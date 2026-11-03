# Mini-ESM 在线纯 Group Rank 模式

## 使用

保留 `scripts/train_mini_esm_online_ddp.py` 的默认 `--loss-type bce`。新增参数：

| 参数 | 默认值 |
| --- | --- |
| `--loss-type` | `bce`，另可选 `group_rank` |
| `--group-negatives` | 2 |
| `--group-margin` | 1.0 |
| `--group-accumulation-steps` | 1 |
| `--group-require-full` | 未启用，允许实际 K 小于上限 |
| `--activation-checkpoint` / `--no-activation-checkpoint` | group 模式启用，BCE 模式关闭 |

A100 正式启动：

```bash
GPU_IDS=0,1 bash scripts/run_immrep25_group_a100.sh
```

默认写入独立目录 `/NAS/luyq/foldabs_group_rank_20261029/lr_0.0001_bs_2_wd_0.01_drop_0.1_group_rank_k_2`，不覆盖原 BCE 输出。可用 `OUTPUT_DIR` 指定其他目录。包装脚本继承原数据划分、缓存、AdamW 参数、10 epochs 和 patience=5；每卡一个预处理 worker，以控制与原任务并行时的 CPU 使用。源缓存不写入联合 embedding。

通用入口示例：在原在线命令后加 `--loss-type group_rank --group-negatives 2 --group-accumulation-steps 1`。此时旧 `--gradient-accumulation 3` 被组级参数取代；`--chunk-rounds` 单位为每卡组数，必须能被组累积数整除。

断点恢复必须沿用输出目录，并传入 `--resume <输出目录>/last.pt`。模式、分组参数、输入指纹、卡数和随机种子不匹配时拒绝恢复。特征交接精度和 activation checkpoint 的有意变更必须显式指定 `--allow-execution-config-change`，变更位置及前后值写入 `run_config.execution_changes`，此选项不放宽其他恢复检查。`--max-steps` 是累计优化步数，用于在线预检；它停止后可恢复，不代表正式训练完成。

在线 BCE/Group Rank 当前默认关闭 activation checkpoint（仍可用 `--activation-checkpoint` 开启），编码器直接返回原生 single/pair 与 FP32 distogram 概率，不再经过 `.half()` FP16 中转；分类头入口 `.float()` 和 BF16 autocast 保留。该默认适用于显存充足的 A100 80GB：K=2 时关闭重计算的活跃张量峰值约 36.6 GiB，明显高于开启时约 7 GiB；更小显存、更多负样本或更长输入应重新评估容量。

## 训练语义

每个训练正例为 anchor，固定自然 TCRαβ 和 HLA，在同 split、同 HLA 的 unique 正例肽池中均匀无放回抽样。排除该 TCR 完整已知阳性黑名单中的肽。`seed + epoch` 确定采样及组顺序。每个出现位置有独立 ID，生物学原 ID 保留为 `original_id`。

整组在同 rank 的一次 DDP forward 内依次执行冻结 encoder 和可训练 head，保留所有 head 计算图，随后反向。损失仅为 `mean(softplus(margin + negative_logit - positive_logit))`，差值和 softplus 使用 FP32，无 sigmoid/BCE 项。组内平均后按组等权；最后不足 A 组的窗口使用实际组数。DDP 对各卡梯度取均值，不能整除卡数的尾部组统一丢弃并记录，禁止补齐复制。

两卡 K=2、A=1 每步通常为 2 组/6 次复合物前向，少数 K=1 组会减少样本数，但组权重不变。动态采样清单保存为 `epoch_NNN_groups.csv`；审计统计为 `epoch_NNN_sampling.json`，含候选不足、零候选、pHLA 正负次数及 DDP 丢弃计数。

## 验证及产物

val/test 原始 records 不变。固定验证组按 TCRαβ、HLA context 和 HLA sequence 配对已有负例，不生成验证负例；无匹配负例的验证 anchor 单独统计。

按验证标准化 `macro partial AUROC(max_fpr=0.1)` 最大值选择模型、早停并驱动 plateau scheduler。pHLA 等权平均，仅双标签 pHLA 可计算 AUROC；`eligible_pmhc` 记录可计算数。主指标全不可计算时直接报错，禁止换指标继续选模型。

- `bs6_lr_0.0001sd_42_epN.pt`：每轮训练结束后、验证前保留的独立恢复 checkpoint；完整规则见 [保存约定](TMP_CHECKPOINT_POLICY.md)。
- `best.pt`：验证 macro partial AUROC 最优；`final.pt`：最终/早停权重；`last.pt`：可恢复状态，另有带超参数名称的符号链接。
- `run_config.json`：loss/group 配置、随机种子、选择指标、输入指纹；无 git_sha/manifest_sha256。
- `history.json`、`learning_curves.csv/png`：训练/验证 group loss、验证 macro partial AUROC/full AUROC、score gap、pairwise accuracy、组数、跳过数及 lr。
- `test_predictions.csv`：仅 test 输出预测文件，含连续 `score`/兼容 `logit`，不输出未经验证阈值选择的二值 prediction，也不宣称 sigmoid 为校准概率。
- `benchmark.json`：最近日志点的峰值显存、累计吞吐、梯度和 DDP 一致性；耗时包含该会话中的采样、预处理、保存等开销，不能当作纯前向延迟。

## 验证命令

```bash
python -m pytest -q tests/test_tmp_group_rank.py tests/test_mini_esm_online_ddp.py \
  tests/test_immrep25_subset_training.py tests/test_tmp_head_checkpoint.py
python scripts/audit_tmp_group_rank.py --splits-dir <splits> --output sampling_audit.json
OUTPUT_DIR=<全新预检目录> bash scripts/run_immrep25_group_a100.sh \
  --smoke-samples 24 --chunk-rounds 4 --log-every 1 --max-steps 2
# 原目录恢复，跑完剩余组，并执行固定 records 子集的验证/测试
OUTPUT_DIR=<同一预检目录> bash scripts/run_immrep25_group_a100.sh \
  --smoke-samples 24 --chunk-rounds 60 --log-every 1 \
  --resume <同一预检目录>/last.pt --smoke-evaluation-samples 12
```

测试覆盖实际双进程 Gloo DDP 与串行参考的一致性、可变 K 的组等权、累积尾窗、checkpoint 重计算、黑名单及不足候选、验证主指标不可计算时停止、仅 test 导出连续分数。GPU 在线验收与正式运行记录见 `TMP_GROUP_RANK_A100_20261029.md`。

预检验证子集至少 6 条，选取会保留一个双标签 pHLA 和一个完整 TCR/HLA 组，以及最长正负样本；只从原 records 选取，不生成新负例。这个子集只验证执行链路，不能用于判断模型效果。正式验证始终使用全部 records。
