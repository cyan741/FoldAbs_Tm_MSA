# TCR–pMHC 特异性预测：无 MSA Protenix / Mini-ESM

本仓库当前用于 **MHC-I 配对 TCRαβ 与 peptide–HLA 的特异性预测**：复用 ESM 单链表征，由冻结的 Protenix encoder 生成四链联合特征，再训练分类或组内排序头。主要数据流程围绕 HiTPH 固定划分、动态肽错配负样本和 IMMREP2025 相关子集展开。

仓库名 `FoldAbS_Tm_MSA` 和 Python 包名 `foldabs_repro` 沿用早期抗体 Tm 重建项目。**当前主线不需要 MSA 建库、搜索或缓存**；相关实现及 `data/tmp_msa` 链接已移除。项目是基于公开方法的工程研究实现，不是 FoldAbS 官方代码，也不声称完整复现其结构特异性模型。

本文按 **2026-11-04 当前工作区实现**整理。实验报告描述其记录日期的结果，不代表远程任务的实时状态。

## 1. 当前方法

```text
固定数据划分、已知正例与 HLA context
    ↓
训练：每 epoch 重采肽错配负例；验证/测试：固定样本
    ↓
TCRα / TCRβ / peptide / HLA 四链输入
    ↓
复用 ESM2-3B 单链缓存
    ↓
冻结 Protenix Mini-ESM，4 cycles，无 MSA / template
    ↓
s、z、p_dist 在线生成并交给可训练分类头
    ↓
PairCondition → 4-block Pairformer → 实体/界面池化 → MLP
    ↓
BCE 分类 或 Group ranking → 固定验证集选模 → 测试评估
```

在线训练只持久化并复用 **ESM 单链缓存**，不将每个正负组合的 Protenix 联合 embedding 写成磁盘特征库。动态负样本复用已有链，但新四链组合仍需经过 Protenix 联合编码，不能用单链向量拼接代替。

Mini-ESM 联合表示的主要字段如下，`N` 为复合物 token 数：

| 字段 | 形状 | 用途 |
|---|---|---|
| `s` | `[N, 384]` | token 上下文表示 |
| `z` | `[N, N, 128]` | token 对表示 |
| `p_dist` | `[N, N, 64]` | 冻结 distogram head 输出的距离分布 |
| `token_asym_id`、`token_residue_index` | `[N]` | 离线缓存中的链与残基身份索引 |

`p_dist` 经可训练投影后注入 `z`，再进入分类头的 Pairformer。在线主线使用 C 模式注意力池化，汇总 4 个实体与 5 个界面，形成 2176 维读出。encoder 与 distogram head 冻结，分类头参与训练。

默认提取停止于 trunk/distogram，不执行结构 diffusion 或 confidence 推理；模型不以最终坐标、PAE 或 ipTM 为输入。**可选实验分支**提供 `--use-d-geomt`：调用冻结 Protenix diffusion，生成 Cα 距离分箱特征 `d_geom`，额外注入 PairCondition；不提供 PAE/ipTM，不能视为 FoldAbS specificity 完整复现。详见 [Diffusion 几何特征实现与验收](docs/plans/TMP_D_GEOM_DIFFUSION.md)。

## 2. 已实现的功能

| 功能 | 当前实现 |
|---|---|
| 输入转换 | 标准 TMP CSV → 四链 Protenix JSON；支持固定评估样本、单 epoch 动态负例与分片 |
| 负采样 | 保持天然 TCRαβ 配对和 HLA，只替换 peptide；使用 split 内同 HLA 候选池及全局已知阳性黑名单 |
| ESM 缓存 | ESM2-3B 单链提取、身份/精度/配置检查、本地按需复制与覆盖验证 |
| 联合特征提取 | **单一入口**支持 Protenix-base 和 Mini-ESM，含多 GPU 分片、缓存校验与断点复用 |
| 在线 BCE | 多 GPU DDP、梯度累积、动态负采样、固定验证/测试、早停与恢复 |
| Group ranking | 正例 anchor 与 K 个错配肽组成一组，组内排序损失、按组等权、组级累积与评估 |
| 编码器池 | 当前工作区已接入 Group 训练：每卡多个冻结 encoder、CUDA IPC 特征交接和 MPS 启动管理 |
| 可恢复训练 | 模型、优化器、调度器、随机状态及数据游标保存；逐 epoch 检查点和执行配置变更记录 |
| 诊断与性能评估 | BCE 捷径诊断、确定性检查、算子/精度/重计算/编码器池的显存与吞吐比较 |

### BCE 与 Group ranking 的区别

| 项目 | BCE | Group ranking |
|---|---|---|
| 训练单元 | 打乱后的单个正/负样本 | 一个正例及其同 TCR/HLA 下的 K 个负例 |
| 损失 | Binary cross entropy with logits | `mean(softplus(margin + negative_logit - positive_logit))` |
| 默认负样本设置 | epoch 总体正负比 1:2；不保证每个 batch 比例 | `K=2`；候选不足时可用较小 K，或要求完整组 |
| 梯度累积参数 | `--gradient-accumulation` | `--group-accumulation-steps` |
| 验证选模指标 | 验证 BCE 最小 | 验证 macro partial AUROC，`max_fpr=0.1`，最大 |
| 结果解释 | 分类 logit 与 sigmoid 分数；校准效果需另行评估 | 连续排序分数，不作为已校准结合概率 |

Group 模式按组等权，不通过全局配额补偿来保证每个 anchor 恰好 K 个负例。两种模式均保留固定验证/测试输入，训练的动态负例不得与固定评估输入重叠。

## 3. 代码组织

```text
.
├── scripts/                      # 数据准备、提取、训练、评估和实验工具
│   ├── make_protenix_input/      # 输入 schema、HiTPH 转换、manifest/JSON
│   ├── negative_sampling/        # peptide mismatch 协议与候选池
│   └── protenix_migration/       # 历史无 MSA 离线缓存的跨主机调度
├── src/foldabs_repro/
│   ├── tmp_prediction/           # 当前模型、池化、指标、缓存和恢复逻辑
│   └── ...                       # 公共模块及早期 Tm 基线
├── tests/                        # 输入、缓存、损失、DDP、恢复及数值回归测试
├── data/immrep25/                # IMMREP2025 TSV 与转换后的 TMP CSV
├── bash/                         # 历史 Mini-ESM 加速离线提取包装入口
├── docs/
│   ├── plans/                   # 方法规范、训练用法和迁移计划
│   ├── deliverys/               # 交付说明、诊断与实验报告（沿用现有目录名）
│   └── benchmarks/              # 原始基准 JSON、CSV、图、日志与归档
├── pyproject.toml                # Python 包和基础依赖
└── requirements-extras.txt       # fair-esm、pytest 等附加依赖说明
```

### 主要入口

| 文件 | 用途 |
|---|---|
| [peptide_mismatch.py](scripts/negative_sampling/peptide_mismatch.py) | 生成候选池、黑名单和固定数据划分产物 |
| [make_tmp_protenix_input.py](scripts/make_protenix_input/make_tmp_protenix_input.py) | 无 MSA 四链输入生成 |
| [prepare_immrep25_subset_training.py](scripts/prepare_immrep25_subset_training.py) | 整理 IMMREP25 相关训练子集及排除信息 |
| [prepare_mini_esm_assets.py](scripts/prepare_mini_esm_assets.py) | 准备 Mini-ESM/ESM 权重资源 |
| [extract_protenix_esm_chains.py](scripts/extract_protenix_esm_chains.py) | ESM 单链缓存提取 |
| [extract_protenix_embeddings.py](scripts/extract_protenix_embeddings.py) | Protenix-base / Mini-ESM 统一离线联合特征提取 |
| [audit_tmp_feature_cache.py](scripts/audit_tmp_feature_cache.py) | 四链联合缓存审计和重试清单 |
| [train_mini_esm_online_ddp.py](scripts/train_mini_esm_online_ddp.py) | **在线训练统一 CLI**，通过 `--loss-type` 选择模式 |
| [train_mini_esm_group_rank.py](scripts/train_mini_esm_group_rank.py) | Group 模式的训练、评估和恢复实现，由统一入口调用 |
| [group_encoder_pool.py](scripts/group_encoder_pool.py) / [launch_group_pool_training.py](scripts/launch_group_pool_training.py) | 编码器池、进程监督与私有 MPS 启动 |
| [train_tmp_specificity.py](scripts/train_tmp_specificity.py) | 离线联合缓存训练，也提供在线训练复用的采样与随机状态工具 |
| [predict_tmp_specificity.py](scripts/predict_tmp_specificity.py) | 离线缓存推理；使用兼容的离线训练 checkpoint |

`scripts/benchmark_*`、`diagnose_*`、`probe_*`、`profile_*`、`report_*` 和 `run_*evaluation.py` 是性能/诊断工具。`run_hitph_cycle4_epochs.py`、`protenix_migration/`、部分 Shell 包装脚本保存了旧离线流程和特定服务器路径，不是通用的在线训练启动命令。抗体 Tm 脚本仍留作历史研究工具。

## 4. 环境与数据准备

需要 Python **3.11–3.12**、适配目标 GPU/CUDA 的 PyTorch、Protenix 及其依赖。历史 Mini-ESM 提取在 Protenix 1.0.0、PyTorch 2.7.1+cu118、fair-esm 2.0.0 环境完成；其他硬件/算子组合需按相应报告配置和验收，不把该组合视为所有服务器的统一环境。

在已配置好 PyTorch、Protenix 的环境中，从仓库根目录执行：

```bash
python -m pip install -e .
python -m pip install -r requirements-extras.txt
export PYTHONPATH="$PWD/src:$PWD/scripts${PYTHONPATH:+:$PYTHONPATH}"
export PROTENIX_ROOT_DIR=/path/to/protenix_assets
export LAYERNORM_TYPE=torch
```

`pip install -e .` 只安装项目声明的基础依赖，不会自动完成 CUDA、Protenix、cuEquivariance 或模型权重安装。在线 encoder 当前使用 cuEquivariance；即使选择 Torch 分类头，仍需准备与环境兼容的 encoder 算子。编码器池的专用启动器还需要 NVIDIA MPS。

外部资产包括：

- `PROTENIX_ROOT_DIR/checkpoint/` 下所选 Protenix 权重，以及 `common/` 下的化学组件等推理资源。
- 已准备好的 ESM 单链缓存（`run_spec.json`、`metadata/`、`embeddings/chains/`）。只在首次提取或增加新链时需要运行 ESM2-3B。
- 正式 splits 目录及其候选池、黑名单。仓库中的 `data/immrep25/` 不等于完整在线训练输入。

标准 TMP CSV 使用 `id, tcra, tcrb, peptide, hla, hla_sequence, label, split`。动态负采样产物则有独立契约：`train.csv` 保存训练正例，`val.csv/test.csv` 保存固定正负例，配套 `*_pep2candidates.npy`、`blacklilst.npy`，并保留 `positive_splits.csv` 用于身份映射。`blacklilst.npy` 是既有接口的实际拼写。

## 5. 常用流程

以下命令中的 `/path/to/...` 需替换为本机实际路径。先准备数据和缓存，再启动训练。

### 生成无 MSA 输入与 ESM 缓存

```bash
python scripts/make_protenix_input/make_tmp_protenix_input.py \
  --input-csv /path/to/tmp.csv --output-json /path/to/input.json

python scripts/extract_protenix_esm_chains.py \
  --input-json /path/to/input.json \
  --cache-dir /path/to/esm_cache \
  --checkpoint-dir "$PROTENIX_ROOT_DIR/checkpoint"
```

也可将输入转换的 `--input-csv` 换为 `--hitph-splits-dir /path/to/splits`；增加 `--train-negative-epoch E` 导出指定 epoch 的负例。该输入生成已不依赖 MSA 注册表。

### 在线训练（主线）

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run \
  --standalone --nnodes=1 --nproc-per-node=2 \
  scripts/train_mini_esm_online_ddp.py \
  --splits-dir /path/to/splits \
  --esm-cache-dir /path/to/source_esm_cache \
  --local-esm-cache /path/to/local_esm_cache \
  --output-dir /path/to/new_bce_run \
  --loss-type bce --epochs 10 --gradient-accumulation 3
```

双卡、每卡一条样本、累积三次时，有效全局 batch 为 6。正负样本在 epoch 内混合打乱，每个 batch 不强制 1:2。

运行 Group 模式时，使用新的输出目录，并将损失/累积参数改为：

```text
--loss-type group_rank --group-negatives 2 --group-margin 1.0
--group-accumulation-steps 1
```

编码器池仅用于 Group 模式，参数为 `--encoders-per-gpu N`；还可设置 `--encoder-fp32-aggregate`、`--head-kernel torch` 和 `--activation-checkpoint`。当前默认 `N=0`（不启用池）、分类头 cuEquivariance、关闭激活重计算；这些默认值不等于所有显存条件下的推荐配置。MPS 启动与 GPU 检查见 [专用启动器](scripts/launch_group_pool_training.py)，训练参数通过 `--` 传入。

恢复时沿用原输出目录，增加 `--resume /path/to/run/last.pt`。代码校验数据、采样、模式和执行配置；可允许的精度/重计算变更须显式声明并记录，不应假定任意配置都可继续原 checkpoint。预检可使用 `--smoke-samples`、`--smoke-evaluation-samples` 和 `--max-steps`，预检指标不作为正式效果结论。

### 离线联合特征（兼容与对照）

```bash
# 旧 Protenix-base，默认 seed=101；兼容旧无 MSA 缓存恢复
python scripts/extract_protenix_embeddings.py \
  --model-name protenix_base_default_v1.0.0 \
  --input-json /path/to/input.json --output-dir /path/to/base_features \
  --cycles 4 --limit 0 --resume

# Mini-ESM，默认 seed=42；使用已提取的 ESM 单链缓存
python scripts/extract_protenix_embeddings.py \
  --model-name protenix_mini_esm_v0.5.0 --cycles 4 \
  --esm-cache-dir /path/to/esm_cache \
  --input-json /path/to/input.json --output-dir /path/to/mini_features \
  --limit 0 --resume
```

两种模式共用一个提取器，均禁用 MSA/template。`--no-msa` 仅为旧命令兼容参数，`--use-msa` 已移除。base 保留旧缓存配置及 FP16 token 索引的读取兼容；Mini-ESM 使用严格模型、序列和缓存配置校验。离线缓存可交给 `train_tmp_specificity.py`，但不是在线训练的必需前置步骤。

## 6. 已有实验结果与边界

| 已记录工作 | 结果 | 证据 |
|---|---|---|
| Mini-ESM 小样本提取及恢复 | 818 条唯一链；484/484 个联合特征成功，约 8.31 GiB；恢复时全部复用，文件大小和修改时间不变 | [提取交付](docs/deliverys/TMP_HITPH_MINI_ESM_NO_MSA_DELIVERY.md) |
| 双卡编码器池 + Torch 分类头基准 | 每卡 3 encoder、关闭重计算，288 样本耗时 135.30 秒，约 2.129 条/秒；每卡峰值约 77.25 GiB | [显存与吞吐报告](docs/deliverys/TMP_TORCH_HEAD_MEMORY_SWEEP_20261101.md) |
| 基准内确定性对照 | 固定环境与样本下，独立重复的逐组 logit/loss/梯度范数及最终参数一致 | [同一报告的确定性记录](docs/deliverys/TMP_TORCH_HEAD_MEMORY_SWEEP_20261101.md) |

484 条是历史小样本接口验收集；吞吐数据是限定硬件、样本长度与配置下的基准，不是完整训练 epoch 的速度承诺。当前工作区已有编码器池训练接入，但旧基准通过不等于最新接入代码已完成完整训练验收。上述工程结果也不代表最终测试集分类性能；效果应根据具体 run 的固定测试集结果报告。

## 7. 验证与训练产物

在完整依赖环境中，可运行受主线影响的回归测试：

```bash
python -m pytest -q \
  tests/test_protenix_extraction_resume.py tests/test_mini_esm_cache.py \
  tests/test_tmp_prediction.py tests/test_mini_esm_online_ddp.py \
  tests/test_tmp_group_rank.py tests/test_tmp_epoch_checkpoints.py
```

测试覆盖输入/负例身份、两种提取缓存的恢复、无 MSA 边界、Group loss、DDP 与检查点等；部分测试依赖 PyTorch/Protenix 和多进程环境。测试通过不替代目标 GPU 上的前后向、显存和恢复验收。本文更新只核对文档、代码接口和链接，不宣称本次重新完成上述测试或 GPU 实验。

在线训练按模式生成 `run_config.json`、`history.json`、验证/测试指标、`best.pt`、`last.pt`、逐 epoch 检查点及预测文件；Group 模式另保存组清单、采样审计和排序指标。`last.pt` 用于恢复，`best.pt` 按该模式的验证指标选择。具体命名及保存时点见 [检查点约定](docs/deliverys/TMP_CHECKPOINT_POLICY.md)。

## 8. 文档导航

- [动态负采样实现与数据契约](docs/deliverys/hitph_dynamic_negative_sampling_implementation_results.md)
- [在线 DDP 实现记录](docs/deliverys/HITPH_MINI_ESM_ONLINE_DDP_20261026.md)
- [Group ranking 使用说明](docs/plans/TMP_GROUP_RANK_USAGE.md)
- [Group loss 设计](docs/plans/TMP_GROUP_LOSS_IMPLEMENTATION_PLAN.md)
- [模型与论文方法差异](docs/plans/TMP_SPECIFICITY_PAPER_ALIGNMENT.md)
- [BCE 诊断](docs/deliverys/BCE_DIAGNOSIS_20261029.md)
- [编码器池与 Torch 分类头评估](docs/deliverys/TMP_TORCH_HEAD_MEMORY_SWEEP_20261101.md)
- [MSA 清理审计与执行记录](docs/MSA_REMOVAL_AUDIT_20261103.md)

`docs/plans/` 和 `docs/deliverys/` 保留了历史方案及服务器路径，部分内容可能滞后于实现；当前入口、参数默认值以代码为准。旧 MSA 文档用于方法追溯，不再提供可执行建库流程。
