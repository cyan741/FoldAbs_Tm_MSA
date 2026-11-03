# HiTPH mini-ESM 无 MSA embedding 交付

日期：2026-10-12。对应 [执行计划](TMP_HITPH_MINI_ESM_NO_MSA_EXPERIMENT_PLAN.md)。

## 环境与权重

使用现有 foldtmp 环境：

```text
/NAS/luyq/miniconda3/envs/foldtmp/bin/python
protenix 1.0.0
torch 2.7.1+cu118
fair-esm 2.0.0
```

官方权重已下载到：

```text
/NAS/luyq/TMP_prediction/protenix_assets/proteinix/protenix_assets/checkpoint/
  protenix_mini_esm_v0.5.0.pt                       541640990 bytes
  esm2_t36_3B_UR50D.pt                            5678116398 bytes
  esm2_t36_3B_UR50D-contact-regression.pt                 6759 bytes
```

`PROTENIX_ROOT_DIR` 指向上一级 `protenix_assets`，复用其 `common` 资源。下载使用安装包提供的官方 URL，采用系统 CA 证书校验，临时文件完整下载后才改名。fair-esm 加载所需的 contact-regression 文件也已准备。无需新增软件包。

[官方模型说明](https://github.com/bytedance/Protenix/blob/main/docs/supported_models.md)介绍了 mini 的 4 cycles 设置及 ESM2-3B。实际执行配置从安装包 runner 读取并核验；mini 的 `load_strict=False` 默认值已在构造 runner 前改为 True。

## 可执行入口

在仓库根目录运行：

```bash
/NAS/luyq/miniconda3/envs/foldtmp/bin/python -u scripts/run_hitph_mini_esm.py --gpu-ids 7,1
```

该命令按顺序完成数据适配、权重检查、ESM 单链预计算、两个 Protenix 分片及逐文件验收。重复运行会严格校验并复用已有缓存。不同模型/精度/seed/cycles/序列的缓存以及缺少身份字段的历史缓存会被拒绝。

小样本独立输出：

```text
/NAS/luyq/TMP_prediction/humanI_uns_hitph_embeddings/protenix_mini_esm_v0.5.0/
  inputs/subset.json
  manifest.csv
  dataset_notes.json
  esm_cache/run_spec.json
  esm_cache/metadata/<sequence-key>.json
  esm_cache/embeddings/chains/<sequence-key>.pt
  features/<sample_id>.pt
  features/run_spec.json
  features/index.json
  features/extraction_stats.json
  features/failures.shard_00.json
  features/failures.shard_01.json
  logs/
  completion_report.json
```

ESM 与 Protenix 使用不同进程。ESM 进程退出后才启动 Protenix，避免两套模型常驻 GPU。单链以序列身份去重，缓存保留 FP32 的逐残基表示；ESM 第 36 层输出的 BOS/EOS 已去除，形状为 `[链长, 2560]`。共享 ESM 写入持有文件锁。Protenix 1.0.0 在 dataloader 中硬编码相对目录，本仓库适配器提供它需要的目录布局和 CSV，并在退出初始化工作目录前将 featurizer 路径固定为绝对路径。

联合特征使用官方 mini 架构、4 cycles、BF16 计算、FP16 浮点缓存、int64 token 索引、seed 42。冻结模型并使用 eval/no_grad；仅调用 trunk 和 distogram head。配置记录 diffusion/confidence 未执行。严格 checkpoint 加载保证 missing/unexpected keys 会立即失败；因此可沿用仓库现有的优化，跳过会被 checkpoint 完全覆盖的 SciPy 随机初始化。

每条缓存写入前和恢复时核验：模型/ESM 配置、软件版本、四链序列、token 数和残基索引、s/z/p_dist 通道、精度、有限值、p_dist 范围及概率和。联合通道为 384/128/64。Protenix 原生 residue_index 为从 1 开始的整数。临时 `.pt` 写完并验证后原子改名；错误记录写入分片失败清单。GPU 分片继承全部 CLI 参数，一个样本只属于一个分片。

## 小样本边界

交付的小样本目录共有 484 条：train 388，validation 48，test 48，各 split 正负比例为 1:1。它沿用历史 level2 数据，TCRα/β 为 CDR3 输入；本次按原样本 ID、序列、label 和 split 提取并验证与原 manifest 一致。

该子集用于验证 mini-ESM 提取和现有 head 接口。正式 L4 实验使用完整自然配对可变区、动态 train peptide mismatch 1:2 和固定 val/test，二者的评估结论应分别报告。

## 正式 L4 数据和计划 epoch

准备正式输入（已执行 baseline 和 epoch 1）：

```bash
/NAS/luyq/miniconda3/envs/foldtmp/bin/python -u scripts/run_hitph_mini_esm.py \
  --mode l4-inputs --epochs 1 --prepare-only
```

输出根目录：

```text
/NAS/luyq/TMP_prediction/protenix_embeddings/hitph_v1/mini_esm_v0.5.0_c4_no_msa
```

`inputs/baseline.json` 和对应 manifest 包含 96,670 条：train 正例 79,465；validation 正例/负例 3,951/7,902；test 正例/负例 1,784/3,568。`inputs/train_negative_epoch_001.json` 和对应 manifest 包含 epoch 1 动态训练负例 158,930 条。

所有身份和抽样均调用现有 `PeptideMismatchSampling` 与 `prepare_hitph_inputs`：沿用同 split/HLA context 的 peptide pool、全局 blacklist、组容量和无放回抽样，不重建固定 val/test。输入不含 MSA/template 路径，也不生成 A3M。

准备更多 epoch 时指定 `--epochs 1 2 3`；正式提取入口去掉 `--prepare-only` 即可顺序准备这些 epoch 的单链及联合缓存。epoch 之间的重复合法组合可复用联合缓存，新组合单独提取。正式大规模联合缓存尚不属于小样本完成结果。

## 下游训练命令

现有 `scripts/train_tmp_specificity.py` 和 head 实现保持原训练接口。正式 L4 训练应在 baseline 及所有将消费的 epoch 联合缓存就绪后执行，每轮训练仍由原抽样器生成样本：

```bash
/NAS/luyq/miniconda3/envs/foldtmp/bin/python scripts/train_tmp_specificity.py \
  --features-dir /NAS/luyq/TMP_prediction/protenix_embeddings/hitph_v1/mini_esm_v0.5.0_c4_no_msa/features \
  --output-dir /NAS/luyq/TMP_prediction/mini_esm_no_msa \
  --negative-sampling peptide_mismatch \
  --splits-dir /NAS/luyq/TMP_prediction/data/hitph_l4_v1/splits \
  --negative-sampling-seed 42 --negatives-per-positive 2 --seed 42 \
  --epochs 1 --pairformer-blocks 4 --pooling-mode C --precision bf16 --gpu 7
```

如果训练更多 epoch，先提取对应 seed 42 的 epoch 联合缓存，再将 `--epochs` 改为该数量；恢复训练使用原 `--resume` 参数。默认学习率、正例打乱、gradient accumulation、scheduler 和固定评估均由现有训练脚本保持。

历史小样本的 head 流程验证命令：

```bash
/NAS/luyq/miniconda3/envs/foldtmp/bin/python scripts/train_tmp_specificity.py \
  --manifest /NAS/luyq/TMP_prediction/humanI_uns_hitph_embeddings/protenix_mini_esm_v0.5.0/manifest.csv \
  --features-dir /NAS/luyq/TMP_prediction/humanI_uns_hitph_embeddings/protenix_mini_esm_v0.5.0/features \
  --output-dir /NAS/luyq/TMP_prediction/mini_esm_no_msa_subset \
  --seed 42 --epochs 1 --pairformer-blocks 4 --pooling-mode C --precision bf16 --gpu 7
```

训练命令作为交付入口；本次执行范围为 embedding 提取。

## 验证结果

缓存边界、ESM 路径恢复回归以及下游 tensor/head 接口测试：22 项通过。实际 GPU 提取和独立逐文件验收均已完成，`completion_report.json` 记录 484/484 条有效缓存；运行统计和恢复检查见下文。

成功提取批次的统计（GPU 7 和 1）：

| 项目 | 结果 |
|---|---:|
| ESM 去重单链 | 818 |
| ESM 推理和缓存写入时间（不含模型加载） | 91.21 秒 |
| ESM PyTorch 分配显存峰值 | 11,582,300,672 bytes |
| 联合特征生成 | 484 / 484 |
| 提取失败 | 0 |
| 平均 token 数 | 218.04 |
| 联合缓存大小 | 8,922,925,160 bytes（约 8.31 GiB） |
| 两个 GPU 分片提取墙钟时间 | 743.23 秒 |
| 平均单条生成时间 | 2.412 秒 |
| trunk 阶段 PyTorch 分配显存峰值 | 965,274,624 bytes |

显存统计为对应提取进程的 PyTorch allocated 指标；trunk 阶段在模型和 dataloader 初始化后重置峰值统计。其他同时运行的实验与 CUDA allocator reserved 显存不计入该数值。

实际恢复验收入口：

```bash
/NAS/luyq/miniconda3/envs/foldtmp/bin/python -u scripts/validate_mini_esm_resume.py --gpu-ids 7,1
```

它先保留初次提取统计到 `initial_extraction_stats.json`，再重跑完整入口，检查新增文件数为零、484 条全部复用、所有特征文件大小和修改时间不变；结果写入 `resume_validation_report.json`。

恢复验收已通过：484 个特征文件的大小和修改时间全部不变，生成 0 条、复用 484 条；恢复提取入口耗时 13.01 秒。完成报告状态为 `complete`，恢复报告状态为 `passed`。两个分片的最终失败清单均为空。
