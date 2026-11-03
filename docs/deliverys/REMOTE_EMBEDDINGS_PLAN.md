# HiTPH cycle 4、无 MSA 联合特征缓存与训练

> 历史方案：以下状态与资源分配是当时实验记录，不代表当前任务状态。2026-11-03 迁移至 docs；旧 base 提取入口已迁移，当前 Mini-ESM 在线训练不依赖该离线缓存编排。

当前统一入口 `scripts/extract_protenix_embeddings.py` 同时支持两种模型：

| 模型选择 | 默认 seed | 缓存恢复 |
|---|---|---|
| `--model-name protenix_base_default_v1.0.0`（默认） | 101 | 兼容旧 `model=protenix-v1` 配置和 FP16 token 索引；权重绝对路径可随主机变化，其他身份参数仍须匹配 |
| `--model-name protenix_mini_esm_v0.5.0 --cycles 4 --esm-cache-dir ...` | 42 | 使用原有 Mini-ESM 严格缓存协议和 ESM 单链缓存 |

两种模式均固定禁用 MSA/template；`--no-msa` 仅为兼容旧命令保留，`--use-msa` 已移除。`--resume` 不重写通过校验的缓存。分片、推理、保存和索引汇总共用同一实现。


计算在 Tang3；`/NAS/luyq/TMP_prediction/protenix_embeddings/hitph_v1` 是 `Foldtmp@ai4ls.myds.me:/home/Foldtmp` 的 SSHFS 挂载，仅存 Protenix 输出。当前项目占用必须低于 20 TB。输入、模型资源、预处理和训练输出放在本机或原有 NAS，不放到 Foldtmp。

规范输入为 `/NAS/luyq/TMP_prediction/hitph_msa/protenix_inputs/all_positive_and_fixed_eval_nomsa.json`，对应 manifest 包含 train/val/test 全部正样本和固定 val/test 负样本，共 96,670 条。四链均无 MSA 字段。`configs/hitph_protenix_cycle4_nomsa.json` 固定 cycles=4、seed=101、bf16 推理和 fp16 缓存，输出根目录名包含这些参数。

| 缓存范围 | 条数 | fp16 张量载荷估算 |
|---|---:|---:|
| 全部正样本与固定评估负样本 | 96,670 | 6.508 TB |
| epoch 1 动态 train 负样本 | 158,930 | 10.714 TB |
| 两者同时存在 | 255,600 | 17.222 TB |

这是张量载荷，实际 `.pt` 文件和元数据略大。训练负样本每 epoch 按现有同 split、同 HLA、全局阳性排除协议确定性重采样；候选池约 5,825 万条，不整体缓存。每次只保留一轮训练负样本。训练完一个完整 epoch 并校验 checkpoint 后，删除该轮负样本，生成下一轮；正样本和固定评估负样本永久保留。训练抽样器按 epoch 打乱所有 train 样本，包含正样本。下游头采用 4-block Pairformer、C 模式全 attention pooling。

正式运行顺序：

1. `findmnt -T /NAS/luyq/TMP_prediction/protenix_embeddings/hitph_v1` 确认挂载；`nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits` 按**剩余显存**选择 GPU。卡上已有进程不构成排除条件。
2. 用两条真实无 MSA 输入试跑 cycle 4，核对 `.pt` 的四链、shape、extraction_config、文件大小、速度和显存峰值。
3. 在输出根目录的 `features/` 中提取 96,670 条永久特征，开启 `--resume`；临时工作目录位于本地 `work/hitph_cycle4_nomsa/`。
4. 每个训练 epoch 通过 `make_tmp_protenix_input.py --hitph-splits-dir ... --no-msa --train-negative-epoch E` 导出该轮负输入，提取到同一个 `features/`，用 `train_tmp_specificity.py --negative-sampling peptide_mismatch --epochs E --pairformer-blocks 4 --pooling-mode C --precision bf16` 训练一个完整 epoch，后续轮次用前一轮 checkpoint 恢复。校验 checkpoint 后删除该轮负特征。下一轮重新抽样并提取。

Protenix 提取入口是 `scripts/extract_protenix_embeddings.py`。`scripts/extract_esm2_embeddings.py` 提取的是 ESM-2，不用于本缓存。`extract_hitph_features.py` 要求 MSA，也不用于当前无 MSA 流程。样本级 `.pt` 文件名仍保持训练 `sample_id`，以便现有数据集直接读取；cycles 等参数写在目录名和每条文件的 extraction_config 中。

缓存预算详见 `docs/HITPH_CYCLE4_NOMSA_CACHE_BUDGET.json`。训练流程在全部永久特征完成前不启动。

## 当前执行状态

cycle 4 / 无 MSA 两条样本试跑通过：每条约 64 MiB，`s=(417,384)`、`z=(417,417,128)`、`p_dist=(417,417,64)`；严格加载 checkpoint，GPU PyTorch 峰值约 3.32 GB。为省去被 checkpoint 覆盖的 SciPy 随机初始化和无模板占位特征的零值几何计算，提取入口增加了等价快速路径；一条样本优化前后所有五个缓存张量逐元素完全相等。双卡（GPU 0、6）4-block/C 头在 humanI 小样本上完成了一个 epoch，训练、验证、测试及 checkpoint 均成功。

永久缓存后台进程与完整命令记录在本地 `work/hitph_cycle4_nomsa/permanent_process.json`，日志为同目录 `permanent_extraction.log`；输出为配置根目录的 `features/`。当前使用 GPU 0、1、4、6、7，通过 `--resume` 按样本续跑。epoch 1 的 158,930 条负输入已经导出至本地 `work/hitph_cycle4_nomsa/train_negatives_epoch_0001.json`，永久缓存结束前不提取其特征。后续训练轮换入口为 `scripts/run_hitph_cycle4_epochs.py`，运行前需明确总 epoch 数和训练 GPU 组合。
