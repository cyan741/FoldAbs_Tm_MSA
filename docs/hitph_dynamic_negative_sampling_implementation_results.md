最终复核已通过：CSV 可由 seed=42 精确重建；固定负例命中全局正例 blacklist 的数量为 0，重复四链输入为 0。训练第 1、2 个 epoch 各生成 158,930 条负例，单 epoch 内无重复，同一 seed 与 epoch 可精确重放。候选池的键已按要求保存为规范化的 `hla_alleles` 字符串。

接下来清空指定的旧工作目录，并删除临时测试产物；保留三份 CSV、四份 NPY、生成代码和修改后的训练脚本。本次未生成文件 SHA。
```
• Ran python - <<'PY' …
  └ OLD_WORKSPACE_REMOVED 45 files; workspace empty
• Ran python - <<'PY' …
  └ DELIVERY test_pep2candidates.npy 3642
    DELIVERY blacklilst.npy 19967808
    FINAL_CLEANUP_PASS
```
已完成，seed=42，数据保存在 `/NAS/luyq/TMP_prediction/data/hitph_l4_v1/splits`。

| 文件 | 正例 | 已保存负例 |
|---|---:|---:|
| train.csv | 79,465 | 0 |
| val.csv | 3,951 | 7,902 |
| test.csv | 1,784 | 3,568 |

同时交付 `train_pep2candidates.npy`、`val_pep2candidates.npy`、`test_pep2candidates.npy`、`blacklilst.npy`，最后一个文件沿用你指定的拼写。

采样固定天然配对 TCR 和 HLA，从本 split、同 HLA\_context 的 peptide 集合中排除全局 blacklist 后无放回抽样。blacklist 使用 `(tcra_v,tcrb_v) → peptide set`，包含 84,904 个已知正例组合。

已确认：

- `hla_alleles_raw` 保留原始 HLA 名称，逐条核对原始表 `hla.allele`，85,200 条全部一致。
- `hla_alleles` 是按 JSON 中相同 `hla_clip` 分组、规范化后提取的共同前缀；当前 31 个前缀对应 31 种 clip。
- CSV 保留指定七列，并按第 4 点增加 `neg_count`。正例记录分配数量，train 为 epoch 1 的规划值，负例行为 0。
- 与之前结论一致：不能保证每条正例都有两个唯一负例；通过记录缺额并重新分配，可以达到整体 1:2，因此 `neg_count` 可大于 2。

代码供审核：

- 生成及共用采样逻辑 (negative\_sampling/peptide\_mismatch.py)
- 训练脚本动态采样 (scripts/train\_tmp\_specificity.py)：使用 `--negative-sampling peptide_mismatch`，每 epoch 动态生成 158,930 条负例，支持确定性重放与断点恢复。

验收通过：固定负例命中 blacklist 为 0，重复四链输入为 0；12 项训练回归测试及新增模式的 CPU 三轮训练、断点恢复检查通过。正式 GPU 训练未启动，新负例需对应的四链联合特征缓存。

指定旧工作目录已清空，临时测试产物已删除，本次未生成任何文件 SHA。
