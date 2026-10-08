# HiTPH peptide mismatch：交付与复现

当前负采样仅有两个生产入口：

- `scripts/negative_sampling/peptide_mismatch.py`：输入校验、HLA context、全局 blacklist、各 split 的候选池、配额分配、无放回抽样和固定 CSV 生成。
- `scripts/train_tmp_specificity.py`：复用上述抽样函数，每 epoch 动态生成训练负例，固定 val/test，联合特征校验、DDP 分片与断点恢复。

`src/foldabs_repro/tmp_prediction/cache_audit.py` 是通用特征校验工具，不实现负采样。
旧静态 TCR 错配、有限负例池、旧协议包装脚本和旧协议测试已删除。
`static` 训练模式继续接受已有带正负标签的 manifest，不构建负例。

## 当前正式数据

输入：`/NAS/luyq/TMP_prediction/data/hitph_l4_v1/splits/positive_splits.csv`。
输出：`/NAS/luyq/TMP_prediction/data/hitph_l4_v1/splits`。

| 文件 | 正例 | 保存负例 |
|---|---:|---:|
| train.csv | 79,465 | 0 |
| val.csv | 3,951 | 7,902 |
| test.csv | 1,784 | 3,568 |

另外只有 `train_pep2candidates.npy`、`val_pep2candidates.npy`、
`test_pep2candidates.npy`、`blacklilst.npy` 四份数据产物。
`blacklilst.npy` 沿用指定拼写。NPY 为本地可信字典，加载方式为
`np.load(path, allow_pickle=True).item()`。

- blacklist：精确 `(tcra_v, tcrb_v)` → peptide set，覆盖全部 split、全部 HLA；79,321 个 TCR，84,904 个唯一已知阳性组合。
- 每 split 候选池：规范化 `hla_alleles` 字符串 → 本 split 同 HLA_context 的 peptide set。
- 保留天然配对 TCR 和实际 HLA clip，只替换 peptide；减去全局 blacklist 后抽样。
- 按 `(TCR,HLA clip)` 共享容量，无放回抽样；负例四链输入在单 epoch 内不重复，不与固定评估输入重复。不同 epoch 允许重复抽中合法负例。
- 先分配每条正例最多 2 个，再把不足配额分给还有容量的组件，保证整体 1:2。整体容量不足会报错。
- 每份 CSV 为指定七列加 `neg_count`。正例记录分配数量，train 为 epoch 1 规划；负例行是 0。重新分配后正例计数可大于 2。
- 零合法候选正例数 train/val/test 为 56/286/55。因此可达到整体 1:2，不能保证逐条均匀 1:2，与先前容量结论一致。

`hla_alleles_raw` 保留原始 HLA 名称，已对照原始 `hla.allele` 校验全部 85,200 条正例。
`hla_alleles` 按 hla2seq JSON 中相同 clip 分组，对规范化两字段名称提取完整字段共同前缀。
当前 31 个前缀对应 31 个实际 clip，视作相同 HLA_context；代码拒绝一个前缀对应多种 clip。

## 生成与换 seed

在仓库根目录运行。`--seed` 默认为 42；正式文件已经存在时会拒绝覆盖，
复现与换 seed 请使用一个新的输出目录。

```bash
/NAS/luyq/miniconda3/envs/foldtmp/bin/python scripts/negative_sampling/peptide_mismatch.py \
  --positive-splits /NAS/luyq/TMP_prediction/data/hitph_l4_v1/splits/positive_splits.csv \
  --hla-map /NAS/luyq/TMP_prediction/data/hitph_l4_v1/raw/Hi-TpH-hla_allele2seq.json \
  --output-dir /path/to/new_seed42_splits --seed 42
```

更换 seed：将上例改为 `--seed 43 --output-dir /path/to/new_seed43_splits`。
同样输入、seed、比例可重建相同 CSV；seed 改变固定评估抽样和 train 的 epoch 1 分配规划。
候选池与 blacklist 只由正例输入决定，不随 seed 改变。

## 动态训练与无文件采样预览

```bash
PYTHONPATH=src /NAS/luyq/miniconda3/envs/foldtmp/bin/python scripts/train_tmp_specificity.py \
  --negative-sampling peptide_mismatch \
  --splits-dir /NAS/luyq/TMP_prediction/data/hitph_l4_v1/splits \
  --negative-sampling-seed 42 --sampling-preview-epoch 1
```

预览只打印审计统计，不使用 CUDA、不读取特征、不写产物。
指定相同 seed 与 epoch 可精确重放；不同 epoch 重新抽样，每 epoch 为 79,465 正例和 158,930 负例。

```bash
PYTHONPATH=src /NAS/luyq/miniconda3/envs/foldtmp/bin/python scripts/train_tmp_specificity.py \
  --negative-sampling peptide_mismatch \
  --splits-dir /NAS/luyq/TMP_prediction/data/hitph_l4_v1/splits \
  --negative-sampling-seed 42 --seed 42 \
  --features-dir /path/to/joint_features --output-dir /path/to/new_run \
  --epochs 3 --gpu 0
```

`--negative-sampling-seed` 控制训练负采样，`--seed` 控制模型初始化等训练随机状态。
val/test 使用已生成的固定 CSV；更改训练采样 seed 不会重建评估数据。
换 seed 生成新的目录后，把相同输入 `positive_splits.csv` 放在该目录以恢复正例特征 ID。
训练需要每个采样组合的四链联合 Protenix 特征；缺失或身份不匹配明确报错。
负例特征名为 `pepneg_<split>_<组件编号>_<候选peptide编号>.pt`，不生成文件 SHA。
使用 `--resume /path/to/complete_epoch_checkpoint.pt` 恢复，必须保持输入、候选池、采样配置及训练随机配置一致。

正式 GPU 训练未启动。工程验收覆盖相同 seed 重建、换 seed、blacklist、去重、零候选配额转移、
固定评估、DDP 分片和 CPU 三轮训练的精确断点恢复；临时验收产物在结束后删除。
