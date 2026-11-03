# TMP-Prediction 数据输入、特征、训练与输出流程

## 1. 实现范围

`tmp-specificity` 分支在原有 Tm/developability 管线旁新增独立的 MHC-I TCR–pMHC 二分类任务。

首版复用当前已验证的冻结 Protenix trunk 输出：

- single representation `s`；
- pair representation `z`；
- softmax 后的 distogram `p_dist`。

然后训练 TMP 专用的 `PairCond + 4 Pairformer blocks + interface-aware classifier`，输出 binding logit、连续概率和可选0/1。

本实现暂不声称复现 FoldAbS antibody-specificity head，因为当前 Protenix 提取器没有运行 diffusion/confidence 路径，也没有输出最终坐标、PAE 或 ipTM。这些信号应作为第二阶段加入，而不能用 `p_dist` 冒充。

## 2. 输入数据契约

### 2.0 HITPH Stage-1 原始数据转换

HITPH 原始列为：

```text
pep,hla,hla.allele,ab,label,beta
```

使用：

```bash
python scripts/make_protenix_input/convert_hitph.py \
  --input-csv data/hitph.csv \
  --hla-map-json /path/to/Hi-TpH/data/Hi-TpH-hla_allele2seq.json \
  --output-csv work/hitph_tmp.csv \
  --seed 42
```

转换规则：

- `pep → peptide`；
- `ab` 按 `TCRα/TCRβ` 拆分；
- `beta` 用于校验拆出的 TCRβ；
- `hla.allele → hla`；
- 原始 34-aa `hla` pseudo-sequence 不作为输入；
- `hla_sequence` 固定来自 Hi-TpH allele mapping 的 `clip sequence`；
- label 分层随机 80/10/10；
- allele 未映射时保留所有已映射 records，同时写 unmatched CSV 并报错退出，不 fallback。



### 2.1 固定 CSV 格式

```csv
id,tcra,tcrb,peptide,hla,hla_sequence,label,split
pair_0001,...,...,GILGFVFTL,HLA-A*02:01,...,1,test
```

八个字段均必需。`label` 必须为 0/1，`split` 必须为
`train/validation/test`。`tcra/tcrb/peptide/hla_sequence` 均不能为空。
HITPH Stage-1 原始数据转换由 `convert_hitph.py` 完成；其他数据源仍需先适配到该固定 CSV。

## 3. 数据流程

```text
固定格式 TMP CSV
  ↓ scripts/make_protenix_input/make_tmp_protenix_input.py
tmp_protenix_inputs.json
  ↓ extract_protenix_embeddings.py（复用现有冻结提取器）
per-complex .pt: token_asym_id, token_residue_index, s, z, p_dist
  ↓ train_tmp_specificity.py
best_model.pt + predictions.csv + metrics.json
  ↓ predict_tmp_specificity.py
submission/prediction CSV + optional pooled latent .pt
```

HiTPH Level IV 正例使用当前 peptide mismatch 流程。生成入口为
`scripts/negative_sampling/peptide_mismatch.py`，训练时动态采样入口为
`scripts/train_tmp_specificity.py --negative-sampling peptide_mismatch`。
完整输入、输出、换 seed 与训练命令见
[动态负采样交付与使用说明](hitph_dynamic_negative_sampling_implementation_results.md)。
旧 `build_tmp_negatives.py`、TCR 错配和有限负例池入口已删除。
生成标签属于 pseudo-negative，不应描述成实验 non-binder。

每个复合物的实体顺序固定为：

```text
TCRα, TCRβ, peptide, MHC-I heavy chain
```

因为顺序是固定契约，不再额外生成 `entity_roles` manifest。

## 4. 构造 Protenix 输入

```bash
python scripts/make_protenix_input/make_tmp_protenix_input.py \
  --input-csv data/tmp_input.csv \
  --output-json work/tmp_protenix_inputs.json
```

Protenix 只需要该 JSON；原 CSV 继续作为标签、split 和样本 ID 的来源。训练前用：

```bash
python scripts/make_protenix_input/make_tmp_manifest.py \\
  --input-csv data/tmp_input.csv \\
  --output-csv work/train_manifest.csv
```

生成 `sample_id,original_id,pmhc_id,label,split` manifest。

## 5. 提取冻结 Protenix latent representation

继续使用仓库现有脚本：

```bash
python scripts/extract_protenix_embeddings.py \
  --input-json work/tmp_protenix_inputs.json \
  --output-dir work/tmp_features \
  --cycles 1 \
  --limit 8 \
  --resume
```

建议先运行8个样本 smoke test。确认后把 `--limit` 设为0，当前提取入口固定禁用 MSA。

每个 `.pt` 包含：

```text
sample_name
token_asym_id
token_residue_index
s             [N, Ds]
z             [N, N, Dz]
p_dist        [N, N, K]
```

训练头使用 `s/z/p_dist/token_asym_id`。Stage 1 不再保存 `s_inputs`，因为后续 TMP 设计未使用该张量；`--resume` 遇到旧缓存时会自动剥离旧 `s_inputs` 后重写。提取结束会额外生成 `extraction_stats.json`，记录 token、缓存大小、耗时和 GPU peak memory。

## 6. TMP 分类头

### 6.1 结构适配

```text
p_dist → linear projection
z + projected p_dist → 2 residual transitions
conditioned z + s → 4 trainable Pairformer blocks
```

### 6.2 显式界面池化

实体级 single pools：

```text
s_TCRα, s_TCRβ, s_peptide, s_MHC
```

界面级 pair pools：

```text
z_TCRα↔peptide
z_TCRβ↔peptide
z_TCRα↔MHC
z_TCRβ↔MHC
z_peptide↔MHC
```

首版使用无参数 masked mean，避免在 latent 导出阶段引入额外未训练的 attention pooling。

拼接后通过三层 MLP 输出一个 logit：

```text
LayerNorm → Linear(512) → GELU → Dropout
          → Linear(512) → GELU → Dropout
          → Linear(1)
```

概率为：

```text
Prediction = sigmoid(logit)
```

0/1 仅由显式阈值派生，默认0.5；benchmark 主结果应使用连续概率。

## 7. 训练

```bash
python scripts/train_tmp_specificity.py \
  --manifest work/train_manifest.csv \
  --features-dir work/train_features \
  --output-dir runs/tmp_v1 \
  --max-steps 4000 \
  --gradient-accumulation 3 \
  --validate-every 100 \
  --pairformer-blocks 4 \
  --auto-pos-weight
```

训练设置：

- BCEWithLogitsLoss；
- AdamW，默认学习率 `1e-4`、weight decay `0.01`；
- batch size 1，适配可变长度复合物；
- 梯度累计和梯度裁剪1.0；
- 按 validation BCE 选择 checkpoint；
- test split 不参与选择；
- train BCE、moving-average train BCE、validation BCE 和 checkpoint 事件写入 `train.log`；
- 不生成 `history.json`。

输出：

```text
best_model.pt
predictions.csv
metrics.json
train.log
```

`best_model.pt` 保存模型参数、完整模型配置、阈值和 feature schema，预测时不需要手工重填通道维度。

## 8. 推理与 latent 输出

```bash
python scripts/predict_tmp_specificity.py \
  --manifest work/immrep25_manifest.csv \
  --features-dir work/immrep25_features \
  --checkpoint runs/tmp_v1/best_model.pt \
  --output-csv runs/tmp_v1/immrep25_predictions.csv \
  --latent-dir runs/tmp_v1/immrep25_latent
```

预测 CSV：

```text
ID,sample_id,pmhc_id,logit,Prediction,prediction_binary,[label]
```

每个 latent 文件保存：

```python
{
  "sample_id": ...,
  "latent": concatenated_entity_and_interface_vector,
  "pools": {
    "s_tcra": ...,
    "s_tcrb": ...,
    "s_peptide": ...,
    "s_mhc": ...,
    "z_tcra_peptide": ...,
    "z_tcrb_peptide": ...,
    "z_tcra_mhc": ...,
    "z_tcrb_mhc": ...,
    "z_peptide_mhc": ...
  },
  "feature_schema": "protenix_trunk_s_z_pdist_v1"
}
```

这些是经过 TMP head 的 PairCond/Pairformer 后的池化 latent，不是原始完整 `N×N×Dz` 缓存。

## 9. 评估输出

有标签时报告：

- overall AUROC；
- overall AUPRC；
- standardized partial AUROC at FPR≤0.1；
- per-pMHC 指标；
- macro per-pMHC AUC0.1。

只有同时含正负样本的 pMHC group 才进入 macro 指标，跳过的 group 数应结合数据报告。

## 10. 当前边界与下一阶段

当前版本已经提供：

- MHC-I αβ TCR输入；
- latent `s/z/p_dist`；
- 显式 TCR–peptide/TCR–MHC/MHC-peptide 界面；
- 二分类训练、连续概率、0/1与latent导出；
- per-pMHC AUC0.1。

下一阶段应扩展 Protenix extractor，真正运行结构/confidence 路径并增加：

```text
predicted coordinates → interface geometry bins
PAE distribution / expected PAE
ipTM / interface confidence
```

然后比较：

```text
z-only
geometry-only
z + geometry
z + PAE
z + geometry + PAE
```

在这些特征真正可用前，当前模型必须称为 `TMPTrunkSpecificityHead`，不能称为完整 FoldAbS specificity 复现。
