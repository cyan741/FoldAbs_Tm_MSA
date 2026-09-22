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

### 2.1 结构编码使用完整 MHC-I 重链

输入 CSV 可以保留 HLA allele 或 pseudo-sequence，但 Protenix 输入必须通过 allele 映射到连续的完整 HLA-I extracellular heavy-chain sequence。pseudo-sequence 不能作为一条可折叠蛋白链。

HLA 映射文件：

```json
{
  "HLA-A*02:01": "FULL_EXTRACELLULAR_HEAVY_CHAIN_SEQUENCE",
  "HLA-B*40:01": "FULL_EXTRACELLULAR_HEAVY_CHAIN_SEQUENCE"
}
```

### 2.2 推理 CSV（例如 IMMREP25）

```csv
ID,TCRa,TCRb,Peptide,HLA
pair_0001,...,...,GILGFVFTL,HLA-A*02:01
```

要求 TCRα、TCRβ、peptide 均非空。首版不悄悄将 α 缺失样本降级为 β-only。

### 2.3 训练 CSV

```csv
ID,TCRa,TCRb,Peptide,HLA,label,split
pair_0001,...,...,GILGFVFTL,HLA-A*02:01,1,train
pair_0002,...,...,GILGFVFTL,HLA-A*02:01,0,train
pair_0003,...,...,NLVPMVATV,HLA-A*02:01,1,validation
pair_0004,...,...,NLVPMVATV,HLA-A*02:01,0,test
```

`label` 必须为0/1；`split` 必须显式为 `train/validation/test`。脚本不会对全表随机拆分，避免先生成负样本再拆分造成实体泄漏。

## 3. 数据流程

```text
TMP CSV + allele→完整HLA序列 JSON
  ↓ make_tmp_protenix_input.py
tmp_protenix_inputs.json + tmp_manifest.csv
  ↓ extract_protenix_embeddings.py（复用现有冻结提取器）
per-complex .pt: token_asym_id, s, z, p_dist
  ↓ train_tmp_specificity.py
best_model.pt + predictions.csv + metrics.json
  ↓ predict_tmp_specificity.py
submission/prediction CSV + optional pooled latent .pt
```

每个复合物的实体顺序固定为：

```text
TCRα, TCRβ, peptide, MHC-I heavy chain, [optional β2m]
```

该顺序会以 `entity_roles` 写入 manifest。训练和预测时，代码把排序后的 Protenix `asym_id` 映射回角色，并在链数不一致时立即报错。

## 4. 构造 Protenix 输入

```bash
python scripts/make_tmp_protenix_input.py \
  --input-csv data/immrep25_test.csv \
  --hla-sequences-json data/hla_class_i_full_sequences.json \
  --output-json work/tmp_protenix_inputs.json \
  --output-manifest work/tmp_manifest.csv
```

如需 β2m：

```bash
  --beta2m-sequence FULL_HUMAN_B2M_SEQUENCE
```

训练输入额外使用：

```bash
  --require-label --require-split
```

输出 manifest 至少包含：

```text
sample_id,original_id,tcra,tcrb,peptide,hla,pmhc_id,
entity_roles,include_beta2m,[label],[split]
```

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

建议先运行8个样本 smoke test。确认后把 `--limit` 设为0，并按需要启用 `--use-msa`。

每个 `.pt` 包含：

```text
sample_name
token_asym_id
token_residue_index
s_inputs
s             [N, Ds]
z             [N, N, Dz]
p_dist        [N, N, K]
```

训练头使用 `s/z/p_dist/token_asym_id`；`s_inputs` 暂不使用。

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
- test split 不参与选择。

输出：

```text
best_model.pt
predictions.csv
metrics.json
history.json
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
    "z_tcrb_mhc": ...
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
- 完整HLA重链映射；
- 可选β2m；
- latent `s/z/p_dist`；
- 显式 TCR–peptide/TCR–MHC 界面；
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
