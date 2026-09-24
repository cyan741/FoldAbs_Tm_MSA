# TMP-Prediction Stage 1 Sanity Execution Plan

> 目标：只验证 **HITPH → HLA clip 映射 → Protenix frozen trunk → TMP classification head → BCE loss 可稳定下降 → checkpoint 可重载预测**。  
> 本阶段不启用 MSA、diffusion、PAE、ipTM，也不追求最终 benchmark 性能。

## 0. 固定约束

- HITPH 已含 `label=0/1`，**Stage 1 不重新构造负样本**。
- split：按 label 分层随机划分 **80% train / 10% validation / 10% test**，seed=42。
- 原始 `hla` 34-aa pseudo-sequence 不作为模型 HLA 输入。
- `hla.allele` 必须映射到 Hi-TpH：
  - `data/Hi-TpH-hla_allele2seq.json`
  - 使用字段 `clip sequence`
- allele 未找到时：
  1. 已成功映射的 records 仍保存；
  2. unmatched records 单独保存；
  3. 转换命令报错退出；
  4. **禁止 fallback 到 pseudo-sequence**。
- Protenix Stage-1 cache 只保留：
  - `token_asym_id`
  - `token_residue_index`
  - `s`
  - `z`
  - `p_dist`
- 不保存 `s_inputs`。
- 训练状态只写 `train.log`；不生成 `history.json`。

---

# Level 1 — Tiny-set Overfit Test

## 目标

证明训练链本身是可学习的，而不是验证泛化。

预期：

```text
train BCE 明显持续下降
并最终能够强行拟合极小训练集
```

如果 Level 1 失败，不进入 Level 2。

## 1. 转换 HITPH

先准备 Hi-TpH HLA 映射文件，然后：

```bash
python scripts/make_protenix_input/convert_hitph.py \
  --input-csv data/hitph.csv \
  --hla-map-json /path/to/Hi-TpH/data/Hi-TpH-hla_allele2seq.json \
  --output-csv work/hitph_tmp.csv \
  --seed 42
```

输出至少包括：

```text
work/hitph_tmp.csv
work/hitph_tmp.csv.conversion.json
```

若存在无法映射的 allele：

```text
work/hitph_tmp.csv.unmatched.csv
```

同时命令以 error 退出。先解决 unmatched allele，再继续后续步骤。

### 必查

```text
columns:
id,tcra,tcrb,peptide,hla,hla_sequence,label,split

hla_sequence:
来自 Hi-TpH "clip sequence"

ab 中 beta:
必须与源 beta 列一致

label:
同时含 0 和 1
```

---

## 2. 构造 tiny subset

从已经完成 80/10/10 划分的数据中构造一个独立 smoke manifest。

推荐规模：

```text
train:       32–64
validation:  16–32
test:        16–32
```

train 尽量保持：

```text
positive : negative ≈ 1 : 1
```

Level 1 的核心不是严格数据统计，而是让模型能 overfit。

---

## 3. 生成 Protenix JSON

```bash
python scripts/make_protenix_input/make_tmp_protenix_input.py \
  --input-csv work/hitph_level1.csv \
  --output-json work/hitph_level1.json
```

---

## 4. 提取 frozen Protenix trunk cache

本阶段：

```text
MSA OFF
template OFF
diffusion OFF
cycles = 1
```

运行：

```bash
python scripts/extract_protenix_embeddings.py \
  --input-json work/hitph_level1.json \
  --output-dir work/hitph_level1_features \
  --cycles 1 \
  --limit 0 \
  --resume
```

每个 `.pt` 应只包含：

```text
sample_name
token_asym_id
token_residue_index
s
z
p_dist
```

不应再包含：

```text
s_inputs
```

---

## 5. Level-1 训练配置

为了最大限度降低调试变量，先不用额外 trainable Pairformer：

```text
pairformer_blocks = 0
dropout = 0
weight_decay = 0
```

推荐命令：

```bash
python scripts/train_tmp_specificity.py \
  --manifest work/hitph_level1_manifest.csv \
  --features-dir work/hitph_level1_features \
  --output-dir runs/tmp_stage1_level1 \
  --max-steps 1000 \
  --learning-rate 1e-3 \
  --weight-decay 0 \
  --gradient-accumulation 1 \
  --validate-every 50 \
  --log-every 10 \
  --train-loss-window 20 \
  --pairformer-blocks 0 \
  --hidden-dim 256 \
  --dropout 0
```

如果正负比例明显失衡，再加：

```text
--auto-pos-weight
```

否则 tiny-set overfit 首轮可以不加。

---

## 6. Level-1 判定标准

### PASS

至少满足：

1. `train.log` 中 `train_bce_moving` 有明确下降趋势；
2. tiny train set 能被明显拟合；
3. `best_model.pt` 正常生成；
4. `predictions.csv` 和 `metrics.json` 正常生成；
5. checkpoint 可被 `predict_tmp_specificity.py` 重载；
6. 同一输入重跑预测流程不报 shape / role / feature schema 错误。

### FAIL

若 train BCE 长期停留在约：

```text
0.69
```

附近且无法下降，优先检查：

1. label 是否和 feature sample 对齐；
2. manifest 的 `sample_id` 是否与 `.pt` 文件名一致；
3. TCRα/TCRβ/peptide/HLA entity 顺序；
4. `token_asym_id` 与 role mask；
5. 正负样本是否实际都有；
6. logit 是否在训练中改变；
7. gradient 是否非零；
8. learning rate 是否过小。

**Level 1 未通过前，不增加 Pairformer blocks，不启用 MSA。**

---

# Level 2 — Small Training Smoke Test

## 目标

验证不是只会记住几十个样本，而是完整的训练 / validation / checkpoint 流程在小规模数据上稳定工作。

这仍然不是最终科学实验。

## 1. 数据规模

推荐：

```text
总计：256–512 samples

例如：
train       400
validation   56
test         56
```

继续保留原 Stage-1 的随机分层 split 逻辑。

如果完整数据类别失衡，不强行下采样；优先使用：

```text
--auto-pos-weight
```

---

## 2. Protenix cache

仍然：

```text
MSA OFF
template OFF
diffusion OFF
cycles = 1
```

```bash
python scripts/extract_protenix_embeddings.py \
  --input-json work/hitph_level2.json \
  --output-dir work/hitph_level2_features \
  --cycles 1 \
  --limit 0 \
  --resume
```

同时记录实际：

```text
样本数
平均 n_token
.pt 总大小
平均 .pt 大小
总耗时
sec/sample
GPU peak memory
```

用这组实测数据再估算几万条 full cache 的时间与磁盘需求，不在 Stage 1 预先批量生成全部缓存。

---

## 3. Level-2 第一轮仍用简单 head

先保持：

```text
pairformer_blocks = 0
```

推荐：

```bash
python scripts/train_tmp_specificity.py \
  --manifest work/hitph_level2_manifest.csv \
  --features-dir work/hitph_level2_features \
  --output-dir runs/tmp_stage1_level2_p0 \
  --max-steps 2000 \
  --learning-rate 3e-4 \
  --weight-decay 1e-4 \
  --gradient-accumulation 3 \
  --validate-every 100 \
  --log-every 20 \
  --train-loss-window 50 \
  --pairformer-blocks 0 \
  --hidden-dim 256 \
  --dropout 0.1 \
  --auto-pos-weight
```

观察：

```text
train_bce_moving
validation_bce
AUROC
AUPRC
partial AUC0.1
```

---

## 4. Level-2 第二轮再加入 1 个 Pairformer block

只有 `pairformer_blocks=0` 的训练已经稳定后：

```text
0 block
→ 1 block
```

不直接跳到 4 blocks。

只改：

```text
--pairformer-blocks 1
```

比较：

```text
是否仍稳定下降
GPU memory
step time
validation metric
```

Stage 1 不要求证明 1 block 一定优于 0 block，只要求训练流程稳定。

---

## 5. Level-2 PASS 标准

满足以下条件即可进入后续开发：

- `train_bce_moving` 稳定下降；
- validation BCE / metrics 可重复计算；
- 不出现 NaN / exploding gradient / OOM；
- checkpoint selection 正常；
- prediction script 可读取 best checkpoint；
- 0-block 与 1-block 两种配置至少都能正常完成训练；
- Protenix cache 时间和空间得到真实测量值。

---

# Stage 1 明确不做的事情

本阶段不做：

```text
MSA
diffusion coordinates
PAE
ipTM
geometry bins
完整 FoldAbS specificity reproduction
全量几万条 Protenix cache
peptide-OOD final split
IMMREP25 benchmark tuning
```

Stage 1 结束条件只有一个：

> **数据转换正确，frozen feature 正确，训练链正确，loss 能下降。**
