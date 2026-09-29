# TMP-Prediction Stage 2：正式模型与跨数据集评测计划

> 分支：tmp-specificity  
> 状态：Stage 1 已完成可训练性与小规模流程验证  
> Stage 2 目标：在冻结 Protenix 表征上建立可审计的 Pairformer TMP 模型，完成 HITPH 正式训练/验证，并在 IMMREP25 与独立外部数据集上进行模型锁定后的迁移评测。

---

## 0. Stage 2 的核心决策

1. **Protenix 继续冻结。** 每个 TCRα–TCRβ–peptide–MHC-I 复合物只提取一次 trunk 表征并缓存。
2. **Stage 2 主模型继续使用 Pairformer。** 目标主配置为 4 个可训练 Pairformer blocks，以接近 FoldAbS specificity 分支公开描述的容量。
3. **0-block 必须保留为消融，而不是后续主模型。** 它用于回答性能提升是否确实来自新增 Pairformer。
4. **HITPH 是模型开发和选择数据集；IMMREP25 与外部数据集是锁模后的迁移评测。** 不使用外部测试集选择 checkpoint、阈值或超参数。
5. **当前 TMPTrunkSpecificityHead 是 Stage 1 工程模型，不等于 FoldAbS specificity 复现。** Stage 2 新建论文对齐实现，同时保留当前模型保证历史结果可复现。
6. **先审计论文，再决定是否加入 geometry、PAE、ipTM。** 这些信号可以作为后续结构消融，但不能未经论文证据就称为 FoldAbS specificity 的组成部分。

---

## 1. 当前冻结表征的数据流

### 1.1 从序列到缓存

当前每个样本的实体顺序固定为：

~~~text
TCRα → TCRβ → peptide → MHC-I heavy chain
~~~

数据流为：

~~~mermaid
flowchart TD
    A["TMP CSV：α、β、peptide、HLA"] --> B["Protenix complex JSON"]
    B --> C["冻结 Protenix trunk"]
    C --> D["s：token single representation"]
    C --> E["z：token-pair representation"]
    C --> F["distogram head + softmax"]
    F --> G["p_dist：距离概率"]
    D --> H["per-complex .pt cache"]
    E --> H
    G --> H
    H --> I["可训练 TMP head"]
    I --> J["latent、logit、probability、0/1"]
~~~

提取脚本为 scripts/extract_protenix_embeddings.py。它执行：

1. Protenix 输入预处理与相对位置编码；
2. 冻结 trunk 前向，得到 s 和 z；
3. 对冻结 distogram head 的输出执行 softmax，得到 p_dist；
4. 将张量转为 CPU float16 后逐样本保存；
5. 不运行 diffusion structure sampling，不保存最终坐标、PAE 或 ipTM；
6. 不保存 s_inputs，因为当前 TMP head 没有消费它。

缓存字段：

| 字段 | 语义 | 典型形状 |
| --- | --- | --- |
| token_asym_id | token 所属链，用于恢复 α/β/peptide/MHC role mask | [N] |
| token_residue_index | 链内 residue index | [N] |
| s | 每个 token 的 single representation | [N, C_s] |
| z | 每对 token 的 pair representation | [N, N, C_z] |
| p_dist | 每对 token 的距离分箱概率 | [N, N, C_dist] |

Protenix-v1 的预期通道通常为 C_s=384、C_z=128、C_dist=64，但训练代码必须从实际缓存动态读取并在运行元数据中记录，不能只依赖硬编码。

### 1.2 冻结边界

冻结的是：

- Protenix 输入 embedding；
- 原始 Protenix MSA/trunk；
- 原始 Protenix distogram head；
- 缓存的 s、z、p_dist。

当前可训练的是：

- PairCondition；
- 新增的 0/1/4 个 Pairformer blocks；
- 最终分类 MLP。

因此，“冻结 Protenix”不等于“模型只有一个 MLP”。新增 Pairformer 是位于冻结 trunk 之后的 task-specific adapter，其参数仍参与反向传播。

---

## 2. 不加与加入 Pairformer 后的数据流

### 2.1 PairCondition

无论 pairformer_blocks 是 0 还是大于 0，当前模型都先执行：

~~~text
p_dist
  → LayerNorm
  → Linear(C_dist, C_z)
  → 与原始 z 相加
  → 2 个 residual transitions
  → conditioned z
~~~

每个 residual transition 为：

~~~text
x + Linear(GELU(Linear(LayerNorm(x))))
~~~

因此，当前 P0 不是“完全不处理 z”的模型，而是：

> 冻结 s/z/p_dist + 可训练 PairCondition + interface pooling + MLP。

P0 与 P1 的比较只隔离“额外 1 个 Pairformer block”的贡献。

### 2.2 0-block 路径

~~~text
frozen s
frozen z + p_dist → PairCondition → conditioned z
s 与 conditioned z → entity/interface mean pooling
→ concatenated latent → 3-layer MLP → logit
~~~

### 2.3 K-block Pairformer 路径

~~~text
frozen s
frozen z + p_dist → PairCondition → z_0
(s_0, z_0)
  → Pairformer block 1
  → ...
  → Pairformer block K
  → (s_K, z_K)
  → entity/interface mean pooling
  → concatenated latent
  → 3-layer MLP
  → logit
~~~

加入 Pairformer 后，张量形状不变，但语义发生变化：

- s 从冻结 trunk single representation 变为 specificity-task-adapted single representation；
- z 从距离条件化 pair representation 变为经过三角更新、pair-biased attention 和 transition 的 task-adapted pair representation；
- 缓存文件本身保持不变，改变只发生在训练/推理 head 内；
- predict_tmp_specificity.py 输出的 pooled latent 是 PairCondition/Pairformer 之后的 latent，不是原始缓存张量。

### 2.4 当前池化与输出

实体 single pooling：

- s_TCRα
- s_TCRβ
- s_peptide
- s_MHC

界面 pair pooling：

- z_TCRα→peptide
- z_TCRβ→peptide
- z_TCRα→MHC
- z_TCRβ→MHC
- z_peptide→MHC

拼接 latent：

~~~text
latent = 4 × C_s + 5 × C_z
~~~

若 C_s=384、C_z=128，则 latent 维度为 2176。

分类路径：

~~~text
latent
→ LayerNorm
→ Linear(hidden)
→ GELU
→ Dropout
→ Linear(hidden)
→ GELU
→ Dropout
→ Linear(1)
→ logit
→ sigmoid
→ continuous binding probability
→ validation-calibrated threshold
→ 0/1
~~~

正式 benchmark 必须保存连续 probability；0/1 只作为阈值派生结果。

---

## 3. Pairformer 模块如何工作

当前代码直接调用 Protenix 的 PairformerBlock。一个 block 依次执行以下更新。

### 3.1 更新 pair representation z

1. **Triangle multiplication outgoing**  
   对 i–k 和 j–k 等共享第三个 token 的 pair 信息进行乘法聚合，使 z_ij 能利用三角路径中的关系。

2. **Triangle multiplication incoming**  
   从相反方向聚合三角关系，补充不同方向的信息传播。

3. **Triangle attention starting node**  
   对共享起点的 pair edges 做注意力更新。

4. **Triangle attention ending node**  
   转置 pair 维度后，对共享终点的 pair edges 做注意力更新。

5. **Pair transition**  
   对每个 z_ij 独立执行前馈 transition。

这些操作都采用残差连接；当前非 inplace 路径还在三角更新处使用 row-wise dropout。

### 3.2 用 z 更新 single representation s

更新后的 z 作为 attention bias，引导 token-level self-attention：

~~~text
s ← s + AttentionPairBias(s, z)
s ← s + SingleTransition(s)
~~~

因此 Pairformer 的核心不是简单地“再做一层 Transformer”，而是让：

- residue/token 表征 s；
- residue-pair/interface 表征 z；

在每个 block 内共同更新。对于 TMP 任务，它有机会把 TCR–peptide、TCR–MHC 和 peptide–MHC 三类界面关系传播回 token 表征。

### 3.3 当前调用细节

- batch size 为 1；
- 每个样本没有 padding，所以当前传入 pair_mask=None；
- triangle multiplication 与 triangle attention 使用 torch 实现；
- K 个 block 串行执行，每个 block 有独立参数；
- Pairformer blocks 是可训练的，原始 Protenix trunk 仍冻结。

### 3.4 预期收益与风险

预期收益：

- 在冻结通用结构表征上学习 TMP-specific interface adaptation；
- 通过三角关系联合建模 α/β、peptide 和 MHC；
- 相比直接 mean pooling，允许界面信号先交互再聚合。

主要风险：

- N×N pair tensor 带来显存和计算开销；
- 数据量小时更容易记忆训练集；
- block 数增加后，训练集提升不等于 unseen-epitope 泛化提升；
- 简单 mean pooling 可能稀释短 peptide 接触区域，需要在后续做 pooling 消融。

---

## 4. Stage 1 结果如何影响 Stage 2 决策

当前 388/48/48 的单次实验中：

| 模型 | split | BCE ↓ | AUROC ↑ | AUPRC ↑ | AUC@FPR≤0.1 ↑ |
| --- | --- | ---: | ---: | ---: | ---: |
| P0 | validation | 0.661 | 0.658 | 0.628 | 0.499 |
| P1 | validation | 0.672 | 0.656 | 0.597 | 0.486 |
| P0 | test | 0.630 | 0.727 | 0.669 | 0.508 |
| P1 | test | 0.571 | 0.786 | 0.670 | 0.477 |

可以确认：

- P1 的 train fitting 明显强于 P0；
- P1 在当前 test 上改善了 BCE 和 overall AUROC；
- AUPRC 基本持平，低 FPR partial AUC 没有改善；
- validation 没有显示一致优势；
- 只有一个 seed，validation/test 各 48 条，不能据此宣称 Pairformer 已被统计验证。

Stage 2 的决策是：

> 继续使用 Pairformer，并以 4 blocks 作为目标主配置；同时保留 P0，使用多 seed 和正式 OOD split 验证 Pairformer 是否真正改善泛化。

不因为一次 P1 结果直接删除 P0，也暂不继续增加到 8 blocks。

---

## 5. Stage 2 模型矩阵

### 5.1 必做模型

| ID | 配置 | 作用 |
| --- | --- | --- |
| P0 | PairCondition + 0 Pairformer + 当前 pooling/head | 最小结构对照 |
| P1 | PairCondition + 1 Pairformer + 当前 pooling/head | 与 Stage 1 连续，检查可重复性 |
| P4 | PairCondition + 4 Pairformer + paper-aligned pooling/head | Stage 2 候选主模型 |
| P4-TMP | P4 + TMP role-aware interface pooling | 最终 TMP 适配模型 |
| Seq baseline | 不使用 Protenix pair representation | 判断收益是否来自 folding representation |

### 5.2 表征消融

先用一个固定 seed 筛选，再对关键配置补多 seed：

| 消融 | 输入 |
| --- | --- |
| S-only | s |
| Z-only | z |
| S+Z | s + z |
| S+Z+Dist | s + PairCondition(z, p_dist) |
| S+Z+Dist+PF4 | 上述输入 + 4 Pairformer |

这样才能回答核心研究问题：

> folding model 内部 interface representation 是否比 single/sequence representation 更有可迁移的 TMP recognition signal？

### 5.3 不建议的做法

- 不在 P1 尚未稳定验证前不断增加 block 数；
- 不根据 IMMREP25 或 external test 表现反向选择 head；
- 不同时改变 block 数、split、负样本、MSA、学习率后再比较；
- 不把 P0/P1 单 seed 的 test 差异描述为已证明的模型提升。

---

## 6. 正式数据集规划

## 6.1 统一数据契约 v2

所有数据集先转换成统一 manifest，至少包含：

~~~text
sample_id
original_id
dataset
tcra
tcrb
peptide
hla_allele
hla_sequence
pmhc_id
label
label_source
split
split_protocol
negative_source
~~~

要求：

1. HLA 统一到确定的 allele 分辨率，例如 HLA-A*02:01；
2. 原始 allele 与 canonical allele 同时保留；
3. label_source 区分 experimental positive、experimental negative 和 pseudo-negative；
4. negative_source 记录 tcr_swap、same_hla_cross_epitope 或数据集原始负例；
5. 同一生物学 pair 在所有数据源中使用稳定 hash，便于跨数据集去重；
6. 转换后生成 dataset audit JSON：样本数、类别、peptide、pMHC、HLA、TCR 重复与交集。

### HLA 序列规则

对于 Protenix/folding 主线，优先使用可折叠的 MHC-I extracellular/clip sequence。34-aa pseudo-sequence 是空间上不连续的位点集合，不应默认伪装成一条真实蛋白链送入 folding trunk。

HLA pseudo-sequence 可以用于：

- sequence-only baseline；
- 单独的输入消融；
- 无法运行 folding trunk 时的轻量模型。

它不作为 Protenix 主配置的默认 MHC 链。

---

## 6.2 HITPH：开发、选择和正式内部测试

HITPH 承担三个不同用途，必须分开命名。

### A. random-stratified diagnostic split

用途：

- 工程回归测试；
- 快速比较 P0/P1/P4 是否能训练；
- 调试学习率、显存和 checkpoint。

限制：不能作为 unseen-epitope 泛化的主结果。

### B. epitope-OOD split：主实验

建议按 peptide 分组，使同一 peptide 在所有 HLA 下只属于一个 split：

~~~text
train peptide ∩ validation peptide = ∅
train peptide ∩ test peptide = ∅
validation peptide ∩ test peptide = ∅
~~~

在可行时同时控制：

- HLA 分布；
- 正负比例；
- TCR 序列相似性；
- 每个 split 的 pMHC 数量；
- validation/test 中可计算 per-pMHC 指标的 group 数。

这是 Stage 2 的主 split。

### C. pMHC-OOD 与 TCR-disjoint 辅助 split

- pMHC-OOD：按 peptide+HLA 组合分组；
- TCR-disjoint：按 TCRαβ exact/cluster 分组，避免同一或高度相似 TCR 跨 split。

两者用于区分模型到底是在记忆 TCR、peptide，还是学到可迁移 interaction signal。

### HITPH 训练规则

- 超参数和 checkpoint 只根据 HITPH validation 决定；
- test 只在模型配置冻结后运行；
- P0 与 P4 至少 3 个 seed；计算允许时使用 5 个；
- 正式结果报告 mean、standard deviation 和 paired/grouped bootstrap CI。

---

## 6.3 IMMREP25：锁模后的外部迁移评测

IMMREP25 不参与：

- 训练；
- early stopping；
- 阈值选择；
- feature/head 选择；
- Pairformer block 数选择。

执行顺序：

1. 完成 IMMREP25 adapter 和 schema audit；
2. 检查与 HITPH 的 exact TCR、peptide、pMHC 和完整 pair 交集；
3. 使用冻结的 Protenix extractor 生成缓存；
4. 加载在 HITPH 上锁定的 P0 与 P4 checkpoint；
5. 输出连续 Prediction、logit、0/1 和 pooled latent；
6. 有公开 label 时运行官方指标；无 label 时只生成合规 submission；
7. 在看到 test label 之前冻结所有规则。

若 IMMREP25 有官方训练/验证数据，可将其作为单独的 in-domain track；但必须与“HITPH 训练 → IMMREP25 zero-shot”结果分开报告。

---

## 6.4 独立 external 数据集

每个 external 数据集单独适配、单独报告，不先合并成一个大测试集。

优先级：

1. 同时含实验正负例的数据集；
2. 有明确 HLA-I、paired TCRαβ 和 peptide 的数据集；
3. 能形成 unseen-epitope 或跨来源评测的数据集；
4. 只有 positive 的数据集最后使用，并明确构建的是 retrieval/decoy benchmark，而不是实验二分类集。

每个 external adapter 必须输出：

- 原始列到统一 schema 的映射；
- 缺失 α链/β链/HLA 的过滤数量；
- label 与 negative 来源；
- 与 HITPH/IMMREP25 的重叠审计；
- HLA 映射成功率；
- 最终 eligible sample 与 eligible pMHC 数。

external 数据集上的任何失败都不得用于重新选择 HITPH 模型；否则它已经变成 validation set。

---

## 7. 评估协议

### 7.1 连续指标

正式报告：

- AUROC；
- AUPRC；
- standardized partial AUROC at FPR≤0.1；
- BCE；
- Brier score；
- calibration curve / ECE；
- per-pMHC AUROC/AUPRC/partial AUC；
- macro 与 micro 两种聚合。

per-pMHC 指标只对同时有正负样本的 group 计算，并报告：

- 总 pMHC 数；
- eligible pMHC 数；
- 每个 group 的 n_pos/n_neg；
- 至少 n_pos≥5、n_neg≥5 的稳健子集结果。

### 7.2 0/1 指标

报告：

- Accuracy；
- F1；
- MCC；
- sensitivity / specificity。

阈值必须由 HITPH validation 选择并写入 checkpoint。默认 0.5 只作为固定参考，不作为唯一结果。阈值选择规则在看到 test 结果前固定，例如最大化 validation MCC。

### 7.3 checkpoint 选择

训练脚本增加 selection_metric 参数。主实验建议：

- specificity ranking 主模型：validation AUPRC；
- 概率校准辅助结果：validation BCE；
- IMMREP25 有官方主指标时，仍只能在 HITPH validation 上用对应指标选模型。

不得在同一 test 上分别挑选“AUROC 最好 checkpoint”和“AUPRC 最好 checkpoint”后拼接报告。

### 7.4 不确定性

- 至少 3 个训练 seed；
- test bootstrap 按 pMHC 分组，而不是只按单条 pair 重采样；
- P0/P4 使用相同 split 和 seed 做 paired comparison；
- 报告均值、标准差、95% CI，不只报告最佳一次结果。

---

## 8. 恢复 FoldAbS specificity 模型结构

## 8.1 当前缺口

当前 TMPTrunkSpecificityHead 已实现：

- frozen s/z/p_dist；
- PairCondition；
- 可配置 Pairformer blocks；
- TMP role-aware mean pooling；
- 三层 MLP；
- BCE 二分类和 latent 输出。

但它仍是面向 TMP 的合理重建，不是完整 FoldAbS specificity public reconstruction。当前缺少一份逐项回答下列问题的论文对齐审计：

- specificity 分支到底使用哪些 trunk/structure/confidence 特征；
- PairCondition 的精确组合方式；
- 4 个 Pairformer 后的 Pool1D/Pool2D 定义；
- 抗体、抗原或界面如何 mask/pool；
- classifier 的层数、宽度、dropout；
- loss、采样、训练步数、scheduler 和 checkpoint metric；
- 是否使用 ESM、coordinates、PAE、ipTM；
- 哪些细节论文未公开。

### 重要原则

developability 分支的 1-block、ESM fusion 和训练协议不能直接当成 specificity 分支配置。两条任务线必须分别审计。

## 8.2 实现策略

### Step 1：先建立论文对齐表

新增：

~~~text
docs/TMP_SPECIFICITY_PAPER_ALIGNMENT.md
~~~

每一项标记：

- paper-confirmed；
- inferred reconstruction；
- TMP adaptation；
- unknown/not disclosed。

如果论文没有公开某项，不把猜测写成作者实现。

### Step 2：新增独立模型类

不直接覆盖 TMPTrunkSpecificityHead。建议新增：

~~~text
FoldAbSSpecificityHeadPublicReconstruction
TMPSpecificityHeadV2
~~~

前者尽可能保持论文公开结构；后者只做必要的实体映射：

~~~text
Ab heavy/light/antigen
→ TCRα/TCRβ/peptide/MHC
~~~

所有 TMP 特有改动在配置和 checkpoint 中显式记录。

### Step 3：4-block 主干

先实现并验证：

- PairCondition；
- 4 个独立 Pairformer blocks；
- paper-aligned pooling；
- paper-aligned classifier；
- BCE/logit 输出；
- pooled latent 输出。

如果论文确认需要其他 tensor，再扩展 extractor 和 feature schema。不要为了“看起来更完整”预先加入未经确认的 geometry/PAE/ipTM。

### Step 4：结构与梯度测试

新增测试至少覆盖：

- s/z/p_dist shape；
- role mask 与链顺序；
- P0/P1/P4 输出 shape；
- frozen cache 不产生梯度；
- PairCondition、Pairformer、classifier 有非零梯度；
- 同一 checkpoint 重载后预测一致；
- latent 拼接顺序和维度稳定；
- 空链、错误 HLA 映射、缺失 feature 明确报错；
- 4-block 的 GPU smoke/OOM 检查。

### Step 5：checkpoint schema v2

checkpoint 保存：

- git commit；
- dataset manifest hash；
- split protocol；
- feature schema/version；
- Protenix checkpoint/version；
- model class 与完整配置；
- seed；
- selection metric；
- validation threshold；
- trainable/frozen parameter counts；
- best step 与 validation metrics。

---

## 9. Stage 2 需要新增的工程组件

建议目录：

~~~text
configs/tmp_stage2/
  hitph_random_p0.yaml
  hitph_epitope_ood_p4.yaml
  immrep25_zero_shot.yaml

src/foldabs_repro/tmp_prediction/
  model_v2.py
  pooling.py
  calibration.py
  audit.py
  adapters/
    hitph.py
    immrep25.py
    external_<source>.py

scripts/
  audit_tmp_dataset.py
  make_tmp_splits.py
  train_tmp_specificity_v2.py
  evaluate_tmp_specificity.py
  compare_tmp_runs.py

tests/
  test_tmp_model_v2.py
  test_tmp_splits.py
  test_tmp_dataset_audit.py
  test_tmp_checkpoint_v2.py
~~~

训练脚本需要补充：

- selection_metric；
- early stopping；
- warmup/scheduler；
- 分参数组学习率；
- resume training；
- full config、git SHA 与数据 hash；
- 每次 validation 的完整 history；
- Brier/ECE/MCC/F1；
- validation threshold calibration；
- 多 seed 汇总。

建议初始学习率分组：

| 参数组 | 初始建议 |
| --- | ---: |
| PairCondition + Pairformer | 1e-4 |
| pooling/classifier | 3e-4 |

这是 Stage 2 的起点，不是最终最优值。weight decay、dropout 和 scheduler 必须只在 HITPH validation 上选择。

---

## 10. 分阶段执行与 Gate

### Stage 2.0：数据与论文审计

交付：

- HITPH/IMMREP25/external adapters；
- HLA canonicalization；
- overlap/leakage report；
- TMP specificity 论文对齐表；
- feature/cache schema v2。

PASS：

- 所有样本来源与 label 类型可追踪；
- 无相同 pair 跨 split；
- epitope-OOD split 无 peptide 交集；
- HLA 序列映射无静默 fallback；
- 外部集没有被用于调参。

### Stage 2.1：P4 工程验证

执行：

- 当前 Level-2 子集；
- P0、P1、P4；
- 固定 seed；
- 相同学习率、step、split；
- 记录显存、step time、最佳 step 和过拟合速度。

PASS：

- P4 无 NaN/OOM；
- checkpoint 可重载；
- latent 可导出；
- train loss 能下降；
- validation 指标不是完全退化；
- 明确 early stopping 区域。

### Stage 2.2：HITPH 正式实验

执行：

- random diagnostic；
- epitope-OOD 主 split；
- pMHC-OOD/TCR-disjoint 辅助 split；
- P0 与 P4 至少 3 seeds；
- 核心表征消融。

PASS：

- 数据与模型配置冻结；
- P4 对主指标的增益在多 seed 下可报告；
- 若 P4 没有稳定优于 P0，仍保留结果，并将 P0 作为紧凑模型；
- 在运行 IMMREP25 前生成 model lock 文件。

### Stage 2.3：模型锁定

新增 model_lock.json，至少写入：

- checkpoint hash；
- git SHA；
- feature schema；
- split hash；
- threshold；
- 预处理版本；
- 主指标；
- 禁止继续调参声明。

### Stage 2.4：IMMREP25 zero-shot

只运行已经锁定的 P0/P4：

- 生成 submission/predictions；
- 导出 latent；
- 按官方协议评估；
- 报告与 HITPH 的 overlap；
- 不回到 HITPH 重新选模型。

### Stage 2.5：独立 external 验证

- 每个数据集单独报告；
- 区分实验负例和 pseudo-negative；
- 报告跨来源性能下降；
- 分析 HLA、peptide 长度、TCR 完整性等 failure modes。

### Stage 2.6：扩展结构信号

只有论文审计和 s/z/p_dist 主线完成后，再开展：

- final geometry；
- PAE；
- ipTM/interface confidence；
- MSA/no-MSA；
- full/clipped HLA 与 pseudo-sequence 消融。

这些属于增强实验，不阻塞 Stage 2 主线。

---

## 11. 推荐的最小运行顺序

1. 完成 specificity 论文对齐审计；
2. 实现 checkpoint/config/history 与数据审计；
3. 在现有 484 条数据上跑 P0/P1/P4 单 seed；
4. 根据 validation 确定 P4 可训练区间和 early stopping；
5. 构建 HITPH epitope-OOD 正式 split；
6. 跑 P0/P4 三个 seed；
7. 完成必要表征消融；
8. 锁定模型与阈值；
9. 运行 IMMREP25 zero-shot；
10. 运行独立 external 数据集；
11. 最后再做 geometry/PAE/ipTM 和 MSA 扩展。

---

## 12. Stage 2 最终需要回答的问题

Stage 2 结束时，不只是回答“模型能不能训练”，而是回答：

1. s、z、p_dist 中哪类冻结 folding representation 对 TMP 最有用？
2. task-specific Pairformer 是否在多 seed、epitope-OOD 下稳定优于无 Pairformer？
3. 4-block 是否比 1-block 带来真实泛化收益，还是只增加过拟合？
4. role-aware interface pooling 是否优于全局 pooling？
5. HITPH 上学习的信号能否 zero-shot 迁移到 IMMREP25？
6. 对不同来源 external 数据集，性能下降主要来自 epitope、HLA、TCR 还是 negative 构建差异？
7. folding 内部 interface representation 是否比 sequence-only baseline 更适合 unseen-epitope generalization？

只有这些问题在冻结数据协议和外部盲测下得到回答，TMP-Prediction 才从工程可运行进入可作为论文实验的阶段。
