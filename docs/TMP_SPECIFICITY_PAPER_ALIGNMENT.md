# FoldAbS specificity head 论文审计与 TMP Stage 1 对照

> 审计日期：2026-10-03（Asia/Shanghai）  
> 论文：用户提供的 `NIPS_2026_foldabs(2).pdf`，标题 *FoldAbS: Repurposing the Protein Folding Model as a Foundation Encoder for Antibody Screening*。以下页码均指 PDF 印刷页码。  
> 论文 SHA256：`e344d5572be2f12b14fbb8a1b1fe067b248b46ed685ade965c30e0116049fbab`。  
> 仓库：`cyan741/FoldAbs_Tm_MSA`，分支 `tmp-specificity`；审计基线 commit：`139641b8c4aa732b78df52dd1ddb0b93b091bd0c`。  
> 范围：审读正文、附录 A.2/A.3/B.1/D.1 和 Figure 1 / Algorithm S1 / Table S6；读取上述 commit 的实际代码及 Stage 1 日志。本文只新增审计记录，不修改模型或历史结果。

## 1. 审计结论

**当前 `TMPTrunkSpecificityHead` 是冻结 trunk 上的 TMP 适配模型，不能称为完整 FoldAbS specificity 复现。** 最大差异不是只少了三个 Pairformer blocks，而是 specificity 所需的结构生成及 confidence 特征链没有执行。

论文确认的 specificity 路径为：

```text
冻结 Protenix-v1
  ├─ trunk → s、z
  └─ diffusion → 预测坐标 X_hat → confidence → PAE、ipTM
                         ↓
              代表原子距离 → geometry distance bins（64 bins，22 Å）
                         ↓
geometry bins + PAE probabilities → PairCondition → z_cond
(s, z_cond) → 4 个 task-specific Pairformer blocks
           → 跨链 pair + 抗体/抗原 single pooling → structural summary
ipTM scalar → 32 维 embedding ──────────────────────┘
           → 融合 → classification MLP → binding logit → sigmoid
```

来源：§3.2/3.3，p.4，行119–142；A.2，p.15，行553–574；A.2/A.3，p.17，行603–627。图中将坐标、PAE、ipTM归到冻结特征链；多个 diffusion sample 如何归约，论文没有说明。

当前 Stage 1 路径为：

```text
冻结 Protenix trunk → s、z
冻结 distogram head(z) → softmax → p_dist
z + p_dist → PairCondition → 0/1/K Pairformer
          → 四个实体均值 + 五个有向界面均值 → 2176 维 latent
          → LN + 三层 Linear MLP → logit → sigmoid
```

论文明确区分 `p_dist`（trunk distogram head 的输出）与 `d_geom`（预测坐标导出的距离特征）。**即使两者都有64个通道，也不能互换名称或认定等价。** `s/z/p_dist` 是论文 developability 的特征集合；当前模型采用这一类条件信号，但它的四实体/五界面读出是 TMP 自定义，亦不是完整 developability 复现。

本次审计还修正 Stage 2 的证据状态：geometry、PAE、ipTM 已由用户提供的原文确认是 specificity 组成部分；若保留无 diffusion 的轻量 TMP 主线，应将其命名为 trunk-only adaptation，并用结构增强分支检验收益。

## 2. 证据等级与逐项对齐

- **paper-confirmed**：该版本论文明确说明。
- **code-confirmed**：审计 commit 的仓库代码或日志直接确认。
- **inferred reconstruction**：合理重建选择，原文未披露精确实现。
- **TMP adaptation**：为了 TCR–pMHC 任务主动调整。
- **unknown**：原文及当前审计范围不能确定。

| 组件 | 原文 specificity | 当前 Stage 1 | 审计判定 |
| --- | --- | --- | --- |
| Backbone | 冻结 Protenix-v1 | 提取缓存时使用 v1.0.0 checkpoint；训练只读取缓存 | 总体对齐；权重身份及运行配置仍需记录 |
| `s/z` | `C_s=384`、`C_z=128` | 从缓存读取通道数 | 对齐维度设计 |
| 坐标生成 | 预测坐标参与几何条件；mini-rollout 32 samples / 20 denoising steps | 直接调用 `get_pairformer_output`，不调用 diffusion | 缺失结构特征链 |
| geometry bins | 代表原子距离，64 bins，最大22 Å | `softmax(distogram_head(z))` | 信号不同；不能视作同一组件 |
| PAE | confidence head logits转概率，64 bins，进入 pair condition | 不提取、不输入 | 缺失 |
| PairCondition | LN、linear projection、residual transition；2个 transition layers | LN(`p_dist`) + Linear → 加到 `z` → 2个 residual transitions | 层数/组件类别相近，输入及精确拓扑不对齐 |
| Pairformer | 4-block task-specific stack，共同更新 `s/z` | 调用 Protenix `PairformerBlock`；默认4，已有P0/P1实验 | 结构族相符；P0/P1属于深度消融，P4尚非已完成实验 |
| Pooling | 跨链 pair、抗体/抗原 single；pooling layers列入 head 参数描述 | 四实体 + 五有向界面，无参数 mean pooling | 池化对象部分对应；公式未确认，TMP adaptation |
| ipTM | 标量嵌入为32维，分类MLP前融合 | 无 | 缺失 |
| Classification | MLP → scalar logit → sigmoid | LN + `2176→512→512→1`、GELU、dropout | 输出目标对齐；层宽/层数是重建选择 |
| Sequence LM | 未在 specificity 主线说明 ESM融合 | 不使用 ESM | 无须为 specificity 主线补 ESM |
| 监督/采样 | BCEWithLogits；正:伪负=1:2；抗体cluster-aware采样 | BCEWithLogits，静态manifest、shuffle；已跑小集1:1 | 损失类别对齐，采样协议不同 |
| 输出latent | pooling summary；affinity继承分类前 hidden | `encode()`输出分类MLP前拼接向量 | 不是论文已确认的倒数第二hidden层 |

## 3. 特征来源：`p_dist`、geometry、PAE、ipTM

### 3.1 当前缓存实际保存什么

`scripts/extract_protenix_embeddings.py` 第159–184行：

```python
_s_inputs, s, z = runner.model.get_pairformer_output(...)
p_dist = torch.softmax(runner.model.distogram_head(z), dim=-1)
```

保存 `sample_name`、`token_asym_id`、`token_residue_index`、`s`、`z`、`p_dist`；浮点缓存转为 CPU float16。`dataset.py` 将三类embedding加载为float32。extractor 默认 `cycles=1`，MSA只有传入 `--use-msa` 才启用，template禁用。不能仅从默认值推断历史缓存实际使用了什么，应查该缓存的 extraction metadata。

runner构造中的 `n_step=1/n_sample=1` 没有触发采样：实际执行路径停在 trunk。因此这两个参数也不能当作论文32 samples / 20 steps的缩小版 rollout 实验。

### 3.2 论文 specificity 的几何特征

§3.2 Eq.(3–4)和A.2行564–565确认：先取得预测坐标，再计算代表原子间距离，离散为64 bins，最大22 Å。

可用于开发的抽象表达：

```text
representative_coordinates: [N,3]
D_ij = ||x_i - x_j||
d_geom = Discretize(D): [N,N,64]
```

上式是单个预测结构的形状示意。**论文未给出**代表原子究竟选CA、CB或其他规则，分箱最小值/边界/溢出桶，编码是否严格one-hot，以及32个samples是平均分布、先选结构还是逐sample预测再聚合。不得默认 `linspace(0,22,64)` 或硬编码CA并标为paper-confirmed。

`p_dist`则对每一对token给出trunk预测的距离概率，即使未产生坐标也能计算。它不是从一套坐标测出的离散距离；其bin定义也不能自动继承specificity的22 Å设置。

### 3.3 PAE与ipTM

- PAE是confidence head的预测aligned-error分布，不是真实结构误差标签；论文确认 logits→probabilities，64 bins，并投影到pair空间。
- geometry提供相对几何，PAE提供对相对位置的置信信息；二者语义不同。
- ipTM是额外的scalar confidence feature，嵌入为32维，在结构池化之后、最终MLP之前融合。
- 缓存只有 `s/z/p_dist` 时，不能把 `p_dist`当作PAE，或用其熵冒充原文ipTM；论文对齐分支需补充diffusion/confidence执行链。

**原文表述不完全闭合**：主文 `R_spec={s,z,d_geom,p_PAE}`未枚举ipTM，Algorithm S1也未展开ipTM；A.2行571–574与D.1行835–836明确写出ipTM。因此判定为“正文简式省略、附录确认”，不能据简式认定specificity不使用ipTM。

## 4. PairCondition：已确认组成，未确认精确计算图

论文Eq.(29–31)：

```text
c_spec = [d_geom, p_PAE]
z_cond = z + PairCond(z, c_spec)
```

A.2确认LN、linear projection、residual transition blocks，行607–608确认两个transition layers。输入两类信号及模块类别可确认；精确操作顺序、激活函数、transition expansion、linear bias、初始化等未披露。

另一个需要实现前固定的歧义：文字先说PAE投影到pair空间，再说两个信号拼接；Eq.(30)只写抽象拼接。因此“先拼接64+64=128再整体投影”是自然的重建方案，不是论文逐算子确认的唯一方案。

当前 `model.py` 第18–43行明确为：

```text
v0 = z + Linear_64→128(LN_64(p_dist))
v1 = v0 + Linear_256→128(GELU(Linear_128→256(LN_128(v0))))
v2 = v1 + Linear_256→128(GELU(Linear_128→256(LN_128(v1))))
```

当前forward返回完整 `v2`，不是仅返回增量。若V2沿用这段实现，外层不得再次执行 `z + pair_condition(...)`，否则重复加 `z`。论文增量接口与代码完整输出接口必须明确区分。

**P0仍包含可训练PairCondition和MLP。** P0/P1只隔离额外Pairformer block，不能将P0称为raw `z` + MLP。若要隔离条件化收益，需另建显式no-condition消融。

## 5. Pairformer的结构及当前实现

### 5.1 论文能确认到哪一级

能确认：4个task-specific blocks，输入 `s/z_cond`，输出共同refine后的 `s/z`，`C_s=384/C_z=128`，下游模块可训练。

不能从该PDF确认：每个block的attention heads、triangle hidden维度、dropout、transition expansion、gating/normalization逐算子细节、block初始化是否来自预训练trunk、是否共享权重、activation checkpointing以及具体Protenix源码版本。

### 5.2 仓库及上游源码确认的block

仓库 `model.py` 第83–95行直接构造Protenix `PairformerBlock`，`ModuleList`各block独立构造；未加载trunk block权重，也未绑权重。

辅助审阅上游官方 `bytedance/Protenix` 的 `protenix/model/modules/pairformer.py`，固定源码参考commit：`f2ba5d04b085060151ec7629a41ad33a7ca5ab32`。该参考用于说明标准block，不代表作者使用版本或Stage 1本地安装版本。仓库 `requirements-extras.txt` 要求单独安装Protenix，却没有固定版本；历史环境应另行核实。

一个block的非inplace路径为：

| 顺序 | 操作 | 更新对象 | 输出形状 |
| --- | --- | --- | --- |
| 1 | Triangle multiplication outgoing + rowwise dropout + residual | `z` | `[B,N,N,128]` |
| 2 | Triangle multiplication incoming + rowwise dropout + residual | `z` | 同上 |
| 3 | Triangle attention starting node + rowwise dropout + residual | `z` | 同上 |
| 4 | 转置pair轴，ending node attention，再转回 | `z` | 同上 |
| 5 | Pair transition + residual | `z` | 同上 |
| 6 | AttentionPairBias：更新后的 `z` 调制single self-attention | `s` | `[B,N,384]` |
| 7 | Single transition + residual | `s` | 同上 |

直觉：`z_ij`可通过第三点k利用 `z_ik/z_jk` 等三角关系；更新后的pair表征再影响token注意力。single attention的简化表达为 `softmax(QK^T/sqrt(d_k) + bias(z))V`。这是标准block机制解释，不是PDF中给出的完整specificity伪代码。

**不要补一个single→pair outer product并称为PairformerBlock。** 所审阅block先更新pair，再pair→single；没有同一block内由更新后的single显式生成pair的outer-product子层。MSA模块里的OuterProductMean是另一组件。

当前构造参数：

| 参数 | Stage 1代码默认 | 证据状态 |
| --- | ---: | --- |
| `n_heads`（single attention） | 16 | code-confirmed；论文未单列 |
| `c_hidden_mul` | 128 | code-confirmed；论文未单列 |
| `c_hidden_pair_att` | 32 | code-confirmed；论文未单列 |
| `no_heads_pair` | 4 | code-confirmed；论文未单列 |
| `dropout` | 0.1 | 同一参数传给block和classifier；不是论文确认的specificity值 |
| blocks | 默认4；日志P0=0、P1=1 | 默认与论文层数一致，历史小实验尚不是P4 |

调用时 `s/z`加batch轴，`pair_mask=None`、torch triangle kernels、`inplace_safe=False`、`chunk_size=None`。角色mask只用于最后pooling，**不会限制Pairformer只处理五个界面**。所有token pairs均可参与更新；当前单样本无padding解释了为什么未给padding mask，但不等于有contact-aware mask。

## 6. Pooling：论文缺少足以唯一复现的公式

### 6.1 原文披露与空白

A.2 p.15行569–574确认pool跨链pair和抗体/抗原single，形成interface-aware summary；并将pooling layers列入可训练head的参数描述。Figure 1只有Pooling & MLP的模块框。

该PDF没有说明：

1. mean、max、attention或contact-weighted pooling；
2. 抗体H/L链先合并还是分别池化；
3. cross-chain是否专指抗体–抗原，是否包含H–L；
4. `z_ij/z_ji`保留一个方向、双向分别池化还是平均；
5. 是否按预测距离、PAE、CDR、affinity-token mask选择界面；
6. pooled summary维度，及pooling中哪些投影/权重可训练。

“pooling layers属于参数”提示作者pooling路径可能有可训练部分，但不能据此唯一推导attention pooling，也不能宣布mean pooling一定错误。B.1的affinity-token masks说明有下游mask数据，不足以推导它们如何进入池化。

developability的 `Pool1D/Pool2D`也是抽象记号；明确mean-pooled的是ESM heavy/light embeddings（p.17行597）。不能把这句话扩大到specificity的 `s/z` mean pooling。

### 6.2 当前Stage 1：可精确确认

`interfaces.py` 第22–58行是无参数masked mean；`model.py` 第134–140行固定拼接顺序。

设实体a的token集合为 `I_a`：

```text
s_a = sum_{i∈I_a} s_i / |I_a|
z_a→b = sum_{i∈I_a,j∈I_b} z_ij / (|I_a|*|I_b|)
```

| 类型 | Pool | 通道数 |
| --- | --- | ---: |
| single | TCRα、TCRβ、peptide、MHC各一个 | `4*384=1536` |
| directed pair | α→peptide、β→peptide、α→MHC、β→MHC、peptide→MHC | `5*128=640` |
| 拼接 | 上述固定顺序 | `2176` |

若α有2个token、peptide有2个token，`z_α→peptide`是四条pair向量的均值，每条权重1/4。无距离筛选、无CDR筛选、无可学习attention权重；远距离pair同样进入均值。

未池化反向pair、α–β或链内pair；但其信息仍可能经Pairformer传播到已选择界面。`z`无需对称，因此有向pooling不能自动视作双向平均。

role mask通过排序后的 `token_asym_id` 与 `tcra,tcrb,peptide,mhc`顺序对应；这是输入顺序假设，后续adapter必须验证token→role映射。

结论：这是清晰的TMP adaptation基线，应该保留。它不是原文已确认的2176维读出，也不能将五个界面数视为FoldAbS要求。

## 7. 分类头与latent：不要借用其他任务的超参数

原文specificity确认分类MLP、logit/sigmoid和32维ipTM融合。**没有明确给出specificity MLP层数、hidden dimension、activation、dropout或融合算子。**

- `3-layer / hidden=512 / dropout=0.1`明确属于developability（p.17行597–598）。
- `3-layer / hidden=128 / dropout=0.1`明确属于affinity regression MLP（p.16行589）。
- 二者不能直接当成specificity的原文设置。

当前代码为：

```text
latent(2176) → LN → Linear(512) → GELU → Dropout
             → Linear(512) → GELU → Dropout → Linear(1)
```

`hidden_dim=512/dropout=0.1`是代码默认值，不代表历史每个run的checkpoint一定用了默认值。`predict_tmp_specificity.py`保存的是classifier前2176维latent，而论文affinity继承的是specificity最终分类层之前的hidden representation（§3.3、A.2 Eq.(32)）。若用于迁移，应显式区分 `pooled_structural_latent`、`confidence_fused_latent`、`classifier_hidden`，不要统称论文latent。

重建分支若把ipTM 32维简单拼到当前2176维latent，输入为2208；这是TMP重建选择，原文没有确认2208维。

## 8. 训练协议及Stage 1实际运行对照

| 项目 | 原文specificity（A.3/B.1） | 当前代码/已有日志 |
| --- | --- | --- |
| Objective | BCE with logits | 相同；可选`auto-pos-weight`是额外实现选择 |
| Optimizer | AdamW | 相同 |
| Learning rate | `1e-4` | 脚本默认相同；Level 2 P0/P1日志实际`3e-4` |
| Batch / accumulation | batch=1，累积3步，8张H20 | 静态读取代码batch=1、累积默认3；Level 2日志world_size=1、effective batch=3 |
| Optimization steps | 4000 | 脚本默认4000；历史总步数需看run日志，不用默认替代 |
| 正:负 | 1:2 | Level 2：194正/194负，即1:1 |
| Sampling | cluster shuffle，每cluster随机一个positive，再生成2个伪负 | 静态manifest + shuffle，非相同训练采样协议 |
| MSA/template | 所有链预计算MSA；抗体UniRef50、抗原标准流程；template禁用 | MSA可选、template禁用；缓存实际MSA状态需查metadata |
| Cropping | 训练长抗原随机连续256 aa，验证/测试不crop | 当前head读取固定缓存，不实施此在线crop协议 |
| Validation | 独立validation；伪负确定性缓存 | 静态manifest；以validation BCE最小选checkpoint |
| Threshold | 没有明确特定阈值/校准协议 | checkpoint写死0.5；predict可覆盖阈值 |

论文没有在specificity段明确weight decay、gradient clipping、scheduler、early stopping、validation频率和selection metric。当前脚本weight decay默认0.01、clip=1.0、每100步验证、最小BCE选模，属于代码选择；Level 2日志实际weight decay=1e-4。不能把affinity/developability段的weight decay或warmup/scheduler移植后标成specificity原文设置。

batch=1在8卡配置下是否per-device/global、全局effective batch究竟多少，原文未说清；不直接声明effective batch=24。

伪负来源：cross-antigen mismatch、CDRH3 mutation、natural CDRH3 swap；A.1行539–546给出负池比例3:1:6，B.1说明cluster-aware采样及固定validation负例。该比例是抗体训练协议，不能未经生物学验证直接移植为TCR CDR3交换/突变标签。生成伪负始终不是实验non-binder观测。

审计发现一个追溯差异：仓库Level 2日志包含DDP/world_size字段，但当前读取的训练脚本没有相应分布式逻辑。日志可证明这些run的运行参数，却不足以证明历史可执行脚本与本commit逐字一致。Stage 2必须记录git SHA、训练环境和checkpoint配置。

## 9. Table S6：消融支持什么、不支持什么

来源：p.23 Table S6及D.1行830–848。

| Origin-1设置 | Macro-antigen AUROC | EF@1% | EF@5% | EF@10% |
| --- | ---: | ---: | ---: | ---: |
| Full FoldAbS-Px | 0.757 | 4.047 | 3.799 | 3.182 |
| Native Protenix-v1 | 0.669 | 3.872 | 2.052 | 2.374 |
| w/o geometry distogram | 0.317 | 0.048 | 0.094 | 0.257 |
| w/o PAE conditioning | 0.718 | 2.815 | 2.910 | 2.420 |

1. 支持在该抗体benchmark中显式预测几何的重要性：去geometry后出现很大的性能下降。
2. `w/o PAE conditioning`同时移除PAE和ipTM，**不是PAE单因素消融**；其差值不能单独归因于PAE。
3. 本表没有P0/P1/P4深度对照，不能用它证明4-block最优或Pairformer本身带来多少收益。
4. 这些结果不能直接推断TCR-pMHC收益，也不能等同于Stage 1 trunk-only模型的预期分数。0.317的异常大下降值得在复现中检查，不凭表格推断原因。

结合TMP核心问题，未来至少分别比较trunk-only、geometry-enhanced、PAE-only、ipTM-only及full条件；同时保留P0/P4。这样才能区分task adaptation、显式几何与confidence信号的贡献。

## 10. Stage 2实施建议与验收条件

### 10.1 保留历史路径，分开命名

1. 保留 `TMPTrunkSpecificityHead` 与schema `protenix_trunk_s_z_pdist_v1`，保证旧checkpoint可加载。
2. 新建 `FoldAbSSpecificityHeadPublicReconstruction`：采用论文确认的geometry+PAE condition、4 blocks、interface pooling、32维ipTM融合及logit；未披露的pooling/MLP等在config中标成重建选择。
3. 新建 `TMPSpecificityHeadV2`：显式适配α/β/peptide/MHC角色与TMP界面；既支持trunk-only也支持结构增强配置，不把原文未给出的实现称为忠实复现。
4. 不以“P4层数对齐”替代feature/rollout/pooling/训练对齐；不覆盖历史类。

### 10.2 新特征schema先于新head

论文对齐分支至少需要：`s/z`、代表原子及valid mask、geometry bins、PAE probabilities、ipTM、token→role映射，以及明确的sample轴/归约协议。保存bin定义、Protenix代码/权重身份、MSA/template/recycle/diffusion设置与seed。

原始坐标是否长期保存可按成本决定，但必须能追溯geometry来源。`s_inputs`不是论文声称的分类输入，但调用Protenix的diffusion/confidence可能需要它作为中间输入；不能因为Stage 1不消费它，就断言新的extractor仍不需要它。

现有缓存可继续用于trunk-only实验；不能将缺失geometry/PAE/ipTM补零后标为full reconstruction。是否能在完全一致trunk配置下追加生成结构，应检查所安装Protenix API及输入/随机状态，不假定当前`.pt`独立足够。

### 10.3 论文未披露部分必须形成决策记录

| 未确认项 | 实施前需要固定的决定 |
| --- | --- |
| 代表原子、64 bins边界 | 原子选择、缺失mask、下界、上界与overflow规则 |
| 32 samples的归约 | 选结构/平均features/平均logits；避免坐标与PAE错配 |
| PAE投影/condition顺序 | 先投影再拼接或拼接后投影；模块返回增量还是完整`z_cond` |
| Pairformer细节 | 固定Protenix版本、独立/共享权重、初始化、dropout、chunk/checkpoint设置 |
| Pooling | 抗体/抗原或四实体分组，pair方向、mean/learned、contact/CDR masks |
| MLP和ipTM融合 | 隐藏维度、层数、activation、dropout、融合算子 |
| Training缺失超参数 | weight decay、scheduler、early stopping、selection metric和阈值校准 |

这些决定可先使用简单可审计版本，不必等待作者代码才能推进TMP，但应持续标为inferred reconstruction/TMP adaptation。

### 10.4 最小验收

- shape/mask/bin/probability检查：有效token、pair/sample轴一致；PAE softmax仅一次；geometry不误用trunk distogram。
- tiny-set检查：旧P0/P1和新P4可训练；检查PairCondition输出无重复残差、head梯度和冻结边界。
- feature ablation独立控制geometry、PAE、ipTM；pairformer深度消融共享相同feature schema、split和seed。
- 核实32×20 rollout在RTX3090上的时间和显存；降采样时命名为低预算重建配置，记录与论文差异。
- 正式评测仍以HITPH epitope-OOD及多seed为主；IMMREP25/external只在锁模后评测。

## 11. 复现边界及来源索引

该PDF的NeurIPS checklist行1182–1187明确：代码开放问题回答No，称接收后公开；A.1行546却提到released code/scripts。当前版本存在表述不一致。本文仅按上传PDF及第三方复现仓库审计，不据此宣称作者实现已经可得，也未核查另一个最新公开发布。

论文证据索引：

| 位置 | 确认内容 |
| --- | --- |
| Figure 1，p.3 | trunk、denoising、confidence与task heads总体关系 |
| §3.2，p.4，行119–131，Eq.(3–4) | `p_dist`与`d_geom`来源不同、冻结边界 |
| §3.3，p.4，行135–142，Eq.(5–6) | specificity feature set、logit/sigmoid |
| A.2，p.15，行553–574，Eq.(29–31) | condition、4 blocks、跨链pooling、ipTM 32维 |
| Algorithm S1，p.16 | spec/dev两条feature路径 |
| A.2，p.17，行603–612 | `384/128`、64+64 bins、22 Å、2 transitions、32 samples/20 steps |
| A.3，p.17，行619–627 | specificity BCE、AdamW、LR、batch、累积、4000步、8 H20 |
| B.1，p.18–19，行679–704 | cluster采样、1:2、crop、MSA、template、mask |
| Table S6/D.1，p.23 | geometry、PAE+ipTM联合移除的消融 |
| Checklist，行1182–1187 | 作者代码可得性的该版本声明 |

仓库证据（以下链接固定到审计commit，避免分支更新后证据漂移）：

- [model.py](https://github.com/cyan741/FoldAbs_Tm_MSA/blob/139641b8c4aa732b78df52dd1ddb0b93b091bd0c/src/foldabs_repro/tmp_prediction/model.py)：PairCondition、Pairformer构造、2176维读出、classifier。
- [interfaces.py](https://github.com/cyan741/FoldAbs_Tm_MSA/blob/139641b8c4aa732b78df52dd1ddb0b93b091bd0c/src/foldabs_repro/tmp_prediction/interfaces.py)：role mapping与无参数有向mean pooling。
- [extract_protenix_embeddings.py](https://github.com/cyan741/FoldAbs_Tm_MSA/blob/139641b8c4aa732b78df52dd1ddb0b93b091bd0c/scripts/extract_protenix_embeddings.py)：trunk-only执行链及缓存。
- [dataset.py](https://github.com/cyan741/FoldAbs_Tm_MSA/blob/139641b8c4aa732b78df52dd1ddb0b93b091bd0c/src/foldabs_repro/tmp_prediction/dataset.py)：必需schema与float32加载。
- [train_tmp_specificity.py](https://github.com/cyan741/FoldAbs_Tm_MSA/blob/139641b8c4aa732b78df52dd1ddb0b93b091bd0c/scripts/train_tmp_specificity.py)：默认参数、loss、选模、固定0.5阈值。
- [predict_tmp_specificity.py](https://github.com/cyan741/FoldAbs_Tm_MSA/blob/139641b8c4aa732b78df52dd1ddb0b93b091bd0c/scripts/predict_tmp_specificity.py)：classifier前latent导出。
- [P0运行日志](https://github.com/cyan741/FoldAbs_Tm_MSA/blob/139641b8c4aa732b78df52dd1ddb0b93b091bd0c/runs/tmp_stage1_level2_p0/train.log)、[P1运行日志](https://github.com/cyan741/FoldAbs_Tm_MSA/blob/139641b8c4aa732b78df52dd1ddb0b93b091bd0c/runs/tmp_stage1_level2_p1/train.log)：已执行的配置与样本数量。
- [Protenix官方Pairformer源码参考](https://github.com/bytedance/Protenix/blob/f2ba5d04b085060151ec7629a41ad33a7ca5ab32/protenix/model/modules/pairformer.py)：标准block机制；不是作者版本确认。

本审计未运行新的训练或重新评测；所有原文结果均来自上传PDF，所有Stage 1实现判定均以固定commit为准。下一步是基于明确的缺失项实现结构增强extractor及独立V2 head，并通过OOD消融检验它们是否服务于TMP研究问题。
