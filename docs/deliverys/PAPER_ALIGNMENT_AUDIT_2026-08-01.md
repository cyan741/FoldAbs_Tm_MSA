# FoldAbS 热稳定性实验逐项对齐审计（2026-08-01）

本审计只覆盖 FoldAbS 的可开发性/热稳定性主实验，不讨论 MSA 加速。信息按三类处理：

- **论文明确**：原文直接给出，可以据此检查代码；
- **确定不一致**：论文明确而当前实现不同，已经修改或提供替代入口；
- **论文未公开**：不得把猜测当成作者实现，本轮保持现有实现并记录边界。

## 1. 原论文中与本实验直接相关的位置

| 原文位置 | 与热稳定性实验相关的信息 |
| --- | --- |
| 3.1-3.4 | 抗体输入包含序列和预计算 MSA；Protenix 固定；可开发性为单属性回归；优化回归损失 |
| Table 4 / 4.3 | 热稳定性 337/30/63；ESM-650M test Spearman 0.784；FoldAbS-Px 0.845 |
| Appendix A.2 | `{s,z,p_dist}`、64-bin distogram、2 个 PairCond transition、1 个 Pairformer、冻结 ESM-2 650M、重/轻链 mean pooling、3 层 512 MLP、dropout 0.1 |
| Appendix A.3 | MSE；AdamW lr 1e-4、wd 0.01；梯度累积 3；clip 1.0；最多 5000 steps；warmup 100；每 100 steps 验证；LR patience 50/factor 0.5/min 1e-5；验证 MSE 选 checkpoint |
| Appendix B.1/B.3, Tables S4-S5 | 抗体 MSA 使用 UniRef50；热稳定性仅由 FLAb 公开来源 Jain2017、Jain2024、Valdes-Tresanco 2023 构成；DSF；430 条；重复重/轻链取中位数；训练集 z-score；按唯一重链划分；训练标签 69.1 +/- 9.0 C |
| Appendix C.3 | ESM-650M 等可开发性基线：固定长度池化表征 + 轻量 MLP；batch size 32、lr 1e-4、3000 steps |
| Appendix C.4 | 可开发性正式主指标为每个属性的 test Spearman rho |
| Appendix D.3 / Table S8 | 热稳定性累计消融：single 0.458 -> distogram 0.621 -> Pairformer 0.720 -> ESM fusion 0.845 |
| Appendix E.1 / Table S10 | ESM 8M/35M/150M/650M/3B 的热稳定性 rho 为 0.641/0.765/0.796/0.784/0.814 |

## 2. 论文完整 FoldAbS 热稳定性头与当前代码

| 项目 | 论文明确 | 当前实现 | 判断/动作 |
| --- | --- | --- | --- |
| Protenix | v1，冻结 | v1.0.0，离线提取冻结特征 | 对齐 |
| 输入 | 抗体 `{s,z,p_dist}` | 相同 | 对齐 |
| 通道 | `c_s=384`, `c_z=128` | 从缓存张量读取，源自 Protenix-v1 | 对齐；运行时仍应在 index 中核验形状 |
| distogram | 64 bins，softmax 后线性投影 | 提取时保存 softmax 概率，训练时 LN + Linear | 主要对齐 |
| PairCond | LN、线性投影、2 个 residual transitions；记作 `PairCond(z,c)` | 默认对条件分支做 2 transitions 后加到 `z` | **精确内部组合未公开**，保持现状，不改 |
| Pairformer | 1 block | Protenix `PairformerBlock` 1 block | 对齐 |
| 池化 | `Pool1D/Pool2D` | 无 padding 的样本内均值 | **精确定义未公开**，保持现状，不改 |
| ESM | 冻结 `esm2_t33_650M_UR50D` | 同一 checkpoint，最后一层 residue mean | 对齐；具体层未另行公开 |
| 链融合 | `e_H,e_L` 分别 mean-pool 后拼接 | 相同；纳米抗体 `e_L=0` | 重/轻链对齐；纳米抗体空轻链策略未公开，保持现状 |
| 回归头 | 3 层，hidden 512，dropout 0.1 | 相同 | 对齐 |
| 目标 | 训练集 z-score 后 MSE | 相同 | 对齐 |
| 优化 | AdamW lr 1e-4, wd 0.01, accum 3, clip 1 | 相同 | 对齐 |
| 调度 | warmup 100；val/100；patience 50；factor .5；min 1e-5 | 相同 | 对齐 |
| 训练长度 | 最多 5000 steps | 默认 5000 | 对齐 |
| 模型选择 | validation MSE | 相同 | 对齐 |
| early stopping | 原文未明确给出 early-stopping patience | 默认关闭 | 不应再声称“缺少论文 early stopping” |

## 3. ESM-650M 基线：已经确认的程序问题

论文 Table 4 的 ESM-650M 热稳定性 rho 为 **0.784**。此前的
`run_public_dsf_esm2_only_control.sh` 是为了和当前 FoldAbS 训练过程做诊断匹配，使用
batch size 1、梯度累积 3、5000 steps。它不是 Appendix C.3 的 ESM 官方协议重构。

Appendix C.3 明确要求 baseline 使用：

- batch size 32；
- learning rate 1e-4；
- 3000 optimization steps。

因此新增 `run_public_dsf_esm650m_paper_baseline.sh`。它强制上述三项，并复用已有
FoldAbS 运行的 `split_manifest.csv`。旧入口保留，仅作为 batch-size-1 匹配诊断，防止
历史实验含义被悄悄改写。

仍未公开、所以没有擅自修改的 ESM baseline 细节：

1. baseline 是完全冻结 ESM 还是只把“fine-tuning”用于描述回归头训练；
2. 重/轻链在 standalone baseline 中的精确拼接方式；
3. 轻量 MLP 的精确层宽、激活与 dropout；
4. baseline 专用 scheduler 和 checkpoint 选择指标。

当前保持“冻结最后层 residue mean；重/轻链分别池化；3 层 512 MLP；验证 MSE 选模型”。
这些必须在结果中标注为 reconstruction choices。

## 4. 数据：能确认的事实与不能确认的部分

论文热稳定性数据 **不包含 in-house 数据**。Table S4 明确列出的来源是 FLAb 中的
Jain2017、Jain2024 和 Valdes-Tresanco 2023。当前公开重建的三部分为：

- Jain2017 DSF：137；
- Jain2024 DSF：43；
- NbThermo DSF (SYPRO)：253；
- 精确重/轻链中位数去重后：433。

当前 433 条的总体标签均值/总体标准差为 69.150/9.247 C；论文给出的 69.1/9.0 C 是
**训练集**统计量，二者不能直接当成同一个量比较。论文最终为 430 条、337/30/63，当前为
433 条、339/30/64。作者没有公布 430-row manifest 和 split manifest，因此不能安全删除
三条来“凑齐”论文，也不能把当前 split 当作官方 split。

确定已经对齐：DSF assay、精确重/轻链组合取标签中位数、只用训练集统计量 z-score、唯一重链划分。需要进一步核验：公开来源中的序列是否已经是与 FLAb 相同的 Fv；论文写明 FLAb 的 Fv 直接使用，ANARCI/Chothia 仅用于其内部全长抗体。未知而不改：精确三条排除项和每个 split 的样本身份。

## 5. MSA：已确认的输入来源不一致

论文的抗体 MSA 构建写明只搜索 UniRef50。当前所谓“普通 MSA”来自 ColabFold 公共服务，
数据库组成并不等同于 UniRef50-only。因此旧结果只能写作 **ColabFold-MSA 近似重构**，不能
写作论文精确普通 MSA。

已新增：

- `run_public_dsf_uniref50_msa_cache.sh`：给定本地 UniRef50 FASTA，用 jackhmmer 建缓存；
- `run_public_dsf_uniref50_msa_features.sh`：以相同 Protenix 表征提取流程生成特征；
- `run_public_dsf_developability.sh msa_uniref50`：训练并在 JSON 中记录 MSA 来源。

论文未给出 UniRef50 版本、数据库快照、jackhmmer iteration/E-value/max-hits；这些参数仍是
显式可配置的重构参数，写入 `cache_report.json`，不冒充作者原始设置。

## 6. 暂不修改的可能问题

下列项可能影响 rho，但论文证据不足，本轮不改：

1. PairCond 是否应在 transition 内联合使用 `z` 与条件特征；
2. Pool1D/Pool2D 是否含特殊 mask、链权重或分链池化；
3. Protenix recycle/cycle 数（当前为 10）；
4. ESM baseline 的精确小头结构与是否解冻 ESM；
5. 训练随机种子、Protenix 推理 seed 和特征保存精度；
6. 纳米抗体缺失轻链的精确作者处理；
7. UniRef50 的精确版本与搜索参数。

## 7. 下一次运行顺序（不涉及 MSA 加速）

1. 先运行 Appendix C.3 ESM-650M baseline reconstruction；这是当前最高优先级。
2. 比较新的 ESM rho 与旧的 0.566，而不是先继续调结构融合。
3. 准备 UniRef50-only 常规 MSA 后重提 Protenix 特征。
4. 在同一固定 manifest 上运行完整 FoldAbS 与 Table S8 累计消融。
5. 只有 ESM baseline 与结构主线都完成后，再讨论 MSA 加速。
