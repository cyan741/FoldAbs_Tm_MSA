# HiTPH peptide错配与split内天然配对TCR候选池执行规范

日期：2026-10-08。分支：tmp-specificity。本文规定后续实现和统计；没有宣称已生成新负例或改变数据划分。

## 1. 用户确定的新方案

先增加pep_mismatch：
对正例(t,p,h)，保留天然配对TCR t=(α精确序列,β精确序列)与HLA h，换成同split、同HLA下已有正呈递/识别记录的其他peptide。排除所有splits中的已知阳性，统计每条正例容量及去重总容量。

若不能满足配置的负例需求，则放弃“只允许同HLA donor”的限制，转为每个split的天然配对TCR池：固定目标正例的pep/HLA，从该split全体天然αβ对随机抽取，不要求donor原来属于相同HLA；使用全局(pep,tcr)阳性blacklist，排除重复负组合。

此处“放弃固定pMHC替换TCR思路”按实际指定算法解释：放弃原同HLA限制的donor池；抽样算法仍保留目标pMHC并替换天然TCR。不能将其实现为任意peptide-HLA乱配或拆散αβ。

不硬性要求1:2。候选池与blacklist独立于每epoch抽样数量；比例是可配置的训练暴露比例和有限候选池预算，不能混淆为独立负例数。

所有生成标签为synthetic/pseudo-negative。排除已知阳性不能排除未知交叉反应。IMMREP25等外部实验测试不重新错配，不参与候选池设计。

## 2. 输入、身份与全局blacklist

输入：
- 固定 positive_splits.csv；val仅适配为validation。
- 全部clean positives用于排除，不限于当前split和种子样本。
- 若有额外已知阳性，可显式合并来源、hash与范围，不读取外部测试隐藏标签。

身份：
- tcr_id = SHA256(alpha_sequence + 明确分隔/长度编码 + beta_sequence)；先规范空白/大小写但不改变残基。
- peptide_id由精确序列确定。
- hla_id保留当前规范名称及对应hla_sequence/边界；当前步骤不自动合并高字段HLA。
- 负例唯一complex_key由alpha、beta、peptide、实际HLA输入序列定义；HLA别名映射到相同输入时不得重复计数。

blacklist B={(peptide_id,tcr_id):在任意内部split、任意HLA存在已知阳性}。
该跨HLA排除比只检查当前三元组更保守。blacklist全局共享用于只读排除；donor和peptide池仍在本split构建，不借其他split对象。

另外保存：
- positives_by_peptide：peptide_id→阳性tcr_id集合，用于快速拒绝。
- known_positive_complexes：审计精确四链矛盾。
- generated_negative_registry：唯一负complex_key、类型、split、来源与生成轮次；它与阳性blacklist是不同对象，不能把合成负例加进阳性blacklist。

## 3. 路线一：pep_mismatch容量

对split s与HLA h建立：
P(s,h)={该split该HLA阳性记录中出现的唯一peptide}。

正例r=(t,p,h,s)的合法候选：
Cpep(r)={q∈P(s,h):q≠p 且(q,t)∉B}。

αβ保持一起，h不变；不是从别的HLA随便拿peptide，也不把同split池换成全数据池。对每个候选写原正例id、q、t、h、来源证明和negative_type=pep_mismatch。

不得逐行盲目展开全部笛卡尔积：缓存(s,h,t)的合法peptide集合，再给原record引用。可先仅统计和保存紧凑索引，按需要物化负例。

### 必交容量统计

每条正例输出：
record_id、split、tcr_id、hla_id、peptide_id、
pool_peptides、excluded_original、excluded_known_positive、legal_pep_mismatch_count、
candidate_set_id、target_count（如设置）、meets_target。

每split输出：
- 正例数P_s、唯一paired TCR/peptide/HLA数。
- legal_pep_mismatch_count的min/median/分位数/max，容量0/1/2/<target的正例数。
- 原始候选次数总和（仅作为展开成本）。
- 去重后合法负四链集合U_s及总容量Nmax_s=|U_s|。
- 最大整体唯一负例比例为1:n_s，n_s=Nmax_s/P_s。
- 全部split汇总比例=sum(Nmax_s)/sum(P_s)，同时审计跨split同输入组合；当前peptide-disjoint通常阻止跨split同peptide负组合，但仍实际核验。
- 每条正例候选数不能直接相加当独立负例容量：同一(t,h)的不同阳性原record可能生成同一个q负组合。
- 逐record都分配k条且全局不重复的可行性，需要容量分配/最大流等匹配检查；每行容量>=k也未必保证可行。报告此项或明确尚未计算。
- 最大整体比例与每个正例都能获得同样比例是不同指标，均要标注。

目前没有真实统计数值，文档不填写臆测的1:n。生成report后才能作结论。

### 分支触发

命令配置target_negatives_per_positive/目标总预算；默认可沿用2作容量诊断，但不强制数据集达到1:2。
输出route_decision.json，包含容量0组、目标不足、唯一容量及分配检查。
用户要求若pep_mismatch不足则转路线二：执行时采用split级route，满足目标的split可使用pep_mismatch；不足的split使用路线二候选池。不静默把两类混合补齐；若显式选择混合模式需配置、标记来源并单独汇报。
未指定目标预算时交付容量报告和路线二池，不无条件展开海量负例；随后抽样预算可直接配置，不需重建候选池。

## 4. 路线二：每split的tcr2candidates.npy

### 池子内容与格式

根目录建议 /NAS/luyq/TMP_prediction/hitph_negative_pools/<protocol_version>/。
每split：
- train/tcr2candidates.npy
- validation/tcr2candidates.npy
- test/tcr2candidates.npy

每个npy是无object的结构化数组，一行一个唯一天然paired TCR：
candidate_index:int64、tcr_index:int64、alpha_chain_index:int64、beta_chain_index:int64。
名字沿用用户指定tcr2candidates.npy，其语义在manifest中明确为split内可抽样天然TCR池，不隐式表示已经展开的所有TCR×peptide组合。

配套tcr_table.tsv记录tcr_index/tcr_id、αβ精确序列或稳定chain_id、来源record、原HLA集合、split；candidate_index到tcr_id的映射必须稳定且有hash。原HLA只作审计，不限制路线二抽样。
使用numpy.load(...,allow_pickle=False,mmap_mode='r')；不存需要pickle的ragged列表，不建立可能巨大的全组合矩阵。

全局：
- blacklist.sqlite：精确(peptide_id,tcr_id)索引与来源；可附只读整数索引方便内存筛选。
- negative_registry.sqlite：已物化唯一负例，唯一约束complex_key。
- protocol_manifest.json：输入hash、池hash、blacklist hash、种子、格式及版本。
- capacity_report.json、route_decision.json。

池中只包含本split原始天然αβ对，不能从α池、β池独立随机重配。所有候选来自正例时不产生新单链序列，可复用全量MSA。

### 抽样算法

对目标正例r=(t0,p,h,s)：
1. 从split s的tcr2candidates.npy均匀抽取候选t。
2. 若(p,t)∈B，拒绝；包括t0及其余已知阳性。
3. 合法负例为(t,p,h)，HLA仍是目标p的原HLA；donor的原HLA不作为准入要求。
4. 查询negative_registry/当前生成批次唯一键，重复组合不新增记录；保留天然αβ。
5. 写negative_type=split_tcr_mismatch、source_record_id、donor_tcr_id及original_hla_context。
6. 达到配置预算或合法容量耗尽停止；记录真实可用数量，禁止无限重试。
7. 重复命中率高时直接用pool减去positives_by_peptide得到合法索引列表，并从剩余组合无放回抽样。

伪代码：

```python
legal = split_pool_indices[s] - known_positive_tcr_indices[p]
# 还需移除该目标(p,h)已经物化的负complex。
chosen = sample_without_replacement(legal_remaining, budget, rng)
for t in chosen:
    register_unique_negative(t, p, h, split=s)
```

对固定p,HLA、split的最大合法容量是split内唯一TCR数减去其中对p已知阳性的TCR数，不等于全池大小。
路线二扩大池但仍可能容量为0；明确报告，不能保证任意比例可实现。

## 5. 动态采样、去重及有限embedding缓存

“重复负样本排除”分别实现：
- 新候选生成时，相同四链输入只物化一次；并发使用数据库唯一约束防重。
- 同一epoch/评估清单内不以重复行凑比例。
- 训练不同epoch允许再次选取已有唯一候选，这是缓存池的重复使用，不是产生新的独立负样本。
若要求跨所有epoch也永不重选，会耗尽有限池；不能默认采用该含义。

训练：
- 维护池、blacklist和registry，不必须每epoch产生全新组合。
- 固定基础seed+epoch派生随机状态，支持恢复；保存实际抽样配置和计数。
- 先选一个预算明确的有限合法负组合池并提取联合特征；每epoch从有有效embedding的池内动态采样。
- 若在线抽样遇到新合法组合，先进入pending_features，完成四链MSA准备及Protenix联合提取后才能训练；不能拼接单链embedding替代。
- 所有池对象可复用，但比例变化只改变抽样/物化预算，不重建天然TCR池或阳性blacklist。
- 报告candidate_capacity、materialized_unique、features_ready和epoch_sampled，区分潜在容量与实际使用。

validation/test：
在容量和预算允许时生成一次固定、无重复清单，版本化保存实际比例；容量不足时如实记录，不复制凑比例。评估报告标明同HLA与跨HLA donor类型，保留pMHC分层、最邻近训练peptide距离分层和覆盖数。

## 6. 原协议与风险检查

本规范替换此前“只允许同split同HLA换TCR、严格1:2”的容量要求；原划分、精确序列及全局已知阳性排除继续保留。
pep_mismatch和路线二属于不同分布，negative_type及target/donor原HLA要保存。
路线二可能增加跨HLA容易负例，需要在后续实验加入TCR-only、pHLA-only等对照及同HLA子集指标，检查HLA/来源捷径；本步骤不因这些风险阻止用户指定的候选池建立。
两条路线都不证明真实不结合；不得写进实验negative标签表冒充湿实验观测。

## 7. 实现顺序与并行

1. canonical输入、天然paired TCR身份、全局blacklist和hash。
2. pep_mismatch容量索引/精确去重统计 与 各split tcr2candidates.npy构建可并行。
3. 输出容量报告1:n、零容量/不足分组和route_decision。
4. 路线二容量统计与有限候选物化；固定validation/test清单。
5. 与hitph_msa索引接通：任何负四链记录引用既有天然链，不新增肽-HLA错误组合。
6. embedding预算/空间检查、候选联合特征、真实训练小跑及后续动态采样。

待实现接口（不是当前已运行命令）：
- build_tmp_global_blacklist.py
- audit_tmp_pep_mismatch_capacity.py
- build_tmp_split_tcr_candidates.py
- sample_tmp_negative_candidates.py
- validate_tmp_negative_protocol.py

## 8. 验收

- 所有clean positives跨split、跨HLA进入同一(peptide,tcr)blacklist；任何生成负例与其交集为0。
- 每个npy仅来自本split、αβ配对天然、唯一、序号/hash可重现。
- pep_mismatch的peptide来自本split同HLA已知阳性池；原tcr/HLA精确保持。
- 每条正例有容量计数；每split有唯一最大容量及真实1:n，原始计数不冒充去重数。
- route2不再限制donor HLA；目标peptide/HLA始终来自合法原record。
- 同四链输入不重复物化，无跨split同输入冲突，失败/容量不足明确报告。
- blacklist、负例registry、候选池有独立身份和职责。
- 不无限拒绝采样；随机状态/恢复受控；比例不固定。
- 所有ready训练样本有配置匹配的联合embedding；仅候选存在不能视作训练ready。
