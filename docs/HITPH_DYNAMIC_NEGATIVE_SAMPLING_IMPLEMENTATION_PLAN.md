> 历史方案：旧负采样入口、TCR 错配路线和有限池代码已清理。当前仅使用 `scripts/negative_sampling/peptide_mismatch.py` 与 `scripts/train_tmp_specificity.py`。当前数据格式、seed 复现和命令见 [交付与使用说明](hitph_dynamic_negative_sampling_implementation_results.md)。

# Hi-TPH Level IV 动态负采样：代码落地执行策略

日期：2026-10-07  
目标仓库：cyan741/FoldAbs_Tm_MSA  
目标分支：tmp-specificity  
状态：方案已确认，本文规定后续代码实现；本文列出的新增脚本、参数和产物尚未实现。

## 1. 已确认的实验边界

- 保留当前全部正样本及既定 peptide 分组划分，不重新切分，不删除与 train 编辑距离为 1 的内部验证/测试 peptide。
- 内部 test 按目标 peptide 与 train peptide 的最近编辑距离分层汇报；负例继承目标 peptide 的距离分层。
- IMMREP25 与当前 train 最近 peptide 编辑距离为 4（用户已确认），保留其原始标签及比例，作为外部测试集。
- 输入为自然配对的 TCRα/β 可变区、peptide、HLA 连续 clip sequence。不拆散 α/β，不改用 CDR3 或 HLA 伪序列。
- TCRα、TCRβ、HLA 分别复用 A/B/H 库产生的 unpaired MSA；peptide query-only。
- 冻结 Protenix，读取每个完整复合物的 s、z、p_dist，训练现有 specificity head。
- 正负比例改为 1:2。若旧规划中写有静态 1:1，以本文为准。

| 数据部分 | 负例策略 | 正负比例 |
|---|---|---|
| Train | 有限候选池预缓存，每个 epoch 重新抽样 | 1:2 |
| Validation | 提前一次生成并固定 | 1:2 |
| 内部 Test | 提前一次生成并固定 | 1:2 |
| IMMREP25 | 使用原始标签，不生成或重采样负例 | 原始比例 |

当前已确认正例数量：train 79,465，validation 3,951，test 1,784。达到严格 1:2 时，每轮 train 为 238,395 条，固定 validation 为 11,853 条，内部 test 为 5,352 条。必须以实际输入审计结果复核这些数量。

## 2. 实现原则及 FoldAbs 对齐

采用 FoldAbs 的 1:2 比例、训练负例动态变化、验证负例固定的思路。参考本仓库 [论文审计](TMP_SPECIFICITY_PAPER_ALIGNMENT.md) 及原论文 §3.4、Appendix A.1/B.1。

本任务有两项明确调整：

1. 每个 epoch 使用全部训练正例一次，不采用 FoldAbs 的抗体簇每簇抽一个正例。
2. 使用同 HLA context 下自然配对 TCR 与目标 peptide 的错配，不引入 CDR 替换/突变。

这里的动态采样是“从固定、已缓存候选池逐轮选取”，不是每轮在线生成任意新复合物。原文完整负例配方与此不同，报告中应说明该差异。

新错配组合必须重新经过 Protenix 提取联合特征。可以复用单链 MSA，但不能拼接不同正例的 s、z、p_dist 作为负例特征。首版训练期间不启动 Protenix 推理。

## 3. 当前代码检查与改动范围

| 路径 | 当前行为 | 后续改动 |
|---|---|---|
| src/foldabs_repro/tmp_prediction/negative_sampling.py | 静态生成 cross-epitope、TCR swap；已知阳性按 TCRα/β+peptide 跨 HLA 排除 | 增加独立候选池接口，明确同 split、同 HLA、自然 α/β、组合去重和容量审计 |
| scripts/build_tmp_negatives.py | 静态脚本，说明文字面向旧 Stage-1 数据 | 保留旧入口兼容性，更新说明，区分旧带标签数据与 Level IV 正例；新增池构建入口 |
| src/foldabs_repro/tmp_prediction/dataset.py | 根据 manifest 查找 sample_id.pt，单样本加载 | 复用文件检查与加载；支持在每轮通过索引选取训练子集，不加载全池张量到内存 |
| scripts/train_tmp_specificity.py | 固定 dataset；shuffle/set_epoch 只改变顺序 | 接入 epoch 负采样索引；更新长度、日志、配置和断点恢复 |
| scripts/make_protenix_input/make_tmp_manifest.py | canonical CSV 转训练 manifest | 复用现有格式，区分候选池总清单与实际评估清单 |
| scripts/make_protenix_input/make_tmp_protenix_input.py | 四链 CSV 转 JSON | 复用，候选池 JSON 按分片输出，保持链顺序 |
| scripts/extract_protenix_local_msa_only.py | 本地 MSA 提取入口 | 复用；补齐角色化 MSA 及 query-only peptide 支持后再批量提取 |

重要问题：现有 tcr_swap 的 donor 过滤没有限制 HLA，因此不能直接作为本方案的候选池构造器。它还可能产生不同 ID 指向同一复合物，必须增加序列组合级去重。旧接口的行为若调整，需要同步更新现有测试和调用者。

拟新增：

- scripts/build_tmp_negative_pool.py：候选池、固定评估负例和可行性审计。
- src/foldabs_repro/tmp_prediction/epoch_sampling.py：纯索引抽样及可复现 epoch 列表。
- scripts/audit_tmp_feature_cache.py：缓存完整性、身份与张量结构检查。
- tests/test_tmp_epoch_sampling.py：抽样、多卡分片和断点恢复的关键测试。
- bash/run_hitph_dynamic_negatives.sh：完成实现后提供串行执行入口；支持先审计、后构建、后训练。

## 4. 候选池规则

### 4.1 输入、标识和排除表

输入正例统一为 canonical CSV：

~~~text
id,tcra,tcrb,peptide,hla,hla_sequence,label,split
~~~

所有 label 必须为 1，split 必须为 train/validation/test。已有划分中若使用 val，适配时映射到 validation，不能重新分配记录。

- TCR 标识：规范化后的 (tcra, tcrb)，自然配对整体去重。
- pMHC 标识：规范化后的 (peptide, hla)，沿用现有 pmhc_id 格式。
- 复合物身份：四条输入序列及固定链角色顺序。HLA 名称保留用于 context 和评估。
- ID 稳定：由规范化组合生成确定性 ID，或持久化一次生成的映射；不得使用 Python 内置 hash() 生成跨进程 ID。
- HLA 名称规范化及名称到 clip 序列的映射必须沿用清洗结果；同一 HLA 出现不一致序列时立即报错。

使用清洗后全部 Hi-TPH 已知正例构建排除集合，不只使用 train 正例。沿用仓库较保守的跨 HLA 规则：

~~~text
known_positive = {(tcra, tcrb, peptide)}
~~~

即某 TCR–peptide 在任意 HLA 下已知阳性，则排除该候选。排除表只用于避免错误负标签，不用验证/测试记录构建训练 donor 池或选择参数；在实验方法中披露此规则。

### 4.2 按 pMHC 建池

对每个 split、每个 pMHC：

1. 统计去重后正例数 n。
2. donor 来源仅为当前 split、相同 HLA 的正例记录。
3. 将 donor 按自然配对 TCRα/β 去重，避免原始记录频次直接决定抽样概率。
4. 保持目标 peptide、HLA、hla_sequence 不变，只替换整个 TCRα/β。
5. 排除 known_positive 中的 TCR–目标 peptide。
6. 对合法候选去重，均匀无放回抽取候选；保留 donor 来源供审计，但来源不得改变样本权重。
7. Train 候选目标数为 pool_multiplier × n；validation/test 固定负例数为 2n。

小规模试运行建议 pool_multiplier=4：每轮从最多 4n 个候选中选 2n。正式规模的池大小根据提取时间和磁盘预算确定；4 是初始建议，不是必须为全量数据预缓存的固定要求。

设该组合法 donor 数为 m：

- m < 2n：不足以实现严格 1:2，报告该组并停止发布训练就绪产物；不得重复记录凑比例、跨 HLA 补齐或删除正例。
- 2n ≤ m < pool_multiplier × n：保留全部 m 个候选，标记池不足目标，但允许达到 1:2。
- m = 2n：该组没有逐轮负例多样性，审计必须单独统计。
- m > 2n：允许跨 epoch 重新抽取，但不保证每一组每一轮的选取一定不同。

使用有界、可终止的合法 donor 选择流程。避免为每条正例遍历全部原始 donor 或无限 rejection loop；优先构建 HLA 去重 donor 索引、peptide 阳性 TCR 集合，再按组筛选。

### 4.3 输出契约

建议输出目录：

~~~text
negative_sampling_v1/
  positives_split.csv
  train_negative_pool.csv
  validation_fixed.csv
  test_fixed.csv
  cache_records.csv
  cache_manifest.csv
  pool_membership.csv
  negative_pool_audit.json
  negative_pool_audit.md
~~~

- train_negative_pool.csv：负例候选，label=0，split=train。
- validation_fixed.csv/test_fixed.csv：对应全部正例及一次固定抽取的负例，均为 canonical 格式。
- pool_membership.csv：pmhc_id、sample_id，供抽样器索引；组内 ID 唯一。
- cache_records.csv：训练正例、全部训练候选、固定评估样本的联合清单，按复合物去重。
- cache_manifest.csv：复用现有 sample_id/original_id/pmhc_id/label/split 字段。
- metadata 可增加 negative_type=same_hla_tcr_mismatch、label_source=generated_mismatch、donor_original_id。
- 如果不同 HLA 名称映射到完全相同四链输入，需要明确复合物 ID 与 context 记录的映射；训练中不得把同一复合物隐式重复加权。

审计至少包含：各 split/pMHC 正例数、合法候选容量、候选数、每轮所需数、不足组、无动态余量组、donor 来源、跨 HLA 排除数、组合重复数、跨 split 相同复合物检查及种子。生成负例明确标记为 pseudo-negative。

## 5. MSA 与特征缓存

A/B/H 库建库和唯一链 MSA 搜索，可以与负例代码开发并行进行。负例只重组已有自然组件，通常不增加唯一链搜索需求。

执行顺序：

1. 从确定的记录提取唯一 TCRα、TCRβ、HLA 序列；按角色路由 A/B/H。
2. 保存链级 unpaired MSA，校验 query 与实际输入序列完全一致；peptide 保持 query-only。
3. 构造 cache_records 对应四链 Protenix JSON，保持 TCRα/TCRβ/peptide/HLA 顺序。
4. 按稳定 sample_id 分片，一条复合物仅由一个提取任务负责。
5. 每个复合物生成独立联合特征；缓存写入采用临时文件加原子改名，支持失败重试。
6. 训练前检查所有所需缓存，失败样本进入重试清单；不能训练时静默丢弃。

提取配置须一致，记录模型版本、recycle、MSA 设置、推理精度及缓存精度；沿用当前已确认配置，不借此调整 recycle 或特征组合。验证四链身份、sample_name、token_asym_id、s/z/p_dist 维度、有限数值及预期 token 数。

磁盘预算按实测缓存大小计算：

~~~text
联合缓存数 ≈ N_train_positive + N_train_pool
           + 3 × N_validation_positive + 3 × N_test_positive
磁盘需求 ≈ 联合缓存数 × 实测平均 .pt 大小 + 中间文件及余量
~~~

若 train 池为 4 倍正例，train 联合缓存约为 5N，而每轮只读取 3N。池大小、计算时间和磁盘容量必须先用小规模提取测量。不要依据 CDR3 的旧长度统计估算当前完整可变区的缓存，也不要将所有张量加载到 RAM。

## 6. Epoch 抽样与训练器接入

### 6.1 纯索引抽样

~~~python
def build_epoch_ids(positive_ids, positives_by_pmhc, pools, base_seed, epoch):
    rng = make_deterministic_rng(base_seed, epoch)
    ids = list(positive_ids)
    for pmhc in sorted(positives_by_pmhc):
        count = 2 * len(positives_by_pmhc[pmhc])
        ids.extend(rng.sample(sorted(pools[pmhc]), count))
    rng.shuffle(ids)
    return ids
~~~

规定 epoch 从 1 开始；基础种子和 epoch 联合决定抽样。输入排序固定，不能依赖 set 遍历顺序。固定采样模式首次选择负例后各轮复用同一负例集合，只重新打乱训练顺序。

Dataset 保持完整缓存索引不可变。Sampler 产出当轮选中的全局索引，或每轮创建 Subset/DataLoader；不得在 persistent workers 启动后仅修改主进程 dataset 列表。只加载当前 batch 的特征。

### 6.2 多卡、梯度累积与恢复

- 各 rank 使用完全相同的全局 epoch 列表，再分片；不能各自独立选择负例。
- 全局抽样列表严格为 N 正例、2N 负例，无组合重复。
- 当前 DistributedSampler(drop_last=False) 会在样本数不整除 world_size 时填充重复样本。必须披露并记录这些 padding，不能把填充后的实际访问声称为严格无重复。
- 首版可保留 DDP padding 以保证等长同步迭代，但统计应区分逻辑样本数、实际访问数和 padding 数；单卡应严格无重复。若实现零权重 padding，需同时正确处理损失归一化与各 rank 同步。
- 每轮长度按实际选中数据及多卡分片规则计算；不得按整个候选池长度计算 scheduler 步数。
- 保留现有按实际 accumulation group 大小归一化逻辑，并校验多卡尾部同步。
- 默认沿用普通 BCEWithLogitsLoss，不额外加入类别加权，否则会改变本次 1:2 方案。
- checkpoint 保存 epoch、抽样种子、优化器、scheduler、早停状态和必要 RNG 状态。首版支持 epoch 边界恢复；不宣称支持任意 batch 精确恢复。
- 恢复时检查候选池 ID 列表和采样配置一致；无需加入 git_sha 或 manifest_sha256。

### 6.3 拟新增 CLI

| 参数 | 建议默认 | 说明 |
|---|---|---|
| --negative-sampling | static | 旧数据默认行为；新实验显式指定 dynamic_pool |
| --train-negative-pool | 无 | dynamic_pool 必填，训练候选 manifest |
| --negatives-per-positive | 2 | 本次协议固定为 2 |
| --negative-sampling-seed | 42 | 独立于模型初始化种子记录 |
| --pool-multiplier | 4 | 候选池构建参数，不是每轮比例 |

训练主 manifest 仅包含 train 正例及固定 validation/test；候选池另行输入，防止旧训练逻辑错误地将整个池每轮都训练一遍。dynamic_pool 模式启动时检查 train 部分只有正例。

新增配置应记录采样模式、比例、种子、池路径和大小。保持此前确定的训练输出规范：

- 每个 epoch 后 validation 一次，按固定 valid_bce_loss 选择最佳模型，早停 patience 默认 5。
- 保留现有模型/曲线命名、学习率调度逻辑。
- train.log 使用普通文本格式，可增加当轮正负样本数、负例覆盖及 padding 数；不输出 JSON event。
- predictions.csv 仅保存 test 预测，prediction 为 0/1；AUROC/AUPRC 继续使用内部连续分数，不能用二值 prediction 计算。
- metrics.json 不写逐 pMHC 具体数值；run_config 和新增产物不引入 git_sha、manifest_sha256。

## 7. 执行顺序与可以并行的工作

| 阶段 | 执行内容 | 依赖 | 可并行部分 |
|---|---|---|---|
| P0 | canonical 适配、阳性排除表、同 HLA donor 容量审计 | 已确定正例划分 | A/B/H 建库、MSA 搜索 |
| P1 | 池构造器和固定验证/测试负例 | P0 | epoch sampler 开发、缓存审计器开发 |
| P2 | 小规模池、JSON、链级 MSA 复用及联合特征提取 | P1；对应 MSA 就绪 | 不重叠的缓存分片 |
| P3 | 训练器接入、多卡与恢复测试 | sampler；小规模缓存就绪 | 其他分片缓存提取 |
| P4 | 小规模 static vs dynamic_pool 验证 | P3 | 独立 GPU 上两组训练 |
| P5 | 按预算生成正式池、提取并审计全量缓存 | 小规模结果、容量估算 | 缓存分片；固定评估缓存优先 |
| P6 | 正式训练及固定内部/外部测试 | 全量必需缓存通过审计 | 独立实验运行 |

代码可以并行开发，但共享文件改动要协调。不能在某个候选池还在变化时开始正式训练；同一次实验的池、固定评估负例和提取配置必须冻结。

### 实现完成后使用的命令模板

以下是拟实现接口的执行模板，当前不能直接执行。运行前设置 POSITIVES、KNOWN_POSITIVES 和 OUT 为本机真实路径。

~~~bash
python scripts/build_tmp_negative_pool.py \
  --positives-csv "$POSITIVES" \
  --known-positives-csv "$KNOWN_POSITIVES" \
  --output-dir "$OUT" \
  --negatives-per-positive 2 \
  --pool-multiplier 4 \
  --seed 42 \
  --audit-only

python scripts/build_tmp_negative_pool.py \
  --positives-csv "$POSITIVES" \
  --known-positives-csv "$KNOWN_POSITIVES" \
  --output-dir "$OUT" \
  --negatives-per-positive 2 \
  --pool-multiplier 4 \
  --seed 42

python scripts/make_protenix_input/make_tmp_manifest.py \
  --input-csv "$OUT/cache_records.csv" \
  --output-csv "$OUT/cache_manifest.csv"

python scripts/make_protenix_input/make_tmp_protenix_input.py \
  --input-csv "$OUT/cache_records.csv" \
  --output-json "$OUT/cache_records.json"
~~~

后两条使用现有入口，仅适合小规模验证。正式全量应先分片，再给四链 JSON 添加角色正确的 MSA 路径，不能把原始 query-only JSON 当成已完成三链 MSA 的输入直接提取。训练调用在保留既有超参数的基础上增加 dynamic_pool、候选池路径及采样种子参数。

## 8. 小规模验收与实验比较

先选择多个 HLA、每个 split 内有多个 peptide 且合法 donor 足够的 pMHC，覆盖大组、小组以及容量不足场景。此子集用于工程验证，不改变正式正例划分。

必要验收：

1. 所有 donor 与目标同 split、同 HLA，α/β 始终自然配对；跨 HLA 已知阳性也被排除。
2. 组合无重复、标签无冲突，容量不足能明确报错；不出现静默少生成负例。
3. 每轮逻辑列表使用全部正例一次，负例恰为两倍；同 seed/epoch 列表一致。
4. 有充足余量的大池在连续多轮产生变化；统计累计负例覆盖率和相邻轮重合率，不要求所有小组每轮必变。
5. 固定 validation/test 的 ID 和标签始终一致；缓存缺失/身份错误在训练前暴露。
6. 多卡全局计划一致，实际分片和 padding 可核对；epoch 边界恢复抽样、scheduler、早停状态一致。
7. 完成少量 epoch 的训练验证测试闭环，确认日志、保存命名、曲线和二值 test 输出保持规范。

static 与 dynamic_pool 对比须共用候选池、正例、固定验证/测试集、模型初始化种子、1:2 比例、最大训练预算和早停规则。static 固定一次选出的 2N 负例，dynamic_pool 每轮重选。记录实际训练 epoch/优化步数；早停造成的预算差异要报告。

动态组可能看到更多不同组合，这正是方案改变的一部分。其效果需要在本任务中实测，不能预先宣称一定优于固定负例。候选池扩大试验应另行记录缓存数、提取时间和覆盖率。

## 9. 完成标准

- 候选池及固定评估集通过完整审计，未改变既定正例划分。
- 联合特征缓存覆盖全部训练正例、候选及固定评估样本，训练期间无在线推理。
- 动态采样、固定验证、早停、多卡和 epoch 边界恢复可以协同工作。
- 小规模对照闭环完成后，再启动正式规模缓存与训练。
- 本文只交付执行策略；本次不修改训练代码、不生成全量负例、不执行 Protenix 或训练实验。
