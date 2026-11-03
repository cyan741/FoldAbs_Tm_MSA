> 历史文档（2026-11-03）：MSA 建库与搜索脚本已从仓库移除，以下命令不再作为当前操作入口；保留本文用于追溯实验方法。

> 2026-10-08 用户要求的当前重构覆盖本计划旧版的 SHA/哈希缓存键/peptide 文件/冗余标准 JSON/代码快照要求：现使用稳定链 ID、精确序列与显式配置，脚本位于 `scripts/MSA_building/`。当前接口与验收边界见 [存储与数据流说明](TMP_MSA_STORAGE_AND_DATAFLOW.md)。下文相关旧要求仅保留为计划历史。

# HiTPH 全量 A/B/H 搜索库与 unpaired MSA 缓存执行规范

日期：2026-10-08。分支：tmp-specificity。本文为执行规范；真实执行结果以 `hitph_msa/reports/full_msa_validation.json` 及 [交付说明](TMP_HITPH_FULL_MSA_DELIVERY.md) 为准，不能仅凭脚本存在判定全量完成。

## 1. 本次目标与完成定义

最终覆盖当前冻结 HiTPH 全部记录及全部唯一输入链：
- 当前文档基线：85,200 条正例，A/B/H/peptide 唯一序列分别为40,343/70,687/31/2,306。执行时重新计算并与输入 hash 对账，不能硬编码替代核验。
- A=TCRα、B=TCRβ、H=HLA-I；三类分别搜索并生成 unpaired MSA。
- peptide 只生成 query-only A3M，不做搜索。
- 输出根目录固定为 /NAS/luyq/TMP_prediction/hitph_msa。
- 同一精确链序列的 MSA可供所有正负组合复用；四链联合特征针对具体组合独立计算。
- 空间不足时先保留完整序号、查询及恢复计划，不强行保存 A3M；有空间后按原身份恢复生成、验证，再交给 Protenix。

“覆盖全部”有两种状态，必须分别报告：
1. 索引覆盖完成：所有链均有稳定 chain_id、精确序列、角色、配置和状态，无漏项。
2. 可提取完成：所有被请求 record 的四条链均有已验证本地 A3M。
pending_space 只满足第1项，不能宣称全量缓存已完成，也不能把序号当作 MSA路径传入 Protenix。
所有蛋白至少得到有效 query-only 或有命中的 A3M；不保证每条链都能获得高深度同源信息。成功搜索后无命中，与搜索失败分开记录。

## 2. 基线与源数据

读取 docs/TMP_ABH_DATABASE_BUILD_USAGE.md 中实际产物：
- 原建库根：/NAS/luyq/TMP_prediction/msa_db/v1/
- 输入：/NAS/luyq/TMP_prediction/data/hitph_l4_v1/splits/positive_splits.csv
- 全量已知阳性：/NAS/luyq/TMP_prediction/data/hitph_l4_v1/clean/positives_clean.csv
- A/B/H 主库107/63/18,679；A/B 未确认候选5,430/5,065。
- UniRef50 2026_03、IPD-IMGT/HLA 3.65.0；以已有 sources manifest 的文件 hash、release 和 commit 为准。
- 原 pilot 是 scope=pilot；只有身份与内容核验后才能通过显式导入建立正式 scope，不能直接改标记为完成。

保留当前 split、HLA精度、181/182 aa 边界；复用现有原始母库，不复制大文件。新库用独立版本，旧冻结库不覆盖。

## 3. 输出结构与稳定索引

输出根目录下：
- inputs/chains.tsv：全部角色化唯一链。
- inputs/record_chain_map.tsv：record_id、split、四链 chain_id。
- inputs/sequences.fasta：精确输入序列。
- databases/abh_v2/：A/B扩展库、H核验库、来源表及冻结 manifest；允许指向经核验的原数据，不依赖临时目录。
- configs/：搜索、边界、链分类、回退规则、软件版本、数据库 hash。
- index/msa_manifest.sqlite：权威状态、事务更新、失败/等待/恢复记录。
- index/msa_manifest.json：兼容导出，外层 entries，包含 sequence、a3m_path、unpairedMsaPath。
- cache/<config_id>/A/、B/、H/、peptide/：最终 A3M。
- pending/：待恢复序号列表、恢复计划及原因。
- reports/：覆盖、质量、空间、恢复与最终验收报告。

chain_id 是稳定序号，如 A00000001；首次按角色和 sequence_sha256 排序分配，追加新查询不重排旧序号。真实缓存身份还必须包含精确序列、角色、库 hash、参数 hash、工具版本和输入边界 hash。序号不替代身份校验。

每条索引至少含：
chain_id、role、sequence、sequence_sha256、input_spec_hash、database_hash、
search_config_hash、cache_key、status、desired_path、a3m_path、a3m_sha256、
nonquery_depth、neff、cdr3_annotation_status、cdr3_coverage_summary、
failure_reason、attempts、updated_at。

a3m_path/unpairedMsaPath 只有文件通过验证后才填写；缺空间时为空，desired_path仅供恢复使用。

## 4. A/B 扩展库：复核候选，再补覆盖

执行顺序：
1. 核验原107/63条主库与来源表、未确认候选的 hash，统计长度和片段标记。
2. 对主库、候选和训练查询做受体类型/可变区识别，优先复用可信原CDR3字段；必要时使用 ANARCI IMGT编号，并映射回真实序列下标。
3. 未确认候选只有获得对应 TCRα/β 类型及合适可变区覆盖证据后才进入扩展库；明确错误链排除，识别失败继续保存为未确认。
4. 按 accession/protein/domain 明确保存原序列或域片段的来源和边界；禁止把域片段写成完整蛋白。
5. 精确去重并合并来源；不加入 HiTPH 查询本身凑库，不把旧纳米抗体代理库混入。
6. 用 train 中不同序列家族、CDR3长度、低覆盖及随机查询检查扩展库。
7. 如仍存在训练家族覆盖空缺，选该家族的 train 代表新增母库搜索。新增种子及来源记录完整。
8. 确定 A/B v2。val/test只使用冻结库和统一回退规则，不参与扩库。

检查全域覆盖及CDR3逐位覆盖。初始统计报告 CDR3覆盖>=80%的非query行数；此阈值是诊断设置，不是生物学保证。CDR3编号失败不冒充覆盖合格，不因某行缺少CDR3就否定全部框架信息。

## 5. H 库复核

复用现有公共 HLA裁剪序列及同源补充：
- 保留公共参考与补充来源差异，对缺少明确MHC身份的补充候选抽查/域识别。
- 181/182 aa 实际输入边界与官方列坐标对应；不补未知残基，不改变真实输入。
- 对已有31条唯一查询核验 query、覆盖和版本。
- 如果改变库内容，发布新 hash/版本，并重做相关MSA；不得静默复用旧缓存。

## 6. 冻结搜索与统一回退

初始继承已验收设置：jackhmmer -N 1、-E/--incE 1e-3、seed=42、非query行覆盖>=70%、最多512条非query行、多样性选择而非截取文件前行。精确软件版本和参数以冻结配置为准。

统一流程：
查询紧凑库 → 验证/统计 → 命中不足或已规定质量不足时查询同一固定母库 → 过滤/转换 → 最终A3M。

“命中不足/质量不足”规则必须在 train pilot 后写入配置并冻结。例如无合法非query命中必触发回退；CDR3覆盖可作为额外诊断，不能临时为某条test更改规则。母库回退命中仅服务当前查询，不追加冻结库。新蛋白类参考来源必须另发版本。

搜索成功但最终没有合法命中：生成 query-only并记录 no_hit_after_fallback；输入序列不丢弃。搜索失败：记录 failed_search/failed_conversion并补跑，不伪装成 no_hit。

## 7. 磁盘不足：保留标记而不写大缓存

执行前检查目标文件系统可用字节、配额（如可获取）及inode，并用真实小批估算A3M、中间文件和并发峰值。默认空间保留预算10 GiB，允许配置；实际预算必须与共享卷环境核对。

严格顺序：
1. 优先持久保存全量索引、序列、record映射、冻结配置和所有待恢复 chain_id。
2. 有足够空间才启动该批搜索；按并发峰值预留空间，而非仅检查最终A3M大小。
3. 空间不足则将未执行/未保存项写为 pending_space，记录序号、精确查询、配置、desired_path及重建计划；不启动注定无法保存结果的长搜索。
4. 若搜索期间写入失败，删除当前任务临时半成品，事务更新状态；不删除已有有效MSA、原始来源证据或用户文件。
5. 保持索引写入预留；若连最小索引都无法保存，应明确报错并导出可保存的位置，不能声称完成。
6. 不把未写入文件登记为ready。空间不足只保留序号无法保存MSA内容，所以之后可能需要重新搜索；索引必须足以自动重建。

状态：
registered → searching → ready_homologous / ready_query_only
registered/searching → pending_space
searching → failed_search / failed_conversion
ready → corrupt（复核失败后）
所有状态保留原因，避免消耗完共享磁盘才发现无法更新manifest。

## 8. 有空间后的恢复与 Protenix 输入

新增 ensure_msa_ready 接口：
1. 接收 record_id/chain_id或分片列表。
2. 按manifest验证ready项身份和A3M hash，复用有效文件。
3. 对pending_space重新检查空间，按冻结查询/数据库/参数生成、验收，原序号不变。
4. 将内部 manifest.entries 的 a3m_path适配成下游 sequence/unpairedMsaPath；只输出四链全部ready的records，未ready项输出明确阻塞列表。
5. 保持 peptide query-only，不设置pairedMsaPath。
6. 用本地MSA验证器检查精确查询、列数、标准残基、重复、路径及版本；随后调用现有四链提取入口。

“直接调用”应实现为一条编排入口先ensure_msa_ready、再组装JSON、再提取；Protenix自身不能从序号恢复丢失的MSA。
未来工作节点也必须能访问已冻结搜索库/工具，避免pending索引可读但来源丢失。

特征索引记录四链序列、每条A3M hash、数据库/配置hash、checkpoint内容hash、cycles、dtype和代码版本。新组合单独提取联合s/z/p_dist。embedding磁盘预算独立检查，不能由MSA缓存空间推断embedding可保存。

## 9. 全量顺序、并行与最终交付

1. 输入/索引导出和候选分类可并行；A/B复核、H复核可并行。
2. train质量检查 → 冻结A/B/H v2和统一回退。
3. A/B/H全量单链搜索分片并行；peptide不依赖搜索，可先完成。
4. 始终生成全量状态及pending序号；磁盘允许时全部缓存，否则明确报告未缓存数。
5. 接通MSA resolver与四链JSON；做真实小批Protenix提取并测embedding空间，才推进全量embedding。

最终报告：
- 每类expected/registered/ready_homologous/ready_query_only/pending_space/failed/corrupt数量与遗漏列表；
- 各split、家族、CDR3长度的质量统计，数据库来源和参数；
- 全部record是否能解析四链；未ready记录的原因；
- 所有链registered覆盖率必须100%；宣称全量MSA完成时ready覆盖率必须100%，failed/corrupt/pending均为0；
- 缺空间时交付“全量索引完成、缓存部分完成”，附可自动恢复清单，不夸大完成状态。

## 10. 已实现入口

统一入口：`bash scripts/MSA_building/run_hitph_full_msa.sh {index|audit|freeze|cache|ensure|extract|validate|all}`。
以下脚本已实现，运行参数参见各自 `--help`：
- freeze_hitph_msa.py audit：候选分类、CDR3覆盖、扩库报告。
- hitph_msa.py index：全量稳定序号及record映射。
- cache_hitph_msa.py：角色路由、冻结回退、空间检查、分片与恢复。
- cache_hitph_msa.py ensure：请求record的缓存补齐及JSON适配。
- validate_hitph_full_msa.py：索引完整性、内容/版本、pending和最终报告。

补充入口：`freeze_hitph_msa.py` 完成 H 域复核、train 家族质量检查与冻结；`hitph_msa.py append` 按角色和精确序列追加新查询且不重排旧序号；`extract_hitph_features.py` 先确保 MSA 就绪，再校验完整特征身份并执行本地四链提取；`validate_hitph_full_msa.py features` 验收真实张量并单独估算 embedding 空间。

实现时复用当前repo可用脚本，但先检查真实接口；必须解决a3m_path与unpairedMsaPath差异、pilot/frozen身份、并发manifest事务以及resume版本校验。
