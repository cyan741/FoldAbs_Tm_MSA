> 执行更新（2026-11-03）：用户已批准清理 MSA_building 目录。21 个专用脚本与目录 README 已删除；通用提取器已合并至唯一入口 scripts/extract_protenix_embeddings.py（支持旧 base 与 Mini-ESM 缓存），审计入口迁至 scripts/audit_tmp_feature_cache.py，历史方案迁至 docs/REMOTE_EMBEDDINGS_PLAN.md。调用方与无 MSA 输入生成已解耦。以下内容及 TSV 为清理前审计快照，其余配置、外部数据及文档删除尚未执行。

# MSA 清理审计清单（待用户审计，尚未执行删除）

审计日期：2026-11-03。目标：后续 Protenix encoder 实验不再依赖 MSA 建库、搜索、缓存及输入注入流程。

本轮仅增加审计文档，未修改现有实现、测试、配置、数据或外部资源。以下“删除候选”均为建议，不代表已获删除批准。清单以当前工作区实际文件为准，包含未提交文件；Git 已记录删除的旧路径不重复计入。

## 1. 关键结论与依赖

- `scripts/MSA_building/` 实际有 **23 个脚本（20 Python + 3 Shell）和 2 个 Markdown**。其中 21 个脚本是 MSA 专用删除候选，另 2 个是通用提取/审计能力，需迁出或合并。目录内 README 的历史数量不能作为当前清单。
- `make_tmp_protenix_input.py:89` 在进入 `no_msa` 分支前就导入 `hitph_msa`、`tmp_msa_common`。因此直接删目录会破坏 **HiTPH 无 MSA 输入生成**，而不仅是旧 MSA 分支。
- `benchmark_hitph_online.py` 把 MSA 目录放到 `sys.path` 前面，并从其中的 `extract_protenix_embeddings.py` 导入 `_load_checkpointed_runner`、`_install_empty_template_fast_path`。旧无 MSA 基准依赖此实现。
- `bash/run_tmp_stage1.sh`、`scripts/resume_immrep25_embeddings.sh`、`scripts/run_hitph_cycle4_epochs.py` 直接执行 MSA 目录中的提取脚本。
- `cache_audit.py` 的通用审计函数仍被 `train_tmp_specificity.py` 使用；后者又是在线 BCE/Group 训练的共享依赖，不能整个删除。
- 当前 Mini-ESM 在线 encoder 显式关闭 MSA、RNA MSA、template。这些关闭设置及拒绝 MSA 输入的测试应保留。
- 外部库中的 MSA 网络模块、checkpoint 参数和 encoder 内部张量不是本次仓库建库清理对象；不修改已安装 Protenix。

依赖顺序：先拆除输入转换对 MSA 工具模块的导入，再迁移/合并通用提取与缓存审计入口，更新调用方，最后删除专用脚本和测试。

## 2. MSA 专用脚本：21 个删除候选

本表路径前缀均为 `scripts/MSA_building/`。

| 文件 | 功能 / 对外入口 |
|---|---|
| `setup_tmp_abh_sources.sh` | 下载固定 UniRef50/IPD-IMGT-HLA 来源，准备和编译 HMMER |
| `prepare_tmp_msa_sources.py` | 验证来源版本、展开母库、生成来源与工具 manifest |
| `export_tmp_unique_chains.py` | 按 split 导出 A/B/H/peptide 唯一链与 MSA 输入规范；无 MSA 输入生成不依赖它 |
| `select_tmp_msa_seeds.py` | `select/search/hla`：训练链多样性种子、母库搜索、公共 HLA 种子 |
| `build_tmp_compact_databases.py` | `ab/hla`：从命中恢复母库蛋白、构建 A/B 紧凑库和裁剪 HLA 参考库 |
| `run_tmp_abh_build.py` | `build/pilot/select/report/finalize`：建库编排、pilot、质量报告与冻结 |
| `run_tmp_abh_build.sh` | A/B/H 建库 Shell 入口 |
| `prepare_tmp_msa_cache.py` | `cache/validate`：按角色 jackhmmer 搜索、A3M 缓存和 manifest 验证 |
| `hitph_msa.py` | `index/append`；SQLite 链注册、记录索引、状态、配置绑定和指标 |
| `freeze_hitph_msa.py` | `audit/freeze`；链域注释、参考库准备、pilot 与冻结搜索参数 |
| `cache_hitph_msa.py` | `cache/ensure`；compact 搜索及母库回退、原子发布、恢复、按请求补齐 |
| `validate_hitph_full_msa.py` | `msa/features`；MSA 完整性验收，以及依赖 MSA 注册表的联合特征验收 |
| `extract_hitph_features.py` | 紧凑链 ID → MSA 四链输入，验证特征身份并调度提取 |
| `extract_protenix_local_msa_only.py` | 验证本地 MSA、安装 socket guard，再调用提取器 |
| `run_hitph_full_msa.sh` | `index/audit/freeze/cache/ensure/extract/validate/all` 总调度 |
| `local_msa.py` | `validate_records()`：校验 A3M、四链顺序、肽 query-only，禁止 paired MSA |
| `tmp_msa_common.py` | 搜索、Stockholm→A3M、FASTA/TSV/JSON、域注释、配置与序列键工具；先替换无 MSA 输入生成使用的 `write_json()` |
| `build_compact_nb_msa_database.py` | 纳米抗体训练种子选择，聚合已有全库 A3M 命中构建 compact 库 |
| `prepare_compact_db_msa_cache.py` | 通用 Protenix JSON → 本地 jackhmmer → A3M 与带 MSA 的 JSON |
| `audit_train_seeded_antibody_compact_capacity.py` | 抗体分层种子与紧凑库容量审计 |
| `run_tm_local_uniref50_comparison.py` | Tm 的 UniRef50/compact MSA 对照，连同特征提取与训练编排 |

这里的“删除”同时放弃仓库内旧抗体 Tm 的 MSA 对照流程；不意味着删除 Tm 回归训练本身。

## 3. MSA 目录内应先迁移的通用能力

| 文件 | 实际作用 | 建议 |
|---|---|---|
| `scripts/MSA_building/extract_protenix_embeddings.py` | 冻结 Protenix-v1 trunk 提取、多 GPU 分片、续跑；支持无 MSA；提供旧基准使用的加载与空模板优化函数 | 对比根级同名实现，迁移必要能力并更新调用方，之后删除旧路径；不能直接覆盖同名文件 |
| `scripts/MSA_building/audit_tmp_feature_cache.py` | 根据 manifest、四链序列和张量审计缓存，生成重试清单；本身不搜索或构建 MSA | 迁出 MSA 目录并保留 |
| `scripts/MSA_building/REMOTE_EMBEDDINGS_PLAN.md` | 标题与内容实际是 cycle-4 **无 MSA** 离线缓存方案 | 移至历史实验文档；不能因所在目录认定为建库文档 |

## 4. 目录外需要局部修改的接口

| 文件 / 位置 | MSA 接口或耦合 | 审计后的处理建议 |
|---|---|---|
| `scripts/make_protenix_input/make_tmp_protenix_input.py:11` | 注入 `scripts/MSA_building` 导入路径 | 删除导入路径依赖 |
| 同文件 `:73`–`:81` | `--msa-root`、`--tcra-msa-manifest`、`--tcrb-msa-manifest`、`--hla-msa-manifest`、`--query-msa-dir`、`--no-msa` | 删除前五类 MSA 参数；`--query-msa-dir` 当前仅声明未使用；无 MSA 改为唯一默认，`--no-msa` 可暂留兼容或同步更新调用方 |
| 同文件 `prepare_hitph_inputs():85` | `msa_root`、`no_msa=False`、注册表查询、`READY/read_snapshot`、MSA 链 ID 输出 | 保留采样、排序、分片、manifest 和标准四链 JSON；移除注册表与 MSA 分支，替换通用 JSON 写出依赖 |
| 同文件 `main():156` 起 | 读取三角色 manifest，注入 `unpairedMsaPath`，肽注入 `unpairedMsa`，调用 `validate_records()` | 删除此注入分支，保留 CSV→JSON 转换 |
| `scripts/extract_protenix_embeddings.py:398` | `--use-msa/--no-msa`，`:437` 的 `--msa-server-mode`，后续预处理与 runner 配置 | 移除启用 MSA 和选择 MSA 服务的用户接口；固定关闭 MSA；保留缓存身份中的 `use_msa=False` |
| 同文件 `_prepare_worker_input()` 等 | `no_msa` 控制输入字段检查 | 无 MSA 应始终检查，不因移除开关而失去校验 |
| `src/foldabs_repro/tmp_prediction/cache_audit.py:33` | `audit_cache(..., require_extraction_config=False)`；启用严格检查时硬编码 `model=protenix-v1` 且必须 `use_msa=True` | 保留通用审计；移除旧 MSA 强制协议或改为按预期 encoder 配置校验，不能删掉张量/身份检查 |
| `scripts/train_foldabs_developability.py:818` | `--protenix-msa-source`，论文对齐报告中的 UniRef50 标识与判断 | 非主线 Tm 元数据接口；若本轮全面退出 MSA，改为明确 no-MSA 来源并保留方法差异说明，不能把 no-MSA 写成论文同配置 |
| `scripts/benchmark_hitph_online.py:24`、`:31` | 导入 MSA 目录提取器的两个通用函数 | 改到迁移后的通用实现；它本身是无 MSA 基准 |
| `bash/run_tmp_stage1.sh:107` | 硬编码旧提取入口 | 改引用；若之后清理 Stage1，另行审计整脚本 |
| `scripts/resume_immrep25_embeddings.sh:14` | 同上 | 改引用或另行归档，不能按建库脚本直接删除 |
| `scripts/run_hitph_cycle4_epochs.py:21`、`:161`、`:200` | 旧提取路径、MSA 目录 PYTHONPATH | 改引用和 PYTHONPATH；离线缓存轮换是否保留属于后续清理 |

### 数据契约与环境接口

- **删除候选的 MSA 输入契约**：`unpairedMsaPath`、`unpairedMsa` 的生成与读取，paired MSA 专用校验，角色 MSA manifest（`sequence/unpairedMsaPath`），紧凑输入的 `schema_version/msa_root/chain_order/records[].chains`，以及 SQLite 链 ID/状态/冻结搜索配置绑定。
- **保留的无 MSA 标准输入**：`name`、四条 `proteinChain.sequence/count/modifications`、`covalent_bonds`；TCRα/TCRβ/peptide/HLA 顺序及身份校验。
- **保留的 encoder 输出契约**：`s/z/p_dist/token_asym_id/token_residue_index`、序列身份、模型和精度元数据、`use_msa=False`。它们不是建库接口。
- MSA 工具链包括 jackhmmer/HMMER、UniRef50、IPD-IMGT/HLA 建库来源及域注释流程。删除相应安装/调用脚本不等于卸载机器环境，环境与外部数据库本轮不操作。
- 删除 MSA `PYTHONPATH/sys.path` 后需检查同名模块解析，尤其两个 `extract_protenix_embeddings.py`。

## 5. 应保留的无 MSA 代码和测试

| 文件 | 保留原因 |
|---|---|
| `scripts/train_mini_esm_online_ddp.py:126`、`:161`、`:203` | 强制 mini-ESM 且关闭 MSA/template/RNA MSA；预处理的 `msa_server_mode` 是外部 API 参数，在 `use_msa=False` 下不能据其名字认定启用搜索 |
| `scripts/train_mini_esm_group_rank.py` | 复用上述在线 encoder；无建库流程 |
| `scripts/protenix_embedding_profile.py:58` | `reject_msa_templates()` 拒绝隐藏 MSA/template 字段，防止无 MSA 实验混入旧输入 |
| `scripts/extract_protenix_esm_chains.py` | ESM 单链特征缓存，不是 MSA 缓存 |
| `scripts/run_hitph_mini_esm.py` | 无 MSA 输入准备与 Mini-ESM 提取；要随输入 CLI 的变化更新调用 |
| `scripts/benchmark_mini_esm_routes.py`、`scripts/resume_mini_esm_accelerated.py` | 无 MSA 推理/缓存路线；名称或配置中的 no_msa 应保留 |
| `tests/test_mini_esm_cache.py` | 拒绝 MSA 输入、特征身份和缓存校验测试 |
| `tests/test_mini_esm_online_ddp.py` | 无 MSA encoder 配置和训练相关检查 |
| `scripts/protenix_migration/`、`configs/hitph_protenix_cycle4_nomsa.json` | 旧无 MSA 迁移与提取配置，非建库代码；其生命周期放到后续离线流程清理审计 |

## 6. 测试清单

| 文件 | 建议 |
|---|---|
| `tests/test_tmp_msa.py` | 随 TMP 建库/搜索实现删除 |
| `tests/test_hitph_msa.py` | 随 HiTPH 注册表/缓存实现删除 |
| `tests/test_compact_nb_msa.py` | 随纳米抗体 compact MSA 实现删除 |
| `tests/test_tm_local_uniref50_comparison.py` | 随 Tm MSA 对照入口删除 |
| `tests/test_tmp_prediction.py:108` 起 | 同一个测试混合 MSA 注册表和 no-MSA 输入生成；只移除 MSA 构造与断言，保留无 MSA 固定样本、动态负例 ID 和四链输入测试 |
| `tests/test_tm_sequence_radius_seeds.py:29` | 有读取 MSA 建库源码的检查；解除此耦合。纯序列种子算法与测试可留待 Tm 路线审计 |

`scripts/select_tm_sequence_radius_seeds.py` 本身只做训练序列多样性选择，不读取 MSA、不建库、不搜索。建议本轮不删；如希望连同 MSA 的种子研究工具一起移除，可将它、对应测试及 `docs/TM_SEQUENCE_RADIUS_SEEDS.md` 作为单独批准项。

## 7. 文档逐项分类

### 7.1 MSA 专用文档：7 个删除/归档候选

| 文件 | 内容 |
|---|---|
| `scripts/MSA_building/README.md` | MSA 脚本清单、合并历史、命令与流程 |
| `docs/TMP_ABH_DATABASE_BUILD_USAGE.md` | A/B/H 来源、建库与恢复操作 |
| `docs/TMP_ABH_DATABASE_MSA_EXECUTION_PLAN.md` | A/B/H 建库、pilot、冻结和搜索方案 |
| `docs/TMP_HITPH_FULL_MSA_CACHE_PLAN.md` | HiTPH 全量 MSA 注册、缓存与验收计划 |
| `docs/TMP_HITPH_FULL_MSA_DELIVERY.md` | HiTPH 全量 MSA 交付记录 |
| `docs/TMP_MSA_STORAGE_AND_DATAFLOW.md` | MSA 数据库、缓存与存储布局 |
| `docs/MSA_ACCELERATION_CORRECTION_PLAN.md` | 旧抗体 MSA 加速方案更正 |

如果保留科研历史，以上可归档；若用户批准彻底移除，则删除并修正文档链接。

### 7.2 混合说明：保留主体，修改当前操作指引或标注历史

| 文件 | 需要处理的内容 |
|---|---|
| `README.md` | 项目定位、MSA 依赖与小库路线、目录介绍改为当前无 MSA 主线 |
| `SHARE_SCOPE.md` | 下载 MSA 数据库、检验 MSA 质量的复用要求 |
| `VERIFICATION.md` | 旧验证范围、文件数量、MSA 数据来源要求 |
| `docs/TMP_PREDICTION_PIPELINE.md` | 旧提取路径与“按需要启用 --use-msa”指引 |
| `docs/TMP_STAGE1_SANITY_PLAN.md` | 本来就是无 MSA sanity；更新提取路径或标注历史 |
| `docs/TMP_STAGE2_PLAN.md` | 旧提取路径和未来 MSA 扩展建议 |
| `docs/TMP_NEGATIVE_MISMATCH_POOL_PLAN.md` | 负采样主协议保留；移除训练前必须准备四链 MSA 的旧流程描述 |
| `docs/HITPH_DYNAMIC_NEGATIVE_SAMPLING_IMPLEMENTATION_PLAN.md` | 动态采样与输入准备的旧 MSA 对接描述；保留采样原则 |
| `docs/TMP_SPECIFICITY_PAPER_ALIGNMENT.md` | 历史源码路径和 MSA 可选状态；保留论文实际使用 MSA 的差异分析 |
| `docs/reproduction_spec.md` | 旧复现协议中的 MSA 方法；作为历史规范标注，不篡改论文方法 |
| `docs/PAPER_ALIGNMENT_AUDIT_2026-08-01.md` | 旧抗体复现审计中的 MSA 条件；保留历史证据或独立归档 |
| `docs/TM_SEQUENCE_RADIUS_SEEDS.md` | 纯序列种子研究，按上一节的独立批准项处理 |

### 7.3 含 MSA/no-MSA 字样但属于无 MSA 或当前训练的文档

以下文件不应仅因关键词命中而删除。涉及旧脚本路径的地方随迁移更新，历史实验数值和设置保持原意。

- `scripts/MSA_building/REMOTE_EMBEDDINGS_PLAN.md`（移出目录）
- `scripts/protenix_migration/README.md`
- `docs/HITPH_ONLINE_NEGATIVE_TRAINING_PLAN.md`
- `docs/HITPH_ONLINE_BENCHMARK_20261015.md`
- `docs/HITPH_ONLINE_BENCHMARK_TANG2_20261015.md`
- `docs/HITPH_ONLINE_CUEQ_TANG2_20261015.md`
- `docs/HITPH_PROTENIX_ACCELERATION_CAPACITY_20261016.md`
- `docs/HITPH_PROTENIX_EXPERIMENT_STATUS_20261016.md`
- `docs/TMP_HITPH_MINI_ESM_NO_MSA_EXPERIMENT_PLAN.md`
- `docs/TMP_HITPH_MINI_ESM_NO_MSA_DELIVERY.md`
- `docs/HITPH_MINI_ESM_ACCELERATED_BENCHMARK_20261017.md`
- `docs/HITPH_MINI_ESM_TRAINING_ACCELERATION_20261018.md`
- `docs/HITPH_MINI_ESM_ONLINE_DDP_20261026.md`
- `docs/HITPH_IMMREP25_CAP50_A100_20261029.md`
- `docs/TMP_GROUP_LOSS_IMPLEMENTATION_PLAN.md`
- `docs/TMP_GROUP_RANK_A100_20261029.md`

`docs/HITPH_CYCLE4_NOMSA_CACHE_BUDGET.json` 和 `docs/benchmarks/` 内报告/日志中的 `use_msa=false`、no_msa 路径也属于实验证据。仓库名 `FoldAbS_Tm_MSA` 只是路径名，不应全局替换或作为删除依据。本轮不展开压缩报告包，不扫描外部挂载的数据内容。

## 8. 配置、资源和生成文件

| 路径 | 建议 |
|---|---|
| `configs/tmp_abh_sources_v1.json` | MSA 来源版本锁，随建库脚本删除 |
| `data/tmp_msa` | 指向 `/NAS/luyq/TMP_prediction/msa_db/v1` 的软链接；清理批准后可只删除链接，不递归操作目标 |
| `.gitignore` 中 `/data/tmp_msa` | 链接退役后同步处理；其他无 MSA work 忽略规则保留 |
| `scripts/MSA_building/output/`、该目录 `__pycache__/` | 前者目前为空；后者是生成缓存，随目录退役清理 |
| `src/foldabs_repro.egg-info/SOURCES.txt` | 安装生成的源文件清单包含旧 MSA 路径；后续重新生成或按生成产物策略清理，不手工当作核心代码维护 |
| `/NAS/luyq/TMP_prediction/msa_db/v1`、`/NAS/luyq/TMP_prediction/hitph_msa` | 外部数据库/注册表/MSA 资源，不纳入本次仓库删除范围；后者历史上也存放无 MSA 输入，不能按目录名整删 |
| `data/Hi-TpH-hla_allele2seq.json`、`Hi-TpH-hla_aliases.json`、`Hi-TpH-tcr_gene2seq.json` | 仍用于序列映射与输入准备，保留；不是 MSA 搜索数据库 |
| `.foldtmp-run/protenix_assets/`、ESM 单链缓存 | encoder 模型与输入特征资源，保留 |

## 9. 用户审计后的执行顺序（尚未执行）

1. 确定 21 个专用脚本、7 份专用文档、4 份专用测试、来源配置和本地软链接的删除/归档范围。
2. 将无 MSA 输入转换改为不依赖 MSA 目录，保留固定样本、动态负例、分片及稳定 ID 行为。
3. 整理两个提取器与通用审计入口，更新旧无 MSA 调用方及文档。
4. 移除 MSA 用户接口及旧强制 MSA 审计条件，保留禁用 MSA 的配置和检查。
5. 删除已批准文件并更新混合测试、文档和生成文件引用。
6. 检查残留导入/路径；运行受影响的输入转换、缓存、在线 DDP、训练和 Group loss 测试。必要时在依赖环境做小型 encoder smoke，不启动正式训练或 MSA 搜索。

本轮验证方式为静态搜索、文件阅读和调用关系核对，未运行 GPU 推理或测试。附录 `MSA_REFERENCE_INVENTORY_20261103.tsv` 提供当前源码、测试、配置和 Markdown 中关键词命中的文件、行号及原文；其中包含 no-MSA 与仓库路径误命中，不能直接当删除清单。
