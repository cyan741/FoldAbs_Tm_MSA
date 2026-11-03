> 历史文档（2026-11-03）：MSA 建库与搜索脚本已从仓库移除，以下命令不再作为当前操作入口；保留本文用于追溯实验方法。

# A/B/H 建库入口与实际执行记录

本文件对应 `TMP_ABH_DATABASE_MSA_EXECUTION_PLAN.md` 的 S0–S5。全部建库数据统一保存在 `/NAS/luyq/TMP_prediction/msa_db/v1/`：原始母库、工具安装与编译目录、输入清单、种子、搜索证据、数据库、pilot 缓存和报告。仓库内 `data/tmp_msa` 仅为指向该目录的兼容符号链接，继续受 Git 忽略规则保护。负样本生成、全量单链 MSA、四链 JSON 组装和 Protenix 联合特征提取是后续任务，不能把本次建库视为那些步骤已完成。

## 固定输入与来源

- 输入：`/NAS/luyq/TMP_prediction/data/hitph_l4_v1/splits/positive_splits.csv`。
- HLA 输入映射：同目录下 `raw/Hi-TpH-hla_allele2seq.json`。
- 原 split 保持不变；仅将字段值 `val` 写成 `validation`。
- A/B 种子各 50 条，只来自包含 train 的唯一链；序列同时出现在多个 split 时，train 中已有的序列仍属于合法候选。共享情况单独报告，不宣称 TCR-disjoint。
- 公共 UniRef50 母库为 `2026_03`：38,840,027 条、12,182,780,325 个残基。官方压缩文件 MD5、压缩/解压文件 SHA256、release note 和下载日期进入 sources manifest。
- IPD-IMGT/HLA `3.65.0`，commit `5b915f27f7f620361cf83cb626eeac8a03c0247c`。该公共参考部分只取实际输入范围 HLA-A/B；UniRef50 的补充部分是跨物种/位点的公共 MHC-I 同源候选。
- HMMER 3.4 在运行目录内编译安装；不改 conda 环境。
- 固定公共源参数保存在 `configs/tmp_abh_sources_v1.json`。下载入口即使指向官方当前 release，也必须通过该锁定版本、大小、官方 MD5 和条数检查，不能随 Latest 漂移。

## 运行

首次准备数据源与工具：

```bash
bash scripts/MSA_building/setup_tmp_abh_sources.sh
```

输入导出、种子选择、源校验、建库和 pilot：

```bash
bash scripts/MSA_building/run_tmp_abh_build.sh --workers 8 --cpu 4
```

下载尚在运行时可添加 `--wait-for-source`。数据源准备、建库、pilot 恢复和冻结验收均默认使用 `/NAS/luyq/TMP_prediction/msa_db/v1`。`TMP_MSA_ROOT` 可同时覆盖这些入口的默认目录，Python 入口的显式 `--root` 优先；本版本的产物根目录固定为上述 NAS 路径。首次启动可用 `--input-csv` 和 `--hla-map-json` 指定同格式的冻结数据。Python 默认使用已有 foldtmp 环境，也可设置 `TMP_MSA_PYTHON`。

主流程持有 `build.lock`；每条搜索保留独立证据与完整性记录。失败会写 `reports/execution_status.json`，重跑上述命令可复用版本与文件校验均通过的搜索结果。不要同时对同一目录启动独立的 cache/seed writer。更改输入或母库时使用新的工作目录。

数据库已建立而 pilot 尚需重跑时，使用 `bash scripts/MSA_building/run_tmp_abh_build.sh --resume-from-pilot --workers 8 --cpu 4`。这会校验现有数据库/输入，复用有效缓存，不重复建库。每个查询另外保留 `raw_search.json` 和原始搜索输出 hash；A3M 转换失败不会要求重新进行已验证的大库搜索。

## 建库与过滤

A/B 使用 0.8×3-mer Jaccard 距离 + 0.2×归一化长度差的 farthest-point 近似选择。没有可信 V/J/CDR3 注释，因此不会宣称完成 V/J 分层。训练标签不参与种子选择。

jackhmmer 使用 `-N 1 -E 1e-3 --incE 1e-3 --seed 42`，保留 Stockholm、sequence/domain tables 与日志。筛选单域 query 覆盖 ≥70%、domain inclusion E-value ≤1e-3、标准残基、母库蛋白长度 ≤700，以及无明确错误链注释。按 accession 一次扫描母库回取完整 FASTA 条目，精确去重；母库自身标注 Fragment/partial 的条目单独记录，不能宣称它们都是生物学完整蛋白。类型不确定的命中保留标记、`A/B_unconfirmed_candidates.fasta`、独立 provenance 与抽查表；主 `A/B_compact.fasta` 仅采用至少有一个来源明确注释为对应 TCRα/β 的序列，再由 train 非种子 pilot 检查覆盖。该保守策略避免把 “Ig-like” 注释当作链类型证据；候选库不会被无记录丢弃。A/B 不加入训练查询本身凑库，也不使用旧 ColabFold 纳米抗体代理库。

H 使用官方分块蛋白比对的列坐标；`-` 表示与参考相同，`.` 表示 gap，`*`/`X` 或短/空块表示未测定、终止或异常区域。短块按该块参考坐标补未知位点，不能左移后续块。验证参考重构和裁剪片段与公共 FASTA 连续匹配，并核对所有输入 allele。

常规 H 输入对应参考全蛋白的零基位置 `[24,206)`。`HLA-A*02:01:110` 与 `HLA-B*35:42:02` 的实际输入只有 181 aa：公共记录在目标区域第一位未测定，输入为已知的连续第 2–182 位。两种区域分别建立公共序列裁剪映射，不补未知残基，不将其称为天然缺失。详细列坐标与检查见 `H_input_boundary_validation.json`。

H 参考库初始包含 15,783 条精确唯一序列。若参考 pilot 的有效多样性不足，可以从公共 HLA 参考选择代表序列搜索同一固定母库补充 MHC-I。H 的补充片段从母库 accession 对应序列的 homologous domain 坐标取出，明确记录为 domain crop；它们不会被写成完整蛋白。公共参考与补充来源分别保留。

## Pilot 与冻结

- A/B 各取 20 条未作种子的 train 查询，覆盖长度极端与序列多样性；比较母库和紧凑库。
- H 检查所有 31 条唯一输入序列。H 的公共参考选择及补充不使用 val/test 或外部 test 来调整库内容。
- A3M 查询必须精确相等；去掉小写插入后，每条匹配列数必须等于 query 长度。处理 Stockholm 的 `.` 和 `-`，保留插入残基、清除重复行。含 X 等非标准残基的命中整行进入 `rejected_alignment_rows.tsv`，不删掉其中的字符来伪装修复；原始 Stockholm 与隔离表同时保留。搜索失败写 failure，不能伪装成 no_hit。
- 深度上限 512 条非 query 行；先过滤 query 列覆盖低于 70% 的行，再按匹配列序列距离做 farthest-point 选择，不直接取文件前 512 行。该设置是本次 pilot 的工程配置，不是已证明最优值。
- Neff 定义为 `sum_i 1/count_j(identity(i,j)>=0.8)`，identity 为非 gap 相同列数除以两行非 gap 长度的较大值，包含 query，只对保留的行计算。
- 缓存键包括 role、精确序列、数据库 hash、配置 hash、工具版本和输入边界 hash；所有缓存另存 A3M hash。可用 `prepare_tmp_msa_cache.py validate MANIFEST` 审计。query-only peptide 不调用搜索。
- `database_manifest.json` 标记 `frozen` 才允许正式全量缓存。大量 no-hit、链类型不确定或过低 Neff 会产生 `built_review_required`，禁止把有待核查的结果发布为已冻结库。
- 此处冻结仅指完成数据库工程验收；本次尚未比较相同 Protenix/head 设置下的验证表现，不能据此宣称性能等价或加速倍数。

## 2026-10-07 已实测内容

- 输入 85,200 条；train 79,465、validation 3,951、test 1,784。无隔离序列，peptide 无跨 split 重叠。
- 唯一 A/B/H/peptide：40,343 / 70,687 / 31 / 2,306。A/B train 唯一链：36,220 / 66,195。
- A/B 跨 split 共享链：540 / 249；H 24；以上不修改原划分。
- 48 个 HLA allele 输入边界检查全部通过。初始公共参考库去重后 15,783 条。
- H 参考 pilot：31/31 成功，无 query-only。首轮按文件顺序取深度时平均 Neff 1.07；改用覆盖过滤和多样性选择后平均 Neff 3.78、非 query 平均列覆盖 97.8%。两次结果都保留，不能把深度当作有效多样性。
- UniRef 下载约 18.5 分钟；验证、解压和规模审计约 275 秒。并发搜索采用 8 workers ×4 CPU，实际检索时间以逐条记录为准。
- 本次新增的格式/隔离/多样性/来源恢复/缓存损坏/失败分类测试及原 compact-Nb 测试，共 19 项通过；另用真实 HMMER 验证了无命中时正确生成 query-only。

截至 2026-10-07，本次 104 个母库种子检索（A/B 各 50、公共 H 4）已全部成功，检索约 1,088 秒。统一回取 8,946 个 accession、清洗和去重约 121 秒。最终主库 A/B/H 为 **107 / 63 / 18,679** 条；A/B 未确认候选分别 **5,430 / 5,065** 条，保留在独立 FASTA 和来源表中。H 包含公共参考及 2,896 条新增母库同源片段，不将片段当作完整蛋白。库与配置状态为 **frozen**，仅表示工程验收完成。

| Pilot | 查询数 | query-only | 平均非 query 列覆盖 | 平均保留 Neff | 平均 jackhmmer 时间 |
|---|---:|---:|---:|---:|---:|
| A 母库 | 20 | 0 | 78.9% | 511.97 | 81.03 s |
| A 主库 | 20 | 0 | 78.2% | 79.73 | 0.101 s |
| B 母库 | 20 | 0 | 79.4% | 513.00 | 80.41 s |
| B 主库 | 20 | 0 | 82.7% | 53.07 | 0.065 s |
| H 公共参考加同源补充 | 31 | 0 | 88.8% | 513.00 | 3.867 s |

Neff 只对 query + 最多 512 条保留行计算，513 是该统计的上限。多样性选择、片段和相似性阈值会影响该数值，不把它当作整个数据库的有效多样性，也不据此宣称模型性能改善。检索时间不包含 A3M 转换；转换耗时另列在 `pilot_report.tsv`。A 母库初次转换被非标准命中拦下，修复整行隔离后复用全部 20 个成功搜索输出；该部分时间由原 HMMER 日志恢复，标明不包含进程启动开销。A/B 母库对照分别隔离 583 / 431 行非标准命中。

A/B 主库的保留行均有明确对应链注释；母库对照的保留行多数注释未确认，且含少量明确其他链。因此母库与保守主库的深度和 Neff 不等价，不宣称信息或模型效果等价。H 补充后的保留行约 42.1% 缺少明确 HLA/MHC 注释，已有 homology 搜索证据及标记，仍需后续人工/模型验证；来源详见合并的 `H_provenance.tsv` 和 `H_mother_provenance.tsv`。

全部 111 份蛋白 pilot MSA（71 个角色化唯一查询）通过查询、列数、残基、重复与版本检查；48 个 HLA 输入 allele 边界检查通过。2,306 条 peptide query-only 文件另行生成并验收，搜索次数 0。最终验收见 `reports/final_database_validation.json`，冻结 manifest 同时记录数据库、来源表、输入清单、种子及配置文件 hash。正式全量单链 MSA、四链组装和 Protenix/head 性能对照尚未运行。

## 主要产物

| 目录 | 内容 |
|---|---|
| `inputs/` | unique_A/B/H/peptide.tsv、record_chain_map.tsv、输入 spec/audit |
| `sources/` | 锁定原始文件、sources_manifest.json、tool_versions.json |
| `tools/` | HMMER 3.4 安装、可执行程序 |
| `seeds/` | A/B train 种子、选择报告、可选公共 H 种子 |
| `seed_searches/` | 每种子原始比对、命中表、日志、search.json |
| `databases/abh_v1/` | A_compact.fasta、B_compact.fasta、H_hla_compact.fasta、来源表、过滤表、构建报告 |
| `pilot/` | A/B 母库对照、H pilot、每查询独立版本化缓存 |
| `reports/` | 状态、阶段耗时、验证、pilot_report.tsv / pilot_summary.json |
| `A/`、`B/`、`H/` | 对应主库、来源表、构建报告的相对符号链接，真实文件位于 `databases/abh_v1/` |

上述目录均相对于 `/NAS/luyq/TMP_prediction/msa_db/v1/`。迁移逐文件核对 SHA256 后更新活动清单路径并重新运行冻结验收，查询、FASTA、A3M、工具二进制和配置身份保持不变。完整迁移校验见 `reports/relocation_validation.json`，修改前的元数据保留在 `reports/relocation_original_metadata/`。原始 HMMER 日志、命令记录和编译配置保留执行时的旧路径，旧路径通过兼容链接仍可访问；这些历史记录不会当作新的输出目录。

## 启动正式训练前的实际缺口（2026-10-07）

建库和 MSA 工程配置已冻结，尚未完成真实数据训练链路。现有负采样、JSON 组装、联合特征提取、缓存验收和训练脚本，不代表相应的全量产物已经生成。

1. **先解决负样本容量约束并冻结规范输入。** 原 CSV 的 `record_id/tcra_v/tcrb_v/hla_name_normalized/hla_clip` 需要映射成 canonical 的 `id/tcra/tcrb/hla/hla_sequence`，保留原 HLA 精度及 split，只将 `val` 适配为 `validation`；全部 clean positives 用于已知阳性排除。按保留的 `hla_name_normalized`、同 split/同 HLA 天然 αβ donor、排除所有 clean positives 的只读核查显示：2,453 个 pMHC 分组中，train / validation / test 分别有 **138 / 47 / 48** 组不能提供每阳性 2 条唯一负样本，涉及 **50,243 / 2,104 / 918** 条阳性。例：train `KLGGALQAK|HLA-A*03:01` 有 14,595 条阳性，需 29,190 条负样本，但合法 donor 仅 346 个。该核查没有生成负样本，也不是正式 negative-pool audit；必须明确容量不足的处理规则，不能默认删阳性、跨 split 借 donor、重复负样本或修改原划分。原阳性四链组合无重复、无跨 split 同组合，保留 HLA 名称对应的 clip 一致。
2. **完成全量、冻结配置的单链 MSA。** 当前唯一 A/B/H 为 40,343 / 70,687 / 31；仅有 A/B 各 20 个 pilot 查询与 H 全 31 个 pilot 查询。pilot 的 `scope=pilot` 与正式 `scope=frozen` 属于不同缓存身份，不应直接把 pilot manifest 宣称为正式缓存。peptide 的 2,306 个 query-only 已生成，但正式记录/缓存清单仍需统一接入。负采样只重组现有天然 αβ 和 peptide/H，因此不产生新单链序列；外部评估的新序列另查冻结库。
3. **接通 MSA 清单并组装正负四链 JSON。** `prepare_tmp_msa_cache.py` 输出包装在 `entries` 中的 `a3m_path`；`make_tmp_protenix_input.py` 当前读取字典/列表中条目的 `sequence/unpairedMsaPath`，二者尚需适配。随后对训练候选池及固定 validation/test 记录分片组装，验收链顺序、查询、路径、不含 paired MSA、peptide query-only 和本地模式。
4. **真实 Protenix 小批验证、空间测算及全量联合特征。** 固定 checkpoint、cycles、dtype、模型配置等，验证四链本地 MSA 提取，测单 record 用时和特征字节数后规划磁盘/GPU。每个新四链组合必须独立提取联合 `s/z/p_dist`，不可拼接单链 embedding。现有旧特征不能仅因文件存在就复用，必须匹配链序列及配置身份并通过 finite/shape 审计。负样本池通常多于每个 epoch 使用的负样本，需要给整个有限候选池预计算特征。

   **特征版本身份还有实现缺口。** 当前 `.pt` 的 `extraction_config` 记录模型名称、包版本、cycles、use_msa 和精度，但没有 checkpoint 内容 hash、各链 A3M/数据库/配置 hash、代码版本；`_cached_index_row` 的 resume 校验只有 sample name 和必需 tensor 字段。当前缓存审计检查四链序列、shape/finite 及缓存内部配置一致性，不能证明其符合这次要求的 checkpoint/MSA 配置。正式提取前须补齐这些身份记录及与目标配置的校验，或使用明确的配置独立目录和完整冻结清单约束，不能让同名旧 `.pt` 被自动复用。

5. **通过完整缓存审计，再做训练启动验证。** 固定 train 正样本与负样本候选池、validation/test 1:2 清单，检查无缺失/损坏/身份冲突，完成真实 GPU 小跑及需要时的 DDP、恢复检查，再启动正式动态负采样训练和 validation early stopping。CPU 合成测试已经覆盖部分逻辑，但尚未验证这套真实全量数据。相同 Protenix/head 设置下的母库与紧凑库性能对照、外部 IMMREP25 评估也尚未运行，不能宣称模型性能等价。

只读容量核查详见 NAS 根目录下 `reports/training_readiness_preflight.json`；该报告保留分组假设及全部容量不足分组，作为正式负样本协议决策前的证据。

空间测算也尚未完成：2026-10-07 迁移最终验收时 NAS 共享卷约剩 190 GiB，余量会随其他任务变化。旧批次统计 `/NAS/luyq/TMP_prediction/protenix_embeddings/immrep25/immrep2025_tmp_features/extraction_stats.json` 记录 8,938 份特征，总计 156,169,637,758 bytes、平均 17,472,548 bytes/条（`use_msa=false`、`cycles=1`、平均 212 tokens）。这仅用于展示存储量级，不能用作新四链 MSA 配置的正式耗时或容量结论。即使按这个旧均值，仅 85,200 条阳性也约需 1.49 TB（十进制），还未包含负样本候选池；应先用真实新配置小批实测，再落实全量特征的存储位置和容量。
