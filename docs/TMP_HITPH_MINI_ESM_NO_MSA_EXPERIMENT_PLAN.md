# Hi-TPH：Protenix mini-ESM 无 MSA 实验准备与代码执行计划

日期：2026-10-12  
目标分支：tmp-specificity  
状态：已确认实验方案；本文是后续代码落地任务清单，不表示新增接口已经实现。

## 1. 本次实验目标与范围

验证公开 protenix_mini_esm_v0.5.0（ESM2-3B）产生的联合 s/z/p_dist 表示，能否在当前 Hi-TPH unseen-peptide 设置下改善结合预测。

已确认：

- 暂不使用 MSA，所有四个组件均关闭 MSA；不生成 query-only A3M，也不启动远程搜索。
- 非 ESM Protenix 及其下游 4-block Pairformer + attention pooling 的训练方案已经确定，作为当前对照；本计划不重新选择该模型的 head。
- 新实验仅更换冻结 encoder 为官方 mini-ESM；下游继续使用同一个 4-block Pairformer + attention pooling 和相同训练协议。
- 不使用作者内部结构微调权重，不对官方 backbone 进行额外微调。
- 不重新清洗、切分或重建固定 val/test 负例。
- 保留自然配对 TCRα/β 可变区、peptide、HLA 连续 clip，四链顺序 tcra/tcrb/peptide/mhc。
- 本轮只提取 trunk 的 s/z/p_dist，不新增 diffusion、geometry、PAE、ipTM。
- 本轮不展开 F0、pooling ABC、cycles 扫描或额外序列 baseline；这些可后续独立安排。

研究结论的边界：ESM 与非 ESM encoder 的比较同时可能涉及模型大小、版本、内部结构等变化；不能将性能差异全部归因为 ESM。超过当前非 ESM 对照不等于已证明超过纯序列模型，相关论文结论需要后续直接序列对照支持。

## 2. 固定路径与隔离规则

数据目录：

~~~text
/NAS/luyq/TMP_prediction/data/hitph_l4_v1/splits
~~~

embedding 根目录（用户指定）：

~~~text
/NAS/luyq/TMP_prediction/protenix_embeddings/hitph_v1
~~~

在指定根目录下新增独立 profile，避免覆盖现有非 ESM 缓存：

~~~text
hitph_v1/
  mini_esm_v0.5.0_c4_no_msa/
    esm_chain_cache/
    prepared/
      run_spec.json
      chain_catalog.csv
      fixed_eval_manifest.csv
      epoch_manifests/
      extraction_shards/
      planned_union_manifest.csv
    features/
      <sample_id>.pt
      index.json
      extraction_stats.json
    audit/
      input_audit.json
      schedule_audit.json
      cache_audit.json
      pilot_budget.json
~~~

--features-dir 指向上述 features 子目录，而不是 hitph_v1 根目录。训练 checkpoint 和日志放入现有训练输出目录体系，单独使用 mini_esm_no_msa 实验名；不与 encoder 特征文件混放。

上述子目录布局是本计划拟新增产物，不表示 NAS 中已经存在。本会话无法访问 NAS，实际文件存在性和资源准备情况由执行端检查。

## 3. 使用最新数据协议，不能回退到旧方案

最新实现说明：[HiTPH peptide mismatch 交付](hitph_dynamic_negative_sampling_implementation_results.md)。

正式 splits 目录包含：

~~~text
positive_splits.csv
train.csv
val.csv
test.csv
train_pep2candidates.npy
val_pep2candidates.npy
test_pep2candidates.npy
blacklilst.npy
~~~

blacklilst.npy 沿用当前实际拼写。可信本地 NPY 以 allow_pickle=True 加载，不接受未知来源文件。

| 文件/逻辑集 | 正例 | 负例 |
|---|---:|---:|
| train.csv | 79,465 | 0 |
| 每个 train epoch | 79,465 | 158,930 |
| val.csv | 3,951 | 7,902 |
| test.csv | 1,784 | 3,568 |

数量来自分支交付说明，执行前必须复核实际输入。源字段为：

~~~text
tcra_v,tcrb_v,peptide,hla_alleles_raw,hla_alleles,hla_clip,label,neg_count
~~~

positive_splits.csv 还含原始 record_id、split 等身份字段，必须保留它们来恢复正例 feature ID。

当前协议：

- 保留 TCRα/β 和 HLA clip，只替换同 split、同 HLA context 下的 peptide。
- 全局 blacklist 按 (tcra_v,tcrb_v) 排除所有 split、所有 HLA 下已知阳性 peptide。
- context 使用当前交付的 hla_alleles/clip 分组；不能重新规范化产生另一套候选索引。
- 按 (TCR,HLA clip) 共享容量，组内无放回采样。
- 正负比例为整体 1:2；部分正例没有合法候选，配额会转移，不能要求每条正例严格对应两个负例。
- 每个 epoch 重新抽样，跨 epoch 可重复组合；一轮内四链负例组合不重复。
- val/test 已固定，训练种子变化不重建它们。
- train epoch 1 中的 neg_count 是规划信息，不能拿它当后续 epoch 固定负例。
- 不使用旧 train-negative-pool/dynamic_pool 接口，不恢复历史 TCR donor 错配路线。

生产入口仅使用：

1. scripts/negative_sampling/peptide_mismatch.py 的现有抽样逻辑；
2. scripts/train_tmp_specificity.py 的 PeptideMismatchSampling 身份映射、动态训练接入。

本次不运行 peptide_mismatch.py 的固定数据生成入口。

## 4. Encoder 配置与资源准备

| 设置 | 本次固定值 |
|---|---|
| model_name | protenix_mini_esm_v0.5.0 |
| ESM 模型 | esm2-3b / esm2_t36_3B_UR50D |
| Backbone | 冻结、eval、no_grad |
| MSA | false，四链均关闭 |
| Template / RNA MSA | false |
| Cycles | 4 |
| Diffusion / confidence | 不执行 |
| 提取计算精度 | BF16，保留敏感操作 FP32 的现有处理 |
| 联合浮点缓存 | FP16，token 索引整数 |
| 提取 seed | 101，显式设置实际 RNG，而不只传 runner 参数 |
| 输出 | s、z、p_dist、token_asym_id、token_residue_index |

官方 mini-ESM 默认 cycles=4、ESM2-3B、use_msa=false。其 trunk 为 mini 配置；不能仅将 base 模型关闭 MSA 就标成 mini-ESM。官方 mini 的 diffusion 默认值与本轮无关，不能把 n_step/n_sample 的占位参数说成已运行结构采样。

必须提前准备：

- 可使用 CUDA/BF16 的 PyTorch、与当前项目兼容的 Protenix 环境；
- mini-ESM Protenix checkpoint；
- ESM2-3B checkpoint 及所安装 fair-esm 加载路径要求的附属文件；
- Protenix common 数据资源、正确的 PROTENIX_ROOT_DIR；
- 检查 get_default_runner 支持 model_name，ESM dataloader/featurizer 能正确准备 token embedding；
- 检查实际安装源码的 ESM 缓存路径、层号、BOS/EOS 去除规则和维度；
- 在正式运行前完成权重下载，本地缓存就绪后不触发 MSA 或临时权重下载。

用官方模型配置构造结构并严格检查权重 missing/unexpected keys。ESM 权重启用必须从 runner.configs 实际读取；模型名写对而 esm.enable=false 的运行必须拒绝。

运行的 Protenix/PyTorch/fair-esm 包版本写入配置，不引入 git_sha、manifest_sha256。

## 5. 当前仓库需修改的位置

### 5.1 scripts/extract_protenix_embeddings.py

当前：checkpoint 检查固定为 protenix_base_default_v1.0.0.pt；get_default_runner 未传 model_name；计算 BF16、保存 FP16；默认 cycles=1、limit=10；resume 主要检查字段。

拟修改：

1. 新增 --model-name，保留 base 旧默认兼容；新实验显式 mini-ESM。
2. 将权重路径检查与实际 model_name 绑定，向 get_default_runner 显式传入 model_name。
3. 新增 --no-msa，与 --use-msa 互斥；本实验强制 false，并清除/拒绝输入中的 MSA/template 路径，防止隐式搜索。
4. 显式关闭 template、RNA MSA；ESM 模型下确认 ESM feature 已生成。
5. 新增 --seed、--dtype、--cache-dtype；示例使用 101/bf16/fp16。
6. 新增 --esm-cache-dir，接入预计算单链表示；路径与 Protenix featurizer 的实际加载规则一致。
7. 保持直接 get_pairformer_output(N_cycle=4) 和 distogram_head 路径；不切换到完整结构 inference。
8. payload 保存 input_sequences、extraction_config、channel_dims，修复现有动态训练身份校验所需信息缺失的问题。
9. resume 校验 model、ESM 模型及配置、cycles、精度、MSA/template 状态、四链序列、token 数和通道；不能只因为同名 .pt 存在就复用。
10. 原子写缓存：先写临时文件，再改名；提取失败输出清单，不能产生半文件。
11. 新增重复 sample_id/组合映射检查；同一 ID 对应不同四链序列直接报错。
12. 支持 ESM 阶段与 Protenix 阶段分离，避免两套模型常驻 GPU；不能假定 torch.cuda.empty_cache() 会释放仍被引用的模型。
13. 多 GPU 子进程继承完整提取配置，每个样本只由一个 shard 写入；共享 ESM 链缓存需提前准备或使用互斥锁。

建议 payload：

~~~python
{
    "sample_name": sample_id,
    "input_sequences": [tcra, tcrb, peptide, hla_clip],
    "token_asym_id": ...,
    "token_residue_index": ...,
    "s": ...,
    "z": ...,
    "p_dist": ...,
    "channel_dims": {"s": Cs, "z": Cz, "p_dist": Cd},
    "extraction_config": {
        "model": "protenix_mini_esm_v0.5.0",
        "esm_model": "esm2_t36_3B_UR50D",
        "esm_enabled": True,
        "cycles": 4,
        "use_msa": False,
        "use_template": False,
        "use_rna_msa": False,
        "seed": 101,
        "inference_precision": "bf16",
        "cache_precision": "fp16",
        "package_version": "...",
        "feature_schema_version": 2,
        "esm_representation_layer": ...,
        "esm_cache_precision": ...,
    },
}
~~~

Cs/Cz/Cd 根据实际输出读取，不能假定换 encoder 后一定与历史缓存相同。

### 5.2 单链 ESM 缓存与数据准备新入口

拟新增 scripts/prepare_hitph_mini_esm_experiment.py，负责：

- 读取已固定 splits，通过当前 PeptideMismatchSampling 恢复身份；
- 输出四链 manifest、计划 epoch 的负例列表、联合提取分片；
- 从所有需要的输入序列去重形成 chain_catalog.csv；
- 不调用固定 val/test 生成函数，不写回原始 splits；
- 提取任务 name 与训练 sample_id 完全一致；
- 输出 run_spec，统一提取器和训练器的预期配置。

拟新增 scripts/cache_tmp_esm_chains.py，或等价地复用 Protenix ESMFeaturizer：

- 包含 TCRα、TCRβ、HLA 和 peptide 四类序列，ESM 路线中的 peptide 也需要序列表示；
- 全局按完整序列去重，保存序列到稳定 ID/文件的映射；禁止只按 CDR3 去重；
- 逐残基 token 表示，不用 mean pooling 的向量替代；
- 与官方 mini-ESM 的层号、维度、特殊 token 处理及精度规则一致；
- 缓存读取必须核对序列和配置，坏缓存不能静默当成零向量；
- 预计算完成后退出 ESM 进程，再加载 Protenix；
- 当前上游 infer_dataloader 会设置 ./esm_embeddings/... 路径，必须检查实际安装版本并适配，不能假定只改 configs.esm.embedding_dir 就有效；
- 所有缓存实际位于用户指定 profile 下，不散落到启动工作目录。

不拼接其他复合物的 s/z/p_dist 构造负例。单链 ESM 可复用，联合特征必须为每个新组合重新计算。

### 5.3 src/foldabs_repro/tmp_prediction/cache_audit.py

当前：validate_feature 已校验 input_sequences；audit_cache 的 require_extraction_config 分支仍要求 protenix-v1 且 use_msa=true。

拟修改：

- 用 expected_profile/expected_config 传入实验预期，替代固定 MSA 断言。
- mini-ESM profile 要求准确模型名、esm_enabled=true、cycles=4、use_msa=false。
- 校验所有缓存配置一致且与外部 run_spec 一致，不能只要求它们彼此相同。
- 保留 sample_name、四链序列、token 数、有限值和通道检查。
- 增加 p_dist 非负、最后一维和约等于 1 的检查，按 FP16 给合理容差。
- 不把缓存类别标签当输入特征；相同四链输入只能对应一致的特征配置。

### 5.4 scripts/train_tmp_specificity.py

动态负采样已经实现，保留 PeptideMismatchSampling，不另写一套采样算法。

拟修改：

1. 新增 --feature-profile/run-spec，启动和每轮缓存审计都检查 mini-ESM 身份。
2. 将采样接口和 sample_id 映射移到可复用模块或公开给准备脚本，保证计划列表与训练列表逐项相同。
3. 保持正例 record_id；负例仍为 pepneg_<split>_<组件编号>_<候选peptide编号>。
4. 从 first payload 动态读取 Cs/Cz/Cd；验证当前可训练 Pairformer 能适配。
5. 四块下游 Pairformer 和 attention pooling 与非 ESM 对照完全一致。
6. 训练 run_config 记录 encoder profile、缓存目录、采样 seed、head 设置；checkpoint 恢复验证配置一致。
7. 保留每轮 validation、BCE 最优模型、早停与 scheduler；test 只评估 validation-best。
8. 将当前动态日志中的 json.dumps(epoch_report) 改为常规文本统计，遵守已确定日志格式。
9. 保存训练与验证曲线；predictions.csv 只保存 test、prediction 为 0/1，AUROC/AUPRC 使用内部连续分数。
10. metrics 不输出 per_pmhc 明细，不添加 git_sha/manifest_sha256。

scripts/extract_protenix_local_msa_only.py 不用于本次实验，因为它要求每条链都存在 A3M。使用更新后的通用 extractor，无需为了绕过校验人为生成 MSA 文件。

## 6. 动态负例缓存覆盖：必须解决的依赖

只缓存 train.csv 的正例或 epoch 1 负例，不能支持后续动态训练。当前训练器在每轮检查该轮联合缓存，缺失立即报错。

本次首先采用“计划 epoch 抽样并集”准备策略：

1. 在 run_spec 中锁定最大 epoch H、negative_sampling_seed、模型 seed、抽样比例和输入列表。
2. 使用现有 epoch(e) 逐轮生成 e=1..H 的抽样 manifest。
3. 合并所有 train 正例、各轮抽到的负例，以及固定 val/test，按 sample_id 和四链身份去重。
4. 流式写 planned_union_manifest 和分片 JSON，不把联合特征张量放进内存。
5. 提取并审计这个并集，然后启动同一 seed、最大 H 的训练。
6. 训练仍然动态抽样；提前规划只是预先知道缓存需求，不固定每轮负例为同一集合。
7. 若改 seed 或增加最大 epoch，先补齐新增组合并审计，再继续训练。

不枚举并缓存所有理论可行负例。也不将现有候选库裁成固定 K 池，因为这会改变已确认的动态采样协议。

缓存 union 可能很大，必须先 dry-run：

~~~text
N_union = N_train_positive + N_unique_selected_train_negative
          + N_fixed_val + N_fixed_test

joint_bytes ≈ N_union × 实测平均联合 .pt 大小
chain_ESM_bytes ≈ Σ_unique_sequence L × 2560 × 每元素字节数
                 + 文件与元数据开销
~~~

联合缓存主要随 token 数平方增长；N 使用完整可变区长度，不能沿用旧 CDR3 token 统计。

若 H 轮并集超出空间预算，采用第二阶段“epoch 前增量缓存”模式，保持相同抽样协议：

- 每轮先由 rank 0 生成与训练完全相同的 epoch manifest；
- 独立提取进程在该轮训练开始前补齐 missing IDs；
- 所有 rank 等待缓存审计通过后开始 head 训练；
- 不在 Dataset.__getitem__ 或 forward 中启动 Protenix；
- 提取失败、空间不足时停止，不换负例、不改变标签和比例；
- 新增可恢复的轮次准备状态，保持断点续训一致；
- 多卡 barrier、子进程资源分配和必要的 head GPU 释放/卸载要明确实现，避免两个模型抢同一张卡。

首版优先实现离线计划并集；资源审计不通过时，完成增量模式后再扩大训练。不能把空间不足解释为可以直接训练缺缓存的数据。

## 7. 下游 attention 设置与训练超参数

当前 repo pooling：

- A：全 mean，不属于本次确认的 attention 配置；
- B：single mean、pair attention；
- C：single/pair 均 attention。

用户已确认 attention，但本次消息未指定 B 还是 C。实施时从已确定的非 ESM 实验 run_config 或实际启动参数读取并沿用；不重新做 ABC，不擅自更换。若对照未明确记录，先将确切 B/C 固化到 run_spec，不能误用训练器默认 A。

同样沿用非 ESM 对照的 learning rate、weight decay、dropout、hidden dimension、gradient accumulation、scheduler、early_stop、precision 和训练预算，逐项记录。

模型初始化 seed 与 negative_sampling_seed 分开记录；第一轮优先和对照一致。抽样 seed 一致保证每轮看到相同负例组合，缓存 encoder 仍不同。不能复用非 ESM head 权重进行该比较；各组独立初始化训练。

## 8. 执行顺序与并行安排

| 顺序 | 工作 | 可并行 |
|---|---|---|
| P0 | 核对 splits、固定评估、baseline 参数，写 run_spec | 下载 mini-ESM/ESM2 权重 |
| P1 | 修改 extractor、缓存配置审计与准备接口 | 单链 ESM 缓存工具开发 |
| P2 | epoch 抽样计划 dry-run、unique chain catalog、磁盘预算 | head profile 接入与 CPU 测试 |
| P3 | 50 条代表性四链输入的 ESM/联合提取检查 | 不重叠序列的独立 ESM 分片 |
| P4 | 小规模完整训练闭环验证 | 正式固定 val/test 缓存准备 |
| P5 | 按预算提取计划并集，完整审计 | 每 GPU 一个独立联合特征 shard |
| P6 | 4-block + 固定 attention 正式训练 | 已确定的非 ESM 对照，可用独立 GPU |
| P7 | 固定内部 test 分层与外部评估 | 后续独立 baseline 工作 |

小规模验证使用现有 train 中的工程子集及固定 val 的明确子集，不改正式源文件；仍然遵守同 split/context/blacklist 规则。临时子集单独命名，不将它的训练/指标冒充正式实验。

小规模检查只验证流程和资源；不反复查看 test 或 IMMREP25 决定超参数。正式协议锁定后测试。

## 9. 命令模板

以下路径与当前已有 preview 接口真实；新增准备/提取参数是拟实现接口，完成代码前不能直接执行全部模板。

~~~bash
PYTHONPATH=src /NAS/luyq/miniconda3/envs/foldtmp/bin/python scripts/train_tmp_specificity.py \
  --negative-sampling peptide_mismatch \
  --splits-dir /NAS/luyq/TMP_prediction/data/hitph_l4_v1/splits \
  --negative-sampling-seed 42 \
  --sampling-preview-epoch 1
~~~

预览不生成或修改 val/test，也不要求 CUDA/cache。

完成新增代码后，先准备配置与预算。下例 H=3 仅用于缓存预算/工程检查；正式 H 应沿用对照已确定预算，不默认把 3 个 epoch 当正式收敛实验。

~~~bash
SPLITS=/NAS/luyq/TMP_prediction/data/hitph_l4_v1/splits
PROFILE=/NAS/luyq/TMP_prediction/protenix_embeddings/hitph_v1/mini_esm_v0.5.0_c4_no_msa
PYTHON=/NAS/luyq/miniconda3/envs/foldtmp/bin/python

PYTHONPATH=src "$PYTHON" scripts/prepare_hitph_mini_esm_experiment.py \
  --splits-dir "$SPLITS" \
  --output-dir "$PROFILE/prepared" \
  --model-name protenix_mini_esm_v0.5.0 \
  --cycles 4 --no-msa \
  --negative-sampling-seed 42 --max-epochs 3 \
  --dry-run
~~~

dry-run 通过后，以同一参数去掉 --dry-run 生成准备产物；准备工具须生成用于小规模检查的 all_records.json 和正式 extraction_shards。然后完成单链 ESM 缓存：

~~~bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 "$PYTHON" scripts/cache_tmp_esm_chains.py \
  --chain-catalog "$PROFILE/prepared/chain_catalog.csv" \
  --output-dir "$PROFILE/esm_chain_cache" \
  --model-name esm2-3b \
  --run-spec "$PROFILE/prepared/run_spec.json" \
  --resume
~~~

该工具接口亦为拟新增；其计算层号和精度从 run_spec/官方 featurizer 读取，不能通过上述模型名猜测。进程完成并退出后再提取联合特征。

提取 smoke 样本（all_records.json 为准备脚本拟生成的工程清单；全量按 extraction_shards 执行）：

~~~bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 "$PYTHON" scripts/extract_protenix_embeddings.py \
  --input-json "$PROFILE/prepared/all_records.json" \
  --output-dir "$PROFILE/features" \
  --model-name protenix_mini_esm_v0.5.0 \
  --esm-cache-dir "$PROFILE/esm_chain_cache" \
  --cycles 4 --no-msa --seed 101 \
  --dtype bf16 --cache-dtype fp16 \
  --limit 50
~~~

全量提取必须显式 --limit 0，并使用同一配置的 --resume。不能在 cycles/模型/输入改变后沿用旧缓存。多卡要确保 --gpu-ids 的子进程完整继承 model/ESM/profile 参数。

训练模板的 POOLING_MODE、MODEL_SEED、TRAIN_EPOCHS 和全部训练超参数从对照/run_spec 读取。BASELINE_ARGS 表示按真实已确定对照补齐的参数列表，不是现有 CLI 名称：

~~~bash
PYTHONPATH=src "$PYTHON" scripts/train_tmp_specificity.py \
  --negative-sampling peptide_mismatch \
  --splits-dir "$SPLITS" \
  --features-dir "$PROFILE/features" \
  --feature-profile "$PROFILE/prepared/run_spec.json" \
  --output-dir "$RUN_OUTPUT" \
  --negative-sampling-seed 42 --seed "$MODEL_SEED" \
  --negatives-per-positive 2 \
  --pairformer-blocks 4 --pooling-mode "$POOLING_MODE" \
  --epochs "$TRAIN_EPOCHS" --gpu 0
~~~

--feature-profile 为本计划新增参数；RUN_OUTPUT 等变量需由执行端设置。模板未列出的学习率、scheduler、精度等仍需显式填写对照值，不能因为默认值存在而忽略一致性。

## 10. 必要检查与验收

准备与身份：

- 固定 val/test 文件不变，train 仍为正例，8 个既定文件完整。
- 黑名单/context 与当前交付一致；source IDs 和 pepneg IDs 与训练映射一致。
- 相同 seed/epoch 的准备清单和训练清单逐项相同，跨轮新的负例被纳入缓存计划。
- 四链序列顺序正确，HLA 使用现有 clip，TCR 使用 α/β 可变区。

ESM 与 encoder：

- runner 的 model_name/esm.enable/use_msa/cycles 与 run_spec 完全一致。
- 四链 ESM token 数与序列一致，special tokens 已按官方规则处理。
- 逐残基 ESM缓存同序列可复用；联合特征新组合独立提取。
- Backbone eval/no_grad；不修改权重、不执行 diffusion。
- 所有浮点张量有限、p_dist 有效、角色数为四，缓存序列和配置匹配。
- model/cycles/MSA 配置不匹配、损坏/旧无身份缓存均被拒绝。

训练与恢复：

- 每轮全局逻辑样本严格 79,465/158,930；明确记录 DDP padding 的额外访问，不能称之为新的合法负例。
- attention 模式是与对照相同的 B/C，不是默认 A。
- 每个 epoch 后使用固定 validation，早停 patience 与对照相同。
- 完整 epoch 恢复时抽样列表、优化器、scheduler、RNG、缓存 profile 一致。
- epoch H 之外的恢复扩展先补缓存，不能运行到缺失才临时改变采样。
- predictions.csv 只输出 test 的二值 prediction；连续分数用于 AUROC/AUPRC。
- 内部 test 按目标 peptide 到 train 的距离分层；IMMREP25 保留原始数据，另设缓存 profile/目录，不能重新生成标签。

测试范围只覆盖关键身份、配置、动态采样/缓存覆盖、resume 和训练闭环；不重复编写镜像实现的测试。正式 GPU 训练和大规模提取由用户执行端运行。

## 11. 最终交付与完成标准

代码实现完成后应交付：

1. mini-ESM 权重/环境准备说明和可执行入口；
2. 数据适配、单链 ESM 缓存及计划 epoch 联合缓存准备工具；
3. profile 严格校验、提取/恢复及必要并发控制；
4. 保留当前动态 peptide mismatch 与固定 val/test 的训练命令；
5. smoke 资源报告与全量缓存覆盖报告；
6. 与非 ESM 对照一致的 head/训练配置和正式实验结果。

本次仅新增实验计划文档；未修改 encoder/head/负采样代码，未访问 NAS，未下载权重、提取 embedding 或启动训练。

## 12. 参考与优先级

- [最新负采样交付说明](hitph_dynamic_negative_sampling_implementation_results.md)：当前协议依据。
- [官方支持模型](https://github.com/bytedance/Protenix/blob/main/docs/supported_models.md)。
- [官方模型配置](https://github.com/bytedance/Protenix/blob/main/configs/configs_model_type.py)。
- [官方 ESM 准备源码](https://github.com/bytedance/Protenix/blob/main/protenix/data/esm/compute_esm.py)。
- [官方 inference dataloader](https://github.com/bytedance/Protenix/blob/main/protenix/data/inference/infer_dataloader.py)。

历史文档中的“必须使用 A/B/H MSA”“train 有限候选池动态选取”不适用于本次实验；以本文件的 mini-ESM/no-MSA 配置和最新 peptide mismatch 交付为准。保持其他已确定实验不变。
