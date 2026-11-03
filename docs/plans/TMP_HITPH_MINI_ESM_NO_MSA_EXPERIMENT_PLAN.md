# Hi-TPH：Protenix mini-ESM 无 MSA执行计划

日期：2026-10-12  
## 1. 本次实验目标与范围
验证公开 protenix_mini_esm_v0.5.0（ESM2-3B）产生的联合 s/z/p_dist 表示，能否在当前 Hi-TPH unseen-peptide 设置下改善结合预测。

已确认：

- 暂不使用 MSA，所有四个组件均关闭 MSA；不生成 query-only A3M，也不启动远程搜索。
- 非 ESM Protenix 及其下游 4-block Pairformer + attention pooling 的训练方案已经确定，作为当前对照；本计划不重新选择该模型的 head。
- 新实验仅更换冻结 encoder 为官方 mini-ESM；下游继续使用同一个 4-block Pairformer + attention pooling 和相同训练协议。
- 不重新清洗、切分或重建固定 val/test 负例。
- 保留自然配对 TCRα/β 可变区、peptide、HLA 连续 clip，四链顺序 tcra/tcrb/peptide/mhc。
- 本轮只提取 trunk 的 s/z/p_dist，不新增 diffusion、geometry、PAE、ipTM。

## 2. 固定路径与隔离规则

训练验证测试数据划分目录：

~~~text
/NAS/luyq/TMP_prediction/data/hitph_l4_v1/splits
~~~
说明文档：[HiTPH peptide mismatch 交付](hitph_dynamic_negative_sampling_implementation_results.md)。

| 文件/逻辑集 | 正例 | 负例 |
|---|---:|---:|
| train.csv | 79,465 | 0 |
| 每个 train epoch | 79,465 | 158,930 |
| val.csv | 3,951 | 7,902 |
| test.csv | 1,784 | 3,568 |

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
- 每轮训练随机打乱正样本
  
相关脚本在：
1. scripts/negative_sampling/peptide_mismatch.py 的现有抽样逻辑；
2. scripts/train_tmp_specificity.py 的 PeptideMismatchSampling 身份映射、动态训练接入。

输出embedding 缓存根目录：

~~~text
/NAS/luyq/TMP_prediction/protenix_embeddings/hitph_v1
~~~

在指定根目录下新增独立 profile目录 mini_esm_v0.5.0_c4_no_msa，避免覆盖现有非 ESM 缓存：

--features-dir 指向上述 features 子目录，而不是 hitph_v1 根目录。训练 checkpoint 和日志放入现有训练输出目录体系，单独使用 mini_esm_no_msa 实验名；不与 encoder 特征文件混放。

## 3. Encoder 配置与资源准备

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
| 提取 seed | 42，显式设置实际 RNG，而不只传 runner 参数 |
| 输出 | s、z、p_dist、token_asym_id、token_residue_index |

官方 mini-ESM 默认 cycles=4、ESM2-3B、use_msa=false。其 trunk 为 mini 配置；不能仅将 base 模型关闭 MSA 就标成 mini-ESM。官方 mini 的 diffusion 默认值与本轮无关，不能把 n_step/n_sample 的占位参数说成已运行结构采样。

必须提前准备：
- mini-ESM Protenix checkpoint；
- ESM2-3B checkpoint 及所安装 fair-esm 加载路径要求的附属文件，所有需要的额外的包安装在foldtmp；
- esm checkpoint安装路径/NAS/luyq/TMP_prediction/protenix_assets/proteinix/protenix_assets/checkpoint；
- Protenix common 数据资源、正确的 PROTENIX_ROOT_DIR，当前base版本的protenix路径在/home/luyq/FoldAbS_Tm_MSA/.foldtmp-run/protenix_assets/checkpoint/protenix_base_default_v1.0.0.pt；
- 检查 get_default_runner 支持 model_name，ESM dataloader/featurizer 能正确准备 token embedding；
- 检查实际安装源码的 ESM 缓存路径、层号、BOS/EOS 去除规则和维度；
- 在正式运行前完成权重下载。

用官方模型配置构造结构并严格检查权重 missing/unexpected keys。ESM 权重启用必须从 runner.configs 实际读取；模型名写对而 esm.enable=false 的运行必须拒绝。

运行的 Protenix/PyTorch/fair-esm 包版本写入配置，不引入 git_sha、manifest_sha256。

## 4. 当前仓库需修改的位置与任务

### 4.1 scripts/extract_protenix_embeddings.py

当前：checkpoint 检查固定为 protenix_base_default_v1.0.0.pt；get_default_runner 未传 model_name；计算 BF16、保存 FP16；默认 cycles=1、limit=10；resume 主要检查字段。

拟修改：

- 新增 --model-name，保留 base 旧默认兼容；新实验显式 mini-ESM。
- 新增 --no-msa，与 --use-msa 互斥；本实验强制 false，并清除/拒绝输入中的 MSA/template 路径，防止隐式搜索。
- 新增 --seed、--dtype、--cache-dtype；示例使用 42/bf16/fp16。
- 新增 --esm-cache-dir，接入预计算单链表示；路径与 Protenix featurizer 的实际加载规则一致。
- resume 校验 model、ESM 模型及配置、cycles、精度、MSA/template 状态、四链序列、token 数和通道；不能只因为同名 .pt 存在就复用。
- 原子写缓存：先写临时文件，再改名；提取失败输出清单，不能产生半文件。
- 支持 ESM 阶段与 Protenix 阶段分离，避免两套模型常驻 GPU；不能假定 torch.cuda.empty_cache() 会释放仍被引用的模型。
- 多 GPU 子进程继承完整提取配置，每个样本只由一个 shard 写入；共享 ESM 链缓存需提前准备或使用互斥锁。



### 4.2 单链 ESM 缓存提取
参考源repo：https://github.com/bytedance/Protenix/blob/main/docs/supported_models.md?utm_source=chatgpt.com
完成模型权重下载， /NAS/luyq/TMP_prediction/humanI_uns_hitph_embeddings目录下是一个可以用来验证流程的小样本子集，已经分好train/val/test，大约400多个样本，可以用来跑通使用ESM+protenix提取的embeddings的流程，下游的训练应该不变，缓存需要在此目录下按照模型名称命名的文件夹下

检查ESM 与 encoder：

- runner 的 model_name/esm.enable/use_msa/cycles 与 run_spec 完全一致。
- 四链 ESM token 数与序列一致，special tokens 已按官方规则处理。
- 逐残基 ESM缓存同序列可复用；联合特征新组合独立提取。
- Backbone eval/no_grad；不修改权重、不执行 diffusion。
- 所有浮点张量有限、p_dist 有效、角色数为四，缓存序列和配置匹配。
- model/cycles/MSA 配置不匹配、损坏/旧无身份缓存均被拒绝。


## 最终交付与完成标准

代码实现完成后应交付：

1. mini-ESM 权重/环境准备说明和可执行入口；
2. 数据适配、单链 ESM 缓存及计划 epoch 联合缓存准备工具；
3. profile 严格校验、提取/恢复及必要并发控制；
4. 保留当前动态 peptide mismatch 与固定 val/test 的训练命令；
5. 子集小样本上的完成报告；
