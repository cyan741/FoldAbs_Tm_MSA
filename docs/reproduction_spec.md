# FoldAbS 小规模复现实验规格

## 目标与边界

本项目的近期目标是复现 FoldAbS 的**开发性（developability）分支**，并在同一可控任务上
验证 MSA 加速策略。它不是对整篇论文的逐比特复现，也不声称复现论文全部四个属性或最终
基准数值。

遵循两条原则：

1. **方法忠实（method fidelity）**：公开写明的模型结构、特征、训练目标和评估协议应尽量
   按论文实现。
2. **规模缩小（scale fidelity）**：遵循老师的建议，先选一个属性、少量样本和短实验，先
   验证数据 - 训练 - 测试链路；只有链路稳定后才扩大数据和 MSA 规模。

当前首选任务为 Jain2017/Jain2024/NbThermo 的 DSF 热稳定性（Tm）回归。

## 论文主配置：开发性分支

根据 FoldAbS 附录 A.2 - A.3，开发性任务的公开配置为：

- 冻结 Protenix-v1，使用单表示 `s`、成对表示 `z`、距离分布 `p_dist`；
- 对 `softmax(p_dist)` 线性投影，通过含两层残差 transition 的 PairCond 注入 `z`；
- 使用一个轻量 Pairformer 适配块共同更新 `s` 和条件化后的 `z`；
- 池化得到 `s_pool` 与 `z_pool`；
- 冻结 ESM-2 `esm2_t33_650M_UR50D`，分别平均池化重链、轻链，得到 `e_H`、`e_L`；
- 拼接 `[s_pool, z_pool, e_H, e_L]`，以隐藏维度 512、dropout 0.1 的三层 MLP 预测一个标量；
- 每个属性独立训练，而不是多任务共享回归头；
- MSE、AdamW（学习率 `1e-4`、权重衰减 `0.01`）、梯度累计 3 步、梯度裁剪 1.0；
- 最多 5000 steps；100 step warmup；每 100 steps 验证；根据验证 MSE 选择 checkpoint。

论文未公开所有实现细节，例如 Pool1D/Pool2D 的精确定义、PairCond 的隐藏层配置、随机种子
和具体 split 清单。对此，本项目会记录明确、可复现的合理实现，不把推定内容伪装成作者原码。

补充：Appendix C.3 的 standalone ESM-650M baseline 与完整 FoldAbS 头不是同一训练协议。
baseline 明确使用 batch size 32、lr `1e-4`、3000 steps；完整 FoldAbS 可开发性头使用
batch size 1、梯度累计 3、最多 5000 steps。两种结果不得混写。

## 当前实现状态

| 项目 | 状态 | 说明 |
| --- | --- | --- |
| Protenix-v1 与 RTX 5090 | 已验证 | CUDA smoke test 和结构推理成功 |
| 冻结 `s/z/p_dist` 表征 | 已验证 | 支持无 MSA 与缓存 MSA 输入 |
| Jain2017 137 条热稳定性基线 | 已验证 | 已有 Ridge 和近似 Pairformer 头结果 |
| 论文对齐 PairCond + ESM-2 头 | 已完成 | 已在公开 DSF 433 条、常规 MSA 特征上完成 5000 step 运行 |
| 公共 DSF 数据集重建 | 已完成 | 三个公开来源去重后得到 433 条，论文报告 430 条 |
| 完整论文 MSA 策略 | 进行中 | 正在以 200 条多样 Nb 种子构建紧凑序列库，并用本地 jackhmmer 检索 |

## 分阶段实验

### 阶段 1：最小可运行模型

1. 选择热稳定性一个属性。
2. 用 8 个样本检查：前向、反向、MSE、梯度和极小集过拟合。
3. 用小规模训练/验证/测试划分运行完整开发性头。
4. 记录 loss 曲线、验证 MSE、测试 Spearman、MAE、RMSE。

这一步对应老师所说的“先把测试搞通；随机参数也可以；先看 loss 降不降”。

### 阶段 2：公开 DSF 数据扩展

1. 合并 Jain2017、Jain2024 和 NbThermo 的 DSF (SYPRO) 数据。
2. 统一序列、以重/轻链组合去重并取 Tm 中位数。
3. 按重链分组，固定并保存 train/validation/test split 清单。
4. 明确报告与论文 430 条、337/30/63 划分之间的差异。

当前执行入口（均可在 AutoDL 远程项目根目录执行）：

```bash
# A. 网络/CPU 为主；manifest 在每条链完成后写盘，可安全重跑。
bash scripts/run_public_dsf_msa_cache.sh  # ColabFold 近似输入，不是论文 UniRef50-only

# B. 可在 A 运行时占用 GPU，建立严格的 no-MSA 对照特征。
bash scripts/run_public_dsf_nomsa_features.sh

# C. A 完成后建立同一批样本的 MSA 特征；--resume 会复用已保存样本。
bash scripts/run_public_dsf_msa_features.sh

# D. 对两种特征分别训练完全相同的 FoldAbS 开发性头。
bash scripts/run_public_dsf_developability.sh nomsa
bash scripts/run_public_dsf_developability.sh msa
```

MSA 缓存的 `cache_report.json` 记录唯一链数、初始命中、当前运行完成数和耗时；
`manifest.json` 才是断点续跑的真实依据。公共服务请求采用每条链一个请求，避免把无关的
重链/轻链误当作 paired MSA。

论文写明抗体 MSA 只搜索 UniRef50。严格的普通 MSA 重构入口为：

```bash
export FOLDABS_UNIREF50_FASTA=/path/to/uniref50.fasta
bash scripts/run_public_dsf_uniref50_msa_cache.sh
bash scripts/run_public_dsf_uniref50_msa_features.sh
bash scripts/run_public_dsf_developability.sh msa_uniref50
```

UniRef50 版本与 jackhmmer 搜索阈值未公开，保留为报告中可见的重构参数。

### 阶段 3：MSA 加速对照

FoldAbS 使用 MSA；AF3-TurboAb 改进的是 MSA 的准备速度，而不是取消 MSA。对照必须固定
下游 Protenix、开发性头、数据划分和随机种子，只改变 MSA 准备方式：

1. 普通 MSA 准备：作为时间和表征基线；
2. 加速 MSA：代表性抗体查询结果聚合/去重形成紧凑抗体数据库，辅以批处理和缓存；
3. 比较 MSA 时间、缓存命中、有效深度、下游指标。

当前实现的加速版本遵循 AF3-TurboAb 的可公开复原思想：从 200 条有代表性的纳米抗体的
**常规 MSA 命中序列**中聚合并去重出紧凑 FASTA 数据库；后续每条抗体链用本地 `jackhmmer`
在该固定数据库中检索，再转换为 Protenix 可读取的 A3M。论文没有公开精确种子、CDR/胚系
注释或数据库快照，因此本项目用“NbThermo 来源、空轻链、长度与 3-mer 序列多样性”选择种子，
并把这一点标为可审计的近似，绝不称为作者原始数据库。

运行顺序如下：

```bash
# 先补齐现有常规 MSA 基线的常数预测器报告；不重新训练、不调用 GPU。
bash scripts/run_public_dsf_baseline_audit.sh

# 从已经完成的常规 MSA 缓存建立紧凑 Nb 数据库；不联网。
bash scripts/run_public_dsf_compact_msa_database.sh

# 用本地紧凑库构建可断点续跑的加速 MSA 缓存。
bash scripts/run_public_dsf_compact_msa_cache.sh

# 保持 10 cycles、Protenix 冻结等条件不变，提取加速 MSA 表征。
bash scripts/run_public_dsf_compact_msa_features.sh

# 与常规 MSA 版本使用相同训练脚本、split 和随机种子。
bash scripts/run_public_dsf_developability.sh msa_compact
```

首次运行建议先做5条抗体的格式与速度 pilot；它只生成前5条输入的紧凑MSA JSON，不能用于
训练。确认 `cache_report.json` 中有合理深度后，以默认的 `0` 限制重新运行同一命令即可断点续跑
全部链：

```bash
FOLDABS_COMPACT_MSA_LIMIT=5 bash scripts/run_public_dsf_compact_msa_cache.sh
bash scripts/run_public_dsf_compact_msa_cache.sh
```

`run_public_dsf_compact_msa_cache.sh` 需要可执行的 `jackhmmer`。若环境尚未安装 HMMER，
它会停止并给出明确提示；不要用在线 ColabFold 搜索替代它，否则就不再是“本地紧凑数据库”
加速实验。

在开发性任务中没有抗原输入，因此 AF3-TurboAb 的“同一抗原 MSA/模板复用”不适用。它要在
未来的特异性或亲和力任务中才能单独验证。

## 结果命名和报告规则

所有结果目录名与 JSON 元数据必须包含：数据集、样本数、属性、是否 MSA、是否 ESM-2、cycle 数。
例如 `tm_jain2017_n137_msa_esm2_c10`。报告时必须标注：

- `smoke`：只验证代码通路；
- `small-scale`：小数据开发性实验，不与论文最终结果直接比较；
- `public-reconstruction`：公开数据重建，说明与论文数据差异；
- `msa-ablation`：只改变 MSA 准备方式的公平对照。
