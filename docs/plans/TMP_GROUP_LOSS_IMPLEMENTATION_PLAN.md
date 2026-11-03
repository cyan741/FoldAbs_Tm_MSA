# TMP 纯 Group Loss 实施清单

日期：2026-10-29。状态：已实现；使用说明见 `TMP_GROUP_RANK_USAGE.md`，验收与远端运行记录见 `TMP_GROUP_RANK_A100_20261029.md`。

## 1. 目标与配置

新增 `loss_type=group_rank`，训练目标仅使用组内排序损失，不混入 BCE。保留原 BCE 模式用于对照。Protenix 冻结、无 MSA、4-block Pairformer＋attention pooling 保持现有设置。

| 参数（拟新增） | 默认值 | 含义 |
|---|---:|---|
| `loss_type` | `bce` | 可选 `bce/group_rank`，兼容旧任务 |
| `group_negatives` | 2 | 每组负例上限；图片配置为 3 |
| `group_margin` | 1.0 | 排序 margin |
| `group_accumulation_steps` | 1 | 每卡累积完整 group 数 |
| `group_require_full` | false | true 时排除负例不足的 anchor |

首轮使用 K=2 保持正负 1:2 的预算。每次复合物前向仍为 1 条样本。

## 2. 修改采样器

- [x] 仅在当前 train 核心集构造 group，继承原有 split。
- [x] 每条正例作为 anchor；固定自然配对 TCRαβ 和 HLA，只替换 peptide。
- [x] 从同 split、同 HLA context 的训练正例 unique peptide 池均匀抽负 peptide；保证负 peptide 在训练集中也具有正例角色。
- [x] 排除 anchor peptide，以及该 TCR 在完整 Hi-TPH 黑名单中的全部已知阳性 peptide；保留未入核心集的已知阳性记录用于排除。
- [x] 同组负 peptide 无放回抽取；按 `seed + epoch` 确定性生成并打乱 groups。
- [x] 候选不足 K 时使用实际 K_g；候选为 0 时跳过并统计。
- [x] 统计有效/跳过组数、实际负例数、每个 pHLA 的正负出现次数。

动态负例只需保存采样清单；沿用在线 GPU 特征生成，不要求落盘保存负例 embedding。

## 3. 实现纯排序损失

每组含一个正例和 K_g 个负例，所有 s 均为 head 原始 logit：

\[
L_g=\frac{1}{K_g}\sum_{k=1}^{K_g}\operatorname{softplus}(m+s^-_{g,k}-s^+_g),\qquad
L=\frac{1}{G}\sum_{g=1}^{G}L_g.
\]

- [x] FP32 计算分数差和 softplus；不先 sigmoid，不加入 BCE。
- [x] 先组内平均，再按有效 group 平均；可变 K_g 时不能直接平均全部比较项。
- [x] 正负 logits 保留梯度；禁止 detach 后计算排序损失。

```python
pos = positive_logit.float().reshape(())
neg = negative_logits.float().reshape(-1)
group_loss = F.softplus(margin + neg - pos).mean()
```

## 4. 改造前向、反向与 DDP

- [x] 按完整 group 分卡，同组正负不得分散到不同 rank。
- [x] 一个组级 `forward` 内依次处理正例及负例：冻结 backbone 使用 `eval/no_grad`，head 保留计算图，整组完成后 backward。
- [x] 保留现有 activation checkpointing；实测组训练峰值显存。
- [x] 新模式不复用旧的“逐样本累积 3 步”；改为累计 A 个完整 group，每组损失除以 A。
- [x] 标准 DDP 梯度平均；各 rank 每次更新参与相同数量的有效组，优化步数保持一致。
- [x] 预先生成有效组清单后再分卡；末尾不能整除卡数时统一丢弃少量组并记录，不静默复制。最后不足 A 的累积窗口按实际组数归一化。

固定 K 时，每次更新的样本数为 `GPU数 × A × (1+K)`。

| 两卡、A=1 | 每卡操作 | 一次更新 |
|---|---|---|
| K=2 | 3 次前向组成一组，1 次组反向 | 2 组、6 条样本 |
| K=3 | 4 次前向组成一组，1 次组反向 | 2 组、8 条样本 |

组内已经平均后，不再额外除以前向次数 3 或 4。梯度裁剪、optimizer step 和按步调度器均在完整累积窗口结束后执行。

## 5. 验证、保存与日志

- [x] 保持原有固定 val/test records 不变；只对 test 输出预测文件。
- [x] 从既有验证 records 按相同 TCR/HLA 构造固定验证 groups，只用已有负例；
- [x] 新模式默认以验证集标准化 `macro partial AUROC(max_fpr=0.1)` 最大值选择模型和早停；按 pHLA 等权平均。
- [x] 若验证主指标不可计算，直接报告并停止模型选择，不静默替换指标。
- [x] 保存验证最优与最终/早停模型，沿用超参数命名体系；配置明确记录 `selection_metric`，不把最优 checkpoint 标为 BCE 最优。
- [x] 日志与曲线记录 `train_group_loss, valid_group_loss, valid_macro_pauc_0.1, valid_macro_auroc, mean_score_gap, pairwise_accuracy, num_groups, skipped_anchors, lr`。
- [x] test `predictions.csv` 增加连续 `score`（logit）用于排序指标；保留 `prediction` 时仅用 validation 预先确定的阈值。不要把 sigmoid(score) 宣称为校准概率。
- [x] run_config 保存 group 参数、种子和选择指标；继续不记录 git_sha、manifest_sha256。

## 6. 验收与执行顺序

1. **采样审计**：验证同组 TCR/HLA 一致、无组内重复负例、无黑名单冲突、同 seed/epoch 可复现。
2. **损失测试**：正例分数提高或负例分数降低时 loss 下降；整体平移 logits 时 loss 不变；检查可变 K_g 的组等权平均。
3. **单卡小样本**：确认梯度仅更新 head、无 BCE 项、无 detach，覆盖候选不足与零候选情况。
4. **在线测速**：记录 K=2 的组耗时、峰值显存、有效吞吐；通过后再启动正式训练。

交付物：新增训练模式、group sampler/loader、损失与 DDP 验证、命令示例、采样审计报告。
