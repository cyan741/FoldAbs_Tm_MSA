# Group Rank：A100 验收与执行记录（2026-10-29）

## 实现与环境

入口 `scripts/train_mini_esm_online_ddp.py --loss-type group_rank`，独立训练实现为 `scripts/train_mini_esm_group_rank.py`，采样/损失为 `src/foldabs_repro/tmp_prediction/group_rank.py`。默认 BCE 保留。命令、参数和产物定义见 [使用说明](TMP_GROUP_RANK_USAGE.md)。

远端 SSH：`Bitahub_A100`。新代码隔离在 `/root/foldabs_group_rank_20261029/repo`，复用 `/root/foldabs_immrep25_20261029/env/bin/python`、原 Protenix 资产及 ESM 链缓存。新产物根目录为 `/NAS/luyq/foldabs_group_rank_20261029`。

未修改原 BCE 运行代码目录、输出目录或进程。两张 A100 80GB 原任务每卡占约 28–29 GiB；新增 group 模式启用 activation checkpointing，预检每卡峰值预留显存约 7.8–9.3 GiB，并行时总占用约 38–39 GiB。显存余量足够，允许启动；两任务共享计算资源，因此吞吐会下降，不能保证原训练耗时不变。

## 验收

- 27 项测试通过，包含真实双进程 Gloo DDP 对照串行梯度更新、可变 K、累积尾窗、checkpoint 重计算、已知阳性排除、候选不足/零候选、HLA context 隔离、默认 BCE 兼容、只导出 test 分数和验证主指标缺失时停止。日志：[tests.log](benchmarks/group_rank_20261029/tests.log)。
- 真实数据 epoch 1/2 采样重复生成完全一致，同组固定自然 TCR/HLA，负肽无重复，无黑名单冲突，DDP 分组无交集。报告：[sampling_audit.json](benchmarks/group_rank_20261029/sampling_audit.json)。
- 每轮 4,960 个有效训练组，4,958 个 K=2、2 个 K=1，合计 9,918 个负例；无跳过 anchor，无双卡尾部丢弃。相比 BCE 的全局补足 1:2，本模式不把短缺负例补到其他组。
- 完整验证集 7,737 records，137 个可计算 AUROC 的 pHLA、2,531 个固定 TCR/HLA 验证组，48 个 anchor 无已有匹配负例。测试集 1,947 records，138 个可计算 pHLA、648 组、1 个无匹配负例 anchor。原 records 均保留。
- 单卡最长样本组在线训练 2 步：梯度有限，encoder 无梯度，峰值预留 7,992 MiB。见 [single_benchmark.json](benchmarks/group_rank_20261029/single_benchmark.json)。
- 双卡在线训练 2 步，再从 step 2 / group round 2 恢复到 step 4 / round 4，共处理 8 组、24 条样本；两 rank 参数校验和相同，encoder 无梯度。恢复阶段峰值预留最高 9,488 MiB，见 [ddp_resume_benchmark.json](benchmarks/group_rank_20261029/ddp_resume_benchmark.json)。
- 预检短窗口约 19.3 秒/组/卡、0.155 样本/秒/卡，包含采样、worker 启动等开销且与原任务共享 GPU，不代表稳定训练吞吐或纯前向耗时。正式运行的后续 benchmark 才适合估计耗时。
- 从 step 4 恢复完成 12 条固定验证 records、12 条固定测试 records 的评估；`smoke_ddp/completion.json` 为 `smoke_complete`，见 [smoke_completion.json](benchmarks/group_rank_20261029/smoke_completion.json)。只有测试预测文件，连续 score 有限。
- 初次随机小子集未覆盖任何双标签 pHLA，按设计抛出“主指标不可计算”错误。预检选择器已改为从已有 records 中覆盖双标签 pHLA、完整 TCR/HLA 组及最长正负样本；没有替换模型选择指标，也没有改变完整 val/test records。预检数值不能用于评估模型效果。

## 正式训练

2026-10-29 **18:08:46 CST** 在 tmux session `immrep25_group_rank_20261029` 启动。使用全新 head，不继承 smoke 权重。

```bash
cd /root/foldabs_group_rank_20261029/repo
bash scripts/run_immrep25_group_a100.sh
```

配置：2 GPU，纯 group rank，K=2，margin=1，每卡累积 1 组；4-block Pairformer + attention pooling；冻结 mini-Protenix，4 cycles，无 MSA/templates；BF16，activation checkpointing；AdamW lr=1e-4、weight_decay=0.01，最多 10 epochs，patience=5。默认模型选择为验证 macro partial AUROC(max_fpr=0.1) 最大值。每轮 2,480 次优化更新。

输出目录：`/NAS/luyq/foldabs_group_rank_20261029/lr_0.0001_bs_2_wd_0.01_drop_0.1_group_rank_k_2`。

日志：`/NAS/luyq/foldabs_group_rank_20261029/train.log`。输出目录内 `benchmark.json` 提供最近训练指标，`progress.json` 提供最近 checkpoint 的位置，`history.json` 在完整 epoch 验证结束后写入。正式训练刚启动，尚无完整验证结论。

原任务 tmux session 为 `immrep25_cap50_a100_20261029`，两者同时存在。运行状态应结合各自日志及 PID 判断，不能只看 GPU 总利用率。

18:10 CST 确认正式训练 step 10 / epoch 1：累计 20 组，train group loss=1.311321，梯度有限、两 rank 参数一致、encoder 无梯度。两卡总显存约 39,276 / 40,305 MiB，空闲约 41,878 / 40,849 MiB。原 BCE PID 17191/17192 仍运行，原日志持续更新。已确认正式配置 smoke_samples=0、max_steps=0、resume=null，首步 last.pt 已保存。启动快照见 [formal_launch_benchmark.json](benchmarks/group_rank_20261029/formal_launch_benchmark.json)。
