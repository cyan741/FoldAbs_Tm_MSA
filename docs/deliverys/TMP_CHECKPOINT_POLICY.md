# TMP 每轮 checkpoint 保存约定

自 2026-10-29 起，TMP 在线 BCE、在线 Group Rank 和离线 `train_tmp_specificity.py` 均保留每轮权重，使用：

```text
bs{batch_size}_lr_{learning_rate}sd_{seed}_ep{epoch}.pt
# 当前双卡、K=2、A=1 示例
bs6_lr_0.0001sd_42_ep1.pt
```

- `batch_size` 是全局名义样本数：BCE 为卡数 × 每卡样本累积数；Group Rank 为卡数 × 每卡组累积数 × (1+K)。可变 K 的实际样本数可能较少，组数/实际 K 仍记录在采样日志中。
- `learning_rate` 使用配置的初始学习率，格式为 Python `:g`；调度后的实际学习率另存 optimizer/scheduler 状态。
- `epoch` 从 1 开始，指该文件权重完成训练的轮次；不要仅依据旧检查点的内部 `epoch` 字段判断完成轮次，在线训练的内部字段有时表示下一轮恢复位置。新归档统一添加 `completed_epoch`。
- 每轮文件真实独立保存，不是指向滚动 checkpoint 的链接，不删除旧 epoch，不以新权重覆盖同名文件。重新进入验证时，若同名归档的 step/权重一致则复用，否则报错。
- 在线模式在每轮训练最后一次 optimizer 更新完成后、验证开始前保存，含模型、optimizer、scheduler、逐 rank RNG、cursor、累计统计及配置；从该点恢复会先完成该轮验证。`checkpoint_stage=training_complete_validation_pending` 标明阶段。
- 离线 TMP 在该轮验证完成后归档同一份训练权重，并保存完整的 epoch-boundary 恢复状态。
- `last.pt`、`best.pt`、Group Rank 的 `final.pt` 继续按原职责保存；它们不替代每轮归档。离线 TMP 的既有 best/final 命名同样保留。
- 新增或重构 TMP 训练入口时必须复用 `foldabs_repro.tmp_prediction.checkpoints` 的归档/命名函数，保持本约定。

## 当前远端运行迁移

Group Rank 已先备份原 step 2,400 checkpoint 为 `resume_before_epoch_archive_update_step2400.pt`，仅停止其 launcher，2026-10-29 22:47:40 CST 从该文件恢复。原进程当时已完成 step 2,480 并开始验证，但旧代码未保存该位置，因此重算该轮最后 80 步，再由新代码归档完整第 1 轮权重。不是从新 head 开始。

原 BCE 不停止：`archive_legacy_online_epochs.py` 在独立 tmux session `bce_epoch_archive_20261029` 中观察旧入口的整轮 checkpoint，并归档到原输出目录。第 1 轮最佳权重已补充保存；第 2～5 轮已被覆盖，无法补回。第 6 轮及之后在旧任务完整验证写入 checkpoint 后归档。任务产生 `completion.json` 后监视器退出。

本地源码和远端两份代码部署目录均已更新，后续启动的 TMP 训练直接使用内置归档。旧 BCE 进程执行的已加载训练循环不变，其原 PID 和任务均保留。

验证：原相关 37 项测试通过；新增 legacy 归档测试后，归档模块 3 项测试通过（总计 38 个不同测试）。覆盖文件名、Group Rank 的 batch 计算、跨 epoch 保留、同轮恢复幂等、冲突拒绝、离线训练每轮归档和旧在线任务完成轮次识别。

2026-10-30 00:05 CST 已实测读取 Group Rank 的 `bs6_lr_0.0001sd_42_ep1.pt`：166,083,806 字节，completed_epoch=1，step=2,480，cursor=2,480，阶段为 training_complete_validation_pending；后续验证正在运行。原 BCE PID 17191/17192 持续存活，独立归档监视器正常运行。

恢复后第 1 轮累计 group loss=1.028774，暂停前同一步为 1.028590；两次训练结果接近但并非逐位一致，未声称 GPU 重算完全确定性。恢复前后原始记录见 `benchmarks/group_rank_20261029/epoch_archive_resume_verification.json`。

2026-10-31 00:04:38 CST：Group Rank 从 `resume_before_native_no_recompute_step20200.pt` 恢复（epoch9/cursor360），去掉FP16特征中转并关闭激活重计算。迁移前历史8轮及归档均保留；后续归档命名不变。新执行配置与变更点写入 run_config.execution_changes，详情见 [迁移记录](TMP_NATIVE_NO_RECOMPUTE_20261031.md)。
