# 在线训练去除 FP16 中转及关闭激活重计算

## 实现

- `OnlineFeatures.infer` 直接返回 single、pair、probability，不调用 `.half()`。实测返回 dtype 为 BF16、BF16、FP32，链 ID 为 int64。
- 分类头入口 `.float()` 与 CUDA BF16 autocast 保留，参数、loss、AdamW 状态保持既有策略；编码器仍 eval + requires_grad_(False) + no_grad。
- 在线 BCE/Group Rank 默认 `activation_checkpoint=false`，A100 group 启动脚本明确传 `--no-activation-checkpoint`。仍支持显式开启。
- 旧 checkpoint 的精度/重计算配置变更需要 `--allow-execution-config-change`，并写入 execution_changes。其他种子、数据指纹、loss、卡数等恢复约束不放宽。
- 本地代码和远端原 BCE、Group Rank 两个部署目录均更新。离线分类头本来默认关闭重计算。

## GPU 对照

A100 80GB；同 epoch 6 权重和 AdamW 状态；2 组预热，11 组/33 条按长度分位选取样本；每组1正2负；单卡同步计时，不包含 DDP 通信、checkpoint、整轮验证。两组均去掉 FP16 中转。

| 指标 | native + 重计算 | native + 不重计算 |
|---|---:|---:|
| 完整组耗时 | 3.1752 s | 3.0233 s |
| 编码器4cycles，每条 | 706.0 ms | 705.1 ms |
| 分类头前向，每条 | 53.25 ms | 52.49 ms |
| 分类头反向，每组 | 577.53 ms | 443.68 ms |
| 最大活跃张量 | 6.97 GiB | 36.62 GiB |
| allocator 最大预留 | 9.86 GiB | 60.83 GiB |

关闭重计算后本次组耗时减少4.79%，吞吐提高5.03%，反向耗时减少23.18%。小样本基准不等于整轮提速；编码器和验证仍占大部分时间。不重计算的内存增长明显，但当前2张80GB卡、K=2可运行。

两次独立带 dropout 的性能测试不是配对数值等价实验：最大组loss差约0.17，不宣称逐位复现。另做同权重同输入、dropout=0的执行等价检查：logits和loss一致；261个参数梯度均有限，整体相对L2误差 0.00034123，方向余弦 0.999999941785。最大绝对误差 0.00195312。逐元素rtol=1%、atol=1e-4在部分近零梯度失败；按BF16整体误差门限relative L2<1%、cosine>0.9999通过。保留原始误差记录，不宣称梯度逐位一致。

30项相关测试通过，覆盖恢复变更显式授权、历史保留、重计算梯度、分组、DDP分片、epoch归档。GPU双卡24条最长组训练与val/test各6条预检通过；DDP checksum一致、encoder_has_grad=false，loss/梯度有限。

## 正式任务迁移

原任务曾短暂挂起用于测量，finally恢复。测试期间原任务完成epoch8验证，进入epoch9。

最终从 `resume_before_native_no_recompute_step20200.pt` 恢复：epoch9、cursor360、step20200，保留前8轮history及epoch归档、model、optimizer、scheduler和逐rank RNG。原停止前最近日志step=20240，该检查点之后未保存步骤会重算。迁移改变后续数值轨迹，并非全程同一精度设置的独立新实验。

启动时间2026-10-31 00:04:38 CST，tmux `immrep25_group_native_20261031`，继续使用原输出目录和 `train.log`。迁移信息记录于run_config.execution_changes。原运行95037/95056/95057退出，新进程重新加载旧训练状态。每轮epoch权重命名/永久归档规则不变。

原始数据见 [benchmarks/native_no_recompute_20261030](benchmarks/native_no_recompute_20261030)。

正式恢复验收：epoch9 step20220、cursor380，累计训练loss 0.292945，梯度范数0.207519，DDP checksum一致、编码器无梯度。rank0峰值allocated 35940.60 MiB、reserved46608 MiB。见 `benchmarks/native_no_recompute_20261030/resumed_benchmark.json`。
