# Group loss 分阶段性能实测（2026-10-30）

当前训练获用户允许后短暂 SIGSTOP，独立测量退出后 finally SIGCONT 原地恢复。17:14:02 恢复，之后确认原进程 95037/95056/95057 存活且 step 15100、DDP checksum 一致。未覆盖训练权重或优化器状态。

## 配置与口径

A100 SXM4 80GB，单卡；读取 group epoch 6 权重及 AdamW 状态。BF16、冻结 mini-ESM 编码器、4 cycles、4-block pooling C 分类头、activation checkpoint 开启。先预热 2 组，再测量 11 组、33 条样本（每组 1 正 2 负），按长度分位选取含最短最长 anchor，实际总 token 405–431。样本非总体随机加权。每阶段前后 CUDA synchronize，耗时为同步 wall time。未包含 DDP 通信、checkpoint 保存、冷启动和数据集初始化。

ESM embedding 已在磁盘缓存；不包含重新运行 ESM 模型。loader_wait 仅为预取后的前台等待，不等于 CPU 特征构建总耗时。infer_total 包含传输、4-cycle 编码器、distogram/softmax 与输出转换。正式训练相邻阶段可重叠，因此同步测量与实际吞吐存在差别。

## 耗时

| 阶段 | 计时单位 | 平均 ms | P50 ms | P95 ms |
|---|---|---:|---:|---:|
| loader_wait | 每条 | 21.072 | 21.012 | 22.722 |
| host_to_device | 每条 | 68.645 | 67.098 | 75.548 |
| encoder_4cycles | 每条 | 703.772 | 702.173 | 757.472 |
| infer_total | 每条 | 775.879 | 774.422 | 836.139 |
| head_forward | 每条 | 52.455 | 52.501 | 56.053 |
| loss | 每组3条 | 0.119 | 0.118 | 0.133 |
| head_backward | 每组3条 | 577.797 | 581.969 | 606.264 |
| optimizer | 每组3条 | 4.881 | 4.860 | 5.188 |

后续 head+loss+backward+optimizer 摊销 246.720 ms/条。每组完整平均 3.1359 秒。均值不可用于推断单条训练独立完成延迟：一组完成前向后才一起反向。

## 显存

下面是本测量进程的 PyTorch allocated，排除仍驻留的被暂停训练进程；不使用 nvidia-smi 的整卡总和。额外峰值为阶段 peak minus start；各阶段不能相加。总峰值包含模型/优化器/输入及前序样本保留的计算图。

| 阶段 | 最大实际总峰值 MiB | 最大阶段新增 MiB |
|---|---:|---:|
| host_to_device | 3178.84 | 244.87 |
| encoder_4cycles | 3920.09 | 636.74 |
| head_forward | 5441.53 | 2090.44 |
| head_backward | 7131.45 | 3084.56 |
| optimizer | 819.01 | 57.11 |

无前序同组 head 图时（每组第一条），编码器阶段总峰值 1576.81–1691.03 MiB。这是含 head/optimizer 常驻状态的测量，不是纯编码器模型权重大小。整个测量预留峰值 10096 MiB；预留是 allocator 缓存，不能等同实际活跃张量。

原始记录：[result.json](benchmarks/group_stage_profile_20261030/result.json)。脚本：`scripts/profile_group_stages.py`。
