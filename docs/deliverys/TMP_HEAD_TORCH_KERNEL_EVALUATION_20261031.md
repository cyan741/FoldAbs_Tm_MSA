# 冻结cuEquivariance编码器 + Torch分类头评估

用户授权仅切换分类头Triangle Multiplication和Triangle Attention为Torch，保持每卡4个冻结Protenix编码器+1个分类头，双卡梯度同步。正式训练不得恢复。

隔离基准：scripts/run_encoder_pool_kernel_evaluation.py；根目录 /NAS/luyq/foldabs_parallel_eval_20261031/head_kernel_v1。

固定条件：4cycles编码器保持cuEquivariance/BF16、稳定FP32分段聚合、MPS开启、重计算关闭、分类头BF16 autocast、dropout=0.1、相同checkpoint/optimizer/样本顺序及seed。分类头通过独立--head-kernel参数选择；没有附加--deterministic-head、没有额外设置CUBLAS_WORKSPACE_CONFIG，以检验仅换kernel是否足够。

依次执行单卡cuEq、单卡Torch、双卡cuEq、双卡Torch及独立重复双卡Torch，每次48组144样本。单卡48次更新、全局batch3；双卡24次更新、全局batch6。速度只在相同卡数与batch下比较，不能将跨卡吞吐比称为相同训练轨迹的加速。双卡Torch两次重复比较每组logits/loss/gradient norm与完整最终head参数SHA256。1秒nvidia-smi记录整卡显存与GPU-util。

正式checkpoint只读，基准使用历史step21400隔离对照；正式恢复点step21600不变。


## 通信初始化问题与复测

v1单卡Torch在最长样本预热完成后的首次NCCL all_reduce处失败，报错Failed to CUDA calloc async 608 bytes，整卡采样峰值80933 MiB。将NCCL communicator初始化提前到加载模型之前，对两种kernel都采用同样的初始化顺序，不改变训练参数、进程数、样本、重计算或算子。该初始化不计入训练段。

v2输出根目录`/NAS/luyq/foldabs_parallel_eval_20261031/head_kernel_v2`。单卡cuEq 112.7337秒/144条，Torch 134.1484秒/144条，均完成48次更新。Torch仅切kernel、关闭重计算可以完成该基准；其显存接近80GB容量，仍需根据峰值明确余量，不能把启动错误当作已经证明数学计算不支持。


## 最终结果（2026-11-01）

|配置|样本数|更新次数|训练秒数|条/秒|整卡显存采样峰值|
|---|---:|---:|---:|---:|---:|
|单卡4encoder + cuEq分类头|144|48|112.734|1.27735|49693 MiB（48.53 GiB）|
|单卡4encoder + Torch分类头|144|48|134.148|1.07344|80847 MiB（78.95 GiB）|
|单卡Torch独立重复|144|48|134.116|1.07370|未在本表重复统计|
|双卡每卡4encoder + cuEq分类头|144|24|57.014|2.52571|每卡50235 MiB（49.06 GiB）|
|双卡每卡4encoder + Torch分类头|目标144|完成23/24|未完成|不能给出有效全程吞吐|GPU0/1：80795/80799 MiB；随后OOM|

单卡Torch相对cuEq耗时增加18.996%，吞吐降低15.963%，显存峰值增加31154 MiB（30.42 GiB，约62.69%）。峰值来自1秒nvidia-smi采样，包含预热；单卡Torch正式训练段峰值80841 MiB。单卡与双卡global batch及更新次数不同，不能直接解释为同一优化轨迹的加速比。

单卡Torch两次独立运行：相同seed，dropout保留0.1，关闭重计算，BF16 autocast及MPS均保留。48组全部logit/loss/梯度范数最大差为0，最终完整分类头state_dict SHA256相同：`355e0c9644b4964077a14c3ae8c52face62f61f60f2c682d5dbf998a2134c89b`。没有附加torch.use_deterministic_algorithms或CUBLAS_WORKSPACE_CONFIG设置。本次验证支持该环境与样本下仅切换Torch即取得重复一致，不是跨环境、任意输入的普遍保证。原始报告保留于head_kernel_v2/gpu1_torch_r0及gpu1_torch_r1（重复的远程来源为head_kernel_v2_repeat）。

双卡Torch第24次更新loss.backward()发生OOM。rank0申请586 MiB时仅余456.12 MiB，PyTorch allocated62.84 GiB、reserved但未使用3.21 GiB；rank1申请612 MiB时仅余360.12 MiB，allocated65.07 GiB、reserved但未使用1.08 GiB。每卡4个编码器各约2.84 GiB。显存碎片/缓存与实际激活压力共同构成容量风险，不能声称绝对最小内存必然超过容量，但当前配置实测失败。并行增加卡数没有减小每卡分类头处理的组大小，因此不会自动缓解这项显存压力。

双卡Torch完整确定性验证未通过执行前提，重复任务按失败跳过，不能把单卡通过当成双卡通过。没有擅自改变重计算、batch、编码器数、allocator策略来掩盖失败。若要保持每卡4encoder和Torch分类头，下一项建议单独评估分类头激活重计算；这是额外配置变化，本轮未执行。

结果与完整失败栈：`docs/benchmarks/parallel_eval_20261031/head_kernel_v2/`，关键文件kernel_comparison.json、summary_detailed.json、dual_torch_failure.json。新增--head-kernel使分类头选择独立于确定性全局开关。正式训练默认及checkpoint未改，未恢复；全部隔离进程与私有MPS退出，两卡均0 MiB。
