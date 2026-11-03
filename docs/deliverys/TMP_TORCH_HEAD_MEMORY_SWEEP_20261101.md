# Torch分类头显存与吞吐对照（2026-11-01）

用户确认按A→B→C→D→E顺序执行。正式训练继续暂停，禁止自动恢复。

运行脚本：scripts/run_torch_head_memory_evaluation.py。
远程结果：/NAS/luyq/foldabs_parallel_eval_20261031/torch_memory_v1。

固定冻结Protenix encoder的cuEquivariance、4cycles、BF16、稳定FP32分段聚合、分类头Torch/BF16/dropout0.1、MPS、checkpoint/optimizer/seed、每rank每组3样本、双卡NCCL梯度平均。仅改变每卡编码器数与分类头激活重计算。NCCL在模型加载前初始化，与head_kernel_v2对照一致。

A：双卡每卡4encoder，分类头重计算开。
B：双卡每卡3encoder，分类头重计算关。
C：双卡每卡2encoder，分类头重计算关。
D：单卡4encoder，分类头重计算开。
A-D各48组144样本；双卡24次更新/global batch6，单卡48次更新/global batch3。
E：从A/B/C已完成配置中选择实测吞吐最高者，独立重复相同48组，比较两rank的逐组logits/loss/gradient norm及完整最终参数SHA256。
F：E逐位一致后，同配置运行96组288样本、48次共同更新，检查长段吞吐和显存。

速度最优仅指本次有限基准，不宣称全局最优；最终还需查看显存余量。1秒nvidia-smi采样记录物理显存与GPU-util；时间排除加载/预热，峰值同时查看全过程。旧结果保留，不重复执行已完成基线。正式恢复点step21600不变，基准只读step21400隔离checkpoint。


## 最终结果

全部A-F成功完成，正式训练未恢复。时间仅统计预热后训练段；显存为整卡所有进程的1秒采样峰值，包含加载/预热。

|实验|GPU数|每卡编码器|分类头重计算|样本数|共同更新次数|秒数|条/秒|每卡峰值GiB|
|---|---:|---:|---|---:|---:|---:|---:|---|
|A_dual_enc4_cp|2|4|开|144|24|76.364|1.8857|23.28/23.29|
|B_dual_enc3|2|3|关|144|24|68.205|2.1113|77.05/77.05|
|C_dual_enc2|2|2|关|144|24|69.423|2.0742|74.21/74.21|
|D_single_enc4_cp|1|4|开|144|48|151.219|0.9523|22.76|
|E_repeat_B_dual_enc3|2|3|关|144|24|67.897|2.1208|77.05/77.05|
|F_long_B_dual_enc3|2|3|关|288|48|135.298|2.1286|77.25/77.25|

已有对照：单卡4encoder+Torch无重计算134.148秒、1.07344条/秒、峰值78.95GiB；双卡每卡4encoder+Torch无重计算第24次更新OOM。双卡每卡4encoder+cuEq分类头57.014秒、2.52571条/秒、49.06GiB/卡。只有同卡数、相同global batch和更新次数的对照用于速度变化结论。

### 重计算的代价与收益

单卡D对已有无重计算Torch配置：时间增加12.725%，整卡峰值80847→23303MiB，下降71.176%（约78.95→22.76GiB）。这是真正仅开关分类头重计算的配对比较。双卡A对B同时改变编码器数量和重计算，不能把两者全部时间差归因于重计算。

### 2与3编码器的吞吐和等待

B（3encoder）68.205秒，C（2encoder）69.423秒，C耗时增加1.786%，但峰值78898→75993MiB，减少2905MiB（2.84GiB）。rank0等待/分类头阶段时间分别为B 21.391/46.757秒，C 32.083/37.284秒。减少encoder使输入等待增加，但分类头阶段变快，抵消了大部分损失。分类头阶段包含前向、反向、梯度同步和optimizer，不是纯算子时间，也可能包含等待另一rank；不能将这组数据宣称为严格的GPU竞争因果分解。

B比A吞吐高11.962%，但A有明显更大的显存余量。B与C仅差约1.8%，只做了有限运行，不能宣称3编码器是所有样本/环境的唯一最优。速度优先可选B；需要更多显存余量可选C；显存压力明显时选择A。

### 确定性

E重复B（相同seed，dropout保留0.1，没有额外确定性开关）：两个rank各24次更新，逐组logit、loss、梯度范数最大差均0，最终完整head state_dict SHA256一致，且同一次运行的两个副本相同。哈希为625599fffd265f3f5f2e8b5a34b0b58bf25946daf54fbd635982ca193fe7dfde。

额外比较已有记录，不增加实验：A与B、B与C的全部逐组logit/loss/梯度范数最大差均0，最终参数相同。D与历史单卡无重计算Torch亦全部差0，最终参数哈希355e0c9644b4964077a14c3ae8c52face62f61f60f2c682d5dbf998a2134c89b。证据支持在当前环境和样本下改变encoder进程数、开关重计算均未改变训练轨迹；不是跨版本/硬件/任意输入的普遍保证。

### 长测

F使用B配置完成96组288样本、48次共同更新：135.298秒、2.12864条/秒；每卡峰值79108MiB（77.25GiB），相对该GPU实际可用约79.25GiB，剩余约2GiB。没有OOM，两rank最终参数一致，所有encoder无梯度。计时段两卡GPU-util平均99.53%/99.67%（1秒采样）；该指标不代表Tensor Core算力利用率，也不能排除亚秒空隙。

### 交付与状态

新增自动执行脚本scripts/run_torch_head_memory_evaluation.py，完成A-D、选择最快完成的双卡方案、E确定性检查、通过后F长测，并在结束退出私有MPS。没有更改正式训练默认配置，也没有启动或恢复正式训练。评估结束后两卡显存均0MiB。

本地结果目录docs/benchmarks/parallel_eval_20261031/torch_memory_v1：summary_detailed.json、determinism.json、cross_configuration_numerics.json、tradeoffs.json，以及各rank原始result.json和日志、GPU遥测。归档torch_memory_v1.tgz仅包含结果与日志，排除encoder特征缓存链接及MPS管道。脚本语法检查通过，远程六项实际实验均退出0；最终核对全部一致性断言通过。
