# 编码器进程数、确定性与 GPU 利用率评估

## 正式训练状态（用户明确要求）

用户要求原实验暂停，优化完成后也必须等待用户命令才能恢复。已停止正式launcher PID196807，保留完整检查点：
`/NAS/luyq/foldabs_group_rank_20261029/lr_0.0001_bs_2_wd_0.01_drop_0.1_group_rank_k_2/resume_before_encoder_pool_sweep_20261031.pt`
checkpoint为epoch9、step21600、cursor1760。不要自动重启。所有当前实验均隔离，不改正式模型/optimizer文件。

## 已定位的数值差异

诊断脚本：`scripts/diagnose_encoder_determinism.py`。远程结果根目录 `/NAS/luyq/foldabs_parallel_eval_20261031/determinism`。

- a/b：相同样本，CPU输入字典逐张量SHA256一致；跨进程全部model.state_dict张量SHA256一致；编码器eval、requires_grad=false；seed42。即使固定序列化输入、同一进程连续推理，输出仍不一致。
- c：插桩逐层哈希。`aggregate_atom_to_token`输入原子张量逐位相同，聚合输出不同；之后input_embedder、4cycle输出均不同。此诊断运行在GPU1、无MPS，排除了多进程共享GPU和CUDA IPC作为必要原因。
- 已安装Protenix 1.0.0调用链：`model/modules/transformer.py:AtomAttentionEncoder.forward` → `model/utils.py:aggregate_atom_to_token` → `utils/scatter_utils.py:scatter_mean` → `scatter_sum` → `Tensor.scatter_add_`。这不是随机权重，也不是MC dropout；GPU原子加法累加顺序不可确定，且此路径为BF16输入/输出，舍入差异向后传播。
- d：实验性确定性替代（FP32分段mean，再转回原dtype）使同一输入重复推理的single/pair/probability/asym全部逐位一致，最大差0。
- e：独立新进程加载相同输入/权重、启用相同替代，与d全部逐位一致，最大差0。

替代实现：`src/foldabs_repro/tmp_prediction/encoder_numerics.py`，仅在实验脚本显式`--stable-aggregate`时安装；正式训练默认不变。输入必须是单复合物、按token连续排序的原子映射，否则显式报错。采用FP32累加是一项执行精度变化，不能把新的稳定输出宣称为与旧BF16非确定性输出逐位一致。

已完成5项测试：FP32/BF16分段均值及映射、dtype、非法/不连续/乱序token映射拒绝；远程pytest全部通过。

## 原聚合模式的进程数扫描

24组72样本、单分类头24次更新、关闭分类头重计算、每生产者最多一组未确认特征、MPS开启。

|编码器进程|总耗时秒|消费者等待秒|分类头调用秒|
|---:|---:|---:|---:|
|1|69.691|55.106|14.496|
|2|57.826|34.660|23.080|
|4|56.361|19.051|37.239|
|6|56.626|11.146|45.407|
|8|56.866|14.947|41.873|

1→2主要收益；4以上进入平台。减少消费者等待不能独立作为优化指标，因为分类头调用也在等待共享GPU调度/竞争；进程增多，初始队列填充和尾部排空开销更大。短实验不能证明绝对全局最优。

## 利用率口径

脚本`report_encoder_pool.py`分三个窗口：完整进程生命周期（含导入/加载/预热/退出）、正式计时、正式计时中间段（去掉首尾各N_encoder组）。每秒采样nvidia-smi，GPU-util是窗口内有kernel运行的时间比例，不等同于算力/带宽占用率；采样不能排除亚秒间隙。

原聚合4编码器：完整生命周期平均56.64%，最低0；正式训练平均99.875%，最低94，98.21%采样≥95%；中段平均99.97%，最低99。故不能声称整个实验生命周期一直95%以上。GPU启动阶段0%正常，正式训练段也不再呈现单编码器的频繁低值波动。

## 后续验证

`encoder_pool_stable_v1`：修复后1/4编码器关闭dropout数值对照，随后4/8/12编码器各48组144样本长一些的吞吐与完整遥测对照。只运行基准，不恢复正式训练。

## 修复后较长基准（已完成）

相同48组144样本，唯一分类头48次更新，分类头使用原cuEquivariance快速算子、关闭激活重计算，编码器启用确定性FP32分段聚合。

|编码器数|总秒数|条/秒|全段输入等待秒|分类头调用秒|稳态等待占比|整卡显存采样峰值GiB|
|---:|---:|---:|---:|---:|---:|---:|
|4|112.843|1.2761|35.289|77.415|27.64%|48.21|
|8|113.006|1.2743|15.010|96.711|0.114%|59.56|
|12|113.749|1.2659|22.451|91.179|0.103%|70.91|

稳态定义去掉首尾各N_encoder组，三个配置稳态组数不同，因此用等待占比比较，不直接比较累计等待。12进程全段等待较多主要反映初始供给填充；不能把全段等待全算成稳态供给不足。

结论：4~8已进入吞吐平台，当前实测最高是4编码器，但差异不足以断言唯一峰值；8/12可以基本消除稳态队列等待，却没有提高总吞吐。优先4编码器可节省显存与进程开销。此结论限于当前A100/样本长度/模型/有界调度，不能称所有配置的全局最优，也不能外推整个epoch无空闲。

|编码器数|完整进程平均GPU-util|正式计时平均|正式计时最低|正式计时≥95%采样占比|中段采样最低|
|---:|---:|---:|---:|---:|---:|
|4|75.51%|99.91%|95%|100%|100%|
|8|74.77%|99.94%|96%|100%|100%|
|12|74.31%|98.97%|13%|96.46%|100%|

完整进程含启动/导入/加载/预热/退出，三者都有0%采样。12进程正式段也有短时低谷。图见 `docs/benchmarks/parallel_eval_20261031/encoder_pool_stable_v1/gpu_timeline.png`，绿色阴影是正式计时区间。每秒采样不能排除亚秒空隙；GPU-util不是SM occupancy或Tensor Core峰值利用率。

## 分类头反向的独立非确定性

修复后1vs4生产者的首组全部3条样本、12个特征张量逐位一致；首个分类头前向logits完全相同。两边均使用cuEquivariance，并非cuEquivariance与Torch的对照。相同初始模型/optimizer、相同组顺序、dropout=0、stable_aggregate=True，均开启activation checkpoint，运行8组24样本、8次更新。跨8组前向记录的最大logit差为0.0625，最大loss差为0.014392；这些前向发生在对应组更新之前，不能表述为“第8次更新后同一固定输入的差”。

最大loss差发生在第6组（前面已完成5次更新）：单编码器0.3906073570，四编码器0.3762152791，相差0.0143920779，约为前者的3.68%。最大logit差发生在第8组，其loss分别为0.0032376586与0.0033663921（差0.0001287335）。原始记录位于encoder_pool_final_reports.tgz中的encoder_pool_stable_v1/{enc1_verify/result.json,enc4_verify/result.json,comparison.json}。本次仅完整保存首组特征用于逐位对照，不能据此宣称全部8组特征均已逐位验证。当前推荐的关闭重计算配置没有用这组数据重新测定累计差异。

进一步固定同一批特征、同一分类头权重、dropout=0、每次seed123，无任何optimizer.step，重复3次反向：
- cuEquivariance路径前向逐位相同，梯度相对L2差约0.0275%~0.0285%，cos>0.99999995；此实验仅一个进程、无MPS，故不是IPC或编码器进程数导致。
- 加torch.use_deterministic_algorithms(True)与CUBLAS_WORKSPACE_CONFIG=:4096:8，cuEquivariance路径仍有约0.03%的梯度差。该开关不能保证自定义CUDA算子确定性。
- 分类头Triangle算子改为Torch路径并启用上述确定性设置，在该固定输入测试中梯度逐位相同（差0）。这证明快速分类头路径是第二个非确定性来源；未将Torch路径默认用于正式训练，也未声称它与cuEquivariance反向数值完全相同。

脚本：`scripts/diagnose_head_determinism.py`。报告位于远程determinism/head、head_deterministic、head_torch_deterministic。严格端到端对照另在encoder_pool_strict_v1。

## 重计算为何增加耗时

分类头4个Pairformer block使用activation checkpoint时，首次前向少存中间激活，反向时再执行部分前向以重建梯度所需激活。因此额外执行计算；不是多更新一次参数。之前相同72样本4编码器对照：开启59.54秒、关闭56.57秒，开启约慢5.3%；整卡显存从约48.2GiB降到18.7GiB。显存够且追求吞吐时，当前优先关闭。

参考：
- https://docs.pytorch.org/docs/main/generated/torch.use_deterministic_algorithms.html
- https://pytorch-scatter.readthedocs.io/en/latest/functions/scatter.html
- https://docs.pytorch.org/docs/stable/checkpoint.html

## 严格端到端对照通过

encoder_pool_strict_v1：确定性编码器聚合＋Torch分类头Triangle算子＋torch.use_deterministic_algorithms(True)＋CUBLAS_WORKSPACE_CONFIG=:4096:8＋dropout0。1与4编码器相同8组24样本、相同初始权重和optimizer、按相同顺序完成8次更新：首组全部特征逐位相同，全部8组logits/loss最大差均0，梯度范数最大差0（见comparison.json）。这验证了多进程生产者/唯一消费者结构可以保持该对照下的一致性。

上表吞吐使用cuEquivariance快速分类头，严格对照使用Torch分类头；二者性能不能混用。严格模式的长时间性能与正式训练集成尚未评估，当前优化为隔离原型，正式默认未改。

最终状态：全部评估已结束，私有MPS已退出；正式训练保持停止，等待用户命令。结果归档 `docs/benchmarks/parallel_eval_20261031/encoder_pool_final_reports.tgz`。


## 全组特征与逐步参数追踪补测

为区分编码器输出差异与分类头参数分岔，使用新增`--trace-numerics`对1与4编码器分别执行相同8组24样本。两边均启用stable aggregate、BF16 autocast、cuEquivariance、activation checkpoint及MPS，关闭dropout，从相同checkpoint加载模型和optimizer。对每组全部输入特征、分类头更新前后state_dict、更新前optimizer张量状态及反向梯度计算SHA256。诊断有CPU拷贝与同步开销，不用于吞吐比较。

结果：全部8组、24样本、96个特征张量（single/pair/probability/asym）哈希均一致。首组head state、optimizer状态、logits和loss一致；首组反向梯度哈希不同，第一次更新后head state哈希不同。第2组起更新前head与optimizer状态不同，第2组logits/loss仍相同，第3组开始logits/loss不同。由此确认本次对照的分岔起点为分类头反向，后续输出差异伴随已经不同的分类头参数，不是编码器输出不同。结合此前相同固定特征/固定权重/不更新参数的重复反向诊断，支持cuEquivariance路径反向非确定性导致更新轨迹分岔；不声称定位到某个具体CUDA kernel。

本次最大logit差0.0625、最大loss差0.0021517575；历史运行最大loss差0.0143920779是另一轮观测，均非误差上界。第二组虽然参数已不同，输出仍可能因扰动尚未反映到输出或有限精度舍入而相同。结论限于已测配置和样本，不宣称所有输入与环境都已验证。

本地逐组结果：`docs/benchmarks/parallel_eval_20261031/pool_trace_v1/comparison.json`，完整原始结果为同目录`enc1/result.json`和`enc4/result.json`。远程目录`/NAS/luyq/foldabs_parallel_eval_20261031/pool_trace_v1`。隔离诊断结束，MPS退出，正式训练未恢复。
