# HiTPH 单机八卡：正例异步读取、负例在线生成执行方案

日期：2026-10-14。本文是执行设计，尚未实现在线训练或启动正式训练。无写盘推理与组合显存试测见 `docs/HITPH_ONLINE_BENCHMARK_20261015.md`；Tang2 独占 GPU 0/6/7 复测见 `docs/HITPH_ONLINE_BENCHMARK_TANG2_20261015.md`；CUDA 11 加速算子安装与速度/一致性试测见 `docs/HITPH_ONLINE_CUEQ_TANG2_20261015.md`。

## 1. 已核实的现状

- 训练集：79,465 正例、每 epoch 158,930 动态负例，共 238,395 条；永久缓存另有固定验证 11,853 条、固定测试 5,352 条。
- 固定协议：Protenix-v1、cycle 4、无 MSA、seed 101、bf16 推理，s/z/p_dist 以 fp16 量化；下游 4-block Pairformer、C pooling。
- `scripts/extract_protenix_embeddings.py` 已经只运行 trunk 和 distogram，不运行 diffusion。在线版应复用这条路径，不能靠“再去掉 diffusion”获得额外加速。
- 当前提取器每条把输出复制到 CPU、torch.save、每条 empty_cache；训练负例在线路径应去掉这些操作。
- `TMPFeatureDataset` 同步 torch.load 后立即将三个大张量转成 fp32；训练 DataLoader 默认没有后台 worker。
- `examples_from_manifest` 要求每条都有文件；`PeptideMismatchSampling.audit_features` 会完整加载缓存，启动和每轮都有调用，而且各 DDP rank 都会执行。直接沿用会产生重复的 TB 级读取，也会拒绝没有文件的在线负例。
- 当前验证由 rank 0 单卡执行，其余卡等待，须改成固定验证集分片评估。
- 本次 Tang3 快照：8×A40，每卡总显存约 46,068 MiB，空闲约 5,159–23,172 MiB；系统可用内存约 326 GiB，swap 已用满；/data 可用约 4.5 TB，根分区仅约 52 GB。以上为共享机器瞬时值，不是资源承诺，也未证明 /data 是 NVMe。

旧吞吐报告：`/data/luyq/hitph_cycle4_train_throughput_20261014/report.md`。Tang2 共享 A40 的 60 条样本训练，冷读 9.37 秒/条，暖读约 2.08 秒/条；后者含头训练和暖数据路径，不能视为独立 GPU 算子耗时。此前提取器“每条约 28 秒”含预处理、推理和保存，不能直接当在线推理耗时。

## 2. 推荐架构与上线次序

采用 8 个 DDP rank，每张卡常驻一份冻结的 Protenix 和一份可训练分类头。DDP 只包装分类头、只同步其梯度。每个 rank 只生成本 rank 要消费的负例，不跨卡传送完整 z/p_dist。

```text
rank 0 生成 epoch 样本计划：沿用现有负采样协议，广播小型元数据
                         ↓ 分片及工作量均衡
每个 rank / GPU：
  Foldtmp 正例 → 后台 CPU 读取/校验 → fp16 RAM 队列 → pinned staging → H2D ┐
  负例四链元数据 → 后台 CPU featurization → 冻结 Protenix → GPU 特征队列 ├→ 分类头 → backward
                                                                       ┘
  分类头梯度 → DDP all-reduce → optimizer
```

第一阶段交付“CPU/I/O 异步、GPU 串行调度”的可用版本：读取下一批正例和准备负例 CPU 输入与当前 GPU 计算重叠；GPU 内按需推理负例，输出直接训练，不等整轮缓存。该版本已满足负例不落盘、边生成边训练。

第二阶段才测试 GPU 双 stream：训练流消费当前特征、推理流生成下一条负例，GPU 队列深度从 1 到 2。事件同步保证就绪，record_stream/明确生命周期保证释放安全。两种模型会争抢同一 GPU 的算力和显存带宽，因此以端到端吞吐决定是否开启，不能假定两者耗时取 max。

首版不采用同卡独立推理进程与训练进程经 CUDA IPC 传大张量，以减少额外 CUDA context 和生命周期管理；也不预设 6 卡推理、2 卡训练。若同卡容量实测不足，再评估生产者/消费者卡组，并计入跨卡通信。若 I 为单条负例推理秒数、H 为单条头训练秒数，分组平衡约需 G_infer/G_train = 2I/(3H)。I=20、H=2.08 时接近 7:1，6:2 会使推理成为瓶颈；这些仅是算例。

## 3. 负例在线路径的必要约束

1. 复用现有 `sample_epoch`：seed 42 + epoch，同 split、同 HLA 候选池、全局已知阳性排除，固定评估集不变。1:2 是全局配额，不能改为机械地给每个正例生成两个负例。
2. 从该 epoch 已采样的负例集合取四链完整联合推理。不能从正例中抽出 TCR/HLA 特征再拼 peptide：z/p_dist 含全复合物交互。
3. 模型只在进程启动加载一次，eval + requires_grad_(False)，用 no_grad 做提取；不把 Protenix 参数加入优化器。无 diffusion，无 MSA，无模板。
4. 输出在 GPU 上 detach 并转为 fp16，模拟原缓存量化；进入头前沿用原训练的 fp32 输入转换与 bf16 autocast。先做等价验证，再另测直接半精度输入，避免同时改变训练数值。
5. no_grad 下产生的常规张量可供分类头反向使用；若改用 inference_mode，须在其作用域外 clone 为常规张量，否则可能无法为头保存反向所需输入。
6. 原提取器逐条重设 Python/NumPy/CPU/CUDA 全局 RNG。直接照搬会改变分类头 dropout 与恢复结果。第一阶段在无并发 GPU 计算时保存/恢复训练 RNG，预处理由独立 CPU worker 使用自己的样本种子。GPU 双 stream 启用前须消除共享 RNG 冲突，使用独立 generator/验证不使用随机数的提取路径；不能把全局 manual_seed 放进并发线程。
7. 原 Protenix dataloader 面向完整 JSON 和初始化预处理；需抽出单条/小块 featurization 适配器。不能每个负例新建 runner/dataloader、重复读全轮 JSON。若实际配置开启链级 ESM，复用本地链输入特征，不能每个负例重跑整套链编码。
8. 训练负例输出不写 Foldtmp、本地磁盘或 RAM disk；只保留可回收 GPU 张量。允许本地保存轻量采样计划、日志、checkpoint 和原有必要输入预处理资源。

## 4. 正例 I/O：先预取，再根据实测加周转盘

训练正例载荷每 epoch 约 5.353 TB；在线负例每轮省去约 10.714 TB 写入和后续同量读取。永久缓存保持原样。

初始参数建议（全部须可配置）：

| 项目 | 初始值 | 调整原则 |
|---|---:|---|
| 正例读取并发 | 每 rank 1 个 CPU reader，共 8 个 | 比较总并发 4/8/16，不无限增加 |
| 正例 RAM 预取 | 每 rank 16 条，另设总字节上限 8 GiB | 64 MiB/条只是典型值；按实际 bytes 限流 |
| pinned staging | 每 rank 1–2 条，总上限约 1 GiB | 不把整个 RAM 队列锁页 |
| GPU 负例队列 | 每 rank 1 条，之后试 2 条 | 两条典型输出约 128 MiB，不含模型/工作区/头激活 |
| CPU 负例特征队列 | 每 rank 2–4 条并设字节上限 | 单独测 featurization 工作集，避免几十 worker 抢内存 |
| 本地正例周转盘 | 默认关闭 | I/O 仍阻塞时再开启 256–512 GiB 上限 |

CPU 队列保留 fp16，完成 H2D 后才做头所需转换。读取时做身份、四链、配置、shape、有限值校验；通过校验的同一 payload 直接训练，不另读一次。开跑只读本地索引和少量抽样；缓存可用性不再每轮由 8 个 rank 全量扫描。索引中的配置要匹配本次 cycle4/no-MSA，不能调用硬编码要求 use_msa=True 的协议检查分支。

正例异步预取只能隐藏延迟，无法突破源端总带宽。仅训练正例的持续有效带宽门槛：24 小时/epoch 约 62 MB/s，12 小时约 124 MB/s，6 小时约 248 MB/s（十进制，未加验证）。在本次共享机器上重新测真实冷数据并发吞吐；不能用 20 个暖文件的速度代替。

若带宽充足、偶发长尾，RAM 队列通常足够；若希望隔离短期抖动，可将后续计划中的正例预取到 /data 的有界周转区，消费后回收。缓冲填充也来自 Foldtmp，同步计入首轮耗时，不会提升长期源带宽。当前 /data 容量不足容纳全训练正例，且需保留其他用户余量；不能承诺一次完整本地化。不得将大缓冲放 /tmp 或根分区。

固定 val 载荷约 795 GB、test 约 359 GB。每轮只做固定 val，最终最佳 checkpoint 再做 test；八卡分片，汇总预测计算全局 BCE/AUROC/AUPRC 和 macro-pMHC，不能平均各卡 AUROC。评估分片不重复填充，或汇总时明确去重。若以后有足够本地空间，优先考虑长期保留固定 val 的本地副本以减少每轮重复网络读取；此项不是启动前置条件。

## 5. 批次、显存与恢复

继续使用每卡物理 microbatch=1；不先实现可变长 padding 的大 batch。用梯度累积提供有效 batch。

建议首个在线实验设 accumulation=3、world_size=8，有效 batch 通常为 24。可按全局窗口调度 8 正+16 负，每个 rank 一正两负，平衡推理负载；从已经完成合法配额分配的正负池独立取样，避免误改为逐正例两负。该调度保持样本集合、全局比例，改变原随机顺序，须在 run_config 记录。也可先沿用原顺序作等价基线，再测试窗口调度。

79,465 不能被 8 整除。最后窗口采用 masked 占位，让各 rank 参与相同次数的同步；按全局实际有效样本数归一化，并补偿 DDP 的梯度平均，不能将补齐样本当真实负例。常规中间窗口亦按实际计数记录损失。

显存预算是“冻结模型常驻 + 训练头/梯度/Adam 状态 + 头激活 + 推理工作区 + 特征队列 + NCCL/allocator”，不能把此前单独提取峰值 3.32 GB 当作组合训练峰值。先测最长样本，必要时为头的 Pairformer block 启用激活重计算；队列满立即背压。第一阶段 GPU 串行调度降低同时活跃的工作区，第二阶段才测试并发峰值。

本次八卡空闲量不足以确认可运行。执行切换时先记录并停掉本项目旧负例写盘 workers、监控器及自动重启 coordinator，释放其资源；保留已有文件，不删除永久缓存，不影响其他用户进程。之后按实测峰值加合理余量判断能否在八卡同时运行。

checkpoint 保存头、optimizer、scheduler、混合精度状态（如使用）、每 rank 训练 RNG、epoch、采样配置/计划标识和已消费位置。只在完成 optimizer step 后记录消费位置；队列中未消费项不计进度，恢复时重生成。首版至少支持完整 epoch 恢复；正式长跑前应补上 optimizer-step 边界恢复，避免故障重算整轮。任意 rank 出错须全组退出并保留最近 checkpoint，不能静默跳过样本或无限等待 DDP。

## 6. 吞吐估算与验收

八卡混合方案、GPU 串行基线的乐观计算时间：

T_compute ≈ (158930 × I + 238395 × H) / 8

I 是不含写盘的负例 GPU 推理均值，H 是头的训练时间。以下暂用旧暖数据端到端 2.08 秒/条代作 H，仅展示量级，不是新方案实测预测；未计读取阻塞、CPU 预处理、DDP、负载差异、验证和 checkpoint。

| 假设 I | 计算阶段估算 |
|---:|---:|
| 5 秒/负例 | 44.8 小时 |
| 10 秒/负例 | 72.4 小时 |
| 20 秒/负例 | 127.6 小时 |

异步方案消除了整轮缓存完成前的等待，训练可在小窗口就绪后开始；它不保证 epoch 从几天变为几小时。想在 24 小时完成训练，暂按 H=2.08，I 需约 ≤1.23 秒且 I/O/其他开销可被隐藏；超过此范围就须进一步优化模型计算，或在新的实验协议中降低 cycles/负例量等。不能暗中降低 cycles 或改负例定义来报告提速。

验收顺序：

1. **单卡分项**：从本地输入抽取 64–128 个真实负例，分别计 CPU featurization、CUDA Event 推理、fp16 输出和头 forward/backward；不写负例文件。单独测新正例冷读分布，明确已读数据比例。
2. **一致性**：抽样将 GPU fp16 输出与原离线提取输出比较，确认身份/四链/config/shape、p_dist 概率、数值误差；同样头权重、输入和训练 RNG 下核对 logit/loss/梯度。与标准离线提取的对照在本地小样本上完成，不向 Foldtmp 新写负例。
3. **容量**：测最长龙、串行和双流各自峰值、队列满载内存、反向是否成功。确认 Protenix 无梯度且头参数确实更新。
4. **八卡管线**：运行至少 500 个全局有效 batch（约 12,000 样本，保持 1:2），使用实际计划及新的正例；分别报告排除/包含启动的吞吐、正例等待、负例等待、NCCL 等待、显存、RAM、远端实际读取字节。禁用双流/启用双流对比，吞吐和容量胜出的配置进入正式运行。
5. **完整数据语义与故障恢复**：核对全轮计划 79,465/158,930，无重复真实负例/跨 split 泄漏；验证末窗口损失权重；做一步边界中断恢复；固定评估多卡结果与单卡一致。
6. 用实际端到端 samples/s 外推 238,395 条，另外加完整 val 时间。若八卡测试仍显示每 epoch 多天，应直接报告实测结果并定位模型/源带宽瓶颈，不将离线写盘时间当作可全部省掉的 GPU 时间。

## 7. 实现落点

| 文件 | 变更 |
|---|---|
| 新增 `src/foldabs_repro/tmp_prediction/online_features.py` | 常驻冻结 runner，单条 CPU 输入适配，GPU fp16 输出，RNG 隔离 |
| `src/foldabs_repro/tmp_prediction/dataset.py` | 区分离线正例和在线负例描述符；仅离线项检查路径；CPU fp16 延迟转换 |
| 新增 `src/foldabs_repro/tmp_prediction/feature_prefetch.py` | CPU reader、有界 RAM/pinned 队列、GPU 就绪事件、背压与异常传播 |
| `scripts/train_tmp_specificity.py` | 在线 provider、一次采样分片、梯度累积/末尾掩码、移除重复全量 audit、分布式 val 和恢复 |
| `src/foldabs_repro/tmp_prediction/model.py` | 可选逐 block 激活重计算，保持模型结构与 pooling |
| 新增 `scripts/benchmark_hitph_online.py` | 无负例落盘的单卡/八卡分项与端到端基准 |
| `scripts/run_hitph_cycle4_epochs.py` | 新增在线模式，跳过负例提取/等待/删除轮换；保留离线模式兼容 |
| 新增独立在线配置 | 保存 provider、buffer、调度、采样和提取精度；不覆盖当前活跃提取配置 |

启动接口设计：在线功能实现后，经 `torchrun --standalone --nproc_per_node=8` 启动训练入口，选择 online negative provider、cycle4/no-MSA、bf16、4 blocks、C pooling、microbatch 1、accumulation 3。当前脚本尚无这些在线参数，本文不提供可误认为立即可执行的正式训练命令。

PyTorch 2.7 的行为依据：[CUDA streams、异步拷贝与生命周期](https://docs.pytorch.org/docs/2.7/notes/cuda.html)、[no_grad 与 inference_mode 的区别](https://docs.pytorch.org/docs/2.7/notes/autograd.html)。
