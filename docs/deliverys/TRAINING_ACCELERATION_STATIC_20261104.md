# 训练加速：静态分析与最小实现

日期：2026-11-04。基线：`tmp-specificity@2536d49dd0231bfcb06aabcb17ffd0e18125c3a9`。开发分支：`feat/accelerate`。

**结论：存在可解释的优化空间，优先减少冻结编码器的重复工作和供给开销。直接换成 Megatron、SGLang 或 ms-swift 不适合当前架构。此次只做静态分析和局部代码修改；按照用户要求，没有运行项目代码、导入模型、训练、推理、编译检查、测试或性能基准，也没有新增依赖或复制第三方实现。所有改动均未经运行验证。**

## 1. 当前真正执行的任务

实际主线为 TCR–pMHC 特异性预测：

```text
固定 split + 每 epoch 动态肽错配负例
  → 复用 ESM2-3B 单链缓存
  → 冻结 Protenix Mini-ESM，4 cycles，无 MSA/template
  → 联合 s[N,384]、z[N,N,128]、p_dist[N,N,64]
  → 可训练 PairCondition + 4-block Pairformer
  → pooling C + MLP
  → BCE 或 Group ranking
```

ESM 单链缓存、BF16、cuEquivariance、冻结 encoder、DDP、梯度累积和 `no_sync` 已存在，不能把它们算成本次新优化。提取器停止于 trunk/distogram，不执行 diffusion/confidence；优化结构生成阶段不会直接加速本任务。

**文档与代码有差异：**本基线 README 引用了 `scripts/group_encoder_pool.py`、`scripts/launch_group_pool_training.py`、`--encoders-per-gpu` 和 `--head-kernel`，但指定提交没有这两个文件，统一训练入口也没有这两个参数。历史 `benchmark_encoder_pool*` 脚本、报告和原始结果存在；不能据此认定编码器池已接入当前训练路径。下文将其列为待集成设计。

## 2. 先检索的开源实现

详细依赖、许可证、维护时间和已知问题见 [开源项目比较](../research/ACCELERATION_FRAMEWORKS_20261104.md)；精确提交和检索证据见 [来源记录](../research/acceleration_sources_20261104.json)。维护时间按默认分支提交判断，不用 stars 或仓库页面更新时间代替。

| 候选 | 核对结果 | 对本项目的结论 |
|---|---|---|
| Protenix | 代码 Apache-2.0，默认分支最近提交 2026-09-21；v2 权重存在专有例外 | 直接上游。优先沿用兼容的 kernel/chunk 接口，不能整体升级依赖后假定数值不变 |
| DeepSpeed / DS4Sci | Apache-2.0，2026-11-04 有提交 | 是蛋白模型专用 fused attention 的直接先例，但张量轴、bias、长度和 backward head dimension 有限制；当前不新增 backend |
| cuEquivariance | 仓库声明 Apache-2.0，最近默认分支提交 2026-10-30；外部 wheel 条款需单独核对 | 当前已使用。先查版本、形状限制和 fallback，不能把再次安装它算作收益 |
| OpenFold | Apache-2.0；最近默认分支提交 2025-12-16 | 成熟设计参考，近期活动不足。已合并的 #560 修复 backend fallback/scaling，值得借鉴 |
| FastFold | Apache-2.0；最近默认分支提交 2023-07-11 | 轴向并行和通信重叠是明确先例，但维护和依赖陈旧，仅作历史参考 |
| Boltz | 代码/权重 MIT；最近默认分支提交 2026-05-29 | 可参考可选 kernel 组织，不能视为可直接替换的当前训练框架 |
| Megatron-LM | 活跃；主许可与第三方许可组合 | 借鉴并行、融合、选择性重计算。冻结 encoder 没有待分片的梯度/优化器状态，完整迁移缺少依据 |
| SGLang | 活跃，Apache-2.0 | 借鉴有界队列与生产/消费调度。其因果前缀 KV 缓存不适用于随 peptide 改变的双向联合表示 |
| ms-swift | 活跃，Apache-2.0 | 将用户所写的 swtft 理解为 ms-swift。可参考配置管理；其 Transformers/PEFT/TRL 适配栈不是 Pairformer 的即插即用 trainer |

Protenix 上游已有 SDPA，并在预缩放 query 后使用 `scale=1.0`。盲目改成默认 SDPA scale 会重复缩放。DS4Sci 也不能在未核对 pair bias 和 backward 约束的情况下直接替换。此次只借鉴设计，未拷贝这些项目的代码。

## 3. 瓶颈证据的强弱

仓库 [2026-10-30 分阶段报告](GROUP_STAGE_PROFILE_20261030.md) 使用 A100 80GB、405–431 tokens、2 组预热、11 组测量、每组 3 条、开启分类头重计算，报告：

| 阶段 | 旧报告均值 |
|---|---:|
| loader 前台等待 | 21.072 ms/条 |
| H2D | 68.645 ms/条 |
| 冻结 encoder，4 cycles | 703.772 ms/条 |
| head 前向 | 52.455 ms/条 |
| head 反向 | 577.797 ms/组 |
| optimizer | 4.881 ms/组 |
| 完整组 | 3.1359 s/组 |

这提示 encoder 是重点，而非 optimizer 状态分片。它排除了 DDP、冷启动、数据初始化及 checkpoint，采用同步分阶段计时，且重计算配置与当前默认不同。**这是仓库历史证据，不是本次测量，不能外推当前吞吐或本轮加速百分比。**

可从代码直接确认的工作包括：逐样本 GPU 标量回读、每个 chunk 重建 loader/预处理、重复 ESM 校验传输、offline 每样本 `empty_cache()`。能证明操作存在，不等于能确定它们在当前硬件上的耗时占比。

## 4. 本次已修改的内容

| 路径 | 修改 | 静态预期与边界 |
|---|---|---|
| 在线 BCE / Group 共用 `OnlineFeatures.infer` | 在 CPU 上读取 `N_token`，再 `to_device` | 避免为一个 Python 配置整数先 H2D 再 D2H；不改变 encoder 配置值 |
| 统一离线提取器 | 同样提前读取 `N_token`；Mini-ESM 精确一致性校验在 CPU 完成 | 保留 shape、FP32 转换、精确 equality 和链缓存有限值/身份检查；少传输一份完整校验 embedding，不减少磁盘读取次数 |
| 离线单卡训练/原模型评估 | 保留 CPU `token_asym_id` 传给 head | head 在 CPU/Python 解析 chain IDs，再将 masks 移至 `s.device`；省去元数据 CPU→GPU→CPU 往返。后续 GPU boolean gather 仍存在；DDP 自动搬运的限制见下文 |
| 统一离线提取器 | 增加 `--reuse-cuda-cache` | 可选跳过逐样本 `empty_cache()`，允许 allocator 复用空闲 blocks；默认关闭，保留原来的共享 GPU 行为 |

静态核对了 Protenix `v1.0.0` 的 `to_device`：递归处理字典并调用 `tensor.to(device)`，不改变 dtype。因此校验移动到 CPU 的依据是只改变校验执行设备，保留原判定逻辑；安装其他上游版本仍需核对该契约。临时校验张量在输入传输前删除。

新选项直接用于 `scripts/extract_protenix_embeddings.py` 的原有命令，追加 `--reuse-cuda-cache` 即可；不适用于在线训练 CLI。多 GPU 提取以 `sys.argv[1:]` 传给 worker，因此该参数随命令传递。显式拼装参数的外层 wrapper 保留旧默认，不自动获得新开关。开关不进入特征 `run_spec`，以免将分配器策略误当作科学特征身份、破坏旧缓存匹配。

该选项可能提高 **reserved VRAM**，不会降低活跃张量所需的 **allocated VRAM**，也不是 OOM 修复。当前 `generated_seconds` 在 `empty_cache()` 之前结束单样本计时，无法衡量此开关的全部影响；将来比较应使用包含释放步骤的整体 wall time，并同时观察 allocated/reserved。

模型结构、PairCondition、pooling 参数、精度、cycles、种子调用、样本顺序、负例数量、loss 权重、优化器、DDP 累积和 checkpoint schema 均未改动。未声明输出/梯度位级一致，亦未声明性能提升已验证。

独立静态审查发现：离线训练的 DDP 使用 `device_ids=[rank]`，PyTorch DDP 的前向包装仍会将 CPU tensor 参数移至该 GPU。因此保留 CPU chain IDs 的收益边界是原模型单卡训练及未包 DDP 的评估；**本次没有消除 DDP 训练中的 chain-ID 同步**，也没有为此扩大修改 DDP 设备配置。

## 5. 更值得投入的后续优化

### 优先级一：有界复用冻结联合特征

**机制：**固定验证/测试样本、每 epoch 重复的正例 anchor，在 encoder 及随机过程固定时，可以复用 encoder 输出，跳过完整 4-cycle 前向。缓存必须位于可训练 PairCondition 之前；即便 head 为 P0，PairCondition 也仍可训练，不能缓存其输出。

缓存键应包含四条有序序列/身份、encoder 权重指纹、模型与特征配置、cycles、seed/RNG 规则、kernel/软件版本、计算精度和存储精度。标签不参与 encoder 计算；sample ID 别名与预测输出顺序仍须保留。只有完全匹配的联合复合物可复用；更换 peptide 后，TCR/HLA 的联合表示也会改变，不能套用 LLM 的前缀 KV 复用。

**资源策略：**先固定评估样本，再在容量允许时缓存重复正例；动态负例只做受预算限制的精确命中缓存。按字节而非仅按条数限制 RAM/VRAM/磁盘，并使用校验、原子写入和淘汰机制，不能把所有 `[N,N,C]` 特征堆入内存。避免为了命中率固定训练负例或改变排序。

设单次编码耗时 `E`，缓存命中后的读取/传输耗时 `R`，一次写入成本 `W`，重复调用次数 `k`。只有 `(k-1) * (E-R) > W` 时才节省时间；远程存储的 `R` 可能抵消收益。FP16 缓存会增加舍入，而在线路径目前刻意避免 FP16 往返；行为保持方案须保留实际 dtype，压缩精度须另作数值消融。

**状态：未实现。**没有数据、容量和数值验证条件，不选择无限缓存、默认压缩或改动数据协议。

### 优先级二：持久化、可恢复的数据供给

`OnlineFeatures.batches` 每次构建记录、检查链元数据、写 JSON、调用上游 preprocess 并创建 DataLoader。训练默认每 96 个 rank round / group 调用一次并销毁 iterator。两个 worker 的 `spawn` 生命周期也随 chunk 反复开始；只设置 `persistent_workers=True` 不能解决外层 loader 被反复销毁。

可借鉴成熟训练框架：每 epoch 准备不可变 rank manifest，以保存的 cursor 驱动 sampler；或采用常驻 CPU worker 服务，队列携带明确任务 ID，有界预取。保留 per-sample RNG、相同 rank 分片、group occurrence ID、DDP 尾部处理以及恢复时不漏/不重复样本。chunk 前台等待不包含全部预处理成本，历史报告不足以给出该项收益。

**状态：未实现。**这涉及恢复和多进程生命周期，静态小改动不足以安全覆盖全部语义。

### 优先级三：冻结 encoder 与训练 head 的有界并行

当前主线逐样本依次执行 encoder 和 head。借鉴 SGLang 的生产/消费调度思路，以及仓库已有独立 encoder-pool 实验，可让冻结编码器提前准备后续样本，用任务 ID、事件/IPC 生命周期和背压控制交接。

并发占用至少包括 encoder 权重副本、队列中的 `s/z/p_dist`、当前 head 的 autograd 图、优化器和临时 workspace。Group loss 需要一组 logits 后统一 backward，保留 `(K+1)` 个 head 图是当前损失实现的一部分，不能作为内存泄漏直接删除。当前历史池报告不能直接作为可用训练实现；MPS、进程失败、退出清理、CUDA IPC 所有权和确定性均需独立验收。

**状态：设计参考，不新增并行框架依赖。**缺少 GPU 时不选择 encoder 数量或给出显存预算数值。

### 优先级四：指标导出与有序索引池化

- BCE 逐样本 `loss.item()`，Group 每组导出 loss/gap/accuracy 三个 GPU 标量。可在设备累积 detached 小张量，到日志/保存/epoch 边界统一导出；保留 Python-double 累积语义、原 checkpoint `running_by_rank` 和恢复值，避免保存 autograd 图。
- pooling C 仍有四次实体和五次界面 boolean gather。可在 CPU 准备升序 role indices，实体用 `index_select`，pair 用有序笛卡尔整数索引，避免 `[N,N]` boolean mask 与动态输出大小处理。必须保留原行优先顺序、方向、query、FP32 scoring/softmax 以及梯度；不要先把非连续 z 全量 reshape，产生隐式大拷贝。

**状态：未实现。**前者关联恢复与统计舍入，后者改变 gather/backward 路径；在用户禁止执行的条件下只提出设计，不声称已验证等价。

## 6. 科学比较与资料边界

本次不会把减少 cycles/head blocks、缩短 TCR/HLA、改变 pooling、减少负例、量化/蒸馏或替换 joint encoder 当作等价工程优化。这些可能有效，但会改变模型或训练问题，需要独立消融。验证/测试划分与 IMMREP25 的逐 pHLA 宏平均标准化 pAUC（`max_fpr=0.1`）应保持，不能用外部测试标签选加速配置。

用户提供的资料中，motivation、IMMREP25 说明、张量流和简要输入说明已读取；ablation 文档返回 `3380004`，无查看权限。张量流和输入说明的内嵌表格未单独读取。本报告用较新的“停止 MSA、采用 pooling C”说明与实际主线相互核对，不臆造未知的 ablation 结论。旧文档约 64.4 MiB/样本、全正例 5.22 TiB 是另一提取设置的样本统计/外推，不能当作当前 Mini-ESM 的完整缓存实测。

## 7. 验证状态与以后允许执行时的验收

本次已完成：源码调用链阅读、上游许可证/依赖/issue 检索、Git diff 静态审阅。未运行任何项目测试或模型代码，未测吞吐、显存、loss/梯度一致性、收敛或最终评价指标。

以后具备条件且用户允许执行时，最小验收应包括：

1. 相同输入、种子、权重与执行配置下比较原版/新版冻结特征、head logits/loss/梯度；CPU 元数据路线覆盖 A/B/C、非连续 chain IDs 和缺失角色错误。
2. 故意不匹配 ESM shape/内容/身份、非法缓存值时仍拒绝；有效缓存恢复仍复用原文件。
3. BCE 和 Group 单/多卡的短程恢复，检查 cursor、样本计数、负例身份、DDP 权重与 checkpoint 兼容。
4. allocator 开关分别记录含预热和冷启动的 wall time、samples/s、allocated/reserved 峰值；不以 kernel microbenchmark 或排除开销的 `generated_seconds` 代替整体收益。

这些是后续验收条件，并未在本次执行。
