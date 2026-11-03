# mini-ESM 加速提取、Foldtmp 与在线训练实测（2026-10-17）

> 状态更新（2026-10-18）：下文记录的是 10 月 17 日实验当时的状态。用户随后要求停止写盘，离线提取已经停止，state.json 为 stopped_by_user，完整验证 2,304 条；不再继续下文所述的恢复提取。最新在线训练加速结果见 [训练加速实验](HITPH_MINI_ESM_TRAINING_ACCELERATION_20261018.md)。

本次对象是 **protenix_mini_esm_v0.5.0**，不是普通 Protenix-v1。保留 mini 实验的 seed 42、cycles 4、无 MSA/template、冻结 encoder，不执行 diffusion/confidence。参考普通版本现状文档 `HITPH_PROTENIX_EXPERIMENT_STATUS_20261016.md` 的 cuEquivariance CUDA 11 加速配置，但下面的数值全部来自本次 mini 实测。

## 执行与精度

mini 提取入口 `scripts/extract_protenix_embeddings.py` 已支持并默认启用 `cuequivariance/cuequivariance`；普通版本入口 `scripts/extract_protenix_embeddings.py` 和在线基准入口也改为默认加速。缺少依赖时直接报错，不自动退回 torch。Tang3 使用 `/data/luyq/hitph_cycle4_worker/accel/enable_cueq_cu11.sh` 的隔离 cuEquivariance 0.5.1 环境。

| 数据/阶段 | 精度 |
|---|---|
| ESM2-3B 单链推理 | BF16 autocast |
| ESM 第 36 层逐残基缓存 | FP32，2560 通道；BF16 结果转为 FP32 存储不恢复 FP32 计算精度 |
| mini-ESM Protenix trunk | BF16 autocast，敏感 softmax 等保留 FP32 |
| 联合 s / z / p_dist | FP16，通道分别为 384 / 128 / 64 |
| token 链/残基索引 | INT64 |
| 4-block Pairformer + C pooling 训练头 | FP32 参数和 AdamW 状态，BF16 autocast；输入与原训练路径一致先转 FP32 |

缓存字段的实际 dtype 已由实测逐样本报告记录。没有改变标签、数据划分或动态负样本协议。测试运行了 BCE 前向、反向和优化器更新，loss 有限；这些检查不代表分类效果或数值等价性验证。

旧 mini 特征来自 torch 算子，仅有 12,772 条。为遵守后续实验全部使用加速版，新特征保存在 **features_cueq**；旧 features 保留作为历史结果，不混入本次加速缓存。新 extraction_config 包含 kernel_backend，恢复时严格校验。

## 实际资源和八卡生成→存储

检查了 Tang3、Tang1、Tang2。三台机均有其他任务占卡，不能称为独占基准。Tang3 八卡足以运行纯 mini 生成，但不能在所有卡上同时常驻 encoder 和训练头。所有实际测试均在 Tang3 A40 完成，无需迁移到其他主机。

单卡生成测试 GPU 1，共 6 条，首条单独保留，其余 5 条统计：

| 热样本指标 | 中位数 |
|---|---:|
| CPU 特征准备（使用本地 ESM 缓存） | 0.521 s |
| mini trunk 推理 | 1.191 s |
| GPU→CPU | 0.048 s |
| Foldtmp 保存 + fsync 请求 | 1.064 s |
| 完整生成→保存 | 2.901 s |

随后在 **GPU 0–7 同时运行 8 个独立生成进程**，各处理 6 条互不重复的样本：

- 48/48 条成功，实际写出 **3,255,262,240 字节（3.255 GB）**。
- 包括八进程启动、模型加载、首条热身和保存的总墙钟 **98.907 秒**，吞吐 **0.485 条/秒**。
- 去除每卡首条后共 40 条：推理中位 **1.228 秒**，保存+fsync 中位 **8.248 秒**，单 worker 每条全流程中位 **10.305 秒**。
- 八卡测试最大 PyTorch 保留显存 **2744 MiB**；共享 GPU 和共享挂载会产生争用。
- 写盘调用请求 fsync，但没有在 Foldtmp 服务端验证物理介质持久化。统计的是挂载返回后的完成时间。

八卡并发时，写盘时间显著高于单卡，不能将单卡 1.19 秒推理直接当成生成到 Foldtmp 的耗时。48 条短测含启动开销，不据此承诺 96,670 条完整提取 ETA。样本为 baseline 前 48 条，未覆盖全数据长度长尾。

## 正样本：缓存读取还是直接生成

同一张 GPU 1、同一组 6 条真实训练正例、相同 4-block 分类头、激活重计算、BF16 head、每条一次 AdamW 更新。表中热样本统计排除首条，缓存读取没有人为清空系统或 SSHFS/server 缓存，因此不标为严格冷读。

| 路径 | 初始化后到首条前向完成¹ | 热样本准备到前向，中位 | 完整前向/反向/更新，中位 | 完整训练步，均值 |
|---|---:|---:|---:|---:|
| 刚生成的加速缓存，从 Foldtmp 读取 | 5.058 s | 1.034 s | 1.884 s | 2.332 s |
| 已有旧缓存，从 Foldtmp 读取² | 6.295 s | 1.529 s | 2.362 s | 2.473 s |
| mini 直接生成，同 GPU 交给 head | 32.042 s | 1.934 s | 2.775 s | 2.811 s |

¹ 计时从 Python 模块导入之后、GPU/模型初始化之前开始；不含 Python 解释器与模块导入。首条时间包括模型加载、数据准备、首次 kernel 启动等，不能视为每条持续耗时。在线首条使用更早的系统缓存状态，初始化差异不可全归因于策略。

² 旧缓存仅用于 I/O/形状一致的时间对照，不用于新加速模型的训练效果比较。

实测说明：**正例预缓存不是启动训练的必要条件，但在本次单卡条件下略快。** 读完旧缓存到前向中位 1.529 秒，直接生成到前向 1.934 秒；并非普通 Protenix 文档中的 9.37 秒冷读对 4.45 秒推理。保存成本也要计入：用本次单卡均值粗算，正例预生成保存 3.048 秒，之后每轮比直接生成节省 0.338–0.479 秒，约 **6–9 次重复训练访问**才抵消首次预生成成本。此计算不含八卡读盘竞争、预取或多机，不能当成生产环境收益保证。

当前继续完成固定 baseline 提取，供多轮训练、固定验证/测试及对照使用；训练不必等待正例缓存全部完成，已有实测在线路径可用于后续正式训练 provider。没有启动正式全量训练或宣称 DDP 已验收。

## 负样本：在线生成与真实 GPU→GPU

从既有 epoch 1 动态负样本输入中取 6 条真实负例，保持原四链序列，label=0。每轮正式训练仍应使用 seed 42 + epoch 的动态抽样计划；该短测没有重新生成固定 val/test。

| 路径 | 热样本 trunk | 特征跨卡传输 | 完整训练步中位 |
|---|---:|---:|---:|
| GPU 1 生成并在同卡训练 | 1.210 s | 无需复制 | 2.832 s |
| GPU 1 生成→GPU 4 训练 | 1.204 s | **0.009 s** | **2.830 s** |

GPU 1→4 的 CUDA peer access 检查为 True；每条约 67 MB 的 s/z/p_dist 通过 GPU tensor `.to()` 交给另一张 GPU 的训练头。两条负例训练路径均没有 `.cpu()` 保存负例联合特征，也没有负例 `.pt` 写盘/重读。跨卡短测采用串行生成→传输→训练，没有实现流水线重叠，因此不能由本结果声称跨卡已经带来吞吐提升。

正例同卡组合峰值保留显存约 8.9 GiB；真实负例同卡测试达到约 **11.7 GiB**，部署需结合长尾再留余量。当前机器没有八张卡都满足该组合显存，故本次完成的是八卡生成写盘与单/双卡训练路径检查，**不是八卡 DDP 在线训练测试**。

本次 mini 在线训练需要的 ESM 单链表示已预先复制到本地 `/data`。训练路径仍有 CPU 数据准备和少量输入 H2D；避免的是体积大的联合特征经过 Foldtmp。后续若要求连单链输入都不读本地磁盘，应另外实现有界 RAM/显存缓存，不能声称当前已做到所有输入零 I/O。

## 恢复提取与后续运行入口

- 持久会话：`tmux` 的 `mini_esm_accel_20261017`。
- 入口：`bash/run_mini_esm_accelerated.sh`。
- Supervisor：`scripts/resume_mini_esm_accelerated.py`，每块 256 条，使用 Tang3 GPU 0–7。
- 输入：原 `inputs/baseline.json`，**96,670 条**，包含训练正例及固定 val/test（含固定评估负例）。
- 输出：`/NAS/luyq/TMP_prediction/protenix_embeddings/hitph_v1/mini_esm_v0.5.0_c4_no_msa/features_cueq`。
- 本地链缓存、预处理、中间索引和日志：`/data/luyq/mini_esm_accelerated_20261017`。
- 状态：该目录下 `state.json`，completed 按完整验证的块更新；块运行时实际 `.pt` 数会更大。
- 本次不离线生成 158,930 条训练负例联合特征。旧 wrapper 也改为默认在线负例；离线需要显式 `--offline-train-negatives`。
- 遇到 I/O 错误，对当前块最多重试三次，恢复时严格验证已生成特征；每块保存验证索引，最终汇总为完整 index 和 `completion_report_cueq.json`。
- 加速缓存尚未完整完成，只有最终报告 status=complete 才表示全部固定数据提取完毕。

## 可复核材料与检查

- 汇总：`docs/benchmarks/mini_esm_20261017/summary.json`。
- 七组逐样本 JSON 均保存在同目录；原日志在 `/data/luyq/mini_esm_routes_20261017`。
- 基准入口：`scripts/benchmark_mini_esm_routes.py` 与 `scripts/run_mini_esm_eight_gpu_benchmark.py`。
- 缓存身份、精度、概率、链序列、加速 backend 不匹配拒绝，以及本地 ESM adapter 路径恢复测试：`pytest tests/test_mini_esm_cache.py -q`，**16 passed**。
- 基准是 startup/training-step 检查；未测试验证集效果、新旧算子数值等价、完整 DDP、多 epoch 收敛和长期 I/O 稳定性。
