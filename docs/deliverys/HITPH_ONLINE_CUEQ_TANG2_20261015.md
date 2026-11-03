# Tang2 Protenix 加速算子安装与无写盘试测

日期：2026-10-15。机器：Tang2，NVIDIA A40 48 GiB，驱动 535.129.03。基准环境 `/data/luyq/hitph_cycle4_worker/env` 为 Protenix 1.0.0、PyTorch 2.7.1+cu118、Triton 3.3.1。只使用用户授权的 GPU 6/7；GPU 0 已有新任务，未改动。其他正在运行的旧缓存 worker 继续使用原环境。

## 安装位置和启用

Protenix 1.0.0 的包元数据要求 cuEquivariance 0.8.0 CUDA 12，但这台机当前 PyTorch 是 CUDA 11.8。0.8.0 CUDA 12 版已经隔离试装到 `/data/luyq/hitph_online_benchmark_20261015/cueq_0_8_overlay`，导入时报 `libcublas.so.12: undefined symbol: cublasGemmGroupedBatchedEx`，故**不能启用**。未修改原 Python 环境。

可运行的 CUDA 11 版安装在 `/data/luyq/hitph_online_benchmark_20261015/cueq_0_5_overlay`，版本为 `cuequivariance`、`cuequivariance-torch`、`cuequivariance-ops-cu11`、`cuequivariance-ops-torch-cu11` 均 0.5.1，另有 `opt-einsum` 3.4.0、`nvidia-ml-py` 13.615.71、`platformdirs` 4.13.0。完整清单在同目录的 `cueq_0_5_requirements.txt`。用下列命令在**新启动**的进程中启用：

```bash
source /data/luyq/hitph_online_benchmark_20261015/enable_cueq_cu11.sh
# 然后调用 Protenix，并显式设置 trimul_kernel="cuequivariance"、
# triatt_kernel="cuequivariance"；仅安装不会改变现有的 torch 路径。
```

隔离目录和 Triton 缓存都在 Tang2 本地 `/data`，未使用 Foldtmp。NVIDIA [cuEquivariance 安装说明](https://github.com/NVIDIA/cuEquivariance) 当前仅列 CUDA 12/13 包；其 [发行说明](https://docs.nvidia.com/cuda/cuequivariance/changelog.html) 说明新版已停止 CUDA 11 支持。Protenix [算子说明](https://github.com/bytedance/Protenix/blob/main/docs/kernels.md) 列出 `cuequivariance` 三角乘法和注意力选项。

## 加速结果

测试使用 `scripts/benchmark_hitph_online.py`、真实 epoch 1 负例、cycle 4、无 MSA、BF16 trunk、FP16 中间特征、4-block Pairformer 分类头和激活重计算；负例 embedding 全程不落盘。GPU 6、7 各执行 24 条，长度分别为 413–424 和 413–422 tokens；首条热身排除。

| 配置 | Protenix 推理中位 | 头训练中位 | 单条全流程中位 | PyTorch 峰值保留 |
|---|---:|---:|---:|---:|
| 原 `torch/torch`，Tang2 三卡前次长测 | 11.836 秒 | 1.101 秒 | 13.443 秒 | 约 10.1 GiB |
| cuEquivariance 0.5.1，GPU 6 | 4.456 秒 | 1.102 秒 | 6.061 秒 | 14,862 MiB |
| cuEquivariance 0.5.1，GPU 7 | 4.448 秒 | 1.097 秒 | 6.055 秒 | 14,288 MiB |

两卡 48 条全流程的外部墙钟 174 秒（含模型加载），热身后推理约快 2.66 倍。基于 158,930 条负例推理、238,395 条总样本分类头训练、八卡完全线性扩展，纯计算乐观估算为 `(158930×4.45 + 238395×1.10)/8/86400 ≈ 1.40 天/epoch`。该估算未包括 Foldtmp 正例读盘、CPU 预处理、DDP、验证、首次编译和长尾负载，也尚未做八卡实测。

短测拆分：仅加速三角乘法时推理中位 8.620 秒；仅启用 Protenix 自带 `triattention` 时为 7.318 秒；两者都用 cuEquivariance 时为 4.415 秒。各配置仅 4 条热样本，拆分数据只用于定位收益。

## 单卡多进程对照

同一时间用 GPU 6 开两个独立进程，各处理 12 条；GPU 7 开一个进程，处理 24 条。都启用 cuEquivariance 0.5.1 和分类头激活重计算。GPU 6 的两个进程合计约占 23.5 GiB，未 OOM；各进程热样本完整流程中位 12.253/12.248 秒。GPU 7 单进程热样本为 6.057 秒/条。GPU 6 完成 24 次计算用约 174.3 秒，GPU 7 用约 168.9 秒，双进程约慢 3.2%。这两个进程各有独立分类头，代表吞吐/显存试测，并非可直接使用的同一模型梯度更新。此测试不支持通过单卡启动更多进程缩短工期。

在原 `torch/torch` 路径下也做过相同结构的 16 次计算：GPU 6 两个进程各 8 条约 253 秒，GPU 7 单进程 16 条约 233 秒；双进程约慢 8.6%。两种算子路径都没有观察到多进程吞吐收益。

## 一致性边界

当前正例缓存由 `torch/torch` 生成，而在线负例若改用 cuEquivariance 会改变数值路径。首次直接对照的 `z` 平均绝对差约 1.27；但同样样本连续用 `torch/torch` 两次，对照差约 1.10。随后复制输入并恢复随机状态后，`torch/torch` 重复差约 0.91，`torch/torch` 对 cuEquivariance 差约 0.76。该小样本现象表明原路径本身存在显著运行间波动，**不能据此声称加速算子等价，也不能将全部差异归因于它**。在训练中混用正例旧缓存与负例新算子前，应扩大跨样本对照，并检验分类头是否能辨识算子来源。这里仅完成依赖安装与测速，没有切换正式训练路径或改写正例缓存。

原始 JSON 和日志位于 Tang2 `/data/luyq/hitph_online_benchmark_20261015/`：`cueq_long_gpu6/7`、`cueq_two_on_6a/6b`、`cueq_one_on_7`、`cueq_0_5_gpu6`、`fast_mul_only_gpu6`、`triatt_only_gpu7`、`torch_repeat_gpu6`、`torch_repeat_fixed_gpu6`、`cueq_compare_fixed_gpu7`。各组 `report.json` 和启用脚本也复制到本地 `/data/luyq/hitph_online_benchmark_cueq_tang2_20261015/` 供复核。前次原生路径试测见 `docs/HITPH_ONLINE_BENCHMARK_TANG2_20261015.md`。
