# FoldAbs 训练加速：开源实现检索与静态适配评估

检索日期：2026-11-04。仅读取 GitHub API、许可证、源文件及 issue；未安装框架、运行模型、执行训练或测量性能。日期表按 GitHub 默认分支最近提交时间（UTC），不能用 star 数或仓库 `updated_at` 代替维护活跃度。issue 是用户报告，除注明已合并修复外，不认定在本项目/最新版中必然复现。

## 当前任务决定了选择范围

`cyan741/FoldAbs_Tm_MSA` 的 `tmp-specificity` README（读取时目标 SHA：`2536d49dd0231bfcb06aabcb17ffd0e18125c3a9`）描述的是：缓存 ESM2-3B 单链表示，在线执行冻结的 Protenix Mini-ESM 四轮 trunk/distogram，再训练分类头 Pairformer；不使用 MSA/template，不执行 diffusion/confidence。动态负例变更四链组合，因此不能把联合表示直接替换成单链拼接，不能把已有 ESM 缓存当作新优化，也不能把冻结编码器全量离线缓存作为不改变训练协议的默认方案。

这使 Protenix 的 pair/triangle kernel、有限队列调度、跨阶段复用和峰值活跃张量管理，比接入一个完整 LLM trainer 更直接。

## 候选比较

| 项目 | 默认分支最近提交 | 许可证核对 | 可借鉴实现 | 依赖、已知问题与取舍 |
|---|---|---|---|---|
| [Protenix](https://github.com/bytedance/Protenix) | 2026-09-21；本次 HEAD 是 CI 更新，不能由此断言所有算子仍持续优化 | 代码 Apache-2.0；README 新增 Protenix-v2 权重专有例外，与文末笼统描述并存。不能把全部权重视为 Apache | 本项目直接上游；PyTorch SDPA、triangle backend、chunk；README 共享变量缓存属于 diffusion 阶段，非本项目 trunk 加速 | 当前 requirements 固定 torch 2.7.1、cuEquivariance 0.8.0、DeepSpeed 0.17.5、Triton 3.3.1；不宜将最新上游依赖整体覆盖历史环境。#265 报告 Blackwell 支持问题；#362 报告末步评估 label size mismatch。优先局部接口兼容与回退 |
| [DeepSpeed / DS4Sci](https://github.com/deepspeedai/DeepSpeed) | 2026-11-04，活跃 | Apache-2.0，源文件含 SPDX | 为 Evoformer 专门设计的 fused attention；已有前后向实现 | CUDA 11+、SM>=7、CUTLASS>=3.1（当前 builder）；输入 `[*, L, H, D]`，限定两种 bias 形状，`L>16`，反向 head dim<=64。可作为受保护的可选 backend，不应以 DeepSpeed engine 替换整个训练系统 |
| [NVIDIA cuEquivariance](https://github.com/NVIDIA/cuEquivariance) | 2026-10-30；仓库推送至 2026-11-03 | README 明确仓库文件 Apache-2.0；GitHub license API 为 null/404，外部分发 kernel wheel 应单独核验分发条款 | 几何网络与 triangle 操作 CUDA kernel，PyTorch/JAX 接口；项目已经使用，应首先核对调用和版本 | README 明示 Beta；CUDA 12/13 wheel 分开。不能将整个仓库许可声明自动外推到所有预编译分发件。适合继续使用已验证版本和明确 fallback |
| [OpenFold](https://github.com/aqlaboratory/openfold) | 2025-12-16；近期默认分支活动不足 | Apache-2.0 代码；README 单独说明 AlphaFold 参数 CC BY 4.0 | 成熟的低显存 attention/chunk、训练重计算、多个 kernel 的互斥选择、形状回退 | 最新合并 [#560](https://github.com/aqlaboratory/openfold/pull/560) 就是 cuEquivariance 外部 fallback 和 scaling 修复；[#582](https://github.com/aqlaboratory/openfold/issues/582) 是梯度累积参数可能未传给 Trainer 的开放报告。作为设计参考，不宣传为当前最活跃依赖 |
| [Boltz](https://github.com/jwohlwend/boltz) | 2026-05-29；近半年默认分支无新提交 | README 明确代码和权重 MIT，LICENSE API 为 MIT | triangle kernel 可选依赖及独立 CUDA extras；训练/推理分离的工程组织 | 当前 pyproject: Python >=3.10,<3.13、torch>=2.2、Lightning==2.5.0，CUDA extras 使用 cuEquivariance>=0.5；README 仍称 Boltz-2 训练代码 coming soon。[#638](https://github.com/jwohlwend/boltz/issues/638) 是已关闭的旧版 trifast/Torch/CUDA wheel 冲突报告，不等于当前版仍有该 bug。非直接替换候选 |
| [FastFold](https://github.com/hpcaitech/FastFold) | 2023-07-11；`pushed_at` 2024-07-16 也明显陈旧 | LICENSE 主体 Apache-2.0，保留第三方通知 | Dynamic Axial Parallelism、通信计算重叠、融合算子、分块；是蛋白模型借鉴大模型并行技术的明确先例 | README CUDA>=11.3、Torch>=1.12、Triton需CUDA11.4；[#194](https://github.com/hpcaitech/FastFold/issues/194) 报告旧 conda 包无法解析。目标不用 MSA，DAP 引入多卡通信与集成复杂度；仅历史参考，不新增依赖 |
| [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) | 2026-11-04，活跃 | LICENSE 首段 BSD-3-Clause 式 NVIDIA 许可，另有按文件 Apache/MIT 等第三方通知；GitHub API `NOASSERTION`，不应简写整个仓库为单一 Apache | 张量/流水/序列并行、选择性重计算、算子融合、通信重叠 | 大规模 transformer 训练栈及 GPU 通信依赖。冻结 encoder 不需要梯度/优化器状态分片；Pairformer 轴语义也不同。借鉴设计，不整体迁移；[#7452](https://github.com/NVIDIA/Megatron-LM/issues/7452) 报告并行组合数值问题，提示并行正确性不是自动保证 |
| [SGLang](https://github.com/sgl-project/sglang) | 2026-11-04，活跃 | Apache-2.0 | 有界队列、连续调度、资源预算、prefill/decode 分离；encoder→head 生产消费调度是本项目的类比方案 | 主要为 LLM/VLM serving；自回归 KV/radix cache 依赖因果前缀不变。四链双向联合表示随 peptide 改变，不能把 TCR/HLA 的联合 K/V 视为不变前缀。可借鉴 encoder→head 异步供给，不能直接复用 KV cache/投机解码 |
| [ms-swift](https://github.com/modelscope/ms-swift) | 2026-11-04，活跃 | Apache-2.0；模型和数据另行遵守原许可 | 训练配置管理、混合精度、checkpoint、梯度累积、冻结组件与可训练组件隔离 | Transformers/PEFT/TRL 等模型适配栈，推荐当前 Torch/CUDA 版本远新于项目历史环境；README 当前 DeepSpeed 范围 >=0.19.3,<0.20。会与 Protenix 依赖矩阵冲突。可借鉴配置验收，不整体引入 |

## 可落地优先级（静态建议，不是速度结论）

1. **优先消除重复调度、重复加载和多余张量存活。** 保持采样顺序、RNG、恢复游标、梯度累积及 loss 归一化不变；队列以字节预算/样本数限制，避免吞吐优化将 VRAM 放大。仓库存在独立 encoder pool 实验，但本基线缺少 README 所述训练接入文件；应核对实验实现的进程所有权与 CUDA IPC 生命周期，再考虑集成。此方向不必增加上游依赖。
2. **按数学等价性做局部张量优化。** 聚合/读出是否先构造完整 `[N,N,C]` 临时量再只读一个子集、是否重复 `cat`/dtype 转换/同步，适合静态定位。真正改变聚合顺序时仍应标注浮点误差风险。
3. **核对既有 SDPA/cuEquivariance 路径，而不是再次“引入 FlashAttention”。** Protenix 当前 `primitives.py` 已调用 `F.scaled_dot_product_attention(..., scale=1.0)`，因为 query 在 `_prep_qkv` 中已经缩放。默认 SDPA scale 会再次除以 `sqrt(d)`。OpenFold #560 的经验是：先判 shape/backend fallback，再确定缩放路径。
4. **checkpoint 是显存换算力，不是默认加速。** 冻结 encoder 已无反向；其 checkpoint 没有常规训练中的收益。分类头重计算是否能换来更多 encoder 并发，要依据硬件和显存峰值实测。没有硬件时不默认打开。
5. **不要在此次静态优化中修改训练任务。** 降低 cycles、缩小网络、量化/LoRA/蒸馏、减少负样本、近似 joint representation 会改变科学基线，需独立消融而不是工程等价优化。

## 来源与复用边界

本次只借鉴公开设计，没有复制第三方代码。建议以后接入新算子采用可选配置、能力探测、原生 fallback，并记录精确 torch/CUDA/driver/GPU/kernel 版本及原始前后向数值差异。源码静态分析可以说明少了哪些计算/分配，不能给出吞吐倍数、VRAM实测或收敛保证。

精确提交、许可核对状态和原始来源链接见同目录 [acceleration_sources_20261104.json](acceleration_sources_20261104.json)。该文件记录来源索引，不是全部源码或许可证全文副本。同类项目通过公开 GitHub 资源检索；目标仓库通过用户指定链接和现有授权读取。未向外部服务上传本项目文档、数据、权重或代码。
