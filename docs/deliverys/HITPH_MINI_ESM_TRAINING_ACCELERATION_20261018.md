# mini-ESM 在线训练加速进展（2026-10-18）

## 当前结论

优先使用 **加速 Protenix + cuEquivariance 训练头 + 关闭激活重计算 + 2 个 CPU 预取 worker**。保留 4 cycles、4-block head 和原输入精度。48 条真实样本复核从 2.796 秒/条降至 **1.702 秒/条**，单步耗时减少 **39.1%**。

不需要为了启动在线训练预生成正负样本联合特征。ESM2-3B 单链缓存已存在，当前在线路径读取本地单链表示，不加载或重新运行 ESM2-3B。不同链组合仍需运行 mini Protenix，不能直接拼接得到新的联合 pair 表示。

离线联合特征写盘保持停止：`/data/luyq/mini_esm_accelerated_20261017/state.json` 为 `stopped_by_user`，完整验证 2,304 条。历史缓存保留。本轮仅保存小型日志和计时 JSON，没有恢复 Foldtmp 联合特征生成。

## 方法和资源

- Tang3 NVIDIA A40，单卡 GPU 1，机器与其他任务共享；不是独占八卡 DDP 测试。
- 冻结 `protenix_mini_esm_v0.5.0`，seed 42，无 MSA/template，不运行 diffusion/confidence。所有 encoder 推理均使用加速版。
- 训练头为 Pairformer + C pooling；每条样本执行 BCE 前向、反向和 AdamW 更新。
- 18 条比较包含 6 正例和 12 负例，固定顺序；前 3 条作为热身，统计余下 15 条。各配置使用同一输入。
- 48 条复核包含 16 个训练正例和 32 个 epoch 1 负例，跨输入列表均匀抽取并纳入两类各自最长样本；413–434 tokens。前 3 条热身，统计余下 45 条。此处“跨数据”指覆盖该实验输入列表，不代表外部独立数据集。
- 在线联合特征通过同卡 GPU tensor 交给 head；本地 ESM 输入读取、CPU 特征准备和 H2D 仍存在。
- 按阶段同步计时；没有实现异步多 GPU 生成/训练流水线。峰值 reserved 是 PyTorch 分配器保留显存，不包括所有进程/驱动开销。

## 18 条实测：保持 4 cycles、4-block head

| 配置 | 平均完整训练步，秒 | 峰值 reserved，GiB |
|---|---:|---:|
| 原 torch head + 激活重计算，无预取 | 2.882 | 14.39 |
| torch head，关闭重计算 | 2.604 | 31.41 |
| cuEq head + 激活重计算 | 2.356 | 6.90 |
| cuEq head + 激活重计算 + 预取 | 1.809 | 6.90 |
| cuEq head，关闭重计算，无预取 | 2.166 | 18.40 |
| **cuEq head，关闭重计算 + 预取** | **1.707** | **18.40** |
| 上项 + BF16 head 输入 | 1.659 | 15.91 |

推荐配置的平均分解：CPU 等待 0.016 秒、H2D 0.094 秒、encoder 1.199 秒、head forward 0.097 秒、backward + step 0.298 秒。后续主要瓶颈已转为 encoder。

首次 cuEq 无重计算试验暴露了多加 batch 维的问题并失败；已修复。表中采用修复后的 `cueq_no_checkpoint_fixed` / `cueq_no_checkpoint_prefetch`，不把失败运行算作成功数据。

## 48 条复核与每轮耗时估算

| 配置 | 平均秒/条 | 中位秒/条 | 峰值 reserved，GiB | 八卡理想 epoch，小时 |
|---|---:|---:|---:|---:|
| 原 torch head + 重计算，无预取，4/4 | 2.796 | 2.775 | 26.05 | 23.14 |
| cuEq head + 无重计算 + 预取，4/4 | **1.702** | **1.693** | **23.53** | **14.09** |

两组均完成 48/48 条，最终 261 个梯度张量全部有限且非零。优化组热样本范围 1.646–1.821 秒，encoder 平均 1.198 秒，约占单步时间的 70%。

每轮按 79,465 正例 + 158,930 负例 = 238,395 条估计，八卡理想值为 `秒/条 × 238395 / 8 / 3600`。若额外假设实际并行效率为 70%–80%，原配置约 28.9–33.1 小时，优化配置约 **17.6–20.1 小时**；该效率是假设，不是测量结果。估算不包括固定验证、checkpoint 保存和完整启动开销，也未验证 DDP 通信及八进程 CPU 争用。

完整复核的显存峰值高于 18 条短测：推荐配置达到 23.53 GiB。部署应留出驱动、通信和数据形状变化余量，不能按早期约 18.4 GiB 直接分配。Tang3 GPU 1 足以完成本次单卡测试，但不据此认定其八张卡当前都可用于训练。

## 精度与正确性检查

ESM 单链缓存 FP32；encoder BF16 autocast；联合 s/z/p_dist FP16；head 参数和 AdamW 状态 FP32、计算 BF16 autocast，默认将 head 输入展开为 FP32。没有实现 INT8/FP8；A40 不具备原生 FP8 Tensor Core 支持。

同一真实联合特征、相同初始 head、dropout=0，连续 3 次 AdamW 步的比较：

- torch 与 cuEq head：logit 差为 0，梯度整体相对 L2 差约 4.25e-7–1.14e-6。
- cuEq 改用 BF16 输入：logit 最大绝对差 0.00097656，梯度相对 L2 差约 0.47%–0.51%。额外提速仅约 2.8%，暂不切换推荐输入精度。
- 两组梯度均有限；初始化后的第一步部分参数梯度为零，后两步 261 个梯度张量均非零。
- 单元检查 `tests/test_tmp_head_checkpoint.py tests/test_tmp_prediction.py tests/test_mini_esm_cache.py`：**27 passed**，14 条既有 matplotlib 弃用警告。

数值检查只覆盖小样本，尚不能证明验证集指标等价。

## 改变模型配置的速度候选（18 条）

以下均为 cuEq head、CPU workers=2、开启激活重计算，且未做分类效果验收：

| cycles / head blocks | 秒/条 |
|---|---:|
| 2 / 4 | 1.216 |
| 1 / 4 | 0.915 |
| 4 / 2 | 1.587 |
| 4 / 1 | 1.479 |
| 2 / 2 | 0.966 |
| 1 / 1 | 0.584 |

48 条另行复核了 **1 cycle / 1-block、cuEq、关闭重计算、预取**：48/48 完成，平均 **0.601 秒/条**，中位 0.592 秒，峰值 reserved **7.72 GiB**，末步梯度全部有限且非零。八卡理想估计 **4.97 小时/epoch**；若假设 70%–80% 效率则约 6.2–7.1 小时。该配置与上表 18 条的重计算设置不同，不能把两组差值单独解释为样本规模影响。

减少 cycles 或 head 深度会改变模型，必须用固定验证集比较 AUPRC/AUROC 等指标和收敛情况后再决定是否用于正式实验。

## 实现与复核入口

- `src/foldabs_repro/tmp_prediction/model.py`：可选 cuEq head kernels，修复 cuEq batch 维，支持激活重计算。
- `scripts/train_tmp_specificity.py`：增加 head kernel 和激活重计算选项。该训练入口仍使用既有特征数据路径，尚未接入完整在线 provider/DDP。
- `scripts/benchmark_mini_esm_routes.py`：真实正负标签、cycles/head 深度、CPU 预取、热身、输入精度和梯度检查。
- `scripts/check_mini_esm_head_kernels.py`：torch/cuEq 与 BF16 输入的三步数值比较。
- 原始记录：`/data/luyq/mini_esm_optimization_20261017`；ESM 测试子集缓存位于其 `esm_cache/`。
- 随仓库保存的逐样本结果：[benchmarks/mini_esm_optimization_20261018](benchmarks/mini_esm_optimization_20261018)。

正式训练、八卡 DDP 吞吐、固定验证集效果、INT8 量化和 torch.compile 均未完成，不将当前计时测试等同于这些任务已验收。
