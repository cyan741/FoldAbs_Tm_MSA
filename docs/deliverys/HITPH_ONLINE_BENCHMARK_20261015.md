# HiTPH 无写盘负例推理与组合显存试测

时间：2026-10-14 23:54 至 2026-10-15 00:07（北京时间）。主机 Tang3，物理 GPU 7，NVIDIA A40 约 46 GiB。脚本：`scripts/benchmark_hitph_online.py`。机器仍在运行 epoch 1 的负例写盘作业和其他用户作业；本试测未暂停它们。

## 测试方法

从现有 epoch 1 的 158,930 条负例 JSON 中均匀取样，测试样本长度 414–421 tokens。Protenix-v1 以原配置 cycle 4、无 MSA、seed 101、bf16 运行，只抽取 `s/z/p_dist`；GPU 内转 fp16，直接交给常驻 4-block Pairformer＋C pooling 分类头。分类头按照当前训练数据集行为在输入端转 fp32，使用 bf16 autocast，执行前向、BCE、反向和 AdamW 更新。测试标签设为 0，仅用于测计算和显存。

计时前后均执行 CUDA synchronize；“GPU 推理”是该段的同步墙钟时间，包含 Python 调度开销，不是仅 GPU kernel 的 CUDA Event 时间。CPU 到 GPU 的输入传输和动态配置更新发生在该段之前，未单独计时。第一条视为热身，汇总计时排除第一条。PyTorch 峰值通过 `max_memory_allocated` 和 `max_memory_reserved` 记录，单位 MiB。`torch.no_grad` 关闭 Protenix 梯度。负例特征不执行 `.cpu()`、`torch.save` 或其他文件写入。输出目录 `/data/luyq/hitph_online_benchmark_20261014/` 总共约 52 KiB，只包含抽样输入和 JSON 报告。

## 结果

| 试测 | 数量 | 排除首条后的结果 |
|---|---:|---:|
| CPU 预处理 | 6 条 | 中位 0.49 秒；范围 0.46–57.4 秒 |
| 无写盘 GPU 推理 | 6 条 | 中位 **35.23 秒/条**；范围 34.30–38.50 秒 |
| 激活重计算下 GPU 推理 | 4 条 | 中位 35.25 秒/条；范围 34.30–37.22 秒 |
| 激活重计算下分类头训练一步 | 4 条 | 中位 **3.40 秒/条**；范围 3.34–3.56 秒 |

每条 fp16 输出约 63–65 MiB。CPU 预处理第一条约 8 秒，另有一条 57.4 秒长尾；其余均低于 0.7 秒。长尾发生在共享主机运行期间，未能定位到单一原因。

| 显存观察 | 结果 |
|---|---:|
| 加载两套模型前 GPU 7 空闲 | 22,854–23,198 MiB |
| Protenix＋分类头常驻后空闲 | 19,932–20,420 MiB |
| 单条推理 PyTorch 峰值分配 | 2,913–3,096 MiB |
| 激活重计算下推理＋分类头一步峰值分配 | **8,644 MiB**（4 条中的最大） |
| 同一步 PyTorch 峰值保留 | **9,616 MiB**（4 条均相同） |

首次尝试**未启用**激活重计算的原始 4-block 头，在首条负例的分类头前向阶段触发 `torch.OutOfMemoryError`：该进程当时约用 22.45 GiB，整卡仅剩 132.81 MiB，申请 172 MiB 失败；同卡其他用户进程约 18.04 GiB，本项目旧提取进程约 3.70 GiB。该次未完成反向，不能报告原始头的完整峰值。启用逐 Pairformer block 激活重计算后，四条连续样本均完成反向和更新。

## 解读与约束

无写盘路径已证明可以在 GPU 内直接生成并消费真实负例；组合显存需要激活重计算。当前共享负载下，**GPU 7 可以运行这个串行组合**。测试结束时 Tang3 八卡空闲显存约为 `0:9820、1:12298、2:6374、3:8919、4:3550、5:8809、6:14330、7:23130 MiB`。按本次约 9,616 MiB 的 PyTorch 保留峰值，再计 CUDA context、工作区和余量，当前不能安全启动八卡相同作业；其中 GPU 2/3/4/5 明显不足。

35 秒/条是在 GPU 7 利用率约 100%、与其他进程共卡时得到的速度，不是独占 GPU 的 Protenix 本征速度。按此速率将 158,930 次负例推理理想均分到八卡，仅推理就约 8.1 天；加上头训练约 1.2 天，仍未包含正例 Foldtmp 读取、CPU 长尾、通信和验证。该外推只说明当前共享负载不能据此承诺一天一轮，不能作为释放旧任务后的正式工期。试测样本长度 414–421，尚未覆盖全量最长复合物，也未测试双 stream 或八卡 DDP。

完整原始数字：`/data/luyq/hitph_online_benchmark_20261014/inference_only/report.json`、`/data/luyq/hitph_online_benchmark_20261014/combined_checkpoint_repeated/report.json`；单条激活重计算复核在同目录 `combined_checkpoint/report.json`。
