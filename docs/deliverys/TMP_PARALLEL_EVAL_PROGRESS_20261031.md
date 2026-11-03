> 最新强制约束：正式训练已暂停于可恢复checkpoint step21600。优化完成后也不得自动恢复，必须等待用户明确命令。前文/历史中的自动恢复计划均已失效。

# 并行评估进度（可续接）

用户已授权暂停正式训练并按顺序完成优化/并行评估，完成后恢复训练。不要从头重跑已完成基准。2026-10-31 原任务在 step21400 备份后停止，停机前最近日志21420。原BCE已结束。

远程SSH：Bitahub_A100。代码目录 `/root/foldabs_group_rank_20261029/repo`，Python `/root/foldabs_immrep25_20261029/env/bin/python`。正式输出 `/NAS/luyq/foldabs_group_rank_20261029/lr_0.0001_bs_2_wd_0.01_drop_0.1_group_rank_k_2`。恢复文件见 `docs/benchmarks/parallel_eval_20261031/pause.json`，完整路径为原输出下 `resume_before_parallel_eval_step21400.pt`。模型/optimizer/scheduler/RNG均已保存，epoch9/cursor1560。

实验输出 `/NAS/luyq/foldabs_parallel_eval_20261031`。Phase A已全部完成：baseline_w1_a/b、timeline、workers2/4、checkpoint_w1。72条样本纯训练约71.67s；workers2/4改善<1%；checkpoint开启75.31s、峰值6.97GiB，关闭36.71GiB。timeline包括trace写出开销，不能拿总耗时比较吞吐。原始trace273MB留远程。

当前正在运行：`scripts/run_group_parallel_evaluation.py --root /NAS/luyq/foldabs_parallel_eval_20261031 --workers 1 --phase streams`，依次生成 streams、serial_correctness、streams_correctness（三个子目录的rank0/result.json）。运行前检查进程及phase_streams.json，禁止重复并发启动同一组测试。

下一阶段：同一代码目录运行 `python -u scripts/run_group_parallel_evaluation.py --root /NAS/luyq/foldabs_parallel_eval_20261031 --workers 1 --phase multiprocess`。独立普通双进程、MPS1/2进程，再依据收益决定3/4进程。MPS使用私有pipe/log目录，并在finally退出daemon。不是共享GPU的NCCL DDP：这些基准测的是独立副本聚合吞吐，不能宣称同一模型训练已加速。

后续：检查streams正确性与吞吐，核对多进程样本不重复、有限loss/梯度。仅采纳经验证的有效优化；若候选无收益/语义未验证，保留每GPU一个DDP进程。整理报告并从step21400恢复正式训练，验证新step、DDP一致和checkpoint。

用户另要求在5h limit重置时自动续接。此项尚未建立定时任务：本机 codex queue 可用，systemd user/atd可用；当前thread id为01a186ea-395a-7192-ba31-3f877b67f81b。app-server proxy读取额度超时，rollout中的重置时间已过期，已通过异步问题向用户询问下一次准确重置时间。不可声称已经设置成功。


## 2026-10-31 后续：用户明确要求唯一分类头

普通多进程独立训练方案已从候选中剔除；独立MPS副本吞吐不视为共同训练同一模型。新原型 scripts/benchmark_encoder_pool.py 使用4个冻结编码器进程和唯一分类头/optimizer，通过CUDA IPC与ACK共享特征。首轮24组72样本：1encoder+checkpoint74.941秒，4encoder+checkpoint59.543秒，4encoder无checkpoint56.568秒（对旧串行71.668秒吞吐+26.7%）；后者物理显存采样峰值49365MiB，平均GPU利用率99.51%，分类头仍等输入19.992秒。

远程输出encoder_pool_v1。数值验证在encoder_pool_verify_v1，关闭dropout后1vs4仍不同，首组编码器特征已存在差异，不能直接部署。追加encoder_pool_verify_repeat单编码器重复以检查基线自身可复现性。完整说明 docs/TMP_ENCODER_POOL_EVALUATION_20261031.md。所有测试完成后从step21400恢复原有正式训练配置，勿将原型直接替换正式DDP。

恢复已确认（2026-10-31 18:09）：正式group训练epoch9 step21430，DDP一致、编码器无梯度，tmux group_resume_20261031_pool。原型测试全部结束、MPS已退出。关闭dropout的单编码器重复也不可复现（maxlogitdiff0.5/maxlossdiff0.12206），因此尚未通过数值等价性；待定位编码器跨次运行差异后再决定正式集成。

## 用户最新约束：禁止自动恢复正式训练
用户2026-10-31明确要求：原配置训练暂停，解决并行优化后仍须等待用户命令才能重新启动。此约束覆盖前文所有自动恢复计划。已复制last.pt至resume_before_encoder_pool_sweep_20261031.pt并向正式launcher PID196807发送SIGTERM。后续只运行隔离评估，不重启正式训练。

## 当前活动任务（继续执行优化）
已定位并验证编码器数值差异来自BF16 atom-to-token scatter_add_，与多进程/IPC无关；实验性FP32 segment mean解决，同进程和跨进程重复均逐位一致。正式默认未改。5项新单元测试通过。
原聚合1/2/4/6/8编码器72样本分别69.69/57.83/56.36/56.63/56.87秒。修复后48组144样本：4编码器112.84秒，8编码器113.01秒；12编码器正在encoder_pool_stable_v1/enc12_long运行。主评估进程run_encoder_pool_stable.py结束会自动退出私有MPS，不会启动训练。
修复后1vs4首组特征逐位一致，首个前向相同，梯度微小差异导致后续8组maxlogitdiff0.0625/maxlossdiff0.01439。已单独验证固定特征同一分类头无更新反向也不完全确定，相对梯度L2差0.0275%~0.0285%，cos>0.99999995。正在用deterministic标志进一步检查。详见TMP_ENCODER_POOL_SWEEP_20261031.md。

## 本轮优化评估完成（禁止自动启动训练）
12编码器144样本113.749秒，无优于4/8；4/8/12稳态队列等待分别27.64%/0.114%/0.103%。4与8吞吐平台，推荐4以省显存；不能宣称唯一全局峰值。固定聚合+Torch确定性分类头的1vs4严格8组对照通过，所有logit/loss/梯度范数差0。快速cuEquivariance分类头有独立约0.03%梯度非确定性，固定样本重复实验和Torch对照已定位。所有bench结束、MPS退出，正式任务仍暂停step21600恢复点，必须等待用户明确启动命令。详见TMP_ENCODER_POOL_SWEEP_20261031.md与encoder_pool_final_reports.tgz。

## 用户要求双卡测试（正式训练仍禁止自动恢复）
新增benchmark_encoder_pool_ddp.py和run_encoder_pool_dual_evaluation.py，实现每GPU一个分类头rank+本卡encoder池，通过NCCL显式平均全部梯度后同步更新同一个模型。全部无激活重计算。输出dual_pool_v1。每卡1encoder/144样本70.122秒，每卡4encoder/144样本56.884秒，吞吐+23.27%；两种最终head参数SHA256跨卡一致。正在复测每卡4encoder/288样本。所有任务是隔离基准，结束退出MPS，不恢复正式训练step21600。

双卡评估已完成：每卡4encoder/288样本113.174秒，2.54475条/秒，每卡峰值49.26GiB，训练段平均GPU-util99.22%/99.17%，最终两卡全部参数哈希一致，encoder无梯度。测试/MPS均已退出。正式实验仍暂停step21600，未重启，原型未接入正式恢复/验证/epoch归档。详见TMP_DUAL_ENCODER_POOL_20261031.md。


## 2026-11-01：Torch分类头评估完成，双卡受显存限制
用户要求encoder保留cuEq，仅head换Torch，其余不变。head_kernel_v2：单卡cuEq112.734秒，Torch134.148秒（+19%），峰值48.53→78.95GiB；Torch重复134.116秒，48组全部logits/loss/梯度范数差0，最终参数哈希相同。双卡cuEq57.014秒，Torch完成23/24次更新后双rank backward OOM，不能声称双卡可用或确定性已通过。新增NCCL提前初始化解决首轮启动calloc失败，但未解决双卡实际激活显存压力。重计算仍关闭，未擅自改变其他训练配置。建议下一步另评估仅head激活重计算，尚未执行。全部评估退出，两卡释放，正式训练继续暂停。详见TMP_HEAD_TORCH_KERNEL_EVALUATION_20261031.md。


## 2026-11-01：用户授权Torch显存优化A-F
已启动run_torch_head_memory_evaluation.py，输出torch_memory_v1；A双卡4encoder+head重计算、B双卡3encoder无重计算、C双卡2encoder无重计算、D单卡4encoder+head重计算，随后自动选择完成的最快双卡配置进行E独立复测，逐位一致才做F 288样本长测。正式训练禁止恢复。续接时先检查summary.json/selection.json/determinism.json和远程进程，不要重复启动。


## 2026-11-01：Torch显存优化A-F全部完成
A双卡4encoder+head重计算：76.364秒、1.8857条/秒、23.29GiB/卡。B双卡3encoder无重计算：68.205秒、2.1113条/秒、77.05GiB/卡。C双卡2encoder无重计算：69.423秒、2.0742条/秒、74.21GiB/卡。D单卡4encoder+head重计算：151.219秒、22.76GiB，比历史无重计算耗时+12.73%、峰值-71.18%。E重复B：67.897秒，两rank全部logits/loss/梯度范数差0、最终参数哈希一致。F长测B288样本：135.298秒、2.1286条/秒、77.25GiB/卡，完成无OOM，两rank参数一致。A/B/C轨迹也完全一致，D与历史无重计算单卡轨迹一致。推荐速度优先B，C仅慢约1.8%且多约2.84GiB余量；A显存余量最大。仅隔离评估，正式训练默认未改，继续禁止自动恢复。所有进程/MPS已退出，两GPU0MiB。完整文档TMP_TORCH_HEAD_MEMORY_SWEEP_20261101.md，结果torch_memory_v1及归档。
