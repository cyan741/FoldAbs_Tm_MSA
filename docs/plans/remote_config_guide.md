# 远程迁移配置指南
| 项目 | 必须准备的内容 |
|---|---|
| 1. 服务器与访问 | 2×A100 80GB、SSH 权限、NVIDIA 驱动、CUDA/NCCL、MPS；确认 GPU 空闲及 CPU、内存、共享内存资源 |
| 2. 项目代码 | 完整仓库 src/、scripts/、configs/、tests/ 及依赖文件；包含编码器确定性聚合修复、Torch 分类头、CUDA IPC 编码器池和梯度同步实现 |
| 3. Python 环境 | 复用已验证组合：Python 3.11、PyTorch 2.6.0+cu126、Triton 3.2.0、Protenix 1.0.0、cuEquivariance 0.5.1；保留完整依赖版本清单 |
| 4. 数据集 | 完整 splits/：训练/验证/测试划分、正样本黑名单、负采样候选池、分组信息和来源校验清单 |
| 5. ESM 特征缓存 | 完整 esm_cache/，包括链特征、元数据、run_spec.json、SHA256 清单；对照/data/luyq/immrep25_cap50_a100_20261029下清单，迁移子集缓存约 12.85 GB到新建文件夹immrep_downsample_cap50 |
| 6. Protenix 资产 | protenix_mini_esm_v0.5.0.pt；CCD components、RDKit molecule 数据、PDB cluster、obsolete release 数据。已有完整 ESM 缓存时，不必重新下载 ESM 权重；当前模式不需要 MSA 数据库 |
| 7. 训练恢复资料 | 完整 checkpoint：分类头参数、optimizer、scheduler、随机数状态、epoch/step/数据游标；同时保留运行配置、历史指标、日志和逐 epoch 权重 |
| 8. 运行配置与验收 | 每卡 3 编码器 + 1 Torch 分类头，冻结编码器用 cuEquivariance，4 cycles、BF16、MPS、关闭重计算；迁移后检查文件哈希、前后向、双卡参数同步、确定性、保存恢复及显存 |

迁移所需文件在本机的路径如下：

代码：/home/luyq/FoldAbS_Tm_MSA
环境：/NAS/luyq/miniconda3/envs/foldtmp
数据：/NAS/luyq/TMP_prediction/data/hitph_l4_v1/splits
缓存：/NAS/luyq/TMP_prediction/protenix_embeddings/hitph_v1/mini_esm_v0.5.0_c4_no_msa/esm_cache
资产：/NAS/luyq/TMP_prediction/protenix_assets/proteinix/protenix_assets


迁移后统一命名：
- 代码：/home/luyq/FoldAbS_Tm_MSA
- 数据：/NAS/luyq/TMP_prediction/data/hitph_l4_v1/splits
- 缓存：/NAS/luyq/TMP_prediction/protenix_embeddings/hitph_v1/mini_esm_v0.5.0_c4_no_msa/esm_cache/immrep_downsample_cap50
- 资产：/NAS/luyq/TMP_prediction/protenix_assets/proteinix/protenix_assets


**尚需完成：**将已验证的“编码器池 + Torch 分类头”接入正式训练的验证、断点恢复和逐 epoch 保存流程；当前通过的是隔离基准，不能直接用旧启动脚本获得新方案。

