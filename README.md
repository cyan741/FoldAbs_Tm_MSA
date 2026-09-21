# FoldAbS Tm 公开重建与抗体 MSA 小库技术分享包

## 这是什么

这是一个**技术参考包**，整理自 FoldAbS 热稳定性（Tm）公开重建与抗体 MSA 小库探索。可供复用或阅读的部分包括：冻结 ESM-2 650M 表征、冻结 Protenix 表征、轻量 FoldAbS 风格回归头、固定划分训练/评估，以及代表性抗体链的 MSA 命中聚合、本地 FASTA 小库检索和 3-mer/长度距离选种子。

它不是 FoldAbS 官方仓库，也不是可直接用于 TCR-pMHC 的成品模型。

## 包含内容

- `scripts/`：特征提取、Tm 训练、本地 jackhmmer MSA、紧凑库构建、种子选择与 GPU 自检。
- `src/foldabs_repro/`：数据加载、指标、标签标准化等公共模块。
- `tests/`：小型合成测试；不下载模型、不运行真实 MSA 或 GPU 训练。
- `docs/`：实现与论文的对齐边界、MSA 更正方案和种子选择说明。

## 明确不包含

- 任何 FLAb/Jain 原始数据、MSA、UniRef 数据库、缓存、特征、模型 checkpoint、预测结果或服务器日志；
- Protenix、ESM-2、ESMFold 的模型权重；
- FoldAbS 或 AF3-TurboAb 论文 PDF；
- 云端凭据、账号信息和特定服务器实验目录。

## 依赖与运行边界

基础 Python 依赖见 `pyproject.toml`；ESM 特征提取额外需要 `fair-esm==2.0.0` 和官方 `esm2_t33_650M_UR50D` 权重；结构特征提取和回归头还需要与当前 GPU/PyTorch 兼容的 Protenix 安装。MSA 检索需要 HMMER 的 `jackhmmer` 和一个用户自己合法获得的本地 FASTA 数据库。

建议先执行：

```bash
python -m pip install -e .
python -m pip install fair-esm==2.0.0 pytest
python scripts/gpu_smoke.py
python -m pytest tests -q
```

上述只是软件检查。真实训练还需自行提供抗体 CSV、固定数据划分、Protenix/ESM 特征目录和本地数据库路径。

## MSA 小库路线

1. 在训练链内、按链类型分层，用重叠 3-mer 集合的 Jaccard 距离与归一化长度差选代表种子；
2. 对种子用完整数据库做常规 MSA；
3. 聚合种子 MSA 的非查询命中、去重成小型 FASTA；
4. 用同一 `jackhmmer` 参数在小库检索所有目标链，再提取结构特征与训练。

本包的 `0.8 × 3-mer Jaccard 距离 + 0.2 × 长度差`、阈值和种子数均是抗体 Tm 数据上的透明实验设定，不是通用最优规则。

## 给 TCR-pMHC 工作的提示

可迁移的是：冻结表征、固定 split、特征审计、训练/验证/测试分离和本地 MSA 缓存的工程框架。不可直接照搬的是抗体的 H/L 格式、Nb 种子选择、3-mer 半径、库内容和 Tm 回归头。TCR-pMHC 应按 TCR α/β、肽和 MHC 的实体定义、泄漏风险和监督任务重新设计数据与模型。

## 引用与共享说明

请称其为“基于公开文献的 FoldAbS Tm 技术重建/工程参考”，不要称为 FoldAbS 官方实现或官方结果。分享前请自行确认所在课题组对代码再分发、外部依赖和数据来源的要求。
