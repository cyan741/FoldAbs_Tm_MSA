# MSA 加速：从 2.2 万条代理库到 UniRef50 主实验

## 已知问题

以前的紧凑库有 22,198 条序列。其构造是：从 NbThermo 中按长度/3-mer 多样性选择
200 条 Nb，再**复用 ColabFold 缓存**中每个种子的前 1,000 行命中并去重。

它的优点是本地检索 610 条独特链仅用了约 1,897 秒；缺点是母库并非论文中应有的
UniRef50。它只能叫“紧凑 MSA 代理库（compact proxy）”，不能作为师兄所说的
“从 UniRef 搜索后再缩小”的最终实现。

## 现在的快速、受控诊断

`run_public_dsf_official_test_compact_msa_branchonly.sh` 保持以下项目和常规 MSA
正式结果完全一致：

- 作者给出的 thermstability test-63；
- 固定的 public train/validation manifest（340/30）；
- 冻结 ESM-2 特征；
- 训练步数、验证集选 checkpoint；
- 当前已验证的 branch-only 融合。

唯一替换的是：Protenix 输入的 MSA 来自已有 22,198 条本地紧凑代理库，而非
ColabFold 常规 MSA。它回答的是“旧紧凑 MSA 结果差，是否主要因为旧的全局融合
归一化？”而不是最终的 UniRef50 加速结论。

## UniRef50 主实验（下一阶段）

师兄建议对应的正确工程链路应为：

1. 取得一个有版本记录的**完整 UniRef50 FASTA**；
2. 对同一 200 条、只按序列多样性选出的 Nb 种子跑 local jackhmmer；
3. 汇总所有命中、去重，形成新的规模可审计的 compact database；
4. 用该库给完整 official-test-aligned 433 条链产生本地 MSA；
5. 重新抽取冻结 Protenix 特征，并用同一个 branch-only FoldAbS 训练；
6. 与常规 MSA 同时报告：一次性建库成本、每条 MSA 检索耗时、深度分布和 test-63 指标。

这一步要准备几十 GB 级的 UniRef50 文件，并且 200 次全库 jackhmmer 是长任务；
在数据库文件的具体版本和下载来源确定前，不应启动它，更不能把 2.2 万条代理库
改名为 UniRef50。
