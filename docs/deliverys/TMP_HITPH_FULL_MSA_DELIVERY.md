> 历史文档（2026-11-03）：MSA 建库与搜索脚本已从仓库移除，以下命令不再作为当前操作入口；保留本文用于追溯实验方法。

# HiTPH 完整 MSA 交付与当前重构

2026-10-08：A/B/H v2 参考库已建立（819/403/18,157 条）；原始全量验收113,367条MSA及85,200条四链记录全部通过。当前按用户要求迁移为无SHA、稳定链ID、直接role缓存目录及peptide内存MSA，迁移后验收结果以 `reports/full_msa_validation.json` 为准。

当前代码统一在 `scripts/MSA_building/`，入口：

```bash
bash scripts/MSA_building/run_hitph_full_msa.sh cache --resume --workers 32 --cpu 1
bash scripts/MSA_building/run_hitph_full_msa.sh validate --require-complete
bash scripts/MSA_building/run_hitph_full_msa.sh ensure --record-ids record_ids.txt --output-json /NAS/luyq/TMP_prediction/hitph_msa/protenix_inputs/request.json
bash scripts/MSA_building/run_hitph_full_msa.sh extract --input-json /NAS/luyq/TMP_prediction/hitph_msa/protenix_inputs/request.json --output-dir /NAS/luyq/TMP_prediction/hitph_msa/features/request --gpu-id 7 --cycles 1 --resume
```

[目录、协议依据与完整数据流](TMP_MSA_STORAGE_AND_DATAFLOW.md)。完整实际迁移记录为 `reports/layout_refactor.json`。旧重复sources已清理约24.01 GiB，记录为 `reports/storage_cleanup.json`。v1仍是当前母库、工具及来源依赖；hitph_msa是当前正式交付。

六条真实Protenix pilot联合特征已生成，FP16缓存约64.4 MiB/record，cycle=1。原始生成平均约4.67秒，GPU峰值约3.04 GiB；该小批测量不是全量吞吐保证。重构期间发现旧提取入口未落实固定随机种子，现已加入每条记录固定种子及确定性算法；同一输入重复提取的 s/z/p_dist 逐元素一致。六条正式 pilot 尚待按新策略重新生成，不能仅修改旧张量元数据。全量正例特征估计5.22 TiB，尚未计负例组合，当前全量未生成。训练前仍需安排存储或按需策略、完成实际正/负例组合特征、身份/输入检查及训练配置检查。

相关测试为 `tests/test_hitph_msa.py`、`tests/test_tmp_msa.py` 和 `tests/test_tmp_prediction.py`，上一轮37项通过，最近确定性提取修改尚待完成最终回归检查。SHA删除后，验证依赖完整query/对齐结构/深度/覆盖/角色/记录映射/显式配置，不再提供任意文件字节级损坏检测。
