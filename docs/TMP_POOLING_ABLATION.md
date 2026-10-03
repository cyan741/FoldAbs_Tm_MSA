# Stage 1 attention pooling：A/B/C 小数据集消融

> 日期：2026-10-03；分支：`tmp-specificity`。  
> 本次只实现代码及运行入口，未执行训练、预测或测试；没有新增实验结果。  
> attention pooling 是 TMP 自定义方案，FoldAbS 原文未披露足以确认其 pooling 算法的细节。

## 1. 本次改动

| 实验 | Single pooling（四实体） | Pair pooling（五有向界面） | 新增参数（384/128通道） |
| --- | --- | --- | ---: |
| A | mean | mean | 0 |
| B | mean | attention | 640 |
| C | attention | attention | 2176 |

每个 attention pool 有独立 query：`scores = LN(features) @ query`，沿当前实体/界面的元素做 softmax，再加权求和**原始**features。query 初始化为0，初始权重均匀；LN无可学习affine参数。没有额外projection、scoring MLP或可学习temperature。

四实体仍为 α、β、peptide、MHC；五界面仍为 α→peptide、β→peptide、α→MHC、β→MHC、peptide→MHC。pooling输出顺序与2176维latent不变，PairCondition、Pairformer、classifier结构不变。attention仅影响读出，不改变Pairformer的mask或信息传播范围。

| 文件 | 改动 |
| --- | --- |
| `src/foldabs_repro/tmp_prediction/pooling.py` | 新增query attention及A/B/C readout |
| `src/foldabs_repro/tmp_prediction/interfaces.py` | 提取共享界面及pool顺序常量；原mean函数保留 |
| `src/foldabs_repro/tmp_prediction/model.py` | 新增`pooling_mode`，默认A；checkpoint配置保存mode |
| `scripts/train_tmp_specificity.py` | 新增`--pooling-mode`；保存seed、manifest hash、git SHA、参数量及`run_config.json` |
| `scripts/run_tmp_pooling_ablation.py` | 同一manifest/缓存/P1配置下顺序执行多mode、多seed；预检数量与缓存文件，禁止覆盖输出目录 |
| `scripts/summarize_tmp_pooling_ablation.py` | 汇总每run和跨seed指标，以及相同seed的B−A/C−A差值 |
| `tests/test_tmp_pooling.py` | 初始mean等价、梯度/mask、方向、排列不变性、旧配置与checkpoint兼容的待执行测试 |

旧checkpoint的`model_config`没有`pooling_mode`时自动使用A；A的pooling没有参数或state keys，保持旧权重strict-load兼容。已有predict脚本通过checkpoint的`model_config`恢复模式，无须额外预测flag。

## 2. 小数据集固定协议

使用已有484条Level 2 manifest：train/validation/test = **388/48/48**，沿用原始划分和`protenix_trunk_s_z_pdist_v1`缓存，不重新生成负例或提取特征。

一键脚本默认：P1、seeds=42/43/44、2000 optimizer steps、LR=3e-4、weight decay=1e-4、batch=1、梯度累积=3、每100步验证、hidden=512、dropout=0.1；validation BCE最小选checkpoint，阈值保持0.5。这是本次统一比较配置，并非声称逐项恢复历史checkpoint超参数；如果原checkpoint的hidden/dropout不同，应在所有A/B/C中统一覆盖。

首次建议用`--modes A B`完成6个runs，再用新的输出目录跑A/B/C完整9个runs，或直接运行完整9个runs。一次完整运行顺序为A的3个seed、B的3个seed、C的3个seed。保持同一seed时公共权重初始化相同；query零初始化不消耗随机数。GPU kernel本身仍可能非完全确定。

## 3. 在Codex/服务器执行

先在已经安装项目、PyTorch、Protenix且能训练Stage 1的环境中切换到最新`tmp-specificity`。以下路径由你替换成实际的Level 2 manifest和缓存目录；不提供猜测的服务器路径。

```bash
export TMP_MANIFEST=/absolute/path/to/level2_manifest.csv
export TMP_FEATURES=/absolute/path/to/protenix_features
export TMP_POOL_RUNS=/absolute/path/to/new_pooling_abc_runs
```

只检查manifest、388/48/48数量、0/1标签及缓存是否齐全，打印命令，不训练、不写文件：

```bash
python scripts/run_tmp_pooling_ablation.py \
  --manifest "$TMP_MANIFEST" --features-dir "$TMP_FEATURES" \
  --output-root "$TMP_POOL_RUNS" --dry-run
```

完整A/B/C × 3 seeds，单GPU顺序执行：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_tmp_pooling_ablation.py \
  --manifest "$TMP_MANIFEST" --features-dir "$TMP_FEATURES" \
  --output-root "$TMP_POOL_RUNS"
```

只跑A/B时加`--modes A B`；更换Pairformer深度时加`--pairformer-blocks 0`或`4`并使用新输出目录，不能与P1结果混合汇总。脚本不实现断点续训，已存在的output-root会报错以保护结果。

单独跑某一个实验，例如B/seed42：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_tmp_specificity.py \
  --manifest "$TMP_MANIFEST" --features-dir "$TMP_FEATURES" \
  --output-dir "$TMP_POOL_RUNS/B_seed42" \
  --pooling-mode B --pairformer-blocks 1 --seed 42 \
  --max-steps 2000 --learning-rate 3e-4 --weight-decay 1e-4 \
  --gradient-accumulation 3 --validate-every 100 \
  --hidden-dim 512 --dropout 0.1
```

可在训练前执行必要验证（本次未运行）：

```bash
python -m pytest tests/test_tmp_interfaces.py tests/test_tmp_pooling.py -q
```

测试中的模型兼容检查需要Protenix；缺少依赖时pytest可能skip，不能将skip视为通过。

## 4. 输出与解读

每个`A_seed42/`等目录：`best_model.pt`、`train.log`、`metrics.json`、`predictions.csv`、`run_config.json`。根目录保存`ablation_config.json`；全部runs成功后自动生成`pooling_runs.csv`和`pooling_summary.json`。

汇总报告train/validation/test的BCE、AUROC、AUPRC、标准化partial AUROC（`auc_0_1`，FPR≤0.1）的跨seed均值及样本标准差；另报告相同seed下B−A、C−A的差值。BCE差值为负较好，其余为正较好。单seed的std为null，不解释为零波动。汇总会检查manifest hash、缓存路径、git SHA、超参数及公共model config一致性。

中途失败后可单独完成缺失run，再手动汇总：

```bash
python scripts/summarize_tmp_pooling_ablation.py --runs-root "$TMP_POOL_RUNS"
```

手动汇总只读取已有完成runs，需自行确认模式与seed齐全；不会自动推断缺失run的结果。缓存路径检查不能证明缓存内容从未改变，比较期间不要修改缓存。

用validation观察收益和过拟合，test只报告；不要根据48条test反复调整attention实现。小集结果只决定是否推进正式HITPH epitope-OOD实验，不足以宣称泛化改善。均值池化A应保留为长期对照。
