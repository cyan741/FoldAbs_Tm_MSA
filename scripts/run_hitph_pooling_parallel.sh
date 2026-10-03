#!/usr/bin/env bash
set -euo pipefail

# Nine matched P1 trials, allocated by free GPU memory regardless of utilization.
# BF16: budget 5 GiB per trial, reserve 4 GiB per GPU; maximum three trials/GPU.
REPO_ROOT=/home/luyq/FoldAbS_Tm_MSA
FOLDTMP_PYTHON=/NAS/luyq/miniconda3/envs/foldtmp/bin/python
DATA_ROOT=/NAS/luyq/TMP_prediction/humanI_uns_hitph_embeddings
RUN_ROOT=${1:-/NAS/luyq/TMP_prediction/pooling_abc_hitph_es_$(date +%Y%m%d_%H%M%S)}

cd "$REPO_ROOT"
exec "$FOLDTMP_PYTHON" -u scripts/run_tmp_pooling_ablation.py \
  --manifest "$DATA_ROOT/manifest.csv" \
  --features-dir "$DATA_ROOT/features" \
  --output-root "$RUN_ROOT" \
  --modes A B C --seeds 42 43 44 \
  --parallel-runs 9 --gpu-ids auto \
  --gpu-memory-per-run-mib 5120 --gpu-reserve-mib 4096 \
  --pairformer-blocks 1 --max-steps 2000 \
  --learning-rate 3e-4 --weight-decay 1e-4 \
  --gradient-accumulation 3 \
  --early_stop --early_stop_patience 5 \
  --lr-scheduler plateau --lr-scheduler-factor 0.5 \
  --lr-scheduler-patience 2 --min-lr 1e-6 --precision bf16 \
  --hidden-dim 512 --dropout 0.1
