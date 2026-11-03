#!/usr/bin/env bash
set -euo pipefail
cd /home/luyq/FoldAbS_Tm_MSA
source /data/luyq/hitph_cycle4_worker/accel/enable_cueq_cu11.sh
export PROTENIX_ROOT_DIR=/NAS/luyq/TMP_prediction/protenix_assets/proteinix/protenix_assets
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
exec /NAS/luyq/miniconda3/envs/foldtmp/bin/python -u scripts/resume_mini_esm_accelerated.py \
  --profile /NAS/luyq/TMP_prediction/protenix_embeddings/hitph_v1/mini_esm_v0.5.0_c4_no_msa \
  --local-root /data/luyq/mini_esm_accelerated_20261017 \
  --gpu-ids 0,1,2,3,4,5,6,7 --chunk-size 256
