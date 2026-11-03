#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export OUTPUT_DIR=${OUTPUT_DIR:-/NAS/luyq/foldabs_group_rank_20261029/lr_0.0001_bs_2_wd_0.01_drop_0.1_group_rank_k_2}
export GPU_IDS=${GPU_IDS:-0,1}
export MIN_FREE_MIB=${MIN_FREE_MIB:-30000}
exec bash "$ROOT/scripts/run_immrep25_a100.sh" \
    --loss-type group_rank --group-negatives 2 --group-margin 1.0 \
    --group-accumulation-steps 1 --no-activation-checkpoint \
    --cpu-workers 1 --chunk-rounds 240 "$@"
