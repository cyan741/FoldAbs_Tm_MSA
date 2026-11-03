#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RUN_ROOT=${RUN_ROOT:-/NAS/luyq/foldabs_immrep25_20261029}
export PYTHON=${PYTHON:-/root/foldabs_immrep25_20261029/env/bin/python}
export ACCEL_ENV=${ACCEL_ENV:-/dev/null}
export PROTENIX_ROOT_DIR=${PROTENIX_ROOT_DIR:-$RUN_ROOT/assets}
export GPU_IDS=${GPU_IDS:-0,1}
cd "$ROOT"
exec bash scripts/run_mini_esm_online_ddp.sh \
    --splits-dir "$RUN_ROOT/splits" \
    --esm-cache-dir "$RUN_ROOT/esm_cache" \
    --local-esm-cache "${LOCAL_ESM_CACHE:-/root/foldabs_immrep25_20261029/esm_cache}" \
    --output-dir "${OUTPUT_DIR:-$RUN_ROOT/train}" \
    --epochs 10 --gradient-accumulation 3 --chunk-rounds 960 --cpu-workers 4 \
    --negatives-per-positive 2 --negative-sampling-seed 42 --seed 42 \
    --checkpoint-every 100 --log-every 10 "$@"
