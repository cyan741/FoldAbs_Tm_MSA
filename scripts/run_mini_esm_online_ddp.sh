#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
PYTHON=${PYTHON:-/data/luyq/hitph_cycle4_worker/env/bin/python}
ACCEL_ENV=${ACCEL_ENV:-/data/luyq/hitph_cycle4_worker/accel/enable_cueq_cu11.sh}
source "$ACCEL_ENV"
export PROTENIX_ROOT_DIR=${PROTENIX_ROOT_DIR:-/NAS/luyq/TMP_prediction/protenix_assets/proteinix/protenix_assets}
export LAYERNORM_TYPE=torch OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${GPU_IDS:?Set GPU_IDS to the verified free GPU IDs}
IFS=',' read -r -a DEVICES <<< "$CUDA_VISIBLE_DEVICES"
declare -A SEEN_DEVICES=()
for GPU in "${DEVICES[@]}"; do
    if [[ ! "$GPU" =~ ^[0-9]+$ || -n "${SEEN_DEVICES[$GPU]:-}" ]]; then
        echo "GPU_IDS must contain unique nonnegative integer indices" >&2
        exit 1
    fi
    SEEN_DEVICES[$GPU]=1
    FREE_MIB=$(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits)
    if (( FREE_MIB < ${MIN_FREE_MIB:-30000} )); then
        echo "GPU $GPU has only $FREE_MIB MiB free; refusing to launch" >&2
        exit 1
    fi
done
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
exec "$PYTHON" -u -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node="${#DEVICES[@]}" \
    scripts/train_mini_esm_online_ddp.py "$@"
