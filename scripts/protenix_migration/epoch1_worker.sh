#!/usr/bin/env bash
# Cache disjoint epoch-1 negative shards on one GPU without touching permanent workers.
set -euo pipefail

if (( $# != 2 )); then
    echo "Usage: $0 GPU_ID SHARD_CSV" >&2
    exit 2
fi

gpu_id=$1
shard_csv=$2
base=/data/luyq/hitph_cycle4_worker
input=$base/job/input/train_negatives_epoch_0001.json
extractor=$base/job/bin/extract_protenix_embeddings.py
features=${HITPH_FEATURES_DIR:?Set HITPH_FEATURES_DIR to the shared Foldtmp features directory}
python=${HITPH_PYTHON:?Set HITPH_PYTHON to the installed Protenix Python}
work=$base/epoch1_work_32
mkdir -p "$work" "$features"

export PROTENIX_ROOT_DIR=$base/job/assets
export LAYERNORM_TYPE=torch OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTHONPATH="$base/job/bin${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES=$gpu_id

kernel_args=()
if [[ ${HITPH_ACCELERATED:-0} == 1 ]]; then
    source "$base/accel/enable_cueq_cu11.sh"
    "$python" -c 'import cuequivariance_torch' >/dev/null
    kernel_args=(--trimul-kernel cuequivariance --triatt-kernel cuequivariance)
fi

IFS=, read -r -a shards <<< "$shard_csv"
for shard in "${shards[@]}"; do
    if (( shard < 0 || shard >= 32 )); then
        echo "Invalid epoch-1 shard: $shard" >&2
        exit 2
    fi
    suffix=$(printf 'shard_%02d_of_32' "$shard")
    if [[ -f "$features/index.$suffix.json" && -f "$features/extraction_stats.$suffix.json" ]]; then
        echo "Epoch-1 $suffix already complete; skipping"
        continue
    fi
    echo "Starting epoch-1 $suffix on GPU $gpu_id accelerated=${HITPH_ACCELERATED:-0} at $(date -Is)"
    "$python" -u "$extractor" \
        --input-json "$input" --output-dir "$features" --work-dir "$work" \
        --cycles 4 --limit 0 --resume --num-shards 32 --shard-index "$shard" \
        "${kernel_args[@]}" \
        > "$work/$suffix.log" 2>&1
    echo "Completed epoch-1 $suffix at $(date -Is)"
done
