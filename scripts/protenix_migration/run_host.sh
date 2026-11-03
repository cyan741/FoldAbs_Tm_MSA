#!/usr/bin/env bash
# Run a consecutive range of globally unique Protenix shards on this host.
set -euo pipefail

if (( $# != 6 )); then
    echo "Usage: $0 ENV_PYTHON GPU_CSV TOTAL_SHARDS FIRST_SHARD SHARED_FEATURES_DIR LOCAL_WORK_DIR" >&2
    exit 2
fi

env_python=$1
gpu_csv=$2
total_shards=$3
first_shard=$4
features_dir=$5
work_dir=$6
bundle_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

IFS=, read -r -a gpu_ids <<< "$gpu_csv"
if (( total_shards < 2 || first_shard < 0 || first_shard + ${#gpu_ids[@]} > total_shards )); then
    echo "Invalid global shard assignment" >&2
    exit 2
fi
if [[ ! -x "$env_python" || ! -f "$bundle_dir/input/permanent.json" ]]; then
    echo "Missing Python environment or input JSON" >&2
    exit 2
fi
if ! (cd "$bundle_dir" && sha256sum -c SHA256SUMS >/dev/null); then
    echo "Migration bundle checksum validation failed" >&2
    exit 2
fi

mkdir -p "$features_dir" "$work_dir"
export PROTENIX_ROOT_DIR="$bundle_dir/assets"
export LAYERNORM_TYPE=torch
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONPATH="$bundle_dir/bin${PYTHONPATH:+:$PYTHONPATH}"

pids=()
for index in "${!gpu_ids[@]}"; do
    shard=$((first_shard + index))
    gpu_id=${gpu_ids[$index]}
    log="$work_dir/shard_${shard}_of_${total_shards}.log"
    echo "Starting global shard $shard/$total_shards on GPU $gpu_id; log: $log"
    CUDA_VISIBLE_DEVICES="$gpu_id" "$env_python" -u \
        "$bundle_dir/bin/extract_protenix_embeddings.py" \
        --input-json "$bundle_dir/input/permanent.json" \
        --output-dir "$features_dir" \
        --work-dir "$work_dir" \
        --cycles 4 --limit 0 --resume \
        --num-shards "$total_shards" --shard-index "$shard" \
        > "$log" 2>&1 &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    wait "$pid" || status=1
done
exit "$status"
