#!/usr/bin/env bash
# Source from epoch1_worker.sh on each compute host.
accel_dir=/data/luyq/hitph_cycle4_worker/accel
export PYTHONPATH="$accel_dir/cueq_0_5_overlay${PYTHONPATH:+:$PYTHONPATH}"
export TRITON_CACHE_DIR="$accel_dir/triton_cache_0_5"
export CUEQ_TRITON_CACHE_DIR="$accel_dir/cueq_tuning_0_5"
mkdir -p "$TRITON_CACHE_DIR" "$CUEQ_TRITON_CACHE_DIR"
