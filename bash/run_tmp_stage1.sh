#!/usr/bin/env bash
# Run TMP Stage 1 sanity checks. Feature extraction is sharded across GPUs;
# training remains single-GPU because the current variable-length loader uses
# batch size 1 and has no distributed-training implementation.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONDA_ENV="${CONDA_ENV:-foldtmp}"
GPU_IDS_CSV="${GPU_IDS:-4,6,7}"
TRAIN_GPU="${TRAIN_GPU:-${GPU_IDS_CSV%%,*}}"
RUN_LEVEL2="${RUN_LEVEL2:-1}"
LEVEL1_BCE_THRESHOLD="${LEVEL1_BCE_THRESHOLD:-0.20}"

IFS=',' read -r -a GPU_IDS_ARRAY <<< "$GPU_IDS_CSV"
NUM_SHARDS="${#GPU_IDS_ARRAY[@]}"
if (( NUM_SHARDS < 1 )); then
  echo "GPU_IDS must contain at least one GPU index" >&2
  exit 2
fi

PYTHON=(conda run --no-capture-output -n "$CONDA_ENV" python)
STAGE_ROOT="$REPO_ROOT/work/tmp_stage1_sanity"
RUN_ROOT="$REPO_ROOT/runs/tmp_stage1_sanity"
CANONICAL_CSV="${CANONICAL_CSV:-$STAGE_ROOT/hitph_tmp.csv}"
SOURCE_CSV="$REPO_ROOT/data/human_I_unseen_hitph.csv"
HLA_MAP="$REPO_ROOT/data/Hi-TpH-hla_allele2seq.json"
HLA_ALIASES="$REPO_ROOT/data/Hi-TpH-hla_aliases.json"

mkdir -p "$STAGE_ROOT/logs" "$RUN_ROOT"

required_gb=3
if [[ "$RUN_LEVEL2" == "1" ]]; then
  required_gb=20
fi
available_kb="$(df -Pk "$REPO_ROOT" | awk 'NR==2 {print $4}')"
required_kb=$((required_gb * 1024 * 1024))
if (( available_kb < required_kb )); then
  echo "ERROR: insufficient disk space on $(df -P "$REPO_ROOT" | awk 'NR==2 {print $1}')." >&2
  echo "Available: $((available_kb / 1024 / 1024)) GiB; required: ${required_gb} GiB." >&2
  echo "Free disk space, then rerun this script. No training was started." >&2
  exit 3
fi

echo "[preflight] conda environment: $CONDA_ENV"
"${PYTHON[@]}" -c 'import torch, protenix; print(torch.__version__, torch.cuda.is_available())'

if [[ ! -f "$CANONICAL_CSV" ]]; then
  "${PYTHON[@]}" scripts/make_protenix_input/convert_hitph.py \
    --input-csv "$SOURCE_CSV" \
    --hla-map-json "$HLA_MAP" \
    --hla-alias-json "$HLA_ALIASES" \
    --output-csv "$CANONICAL_CSV" \
    --seed 42
fi

merge_extraction_metadata() {
  local feature_dir="$1"
  "${PYTHON[@]}" - "$feature_dir" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for path in sorted(root.glob("index.shard_*_of_*.json")):
    rows.extend(json.loads(path.read_text(encoding="utf-8")))
rows.sort(key=lambda row: str(row["sample_id"]))
(root / "index.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

parts = [
    json.loads(path.read_text(encoding="utf-8"))
    for path in sorted(root.glob("extraction_stats.shard_*_of_*.json"))
]
total_bytes = sum(int(path.stat().st_size) for path in root.glob("*.pt"))
summary = {
    "samples_total": len(list(root.glob("*.pt"))),
    "samples_generated_this_run": sum(int(x["samples_generated"]) for x in parts),
    "samples_reused_this_run": sum(int(x["samples_reused"]) for x in parts),
    "mean_n_token": sum(float(x["mean_n_token"]) * int(x["samples_total"]) for x in parts)
    / max(1, sum(int(x["samples_total"]) for x in parts)),
    "total_cache_bytes": total_bytes,
    "mean_cache_bytes": total_bytes / max(1, len(list(root.glob("*.pt")))),
    "generated_seconds": sum(float(x["generated_seconds"]) for x in parts),
    "generated_seconds_per_sample": sum(float(x["generated_seconds"]) for x in parts)
    / max(1, sum(int(x["samples_generated"]) for x in parts)),
    "max_shard_wall_seconds": max((float(x["total_wall_seconds"]) for x in parts), default=0.0),
    "max_gpu_peak_memory_bytes": max((int(x["gpu_peak_memory_bytes"] or 0) for x in parts), default=0),
    "num_shards": len(parts),
}
(root / "extraction_stats.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
PY
}

extract_features() {
  local input_json="$1"
  local feature_dir="$2"
  local level_name="$3"
  mkdir -p "$feature_dir"
  local pids=()
  for shard_index in "${!GPU_IDS_ARRAY[@]}"; do
    local gpu="${GPU_IDS_ARRAY[$shard_index]}"
    echo "[$level_name] shard $shard_index/$NUM_SHARDS -> GPU $gpu"
    CUDA_VISIBLE_DEVICES="$gpu" "${PYTHON[@]}" scripts/extract_protenix_embeddings.py \
      --input-json "$input_json" \
      --output-dir "$feature_dir" \
      --cycles 1 \
      --limit 0 \
      --resume \
      --num-shards "$NUM_SHARDS" \
      --shard-index "$shard_index" \
      >"$STAGE_ROOT/logs/${level_name}_extract_shard_${shard_index}.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
  merge_extraction_metadata "$feature_dir"
}

prepare_level() {
  local level_name="$1"
  local train_size="$2"
  local validation_size="$3"
  local test_size="$4"
  local level_dir="$STAGE_ROOT/$level_name"
  mkdir -p "$level_dir"
  "${PYTHON[@]}" scripts/make_protenix_input/make_stage1_subset.py \
    --input-csv "$CANONICAL_CSV" \
    --output-csv "$level_dir/hitph_${level_name}.csv" \
    --train-size "$train_size" \
    --validation-size "$validation_size" \
    --test-size "$test_size" \
    --seed 42
  "${PYTHON[@]}" scripts/make_protenix_input/make_tmp_protenix_input.py \
    --input-csv "$level_dir/hitph_${level_name}.csv" \
    --output-json "$level_dir/hitph_${level_name}.json"
  "${PYTHON[@]}" scripts/make_protenix_input/make_tmp_manifest.py \
    --input-csv "$level_dir/hitph_${level_name}.csv" \
    --output-csv "$level_dir/hitph_${level_name}_manifest.csv"
  extract_features \
    "$level_dir/hitph_${level_name}.json" \
    "$level_dir/features" \
    "$level_name"
}

reload_checkpoint() {
  local manifest="$1"
  local features="$2"
  local checkpoint="$3"
  local output_csv="$4"
  CUDA_VISIBLE_DEVICES="$TRAIN_GPU" "${PYTHON[@]}" scripts/predict_tmp_specificity.py \
    --manifest "$manifest" \
    --features-dir "$features" \
    --checkpoint "$checkpoint" \
    --output-csv "$output_csv" \
    --device cuda
}

echo "[level1] preparing 32/16/16 balanced subset"
prepare_level level1 32 16 16
LEVEL1_DIR="$STAGE_ROOT/level1"
LEVEL1_RUN="$RUN_ROOT/level1_p0"
CUDA_VISIBLE_DEVICES="$TRAIN_GPU" "${PYTHON[@]}" scripts/train_tmp_specificity.py \
  --manifest "$LEVEL1_DIR/hitph_level1_manifest.csv" \
  --features-dir "$LEVEL1_DIR/features" \
  --output-dir "$LEVEL1_RUN" \
  --max-steps 1000 \
  --learning-rate 1e-3 \
  --weight-decay 0 \
  --gradient-accumulation 1 \
  --validate-every 50 \
  --log-every 10 \
  --train-loss-window 20 \
  --pairformer-blocks 0 \
  --hidden-dim 256 \
  --dropout 0

reload_checkpoint \
  "$LEVEL1_DIR/hitph_level1_manifest.csv" \
  "$LEVEL1_DIR/features" \
  "$LEVEL1_RUN/best_model.pt" \
  "$LEVEL1_RUN/reloaded_predictions.csv"

"${PYTHON[@]}" - "$LEVEL1_RUN/train.log" "$LEVEL1_BCE_THRESHOLD" <<'PY'
import json
import sys

events = [json.loads(line) for line in open(sys.argv[1], encoding="utf-8")]
train = [row for row in events if row.get("event") == "train"]
if not train:
    raise SystemExit("Level 1 FAIL: train.log has no train events")
first = float(train[0]["train_bce_moving"])
last = float(train[-1]["train_bce_moving"])
threshold = float(sys.argv[2])
print({"level1_first_train_bce_moving": first, "level1_last_train_bce_moving": last})
if last >= first or last > threshold:
    raise SystemExit(
        f"Level 1 FAIL: moving BCE {first:.6f} -> {last:.6f}; required final <= {threshold:.6f}"
    )
print("Level 1 PASS")
PY

if [[ "$RUN_LEVEL2" != "1" ]]; then
  echo "RUN_LEVEL2=$RUN_LEVEL2; stopping after Level 1 PASS"
  exit 0
fi

echo "[level2] preparing 400/56/56 balanced subset"
prepare_level level2 400 56 56
LEVEL2_DIR="$STAGE_ROOT/level2"
for blocks in 0 1; do
  run_dir="$RUN_ROOT/level2_p${blocks}"
  CUDA_VISIBLE_DEVICES="$TRAIN_GPU" "${PYTHON[@]}" scripts/train_tmp_specificity.py \
    --manifest "$LEVEL2_DIR/hitph_level2_manifest.csv" \
    --features-dir "$LEVEL2_DIR/features" \
    --output-dir "$run_dir" \
    --max-steps 2000 \
    --learning-rate 3e-4 \
    --weight-decay 1e-4 \
    --gradient-accumulation 3 \
    --validate-every 100 \
    --log-every 20 \
    --train-loss-window 50 \
    --pairformer-blocks "$blocks" \
    --hidden-dim 256 \
    --dropout 0.1 \
    --auto-pos-weight
  reload_checkpoint \
    "$LEVEL2_DIR/hitph_level2_manifest.csv" \
    "$LEVEL2_DIR/features" \
    "$run_dir/best_model.pt" \
    "$run_dir/reloaded_predictions.csv"
done

echo "TMP Stage 1 completed successfully"
