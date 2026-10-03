#!/usr/bin/env bash
set -euo pipefail

# Resume only missing IMMREP25 caches. Existing valid .pt files are loaded for
# validation/index metadata but are never rewritten.
REPO_ROOT="/home/luyq/FoldAbS_Tm_MSA"
FOLDTMP_PYTHON="/NAS/luyq/miniconda3/envs/foldtmp/bin/python"
INPUT_JSON="/NAS/luyq/TMP_prediction/protenix_embeddings/immrep25/immrep2025_tmp.json"
OUTPUT_DIR="/NAS/luyq/TMP_prediction/protenix_embeddings/immrep25/immrep2025_tmp_features"

export PROTENIX_ROOT_DIR="${REPO_ROOT}/.foldtmp-run/protenix_assets"

cd "${REPO_ROOT}"
exec "${FOLDTMP_PYTHON}" scripts/extract_protenix_embeddings.py \
  --input-json "${INPUT_JSON}" \
  --output-dir "${OUTPUT_DIR}" \
  --cycles 1 \
  --limit 0 \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --resume "$@"
