#!/usr/bin/env bash
# tang3的家目录下面缓存空间不够，缓存先存放到nas
# FEATURE_CACHE_PATH="/NAS/luyq/TMP_prediction/tmp_stage1_sanity/level1/hitph_level1_features"
python ../scripts/extract_protenix_embeddings.py \
  --input-json /NAS/luyq/TMP_prediction/protenix_embeddings/immrep25/immrep2025_tmp.json \
  --output-dir /NAS/luyq/TMP_prediction/protenix_embeddings/immrep25/immrep2025_tmp_features \
  --cycles 1 \
  --limit 0 \
  --gpu-ids all \
  --resume