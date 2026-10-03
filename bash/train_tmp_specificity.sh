#!/usr/bin/env bash
# FEATURE_CACHE_PATH="/NAS/luyq/TMP_prediction/tmp_stage1_sanity/level1/hitph_level1_features"
# PREFIX="/home/luyq/FoldAbS_Tm_MSA/work/tmp_stage1_sanity/level1"
# python ../scripts/train_tmp_specificity.py \
#   --manifest ${PREFIX}/hitph_level1_manifest.csv \
#   --features-dir ${FEATURE_CACHE_PATH} \
#   --output-dir ../runs/tmp_stage1_level1 \
#   --max-steps 1000 \
#   --learning-rate 1e-3 \
#   --weight-decay 0 \
#   --gradient-accumulation 1 \
#   --validate-every 50 \
#   --log-every 10 \
#   --train-loss-window 20 \
#   --pairformer-blocks 0 \
#   --hidden-dim 256 \
#   --dropout 0

LR=3e-4
BS=1 # batch size默认是1，gradient accumulation=batch size=1*1=1    
WD=0
DROP=0
python ../scripts/train_tmp_specificity.py \
  --manifest /NAS/luyq/FoldAbS_Tm_MSA_level2/work/hitph_level2_manifest.csv \
  --features-dir /NAS/luyq/FoldAbS_Tm_MSA_level2/work/hitph_level2_features \
  --output-dir /NAS/luyq/FoldAbS_Tm_MSA_level2/runs/tmp_stage1_level2_p0/lr_${LR}_bs_${BS}_wd_${WD}_drop_${DROP} \
  --max-steps 2000 \
  --learning-rate ${LR} \
  --weight-decay ${WD} \
  --gradient-accumulation ${BS} \
  --validate-every 100 \
  --log-every 20 \
  --train-loss-window 50 \
  --pairformer-blocks 0 \
  --hidden-dim 256 \
  --dropout ${DROP} \
  --gpu 0 \
  --auto-pos-weight &


LR=3e-4
BS=1 # batch size默认是1，gradient accumulation=batch size=1*1=1    
WD=0
DROP=0
python ../scripts/train_tmp_specificity.py \
  --manifest /NAS/luyq/FoldAbS_Tm_MSA_level2/work/hitph_level2_manifest.csv \
  --features-dir /NAS/luyq/FoldAbS_Tm_MSA_level2/work/hitph_level2_features \
  --output-dir /NAS/luyq/FoldAbS_Tm_MSA_level2/runs/tmp_stage1_level2_p1/lr_${LR}_bs_${BS}_wd_${WD}_drop_${DROP} \
  --max-steps 2000 \
  --learning-rate ${LR} \
  --weight-decay ${WD} \
  --gradient-accumulation ${BS} \
  --validate-every 100 \
  --log-every 20 \
  --train-loss-window 50 \
  --pairformer-blocks 1 \
  --hidden-dim 256 \
  --dropout ${DROP} \
  --gpu 1 \
  --auto-pos-weight 