# FoldTMP: optional frozen diffusion geometry `d_geomt` (2026-11-04)

## Contract

This branch adds an **optional ablation** on top of the existing
`s / z / p_dist` trunk-only model. The frozen Protenix-v1.0.0
diffusion sampler consumes the *same* `s_inputs, s, z` and input-feature
dictionary used by the trunk. No new structure or ESM checkpoint is trained.

- `p_dist [N,N,64]`: softmax of the **trunk distogram head**; unchanged.
- `d_geomt [N,N,64]`: Cα pair-distance bins from **sampled diffusion
  coordinates**, averaged over `diffusion_samples`. This branch calls
  the feature `d_geomt` as requested; it corresponds to the geometry
  feature denoted `d_geom` in the FoldAbS audit.
- Bin implementation: 64 categorical bins with 63 thresholds equally
  spaced from >0 to 22 Å; final bin contains distances >22 Å. The
  atom choice (Cα), edges, and ensemble averaging are **TMP
  implementation choices**, not values independently confirmed from
  the unpublished FoldAbS code.
- All four chains must be proteins, every token must have exactly one
  Cα atom, and geometry must be finite and normalized. Missing features
  raise an exception; there is **no zero filling**.
- Multiple independent structures are summarized as a probability
  histogram, not by averaging 3D coordinates across unaligned samples.
- The model condition is
  `z + Linear(LN(p_dist)) + Linear(LN(d_geomt))`, followed by the
  same two residual transitions and 4-block Pairformer. The new
  projection is trainable; Protenix trunk and diffusion remain frozen.

**This is a geometry-only ablation, not a full FoldAbS specificity
reproduction**. It does not implement PAE probabilities, ipTM, confidence
head, or the paper's complete sample-selection protocol. The earlier
audit lists those missing items in `docs/plans/TMP_SPECIFICITY_PAPER_ALIGNMENT.md`.

## Online BCE / Group (no pair embedding disk cache)

Use the existing unified CLI. The original default is unchanged
(`--use-d-geomt` is **off**). Use a **new output directory**:

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run \
  --standalone --nproc-per-node=2 scripts/train_mini_esm_online_ddp.py \
  --splits-dir /path/to/splits \
  --esm-cache-dir /path/to/esm_cache \
  --local-esm-cache /path/to/local_esm_cache \
  --output-dir /path/to/geometry_bce_smoke \
  --loss-type bce --use-d-geomt --diffusion-steps 20 \
  --diffusion-samples 1 --smoke-samples 12 --max-steps 1
```

For Group replace `--loss-type bce` with
`--loss-type group_rank --group-negatives 2`, and choose a new directory.
The feature is computed by `OnlineFeatures.infer`, then passed to
the head in both BCE and Group mode.

`--diffusion-samples 32 --diffusion-steps 20` is much more expensive
than `1 × 20`. Benchmark **speed and peak GPU memory first**,
especially when running on multiple encoder processes. Lower-budget
experiments must report their sample/step counts.

Resume is only allowed for matching geometry enablement, steps, and
samples; different feature configurations must not reuse a checkpoint.

## Offline extraction / training / prediction

```bash
python scripts/extract_protenix_embeddings.py \
  --model-name protenix_mini_esm_v0.5.0 --cycles 4 \
  --input-json /path/to/input.json --esm-cache-dir /path/to/esm_cache \
  --output-dir /path/to/geometry_features --limit 1 \
  --use-d-geomt --diffusion-steps 20 --diffusion-samples 1

python scripts/train_tmp_specificity.py \
  --manifest /path/to/manifest.csv \
  --features-dir /path/to/geometry_features \
  --output-dir /path/to/geometry_offline_train \
  --use-d-geomt
```

Geometry caches include `d_geomt` and a `diffusion_geometry`
configuration fingerprint. The `--resume` path validates all tensors
and rejects caches with mismatched geometry settings. Default trunk-only
cache files and their extraction profiles remain unchanged.

Offline inference automatically reads the checkpoint's
`model_config.use_d_geomt` flag. Prediction from a geometry head
requires feature files with `d_geomt`.

## Verification

CPU/shape checks:
```bash
python -m pytest -q tests/test_tmp_d_geomt.py
python -m pytest -q tests/test_protenix_extraction_resume.py tests/test_tmp_prediction.py
```

GPU checks still required on the target Protenix environment:

1. Run the online BCE smoke (one update) and confirm finite loss and
   `d_geomt` is `[N,N,64]`, symmetric and normalized.
2. Repeat the same geometry input twice with the same seed; compare
   outputs and the logit. Verify no accidental global RNG reset.
3. Compare enabled versus disabled GPU memory, extraction throughput,
   and validation behavior on **the same fixed inputs**.
4. Test Group smoke, checkpoint/resume and offline geometry extraction
   with the installed Protenix-1.0.0 checkpoint.

No real GPU diffusion experiment or full training run was executed
while preparing this branch. Do not report model performance gains
until the GPU/regression checks and fixed-split comparisons pass.
