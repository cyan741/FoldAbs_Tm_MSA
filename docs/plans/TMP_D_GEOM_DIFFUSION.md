# FoldTMP: FoldAbS-aligned specificity features (2026-11-04)

## Paper-derived specificity architecture

The uploaded FoldAbS paper defines the specificity inputs as {s, z, d_geom, p_PAE} (main text §3.2–3.3; Appendix A.2; Algorithm S1). The geometry map is computed from predicted coordinates, d_geom = Dist(X_hat), by discretizing distances between representative atoms into 64 bins with a maximum distance of 22.0 Å.

For each token pair, the two pairwise signals are concatenated first:

    c_spec = concat(d_geom, p_pae)       # [N_token, N_token, 128]
    z_cond = z + PairCond(z, c_spec)

PairCond uses layer normalization, linear projection and two residual transition blocks. The conditioned z and single representation s are refined by a 4-block task-specific Pairformer. Cross-chain pair features and antibody/antigen single features are pooled. The scalar Protenix iPTM is embedded into 32 dimensions and concatenated with the pooled structural summary before the final classification MLP.

The paper's Appendix A.2 specifies 64 geometry bins, 64 PAE bins, 22.0 Å maximum geometry distance, 32 diffusion samples and 20 denoising steps, plus two PairCond transitions and four Pairformer blocks.

## Implementation mapping

- d_geom [N,N,64]: sampled coordinates → Protenix representative atoms (the distogram_rep_atom_mask) → pairwise Euclidean distances → 64 bins up to 22 Å. This avoids assuming a hard-coded atom name. The exact atom-selection rule is delegated to Protenix's own mask.
- p_pae [N,N,64]: Protenix confidence head PAE logits → softmax over bins → mean probability distribution over diffusion samples.
- iptm []: official Protenix calculate_iptm applied to each sample's PAE probabilities, then averaged over samples.
- d_geom is an ensemble histogram (mean of one-hot distance bins); p_pae and iptm are also averaged over the 32-sample mini-rollout. The paper specifies 32 samples but does not detail the ensemble reduction; mean aggregation is an explicit TMP implementation choice.
- Protenix trunk, diffusion denoiser and confidence head remain frozen. Only PairCond, Pairformer, pooling, the 32-D iPTM embedding and classifier are trained.
- The old trunk-only baseline remains available when --use-d-geom is off: it conditions on p_dist. When enabled, the specificity branch instead conditions on concatenated d_geom + p_pae and fuses iptm after pooling, matching the paper's feature topology.
- The paper says "representative atoms" but does not name the atom-selection mask in the described text. Using Protenix's own distogram_rep_atom_mask is an implementation alignment to the upstream model, not a claim that the paper explicitly names this key.

## Online BCE / Group

The flag is --use-d-geom (not --use-d-geomt). Defaults when enabled match the paper's mini-rollout: 32 samples × 20 denoising steps.

    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run \
      --standalone --nproc-per-node=2 scripts/train_mini_esm_online_ddp.py \
      --splits-dir /path/to/splits \
      --esm-cache-dir /path/to/esm_cache \
      --local-esm-cache /path/to/local_esm_cache \
      --output-dir /path/to/foldabs_aligned_bce \
      --loss-type bce --use-d-geom --diffusion-steps 20 \
      --diffusion-samples 32 --smoke-samples 12 --max-steps 1

For Group ranking, use --loss-type group_rank --group-negatives 2. Always use a new output directory when changing feature topology or diffusion configuration. Resume checks the feature flag, sample count and step count.

## Offline extraction / training / prediction

    python scripts/extract_protenix_embeddings.py \
      --model-name protenix_mini_esm_v0.5.0 --cycles 4 \
      --input-json /path/to/input.json --esm-cache-dir /path/to/esm_cache \
      --output-dir /path/to/foldabs_aligned_features --limit 1 \
      --use-d-geom --diffusion-steps 20 --diffusion-samples 32

    python scripts/train_tmp_specificity.py \
      --manifest /path/to/manifest.csv \
      --features-dir /path/to/foldabs_aligned_features \
      --output-dir /path/to/foldabs_aligned_offline_train \
      --use-d-geom

The feature cache stores d_geom, p_pae and scalar iptm, along with a configuration fingerprint. Geometry-mode training and prediction reject incomplete feature files. Trunk-only caches remain valid for the trunk-only baseline but cannot be used with a FoldAbS-aligned checkpoint.

## Validation

    python -m pytest -q tests/test_tmp_d_geom.py
    python -m pytest -q tests/test_protenix_extraction_resume.py tests/test_tmp_prediction.py

GPU checks still required on the target Protenix environment:
1. One-step online BCE and Group smoke tests with the 32×20 mini-rollout.
2. Offline extraction smoke test; verify d_geom and p_pae are [N,N,64], normalized and finite, and iptm is a finite scalar.
3. Verify the representative-atom mapping and confidence-head outputs for the actual four-chain TMP inputs.
4. Compare runtime, peak GPU memory and fixed-split validation metrics against the trunk-only p_dist baseline.

No GPU execution or predictive-performance gain is claimed by this code change. The FoldAbS paper also reports the geometry and PAE ablations; those should be reproduced separately if assessing feature contribution.
