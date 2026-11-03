# HiTPH mini-ESM online DDP training (2026-10-26)

## Restart update (2026-10-27)

The initial run failed with CUDA illegal memory access. After an all-valid
triangle-attention mask workaround, full-head memcheck and 65 regression tests
passed, followed by a two-rank checkpoint replay from step 2900 to 3000.
Formal training resumed at 20:02:30 on Tang2 GPUs 3,5 using the validated
step-3000 continuation. The original checkpoint is retained under
`train/pre_mask_fix_20261027/`.

Current log: `train_resume_maskfix_20261027.log`; current tmux session:
`mini_esm_ddp_train_maskfix_20261027`. The older `train.log` and
`train_exit_code` describe the original failed run. See
`docs/HITPH_MINI_ESM_MASK_FIX_20261027.md` for evidence and operational details.

## Entry points and data flow

- Training: `scripts/train_mini_esm_online_ddp.py`.
- Single-node launcher: `scripts/run_mini_esm_online_ddp.sh`.
- Tests: `tests/test_mini_esm_online_ddp.py`.
- Each rank reads existing ESM2-3B single-chain tensors, executes frozen
  `protenix_mini_esm_v0.5.0` with four recycling cycles, and immediately trains a
  four-block Pairformer/C-pooling classifier on the same GPU.
- Encoder and head use BF16 autocast and cuEquivariance. Joint features use FP16
  tensors, converted to FP32 head inputs under autocast. Activation checkpointing
  is disabled. Neither ESM2-3B weights nor diffusion inference are used.
- Both positive and negative joint features are computed online. No joint
  embedding files are written. The local cache contains only validated existing
  single-chain ESM tensors, staged on demand from the read-only source cache.
- Two spawned CPU workers per rank prepare features. Chunk boundaries bound the
  prefetch queue; first-epoch cold staging and worker startup add overhead beyond
  the previously measured warm single-GPU timing.

## Training semantics

The existing `PeptideMismatchSampling` implementation supplies fresh epoch-specific
negatives (seed 42, 1:2 positive:negative ratio), original natural TCR pairs,
split-specific peptide pools and the all-positive exclusion blacklist. Validation
and test inputs remain the delivered fixed splits.

Every epoch is shuffled using seed + epoch. Each GPU processes one complex at a
time. DDP synchronizes only the head; the encoder is frozen. Gradient accumulation
defaults to three, so two GPUs give an effective batch size of six. Incomplete
tail windows use zero-weight placeholders and the actual global sample count;
they neither drop valid records nor count padding as extra samples.

Validation runs online across disjoint rank shards after each complete epoch.
Predictions are gathered before global AUROC/AUPRC and macro-pMHC metrics are
calculated. Validation BCE drives ReduceLROnPlateau (factor 0.5, patience 2,
minimum LR 1e-6), best-checkpoint selection, and early stopping (patience 5).
The maximum is ten epochs by default. Test runs once using the validation-best
head after training stops. AdamW starts at LR 1e-4, weight decay 0.01.

`last.pt` is saved atomically after the first optimizer step, every 100 optimizer
steps, and each epoch. It includes model, optimizer, scheduler, per-rank RNG,
epoch, consumed round, accumulated epoch loss and sampling-input fingerprints.
Resume requires the same world size, sampling inputs and optimization settings.
Only consumed optimizer windows advance the cursor; CPU-prefetched records are
regenerated. Use the original output directory so `best.pt` remains available.

Protenix's internal `DIST_WRAPPER.world_size` is deliberately one: each rank owns
its independent frozen inference runner and its already-sharded input list.
The training script separately initializes the real NCCL process group and wraps
the head in PyTorch DDP. A Protenix log saying world size one does not mean that
the head is trained independently. Training logs report the actual world size
and periodically compare rank parameter checksums.

## Initial resource selection

At inspection, no candidate host had eight GPUs meeting the full-model memory
budget. Tang3's largest free memory was approximately 22 GiB. Tang1 GPUs 2/6 had
approximately 30/35 GiB free, while Tang2 GPUs 3/5 each had approximately 35 GiB.
Tang2 GPUs 3 and 5 were selected. No existing process was stopped.

The initial real two-rank smoke run completed 48 samples (16 positive, 32
negative), including the longest positive and negative in the epoch input.
All eight optimizer steps had finite gradient norms and equal rank parameter
checksums. Per-rank peak PyTorch reserved memory was 22,054 / 18,224 MiB. These
measurements exclude other processes and do not guarantee memory availability
throughout a shared-server run.

The interrupted run saved at optimizer step 4 / round 12 and resumed through
step 8 / round 24, consuming exactly 48 valid samples. Against an independent
uninterrupted run, the final accumulated BCE was 0.66305651 versus 0.66305170,
model relative L2 difference was 0.00012282, and maximum absolute parameter
difference was 0.00060713. Independent accelerated runs already differed slightly
at step one; this verifies successful recovery and close numerical agreement,
not bitwise reproducibility. Online evaluation also completed on six fixed
validation and six fixed test records. These tiny smoke subsets are functional
checks, not model-quality estimates.

The targeted test command passed **32 tests**:

```bash
PYTHONPATH=src /NAS/luyq/miniconda3/envs/foldtmp/bin/python -m pytest \
  tests/test_mini_esm_online_ddp.py tests/test_tmp_head_checkpoint.py \
  tests/test_mini_esm_cache.py -q
```

## Full training launch

Full training started on Tang2 at **2026-10-26 21:54:46 Asia/Shanghai**, in tmux
session `mini_esm_ddp_train_20261026`. Epoch one contains 238,395 valid samples:
79,465 positives and 158,930 dynamic negatives. This run starts from a fresh head;
it does not use smoke-trained weights.

The first checkpoint was saved after optimizer step one. The live process was
then verified at step ten / 60 valid samples, with finite gradient norms and equal
rank parameter checksums. At that observation, rank peak reserved memory was
22,062 / 22,878 MiB. This is evidence of active full training, not completion of
an epoch or of model-quality validation.

Runtime root: `/data/luyq/mini_esm_online_ddp_20261026` on Tang2. Main log:
`train.log`; artifacts: `train/`; latest recovery checkpoint: `train/last.pt`.
`train/progress.json` describes the most recently saved checkpoint, not every
live step. The main log reports every ten optimizer steps. `train_exit_code`
appears only when the launcher exits; zero alone does not replace checking the
appropriate completion marker.

```bash
ssh LDS_Tang2 'tail -n 30 /data/luyq/mini_esm_online_ddp_20261026/train.log'
ssh LDS_Tang2 'cat /data/luyq/mini_esm_online_ddp_20261026/train/progress.json'
ssh -t LDS_Tang2 'tmux attach -t mini_esm_ddp_train_20261026'
```

## Tang2 paths and commands

The local source of truth is `/home/luyq/FoldAbS_Tm_MSA/scripts`. Tang2 uses an
isolated source snapshot under `/data/luyq/mini_esm_online_ddp_20261026/repo`.
The remote Python environment is `/data/luyq/hitph_cycle4_worker/env`.

```bash
cd /data/luyq/mini_esm_online_ddp_20261026/repo
GPU_IDS=3,5 bash scripts/run_mini_esm_online_ddp.sh \
  --splits-dir /NAS/luyq/TMP_prediction/data/hitph_l4_v1/splits \
  --esm-cache-dir /data/luyq/hitph_cycle4_worker/foldtmp/mini_esm_v0.5.0_c4_no_msa/esm_cache \
  --local-esm-cache /data/luyq/mini_esm_online_ddp_20261026/esm_cache \
  --output-dir /data/luyq/mini_esm_online_ddp_20261026/train \
  --epochs 10 --gradient-accumulation 3 --chunk-rounds 96 \
  --checkpoint-every 100 --log-every 10
```

To resume, repeat the identical command and add
`--resume /data/luyq/mini_esm_online_ddp_20261026/train/last.pt`.
Do not run a second copy against a live output directory. Recheck free GPU
memory before any launch; changing `GPU_IDS` does not make occupied GPUs safe.
The launcher rejects duplicate IDs and refuses GPUs below 30,000 MiB free by
default (`MIN_FREE_MIB` overrides this explicit launch-time check).
For an eight-GPU host, set all eight verified GPU IDs; the launcher derives the
process count from the list. A checkpoint created with two ranks cannot be
resumed with eight by this implementation.

Smoke mode uses `--smoke-samples 48`; `--smoke-evaluation-samples 6` additionally
checks six fixed records from each evaluation split. `--max-steps` provides an exact optimizer
step limit for interruption/recovery checks; it does not imply a complete epoch.
Progress is recorded in `progress.json`; `completion.json` only appears after
the requested smoke or full train/validation/test procedure actually completes.
