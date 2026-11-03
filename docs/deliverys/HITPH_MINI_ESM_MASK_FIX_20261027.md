# mini-ESM online DDP: triangle-attention backward workaround

## Failure and scope

The original Tang2 run stopped with CUDA illegal memory access after the last
logged optimizer step 2940. Its intact recovery checkpoint is step 2900, epoch 1.
Its SHA256 is `43c63408cca67692bc575eb3f8d59bb889ec1c3c5dcafe9f4236553bcea2382f`.

In the CUDA 11 / cuEquivariance 0.5.1 environment, independent triangle attention
and the four-block production head reproduce 515 Compute Sanitizer errors with
`PYTORCH_NO_CUDA_MEMORY_CACHING=1`. The masked attention backward kernel reads
beyond the all-valid mask allocation at length 416. Enabling allocator caching
can conceal this overread from memcheck. The original asynchronous traceback at
`loss.item()` is not a reliable identification of the failing operation.

The current head processes one unpadded four-chain complex at a time. Although
the head passes `pair_mask=None`, Protenix materializes an all-valid boolean mask
before calling cuEquivariance. Omitting this redundant mask selects an unmasked
kernel path without changing which token pairs participate in attention.

## Implementation

- `src/foldabs_repro/tmp_prediction/triangle_attention.py` installs an idempotent
  wrapper around the Protenix cuEquivariance attention adapter.
- Only boolean masks whose elements are all true become `None`. Nontrivial
  masks, non-boolean masks, attention bias, and scale remain unchanged.
- A head configured with cuEquivariance attention installs the wrapper during
  construction. No installed package or CUDA binary is modified.
- This adapter is process-wide, so subsequent frozen encoder calls also use it.
- Parameters, state-dict keys, optimizer structure, and batch semantics do not
  change. Existing checkpoints can be resumed.
- Online run metadata records
  `triangle_attention_mask_policy=omit_all_valid_boolean_masks`.

This is a targeted workaround for the unpadded online workflow, not a repair of
cuEquivariance's general masked kernel. A future padded/masked batching path
still needs independent kernel validation.

## Validation evidence

1. Tang2 production head: two optimizer iterations for each of lengths
   413, 415, 416, 418, 419, 421, 423, 434; all 16 forward/backward/update cycles
   completed with finite gradient norms. Log:
   `/data/luyq/mini_esm_online_ddp_20261026/diagnostic_20261027/fixed_head_shapes.log`.
2. Tang1 full head, length 416, workaround enabled, CUDA allocation caching
   disabled: forward/backward/update completed, memcheck **0 errors**, exit 0.
   The earlier 300-second test was inconclusive; this extended run completed.
   Log: `/data/luyq/mini_esm_kernel_debug_20261027/full_head_none_extended12_memcheck.log`.
3. Torch versus patched cuEquivariance full-head comparison on a real cached
   complex, loading the step-2900 trained head: three AdamW updates, all 261
   gradient tensors finite and nonzero. Logit absolute differences were
   0, 0, 0.0078125; gradient relative L2 differences were approximately
   0.213%, 0.244%, 0.609%. This comparison disables dropout and uses activation
   recomputation in both paths; it is a numerical regression, not a bitwise
   equivalence claim for production training.
   Results on Tang3:
   `/data/luyq/mini_esm_mask_fix_20261027/trained_reference/head_kernel_gradient_check.json`.

4. Tang3 production code (automatic constructor installation, no probe-specific
   monkey patch), lengths 416 and 419, allocation caching disabled: both complete
   forward/backward/update cycles passed, memcheck **0 errors**, exit 0. Log:
   `/data/luyq/mini_esm_mask_fix_20261027/mini_esm_mask_fix_production_memcheck.log`.
5. CPU regression suite: **65 passed**, including mask preservation and numerical
   gradient equivalence, checkpointing, online DDP sharding/loss normalization,
   ESM cache validation, pooling, interfaces, and training/resume tests. The
   14 warnings are Matplotlib/Pyparsing deprecations. Log:
   `/data/luyq/mini_esm_mask_fix_20261027/mini_esm_mask_fix_pytest.log`.

6. Tang2 two-rank production-entry replay: restored a copy of step 2900 and
   completed 100 optimizer updates / 600 additional valid samples through step
   3000 (cursor 9000, cumulative 18,000 samples). Exit 0, finite gradient norms,
   matching rank parameter checksums at every logged step. Peak reserved GPU
   memory was 24,290 / 22,530 MiB. This crossed the previous failure region.
   Log: `/data/luyq/mini_esm_online_ddp_20261026/diagnostic_20261027/mask_fix_replay.log`.

## Formal restart

The formal run was relaunched on **2026-10-27 20:02:30 Asia/Shanghai**, Tang2 GPUs
3 and 5. Both had 35,949 MiB free before launch. The validated step-3000 replay
checkpoint was promoted rather than discarding its 100 valid training updates.
Its model configuration, input fingerprints, sampling, optimizer hyperparameters,
world size, cursor, RNG states, and cumulative sample count were checked before
promotion. All saved model and optimizer tensors were finite. Neither checkpoint
had completed an epoch or created a best-validation checkpoint.

- Runtime root: `/data/luyq/mini_esm_online_ddp_20261026` on Tang2.
- Original step-2900 checkpoint/config/progress backup: `train/pre_mask_fix_20261027/`.
- Promoted step-3000 SHA256:
  `ee79c743f69e893f6539bf4a859ac0157a22a87a53eebc9945bb54e3f3e3aa77`.
- Current log: `train_resume_maskfix_20261027.log`.
- Current exit marker: `train_resume_maskfix_20261027.exit` (created on exit only).
- tmux session: `mini_esm_ddp_train_maskfix_20261027`.
- Output/checkpoints remain in `train/`; the formal run has no `--max-steps` cap.
- The original `train.log` and `train_exit_code=1` are historical failure records,
  not the status of the restarted process.

At **20:04:33**, the formal run was verified at step **3010**, epoch 1,
18,060 cumulative valid samples, cumulative train BCE 0.3311469, finite gradient
norm 3.0960, and equal parameter checksums on both ranks. Both training workers
and the tmux session were alive, with no exit marker. This is an observed running
state, not a promise that subsequent epochs cannot encounter unrelated failures.

```bash
ssh LDS_Tang2 'tail -n 30 /data/luyq/mini_esm_online_ddp_20261026/train_resume_maskfix_20261027.log'
ssh LDS_Tang2 'cat /data/luyq/mini_esm_online_ddp_20261026/train/progress.json'
ssh -t LDS_Tang2 'tmux attach -t mini_esm_ddp_train_maskfix_20261027'
```

Successful regression checks and restart do not constitute completed training
or model-quality validation; epoch validation/test results are still pending.

## Operational constraints

Keep the two-rank configuration, gradient accumulation 3, effective batch 6,
four encoder cycles, four head blocks, and the existing optimizer/scheduler
states when resuming. Changing to eight ranks is not supported by this exact
checkpoint-resume implementation. Recheck GPU availability before launching.

Do not run two writers against the same output directory. Diagnostic replay
uses a separate copy of `last.pt`; retain the original checkpoint and logs
before promoting any validated continuation or restarting formal training.
