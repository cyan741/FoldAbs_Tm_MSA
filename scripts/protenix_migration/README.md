# HiTPH Protenix cycle-4/no-MSA migration bundle

This bundle contains the exact 96,670-record permanent input, the current
extraction script, the Protenix v1.0.0 checkpoint and chemical-component data.
`foldtmp_env.tar.gz` is a separate conda-pack archive of the Python 3.11.16
environment (PyTorch 2.7.1+cu118, Protenix 1.0.0). The extraction uses seed
101, bf16 inference, fp16 cache, and saves only `s`, `z`, `p_dist`,
`token_asym_id`, and `token_residue_index`.

## Transfer and validate

Copy `hitph_job.tar.gz`, `foldtmp_env.tar.gz`, and `SHA256SUMS` to each compute
host. Keep the private SSH key outside the bundle. On each host:

```bash
mkdir -p /scratch/hitph/job /scratch/hitph/env
tar -xzf hitph_job.tar.gz -C /scratch/hitph/job
tar -xzf foldtmp_env.tar.gz -C /scratch/hitph/env
PATH="/scratch/hitph/env/bin:$PATH" /scratch/hitph/env/bin/conda-unpack
cd /scratch/hitph/job
sha256sum -c SHA256SUMS
PROTENIX_ROOT_DIR="$PWD/assets" LAYERNORM_TYPE=torch \
  /scratch/hitph/env/bin/python -c 'import torch,protenix; print(torch.__version__,torch.cuda.is_available())'
```

`SHA256SUMS` in the job checks the code, input JSON and model assets. The
separate top-level `SHA256SUMS` checks both transfer archives.

Run a one-record smoke test on a free GPU before production. Use a local
output directory so the test cannot publish `index.json` to the shared cache:

```bash
cd /scratch/hitph/job
/scratch/hitph/env/bin/python -c 'import json; x=json.load(open("input/permanent.json")); json.dump(x[:1],open("/scratch/hitph/smoke.json","w"))'
CUDA_VISIBLE_DEVICES=0 PROTENIX_ROOT_DIR="$PWD/assets" LAYERNORM_TYPE=torch \
  OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH="$PWD/bin" \
  /scratch/hitph/env/bin/python bin/extract_protenix_embeddings.py \
  --input-json /scratch/hitph/smoke.json \
  --output-dir /scratch/hitph/smoke_features \
  --work-dir /scratch/hitph/smoke_work \
  --cycles 4 --limit 1
```

## Shared cache and exclusive partitions

Mount `Foldtmp@ai4ls.myds.me:/home/Foldtmp` on every compute host, using your
own SSH credentials. Point all hosts at the **same**
`embeddings_cycle4_nomsa_seed101_bf16_fp16_szp_protenixv1/features` directory.
The mount is output storage only; all preprocessing, model loading and GPU
inference run on compute hosts. Keep `LOCAL_WORK_DIR` on fast local storage.

Choose `TOTAL_SHARDS` as the total number of GPU worker processes across all
hosts. Give every worker a different global shard index from `0` through
`TOTAL_SHARDS-1`. The input records are partitioned by
`input_index % TOTAL_SHARDS`, so no two workers receive the same record.
Each worker uses atomic per-sample writes and `--resume` validates existing
files. For example, with 7 GPUs here and 8 GPUs on a second host:

```bash
# First host, indices 0..6
bash bin/run_host.sh /scratch/hitph/env/bin/python 0,1,3,4,5,6,7 15 0 \
  /path/to/Foldtmp/embeddings_cycle4_nomsa_seed101_bf16_fp16_szp_protenixv1/features \
  /scratch/hitph/work

# Second host, indices 7..14
bash bin/run_host.sh /scratch/hitph/env/bin/python 0,1,2,3,4,5,6,7 15 7 \
  /path/to/Foldtmp/embeddings_cycle4_nomsa_seed101_bf16_fp16_szp_protenixv1/features \
  /scratch/hitph/work
```

The GPU list may repeat an ID, such as `7,7`, to run two *different* shard
workers on one card. On the source A40, two workers used about 8 GiB total
and raised that card's combined output from roughly 4 to 5.2 records/minute
in a short trial. Both workers slowed individually, so benchmark each new
GPU type before adding more processes per card. Count every extra process in
`TOTAL_SHARDS` and give it its own global shard index.

The modulus-15 example is illustrative; the active permanent job below uses
modulus 25. Do not launch an overlapping partition of the same input.

After all global workers complete, publish the permanent index once:

```bash
/scratch/hitph/env/bin/python bin/merge_shards.py \
  --input-json input/permanent.json \
  --features-dir /path/to/Foldtmp/embeddings_cycle4_nomsa_seed101_bf16_fp16_szp_protenixv1/features \
  --num-shards 15
```

The merge validates every shard's index, exact record assignment, sample
identity, feature filename, and presence of every `.pt` before atomically
publishing `index.json` and `extraction_stats.json`. Keep total permanent plus
one epoch of negatives below the 20 TB storage limit.

## Active Tang3 / Tang1 / Tang2 run

The migration is active with 25 exclusive shards. Tang3 owns 0–7 on GPUs
`0,1,2,3,4,5,6,7`; Tang1 owns 8–16 on GPUs
`0,1,1,2,2,4,5,6,7`; Tang2 owns 17–24 on GPUs
`0,1,2,2,5,5,6,7`. Each host uses
`/data/luyq/hitph_cycle4_worker/job` and local `work_25`. Tang1 and Tang2
mount Foldtmp at `/data/luyq/hitph_cycle4_worker/foldtmp`; Tang3 retains its
existing `/NAS/luyq/TMP_prediction/protenix_embeddings/hitph_v1` mount.
Current parent logs are `launch_reverted.log` and worker logs are under
`work_25_reverted/` on each host. Log in from Tang3 with `ssh Tang1` and
`ssh LDS_Tang2`.

Tang3's `monitor_permanent.py` waits for all 25 shard metadata pairs, runs
`merge_shards.py`, then invokes `run_hitph_cycle4_epochs.py` to cache epoch 1
negatives. Its log is `/data/luyq/hitph_cycle4_worker/monitor_25.log`.
The active process IDs and host assignments are recorded in
`configs/hitph_protenix_cycle4_nomsa.json`. Existing permanent feature files
from the original 7-shard run are reused after validating identity and
extraction parameters; only uncached records are inferred.

The workers on each host share one SSHFS connection. A multi-mount experiment
on 2026-10-13 reduced four concurrent 64 MiB write-test files from 19–30
seconds to 8–11 seconds on Tang2, but the actual three-host cache rate fell
from about 60 to about 40 records/minute. The multi-mount jobs were stopped
and all three hosts restored to their original single SSHFS mounts. Current
worker logs are in `work_25_reverted/`; `work_25/`, `work_25_multi/`, and
`lanes_25/` preserve the earlier logs. The shard numbers, GPU assignments,
remote output directory, and cache format never changed. Previously finished
files were validated and reused after each restart.

## Epoch-1 negative pre-cache on spare GPUs

Seven independent epoch-1 negative shards (`0..6` of 32) started on 2026-10-13
while permanent extraction was running. Tang3 uses GPU 3 for shard 0; Tang1
uses GPUs 0,4,5,7,6 for shards 1,2,3,4,6; Tang2 uses GPU 3 for shard 5.
Their input has 158,930 unique `pepneg_train_*` names and no overlap with the
96,670 permanent names. They write to the same Foldtmp features directory,
but use `index.shard_XX_of_32.json` metadata, distinct from the permanent
`index.shard_XX_of_25.json` files. Local logs are in `epoch1_work_32/` on each
host. The input SHA-256 and initial launcher PIDs are in the config.

When the permanent 25-shard monitor finishes, `run_hitph_cycle4_epochs.py`
calls `epoch1_distributed.py finish`. It starts shards 7..31 on GPUs with at
least 8 GiB free (up to 18 new workers), waits for all 32 shard metadata pairs,
then validates and publishes the epoch-1 index. The permanent index is copied
to `permanent_index.json` before the epoch index replaces `features/index.json`.
Do not run the ordinary 5-GPU epoch-1 extractor at the same time: it would
recompute records owned by the distributed shards.

The permanent index completed at 2026-10-14 11:31 CST. Epoch-1 now has 25
worker processes covering all 32 disjoint shards. The original coordinator
exited after shard 19 repeatedly failed CUDA initialization on Tang1 GPU 4;
that shard was moved to Tang1 GPU 1 and resumed without deleting any cache.
The coordinator was restarted in a separate session; its current log is
`/data/luyq/hitph_cycle4_worker/epoch1_resume_20261014.log`. On future worker
failure, it chooses another GPU with at least 12 GiB free before retrying.

The epoch-2 input and manifest were generated in `work/hitph_cycle4_nomsa/`
without starting inference. Its 158,930 negatives overlap epoch 1 by 5,942
IDs (3.74%). The training rotation now retains those shared feature files
after verifying the epoch-1 checkpoint, and deletes only the remaining
epoch-1 negatives before caching epoch 2.
