# Stage81: sequence-only adaptive seed selection

This is a deployable selection rule: it requires the fixed training-chain
sequences but no candidate MSA results. It does not construct a database or
launch MSA, feature extraction or model training. Existing experiments are untouched.

## Rule held constant

- Strata: nanobody heavy, paired heavy, paired light; training membership only.
- Represent each chain by its set of overlapping 3-mers.
- Distance: `0.8*(1 - intersection_size/union_size) + 0.2*abs(length_a-length_b)/max(stratum_length_span,1)`.
- The length span is fixed across ALL training candidates of that stratum.
- Initialize with shortest and longest chains; deterministic lexical tie handling.
- Then repeatedly select the candidate maximizing distance to its nearest selected seed.

The old 80/60/60 quota is replaced by a maximum nearest-seed distance threshold,
applied independently to each stratum. Every chain must have a SAME-stratum seed
within that threshold. The extrema initialization is retained even if one seed
would satisfy a loose threshold: this isolates stopping-rule changes but is not
a globally smallest-set algorithm. Equal k-mer sets and lengths can have zero
distance even when sequence order differs; this is a sequence descriptor proxy.

## Exploration, not a selected optimum

Default radii 0.15/0.20/0.25/0.30/0.35/0.40 are a transparent exploratory grid,
not paper-derived or validated cutoffs. Smaller radius means stricter geometric
coverage and at least as many selected seeds. A radius of 0.20 DOES NOT mean
80% MSA-hit retention. No prediction or label chooses a threshold here.

Outputs include the entire sequence-only coverage curve and selected lists for
each threshold, plus the geometric radius reached by the old 80/60/60 budget.
If a stratum has too few chains, its baseline row is capped and JSON flags that
the exact historical budget was unavailable. Unique queries are deduplicated
across stratum rows. Lists use `stratum, sequence, length, seed_index` fields.

Do not launch downstream training for every output automatically. First inspect
counts and geometric coverage, fix an intended budget/threshold without using
official test scores, then build a library with that list. For a clean comparison
with original200, preserve the original hit filtering and library construction;
do not silently adopt Auto144's changed short/self-hit filtering.

## Run remotely

```bash
cd /root/autodl-tmp
unzip -o foldabs-repro-stage81-sequence-radius-seeds.zip
cd foldabs-repro
bash scripts/run_tm_sequence_radius_seed_selection.sh
```

The wrapper activates the EXISTING environment, tests the code, runs selection,
and prints summary.md. It uses only the Python standard library. Output goes to
`foldabs-work/msa_accelerated/tm_sequence_radius_TIMESTAMP/`. No GPU or remote
service is required. Never overwrite a prior output directory.

Optional threshold override:

```bash
FOLDABS_SEED_RADII=0.20,0.25,0.30 bash scripts/run_tm_sequence_radius_seed_selection.sh
```

## Provenance and limits

K-mer sequence descriptors and farthest-first representative selection are
established ideas, e.g. Gonzalez (1985), DOI 10.1016/0304-3975(85)90224-5.
This does not mean the particular k=3, 0.8/0.2 weights, biological strata,
initialization or thresholds are taken from that paper or proven optimal.
Those exact choices are reconstruction heuristics. No classical approximation
guarantee is asserted for this implementation's two-extreme initialization.
Good original200 downstream results validate that tested configuration, not a
universal guarantee of homolog coverage or a proof that k=3 beats k=2 or k=4.

Tests check exact prefix equivalence against the existing selector, deterministic
ties, radius correctness, monotone counts, singleton/empty strata, invalid
thresholds, held-out/label independence and refusal to overwrite results.
