# Public pressure development panel

This compiler implements the frozen
[pressure-panel protocol](../research/pressure-panel-protocol-2026-09-26.md)
from [Decision Models Under Pressure v3](https://github.com/gazelle93/decision-models-under-pressure/tree/21820dc8b455e37aebaf5a83122233b6e564c109/dataset/v3).
It is a **public development diagnostic**. It must not be used as a final
held-out release gate or merged into the CSS15/family-disjoint final ranking.

The pinned source commit is `21820dc8b455e37aebaf5a83122233b6e564c109`.
The compiler checks full SHA-256 digests for `items.jsonl`, `manifest.json`,
`gates.json`, and all three split files, plus every file and byte count listed
in the source manifest. It requires exactly 1,000 unique source UIDs, their
declared domain counts, intact text hashes, and unchanged tracked files.
The emitted manifest records those source digests, generated file digests,
compiler digest, prompt recipe, and the source's failed G3 text-blind picker
gate. The upstream data are **CC BY-SA 4.0**, including derived files that
contain item text; credit the underlying CLINC, MTOP, GoEmotions, DBpedia,
and financial tweet datasets and preserve ShareAlike if redistributing.
The financial tweets were written by third parties; the uploader's MIT
metadata does not settle rights in the underlying posts.
The upstream repository has no root code license at this revision; the reader
and scorer here were written independently.

## Build on an experiment host

Use a clean upstream checkout at the pinned commit. `--output-dir` must not
exist so an earlier panel cannot be overwritten accidentally.

```bash
PYTHONPATH=/work/source python3 -m transfer.pressure_build \
  --source-root /work/external/decision-models-under-pressure \
  --output-dir /work/runs/pressure-dev-v1
```

Three independently hash-pinned slices are emitted:

| Slice | Gold-free main requests | Additional control requests | Use |
| --- | ---: | ---: | --- |
| `rq3_shared_hardness` | 3,200 | 0 | Same UID/K under near and far distractors, K=2,4,8,16 |
| `rq2_shared_order` | 8,000 | 1,600 | Five permutations of one K16 set, plus one identical-order repeat per UID/tier |
| `rq1_extended_capacity` | 5,600 | 0 | Nested ext-pool K=2,4,8,16,32,64,128 |

For each slice, only `<slice>.prompts.jsonl` goes to the model. Every prompt
has exactly `id`, `state`, and one typed `choice` question named `decision`.
The ID is an opaque digest; no source UID, domain, leakage flag or gold field
is exposed. `<slice>.gold.jsonl` remains private to the scoring process and
holds the original UID, domain, leakage flag, K/pool/permutation, complete
ordered labels, gold, option-set hash, option-order hash, and exact prompt
payload hash. The order seed is SHA-256 of compact UTF-8 JSON
`[uid,pool,K,permutation]`. RQ2's repeat has a distinct ID and an identical
payload hash to permutation zero. Its 1,600 controls are excluded from the
main accuracy denominator.

## Score predictions

Each native adapter must read the same gold-free prompt file and return the
unified prediction contract: `id`, `answers.decision`, and the echoed
`source_input_sha256`, `model_id`, `model_revision`, and `adapter_version`.
The scorer requires every present receipt identity to equal its report arguments.
A missing prediction counts as a miss; a mismatched source hash, identity, or
an ID outside the panel aborts scoring. An answer may omit
probabilities, but a supplied map must cover all candidates, contain finite
values in `[0,1]`, sum to within 0.02 of one, and agree with the returned
choice within 0.02. Brier and NLL are calculated only for valid normalized
maps; their coverage is reported. Keep the same inference limits and native
readout for every model, and record the deployment profile separately from
quality metrics.

```bash
PYTHONPATH=/work/source python3 -m transfer.pressure_score \
  --manifest /work/runs/pressure-dev-v1/pressure-manifest.json \
  --slice rq3_shared_hardness \
  --prompts /work/runs/pressure-dev-v1/rq3_shared_hardness.prompts.jsonl \
  --gold /work/runs/pressure-dev-v1/rq3_shared_hardness.gold.jsonl \
  --predictions /work/runs/model-pressure-rq3.predictions.jsonl \
  --model model-name --revision pinned-revision --adapter-version adapter-id \
  --deployment-profile matched-single-request \
  --source-overlap not-audited \
  --bootstrap-iterations 1000 --bootstrap-seed 20260926 \
  --output /work/runs/model-pressure-rq3.score.json
```

The RQ3 report gives all-item and per-domain/leakage/K accuracy, near/far
paired accuracy difference, joint correctness, and probability difference.
RQ2 reports changed semantic choices across reordered identical candidate
sets, a separate identical-order repeat rate, and invalid-inclusive rates.
RQ1 gives the K curve with response/admission coverage and invalid reasons.
Intervals resample *source UIDs*, not repeated queries. Report any
context-overflow or candidate-limit errors as missing capability; never
silently truncate a candidate list. K>16 is an extended capacity view since
some native model interfaces only admit 2–16 candidates.

The source's G3 gate fails in six of ten domain/tier cells at K16. Public
items may also overlap pretraining. Consequently the absolute accuracy level
is diagnostic only. The paired near/far and order effects can be useful for
development, with their own prompt-order and option-format limits disclosed.
