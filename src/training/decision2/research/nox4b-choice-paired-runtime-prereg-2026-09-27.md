# Own-Nox 4B: matched-runtime Choice weighting screen

**Prospective development experiment.** The earlier Choice-weight v1 and v2
attempts stopped at zero-step comparisons against a historical process. This
new experiment compares a fresh control and treatment produced by the same
source and container on two reserved physical GPUs. Neither earlier attempt
is reinterpreted as a success. No protected JevArena v3 labels, CSS15 labels,
public JevBench labels, calibration labels or model publication enter this
screen. The frozen panel has already been opened during other 4B research, so
any later run there is same-panel validation, not a virgin blind test.

## Causal question and inputs

Does 1.5× loss weight on genuine human-labeled Choice rows improve transfer
and Choice accuracy when the dataset, model initialization, schedule, option
renderer and native head are identical? The only treatment difference is the
weight of 2,240 Choice rows from `google_goemotions_official_train` (1,400),
`legacy:cosmos_qa` (448), `legacy:snli` (272), and
`css_flute_official_train` (120). The original control command requested the
same four source IDs at weight 1.0; the execution correction below explains
why it must instead omit the inactive source list. All other examples have
weight 1.0 in both arms. Within an
accumulation window, each arm normalizes by its own sum of weights.

| Frozen input | Identity |
| --- | --- |
| Initial weights | Our `llm-semantic-router/Decision-1.0-Nox-4B` at `0bb833504965c0eabdb9630b7bbd385cb2fe5cd4`; local 97-file release-manifest SHA-256 `50c2f77c7c3f6c1efae7014ccfc4aedc3b185e52731ba721a1c6d6192668d945` |
| TRAIN | Rights-clean v2, 7,455 rows, 4,194,465 unpadded tokens, maximum 6,596, SHA-256 `61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755` |
| SELECT | 700 rows, SHA-256 `32a4352d8ed93ce82430db80175339ad8e4d40c618f2866608fdb6ef5120f2a6` |
| CAL | 700 rows, lineage/isolation audit only, SHA-256 `3e34f6cb5a32c9f14d0fee0897ee3f2318e59d66fe1ff0a95e2ea5eb2497f60a` |
| Code | Local branch `xunzhuo/decision2-own4b-paired` at parent `931a0f79c`; `training/model/train.py` SHA-256 `bab25d3ae9ef9f3faaeb8241fb83f52f13d48cd0a1c26d3bcdd7f5c46d65ab13` |
| Four-cell comparator | `research/nox4b_compare_paired_zero_step.py` SHA-256 `04cea5cf8f05640b50247b2d59ab801158dd0d436496b46d42d78419c3e82eaa`; its receipt contains no labels or raw prompts. |
| Runtime | Same pinned ROCm training image `sha256:f83b1d10f14dbe46ea14ee56fd3e5d01849673f3739fed5311c99ba54cbc2d54`; retain its optimized `fla` import path. |

Both arms use Decision-1.0 native dynamic-head initialization, rank-16,
alpha-32, dropout-.05 LoRA, LoRA LR 1.5e-5, head LR 7.5e-6, CE + .5 Brier,
BF16 autocast/FP32 head, seed 20260926, microbatch 1, accumulation 16,
8192-token limit, one epoch, and the planned 466-step warmup/cosine schedule.
The prospective **only** weight observation is durable step 128; preserve
the 466-step schedule while stopping each arm immediately after that atomic
checkpoint. Do not select another step, substitute the historical checkpoint,
or complete later updates under this experiment name. Same train row order,
input-token trace, and SELECT/CAL bytes are required.

## Sequential gates, fixed selection and budget

1. Atomically reserve two currently idle cards, confirm no KFD processes and
   zero HBM occupation on both, verify all 97 source files against the release
   manifest, all data/code hashes, exact four-source 2,240-row cohort, and
   single-device BF16 plus optimized `fla`. Never touch other GPU owners.
2. Run four separate **zero-step-only** processes, control and treatment on
   each of the two cards, before an optimizer step. All four SELECT700 records
   must have identical ordered IDs, native prompt/token hashes, task type and
   option domains. Every pair must have zero category changes and maximum
   absolute offered-option probability drift <=1e-4. Seal each raw prediction
   SHA-256 and a comparison receipt before proceeding. A failure stops both
   arms; no historical-control parity requirement is applied to this fresh
   causal pair.
3. Run a separate treatment-only one-update smoke using the planned settings
   except `max-steps=1`. Require one visible BF16 GPU, finite loss and gradient,
   at least one weighted row in the window, a durable checkpoint, and projected
   step-128 training cost at most 0.40 GPU-hour per arm. The different schedule
   makes this smoke unusable as a score.
4. Run the matched full-horizon processes in parallel only after gates 1–3;
   control on the first reserved card and treatment on the second. Stop each
   after its atomic step-128 checkpoint and SELECT700 receipt; retain both
   checkpoints and all failed attempts. Abort for changed dataset/token roster,
   nonfinite numbers, resource collision, missing checkpoint or cumulative
   allocation above 1.20 GPU-hours (including four zero-step runs, smoke and
   both training arms). Log actual wall allocation and GPU-hours.
5. SELECT advancement is measured treatment versus **fresh control step128**.
   Require at least +6/200 correct on the GoEmotions Choice slice; at most five
   fewer correct on all SELECT700; Score >= control −2/90; family macro accuracy
   >= control −.005; all 700 valid. This fixed screen tests whether the weight
   helps its intended human Choice target without collapsing other skills.
6. Only if step 5 passes, run one uncalibrated native typed DEV1600 and CSS
   pilot1430 diagnostic for **both** saved step128 packages. The exact same
   prompts, adapter and scorer are required. Promote to a later separate full
   training experiment only if the development proxy
   `100 sqrt(T_DEV * H_pilot)` is >=56.0 and >=fresh-control+1.0 point,
   treatment Choice improves >=12/800, and neither panel has added invalid
   answers. All type/task regressions, Brier and uncertainty must be reported;
   individual Score or human-task regression is disclosed but is not silently
   converted into a new hard cap. Only a promoted later arm may run the public
   231-subset diagnostic; this short screen never runs protected formal labels.

The old Nox clean-v2 run completed 466 steps in about .656 GPU-hour; each new
128-step training arm is estimated .20–.35 GPU-hour plus startup/checkpoint.
The new pair has a capped 1.20 total GPU-hours. The purpose is one causal
conclusion per bounded GPU-hour, not utilization for its own sake. Decider 4B
and Hopper were selected from Decision Index 0.2.1 as peer candidates, but
their historical Index values are never mixed with our development or formal
scores. They require same-panel native reruns before any rank entry.

## Execution correction, 2026-09-27 10:46 UTC

The first **control** zero-step process failed in argument validation before
model/data loading: the trainer enforces `(weight != 1.0) == bool(source_ids)`.
The requested `source_ids=[four IDs], weight=1.0` combination is invalid.
The parallel **treatment** zero-step process exited successfully and wrote its
unscored baseline, but no comparator ran, no control baseline exists yet, no
optimizer update occurred and no development score was read before this
correction. The failed control process used approximately 4.5 seconds of
allocated GPU time; retain its private log.

For both control zero-step runs and its later training process, omit
`--choice-source` and `--choice-source-weight`; the validated defaults are an
empty source list and weight 1.0 for **every** TRAIN row. The treatment keeps
the exact four source IDs and weight 1.5. Both arms still load precisely the
same 7,455 rows and encode them in the same order; the only non-metadata loss
difference is the weight of the 2,240 predeclared rows. Recompute and compare
all four zero-step predictions as originally specified before any update.
All SHA-256 identities, numerical gates, model initialization, learning-rate
schedule, fixed step128 observation, GPU-hour cap, datasets and downstream
promotion rules above are unchanged. This is an execution correction made
before training or outcome comparison, not a retrospective pass for the
failed process.
