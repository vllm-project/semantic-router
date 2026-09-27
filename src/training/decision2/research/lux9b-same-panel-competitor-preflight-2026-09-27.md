# 9B release path and same-panel competitor preflight

**Status: CPU/read-only preflight.** This note uses no sealed JevArena v3
labels and reports no new model score. The exact 9B checkpoint, data, and
source identities are recorded in the earlier
[BEST224 preflight](lux9b-structured8360-best224-rights-package-preflight-2026-09-27.md)
and [source-revision audit](lux9b-source-revision-rights-gate-2026-09-27.md).

## Current own-lineage candidate

`lux9b-human-structured8360-low-fla-r1` BEST224 initializes from our published
`Decision-1.0-Lux-9B` at immutable revision
`bd45a30aee8c84032791c245c70f86dee5389cc8`; all 16 recorded source
files matched that snapshot. This meets the Decision 2.0 initializer policy.
The unmerged native model loads 7,984,174,080 parameters. Its typed DEV
1,398/1,600, CSS three-task pilot median macro-F1 0.573976, and public
JevBench 183/231 are development/public diagnostics, not JevArena v3 release
results. The Lux 1.0 development observations were 86.750%, 0.57011, and
183/231, respectively. The independent human-only 9B arm had higher CSS pilot
median macro-F1 (0.580324), so BEST224 is not uniformly better.

The frozen structured TRAIN has 8,360 rows, 19 merged source keys, 2,536 added
rows, and 600/900 SELECT/CAL rows. Its manifest SHA-256 is
`f2e2e35ac552731495679a00a9674fd4347ee9f07ff5cf7a5cf66e42e810c2da`.
The 2,536-row addition has source evidence, but the inherited 5,824 rows do
not yet have an exact-source public-weight determination. The mixed Stage3 and
TweetEval source keys, inherited MultiNLI rows, and source model's own lineage
must be reviewed separately. The schema-aware publication verifier now reads
`merged_counts` and `limits`, binds the exact run and partitions, and requires
a reviewed decision for every merged source key. It cannot create that rights
decision from a builder manifest. BEST224 stays HOLD. There is no merged full
package, package-native full-panel parity, or frozen v3 prediction set.

## Proposed 9B comparison roster

| Role | Immutable model identity and native contract | Admission action |
| --- | --- | --- |
| Own-source baseline | [`Decision-1.0-Lux-9B`](https://huggingface.co/llm-semantic-router/Decision-1.0-Lux-9B) at `bd45a30aee8c84032791c245c70f86dee5389cc8`; dynamic Choice/Noul/Score head | Re-run on the same v3/JevBench inputs and runtime; retain paired outputs. |
| Strong external 9B | [`kirp/jpt-9b`](https://huggingface.co/kirp/jpt-9b) at `7114b0c3d9bea6b82dfa2d0691e8d5562cd26d4e`; full weights, published three-type `llm2jev` interface, T=1.087 | Download full pinned package; smoke Choice/Noul/Score through its actual scorer, including long-input and invalid-response behavior, before a full panel. |
| External adapter 9B | [`bespokelabs/Bespoke-Nimble-9B-v2`](https://huggingface.co/bespokelabs/Bespoke-Nimble-9B-v2) at `4b8c04d1ac2cea3e41e5e3c4d2130bcead2c0abe`; pinned official Qwen3.5-9B base `c202236235762e1c871ad0ccb60c8ee5ba337b9a`, PEFT adapter and included `ParallelScorer`, T=2.179078721266035 | Verify base+adapter bytes and calibration once; use native typed probabilities. Its 8,192-token limit counts overlength cases as failures, without truncation. |
| General-source baseline | [`Qwen/Qwen3.5-9B`](https://huggingface.co/Qwen/Qwen3.5-9B) at `c202236235762e1c871ad0ccb60c8ee5ba337b9a` | Include only after freezing a native decision adapter, temperature policy and gold-free smoke. |

The JPT model card describes a separate public 231-item result and a Decision
Index 0.2 result; Nimble has its own published tests. None is a substitute for
running both on our **same** JevArena v3 and public JevBench versions. Index
0.2.1 is a different, post-release external evaluation and its published score
cannot be copied into JevArena. Actual loaded parameter counts must be measured
before plotting Pareto points; model-name size labels are approximate.

## Next discriminating experiment

Do not spend a GPU-hour merging or formally scoring BEST224 until its exact
source-use decision and package parity are complete. If those cannot be
cleared, freeze a separate official `Qwen/Qwen3.5-9B` or audited own-Lux start
on rights-clean v2, with a short zero-step and numerical/length preflight,
precommitted one-epoch/token budget and SELECT-only checkpoint choice. First
use a fixed gold-free roster to qualify the JPT and Nimble native adapters;
then run the strongest available 9B peers and our eligible candidate on one
common panel. The previously completed structured replay and human-only arms
remain read-only controls; do not retrain them.
