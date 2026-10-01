# DEV2.0 runtime: forward token budget for long multi-question requests — release hand-off (2026-10-01)

Owner: eval track (IX1 follow-up A). Status: fix merged into `xunzhuo/decision-2-training`; **no Hub revision
published**. The release track builds and publishes the runtime-only revisions below.

## What failed

The shipped runtime (`QwenDecision.system_one`) answers all questions of a request in one padded batch: every
question re-encodes the full prompt, so the forward holds questions × padded-longest-prompt tokens. FLA 0.5.2's
gated-delta forward (`chunk_gated_delta_rule` with in-kernel q / k L2-norm, as transformers' Qwen3.5
`GatedDeltaNet` calls it) computes some element offsets in 32-bit integers (`chunk_fwd.py`: `i_bh` from
`tl.program_id(1)`; `l2norm.py`: `i_t`). Once a q / k / v tensor (padded tokens × value heads × 128, q / k repeated
to the value-head count) passes 2^31 − 1 elements, rows past the boundary are wrong, logits go non-finite
(`invalid_model_output`) or the GPU page-faults. The limit is 349,525 padded tokens on DEV2.0-27B (48 value heads),
524,287 on 9B / 4B (32), 1,048,575 on 2B / 0.8B (16).

Two long multi-question requests from the private eval panel hit it on DEV2.0-27B `4e89288d`: 32 questions each,
longest prompts 14,223 and 22,676 tokens, padded 455,168 and 725,760 tokens (one `invalid_model_output`, one
memory-access fault). The other sizes refuse the 22,676-token question (`max_length_exceeded`, 16,384-token cap)
and stay under their limit on the other request. Request content stays in the private eval directories.

Forwards between about 2^30 and 2^31 elements (21–24 rows × ~14,224 tokens) also hung (host thread spinning, GPU
idle) or segfaulted in the HIP runtime on DEV2.0-27B; they passed with `AMD_SERIALIZE_KERNEL=3` or a synchronize
after every module. Not isolated further; the budget therefore stops at 2^30.

Nothing caps the questions per request, so every Qwen3.5-family package can reach the limit within its own
`max_input_tokens` (e.g. 9B: 32 questions of 16,384 tokens; 0.8B: about 64). The dense DEV2.0-0.6B (Qwen3, no
gated-delta layers) cannot.

## What changed

`src/training/decision2/v2/release/runtime/qwen.py` only (commits `8e6bdfc33`, `e876fbefc` on
`xunzhuo/decision-2-eval-index`; merged with the training branch's label-token readout change, conflict limited to
the two new constructor arguments):

- `forward_token_budget(config)`: `(2**31 − 1) // 2 // (linear_num_value_heads × max(linear_key_head_dim,
  linear_value_head_dim))` padded tokens, `None` for a backbone without gated-delta layers. 27B 174,762; 9B / 4B
  262,143; 2B / 0.8B 524,287; 0.6B `None`.
- `micro_batches(lengths, budget)`: one group when the padded batch fits (the old path, unchanged); otherwise the
  questions, longest first, are cut into consecutive groups that each fit. A single question over the budget raises
  (not reachable under the current caps).
- `QwenDecision.load` sets the budget on GPU only; `system_one` runs each group as its own forward and reassembles
  the logits in question order. Answers are per question, so splitting changes them only by BF16 batch-shape noise.
- Tests: `v2/release/tests/test_long_request.py` (CPU: budget values, grouping, single-question error, split vs
  whole-batch answers through `system_one` with a row-wise fake model) and `v2/release/tests/gpu_long_request.py`
  (GPU synthetic regression: a generated-word request of N questions with one long prompt; each answer vs the same
  question asked alone; public text only).

Released `decision2/qwen.py` (0.8B `e13a40f8`, 9B `b4f65fa8`, 27B `4e89288d`): sha256 `59eb1605ece2…`. Fixed
(pre-merge, as verified below): `693154e796b1…`. The merged file differs from the verified one only by the
label-token change already on the training branch.

## Evidence (node D; packages restaged with only `decision2/qwen.py` replaced)

Synthetic regression (`gpu_long_request`, DEV2.0-27B, 32 questions, tolerance 0.02 vs alone):

| Long prompt (tokens) | Unfixed runtime | Fixed: forwards (rows × padded) | Invalid | Differ from alone | Max drift |
| ---: | --- | --- | ---: | ---: | ---: |
| 14,224 (×2) | 2 invalid, 9 / 32 differ | 12 × 14,232 + 20 × 432 | 0 | 0 | 0.011 |
| 18,000 | — | 9 × 18,008 + 23 × 456 | 0 | 0 | 0.010 |
| 22,676 | GPU page fault | 7 × 22,784 + 25 × 488 | 0 | 0 | 0.015 |

Both private requests on the fixed 27B runtime (each twice): 32 / 32 valid answers, 32 / 32 equal to the question
asked alone (max drift 0.0034 and 0.0022), wall 23–35 s, peak 95.2 and 93.4 GiB.

Release parity (`examples.py parity`, the BF16-resident rollout's inputs and frozen caches, tolerance 1e-4):

| Package (fixed) | Manifest | typed-final 1,600 | css15 6,547 | public231 231 | mlx-diag 2,275 | Max drift |
| --- | --- | --- | --- | --- | --- | ---: |
| 0.8B from `e13a40f8` | `ec82f6ca…` | 0 changes | 0 | 0 | 0 | 8.9e-16 |
| 9B from `b4f65fa8` | `9ff154ef…` | 0 | 0 | 0 | 0 | 8.9e-16 |
| 27B from `4e89288d` | `22fed33a…` | 0 | 0 | 0 | 0 | 0 |

0 missing and 0 input mismatches everywhere; caches `5e37a143…` / `5604ffdc…` / `f474e2e9…` (digest-checked).
Every parity request fits its budget, so it takes the unchanged single-batch path.

Latency (`runtime_bench.py`, 400 typed-final requests after 400 warm-up, released vs fixed package on the same GPU,
fresh frozen-cache copy each side):

| Size | Bit-identical | p50 ms old → new | p95 ms old → new | Request peak GiB old → new |
| --- | ---: | --- | --- | --- |
| 0.8B | 400 / 400 | 25.6 → 23.2 | 33.1 → 35.0 | 2.01 → 2.01 |
| 9B | 400 / 400 | 28.7 → 28.6 | 31.0 → 29.8 | 16.83 → 16.83 |
| 27B | 400 / 400 | 98.9 → 98.8 | 107.6 → 107.4 | 52.07 → 52.07 |

Package manifests old → new: 0.8B `3fb9e83e…` → `ec82f6ca…`, 9B `551fff70…` → `9ff154ef…`, 27B `c0d8c65a…` →
`22fed33a…`. Requests that fit the budget run the old single-batch code path; the split adds one forward per extra
group only for requests that would otherwise fail.

## Runtime-only revisions needed (release track)

| Repository | Verified against | Why |
| --- | --- | --- |
| DEV2.0-27B | `4e89288d` | reachable within its 32,768-token cap (the two observed failures) |
| DEV2.0-9B | `b4f65fa8` | reachable (32 questions at the 16,384 cap) |
| DEV2.0-4B | `4f560ae5` | reachable (same head layout as 9B) |
| DEV2.0-2B | `56950ec5` | reachable with ≥ 64 long questions |
| DEV2.0-0.8B | `e13a40f8` | reachable with ≥ 64 long questions |
| DEV2.0-0.6B | `def20a1c` | optional: no gated-delta layers, the budget is `None` and behaviour is unchanged |

Weights, calibration and every other file stay byte-identical; only `decision2/qwen.py` (and the manifest digests)
change. Suggested gate per tier: the rollout's four-panel parity (expect 0 changes, as above) plus
`gpu_long_request --tokens <cap − 300> --questions 32` on the 27B and 9B packages.

The 2.0 auto_map remote code (`modeling_decision2.py`, branch `xunzhuo/decision-2-automap`) loads the package's own
`decision2/` runtime and calls its `system_one`, so it picks the fix up with the vendored runtime; no wrapper change
is needed. The auto_map worker is publishing remote-code revisions on top of the BF16-resident revisions (0.6B and
0.8B first), so build each runtime-only revision from that repository's current `main` (carrying the remote code and
its manifest entries forward) and do not push to a repository while the auto_map worker is pushing to it.

All numbers above are counts, shapes, hashes and latencies; no eval-panel scores.
