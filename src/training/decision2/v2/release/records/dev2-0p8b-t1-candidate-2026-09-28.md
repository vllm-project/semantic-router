# DEV2.0-0.8B: retroactive calibration check and a prepared T = 1 revision (2026-09-28, not uploaded)

**Published 2026-09-29 ≈00:12 UTC+8** as private `llm-semantic-router/DEV2.0-0.8B@7d08d0e12082ee6810a4221063342cefe45b6ac3`
(manifest `2a19a65a…`: this candidate rebuilt from `6d7e7a148`, only `builder.source_commit` differs).
Final decision `fedb18fa…`. See the [release record](dev2-0p8b-release-2026-09-28.md).

**Result: the released DEV2.0-0.8B's CAL698 temperatures fail the coordinator's 23:15 development rule.
A calibration-only T = 1 candidate is built and verified on node A but not uploaded; publishing is the
coordinator's decision.** The released package is `llm-semantic-router/DEV2.0-0.8B@0b631a85c19fb573aee34fc68bb413271ebe89f4`
(manifest `0af27b1c…`, CAL698 `calibration.json` `9f76867d…`: Choice 1.123, Noul 1.071, Score 0.395).

## Development-panel check

The decoder track's stored development predictions of the E8F soup (`v2/dec` readout, identity
`60356482…`, 8,192 tokens, all answers valid) were produced with its CAL700 development calibration
(`4728f1e8…`); `v2/release/dev_calibration.py` undid those temperatures (T = 1) and applied CAL698, with
0 answer changes, and scored both with the development scorers
([receipt](../dev2-0p6b-release-2026-09-28/devcal/dev2-0p8b.json)):

| Development metric | Raw (T = 1) | CAL698 |
| --- | ---: | ---: |
| Typed DEV Brier / ECE-10 | 0.269 / 0.128 | **0.300 / 0.180** (worse) |
| CSS pilot median task Brier-sum / ECE-15 | 0.834 / 0.127 | 0.821 / 0.109 |

Typed-DEV Brier and ECE worsen: CAL698 is rejected under the rule. (For the record only, not a decision
input: on the formal panels T = 1 has typed Brier / ECE 0.254 / 0.103 versus 0.278 / 0.150 with CAL698,
human-transfer median task Brier-sum / ECE 0.650 / 0.164 versus 0.630 / 0.147, public 231 0.200 / 0.114
versus 0.200 / 0.116.)

## Prepared T = 1 candidate (built and verified locally)

- **Scored bindings:** the sealed CAL698 formal run `m2-E8F-soup-nodeA` was returned to T = 1 with
  `v2/release/retemper_predictions.py --undo` (typed 1,600, transfer 6,547, public 231, mlx-diag 2,275
  rows; 0 answer changes; [receipts](dev2-0p8b-t1-candidate-2026-09-28/derived-run/)) and adopted with
  `same_panel adopt` (reason stated in `ADOPT.json`), sealed (`522e813d…`), reported
  ([REPORT](dev2-0p8b-t1-candidate-2026-09-28/derived-run/REPORT.json) `1f5cf33d…`: v3 50.236, typed
  529 / 611 / 107, public 156) and paired against the same adopted Eos 1.0 run (`fa62c29a…`: +7.69
  [+3.65, +13.32], unchanged).
- **Spec** [`specs/dev2-0p8b-t1-candidate.json`](../specs/dev2-0p8b-t1-candidate.json): the released
  spec with `calibration: null`, those bindings, and a calibration line stating the evaluated-and-rejected
  temperatures; everything else (C1 line, disclosures, limits) unchanged. Build binding: draft
  [`DEV2.0-0.8B.decision.t1-candidate.json`](dev2-0p8b-t1-candidate-2026-09-28/DEV2.0-0.8B.decision.t1-candidate.json)
  (`064afbc0…`).
- **Local verification** (`release.sh` without `--upload`, node A GPU0 shared, the release's kernel
  runtime: `--site /opt/decision-fla --require-kernels`, a fresh copy of the scored Triton autotune cache;
  [receipts](dev2-0p8b-t1-candidate-2026-09-28/local/)): package manifest
  **`a1f5c3324013e309edaf5064e3d2d59604243f8cfe9ca29656d2c8f1b4c9bef0`**, 30 files, 753,446,208 loaded
  parameters; every model file and the identity equal the released revision, and only `calibration.json`
  (removed), `config.json`, `README.md` and `evaluation/*` change. Gated-delta and causal-conv1d kernels
  bound; native examples bit-identical in two processes; card example reproduced; parity against the
  derived T = 1 predictions on 600 prompts: 0 changes, drift ≤ 4.4e-16. The card shows typed Brier / ECE
  0.254 / 0.103.
- **To publish (coordinator):** write a final decision for `DEV2.0-0.8B` naming report `1f5cf33d…` and
  paired `fa62c29a…`, point the candidate spec's `gate_receipt` at it, and run the same `release.sh`
  command with `--upload --collect` (hub upload now removes the stale `calibration.json`, fix `89294e89f`).
  C1 event 1 scores are unaffected (answers identical).
