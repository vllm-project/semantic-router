# 0.8B / 2B: M16's Index-path candidates, the Index-first choice and the releases (2026-10-02)

Continuation of 1afc17e8 (COORDINATION 2026-10-02 02:05, 04:25, 04:40, 09:40), superseded in its rule by the user's
Index-first decision (09:55). State: [`dec-08bfast-indexpath-state.md`](dec-08bfast-indexpath-state.md). Ops:
[`dec-08bfast-indexpath-2026-10-02/ops/`](dec-08bfast-indexpath-2026-10-02/ops/). **Index values are private**
(node C / node A private trees and `decision2-program/private/m16-indexpath/`); this record holds verdicts, public
scores, counts and hashes only.

## Bottom line

- **All three candidates qualify on the Index path** (item 1′: v3 not significantly below both bars and the private
  Index delta vs the tier's release significantly positive; item 6(b)′: v3 without the flagged items not
  significantly below; items 2–7 unchanged and passed in M16).
- **Under the Index-first rule (09:55) the per-tier winners are `08b-RA-a75` (0.8B) and `2b-RASD-a25` (2B):** the
  largest Index-gain 95% lower bound per tier. At 0.8B, `08b-RA-a75` ranks above M12's frozen `08b-RA` (the
  coordinator's requested comparison) and above `08b-RASD-a75`. `08b-RA-a75` and `2b-RASD-a25` also keep a
  significantly positive transfer-only Index delta (private values).
- **Item 8 (C1) was not run:** the 09:55 rule makes it a reference, run only if already in progress (it was not).
- Releases: section 6. Eos 0.8B shipped `08b-RA-a75`; Sol 2B shipped M15's `2b-RASDML` (user directive 15:35, the
  largest 2B Index lower bound once the Index sweep measured it).

## 0. Rules

- **02:05 (Index path, coordinator):** item 1′ = v3 not significantly below (paired 95% CI upper bound > 0) and the
  private Jev Decision Index delta vs the release significantly positive (paired bootstrap over rows within benchmarks
  through the board's area weights, ≥ 2,000 replicates, 95% lower bound > 0); item 6(b)′ = v3 with the flagged items
  removed not significantly below; items 2–8 unchanged; per tier the largest Index-gain lower bound; the transfer-only
  delta (without HoVer, When2Call, iSarcasmEval, GSM8K, BPoMP) always recorded privately.
- **09:55 (Index-first, user):** the only quality gate is the Index delta (as above); integrity checks are exact
  package parity, the Hub / `trust_remote_code` checks, the row-level Index contamination audit and no collapsed
  decision type; v3, human transfer, mlx-diag / MLX-DEV2, public 231 and C1 are reported references; C1 item 8 runs
  only if already in progress; never publish concurrently.

## 1. The Index runs of the frozen finalists (found complete)

The predecessor ran them on node C GPU1–7 (mirror `00401c9b5`, 20:22–21:04Z): each candidate's FP32 checkpoint (the
M16 formal package, files lists `bca5ec1d…` / `a57e236b…` / `b5248c80…`) restaged onto the tier's IX1 package
(DEV2.0-0.8B `bede7938` = the current Eos weights `3f02f0e5…`; DEV2.0-2B `a53cf66a` = the current Sol weights
`32872f29…`).

| Candidate | Parity gate | Full run (panel-7) | Dual scoring | Results | GPU-h |
| --- | --- | --- | --- | --- | --- |
| `M16-08b-RA-a75` | 86 / 86 ok, max \|Δp\| 0.0 | 120,226 rows: 120,224 ok, 2 unsupported, 0 errors | PASS | `23b90341…` | 1.28 |
| `M16-08b-RASD-a75` | 86 / 86 ok, max \|Δp\| 0.0 | same counts | PASS | `3697e0f8…` | 1.22 |
| `M16-2b-RASD-a25` | 86 / 86 ok, max \|Δp\| 0.0 | same counts | PASS | `aabe4111…` | 1.35 |

Bootstraps (this continuation, node C CPU, `paired_boot.py` of `42960045f`, 2,000 replicates, seed `20261002`), each
vs its tier's DEV2.0 IX1 run: full panel (no exclusion) and transfer-only. M12's `08b-RA` (IX1 run
`DEV2.0-0.8B-08bRA`, results `9c28a430…`, copied node D → node C through node A, hash equal) was bootstrapped the same
way for the 09:55 comparison.

| Run | Full-panel bootstrap | Transfer-only bootstrap | 95% lower bound > 0 |
| --- | --- | --- | --- |
| `M16-08b-RA-a75` | `e50577de…` | `544c3fb9…` | yes |
| `M16-08b-RASD-a75` | `dc2634d3…` | `0aa5acd0…` | yes |
| `M16-2b-RASD-a25` | `982430b8…` | `9611e6bd…` | yes |
| `DEV2.0-0.8B-08bRA` (M12 `08b-RA`) | `fbd0ff1e…` | `8f397e3f…` | yes |

## 2. Items 1′ and 6(b)′ (M16 two-bar successor files on node A)

v3 parts against both 0.8B bars (bar-t1 = the stored DEV2.0-0.8B run, bar-b = its node-B M16 collection; at 2B the
two bars give identical answers):

| Candidate | v3 | Item 1′ v3 vs bar-t1 / bar-b | Item 6(b)′ reduced v3 vs bar-t1 / bar-b | Rule-5 parts of 6(b) | Item 1′ Index | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| `08b-RA-a75` | 53.913 | +3.68 [−2.76, +4.82] / +3.65 [−2.68, +4.71] | [−2.87, +4.88] / [−2.76, +4.76] | own 1.0 low > 0; 6 / 6 reproduced | lower bound > 0 | **qualifies** |
| `08b-RASD-a75` | 51.685 | +1.45 [−4.27, +4.63] / +1.42 [−4.27, +4.55] | [−4.39, +4.75] / [−4.39, +4.72] | own 1.0 low > 0; 6 / 6 | lower bound > 0 | **qualifies** |
| `2b-RASD-a25` | 53.488 | +.05 [−.60, +1.67] | [−.57, +1.71] | own 1.0 low > 0; v3 ≥ 44.5; H vs Decider 2B ≥ 0; 6 / 6 | lower bound > 0 | **qualifies** |

Items 2, 3, 4 (mlx-diag card-eligible −.0037 / −.0015 / −.0017, upper bounds ≥ 0), 5, 6(a) and 7 passed in M16 (results
record `949ff508c`).

## 3. The Index-first choice

Ranked by the Index-gain 95% lower bound (values private): 0.8B `08b-RA-a75` > `08b-RA` > `08b-RASD-a75`; 2B
`2b-RASD-a25` (the only measured 2B candidate). The Index sweep (COORDINATION 10:00) measures further 0.8B / 2B / 9B
candidates and compares them against these releases.

## 4. Integrity checks

- **Types (scored runs):** `08b-RA-a75` choice / Noul / Score OK; `2b-RASD-a25` OK (`v2.eval.gates types`).
- **Row-level Index contamination audit** (IX1 method, planted control 200 / 200): the RASD arms' TRAIN files are hard
  links of M12's RA files (M13 data lock), so one audit per tier covers every candidate. 0.8B RA TRAIN `12bd63d8…`
  (309,225 lines): 18 duplicate-class rows, 1 item row (the same public-split BANKING77 item as the DEV2.0-0.8B TRAIN,
  which has 20 / 1). 2B RA TRAIN `08140409…` (91,146 lines): 76 duplicate-class rows, 0 item rows (DEV2.0-2B: 80 / 0).
  The card footnote stays "audited".
- **Exact package parity and the Hub / `trust_remote_code` checks:** release steps (section 6).

## 5. References (not gating; the scored runs adopted from the M16 formal runs, T = 1)

| | Eos 0.8B successor `08b-RA-a75` | Sol 2B successor `2b-RASD-a25` |
| --- | --- | --- |
| v3 vs the current release | 53.913 vs 50.236: +3.68 [−2.76, +4.82] | 53.488 vs 53.437: +.05 [−.60, +1.67] |
| human transfer (H) vs current | [−.046, +.079] | [−.018, +.023] |
| own 1.0 | vs Eos 1.0 +11.37 [+2.83, +15.86] | vs Sol 1.0 16K +7.71 [+4.09, +11.07] |
| peers | Intern-Decision-0.8B +10.38 [+1.70, +12.35]; Kev-0.8B +10.70 [+2.06, +13.54] | Decider 2B +3.99 [−1.89, +6.09]; This-That 1.2 +7.38 [−1.33, +14.01] |
| card-eligible mlx-diag vs current | −.0037 [−.0139, +.0065] | −.0017 [−.0076, +.0040] |
| public 231 vs current | 162 vs 156 (McNemar p .327) | 171 vs 171 |
| exposure (arm TRAIN) | 0 groups | 0 groups |
| C1 post-key | not run (09:55) | not run (09:55) |

## 6. Releases

Coordinator directive 11:40 UTC+8: one direct release per tier of the highest-scoring candidate; no reference-only
evals (v3, C1, human transfer, mlx-diag) are run for it.

### Decision-2.0-Eos-0.8B: M16 `08b-RA-a75`, revision `3de61185ade42cc307c26a6062df4baac327eac5`

- **Weights:** `v2.release.bf16_copy` of the FP32 point `fee82408…`, identity `d9712799…` (receipt `d1f51ee0…`).
- **Index evidence:** the private Index ran on exactly these weights (node A, panel-3, 120,224 `ok` + 2
  `unsupported`, scorer gate PASS). Paired bootstraps vs the current release (full panel and transfer-only, 2,000
  replicates): lower bound > 0. Values are private.
- **Integrity checks:**
  - types OK (no collapsed type);
  - RA TRAIN contamination audit `12bd63d8` (planted control 200 / 200);
  - exact package parity of typed-final 1,600, css15 6,547 and public231 231, before and after the real download,
    plus AutoModel against the native runtime;
  - Hub `trust_remote_code` smoke (Transformers 5.18 site `93df9002…`);
  - verify_bundle, revision diff, collection order, card HTTP and links;
  - gate evaluate PASS (`index-first`).
- **Card:** default generator (`v2.release.card_index`, audited footnote, Index input `42055aca…`), banner A, assets
  receipt `980fe5c6…`. Spec `dev2-0p8b-ixf.json`; decision `Decision-2.0-Eos-0.8B.decision.ixf.json`.
- **Purge:** the superseded weights of `9c7f3ea0` (2.02 GB) were deleted with `rewrite_history=False` after
  `hf_headroom.sh`. Commits and refs are unchanged, the old weights are no longer served and main is still served.
  Headroom is 47.45 GB; the node copy `inputs/dev2-0p8b-bf16/checkpoint` stays.

### Decision-2.0-Sol-2B: M15 `2b-RASDML`, revision `1b7c47eafa3ffec1f4f4b79b0abfe9309d583439`

The coordinator's 13:05 choice (M13 `2b-RASD`) was superseded at 15:30 / 15:35 UTC+8 by the user's directive: release
the 2B candidate with the largest Index lower bound, M15's soup `2b-RASDML` measured on its BF16 release copy by the
Index sweep. `2b-RASD` (staged, not uploaded) and `2b-RA-a75` are not released.

- **Weights:** M15's soup (node F build, FP32 `8c8e98e3…`) → `v2.release.bf16_copy` identity `e20df76c…` (receipt
  `51b10eae…`), equal to the package the Index sweep scored.
- **Index evidence:** the sweep's IX1 run `IS-2b-RASDML-bf16` on exactly these weights (node C, 120,226 rows);
  full-panel and transfer-only paired bootstraps vs DEV2.0-2B (2,000 replicates, seed `20261002`): both 95% lower
  bounds > 0. Values are private.
- **Formal collection:** M15 ran none. It ran on node B GPU1 (M16 formal path, `M6_FORCE_T1`; select `select-ixf2`, the
  soup copied from node F with equal SHA-256 lists): smoke and collection in 6 minutes, sealed, relayed to node A,
  scored and adopted unchanged (T = 1, no calibration).
- **Integrity checks:**
  - types choice / Noul / Score OK (no collapsed type);
  - row-level contamination audit of the whole TRAIN `97157068…` (102,402 lines: M12's RA TRAIN plus M15's copies
    of released multilingual rows), IX1 method, planted control 200 / 200: 76 duplicate-class rows and 0 item rows,
    as for the RA TRAIN alone, so the copies add none;
  - exact package parity of typed-final 1,600, css15 6,547 and public231 231 (max |Δp| 0.0, 0 category changes)
    before and after the real download, plus AutoModel against the native runtime on every scored prompt;
  - Hub `trust_remote_code` smoke (stock site and Transformers 5.18 site `93df9002…`);
  - verify_bundle, revision diff, collection order and gate evaluate PASS (`index-first`). Card HTTP and links failed
    once on a GitHub 504 for the repository link (every Hub target passed); both re-run PASS before the purge.
- **References (not gating):** v3 52.074 vs 53.437, −1.36 [−3.18, +3.62]; human transfer [−.044, +.075]; vs Sol 1.0
  16K +6.29 [+3.53, +10.57]; Decider 2B +2.57 [−3.10, +5.85]; This-That 1.2 +5.96 [−1.88, +12.63]; public 231 166 vs
  171 (McNemar p .227); mlx-diag and C1 not run.
- **Card:** default generator (`v2.release.card_index`, audited footnote, Index input `97a95897…`), banner A, assets
  receipt `c7ed5810…`. Spec `dev2-2b-ixf.json`; decision `Decision-2.0-Sol-2B.decision.ixf.json` (`813d0fc2…`).
  Published in parallel with the Vega-27B release (15:35: one publisher per repository).
- **Purge:** the superseded weights of `8ed41433` (4.79 GB) were deleted with `rewrite_history=False` after
  `hf_headroom.sh` (47.43 GB before the upload). Commits and refs are unchanged, the old weights are no longer served
  and main is still served. Headroom is 39.94 GB; the node copy `inputs/dev2-2b-bf16/checkpoint` stays.
