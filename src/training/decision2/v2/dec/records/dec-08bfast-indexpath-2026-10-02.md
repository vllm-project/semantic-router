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
- Releases: section 6.

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

RELEASE-SECTION
