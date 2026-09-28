# Milestone 2 deferred peers and Jev reference status

Eval & peers track, 2026-09-28, amendments A4–A4.6. **Post-key same-panel** evidence;
reports in [`m2-reports/`](m2-reports/). Paired intervals: joint v3 bootstrap, 5,000
draws, seed 20260927, against the tier's own 1.0 model on the same node (27B: against
AutoJev on node B, as there is no 27B 1.0 model).

| Model @ pinned revision | Tier / node | Loaded params | v3 | T | H | Choice / Noul / Score | Public 231 (E/S/H) | Invalid CSS / pub | Δ v3 [95% CI] | mlx-diag type-macro (non-EN) | Dev P |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- | ---: |
| Intern-Decision-0.8B `85a0cc5a` | 0.8B / A | 852,985,920 | 43.535 | .4956 | .3824 | 495 / 431 / 88 | 164 (47/57/60) | 115 / 0 | +0.989 [−2.328, +6.084] vs Eos1 | 58.3 (57.4) | 35.73 |
| Kev-0.8B `9a45d25e` | 0.8B / A | 752,917,824 | 43.217 | .4794 | .3896 | 489 / 407 / 82 | 147 (48/58/41) | 13 / 0 | +0.670 [−3.027, +6.828] vs Eos1 | 64.8 (63.9) | 33.50 |
| This-That 1.2 `c4d1c30b` | 2B / A | 1,881,825,088 | 46.112 | .5250 | .4050 | 529 / 431 / 79 | 147 (48/64/35) | 169 / 37 | +0.531 [−6.362, +9.175] vs Sol1 | 74.2 (73.3) | 47.06 |
| Jet v6.2 `fbc3d2da` | 4B / A | 4,205,751,296 | 60.375 | .6781 | .5375 | 720 / 640 / 125 | 174 (48/69/57) | 3 / 0 | **+3.904 [+0.287, +8.141] vs Nox1** | 79.6 (78.8) | 63.55 |
| Hopper (G) 1.2 `71d991f4` (research-only) | 4B / A | 4,205,751,296 | 58.396 | .6438 | .5297 | 704 / 590 / 136 | 194 (48/70/76) | 0 / 0 | +1.926 [−1.109, +5.381] vs Nox1 | 78.6 (77.8) | 64.95 |
| Nimble 9B v2 `4b8c04d1` | 9B / A | 9,453,092,080 | 62.056 | .7188 | .5358 | 784 / 654 / 107 | 185 (48/68/69) | 21 / 0 | **−3.752 [−7.828, −0.567] vs Lux1** | 76.3 (75.3) | 64.93 |
| Eikos-27B `103a5647` (BF16 sibling) | 27B / B | 27,781,427,952 | 69.201 | .8144 | .5880 | 784 / 588 / 331 | **212** (48/72/92) | 4 / 0 | −3.109 [−5.805, +0.167] vs AutoJev 27B | — | 76.11 |
| Jebadiah 27B `c68db2b5` | 27B / B | 26,895,998,464 | 65.472 | .7419 | .5778 | 616 / 721 / 250 | 177 (48/70/59) | 153 / 36 | **−6.838 [−8.725, −2.703] vs AutoJev 27B** | — | 75.70 |

- **0.8B.** Intern and Kev are within noise of Eos1 (42.547). The tier now has three
  peers: Intern 43.535, Kev 43.217 and JPT-0.8B 40.085. Intern's 115 CSS invalids are
  native 8,192-token rejections; Kev's 13 are its strict state limit.
- **2B.** This-That 1.2 is within noise of Sol1 (45.580) and below Decider 2B (49.499).
  Its invalids are native 1,536-token state overflows. Noul and Score are option
  projections of its native Choice head.
- **4B.** Jet v6.2 is the only deferred peer clearly above its tier's own 1.0 model
  (Nox1 56.470). It remains below Decider 4B (61.882; not paired). It runs on
  unvalidated ROCm (the release targets CUDA). Hopper (G) is within noise of Nox1 and
  has zero invalid answers. It is research-and-demo licensed, so it is for internal
  comparison only and is excluded from cards by the licence filter. Its own kernel
  check found no slow kernels on this image, so the permitted PyTorch fallback was not
  used.
- **9B.** Nimble v2 is significantly below Lux1 on node A (65.808). Its 21 CSS invalids
  are native 8,192-token rejections.
- **27B.** Eikos-27B's public 212/231 is the highest open-weight count so far (AutoJev
  200). Its v3 is 3.1 below AutoJev; the interval includes zero. The artifact is the BF16
  sibling of the board's FP8 entry, which is disclosed. Jebadiah 27B is significantly
  below AutoJev. Its renderer cuts states over 2,048 tokens; the collector never scores
  a cut prompt, which yields 153 CSS and 36 public invalids. It has a large gap between
  typed-DEV (T .873) and typed-FINAL (T .742), mostly on Choice. Both 27B peers ran
  formal and dev panels only.
- **Proxy check (out of sample).** v3 ≈ 19.12 + 0.629·P predicts these models with the
  following errors:

  | Model | Error |
  | --- | ---: |
  | Intern | +1.9 |
  | Kev | +3.0 |
  | This-That 1.2 | −2.6 |
  | Jet | +1.3 |
  | Hopper (G) | −1.6 |
  | Nimble v2 | +2.1 |
  | Eikos-27B | +2.2 |
  | Jebadiah 27B | −1.3 |

  All eight are inside the calibrated ±3 except Kev, which sits on the boundary. Nimble
  and Hopper have the same P (64.9) but differ by 3.7 in v3. This is the |ΔP| < 4 tie
  case the rule sends to the formal runner.

## Technical stops (each recorded, then corrected once)

- **Q2 (Intern).** First stopped at engine load (14.1 s, no prediction) because the
  release's dataclasses need their module in `sys.modules`. Fixed with a regression
  test; Q2b ran once.
- **Q5 (Jet).** Stopped by the operator after 224 s (nothing sealed). Jet's API rejects
  Noul `criteria`, and the first collector counted that as a native rejection. Adapter
  v2 fixed the collector: it appends the true/false meanings to the instructions, as
  the APUS adapter does, maps answers onto the scorer's fields, and treats only the
  16,384-token message as invalid. Two 20-item gold-free smoke runs checked answer
  shapes, then Q5b ran once.

- **Smoke-first rule.** After the Q5 stop, Nimble, Jebadiah and Hopper each ran
  20-item gold-free smoke runs before their single full run. The smoke runs checked
  answer shapes against the scorer's field rules and covered all three types where the
  panels allowed. All three full runs then ran once with no stop.

Shared-adapter changes (separate commits with tests): the Kev and Eikos size tables, the
This-That version table, and the new `inference/intern_decision.py`. The new collectors
`v2/eval/native_{jet,nimble,jebadiah,hopper}.py` live outside `inference/` because Jet's
and Nimble's releases ship their own top-level `inference` module, which would collide
with ours. Existing defaults are unchanged.

**Not run: Rune 26B-A4B.** Its native runtime (surogate) is C++/CUDA 13 only. The
documented transformers fallback would need a parity check against that native path,
and no CUDA hardware is available here, so there is no cheap ROCm path. Every other
deferred peer has now run.

**M2 GPU-hours for peers:** 1.745 in total. That breaks down into 1.59 for full runs,
0.07 for technical stops and 0.08 for smoke runs.

## Jev reference (approved 2026-09-28 13:40, private)

The official API run on v3 and public 231 completed. The model was pinned to
`jev-1.13.0`, which the `jev-latest` alias resolved to. Requests took about 0.15 s each,
and none failed. Its predictions, seal and report stay in private storage on node A and
are not published in this branch, the gist or any card, per the coordinator's rule. It
is labelled separately from open-weight ranks and is never a training target.
