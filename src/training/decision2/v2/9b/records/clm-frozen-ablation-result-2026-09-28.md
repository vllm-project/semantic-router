# 9B CLM architecture experiment: frozen-feature ablation result (Milestone 1)

**Closed (coordinator decision, 2026-09-28):** the CLM disaggregated readout is
dropped as a 9B decision architecture; this is a completed negative finding.
The cached shortlister for ≥ 32 options remains only an optional serving note.
Hard negatives, the one helpful CLM component, are carried to the fine-tuned
route as a candidate factor ([Milestone 2 preregistration](lux9b-m2-continuation-prereg-2026-09-28.md)).

**Disposition: development HOLD for every frozen-backbone head, including every
CLM-style arm; the CLM disaggregated route is decisively worse than the
ordinary joint Decision head on transfer.** All numbers below are development
readouts (SELECT 700, CAL 700, typed DEV 1,600, CSS pilot 1,430); none is a
release score, and no JevArena v3, public JevBench or Hugging Face run
followed. Protocols: [extraction](frozen-feature-extraction-protocol-2026-09-28.md),
[preregistration](clm-frozen-ablation-prereg-2026-09-28.md),
[CLM source study](clm-source-study-2026-09-28.md). Aggregate numbers:
[`clm-frozen-ablation-result-2026-09-28.json`](clm-frozen-ablation-result-2026-09-28.json).

## Sequence (all gates in order)

1. Pins: official `Qwen/Qwen3.5-9B@c2022362…` and `-Base@68c46c4b…`
   (Apache-2.0, equal to the Hub head), own `Decision-1.0-Lux-9B@bd45a30a…`;
   7,936,684,544 loaded text parameters each.
2. Extraction preflight r1 failed its frozen parity gate and exposed prose
   token inflation; the prospective amendment (single-prompt joint forwards,
   native-segment disaggregated texts) passed preflight r2 and full
   extraction for all three sources (TRAIN/SELECT/CAL all valid).
3. Own-Lux TRAIN teacher gate passed: published example max drift 0.0199,
   categories identical, 7,324/7,324 rows (file `abaa1113…`).
4. Grid r1 was invalid (a loss-dictionary bug removed every gradient) and was
   also killed by a duplicate launch; both fixed, grid r2 ran all 17
   source×arm cells × 3 seeds (51 heads) with every technical gate passing
   (nonzero step-1 gradients, finite losses, zero-drift reload).
5. Heads sealed and pushed (`59c4b9ed2`, seal `f152c1dc…`) before any DEV/CSS
   feature existed; readout features extracted gold-free; joint prediction
   seal `0aaef4ad…` written before either key was read; unchanged scorers.
6. Post-seal pipeline check: Lux's **own published head** on the frozen Lux
   readout features gives T .875625, H .568818, proxy **70.574** (Choice
   799/800, Noul 272/400, Score 330/400) versus Lux 1.0's historical same-input
   receipt T .8675, H .57011, proxy ≈70.326. The frozen feature path is
   faithful; the results below are real transfer outcomes of the new heads.

## Development readout (seed mean ± SD over 3 seeds; per-type counts: primary seed)

| Source | Arm | Layer | SELECT /700 | Typed T | CSS H | Proxy | Choice/Noul/Score |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| official posttrained | A0J ordinary head, joint | 32 | 515.0 | .4390 | .2892 | **35.62 ± 0.95** | 323/187/208 |
| official posttrained | A0D ordinary head, disaggregated | 32 | 474.0 | .3958 | .0488 | 13.90 ± 0.12 | 234/192/208 |
| official posttrained | A1 raw embedding | 16 | 284.0 | .3881 | .0591 | 15.15 ± 0.00 | 221/192/208 |
| official posttrained | A2 dual projection + InfoNCE | 16 | 369.7 | .4010 | .0500 | 14.15 ± 0.67 | 235/192/208 |
| official posttrained | A3 + hard negatives | 16 | 454.7 | .4002 | .0622 | 15.70 ± 1.33 | 227/192/208 |
| official posttrained | A4 + own-Lux replay | 16 | 450.3 | .3750 | .0817 | 17.30 ± 3.87 | 234/192/208 |
| official Base | A0J | 32 | 517.7 | .4352 | .2347 | 31.96 ± 0.46 | 284/187/208 |
| official Base | A0D | 32 | 446.3 | .4008 | .0707 | 16.80 ± 1.04 | 230/203/208 |
| official Base | A1 | 32 | 266.7 | .4113 | .0958 | 19.85 ± 0.00 | 242/208/208 |
| official Base | A2 | 16 | 366.3 | .3481 | .0623 | 14.48 ± 1.88 | 243/192/85 |
| official Base | A3 | 16 | 448.3 | .3290 | .0836 | 16.35 ± 2.65 | 216/192/107 |
| official Base | A4 | 16 | 449.3 | .3417 | .1007 | 18.28 ± 2.92 | 229/192/107 |
| own Lux 1.0 (frozen) | A0J | 16 | 610.7 | .7712 | .3072 | **48.67 ± 1.68** | 779/198/230 |
| own Lux 1.0 (frozen) | A0D | 24 | 568.3 | .5096 | .1220 | 24.90 ± 1.51 | 325/210/285 |
| own Lux 1.0 (frozen) | A1 | 24 | 321.3 | .4496 | .1094 | 22.18 ± 0.16 | 260/193/256 |
| own Lux 1.0 (frozen) | A2 | 24 | 524.7 | .4708 | .1317 | 24.84 ± 2.41 | 326/237/142 |
| own Lux 1.0 (frozen) | A3 | 24 | 557.0 | .4800 | .1404 | 25.95 ± 0.42 | 310/243/183 |

Comparators, not re-run: official LoRA shared head 61.255 (T .683, H .549,
Score 174/400); Score-cardinality 63.655; Lux 1.0 historical 70.326; Lux's own
head through this frozen path 70.574. All CSS invalids (2/1,430 per run) are
the same two over-length items as prior arms; typed DEV had zero invalids.

SELECT did not predict transfer: every frozen head was competitive in-family
(for example posttrained A0J 515/700, Lux A0J 611/700) but collapsed on the
out-of-family typed rule families and the human CSS tasks. Many heads predict
a single Noul polarity (≈192/400) and a single Score level (208 = the gold
majority) on DEV; disaggregated heads often predict one label per CSS task.
Even on Lux's own backbone, a head retrained on the current 7,324-row TRAIN
loses 21.9 proxy points against Lux's co-trained head, so the limiting factor
is the readout's training data and coupling, not the frozen features' content.

## Paired 95% intervals (primary seeds, 2,000 draws, typed within family, CSS within task)

| Contrast | Δ proxy | 95% interval |
| --- | ---: | --- |
| CLM A3 − ordinary A0J (posttrained) | −18.18 | [−21.49, −15.26] |
| CLM A4 − ordinary A0J (posttrained) | −16.70 | [−19.79, −14.07] |
| A0D − A0J: disaggregated vs joint encoding, same head | −21.31 | [−23.78, −18.24] |
| A3 − A2: explicit hard negatives | +3.13 | [+0.33, +5.04] |
| A4 − A3: own-Lux soft replay | +1.48 | [−1.20, +4.12] |
| A2 − A1: dual projection vs raw cosine | −1.11 | [−2.98, +1.64] |
| A3 − A0D: CLM objective vs ordinary head on disaggregated features | +3.13 | [+0.32, +5.12] |
| Base A0J − posttrained A0J | −3.82 | [−6.90, −1.23] |
| Lux A0J − posttrained A0J | +10.95 | [+7.17, +14.46] |
| Lux A3 − Lux A0J | −20.49 | [−23.95, −16.99] |
| Base A3 − Base A0J | −12.52 | [−15.34, −9.58] |

Preregistered CLM rule: the CLM route is **worse** (not within 1.0 of A0J) on
every source. Against the source-study hypotheses: (1) frozen official
features with heads trained on the current TRAIN do not suffice (35.6 versus
61.3 for the LoRA-adapted backbone on the same data), although frozen Lux
features with Lux's co-trained head do (70.6); (2) separate state/candidate
encoding is the dominant cost and is rejected; (3) learned projections over
raw cosine are not established; (4) explicit hard negatives help modestly;
(5) soft replay is not established; (6) agent trajectories were not tested.

## Absolute versus relative Score (typed DEV three-level Score, primary seeds)

| Source / arm | Absolute acc / Brier / ECE-10 / RPS | Relative acc / Brier / ECE-10 / RPS | Middle-level recall (abs) | Rule |
| --- | --- | --- | ---: | --- |
| Lux A0D | .713 / .204 / .068 / .114 | .685 / .234 / .193 / .121 | .81 | pass |
| Lux A1 | .640 / .262 / .162 / .153 | .268 / .344 / .103 / .241 | .68 | fail (ECE) |
| Lux A0J | .575 / .299 / .266 / .173 | .555 / .382 / .384 / .284 | .24 | fail (ECE) |
| posttrained A0J | .520 / .308 / .089 / .210 | .460 / .307 / .032 / .203 | .00 | fail (Brier) |
| posttrained A3 | .520 / .310 / .081 / .212 | .268 / .567 / .499 / .344 | .00 | pass, degenerate |

Where the representation carries Score information (Lux features), the
absolute cumulative-link head is better calibrated than the candidate-relative
softmax of the same model, with a non-trivial middle level (Lux A0D: 285/400,
the best Score of any new head; Lux's own head scores 330/400). On official
frozen features both readouts collapse to the majority level; the rule's
"passes" there only reflect base-rate calibration and are disclosed as
degenerate, not as a validated Score.

## Serving: latency, throughput and candidate cache (one MI325X, BF16 compute)

Official posttrained backbone, heads timed only. Production kernels (FLA 0.5.2
from the pinned image) and the reference Gated DeltaNet path used for features:

| Options | Joint p50 (FLA / ref) | Disaggregated cold | Cached candidates | Encoder tokens joint → cached |
| ---: | --- | --- | --- | --- |
| 2 | 27.9 / 49.1 ms | 54.5 / 85.4 ms | 27.2 / 40.4 ms | 135 → 54 |
| 10 | 34.7 / 61.2 ms | 57.3 / 83.1 ms | 27.4 / 40.5 ms | 383 → 54 |
| 32 | 56.0 / 103.7 ms | 72.6 / 109.8 ms | 27.5 / 38.4 ms | 1,087 → 54 |
| 128 | 164.8 / 332.4 ms | 174.5 / 274.5 ms | 28.7 / 43.5 ms | 4,187 → 54 |
| 255 | 331.1 / 650.4 ms | 306.6 / 475.6 ms | 28.8 / 39.6 ms | 8,483 → 54 |

A full cache hit (state and candidates) costs 0.1 ms. Batch-32 throughput:
158 (FLA) / 115 (ref) requests/s joint versus 293 / 192 with cached
candidates. On the real panels the cache cuts encoder tokens from 362,011 to
230,972 (typed DEV) and from 421,265 to 182,943 (CSS pilot). The preregistered
serving criterion (≥2× at ≥10 options, latency or tokens) is met on tokens
(7.1× at 10 options) but latency reaches 2× only from 32 options with
production kernels; at the 2–5 options typical of System One questions the
cached path is 1.0–1.3× faster and a cold request costs twice a joint pass.
Joint prompts are also batch-shape sensitive in the reference runtime
(padded-batch versus single final-layer cosine down to 0.969); disaggregated
short texts are not (≥ 0.9997). Batch-shape sensitivity under FLA was not
measured.

## Failures and stop records

| Event | Cause | Disposition |
| --- | --- | --- |
| Extraction preflight r1 | Batched joint parity 0.987 < 0.999; prose inflation 3,888 → 4,421 tokens | Recorded; method amended before any full run; threshold unchanged |
| Grid r1 (two sources) | Loss dictionary returned detached terms (zero gradients); a duplicate launch plus name-based cleanup killed both containers after 95 s | Invalid; fixed with fail-closed gradient check, per-arm gradient test and a locked launcher; restarted as r2 from scratch |
| Readout mirror attempt | Backgrounded tar lost its input when the session closed | Nothing ran; mirror and launch separated |

## Resources

1.720 GPU-hours on node A GPU2–4 (container wall × one GPU): extraction and
preflights 0.809, readout-time extraction 0.186, head grids 0.651 (including
0.053 for the killed r1 grids), predictions 0.009, benchmarks 0.064. Teacher,
sealing, scoring, Score validation, bootstrap and the Lux pipeline check ran
without a GPU.

## Key digests

Extraction manifests: posttrained `d5c47aad…`, Base `cc410c42…`, Lux
`72940c57…`; readout manifests `aad28df0…`, `f5cee22e…`, `82c1ba62…`; teacher
`abaa1113…` (summary `71130156…`); heads seal `f152c1dc…`; prediction seal
`0aaef4ad…`; scores `8e71e620…`; Score validation `9b53c56f…`; paired
`7de4bf14…`; Lux pipeline check `cd744e83…`; benchmarks `fde0f3d1…` (ref),
`f3475afd…` (FLA). Private run artifacts stay on the node.

## Route recommendation for Milestone 2

1. **Lux continuation is the 9B release route.** Keep Lux's joint encoder and
   co-trained candidate head (70.3–70.6 development proxy; current-package
   post-key v3 66.268 / public231 183) and continue it on admitted data arms
   with own-Lux soft replay (teacher already gated). The official Qwen3.5-9B
   fine-tuned route stays a control (61.3–63.7).
2. **No frozen-backbone head release**, CLM or ordinary: the best is 48.7.
3. **CLM disaggregated readout is not the decision path.** It is worth keeping
   only as an optional cached shortlister for ≥32-option Choice, with the
   joint model deciding.
4. **Absolute Score readout** is the transferable piece: test a zero-initialized
   cumulative-link Score readout on the Lux continuation, where Score is
   Lux's weakest v3 axis (228/400).
