# Decoder Milestone 6 — results (successors for DEV2.0-4B / 2B / 0.8B; 2026-09-29)

Preregistration [`dec-m6-prereg-2026-09-29.md`](dec-m6-prereg-2026-09-29.md) (`68d31c358`); data lock
[`dec-m6-datalock-2026-09-29.md`](dec-m6-datalock-2026-09-29.md) (parts 1 / 2a / 2b `8f5699bdf`, `54b18b6c3`,
`4ee4f66b2`); running state [`m6-state.md`](m6-state.md). Development readouts are never release scores; v3 / public
231 are post-key same-panel comparisons. Nothing was uploaded to HF.

## Bottom line

- **No successor in any tier.** DEV2.0-4B `452f1332`, DEV2.0-2B `5ad3e9a3` and DEV2.0-0.8B `f458c34c` stand.
- **4B:** no preregistered finalist, so no formal run.
  - N6D (2× XL dose, own-Lux on all rows) gave the largest typed-DEV gain: T .729 vs .704 at β = 1.
  - But every N6D point's CSS-pilot three-task mean was below the incumbent's (.555–.560 vs .5625), so the rule's
    human-transfer guard removed the whole line.
  - N6A (N4XF recipe, new seeds) peaked at G = +.0087, under the 0.01 minimum. N5BN and Nox had no eligible point.
- **2B:** the finalist `2b-S6X-b1_3` (⅓ toward the XL-mixture S6X soup) is −1.88 [−2.52, +1.15] vs the current
  revision. It fails item 1.
- **0.8B:** the finalist `08b-E6K-b1_3` (⅓ toward the own-Eos-KL soup) is −0.39 [−1.61, +2.16]. It fails item 1.
- **Typed-DEV gains did not transfer to typed FINAL** in either formal run: 2B T .546 vs .543, 0.8B .568 vs .573,
  against development gains of +.013 and +.040. This is the known typed-DEV / FINAL composition gap.

## Development selection (16K, kernel path; `m4_rules.py alpha`, reference = incumbent)

T is typed-DEV family macro, H3 the CSS-pilot three-task mean, P the proxy.

| Tier | Reference I (T / H3 / P) | Line: best points (T / H3) | Pick |
| --- | --- | --- | --- |
| 4B | .7044 / .5625 / 61.47 | N6D β1 .7294 / .5550, β⅔ .7281 / .5578; N6A β½ .7131 / .5647; N5BN β⅓ .7131 / .5596; Nox γ⅓ .7137 / .5524 | none (N6D, N5BN, Nox: no eligible point; N6A: G\* < 0.01) |
| 2B | .6100 / .4278 / 47.14 | S6X β⅓ .6231 / .4361 (β ½, ⅔, 1 fail the Score floor); S6D best β⅓ .6038; Sol flat | `2b-S6X-b1_3` |
| 0.8B | .6131 / .3872 / 40.73 | E6K β½ .6562 / .3916, β⅓ .6531 / .3949, β1 .6556 / .3831; Eos γ⅓ .6362 / .3437 | `08b-E6K-b1_3` (smallest eligible β with G ≥ ¾·G\*) |

Selection files are on each tier's node under `/data/dev2/runs/dec/m6/select/<tier>-{rules,finalists}.json`: node B
for 4B and 2B, node A for 0.8B.

## Formal (post-key same-panel, 16,384 tokens, T = 1; scored on node A)

| Item | 2B `2b-S6X-b1_3` | 0.8B `08b-E6K-b1_3` |
| --- | --- | --- |
| Collection | node B, `cache-frozen-2b` from the node-B S2T reference (0 differing answers vs the node-A run and the T = 1 binding) | node A GPU5, copy of E8F's persisted cache |
| Revision (per-file list) | `e5131501…` | `e3f04552…` |
| v3 / T / H | 51.561 / .5463 / .4867 | 49.848 / .5678 / .4376 |
| Current revision | 53.437 / .5434 / .5255 | 50.236 / .5734 / .4401 |
| 1. v3 vs current revision | −1.88 [−2.52, +1.15] **FAIL** | −0.39 [−1.61, +2.16] **FAIL** |
| 2. H vs current revision | [−.050, +.017] not significantly below | [−.023, +.041] not significantly below |
| 3. Types | OK; C / N / S 451 / 573 / 169 (S2T 445 / 567 / 175) | OK; 522 / 611 / 102 (E8F 529 / 611 / 107) |
| 4. mlx-diag, card-eligible parts | not collected (item 1 decides) | +.0070 [+.0010, +.0130] |
| 5. Tier gate vs own 1.0 | vs Sol 1.0 16K +5.78 [+3.46, +10.00] | vs adopted Eos 1.0 +7.30 [+4.04, +13.62] |
| 7. public 231 (`gates public231`) | 171 vs 171, OK (p 1.0) | 160 vs 156, OK (p .29) |
| Best same-size peer | Decider 2B +2.06 [−3.12, +5.48] | Intern-0.8B +6.31 [+0.96, +10.60] |

- Run directories are node A `/data/dev2/runs/dec/formal/m6/m6-{2b-S6X-b1_3,08b-E6K-b1_3}`. The successor records
  are in `formal/m6/successor/`.
- Item 6 was not run to completion: item 1 already fails. The new training files have empty exposure receipts
  (data lock).

## Findings

1. **At 2B the XL mixture does not carry over the way it did at 4B.**
   - The pure S6X soup was the best 2B development artifact (P 49.39 vs 47.14; typed Choice 550 vs 489).
   - But its Score fell (212 vs 257), so only the β = ⅓ point was eligible.
   - Formally, CSS15 human transfer dropped −.039, driving the v3 loss.
   - The 2× dose (S6D) was worse than 1× on development panels.
2. **Own-Eos KL at 0.8B** held human transfer and raised the development typed score, but typed FINAL went down
   slightly. The formal −0.39 is within noise; the arm does not beat E8F.
3. **4B's strongest development signal is N6D's typed gain**, and it came with a small CSS-pilot drop. The pilot is
   a three-task development signal that the eval track rates as unreliable for human transfer (HT-DEV note 09:25). So
   the guard may have removed a real gain, or a real loss. Only a preregistered formal test can tell.
4. **The development proxy again did not predict the within-tier formal order.** Both finalists were ahead on P and
   behind on v3.

## Items for the coordinator

1. No successor: keep DEV2.0-4B / 2B / 0.8B at their current revisions. No release hand-off.
2. **Proposed next 4B test (not run; needs a preregistration):**
   - a formal run of the N6D soup and N6D β = ⅔, with the successor rule as the only selection;
   - reason: its typed-DEV gain (+.025) is the largest any 4B arm has shown, and the CSS-pilot guard that removed it
     is a weak signal;
   - cost ≈ 0.6 GPU-h; the weights are on node B `m6/soup/N6D/build/N6D-soup` and `m6/lines/4b/4b-N6D-b2_3`.
3. **Near-match group.** The 3 HotpotQA rows the 17:15 note quarantines are in `c7d51219`, so they are in N6A, S6X
   and the nested 59m mixture as well as in the released DEV2.0-4B. The mixtures were locked before the note. This
   touches public 231 only.
4. **Tooling fix.** `m6-soup.sh` read the wrong seed-marker name, so the chains' soups all reported FAILED. It was
   fixed in `ddc5d2ffe` and the soups were rebuilt on CPU from the finished seeds. The chain FAILED markers are kept
   under `m6/attempts/`.
5. **Shared module.** `v2.dec.teacher_label` now falls back to a 1.0 package's `config.json` temperature
   (`6fe568271`, with tests).
6. **Item-7 tooling.** `m6_successor.py` still marks item 7 "report only". The `gates public231` verdicts above were
   run by hand. A follow-up should make it gate.

## GPU-hours

**≈ 17.8 of 36**:

| Item | GPU-h |
| --- | ---: |
| Node B training: N6D 4.93, N6A 2.82, S6X 1.81, S6D 2.51 (Sol labels 0.17 counted once) | 11.90 |
| Node A training: E6K incl. Eos labels 0.30 | 3.95 |
| Readouts, references, formal and mlx-diag (node B 1.52, node A 0.39) | 1.91 |

No preflight failed and no arm was rerun.
