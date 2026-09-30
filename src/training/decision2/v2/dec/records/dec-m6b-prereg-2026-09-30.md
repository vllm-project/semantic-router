# Decoder Milestone 6b — preregistration (formal test of the N6D lead against DEV2.0-4B; 2026-09-30)

Written 2026-09-30 ≈02:40 UTC+8, before any M6b GPU job. Assignment: coordinator note 2026-09-30 01:50, decision 1
("M6b approved: a preregistered formal test of the N6D soup and its ⅔ point under items 1–8"). Development readouts
are never release scores; v3 / public 231 / C1 are post-key same-panel comparisons. Nothing goes to HF.

## Question and why it is tested formally

M6 ([results](dec-m6-results-2026-09-29.md)) found the largest 4B typed-DEV gain so far on the N6D line (N4XF's XL r2
recipe at 2× dose, own-Lux KL on all rows), but every N6D point's CSS-pilot three-task mean H3 was below the
incumbent's, so the M6 rule's human-transfer guard removed the line. The pilot is a weak human-transfer signal (the
eval track's HT-DEV note: no development panel orders within-tier candidates on formal H). M6b asks one question:
**do the N6D soup or its ⅔ point pass successor-rule items 1–8 against DEV2.0-4B `452f1332`?** It is a formal test of
a lead identified after the M6 development readouts were seen; that post-hoc choice is disclosed with any result.

## Finalists (fixed; no development selection)

Both are existing CPU-built weights on node B; nothing is trained. `I` = the released N4XF soup.

| Slot | Point | Weights | Checkpoint (node B) and per-file list | M6 16K development readout (T / H3 / P; typed C / N / S) |
| --- | --- | --- | --- | --- |
| 1 | `4b-N6D-b1` | the N6D 3-seed soup (`m6-xl-full-59m` `160812e2…`, own-Lux `lux-all-59m` `a1bafad5…`, KL 1.0) | `m6/soup/N6D/build/N6D-soup`; list `m6/lines/4b/4b-N6D-b1/files.sha256` `6e270c46…` | .7294 / .5550 / 62.49; 540 / 302 / 325 |
| 2 | `4b-N6D-b2_3` | ⅔ N6D soup + ⅓ `I` (`v2.dec.soup`, members `[I, A, A]`; model `9e18fdae…`) | `m6/lines/4b/4b-N6D-b2_3/build/4b-N6D-b2_3`; list `76b2db85…` | .7281 / .5578 / 63.25; 532 / 284 / 349 |
| ref | `4b-I` (DEV2.0-4B weights, `11b5ca1c…`) | — | — | .7044 / .5625 / 61.47; 501 / 264 / 362 (M6 16K reference) |

Both failed two M6 development rules: H3 below the incumbent's .5625, and the typed-DEV Score floor (349 and 325 <
350). The formal types gate (item 3) tests collapse, not that floor; the Score counts are reported.

`v2/dec/ops/m6b/m6b_finalists.py` writes these two rows (with their M6 development values) in the M6 finalists schema
to `/data/dev2/runs/dec/m6b/select/4b-finalists.json` on node B, so the M6 formal scripts stage them unchanged
(`M6_SELECT`).

## Formal runs (the 4B's scored settings; the M6 prereg's 4B procedure)

- **Collection:** node B, image `dbe5f32b…`, 16,384 tokens, the frozen runner `v2/eval/run_same_panel.sh` (gold
  never mounted), each run on its own `cp -a` copy of `formal/m5/cache-frozen` (`f6d0f920…`). This is the procedure
  whose node-B N4XF reference (`formal/m5/m5-ref-N4XF-soup`) reproduced the node-A scored answers exactly.
- **Commands:** `M6_FORMAL_ROOT=/data/dev2/runs/dec/formal/m6b M6_SELECT=/data/dev2/runs/dec/m6b/select`
  `ops/m6/m6-formal.sh 4b smoke|finalist <point>` (GPU3 / GPU4, one job per GPU, recorded co-tenant lease entries).
  `stage` makes a 16K CAL698 fit and applies the 23:15 rule (`v2.release.dev_calibration`); the package ships T = 1
  unless the rule adopts CAL698. Answers are unaffected either way.
- **Scoring:** relayed gold-free to node A with manifest checks (`m6-relay.sh pull`), then `m6-score.sh 4b <run>`
  (seal check, report, paired CIs vs the bar, the tier's own 1.0 and peers, types gate), `m6-relay.sh mark`, the
  `mlx-diag` collection on node B (copy of `cache-frozen-mlx` `65d7d38f…`), `m6-score.sh 4b mlx <run>` (paired vs
  `formal/m5/m5-ref-N4XF-soup-mlx`), `m6-score.sh 4b overlap <run>` and `m6-score.sh 4b successor <run> <run>`.
- **Smoke first:** 8 items per finalist on a throwaway cache copy. A failed smoke or collection stops that finalist;
  nothing is rerun.

## Successor rule (items 1–8, vs the current revision's T = 1 scored run)

Bar: DEV2.0-4B `452f1332`, weights N4XF soup `11b5ca1c…`, scored run `runs/release/dev2-4b-t1-derived` (v3 63.151).

1. v3 paired 95% CI lower bound > 0 (`same_panel compare`, joint bootstrap, 5,000 replicates, seed 20260927).
2. Human transfer not significantly below: `axis_ci95.H.delta.high` ≥ 0.
3. No type collapsed (`v2.eval.gates types`).
4. `mlx-diag` card-eligible type macro (Choice + Noul) paired 95% upper bound ≥ 0 (`v2.dec.mlx_paired`, vs the
   node-B N4XF reference).
5. Tier gates: lower bound > 0 vs the adopted Nox 1.0; v3 ≥ 55.7; H not significantly below Decider 4B or Jet v6.2;
   no type collapsed.
6. No overlap exposure: (a) the exposure receipt of `m6-xl-full-59m` (`8453af8a…`, 0 groups) — the only new
   training file; the ⅔ point inherits N4XF's exposure (none); (b) rules 1 and 5 hold with the 84 flagged items
   removed (`overlap_effects run`).
7. **`python3 -m v2.eval.gates public231 --left <finalist> --right <bar-t1>` is not REGRESSION** (loss with exact
   McNemar p < .05). This now gates in `m6_successor.py` (fix committed with this record, with tests).
8. **JevArena-C1 v1.2 post-key guard, through the eval custodian:** `gates c1` against the registered 4B baseline
   (48.38, `m4/c1-event3/cand4b`) is not REGRESSION. This track never opens C1.

**Choice and item 8.** Among finalists passing items 1–7: the highest paired lower bound vs the best same-size peer
(Decider 4B, 61.882), then the higher lower bound vs the bar, then slot. Only that one (the "C1 candidate") goes to the
custodian: the ledger allows one successor per tier baseline, and a second collection would need a coordinator
approval. Hand-off: its staged package is relayed to node A (hash-checked against its list) and wrapped as a frozen
package (checkpoint + `MODEL_MANIFEST.json` with per-file SHA-256 and the weights identity), with its formal run's
autotune cache, its stored typed FINAL + public 231 predictions, and a `dev2-c1-postkey-spec/1` spec (role
successor, image `kernel`, the formal adapter spec). The custodian verifies, preflights (exact parity) and collects.
Also requested of the custodian: the content recheck its ledger requires for a model trained on an unreleased-arm pool
(`m6-xl-full-59m`).

A candidate passing items 1–8 is handed to the coordinator for release (the 16:05 fast path) with its package paths
and hashes. **Multiplicity:** two finalists are tested; the result is reported with both.

**Disclosures carried by either point:** the 3 HotpotQA near-match rows (17:15 quarantine note) are in
`m6-xl-full-59m`, as in the released DEV2.0-4B (public 231 only); the post-hoc choice of this line.

## Budget, GPUs, stop rules

- **Cap 1.5 GPU-h** within the 30 GPU-h shared with Milestone 7: two smokes, two CAL698 16K fits, two formal
  collections, two `mlx-diag` collections. Scoring and the successor evaluation are CPU on node A.
- **GPUs:** node B GPU3 (slot 1) and GPU4 (slot 2), lease entries per the co-tenant rule.
- **Stop rules:** a failed smoke, calibration or collection stops that finalist, recorded, not rerun. No other point
  of the line is tested.
- **Chain rule (16:00 note):** every script is uploaded, size- and SHA-256-checked, then launched in a separate step;
  liveness by container name or PID.

## Code

- `v2/dec/ops/m6/m6_successor.py`: item 7 gates (`--public231`), item 8 read from the custodian's `SUMMARY.json`
  (`--c1`), `status_1_7` vs `status`; tests in `v2/dec/tests/test_m6_successor.py`.
- `v2/dec/ops/m6/m6-score.sh successor`: runs the public-231 guard vs the bar and passes item 8 when present.
- `v2/dec/ops/m6/m6-formal-lib.sh`: `M6_SELECT` overrides the select directory.
- `v2/dec/ops/m6b/m6b_finalists.py`, tests in `v2/dec/tests/test_m6b.py`.

No shared module changes.
