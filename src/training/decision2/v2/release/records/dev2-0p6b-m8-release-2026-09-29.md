# DEV2.0-0.6B progressive update: `m8-s5-b05` released as revision `b2131337`

Coordinator approval 2026-09-29 18:25 UTC+8 (first progressive-update successor; hand-off
`v2/06b/records/m8-handoff-2026-09-29.md`). The work was interrupted by the Cursor usage limit at ~19:04 and
resumed at 20:45 (nothing had been uploaded or committed before the stop).

## Result

- **Private `llm-semantic-router/DEV2.0-0.6B@b21313375ad77ddf4a8e420fa5195e6e09582043`** (`main`),
  `MODEL_MANIFEST.json` `816827dd836d21ab0b7555605fdc56dd5189ee4462a541a84de27de97f637796`, 33 files,
  597,103,104 loaded parameters, licence apache-2.0. Supersedes `99c4e799` (`m4-t-a7-soup`).
- **Final decision** [`DEV2.0-0.6B.m8.decision.json`](dev2-0p6b-m8-release-2026-09-29/DEV2.0-0.6B.m8.decision.json)
  `3fbe847bca4cde30…` (status final; decided by the coordinator at 18:25), sealed in
  [`release/receipts/gate.json`](dev2-0p6b-m8-release-2026-09-29/release/receipts/gate.json) with profile
  `successor` and `supersedes` = {`99c4e799`, gate `787c31c9…`}. All 12 gate items pass.
- **Weights:** the BF16-storage copy of the unchanged scored `m6-mxcx-soup` weights (`01fae750…` →
  package identity `bb806f31a14d4532a1b5e00984442128f47f46cbbc68e57e7623327c81cfb983`) plus the fixed
  five-level Score offsets +0.039188, +0.203049, +0.079362, −0.151620, −0.169979, rebound to the BF16 hash
  (`score_bias.json` `7d3a060f…`; from the packaged public file `725c16f9…`, scored file `03f0c9c7…`).
- **T = 1** (23:15 rule): CAL698 temperatures rejected on the development panels.
- **Collection** "🎲 Decision 2.0" unchanged: 0.6B, 0.8B, 2B, 4B, 9B, 27B; title unchanged, private.
- **Superseded weights purged** with `rewrite_history=False` (below). Storage 62.09 → 61.20 GB of 100.

## Successor gate profile (`v2/release/gate.py`, `9d63aa817`)

`gate_profile.name = "successor"` replaces item 1 with R1–R7 of the 16:05 / 17:15 successor rule against the
current revision's scored run, reached through that revision's sealed gate receipt → decision → report
(`787c31c9…` → `e22fe181…` → `72cf01de…` = `m4-t-a7-soup/REPORT.json`). The decision binds the profile,
`current_revision` and the SHA-256 of eight evidence files. `python3 -m v2.release.gate profile --spec S`
evaluates it before upload. Tests: `v2/release/tests/test_gate_successor.py` (11).

| Item | Evidence (node A, before upload and again at seal) |
| --- | --- |
| R1 v3 | 48.636 vs 43.541: +5.09 [+2.65, +9.30] (`paired-vs-released.json` `61700400…`) |
| R2 human transfer | Δ 95% [−0.064, +0.061] |
| R3 types | Choice, Noul, Score OK (`types.json` `ce2ca8bc…`) |
| R4 mlx-diag | card-eligible Choice + Noul macro +0.0044 [−0.0058, +0.0144] (`mlx-paired.json` `72660c42…`; released side = the current revision's mlx predictions) |
| R5 tier gates | vs Kai 1.0 low +10.00; v3 ≥ 90% × GLiNER2.5-Decide 42.524 = 38.272; H vs GLiNER2.5-Decide [−0.013, +0.161]; no collapse |
| R6 overlap | 6 M6 training files, 0 exposed groups (`exposure-m6-mxcx-soup.json` `4c2a55d2…`) |
| R7 public 231 | `v2.eval.gates public231` rerun by this worker: 152 vs 142 (+10; McNemar p .087), OK (`dff304c9…`) |

## BF16 (`eb5152033`)

`bf16_copy` gained the single-file backbone layout and `--score-bias` (same offsets rebound to the copy's
hash, provenance in `fit.bf16_rebind`); the builder accepts a `bf16_copy` spec entry (receipt pinned by hash;
the checkpoint must be exactly that copy) and then binds the offsets to the scored run through the receipt's
`source_model_sha256`. Copy receipt `2a991bfb…`: 196 Linear matrices BF16, 114 tensors FP32 bit for bit,
2.38 → 1.50 GB tensor bytes. Staging parity (no upload,
[receipts](dev2-0p6b-m8-release-2026-09-29/bf16/parity-receipts/)) at tolerance 0 on every scored prompt:
typed 1,600, CSS15 6,547, public 231 231, mlx-diag 2,275 — **0 answer changes, max drift 0.0** →
adopted. The release run repeated this before and after the real download (same result).
First staging launch refused its repo name (`dev2-release-staging-06bm8-bf16` does not match the staging
pattern; renamed `…-06bm8bf16`, `a7e1c7dd9`); no GPU work was done by the refused launch.

## Calibration (23:15 rule; `95a683e50`)

`calibrate_frozen --score-bias` fits CAL698 on the logits the package scores (offsets added to Score rows).
CAL698 `19cc1a8c…` on the BF16 package, node A GPU0 (29 s): Choice 0.584, Noul 0.554, Score 0.450
([calibration.json](dev2-0p6b-m8-release-2026-09-29/devcal/calibration.json)). `dev_calibration` on the stored
`m6-mxcx-soup` development readouts (identity `01fae750…`, T = 1; the offsets touch neither panel: typed-DEV
Score is 3-level, the CSS pilot has no Score): typed-DEV Brier .271 → .282, ECE-10 .089 → .114; CSS-pilot
Brier .766 → .769, ECE-15 .070 → .093 — all four worsen → **T = 1**
([receipt](dev2-0p6b-m8-release-2026-09-29/devcal/devcal-m8-s5-b05.json)); disclosed on the card.

## Card

Rendered text: [`package-text/README.md`](dev2-0p6b-m8-release-2026-09-29/package-text/README.md); spec
`v2/release/specs/dev2-0p6b-m8-release.json` (generated by `ops/make_spec.py`).

- Recipe and weight origin (six full fine-tunes, two recipes × three seeds, uniform soup, then offsets);
  training data regenerated from the six M6 train files ([credits](dev2-0p6b-m8-release-2026-09-29/credits/):
  232,754 distinct rows, 142,983 from 37 public datasets, 89,771 program-generated; no NC / research-only /
  unknown source; Lux soft targets on all 232,754 rows); licence components unchanged → apache-2.0.
- Same-panel table and charts: Kai 1.0 (own 1.0), GLiNER2.5-Decide, Bosun v3.1 0.6B, Lex 1.0 (licence filter
  unchanged). Tradeoffs vs Kai: one row (CSS15 mrf 55.5% vs 56.6%).
- Disclosures 1–5 and 8 verbatim from hand-off §5; the public-231 note in the 17:15 wording (card renderer
  `b6dbf0a6a`), with paired gaps: previous revision +10 [0, +20], Bosun +19 [+7, +31], GLiNER2.5-Decide
  +36 [+23, +49], Kai +38 [+23, +53] (+25 [+10, +41] at 8K), Lex +39 [+24, +55]; no peer ahead by more than
  three items in any family, so the two-skill sentence does not apply
  ([receipts](dev2-0p6b-m8-release-2026-09-29/public231/)).
- C1: "The sealed JevArena-C1 comparison measured the previous revision (`99c4e799`); it has not been
  repeated for this revision." Precision for the record: C1 event 2 scored revision `e61b2b44`, whose model
  files are byte-identical to `99c4e799` (card-only difference).

## Verification ([receipts](dev2-0p6b-m8-release-2026-09-29/release/))

All 17 `release.sh` steps pass: build; examples bit-identical in two processes before and after download;
card example; parity pre and post (above); ensure private; upload; real `hf download` of `b2131337` and
re-hash 33/33; loaded parameters 597,103,104; readback (private, no card problems); seal; collect with
`--already-collected`; readback after collect. Extras: `gate evaluate` passes, Hub HTTP card check passes,
`hub_links` 14/14, collection order unchanged.

## Purge of the superseded weights ([tool](dev2-0p6b-m8-release-2026-09-29/ops/purge_superseded_lfs.py.txt), `6fe2eb03f`)

Targets: `backbone/model.safetensors` `7e2d1f9d…` (2,384,233,112 B) and `decision_head.safetensors`
`70dc7451…` (4,213,584 B) of `99c4e799` (also the weights of `7b5d3ff2`, `cb2bfd76`, `a87eeb72`, `e61b2b44`).
`tokenizer.json` is shared with the new revision and kept. Node copy
`/data/dev2/runs/release/inputs/dev2-0p6b/staging-06bm4-62c61c10/` re-hashed equal before deletion; it is the
durable store. `permanently_delete_lfs_files(..., rewrite_history=False)`; after 20 s: commits (7) and refs
unchanged, exactly the two objects gone (LFS objects 6 → 4), old weight paths return 403, every LFS file of
`main` still served ([plan](dev2-0p6b-m8-release-2026-09-29/release/purge/plan.json),
[apply](dev2-0p6b-m8-release-2026-09-29/release/purge/apply.json)). The first plan (`plan-v1-path-bug.json`,
nothing deleted) exposed a path-prefix bug in the tool's served check; fixed before apply.
Old revisions' weights are no longer downloadable from the Hub; C1 / older records that cite them rest on the
node copy and the per-file SHA-256.

## Incidents and notes

- Node A mirror race (18:52): two concurrent `mirror_to_node.sh` runs of `d957ba1b9`; the loser's staging
  dir was moved inside the finished mirror (`mv` into an existing directory). The stray nested `.incoming-*`
  copy was removed and the mirror re-verified. Tooling suggestion for the coordinator: `mv -T` / re-check the
  target before the final move.
- Shared-module changes (small separate commits with tests): `gate.py` successor profile; `bf16_copy.py`
  and `build.py` BF16 link; `card.py` public-231 note (affects every card at its next render);
  `calibrate_frozen.py --score-bias`.

## GPU-hours

Node A GPU0 (shared lease `owner.release`, now idle): BF16 parity staging 301 s, CAL698 fit 29 s, release
run 568 s → **0.249 GPU-h**. CPU only: BF16 copy, development calibration, public-231 guards, purge.

## Commits

`9d63aa817` gate profile · `eb5152033` BF16 tooling · `b6dbf0a6a` card note · `a957198e3` / `a7e1c7dd9`
BF16 staging spec · `95a683e50` calibrate_frozen · `dcd1ad55d` spec, decision, receipts, credits ·
`6d3de6070` / `6fe2eb03f` purge tool · this record.
