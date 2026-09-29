# DEV2.0-8B (9B tier, candidate K-a13) release: state file

Updated: 2026-09-29 15:50 UTC+8 (release-9b worker 1; DONE up to, not including, `--collect`)
Branch `xunzhuo/decision-2-training-release-9b` (worktree `/home/xunliu/code/vllm-sr-dev2-release-9b`, merge-only).
Gist file `07f-decision-2-release-9b.md`. Full record: `../dev2-8b-release-2026-09-29.md`.

## Result

- Private `llm-semantic-router/DEV2.0-8B@0dee801731a69a89c4219a2bf26381ce4d59af48`, manifest `80770483735c…`, 40 files,
  17,947,416,716 bytes, identity `b1ed5a71038b…` (BF16-storage copy of the scored FP32 `b9d973b3…`), T = 1, 7,940,895,744
  loaded parameters, apache-2.0. Not in the collection.
- Verified draft decision `DEV2.0-8B.decision.draft.json` `1db7683ff065…` (node A `runs/release/decisions/`; gate.check
  passes); build-time draft `5db122b0ffe5…`.
- HF storage after: 62.57 GB used, 37.43 GB free. GPU-hours node A GPU6: 0.628 (lease `status=idle`).

## Done (UTC)

- 05:59 GPU6 lease taken; 06:03 calibration rule: CAL698 rejected (typed-DEV and CSS-pilot Brier worsen) -> T = 1.
- 06:10 T = 1 derivation of the sealed CAL698 run (0 answer changes); derived REPORT `66106ef9…`, paired `ae559e9a…`.
- 06:15 BF16-storage copy `b1ed5a71…` (17.95 GB); 06:33-06:43 no-upload parity run: 0 answer changes on all four panels.
- 07:07 and 07:12 upload runs stopped before any upload (card calibration sentence; eval 9B gate record appeared).
- 07:22-07:41 release run from mirror `05914b8e8` (gate disclosures on the card, `vendor_source` = scored mirror
  `3277dec9d`): 15/15 steps, parity 0 changes pre and post upload, card HTTP 12/12, links 15/15, gate evaluate 6/6.
- Receipts copied into this directory; verified draft written and installed.

## Next (coordinator)

1. Final decision `runs/release/decisions/DEV2.0-8B.decision.json` (status final, decided_by; same identity, report,
   paired file); point the spec's `gate_receipt` at it (commit, push, mirror).
2. JevArena-C1 line in `card.text.confirmation` (card-only revision), then `ops/release-upload.sh` with `--upload --collect`.
