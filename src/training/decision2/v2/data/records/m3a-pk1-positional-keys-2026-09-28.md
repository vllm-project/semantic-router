# M3a item 3 — positional option keys (pk1) for A0 / A0s / A0p / RP-v1q / R2 (2026-09-28)

Plan: [`m3a-prereg-2026-09-28.md`](m3a-prereg-2026-09-28.md) §6. Code: `v2.data.m3.renumber` (rows from the
published files at revision `002e5b42…`; rule `v2.common.option_keys`, A7 rule 7d), launchers
`v2/data/m3/pk1_nodeA_lux.sh` and `pk1_nodeB.sh`. Count-only receipts and reports:
[`m3a/pk1/`](m3a/pk1/), registry [`m3a/m3-registry.json`](m3a/m3-registry.json).

**Published:** private `llm-semantic-router/decision-2.0-training-data` revision
**`d8eae3e4fb5b91871c5aa7c13f0d94ea96e86ea7`**, tree `m3/pk1/` (25 files, readback verified 25 / 25,
`m3/registry.json`, `m3/README.md`). **Release candidates use these files instead of the v1 ones.**

| pk1 file | Replaces (v1) | Rows | Renumbered | SHA-256 |
| --- | --- | ---: | ---: | --- |
| `m3/pk1/A0/train.jsonl` | root TRAIN `61740be4…` | 7,455 | 117 | `766249e76e2ced4b0d238749903965e594a3ffb4472cae848867771aef9742ae` |
| `m3/pk1/A0s/train.jsonl` | `v2/arms/A0s` | 7,299 | 117 | `35c1f98ff0f406feef8bfd1c7ed8f254c096b4f7b3a6e0858bcc92ef43ddc20b` |
| `m3/pk1/A0p/train.jsonl` | `v2/arms/A0p` | 3,709 | 108 | `18bcd366dd13bf9878fbdc3040c0ecc0c95ead451136d2efbc84952038bd9052` |
| `m3/pk1/RP-v1q/train.jsonl` | `v2/replay/RP-v1q` | 1,517 | 22 | `1eb960bc7554532a4a2a834a73a639b4cba0149bb0f67b81fd160e3b26c26b01` |
| `m3/pk1/lux1/A0-train.canonical.jsonl` | `m2/teachers/lux1/A0-train.canonical.jsonl` `dd160420…` | 7,455 | 117 re-derived | `56627939e01b60d867ff3fa7ca8073e7434b63b859eee16b407860ca2ef508d2` |
| `m3/pk1/R2/kai/replay.jsonl` | `v2/replay/R2/kai` | 1,288 | 22 re-derived | `35917bc8370789d4a43a0faf7a23d2a24104664426a74aa4906a22b0292529d4` |
| `m3/pk1/R2/lex/replay.jsonl` | `v2/replay/R2/lex` | 1,288 | 22 re-derived | `8c8a03574fba944000f329d5cfb1368ec65c4fef3eba29479b21fbdf794b0efc` |
| `m3/pk1/R2/eos/replay.jsonl` | `v2/replay/R2/eos` | 1,517 | 22 re-derived | `3181b172f4b77f96211b4e3d5773e5d4dfa538ff80a114b7776744a8898500f6` |
| `m3/pk1/R2/sol/replay.jsonl` | `v2/replay/R2/sol` | 1,517 | 22 re-derived | `9ca626a8c275be51607529083769a2315d8fbc9c11c60caaa1bd2a76bd5ad6b6` |
| `m3/pk1/R2/nox/replay.jsonl` | `v2/replay/R2/nox` | 1,517 | 22 re-derived | `ae7510a80e4b38382c177e68362d5f1a6ce3f50483df4261a524fcb28397a8e7` |
| `m3/pk1/R2/lux/replay.jsonl` | `v2/replay/R2/lux` | 1,517 | 22 re-derived | `5a77f9df060e79f504108430fcc0154c0d6d321c20fdff0db757ea8678cba029` |

Checks: every source file was canonical JSON lines and every unchanged row is byte-identical in its
pk1 file; after renumbering no row of any pk1 file still has construction-order keys (A0 / A0s: 131
`result_<n>` rows, all positional; A0p 111; RP-v1q and R2 23). Every pk1 row passes `validate_row`
(replay rows with `teacher_probs` over exactly the new keys). A0s and RP-v1q rows equal their A0 rows,
so the renumbered ids are shared; the 117 renumbered A0 rows are all in A0s (families
`stage4_replay_*`: CLINC 35, BANKING 28, policy 19, rubric 13, evidence 11, mapping 8, authorization 3).
SELECT700, CAL700 and CAL698 have no `result_<n>` keys.

## Teacher re-derivation (never moved with the old keys)

| File | Node / runtime | Prompts | Re-derived | Controls (unchanged rows) vs the old file |
| --- | --- | ---: | ---: | --- |
| own-Lux A0 canonical | node A GPU2; image `f83b1d10`, Lux-9B `bd45a30a`, `inference.run --backend lux --over-budget-invalid`, private copy of the post-run autotune cache of the canonical predictions (`c94e66c5`); mirror `6533ef531`; 34 s | 181 | 117 / 117 | 64 / 64 bitwise identical (max diff 0.0) |
| R2 Kai, Lex | node B GPU7; Milestone 1 mirror `0a42f3dd`, image and snapshots | 64 | 22 / 22 | 42 / 42 bitwise identical |
| R2 Eos | same | 64 | 22 / 22 | 0 / 42 identical, argmax 40 / 42, max diff 0.0089 (its Milestone 1 runtime has no persisted autotune cache; M1 repeat drift was 0.0082) |
| R2 Sol, Nox, Lux | same | 64 | 22 / 22 | 42 / 42 bitwise identical |

No renumbered row lacks a target. Every re-derived receipt was checked by
`v2.data.replay_targets.convert` (model id, attested revision, validated runtime, prompt digest of the
pk1 row).

**Did Lux use the leaking keys?** On the 117 renumbered A0 rows, own-Lux1 picks the gold before and
after renumbering on 117 / 117; mean gold probability 0.977 → 0.975; no argmax change; largest
per-option change 0.22 (one row). By gold-key rank before renumbering (largest-but-one 75, largest 18,
smallest 13, other 11) the gold probability moves by at most 0.003. So the leak barely shaped the Lux
targets on A0, and the pk1 targets are nearly the same distributions under non-leaking keys.

## GPU-hours

Node A GPU2 0.009 (Lux, 181 prompts); node B GPU7 0.058 (six R2 teachers, 208 s).
