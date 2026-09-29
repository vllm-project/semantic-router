# DEV2.0-0.6B card-only revision: the post-key C1 line (2026-09-30)

Release worker (worktree `vllm-sr-dev2-release`), resumed once after the C1 card pass for the line the coordinator
approved at 2026-09-30 00:30 UTC+8, after that pass had started. Receipts: [`dev2-0p6b-c1postkey-card-2026-09-30/`](dev2-0p6b-c1postkey-card-2026-09-30/)
(`0p6b/` receipts, extra checks, logs and package text; `ops/` scripts). Everything stays private.

## Result

- **Private `llm-semantic-router/DEV2.0-0.6B@476fe984a2316519f3e583b7f31b1670295b2477`** (`main`), manifest
  `5a3317a2034bf7fe9cc75ac228d6434907730ef97e0be7446c639851ad721dde`. It supersedes `188eb4c8` (C1 card pass).
- **Final decision** [`DEV2.0-0.6B.decision.json`](dev2-0p6b-c1postkey-card-2026-09-30/DEV2.0-0.6B.decision.json)
  `54a0f25f373be7002eed7c3caacc89744b0fb3cc0ef391468824a0bf8b5f1feb`, superseding `453b4b7b…`. It keeps the successor
  profile (R1–R7 against `99c4e799`); gate.json `9d67b209…`, 12 of 12 items.
- **Card** (spec `specs/dev2-0p6b-card-c1pk.json`, derived by [`ops/make.py`](dev2-0p6b-c1postkey-card-2026-09-30/ops/make.py), `--check`):
  - Under the sealed-set line, which keeps measuring the previous revision, the approved line verbatim: "**JevArena-C1
    v1.2, post-key (not an independent validation):** 36.92 on this revision vs 33.02 for the previous revision
    (+3.89, 95% CI [+2.16, +5.60]; 2,840 items, paired by source group). The independent sealed confirmation above
    measured the previous revision."
  - A new limit discloses the post-key C1 declines against the previous revision (eval record
    `v2/eval/records/c1-postkey-guard-2026-09-29.md` §5): star_rating 21.3 vs 30.5, moment_type 53.2 vs 60.1,
    is_rapport 43.6 vs 49.0, likelihood 10.2 vs 13.1; Russian .263 vs .353, Finnish .426 vs .500.
- **Weights:** both weight files byte-identical to `188eb4c8` (and `b2131337`); only `README.md` and
  `MODEL_MANIFEST.json` changed (Hub revision diff).
- **Collection** "🎲 Decision 2.0": private, title unchanged, 0.6B, 0.8B, 2B, 4B, 9B, 27B; nothing moved.

## Verification

`ops/card.sh 0.6B --gpu 0` (the C1 pass runner restricted to this revision) ran `release.sh --upload --collect
--already-collected` from mirror `2f21790ba` on node A GPU0 under the shared lease `owner.release-c1pk`. A CPU preview
first showed that only the two additions changed.

- Subset parity, before and after upload, at tolerance 0: typed-final 200, CSS15 300, public 231 100, mlx-diag 100;
  0 changed answers, max drift 0.0.
- Real `hf download` and re-hash of 33 files; 597,103,104 loaded parameters; examples bit-identical across processes
  and after download; the card example reproduced.
- Readback private with no card problems; gate sealed; collected readback passed.
- Card HTTP 12 / 12 (anonymous 401); links 14 / 14; `gate evaluate` passed.

## GPU-hours, storage, commits

- 17:43:24–17:46:02Z on node A GPU0: **0.044 GPU-h**. The preview was CPU only.
- Storage unchanged at 61.28 / 100 GB; nothing deleted.
- `2f21790ba` spec, decision and ops; this record and the receipts follow on the same branch, merged into
  `xunzhuo/decision-2-training`.
