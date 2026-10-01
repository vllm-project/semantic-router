# IB1 state (research & data; hand-off file)

Worktree `/home/xunliu/code/vllm-sr-dev2-data-ib1`, branch `xunzhuo/decision-2-training-data-ib1` (from
`origin/xunzhuo/decision-2-training` at `3ee34c625`). Prereg `records/ib1-prereg-2026-10-01.md` (`1f6823fae`) with
amendment 1 (`a21ee5f8a`); licence registry `records/license-registry-ib1.json`; reviewer instructions
`records/dq/reviewer-ib1.md`; tooling `v2/data/ib1/` (runner `node_a.sh`). Results
`records/ib1-results-2026-10-01.md`; receipts and data card `records/ib1/`. Private data on node A under
`/data/dev2/private/data/ib1/` (mode 700): `raw/` (pinned publisher files; `download.log` holds every revision and
commit) and the run directory **`b2/`** (candidates, scans, quarantine, shortcut receipts, screen and review keys and
answers, final files, upload tree, logs; superseded outputs kept as `final-v1/`, `freeze-v1/`, `hf-v1/`). `b1/`
stopped at the build on a reader bug and holds nothing. Local review packets and answers (private, mode 700):
`/home/xunliu/.cache/dev2-ib1-review/`.

Rules carried by this job: Index numbers never leave the private directories (not in commits, records, gist, cards);
no Index row is trained on (G0); C1 registry datasets and the sealed directory are never read; HR2-r2's worktree is
not touched.

**Status: DONE, flagged NOT release-safe.** Private
`llm-semantic-router/decision-2.0-training-data@1371c28777ae24e2632145dff3ea55050d47c014`, `m6/ib1/`: TRAIN 37,314
rows (`7f2d0003…`), DEV 2,664 (`b412ab93…`), 10.47M native tokens. Decisive blind review 13 / 216 = 6.02% (P1 / P2
fail, P3 pass).

## Log (newest first)

- 2026-10-01 13:06 UTC+8 — merged into `xunzhuo/decision-2-training` (`6ef637ebd`; IB1, HR2, guard, overlap,
  shortcut and freeze tests pass) and gist 02 entry added. **Worker done.**
- 2026-10-01 13:00 UTC+8 — uploaded `m6/ib1/` (43 files, read-back SHA-256 equal, private before / after; revision
  `1371c287`); receipts, data card, results record written. Remaining: gist 02, merge into integration.
- 2026-10-01 12:50 UTC+8 — final files (`dada31e38` assemble; upload copy of the G0 receipt without suite
  statistics); the leak guard flagged 12 `w2c` rows (IPv4-like strings) → `final` re-run without them (old outputs
  `final-v1/`, `freeze-v1/`); G5, G7 PASS; node-side check for node addresses / hostnames 0.
- 2026-10-01 12:36 UTC+8 — **stage R FAILED**: 13 / 216 = 6.02% [3.24, 10.07], weighted 8.33%; P3 pass (max 3 of 18
  in `sentfin`, `sumedit`, `wands`). No fix rule → published flagged; 13 + 4 rows dropped.
- 2026-10-01 12:22 UTC+8 — stage R sample (216 rows, 18 per family, 12 families; no screened row or group);
  R1 / R2 as four fresh subagents, R3 on 3 splits.
- 2026-10-01 12:20 UTC+8 — stage S screen (one fresh reviewer, 120 rows): 4 disagreements (sentfin, snips_rel,
  sumedit, w2c one each); no family dropped; the 4 rows leave TRAIN.
- 2026-10-01 12:11 UTC+8 — pass 1: quarantine 530 groups (G2) + 3,273 groups (G0; mostly When2Call train rows sharing
  tool schemas with Index tool rows); DEV near TRAIN 1,064 groups. **G4 dropped `maud` (option-only .655 vs .430),
  `medmcqa` (option-only .342 vs .254), `procb` (state-removed .500 vs a cross-validated majority of .430;
  small-sample artefact at chance, not re-run) and `wanli` (hypothesis-only .440 vs .320).** 12 families left.
- 2026-10-01 12:10 UTC+8 — scans: G0 positive controls 2,000 / 2,000 exact and 2,000 / 2,000 perturbed copies flagged
  (PASS); G1 PASS (lineage: CommonsenseQA already in 2.0 training, disclosed); G3 C1 source terms 0 found.
- 2026-10-01 12:12 UTC+8 — amendment 1 (`a21ee5f8a`): `expqa` empty under the rule (every `Missing` claim lacks
  evidence text) and dropped; G5 per option count.
- 2026-10-01 12:02 UTC+8 — candidates (`17359493`, run `b2`): TRAIN 63,359 / DEV 6,655, 16 families with rows.
- 2026-10-01 11:58 UTC+8 — prereg pushed (`1f6823fae`) before any row was converted.
- 2026-10-01 11:55 UTC+8 — source audit done (16 sources admitted; HellaSwag repo DMCA-blocked, RAGTruth / VAST /
  FinEntity / Humicroedit / ANLI / ACORD excluded); raw files downloaded on the node A host; iSarcasmEval test files
  deleted unread.
- 2026-10-01 11:07 UTC+8 — job started; worktree created.

## Next steps (for the coordinator)

1. Decide: IB1 for experiments only (not release-safe), or a separately preregistered IB1-r2 (for example without
   `sentfin` neutral rows, SummEdits' Shakespeare domain and WANDS Partial rows; fresh decisive review).
2. Before any C1-scored model trained on IB1: the custodian C1 content recheck (IB1 is inside the rescan roots).
3. Uncovered target families for a later round: math verification / GSM8K-style numeric options, contracts,
   knowledge MCQ, adversarial NLI, select-all-that-apply (all removed by a gate or deferred).
