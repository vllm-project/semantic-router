# IB1 state (research & data; hand-off file)

Worktree `/home/xunliu/code/vllm-sr-dev2-data-ib1`, branch `xunzhuo/decision-2-training-data-ib1` (from
`origin/xunzhuo/decision-2-training` at `3ee34c625`). Prereg `records/ib1-prereg-2026-10-01.md` (`1f6823fae`) with
amendments 1 (`a21ee5f8a`), 2 (`1464c28df`, round 2) and 3 (`74a4cad41`, round 3; addendum `b95b2ee34`);
licence registry `records/license-registry-ib1.json`; reviewer instructions
`records/dq/reviewer-ib1.md`; tooling `v2/data/ib1/` (runner `node_a.sh`). Results
`records/ib1-results-2026-10-01.md` (round 1), `records/ib1-r2-results-2026-10-01.md` (round 2) and
`records/ib1-r3-results-2026-10-01.md` (round 3); receipts and data card `records/ib1/`. Private data on node A under
`/data/dev2/private/data/ib1/` (mode 700): `raw/` (pinned publisher files; `download.log` holds every revision and
commit) and the run directory **`b2/`** (candidates, scans, quarantine, shortcut receipts, screen and review keys and
answers, final files, upload tree, logs; superseded outputs kept as `final-v1/`, `freeze-v1/`, `hf-v1/`) and the round-2 run directory **`r2/`** (same layout plus `pass1-3/`,
`rescan1-3/`, `rescanfinal*/`; superseded `final-v1/`, `freeze-v1/`, `leak-v1/`, `shortcut-pass2/` and the unreviewed
214-item draw `review-v1-unreviewed/`) and the round-3 run directory **`r3/`** (same layout, `pass1-4/`,
`rescan1-4/`, `rescanfinal*/`; superseded `final-v1/`, `freeze-v1/`, `leak-v1/`, `shortcut-pass2/`, `shortcut-pass3/`
and the unreviewed first draw `review-v1-unreviewed/`, identical items to the reviewed sample). `b1/`
stopped at the build on a reader bug and holds nothing. Local review packets and answers (private, mode 700):
`/home/xunliu/.cache/dev2-ib1-review/` (round 2 in `review-r2/`, round 3 in `review-r3/`).

Rules carried by this job: Index numbers never leave the private directories (not in commits, records, gist, cards);
no Index row is trained on (G0); C1 registry datasets and the sealed directory are never read; HR2-r2's worktree is
not touched.

**Status: round 3 DONE, RELEASE-SAFE.** Private
`llm-semantic-router/decision-2.0-training-data@31b200a34759d6b2ca197737603dfaa253c471cd`, `m6/ib1/`: TRAIN 24,325
rows (`1e1b08f3…`), DEV 2,067 (`3f56aa41…`), 4.75M native tokens, 9 families. Fresh decisive blind review 3 / 216 =
1.39% [0.29, 4.01], weighted 2.14% (P1, P2, P3 pass). Rounds 1 (`1371c287`) and 2 (`82bf70a7`) are earlier revisions,
both NOT release-safe.

## Log (newest first)

- 2026-10-01 15:30 UTC+8 — round 3 uploaded (revision `31b200a3`, 85 files, read-back equal, private,
  `release_safe: true`); records, card, status, results; gist 02 and merge into integration.
- 2026-10-01 15:15 UTC+8 — round-3 final: 3 review drops → final re-scan flagged 1 group → `final` re-run → re-scan
  clean; leak guard 0; G5 / G7 PASS.
- 2026-10-01 15:05 UTC+8 — **round-3 review PASSED**: 3 / 216 = 1.39% [0.29, 4.01], weighted 2.14%, max 1 per family;
  R1 = R2 on all 216 (no R3).
- 2026-10-01 14:55 UTC+8 — first draw had no `sumedit` item (all 17 groups sampled in rounds 1–2) → addendum
  `b95b2ee34`: `sumedit` leaves; pass 4 + re-scan clean; redraw identical items (216, 24 × 9).
- 2026-10-01 14:45 UTC+8 — run `r3`: build (two-option `sentfin`), scans, pass 1, re-scans 53 + 7 → 2 → 0;
  G4 drops `medmcqa`, `procb`, `wanli`; controls 2,000 / 2,000 twice.
- 2026-10-01 14:25 UTC+8 — amendment 3 pushed (`74a4cad41`) before any round-3 row was built.

- 2026-10-01 14:15 UTC+8 — round 2 uploaded (revision `82bf70a7`, 80 files, read-back equal, private; stale round-1
  stage-S receipts removed in the same commit); receipts, card, status and results record written; gist 02 and merge.
- 2026-10-01 14:00 UTC+8 — round-2 final: review drops (12 + 17 carried) → re-scan of the final files flagged 1 group
  → `final` re-run (old outputs `final-v1/`, `freeze-v1/`, `leak-v1/`) → re-scan clean, leak guard 0, G5 / G7 PASS.
- 2026-10-01 13:55 UTC+8 — **round-2 review FAILED**: 12 / 225 = 5.33% [2.79, 9.13], weighted 5.97%, `wands` 4 / 19
  (P3). 5 of 12 errors are reviewers choosing the class kept as an option but no longer gold.
- 2026-10-01 13:37 UTC+8 — round-2 sample 225 (19 per family, `sumedit` 16); five fresh foreground reviewers (R1 / R2
  on two packets each, R3 on 4 splits).
- 2026-10-01 13:33 UTC+8 — first draw 214 < 216 (`sumedit` group-limited) → set aside unreviewed; addendum
  `602fc4e9c` (quota top-up) before any review.
- 2026-10-01 13:30 UTC+8 — run `r2`: candidates byte-identical to `b2`; scans, pass 1 (constructions + carried ids),
  re-scans 62 + 7 → 2 → 0 groups (pass 3 sampled); G4 drops `medmcqa`, `procb`, `wanli`; `maud` emptied by re-scan.
- 2026-10-01 13:08 UTC+8 — amendment 2 pushed (`1464c28df`) before any round-2 row was built or sampled.

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

1. IB1 round 3 is release-safe: the 27B M6 and 4B M10 IB1 arms can start from revision `31b200a3`.
2. Before any C1-scored model trained on IB1: the custodian C1 content recheck (IB1 is inside the rescan roots).
3. IB2 (uncovered target families): contracts, knowledge MCQ, maths verification, NLI substitute,
   select-all-that-apply; plus faithfulness and product-search relevance, which IB1 round 3 no longer covers.
