# IB1 state (research & data; hand-off file)

Worktree `/home/xunliu/code/vllm-sr-dev2-data-ib1`, branch `xunzhuo/decision-2-training-data-ib1` (from
`origin/xunzhuo/decision-2-training` at `3ee34c625`). Prereg `records/ib1-prereg-2026-10-01.md` (`1f6823fae`) with
amendment 1 (`a21ee5f8a`); licence registry `records/license-registry-ib1.json`; reviewer instructions
`records/dq/reviewer-ib1.md`; tooling `v2/data/ib1/` (runner `node_a.sh`). Private data on node A under
`/data/dev2/private/data/ib1/` (mode 700): `raw/` (pinned publisher files; `download.log` holds every revision and
commit) and the run directory **`b2/`** (`b1/` stopped at build on a reader bug; nothing written). Index suite rows
(read-only): `/data/dev2/private/eval/index021/suite-0.2/`.

Rules carried by this job: Index numbers never leave the private directories (not in commits, records, gist, cards);
no Index row is trained on (G0); C1 registry datasets and the sealed directory are never read; HR2-r2's worktree is
not touched. Runner: `IB1_RUN=b2 bash /data/dev2/src/<sha>-src_training_decision2/src/training/decision2/v2/data/ib1/node_a.sh <sha> <stage>`.

**Status: stage R blind review running (R1 / R2, four fresh reviewers, packets of 108 items).**

## Log (newest first)

- 2026-10-01 12:22 UTC+8 — stage R sample (216 rows, 18 per family, 12 families; no screened row or group);
  R1 / R2 launched (answers go to the local private review directory, then to `b2/review/answers/`). PASS needs
  ≤ 9 errors, weighted ≤ 5%, and no family with ≥ 4 of 18.
- 2026-10-01 12:20 UTC+8 — stage S screen (one fresh reviewer, 120 rows): 4 disagreements (sentfin, snips_rel,
  sumedit, w2c one each); no family dropped; the 4 rows leave TRAIN.
- 2026-10-01 12:11 UTC+8 — pass 1: quarantine 530 groups (G2) + 3,273 groups (G0; mostly When2Call train rows sharing
  tool schemas with ToolRet / When2Call test rows); DEV near TRAIN 1,064 groups. **G4 dropped `maud` (option-only
  .655 vs .430), `medmcqa` (option-only .342 vs .254), `procb` (state-removed .500 vs a cross-validated majority of
  .430; small-sample artefact at chance, not re-run) and `wanli` (hypothesis-only .440 vs .320).** 12 families left;
  pass-1 TRAIN 37,355 rows.
- 2026-10-01 12:10 UTC+8 — scans: G0 positive controls 2,000 / 2,000 exact and 2,000 / 2,000 perturbed copies flagged
  (PASS); G1 PASS (lineage: CommonsenseQA already in 2.0 training, disclosed); G3 C1 source terms 0 found.

- 2026-10-01 12:12 UTC+8 — amendment 1 (`a21ee5f8a`): `expqa` empty under the rule (every `Missing` claim lacks
  evidence text) and dropped; G5 per option count. Scans started on `b2` with `a21ee5f8a`.
- 2026-10-01 12:02 UTC+8 — candidates (`17359493`, run `b2`): TRAIN 63,359 / DEV 6,655, 16 families with rows.
- 2026-10-01 11:58 UTC+8 — prereg pushed (`1f6823fae`) before any row was converted.
- 2026-10-01 11:55 UTC+8 — source audit done (16 sources admitted; HellaSwag repo DMCA-blocked, RAGTruth / VAST /
  FinEntity / Humicroedit / ANLI / ACORD excluded); raw files downloaded on the node A host; iSarcasmEval test files
  deleted unread.
- 2026-10-01 11:07 UTC+8 — job started; worktree created.

## Next steps

1. Scans (G0 Index rows incl. positive controls, G1, G2, G3) → `pass1` (quarantine, G4 shortcuts).
2. Stage S screen: `screen` → one fresh reviewer on `b2/screen/sample/packet.s1.*.jsonl` → answers to
   `b2/screen/answers/s1.*.jsonl` → `screen-score`.
3. Stage R: `review` → R1 / R2 (packets `b2/review/sample/packet.{r1,r2}.{1,2}.jsonl`) → `splits` → R3 → `score`.
4. `final`, `leak` (re-run `final` if rows are flagged), data card + `status.json`, `hf-assemble`, `hf_headroom.sh`,
   `hf-upload`; results record; gist 02; merge into `xunzhuo/decision-2-training`.
