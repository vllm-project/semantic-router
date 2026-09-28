# JevArena-C1 blind quality review — round 2 (2026-09-28)

This round re-reviewed the three tasks whose templates were fixed once after
[round 1](m3-sealed-c1-review-round1-2026-09-28.md). Three fresh reviewer subagents
(r4–r6) saw the same sampled items (32) under the fixed templates (build-4: prompts
`4ba60968…`, gold `7e71d8b4…`; selection unchanged for every task). Receipt SHA-256
`ae213452…` (node A, private).

| Task | n | Flagged by ≥ 2 | Majority = gold | Chance | Fleiss κ | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| mcjudgebench/constraint | 16 | **6 (38%)** | 0.69 (within 1 level: 0.88) | 0.33 | 0.80 | FAIL (second failure: excluded) |
| tutormoments/is_rapport | 8 | 1 | 0.88 | 0.50 | 1.00 | PASS |
| tutormoments/moment_type | 8 | 0 | 1.00 | 0.50 | 1.00 | PASS |

- **MCJudgeBench is excluded** (`c1-config.json` v3). All three reviewers found the same
  intrinsic ambiguity, which the source's own annotation criteria do not resolve:
  - "each/all/at least N" requirements met in some places and broken in others;
  - permission-style ("may/can") or persona requirements on very short outputs;
  - requirements written against a reference answer;
  - "must not" rules on responses that end early.

  With it goes C1's only instruction-following judgement task.
- **tutormoments passes** under the fixed template. The remaining ambiguity is in short
  spans that mix feelings with a teaching decision.
- Reviewers noted minors' personal remarks in the tutoring transcripts. The names look
  pseudonymised in the source release. Nobody flagged them as sensitive, but the seal
  record discloses it.

Final C1 v1 composition: 14 tasks from 8 sources. GAPA stays in the config, but the build
gate drops it with 3 screenable items. See build-5 in the seal record.
