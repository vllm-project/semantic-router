# 27B pooled schedule: CPU admission result

**Status: HOLD; zero GPU-hours.** This is a prospective, distinct schedule
candidate. The earlier two exact-quota admissions failed and remain failed.
This audit did not train, infer, select a checkpoint, inspect evaluation
answers, package weights, or change Hugging Face state.

The fixed 2,560-row/160-update schedule has receipt SHA-256
`d76eb49a12a6dd04861239a7d7e6697f6f96c0f80015246b4060362c5bc156d7`
and ordered ID/input/token digest
`6ea5e54d207535bd1ca06e9f2ad4ff19fbdae0d4c6143fce0b20be8fcbe52181`.
The CPU admission implementation SHA-256 is
`bdcde2ff17fa36fb1205d179e2d56cb3579cd5c6ceb5eb83dc325b7f78823b2b`.
It ran from an exact local-source mirror in the pinned CPU image
`sha256:dbe5f32b2263b2671ba0b9aaaf18ee20abda189541fc22107e216a2f37d440b1`.
The final private 0600 aggregate receipt SHA-256 is
`f7a9c2ca05a1664caf1a2cbeff17f18dec899e96ba40b96a4cfd65ec36c8bad9`.
The receipt and schedule remain private because the schedule contains row IDs.

| Check | Result |
| --- | --- |
| Scheduled row and native-token identity against pinned TRAIN | 2,560/2,560 pass; Choice 1,026, Noul 1,027, Score 507 |
| TRAIN rights/source ledger | Seven declared entries cover all nine raw TRAIN source IDs under the explicit frozen source mapping; declared permissions are not an independent legal conclusion |
| Teacher distributions | Both pinned teacher files, all 7,455 ordered row/input/option vectors and frozen revision identity pass |
| Teacher mask minima | 990 eligible: Choice 347, Noul 441, Score 202; three-level Score 63; frozen minimums pass |
| Eight-role protected input inventory | All 20,263 projected input-only records and file hashes pass |
| Exact complete native-input overlap | Zero across all seven disjoint protected roles |
| Exact state-evidence overlap | Zero across all seven disjoint protected roles |
| Bounded near state-evidence pairs | SELECT 52, CAL 89; other five protected roles zero |
| Fully selected source/group IDs | 1,558 of 2,158 selected groups |
| Selected groups missing a task type | **600** |
| Selected groups missing any row | **600**; same-task-type missing rows zero |
| Extra cross-type rows required to complete selected Score groups | **0** |

The reusable broad span checker also reports many pairs sharing generic
instructions or option wording. Those are not evidence of full-row reuse: an
independent complete-input/state view reports them separately and prevents
common scaffolds from being mistaken for exact input identity. The 141
SELECT/CAL near-state pairs are **possible** overlap signals, not confirmed
duplicate or leakage cases. They remain blocking until reviewed at source and
semantic level; a zero exact-match count does not clear them.

This admission requires whole `source/group_id` selection across all task
types. The planner preserved whole groups only inside each source/task type,
and **600 groups violate the stricter policy**. This candidate therefore
cannot be admitted even if every near-state pair is later resolved. A future
schedule must prospectively meet the whole-group rule, context budget and
teacher-mask thresholds, followed by a fresh rights and full-input screen;
the current schedule cannot be silently relabeled or pruned after seeing its
diagnostics. Conversely, relaxing the group policy would be a newly versioned
experiment, not a pass for this one.

The bounded exact/SimHash near-input audit may miss paraphrases and source
reuse; pretraining exposure and independent permission review are outside its
scope. No zero-step model parity, optimizer stability, SELECT, CAL, DEV,
JevArena, JevBench or upload permission follows from these CPU checks. The
next useful work is to review the near-state pairs without opening any
evaluation answers and plan a **new** group-atomic schedule using the same
frozen TRAIN and teacher identities. Only a separately frozen, fully passing
arm may reserve a GPU.
