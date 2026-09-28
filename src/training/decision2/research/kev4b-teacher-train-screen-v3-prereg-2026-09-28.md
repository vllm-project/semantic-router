# Kev 4B TRAIN-only teacher signal: third, disjoint screen

**Prospective v3; no v3 teacher inference at registration.** V1 stopped on
rounded-distribution validation, and v2 stopped before inference because a
container mount hid linked Git metadata. Neither failed roster is retried.
This screen measures whether the same public native Kev release offers a
useful TRAIN-only soft-label signal. Kev is an external teacher and peer, not
a direct Decision 2.0 weight initializer. Its signal does not establish any
student gain or release result.

| Frozen component | v3 commitment |
| --- | --- |
| TRAIN | Rights-clean v2, 7,455 rows, SHA-256 `61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755` |
| Teacher | `jaredpalmer/kev-4b@139fdd94f1b6a6ad80cc15e08fcb99cac885a101`; publisher source `6d02f5d066cd34958dfd15ffa5d2f6f0f4c21a63`, native Checkpoint/API readout with its checkpoint temperature |
| Roster | Exactly 96 independent TRAIN groups, 32 per Choice/Noul/Score; exclude all group IDs from **both** v1 and v2, then sort SHA-256 of `kev-v3`, type, original group and row ID. CPU dry-run fixes the resulting SHA-256 before any teacher readout. |
| Code | [`kev_teacher_train_pilot_v3.py`](kev_teacher_train_pilot_v3.py), SHA-256 `5bd612d7e0db655ba6bda6681cd5cb192a40173062042bcad59e14f157dac4f9`; two new CPU contracts plus existing v2 numeric contracts passed locally. |
| Numeric rule | Reuse the signed v2 `rounded_probabilities` and `aggregate` unchanged: exact type/keys, finite [0,1] values, `abs(sum−1) ≤ K × 0.00005 + 1e-8`, then normalize. Noul uses its native yes probability and complement. Tied argmax is wrong; malformed output stops; overflow is invalid. |
| Report | Full-denominator per-type valid/invalid/tie and TRAIN hard-label agreement, mean gold probability and half multiclass Brier; aggregate only, no raw rows or per-item predictions. |
| Runtime | Pinned container image SHA-256 `dbe5f32b2263b2671ba0b9aaaf18ee20abda189541fc22107e216a2f37d440b1`; one exclusive GPU, one ≤15-minute invocation; no changed model, roster, numeric rule, prompt or same-roster retry. |

The **exact future container mount layout** keeps the private experiment root
at its original absolute path, read-only, including both the Kev linked
worktree and its referenced parent Git metadata. Only a task-owned private
receipt directory is mounted writable. The pinned offline HF cache lives in
that same read-only root. Before reserving a GPU, run `--dry-run` and
`--preflight` inside this very layout without GPU devices. The preflight
checks local model revision metadata, all publisher source hashes and the
linked worktree HEAD, adapter bytes, base revision and all indexed base weight
shards. Any failed check ends this arm. Then recheck live GPU/process/container
occupancy. Do not download new large weights or access protected evaluation
labels.

The descriptive threshold for a **future, separately preregistered**
three-type distillation comparison remains all 96 native-valid, at least
22/32 Choice, 26/32 Noul and 20/32 Score TRAIN hard-label matches, and Score
half Brier ≤0.25. Failure authorizes no three-type student. A pass authorizes
only designing source-reviewed student/control arms with matched budgets and
independent development promotion; this screen itself authorizes no training.
Keep the receipt mode `0600` and publish only public-safe aggregates and hashes.

## Pre-GPU CPU lock

After the signed prospective registration, the exact source mirror and future
container mount layout passed a CPU-only dry-run and provenance/cache preflight.
The v3 roster is **96 rows from 96 independent groups**, 32 per type, with
**zero group overlap** with either v1 or v2. Its ordered identity SHA-256 is
`f35e56d4127d9c573671e232880b9de76abe67a69115e97a22f6187259b2cdae`.
The mirrored script and TRAIN bytes matched their frozen SHA-256 values.
The same pinned image attested the Kev model revision, all 30 publisher source
files, the linked source worktree HEAD, and both official base weight shards
from offline cache. Model fingerprint:
`8e2c7fff2ef6ad7b195443fac287fb1ae4cd83c8c9af1a12dfb743501a3ec3e9`.
No GPU or protected evaluation labels were used in this preflight. A result
with any other roster digest is invalid for this protocol.
