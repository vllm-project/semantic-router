# Kev-4B external teacher: v2 pre-inference stop

**Decision: HOLD.** The separate v2 TRAIN-only teacher screen did not reach
model inference and produced no aggregate receipt. Its frozen 96-group roster
must not be retried for a teacher result.

The [v2 prospective protocol](kev4b-teacher-train-screen-v2-prereg-2026-09-28.md)
locked the rights-clean TRAIN SHA-256
`61740be433c6cd714810a9908432ad29597c570d0d267d2731ab78cdad243755`,
96 independent groups with roster SHA-256
`cbf6b8e3292428508cf8cbb6a9721232df74ffcdcefbdb34e354cb6b3b685692`,
the published Kev model and source revisions, and pilot-script SHA-256
`f6d5d44d37114a901c94d12b6b1ee9de549778a5e443a7a7661df7e72ffb5ca9`.
The runtime image ID was
`sha256:dbe5f32b2263b2671ba0b9aaaf18ee20abda189541fc22107e216a2f37d440b1`.

The only v2 container exited with status 1 after **0.427853 seconds** from
start to die events, an upper bound of **0.000119 GPU-hours**. During source
provenance verification, `git rev-parse HEAD` exited 128 because the mounted
Kev source is a linked worktree but the mount omitted the referenced parent
Git metadata. This happened before `Checkpoint.load` and before any TRAIN
decision; the container window is not a claim of GPU computation. No score,
raw prediction, or aggregate receipt exists. The container exited and no task
process remained.

The source-path mount contract therefore joins the native model revision,
TRAIN bytes and output validation as a required **CPU/container preflight**
for any distinct future screen. A new roster and a new signed prospective
protocol are necessary; changing only the mount and rerunning v2 would violate
its one-run rule. No student arm is justified by this failure.
