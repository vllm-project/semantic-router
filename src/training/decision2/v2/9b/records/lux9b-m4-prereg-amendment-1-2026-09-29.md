# 9B Milestone 4 preregistration, amendment 1: no Hugging Face staging of finalists

Frozen 2026-09-29 while K-s1 and P-s1 train, before any Milestone 4 result from the K, P, KN, U or
K-line artifacts (the D-line development readout is the only Milestone 4 result so far, and it
does not involve staging). Trigger: the coordinator's storage policy of 04:55 and 05:25 UTC+8
(COORDINATION "Cross-track notes") supersedes the staging parts of the 02:45 / 04:10 notes that
the [preregistration](lux9b-m4-prereg-2026-09-29.md) followed.

- The preregistration's last "Finalists, formal runs and gate" bullet is replaced by: **a finalist
  that passes the gate is NOT uploaded to Hugging Face.** Its durable copy is the scored
  checkpoint directory on node A (`/data/dev2/runs/9b/m4/<name>-build/soup` plus `<name>-cal/`),
  with a per-file SHA-256 manifest written next to it and re-hashed once. It is reported to the
  coordinator immediately with its scored run directory. Only a coordinator-approved release
  candidate is uploaded, directly to its final release repo (release engineering, after the
  steward's `v2/common/hf_headroom.sh` check).
- Failing finalists and every other Milestone 4 artifact get the same node-copy treatment.
  Nothing from Milestone 4 is uploaded.
- Nothing else changes (arms, data, α rule, finalist priority, gate, budget).
