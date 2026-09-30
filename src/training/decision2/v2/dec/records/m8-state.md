# Decoder Milestone 8 — state (keep current; newest first)

Assignment: coordinator note 2026-09-30 17:00 (user plan decision: 4B M8 = cross-size distillation from DEV2.0-27B
A20r, matched controls, starts now, HR2 only by amendment). Budget 24 GPU-h. GPUs: node A GPU5, node B GPU3–4
(decoder), node A GPU0–1 shared leases for the A20r labels. Preregistration:
[`dec-m8-prereg-2026-09-30.md`](dec-m8-prereg-2026-09-30.md).

## Now

- 2026-09-30 ≈18:05 UTC+8 (10:05Z) — **Running.** Prereg `a629a6ce2`; data lock part 1 `862c99b1e`; formal wrapper +
  compressed relays `a1f4dbc55`. Mirror on both nodes: `a629a6ce29fe2b9a27aeac8f8b611667f654d8d9-src_training_decision2`
  (tests pass in the image: 17). Nothing uploaded.
  - Slice: 12,794 rows / 6.32M tokens (`bd529767…`); human-rated 3,467 rows / 0.92M tokens; C teacher `7ddaff84…`;
    exposure 0 groups; node B `m8/data/READY` written.
  - node B chains (PIDs in `m8/chains/chain-b{3,4}.pid`): C-m1 preflight PASS 09:47Z, done 09:57Z (167 updates);
    C-m2 done 09:58Z; C-m3 training (~10:09Z); b3 waits for `m8/teacher-a20r/READY`.
  - node A label shards 0 / 1 / 2 on GPU0 / GPU1 / GPU5 since 09:57Z (PIDs `m8/label/logs/shard-<k>.pid`, containers
    `dec-m8-label-<k>`), ≈15–20 min each.
  - node B GPU3: `m8-lines.sh refs` (4b-I Score5-typed-DEV; other panels reused from M7).
  - Next: relay shard predictions + smoke receipts to node B (`m8/label/run-<k>/`), `m8_teacher.py convert` on node B
    → `m8/teacher-a20r/{D1,D2}`, lock part 2 commit, then `m8/teacher-a20r/READY` (format: `<sha>  D1/teacher.jsonl`,
    `<sha>  D2/teacher.jsonl`). The C soup builds when C-m3 finishes; then `M8_GPU=3 m8-lines.sh line C`.
  - The dec-small (2B / 0.8B) worker may read the A20r targets on node B by row id + input hash:
    `/data/dev2/runs/dec/m8/teacher-a20r/D1/teacher.jsonl` (every slice row).

## Plan

| Step | Where | Status |
| --- | --- | --- |
| Data build (slice, S split, C teacher, prompts, exposure) | node B CPU | pending |
| A20r labels, 3 shards (+ parity smoke) | node A GPU0 / GPU1 (shared) / GPU5 | pending |
| Chains b3: C:1 D1:1 D1:2 D2:2; b4: C:2 C:3 D2:1 D2:3 D1:3 | node B GPU3 / GPU4 | pending |
| Lines L-D1 / L-D2 / L-C (α 1, ⅔, ⅓) + refs + hs1-dev | node B (co-tenant readouts) | pending |
| HT-DEV v2 / Score5-typed-DEV scoring, rules → finalists | node A CPU | pending |
| Formal (≤ 3), successor items 1–7, item-8 hand-off | node B → node A | pending |

## Liveness and operations

- Never `pgrep -f`; check PIDs (`chains/chain-b3.pid`, `label/logs/shard-<k>.pid`) and container names
  (`dec-m8-*`) with `docker ps`.
- Chain rule: upload/mirror, verify size + SHA-256, launch in a separate step, confirm the first log line.

## GPU-hours

| Item | GPU-h |
| --- | ---: |
| (none yet) | 0 |
