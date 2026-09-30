# Decoder Milestone 8 — state (keep current; newest first)

Assignment: coordinator note 2026-09-30 17:00 (user plan decision: 4B M8 = cross-size distillation from DEV2.0-27B
A20r, matched controls, starts now, HR2 only by amendment). Budget 24 GPU-h. GPUs: node A GPU5, node B GPU3–4
(decoder), node A GPU0–1 shared leases for the A20r labels. Preregistration:
[`dec-m8-prereg-2026-09-30.md`](dec-m8-prereg-2026-09-30.md).

## Now

- 2026-09-30 ≈19:40 UTC+8 (11:40Z) — **Finalists fixed; formal runs on node B.** All nine members DONE (chains
  exited 11:02Z); soups C / D1 / D2 built; every line read and scored. Rules (`m8/select/4b-finalists.json`
  `08273233…`, node A and node B):

  | Line | α 1 | α ⅔ | α ⅓ | Pick |
  | --- | --- | --- | --- | --- |
  | L-D1 | T .701, HT −.027 **FLAG** | T .710, HT −.016 TIE | T .714, HT −.009 TIE | **slot 1 `4b-D1-a2_3`** |
  | L-D2 | T .708, HT −.034 **FLAG** | T .717, HT −.021 **FLAG** | T .714, HT −.010 TIE | **slot 2 `4b-D2-a1_3`** |
  | L-C | T .748, HT −.017 TIE | T .743, HT −.012 TIE | T .725, HT −.006 TIE | **slot 3 `4b-C-a1`** |

  (4b-I typed DEV T .704; Score5-typed-DEV: no flag at any point; every point passes the typed / Noul floors.)
  hs1-dev (report only): adopt C .793 / D1 .721 / D2 .693 (I .775); false yes .280 / .268 / .295 (I .262).
  - Formal (`m8-formal.sh`, mirror `4d8b40d48…`): node B GPU3 slots 1, 3 and GPU4 slot 2 since 11:34Z; runs under
    `/data/dev2/runs/dec/formal/m8/`, markers `formal/m8/status/`. Next per run: `M6_FORMAL_ROOT=/data/dev2/runs/dec/formal/m8
    m6-relay.sh pull <run>` → node A `m6-score.sh 4b <run>` → `m6-relay.sh mark <run>` → node B `m6-formal.sh 4b mlx
    <run>` → pull `<run>-mlx` → `m6-score.sh 4b mlx <run>`; `M6_EXPOSURE=` the slice receipt for `overlap`; then
    `successor`.
- 2026-09-30 ≈18:45 UTC+8 (10:45Z) — **D arms in members 2–3; C line scored.** Lock part 2 `4d8b40d48`
  (A20r targets: 12,794 / 12,794 rows, parity smoke PASS on all shards, drift 0.0; D1 `0ead0d1b…`, D2 `015de068…`);
  `m8/teacher-a20r/READY` written 10:23Z. Mirror `4d8b40d48…-src_training_decision2` on node B (formal wrapper).
  - Early rules (`m8/early/*.json`): D1 continue (SELECT700 .9000 vs C-m1 .8975), D2 continue (.9033).
  - Running: b3 D1-m2 → D2-m2; b4 D2-m3 → D1-m3 (≈11 min each); soups ≈11:05Z.
  - L-C (vs 4b-I, typed DEV T .704): α 1 T .748 (C / N / S 562 / 277 / 358), HT-DEV v2 −.0165 [−.027, −.006] TIE;
    α ⅔ .743, −.012 TIE; α ⅓ .725, −.006 TIE; Score5-typed-DEV no flags anywhere (4b-I top share .39). Every C point
    passes the floors → C pick α 1 (rules not run yet). C soup hs1-dev: adopt .793 (I .775), false yes .280 (I .262).
  - Next: `M8_GPU=3 m8-lines.sh line D1`, `M8_GPU=4 m8-lines.sh line D2` when `m8/status/D{1,2}.DONE` appear; hs1-dev
    diag of each soup; `m8-relay.sh lines <points>`; node A `m8-score.sh points …` then `m8-score.sh rules`;
    `m8-relay.sh select`; formal via `m8-formal.sh launch <mirror 4d8b40d48…> 4b 3|4 <slots>`.
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
