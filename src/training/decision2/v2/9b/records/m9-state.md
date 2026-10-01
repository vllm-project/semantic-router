# 9B M9 state (resume file)

## Now (2026-10-02 00:35 UTC+8, 16:35Z; continuation worker cba71646): **M9 CLOSED — no successor; DEV2.0-9B stands**

- **Stage 3 result** [`lux9b-m9-stage3-result-2026-10-01.md`](lux9b-m9-stage3-result-2026-10-01.md): the official
  rules named **K-a13IB** (passes all seven development gates); K-a13IBX fails the Noul floors and Y1.
- **Formal K-a13IB (16:09–16:22Z, node A GPU6): v3 68.024 vs 67.737, +0.29 [−1.57, +1.20] → item 1 FAILS.** Items 2–7
  pass (H +.007 [−.021, +.020]; types OK; mlx −.004 [−.011, +.003]; vs Lux1 16K +2.22 [+0.11, +3.95]; exposure 0;
  public 231 182 vs 178). Typed FINAL .8069 vs .8106 (the +.009 typed-DEV gain did not carry). `formal-m9/K-a13IB.gates/
  items.json` `3e00533b…`, `successor.json` `5e4a7258…`.
- Therefore: no C1 item 8, no release, no private Index run of a release; the optional Noul T+b study (passers only)
  not run. (The custodian's C1 content recheck r1, `v2/eval/records/c1-recheck-r1-2026-10-01.md`, maps K-a13IB /
  K-a13IBX with exposure 0, so item 8 would have been allowed.) Integration fast-forwarded to `aa624ac1b`. Every M9 chain has ended; node A GPU6–7 and node C GPU1–5 leases released (16:27Z; node C GPU6–7 were
  released 14:05Z). **≈ 41.5 of 120 GPU-h used.**
- **Proposed next lever (not launched):** K-a12IB = ½ point of the existing KIB soup (≈ 1 GPU-h, amendment first), then a
  five-seed KIB soup (≈ 6 GPU-h), then typed-row self-distillation + IB additive (≈ 9 GPU-h) if the typed gain stays
  short; IB3 when release-safe. Details in the stage-3 record. Optional: a private Index diagnostic of K-a13IB (≈ 2
  GPU-h) to size the 9B breadth effect.

## Log (continuation cba71646)

**16:06Z official stage-3 rules: finalist K-a13IB** (`select/9b-finalists-s3.json` `4b02f31f…`, readout
`lines/readout/m9-s3.json` `d312f8e7…`). K-a13IB passes every gate (T .9344 vs .9250; HT-DEV v2 −.006 TIE; Y1 −.015
[−.028, −.003]; Y3 .134 vs .152). K-a13IBX fails the Noul type floor (319 < 326), the `rule_precedence` floor (319 <
334) and Y1 (+.032 [+.019, +.045]). COORDINATION re-read (newest 2026-10-02 00:10 UTC+8; no 9B formal-path change;
the custodian C1 content recheck for IB1-r3 + IB2 runs centrally, 3c7679b0). `lock.sh` 16:10Z →
`select/formal-lock-s3.json` `bc4b428c…`, `soup/K-a13IB/SHA256SUMS` `49c6d942…` (16 files); lock record
`records/lux9b-m9-formal-lock-s3-2026-10-01.md` (`5d0c1c234`); `status/formal-s3.GO` 16:09:13Z; **formal K-a13IB
running on node A GPU6 since 16:09:28Z** (chain pid 214241 → `formal.sh` pid 354110; CAL698 fit done, smoke
collecting; ETA ≈ 16:50Z; log `logs/formal-K-a13IB.log`). Next: `items.py verdict` (copy `exposure/kib-subset.json`
node C → node A first), items 1–7 into the stage-3 result record. GPU-hours ≈ 41.3 of 120 at 16:12Z (node C 38.81 by
`gpuh.py`; node A readouts 2.29; C0F parity 0.20) + formal.
Poll 15:37Z: **K-a13IBX seeds DONE** (s2 15:26Z, s1 15:28Z, s3 15:32Z; BEST = checkpoints 1,999 / 1,489 / 1,745)
and the node-C soup `soup/KIBX/build/KIBX-soup` built 15:34Z (`model_sha256` `cc25e396…`); node C GPU1–4 / 6–7 idle
(9B leases, status idle). Node A `post-a-KIBX` pulls it, builds K-a13IBX and reads the panels on GPU7, then runs the
shared readout / contrasts / rules (ETA ≈ 16:10Z); `formal-s3` alive and waiting. Integration merged (`a9df14980`, signed off). New: `lux9b/m9/items.py` (`f1bb2a50f`; items 1–7 from
`successor.json`, the exposure receipts and a line-level TRAIN ⊂ x60 ∪ IB1-r3 ∪ IB2 check), committed before any
formal run. **Item 6 subset checks done (node C, mirror `f1bb2a50f`):** `exposure/kib-subset.json` (`1dfb1843…`):
151,015 of 151,015 K-a13IB TRAIN rows (`2cd09292…`) occur in x60 (`a66131b1…`, the released K file) ∪ IB1-r3 TRAIN
(`1e1b08f3…`, 24,325) ∪ IB2 TRAIN (`ee137efa…`, 24,518); `exposure/kibx-subset.json` (`af6ca288…`): 145,480 of 145,480
(`548b61a5…`). IB exposure receipt `exposure/ib1-ib2-train.json` (node A, `9fd9d3db…`): 0 groups.

Updated: 2026-10-01 22:40 UTC+8 (14:35Z) by continuation worker b23ed249 (hand-off point). **Stage 2 closed 13:28Z: NO FINALIST**
(record `lux9b-m9-stage2-result-2026-10-01.md`; rules `select/9b-finalists-s2.json` `df0ef550…`, readout
`lines/readout/m9-s2.json` `cda87bcf…`; the s2 formal chain stopped by rule). L9IB fails the Choice / Score type
floors, HT-DEV v2 (FLAG −.021), Y1 (+.117) and Y3 (.229 vs .152); L9IBX fails HT-DEV v2 (FLAG −.022), Y1 (+.058) and
Y3 (.271). **Stage 3:** K-a13IB seeds DONE 13:23–13:28Z, soup `soup/KIB/build/KIB-soup` built 13:30Z (node C);
K-a13IB built and scored on node A (13:35 / 13:57Z; passes every gate in a scratch preview, see "Stage 3 interim");
K-a13IBX seeds on GPU2 / 1 / 4 since 12:51–12:57Z (all preflights PASS), 1,273 / 1,282 / 1,215 of ≈ 1,990 at 14:35Z,
ETA ≈ 15:30–15:35Z; every chain PID alive (node C `chain-c{2,1,4}-s3`, `post-c-KIBX`; node A `post-a-KIBX`, `formal-s3`). **Stage 1 closed (no finalist); stage 2
training (unattended); stage 3 (amendment 3) launched 10:47Z.** Branch `xunzhuo/decision-2-training-9b-m9`
(worktree `vllm-sr-dev2-9b-m9`). Prereg `records/lux9b-m9-prereg-2026-10-01.md` (`8570a5896`); amendment 1
(`0b84e0db4`), amendment 2 (`51c80ddc9`, stage 2), **amendment 3 (`787abdc54`, stage 3 = K-a13 recipe + IB1-r3 + IB2 at
matched tokens; pushed before any stage-3 job)**; stage-3 Lux fallback commit `624f94027`.

## Stage 1 verdict (10:09Z): NO FINALIST (record `lux9b-m9-stage1-result-2026-10-01.md`, `8ab82f069`)

L9 fails the Noul floor, HT-DEV v2 (FLAG −.031) and Y1 (+.042); L9L fails Y1 only (+.086). C0 (DEV2.0-9B) retention
.792 ≥ base .781. Typed readout `lines/readout/m9.json`, rules `select/9b-finalists.json`.

## Running chains (detached; liveness by PID + container name; a failed step writes a marker and is never rerun)

| Where | Chain (mirror) | State at 11:00Z |
| --- | --- | --- |
| node C GPU5 / 2 / 4 / 1 | stage 2 `chains2.sh` (`1b0830c0c`): L9IB-s1 / L9IB-s2 / L9IBX-s1 / L9IBX-s2 | all DONE (10:55–12:53Z), chains ended |
| node C (GPU5 / GPU4) | stage 2 `post-c.sh` L9IB / L9IBX (`1b0830c0c`): LoRA merges + soup | done 12:56–12:58Z |
| node A GPU6 / GPU7 | stage 2 `post-a.sh` `M9_STAGE=2` L9IB / L9IBX → `select/9b-finalists-s2.json` | done: scored 13:19–13:21Z, rules 13:28Z (no finalist) |
| node A | stage 2 `formal-chain.sh` `M9_STAGE=2` (`54c420357`) | stopped by rule 13:28Z: no finalist |
| node C GPU3 / 6 / 7 | **stage 3 `chains3.sh` (`787abdc54`)**: K-a13IB-s1 (pre-warm, alone first) / -s2 / -s3 | all DONE 13:23–13:28Z (BEST = last checkpoint: 2,083 / 2,081 / 2,084); GPU3 idle, GPU6–7 released 14:04Z |
| node C GPU2 / 1 / 4 | stage 3 `chains3.sh`: K-a13IBX-s1 / -s2 / -s3 | full runs since 12:53 / 12:51 / 12:57Z (preflights PASS; GPU4 waited for the L9IBX merges), ETA ≈ 15:40Z |
| node C (CPU) | stage 3 `post-c.sh` KIB / KIBX (`787abdc54`): three-seed FP32 soups | KIB built 13:30Z (`soup/KIB/build/KIB-soup`); KIBX waiting |
| node A GPU6 / GPU7 | **stage 3 `post-a3.sh` KIB / KIBX (`624f94027`, pids 215950 / 215963)**: pull → K-a13IB / K-a13IBX = [arm soup, Lux, Lux] (α ⅓) → 10 panels → score vs C0 → `readout/m9-s3.json` → contrasts → `select/9b-finalists-s3.json` | KIB: K-a13IB scored 13:57Z, α 1 IB DEV read 14:00Z, chain ended (its `finish` returns until K-a13IBX is scored); KIBX: waiting for node C's soup, then it runs the shared readout / contrasts / rules |
| node A | stage 3 `formal-chain.sh` `M9_STAGE=3` (`787abdc54`, pid 214241) | waiting for the s3 rules, then **`status/formal-s3.GO`** |

Liveness: node C `chains/chain-c{5,2,4,1}-s2.pid`, `chains/chain-c{3,6,7,2,1,4}-s3.pid`, `chains/post-c-{L9IB,L9IBX,KIB,KIBX}.pid`;
node A `chains/post-a-{L9IB,L9IBX,KIB,KIBX}.pid`, `chains/formal-s2.pid`, `chains/formal-s3.pid`. Logs: both nodes
`/data/dev2/runs/9b/m9/{OPERATIONS.log,logs/,arms/OPERATIONS.log,lines/OPERATIONS*.log}`.

## Stage 3 inputs and checks (amendment 3)

- **TRAIN at K-a13's 60,183,732 tokens per seed** (`data/READY3.json`; `m9/stage3.sh build`, 10:47Z):
  - `kib` `2cd09292…` (teacher `7e3f8bf7…`): 151,015 rows = 102,172 x60 rows (cut budget 50,010,377, realized
    50,270,072) and 48,843 IB rows (10,173,355 tokens); 60,443,427 tokens (+0.43%); IB share 16.8%.
  - `kibx` `548b61a5…` (teacher `65c0c648…`): 145,480 rows = 107,588 x60 + 37,892 IB (7,504,930); 60,383,450 tokens
    (+0.33%); IB share 12.4%. The kib x60 rows are a subset of kibx's (nested, checked). 178 strata.
- **Trainer contract = K-s1's** except the arm name, data hashes, `teacher_partial: true` (IB rows gold only) and
  2,083 planned updates vs 1,624 (IB rows are shorter, ≈ 208 vs ≈ 490 native tokens per row).
- **Lux fallback taken by rule (10:55Z, before any seed finished):** `v2.dec.soup` would refuse K-a13's own Lux
  checkpoint (`m3/pf-D-s1-zero/run/checkpoint-0000000`): its `full_training_source.source_name` is `model` (the package
  was mounted as /model), M9's is `lux`; every fingerprinted file hash is equal. K-a13IB-s1's zero-step checkpoint is
  the α ⅓ base instead; its 9 backbone shards and head are **byte-identical** to K-a13's base (hashed on both nodes;
  `post-a3.sh` re-checks after the pull). The two node-A post chains were idle and were relaunched from `624f94027`.
- x60 ids file (per-row native tokens / pools) copied node A → node C, `7843afb7…` verified.

## Stage 3 interim (K-a13IB scored 13:57Z; the official stage-3 rules run once K-a13IBX is scored)

- **K-a13IB** = [KIB three-seed soup, Lux 1.0, Lux 1.0], built on node A 13:35Z (readout identity `4701ba41…`).
  Screens vs C0 (written by `score.sh points`): HT-DEV v2 −.006 [−.017, +.005] TIE; PN1-dev clean gold-no −.015
  [−.028, −.003] (**less** yes-bias than C0), hop +.000; `hs1-dev` false-yes .134 vs .152; MLX-DEV-9B Noul −.001
  [−.008, +.005], Choice −.000 [−.015, +.014]; retention macro −.000 [−.009, +.009]; Score5-typed-DEV clean.
- **Scratch preview of the gate code** (`scratch/s3-preview-readout.json`, `scratch/s3-preview-rules.json`; NOT the
  rules output, which is `select/9b-finalists-s3.json`): typed T .9344 vs .9250 (C / N / S 800 / 344 / 351 vs 799 /
  338 / 343), `rule_precedence` 344, every family floor held, H3 .5566 vs .5622 → **no failing gate**.
- IB DEV (report): K-a13IB IB1 .963 (+.040 [+.031, +.048] vs C0), IB2 .890 (+.081 [+.061, +.102]); the α 1 soup KIB
  .971 / .910 (+.048 / +.101), so the ⅓ interpolation keeps ≈ 80% of the IB gain (H5).

## Hand-off steps (stage 3)

1. **Wait for `select/9b-finalists-s3.json`** on node A (K-a13IBX seeds ETA ≈ 15:40Z → node-C soup → node-A ⅓ point
   and 10 panels → rules ≈ 16:15Z; log `logs/post-a-KIBX.log`).
2. **If it names finalists:** re-read COORDINATION; then on node A, from `/data/dev2/runs/9b/m9`:
   `bash /data/dev2/src/624f940270e70991cb8d733f24dadb1fbdc92fab-src_training_decision2/src/training/decision2/v2/9b/lux9b/m9/lock.sh
   select/9b-finalists-s3.json lines/readout/m9-s3.json NAME…` (it writes `soup/NAME/SHA256SUMS` once and prints the
   lock JSON; a node-A-built point records `<artifact>.built.json`). Commit the JSON as
   `records/lux9b-m9-formal-lock-s3-2026-10-01.md`, push, then
   `date -u +%FT%TZ > /data/dev2/runs/9b/m9/status/formal-s3.GO`. The waiting chain `formal-s3` (pid 214241, mirror
   `787abdc54`) runs `formal.sh` per finalist on GPU6 / GPU7 (≈ 40 min, ≈ 0.25 GPU-h each) and writes
   `formal-m9/NAME.gates/successor.json`. The C0F formal-path parity is exact, so the stored T = 1 run is the bar
   (v3 67.737).
3. **Items 1–7** from `successor.json` and the gate files: (1) `PAIRED-vs-DEV2.0-9B-T1.json` v3 `ci95.low` > 0;
   (2) `axis_ci95.H.delta.high` ≥ 0; (3) every `types.json` verdict `OK`; (4) `MLX-PAIRED-vs-DEV2.0-9B-T1.json`
   card-eligible Choice + Noul `ci95.high` ≥ 0; (5) vs Lux1 16K (`formal-m9/NAME-16k/PAIRED-vs-Lux1-16K.json`)
   `ci95.low` > 0, `H.delta.high` ≥ 0 vs Lux1 and `PAIRED-vs-Nimble2.json`, types `OK`; (6) exposure: the stage-3
   TRAIN is a subset of x60 ∪ IB1-r3 ∪ IB2 TRAIN, both receipts 0 groups; (7) `PUBLIC231-vs-DEV2.0-9B-T1.json` not
   `REGRESSION`.
4. **A passer of items 1–7:** the T = 1 derivation (`v2.release.retemper_predictions --undo`, as `lux9b/m7/derive_t1.sh`);
   a frozen package dir on node A (checkpoint, `SHA256SUMS`, calibration, scored run, gate files); **the custodian's
   C1 content recheck first (IB-trained)**, then a C1 item-8 spec draft (template
   `v2/eval/sealed/c1-postkey/dev2-27b-a20r.json`); a release hand-off that builds on **the 9B forward-budget fix
   revision `main` `5de3f9ed`** (COORDINATION 21:25; it supersedes `41cb6a08`), with card facts: Lux 1.0 → K-recipe
   full fine-tuning on x60 (stratified cut) + IB1-r3 + IB2 → three-seed soup → ⅓ toward the soup from Lux; own-Lux
   targets on x60 rows, IB rows gold only; IB source licences / attribution per the IB1-r3 and IB2 records (CC BY-SA
   sources need attribution); and a private Index request for IX1 (frozen package; numbers only in
   `decision2-program/private/`). Optional: the CAL-only Noul T+b study (prereg "Calibration") for the chosen finalist.
5. **No finalist / no passer:** stage-3 section of the final results record, gist 05, release node-A GPU6–7 leases.

Stage 2 needs nothing more: no finalist (13:28Z), its formal chain stopped by rule.

Leases: node C GPU6–7 were borrowed from IX1 (released at 09:53Z) and released again at 14:04Z. IX1's owner files are
single-line, which the launch-time archive check did not recognise, so their archives were restored from the
worker's log as `owner.before-9b-m9-20261001T104751Z`. Node C GPU3 (track) is idle; GPU1 / 2 / 4 hold K-a13IBX.

## GPU-hours

≈ 36.1 on node C at 14:35Z (running containers included; stage 1 L9 5.38 / L9L 5.42, stage 2 L9IB 6.27 / L9IBX 6.05,
stage 3 K-a13IB 7.74 / K-a13IBX 5.05 so far, merges 0.17) + node A readouts ≈ 1.97 + formal-path parity 0.20 → **≈ 38.3
of 120**. Stage 2 ≈ 13.0 (closed). Projection: stage 3 ≈ 16.5 (cap 30), formal ≈ 0.25 per finalist → M9 ≈ 40.

## Deviations (this worker)

- The integration merge `135e7c523` (11:00Z) lacks a DCO sign-off; it is on the integration branch, so history is not
  rewritten. Later merges use `--signoff`.
- Node C GPU6–7 lease archives were skipped at launch (single-line foreign owner files) and restored from the log. (`m9/gpuh.py table --running` on node C; node A
`gpuh.py table` double-counts 0.08 of merge receipts copied in with the soup side files.)
