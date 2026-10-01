# 9B M9 state (resume file)

Updated: 2026-10-01 20:20 UTC+8 (12:20Z) by continuation worker b23ed249. **Poll 12:20Z:** L9IB-s2 1,891 / 2,344,
L9IBX-s1 1,821 / 2,196, L9IBX-s2 1,801 / 2,187 (ETA ≈ 12:48–12:54Z); K-a13IB-s1 / s2 / s3 1,190 / 1,185 / 1,134 of
2,083 (≈ 13 updates / min, ETA ≈ 13:29–13:33Z); every chain PID alive. **Stage 1 closed (no finalist); stage 2
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
| node C GPU5 / 2 / 4 / 1 | stage 2 `chains2.sh` (`1b0830c0c`): L9IB-s1 / L9IB-s2 / L9IBX-s1 / L9IBX-s2 | **L9IB-s1 DONE 10:55Z**; the others ≈ 400 / 2,344 at 10:25Z, ETA ≈ 13:05–13:15Z |
| node C (GPU5 / GPU4) | stage 2 `post-c.sh` L9IB / L9IBX (`1b0830c0c`): LoRA merges + soup | waiting for both seeds |
| node A GPU6 / GPU7 | stage 2 `post-a.sh` `M9_STAGE=2` L9IB / L9IBX (pids 137614 / 137627) → `select/9b-finalists-s2.json` | waiting for node C's soups |
| node A | stage 2 `formal-chain.sh` `M9_STAGE=2` (`54c420357`, pid 206896) | waiting for the s2 rules, then **`status/formal-s2.GO`** |
| node C GPU3 / 6 / 7 | **stage 3 `chains3.sh` (`787abdc54`)**: K-a13IB-s1 (pre-warm, alone first) / -s2 / -s3 | s1 preflight PASS 10:54Z (pre-warm marker 10:52Z), s2 / s3 preflights passing; full runs ETA ≈ 13:30–13:40Z (2,083 updates each) |
| node C GPU2 / 1 / 4 | stage 3 `chains3.sh`: K-a13IBX-s1 / -s2 / -s3 | queued on the GPUs' flocks behind the stage-2 chains (GPU4 also after node C's L9IBX merges); ETA start ≈ 13:15–13:25Z, end ≈ 16:00Z |
| node C (CPU) | stage 3 `post-c.sh` KIB / KIBX (`787abdc54`): three-seed FP32 soups | waiting for the seeds |
| node A GPU6 / GPU7 | **stage 3 `post-a3.sh` KIB / KIBX (`624f94027`, pids 215950 / 215963)**: pull → K-a13IB / K-a13IBX = [arm soup, Lux, Lux] (α ⅓) → 10 panels → score vs C0 → `readout/m9-s3.json` → contrasts → `select/9b-finalists-s3.json` | waiting for node C's soups |
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

## Hand-off steps

1. **Stage 2 rules** (`select/9b-finalists-s2.json`, ETA ≈ 14:00–14:30Z). If finalists: re-read COORDINATION; on node A
   `bash /data/dev2/src/<mirror>/src/training/decision2/v2/9b/lux9b/m9/lock.sh select/9b-finalists-s2.json
   lines/readout/m9-s2.json NAME…` (from `/data/dev2/runs/9b/m9`), commit its JSON as
   `records/lux9b-m9-formal-lock-s2-2026-10-01.md`, push, then `date -u +%FT%TZ > /data/dev2/runs/9b/m9/status/formal-s2.GO`.
   The chain runs `formal-m9/NAME-16k` (+ `-smoke`, `-16k-mlx`) → `formal-m9/NAME.gates/successor.json` (items 1–7
   numbers; the stored T = 1 run is the bar, C0F parity exact).
2. **Stage 3 rules** (`select/9b-finalists-s3.json`, ETA ≈ 16:30–17:15Z): the same lock / GO procedure with
   `select/9b-finalists-s3.json`, `lines/readout/m9-s3.json`, record `records/lux9b-m9-formal-lock-s3-2026-10-01.md`
   and `status/formal-s3.GO` (≤ 4 formal candidates in M9 overall).
3. **A passer of items 1–7:** T = 1 derivation, frozen package (built on top of the 9B forward-budget fix revision of
   `main` `41cb6a08`), C1 item-8 spec (template `v2/eval/sealed/c1-postkey/dev2-27b-a20r.json`) **after the custodian's
   C1 content recheck (IB-trained)**, release hand-off, private Index request for IX1 (numbers only in
   `decision2-program/private/`). Optional CAL-only Noul T+b study (prereg "Calibration") for the chosen finalist only.
4. **No finalist in a stage:** record its development results (stage-2 / stage-3 sections of the final results record)
   and the gist 05 entry.

## GPU-hours

≈ 16.95 on node C at 10:47Z (running containers included) + node A ≈ 1.23 → **≈ 18.2 of 120**. Projection: stage 2
≈ 16 in total, stage 3 ≈ 18 (cap 30), formal ≈ 0.2 per finalist. (`m9/gpuh.py table --running` on node C; node A
`gpuh.py table` double-counts 0.08 of merge receipts copied in with the soup side files.)
