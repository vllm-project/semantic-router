# 9B M9 state (resume file)

Updated: 2026-10-01 18:25 UTC+8 (10:25Z). **Hand-off point: stage 1 is closed (no finalist; result record `lux9b-m9-stage1-result-2026-10-01.md`); stage 2 runs unattended.** Branch `xunzhuo/decision-2-training-9b-m9` (worktree `vllm-sr-dev2-9b-m9`).
Prereg `records/lux9b-m9-prereg-2026-10-01.md` (`8570a5896`); amendment 1 (`0b84e0db4`, B0 read through the base's
untied LM head); amendment 2 (`51c80ddc9`, stage 2 = L9 recipe + IB1-r3 + IB2 and the transfer-only ablation, started
as node-C GPUs free). Every amendment preceded the GPU jobs it governs; no stage-1 arm had been read.

## Stage-1 verdict (10:09Z): NO FINALIST, no stage-1 formal run

Rules `select/9b-finalists.json` (`7b43dad3…`), typed readout `lines/readout/m9.json` (`4ab78036…`); development
readouts, never release scores. Every contrast is against C0 (DEV2.0-9B) on the M9 node-A path.

| Point | typed T | C / N / S | RP | H3 | HT-DEV v2 Δ | PN1 clean gold-no (Δ [CI]) | hop | hs1 false-yes | MLX-DEV-9B Noul / Choice Δ | retention macro (Δ [CI]) | gates failed |
| --- | ---: | --- | ---: | ---: | --- | --- | ---: | ---: | --- | --- | --- |
| C0 | .9250 | 799 / 338 / 343 | 338 | .5622 | — | .262 | .987 | .152 | — | .792 | (reference) |
| L9 | .9369 | 789 / 330 / 380 | 330 | .5841 | −.031 [−.046, −.018] FLAG | .305 (+.042 [+.023, +.060]) | .987 | .140 | +.025 / +.007 | .814 (+.022 [+.011, +.034]) | Noul floor (330 < 334), HT-DEV v2 FLAG, Y1 |
| L9L | .9287 | 784 / 336 / 366 | 336 | .5558 | −.006 [−.019, +.006] TIE | .348 (+.086 [+.066, +.106]) | .987 | .152 | +.021 / +.007 | .807 (+.015 [+.004, +.026]) | Y1 only |

- L9 − L9L: HT-DEV v2 −.025 [−.040, −.012] FLAG, retention +.008 [−.003, +.019], PN1 clean gold-no −.044
  [−.062, −.025], typed +.008 (Score +14, Noul −6, Choice +5).
- H1 (retention): supported (L9 +.022, CI above 0). H2 (transfer without the yes-bias): not supported. H3: the
  base start transfers worse to held-out decision families than the Lux start at 9B, with less near-miss yes-bias.
- The formal chain stopped by rule ("no finalist: no formal run"). Stage 2 (L9 recipe + IB) keeps running as
  preregistered (amendment 2); it is gated on its own.

## Everything runs unattended (detached chains; liveness by PID + container, first log lines confirmed)

| Where | Chain | State at 08:05Z |
| --- | --- | --- |
| node C GPU1–4 | stage 1 `chains.sh` c1 L9-s1, c2 L9-s2, c3 L9L-s1, c4 L9L-s2 (mirror `8570a5896`) | **all four DONE 09:40–09:47Z.** BEST (SELECT700 family macro): L9-s1 1,624 (.8766), L9-s2 1,621 (.8689), L9L-s1 1,624 (.8646), L9L-s2 1,216 (.8763) |
| node C GPU1 / GPU3 | `post-c.sh` L9 / L9L (`fb2cc87cc`): LoRA merges + soup after both seeds | **done 09:49–09:50Z**: L9 soup `56237917…`, L9L soup `baf566e1…` (dec identities); merges agree with the adapters on 128 SELECT rows |
| node A GPU6 / GPU7 | `post-a.sh` L9 / L9L (`fb2cc87cc`) | done: scored 10:06–10:08Z, rules 10:09Z |
| node A | `formal-chain.sh` (`ad272def4`), stage 1 | **stopped by rule: no finalist** (10:09Z) |
| node A | `formal-chain.sh` with `M9_STAGE=2` (`54c420357`, pid 206896): reads `select/9b-finalists-s2.json`; no finalist → stop; else **waits for `status/formal-s2.GO`**, then `formal.sh` per finalist (GPU6 / GPU7) | waiting for the stage-2 rules |
| node C GPU5 / 2 / 4 / 1 | stage 2 `chains2.sh` (`1b0830c0c`): L9IB-s1 (GPU5, started 07:46Z), L9IB-s2 (GPU2), L9IBX-s1 (GPU4), L9IBX-s2 (GPU1, after the L9 merges) | at 10:19Z: L9IB-s1 update 1,892 / 2,343 (ETA ≈ 10:55Z); L9IB-s2 296 / 2,344, L9IBX-s1 304 / 2,196, L9IBX-s2 248 / 2,187 (≈ 12 / min → ETA ≈ 12:40–12:50Z); every chain PID alive |
| node C GPU5 / GPU4 | `post-c.sh` L9IB / L9IBX | waiting |
| node A GPU6 / GPU7 | `post-a.sh` with `M9_STAGE=2` (L9IB / L9IBX): + IB1 / IB2 DEV diagnostics, rules `select/9b-finalists-s2.json` | waiting |

Logs: node C / node A `/data/dev2/runs/9b/m9/{OPERATIONS.log,logs/,arms/OPERATIONS.log,lines/OPERATIONS*.log}`.

## Inputs and checks

- x60 TRAIN `a66131b1…` / own-Lux targets `cdcd99c1…`, SELECT700 `32a4352d…`, CAL698 `19cc1a8c…`, Lux 1.0 package
  (tree `6075a339…`), node-C training cache (copy of `m4/triton-cache`, tree `579ca841…`): `data/READY.json`.
  Qwen3.5-9B-Base `68c46c4b…` via the host HF CLI on both nodes, byte-identical to node A's older copy.
- **Stage-2 TRAIN** (`data/READY2.json`): L9IB `a2398217…` (171,494 rows; IB tokens 10.17M), L9IBX `ec17d822…`
  (160,543 rows; IB tokens 7.50M); IB1-r3 / IB2 TRAIN + DEV match the release records. The first two builds stopped
  on a too-strict lineage check (IB groups hold several rows; fixed in `1b0830c0c`; partial outputs kept as
  `data/l9ib.failed-build-{1,2}`, CPU only).
- **Item 6 for stage 2:** exposure receipt of the IB1 + IB2 TRAIN files vs the K-a13 payload `2194716a…`:
  `groups: []`, methods agree (`exposure/ib1-ib2-train.json` `9fd9d3db…`). x60's receipt is `groups: []`.
- **M9 probe panel:** the M10 panel minus 250 GSM8K items with x60 13-gram hits → 2,839 items (MMLU 1,265, ARC 824,
  GSM8K 750); prompts `4c2ab4ad…`.

## Reference readouts (node A M9 path, T = 1; development only)

- **C0 path parity:** C0's M9 readouts equal M7's stored `ref-ka13` answers on typed DEV, CSS pilot, HT-DEV v2 and
  PN1 dev (0 differences; probability drift ≤ .18 only because M7 read with CAL698 temperatures).
- **Retention probe macro (MMLU / ARC / GSM8K):** C0 (DEV2.0-9B) .792; Lux 1.0 .760 (Δ −.032 [−.044, −.020]);
  B0 = Qwen3.5-9B-Base through its own LM head .781 (Δ −.011 [−.027, +.004]; MMLU .760, ARC .954, GSM8K .628).
  **Unlike 4B, the released 9B lineage did not lose knowledge against its base on these probes** (C0 ≥ base).
- Lux 1.0 vs C0 (report): HT-DEV v2 −.003 TIE; PN1 clean gold-no −.075 [−.095, −.057], hop −.025; hs1-dev
  false-yes .128 vs .152; MLX-DEV-9B Noul-ML −.045, Choice-ML −.074.

## Formal-path parity (preregistered first step; run early because it needs no M9 result)

`formal.sh 6 C0F` (runner mirror `ad272def4`, image `f83b1d10…`, one copy of the frozen `formal-m3` cache, tree
`af623300…`): the CAL698 fit reproduces K-a13's temperatures exactly (Choice 1.4203, Noul 1.0368, Score 0.5626), and the
collection has **0 answer differences** against the stored bar `release/dev2-8b-t1-derived` on typed FINAL, CSS15
and public 231 (`formal-m9/C0F.parity.json`; probability drift ≤ .29 only from the temperatures). v3 67.737 (T .8106,
H .5660), paired vs the bar 0.0 [0.0, 0.0], public 231 178, mlx-diag Δ 0.0. **The stored run is the paired bar.**

## Dry test of the gate code (scratch, not a rules output)

`m9_rules.py` on Lux 1.0 as a pseudo point against C0 (`scratch/rules-test.json`) runs end to end: C0's typed
readout reproduces M7's R exactly (T .9250, C / N / S 799 / 338 / 343, `rule_precedence` 338, H3 .5622); Lux 1.0 fails
the Noul floors, MLX-DEV-9B and retention, and passes the yes-bias guard, as expected.

## GPU-hours

**≈ 16.3 used at 10:19Z** (node C 15.07 incl. running containers: L9 5.38, L9L 5.42, L9IB 3.11, L9IBX 1.04, merges
0.08, B0 0.04; node A: readouts 1.03, formal-path parity 0.20). Stage 1 ≈ 12.1 (closed). Stage 2 projection ≈ 16
including its readouts; any stage-2 formal ≈ 0.2 per finalist. Cap 120. (`m9/gpuh.py table --running` on node C;
node A `gpuh.py table` double-counts 0.08 of merge receipts copied in with the soup side files.)

## Hand-off: finishing stage 2 (a continuation worker; everything below is already running or scripted)

1. **Wait** for node C's stage-2 seeds (`status/m9-L9IB-s{1,2}`, `m9-L9IBX-s{1,2}` `.DONE`; ETA ≈ 10:55Z for
   L9IB-s1, ≈ 12:45–13:05Z for the others), the node-C soups (`soup/L9IB|L9IBX/DONE`) and node A's stage-2 post
   chains (pull → 10 panels incl. `ib1dev` / `ib2dev` → `score.sh points C0` → `readout/m9-s2.json` → contrasts vs L9
   → `select/9b-finalists-s2.json`). Liveness: node C `chains/chain-c{5,2,4,1}-s2.pid`, `chains/post-c-L9IB|L9IBX.pid`;
   node A `chains/post-a-L9IB|L9IBX.pid`, `chains/formal-s2.pid`. A failed step writes `status/failed-<ARM>` and is
   never rerun.
2. **If the stage-2 rules name finalists:** re-read COORDINATION; on node A run
   `bash /data/dev2/src/<mirror>/src/training/decision2/v2/9b/lux9b/m9/lock.sh select/9b-finalists-s2.json
   lines/readout/m9-s2.json NAME…` (from `/data/dev2/runs/9b/m9`), commit its JSON as
   `records/lux9b-m9-formal-lock-s2-2026-10-01.md`, push, then
   `date -u +%FT%TZ > /data/dev2/runs/9b/m9/status/formal-s2.GO`. The chain runs `formal-m9/NAME-16k` (+ `-smoke`,
   `-16k-mlx`) and writes `formal-m9/NAME.gates/successor.json`; the formal-path parity of DEV2.0-9B is already exact
   (`formal-m9/C0F.parity.json`), so the stored T = 1 run is the bar. Items 1–7 read `successor.json` (item 4:
   `MLX-PAIRED-vs-DEV2.0-9B-T1.json` `ci95.high` ≥ 0; item 5: `PAIRED-vs-Lux1-16K.json` + Nimble2; item 6: exposure
   receipts `exposure/ib1-ib2-train.json` and x60's, both 0 groups; item 7: `PUBLIC231-vs-DEV2.0-9B-T1.json`).
3. **For a passer of items 1–7:** the T = 1 derivation (`v2.release.retemper_predictions --undo`), a frozen package
   for release engineering (card facts: Qwen3.5-9B-Base → merged LoRA; x60 + IB1-r3 + IB2; own-Lux teacher), a C1
   item-8 spec (the 27B spec `v2/eval/sealed/c1-postkey/dev2-27b-a20r.json` is the template; image `host2`, the
   formal-m9 cache; **the custodian's C1 content recheck first: the model is IB-trained**) and a private Index request
   for IX1 (numbers only in `decision2-program/private/`).
4. **If no stage-2 finalist:** M9 ends with no successor; write the final results record (stage 2 section) and the
   gist 05 entry. The optional CAL-only Noul T+b study applies only to a finalist and was not run.
