# 9B M9 state (resume file)

Updated: 2026-10-01 17:55 UTC+8 (09:55Z). Branch `xunzhuo/decision-2-training-9b-m9` (worktree `vllm-sr-dev2-9b-m9`).
Prereg `records/lux9b-m9-prereg-2026-10-01.md` (`8570a5896`); amendment 1 (`0b84e0db4`, B0 read through the base's
untied LM head); amendment 2 (`51c80ddc9`, stage 2 = L9 recipe + IB1-r3 + IB2 and the transfer-only ablation, started
as node-C GPUs free). Every amendment preceded the GPU jobs it governs; no stage-1 arm had been read.

## Everything runs unattended (detached chains; liveness by PID + container, first log lines confirmed)

| Where | Chain | State at 08:05Z |
| --- | --- | --- |
| node C GPU1–4 | stage 1 `chains.sh` c1 L9-s1, c2 L9-s2, c3 L9L-s1, c4 L9L-s2 (mirror `8570a5896`) | **all four DONE 09:40–09:47Z.** BEST (SELECT700 family macro): L9-s1 1,624 (.8766), L9-s2 1,621 (.8689), L9L-s1 1,624 (.8646), L9L-s2 1,216 (.8763) |
| node C GPU1 / GPU3 | `post-c.sh` L9 / L9L (`fb2cc87cc`): LoRA merges + soup after both seeds | **done 09:49–09:50Z**: L9 soup `56237917…`, L9L soup `baf566e1…` (dec identities); merges agree with the adapters on 128 SELECT rows |
| node A GPU6 / GPU7 | `post-a.sh` L9 / L9L (`fb2cc87cc`): pull soup → 8 panels → scoring vs C0 → typed readout + rules (`select/9b-finalists.json`) | soups pulled (content manifests equal), readouts running from 09:51Z |
| node A | `formal-chain.sh` (`ad272def4`): no finalist → stop; else (parity already done, see below) **waits for `status/formal.GO`** (write it only after re-reading COORDINATION and pushing a finalist lock record) | waiting for the rules |
| node C GPU5 / 2 / 4 / 1 | stage 2 `chains2.sh` (`1b0830c0c`): L9IB-s1 (GPU5, started 07:46Z), L9IB-s2 (GPU2), L9IBX-s1 (GPU4), L9IBX-s2 (GPU1, after the L9 merges) | L9IB-s1 full run (ETA ≈ 10:55Z); L9IB-s2 (GPU2), L9IBX-s1 (GPU4), L9IBX-s2 (GPU1) started 09:46–09:50Z after the stage-1 chains / L9 merges; ETA ≈ 13:00Z |
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

≈ 3.0 at 07:47Z (node C preflights + running seeds; node A readouts ≈ 0.4; B0 0.03). Projection: stage 1 ≈ 12,
stage 2 ≈ 16, readouts / formal ≈ 4. Cap 120.

## Next (if this worker has handed off)

1. When `select/9b-finalists.json` exists (≈ 10:30Z): read it and `lines/readout/m9.json`; re-read COORDINATION;
   for finalists, push a lock record (soup `SHA256SUMS` on node A, rules-output hashes), then
   `date -u +%FT%TZ > /data/dev2/runs/9b/m9/status/formal.GO` on node A. The formal chain then runs each finalist
   (`formal-m9/NAME-16k`, gates in `formal-m9/NAME.gates/`); apply items 1–7 from `successor.json` and the C0F parity.
2. Stage 2 lands ≈ 13:00–13:30Z: rules in `select/9b-finalists-s2.json`; formal for ≤ 2 passers the same way.
3. Results record, gist 05 entry, private Index request for frozen finalists (IX1 harness), C1 item-8 spec.
