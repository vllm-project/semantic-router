# 9B M9 state (resume file)

Updated: 2026-10-01 16:05 UTC+8 (08:05Z). Branch `xunzhuo/decision-2-training-9b-m9` (worktree `vllm-sr-dev2-9b-m9`).
Prereg `records/lux9b-m9-prereg-2026-10-01.md` (`8570a5896`); amendment 1 (`0b84e0db4`, B0 read through the base's
untied LM head); amendment 2 (`51c80ddc9`, stage 2 = L9 recipe + IB1-r3 + IB2 and the transfer-only ablation, started
as node-C GPUs free). Every amendment preceded the GPU jobs it governs; no stage-1 arm had been read.

## Everything runs unattended (detached chains; liveness by PID + container, first log lines confirmed)

| Where | Chain | State at 08:05Z |
| --- | --- | --- |
| node C GPU1–4 | stage 1 `chains.sh` c1 L9-s1, c2 L9-s2, c3 L9L-s1, c4 L9L-s2 (mirror `8570a5896`) | all four preflights PASS (07:06–07:11Z); full runs at ≈ 11 updates / min of 1,624 → ETA ≈ 09:40–09:50Z |
| node C GPU1 / GPU3 | `post-c.sh` L9 / L9L (`fb2cc87cc`): LoRA merges + soup after both seeds | waiting |
| node A GPU6 / GPU7 | `post-a.sh` L9 / L9L (`fb2cc87cc`): pull soup → 8 panels → scoring vs C0 → typed readout + rules (`select/9b-finalists.json`) | waiting |
| node A | `formal-chain.sh` (`ad272def4`): no finalist → stop; else parity run C0F (GPU6) vs the stored T = 1 bar, then **waits for `status/formal.GO`** (write it only after re-reading COORDINATION and pushing a finalist lock record) | waiting for the rules |
| node C GPU5 / 2 / 4 / 1 | stage 2 `chains2.sh` (`1b0830c0c`): L9IB-s1 (GPU5, started 07:46Z), L9IB-s2 (GPU2), L9IBX-s1 (GPU4), L9IBX-s2 (GPU1, after the L9 merges) | s1 running; the others wait on the GPU flocks |
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
