# Arm factory — state (running log, newest first; prereg `af-prereg-2026-10-02.md`)

Index values stay private (node private stores and `decision2-program/private/arm-factory/`); this file has none.

## 2026-10-02 16:00Z (00:00 UTC+8) — first soups built; first Index run on node F

- **Seeds DONE** (no failure, no cap stop): node C `4b-LHS17IB4` s4 / s5, `4b-SDMLIB4` s4 / s5, `4b-LHS17UP` s3; node F
  `4b-LHS17IB4ML` s1 / s2, `4b-LHS23IB4` s1 / s2 (15:45–15:50Z). Still training: node C `4b-LHS17IB4-lrh` s1 / s2,
  `4b-LHS17UP` s4 (≈ 17:05Z); node A all six 9B seeds (≈ 17:45–18:05Z).
- **Soups** (each LoRA seed merged first; SELECT agreement 127–128 / 128, max drift ≤ .011):
  `4b-LHS17IB4-s45` `ce5c1401…`, `4b-SDMLIB4-s45` `16b34cfb…` (node C, shipped to node F, lists equal),
  `4b-LHS17IB4ML` `9cb18a6a…`, `4b-LHS23IB4` `2d0a50bd…` (node F). Building on node F: `4b-LHS17IB4-x5`, `4b-SDMLIB4-x5`.
- **Index:** `AF-4b-LHS17IB4ML-bf16` staged (BF16 copy + restage), parity PASS, shards on node F GPU2–5 since 15:57Z;
  `AF-4b-LHS23IB4-bf16` follows when its shards end.
- **Amendment 3** (`23d9af8aa`): references = the current releases' runs (Nox `DEV2.0-4B-SDMLxALL-bf16`, Lux
  `M10-KIB4-a40-bf16`), copied to nodes F / A with the results hash pinned. `4b-LHS17IB4ML`'s chain was started
  before the switch, so it bootstraps against `IS-4b-LHA10SDML-bf16`; its current-release bootstrap is added on CPU.
- **Operational incident (no effect on any result):** a `pkill -f` pattern used to restart two waiting measurement
  pipelines also matched the restarting ssh shell. Only the waiting scripts ended; the running Index chain, the
  built soups and every training seed were unaffected; both pipelines were relaunched.
- Node C GPU6 / GPU7 are idle and released (not needed before ≈ 17:10Z).

## 2026-10-02 15:20Z (23:20 UTC+8) — unattended soups and Index pipelines armed

- `4b-AFxALL`'s base set (amendment-2 rule, decided on M17's two Index runs, values private): `SDMLxALL9`'s nine.
- Detached node-side pipelines (mirror `91cdf96cc`), each waiting for its seeds:
  - node F: soups `4b-LHS17IB4ML` (merges on GPU2), `4b-LHS23IB4` (GPU4), then their Index runs (panel-8, GPU2–5,
    8 shards greedy) as soon as each is staged;
  - node C: soups `4b-LHS17IB4-s45` (GPU7), `4b-SDMLIB4-s45` (GPU6), `4b-LHS17UP-s34` (GPU5), `4b-LHS17IB4-lrh` (GPU3);
  - node A: `KF`, `KF-a40`, `KF-a50`, `KFK`, `KFxKIB-a40`, `KIB4W2-a40`, `KIB4L2-a40`, `KIB4Q` built in sequence (CPU),
    then the Index runs of `KF-a40`, `KF-a50`, `KFxKIB-a40` (panel-7, GPU1–6, 7 shards greedy).
- Seed ends (estimates): node C ≈ 15:30–15:55Z then ≈ 17:05Z (second items); node F ≈ 16:20Z; node A ≈ 17:45–18:05Z
  (≈ 3.4 GPU-h per 9B seed).

## 2026-10-02 15:05Z (23:05 UTC+8) — all 15 preflights PASS; candidates pinned (amendment 2)

- All 15 seeds passed their preflights (node A 14:29–14:38Z, node C 14:27–14:35Z, node F 14:38–14:44Z) and are in
  their full runs. Training GPU-h so far ≈ 4.7.
- Amendment 2 (`b1b34a3a9`): one arm, one vote (factory seeds extend M17's arm soups to `-x5` / `-x4`), the exact
  members of `4b-AFxALL` / `4b-AFxALL2`, the 9B family soup `KF` and its points, and the measurement order.
- Imported read-only to node A (node B → C → A, per-file SHA-256 lists equal): M10 `KIB4P` (KIB4 s1–s3, model
  `159bfda1…`) as `soup/m10-KIB4P`; K-a13IB's arm soup `KIB` (model `794ebfd2…`) as `soup/m9-KIB`.
- Node F is a full 4B Index node now: the scoring environment and `IS-4b-LHA10SDML-bf16`'s merged results (SHA-256
  `a459ce7c…`) copied from node C.
- IX1 entries `AF-<name>-bf16` (`6df5d9264`, a separate commit to `v2/eval/ix1/launch.sh`).

## 2026-10-02 14:35Z (22:35 UTC+8) — 15 GPUs training

| Node / GPU | Chain (items in order) | Since |
| --- | --- | --- |
| A1 | `KIB4` s4 (seed 3; pre-warm, preflight PASS 14:29Z) | 14:23Z |
| A2 / A3 / A4 / A5 / A6 | `KIB4` s5 (4); `KIB4W2` s1 (5) / s2 (6); `KIB4L2` s1 (7) / s2 (8) | 14:30Z |
| C3 | `4b-LHS17IB4` s4 (20260929; pre-warm, PASS 14:27Z), then `4b-LHS17IB4-lrh` s1 (20260926) | 14:22Z |
| C4 | `4b-LHS17IB4` s5 (20260930), then `4b-LHS17IB4-lrh` s2 (20260927) | 14:28Z |
| C5 / C6 / C7 | `4b-SDMLIB4` s4 (20260929) then `4b-LHS17UP` s4 (20260929); `4b-SDMLIB4` s5 (20260930); `4b-LHS17UP` s3 (20260928) | 14:28Z |
| F2 / F3 / F4 / F5 | `4b-LHS17IB4ML` s1 (pre-warm) / s2; `4b-LHS23IB4` s1 / s2 (amendment 1) | 14:33Z |

- Data locks (`READY-af.json` per node): wave-1 4B entries equal M17's locks; `KIB4` equals M10's node B lock
  (`2e72bcfd…` / `377f8878…`); `KIB4W2` adds weights `ea81c621…` (9,459 IB4 rows ×2; IB4 loss-weight share .063 →
  .118); wave-2 4B files per amendment 1 (audit PASS).
- Node F GPU2–5: stale harness leases of M17's finished `SDMLxALL15` run moved to `owner.prev-af-*`.

## 2026-10-02 14:20Z (22:20 UTC+8) — prereg and ops

- Prereg and ops (`v2/af/ops/`: `af-launch.sh`, `af-arm.sh`, `af-chain.sh`, `af_arms.py`, `af_gpuh.py`,
  `af_weights.py`) committed before any arm-factory GPU job.
- Free at 14:05Z: node A GPU1–6 (GPU7 runs M10's KIB4-a40 prerelease), node C GPU3–7. Node F GPU2–7 and node B
  GPU2 / 4 / 6 / 7 hold the M17 `SDMLxALL15` and M10 `KIB4P-a33` Index runs.
- Assets relayed node to node: M17's 4B locks, data, Triton caches and Qwen3.5-4B-Base (node F → node C through
  node A); M10's KIB4 TRAIN / teacher and IB4 p1 (node B → node C → node A).
