# 9B M5 state (resume file)

Updated: 2026-09-29 17:25 UTC+8 (M5 worker; prereg `567e7fd39` frozen and pushed; chains running)
Branch: `xunzhuo/decision-2-training-9b` (merge-only into `xunzhuo/decision-2-training`)
Prereg: `records/lux9b-m5-prereg-2026-09-29.md` (arms KD / KG, lines KD / KG / UM5, successor test vs the released T = 1 run)

## Done

- Merged origin `xunzhuo/decision-2-training` (fast-forward to `28f05ace7`, the DEV2.0-8B release merge).
- Power check (CPU, node A `m5/power/`): vs the released T = 1 run `/data/dev2/runs/release/dev2-8b-t1-derived` —
  K-a13 CAL698 run 0 [0, 0]; U-a13 +0.129 [−2.199, +1.014]; KN-a12 +0.244 [−1.514, +2.282]; DW +0.834 [−4.483, +3.374].
- Data (CPU, `4c375723f`; node A `m5/data/`): KD train `86a41afb…` (171,351 rows / 84.18M tokens = x60 byte-identical +
  48,700-row / 24.0M-token family-equal A7 Stage dose), KD teacher `0f9e6ed2…`, KG teacher `cdcd99c1…` (= M4 x60);
  exposure `5639cfa3…` groups [] (methods agree).
- Tools: `4e8d963e3` (m5_rules, mlx_paired, score_levels; KN-a12 vs K-a13 mlx card parts −.0198 [−.0294, −.0104]; K-a13 Score
  level-0 recall .514); wrappers `facd4f846` + `a7b326315` (`lux9b/m5/`; readouts / formal on runner mirror `3277dec9d`).

## Running

- Mirror node A `/data/dev2/src/567e7fd399f951932a0d7980a0b11718d9efb813-src_training_decision2` (tree `52cc38ae…`,
  `mirror_to_node.sh --verify` OK). Chain files (`lux9b/m5/chains/`) uploaded to node A `m5/logs/chains/` and verified
  (`upload_chain.sh`): `m5-gpu6.sh` 1,292 B `49a81cf3…`, `m5-gpu7.sh` 1,468 B `48a18381…`; then launched separately with
  `launch.sh` (arg: the mirror SHA).
- **m5-gpu6** PID 3218725 (09:18:18Z): ref-ka13 → KD-s1 (preflights) → KD-s2 → KD-soup → KD-a13 / a12 / a23.
- **m5-gpu7** PID 3219064 (09:18:20Z): ref-lux → KG-s1 (preflights) → KG-s2 → KG-soup → KG-a13 / a12 / a23.
- Liveness check 09:19Z: both PIDs alive (`alive.sh`), containers `d2-9b-m5-ref-ka13-dev-g6` / `d2-9b-m5-ref-lux-dev-g7` up,
  first step-log lines present, leases `track=9b-m5`. Check with
  `bash $L/alive.sh m5-gpu6` (L = the mirror's `v2/9b/lux9b/m5`); logs node A `/data/dev2/runs/9b/m5/logs/`.

## Next

1. Verify ref-ka13 reproduces M4's K ⅓ readout (T .925, 799 / 338 / 343, H3 .562, P 73.22).
2. After soups: `score.sh` + `m5_rules seed`; after lines: `m5_rules alpha` per line → lock per finalist → formal.
3. Successor test vs I (released T = 1 run, or the latest M5 successor handed off); record; gist 05; merge.

## GPU-hours

M5: 0 so far (cap 24; projected ≈ 15.5). CPU only: data builds, exposure, power check.
