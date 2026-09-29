# 9B M5 state (resume file)

Updated: 2026-09-29 17:30 UTC+8 (M5 worker; prereg frozen, chains not yet launched)
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

- nothing yet.

## Next

1. Commit + push prereg; mirror HEAD to node A; upload + verify chain files; launch GPU6 / GPU7 chains (chain rule).
2. After soups: `score.sh` + `m5_rules seed`; after lines: `m5_rules alpha` per line → lock per finalist → formal.
3. Successor test vs I (released T = 1 run, or the latest M5 successor handed off); record; gist 05; merge.

## GPU-hours

M5: 0 so far (cap 24; projected ≈ 15.5). CPU only: data builds, exposure, power check.
