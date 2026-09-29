# 9B M5 state (resume file)

Updated: 2026-09-30 01:20 UTC+8 (M5 worker; **M5 complete: no finalist, no successor**)
Branch: `xunzhuo/decision-2-training-9b` (merge-only into `xunzhuo/decision-2-training`)
Prereg: `records/lux9b-m5-prereg-2026-09-29.md` (`567e7fd39`); result: `records/lux9b-m5-result-2026-09-29.md`.

## Done

- Data (`4c375723f`): KD / KG TRAIN frozen (train `86a41afb…`, exposure clean). Tools `4e8d963e3`, wrappers `facd4f846` /
  `a7b326315`, chains `lux9b/m5/chains/`.
- References re-read = M4 exactly (R = K-a13 T .925, H3 .5622, P 73.22).
- Arms KD and KG, two seeds each: preflights PASS, all steps exit 0; both soups won the seed rule.
- Lines KD / KG / UM5 (α ⅓, ½, ⅔, 1) read and scored. `lux9b.m5_rules`: KD no eligible α, KG no eligible α, UM5 ⅓ eligible
  but G* = .005 < .01 → **no finalists** (`m5/rules/finalists.json` `f566bd56…`). Per the prereg: no formal runs, no successor.
- Cause (development): the A7 dose collapses Noul rule_precedence (soups 211–221 / 400 vs K soup 309) while raising Score.
- Node A: KD / KG soups kept with `SHA256SUMS` (`e377ca2e…`, `1aeccdeb…`); GPU6–7 leases set idle 17:11Z.
- Result record, gist 05 entry, merge into integration.

## Running

- nothing.

## GPU-hours

M5 total 14.88 GPU-h (job receipts; ≈ 15.3 chain wall-clock) + 0.55 CPU-h soups; cap 24.
