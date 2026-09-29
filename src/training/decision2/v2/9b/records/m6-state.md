# 9B M6 state (resume file)

Updated: 2026-09-30 03:00 UTC+8 (M6 worker; **data frozen; training chains launching**)
Branch: `xunzhuo/decision-2-training-9b` (merge-only into `xunzhuo/decision-2-training`)
Prereg: `records/lux9b-m6-prereg-2026-09-30.md` (`f41402e68`) + amendment 1 (`7380a3cbf`). Code / wrappers mirror
`7380a3cbf` on node A (created, verified); runtime mirror `3277dec9d` (verified, reused).

## Plan

- Arms on the K recipe (x60), seeds 3 / 4 as `-s4` / `-s5`: **KA** (AutoJev-27B on the human-rated rows S), **K**
  (matched own-Lux control, M4 x60 in place), **KH** (HS1 F3 + ½ F1 substituted at matched tokens, block gold only).
- Early stop: each treatment arm's first seed at ⅓ toward Lux vs K-s4's ⅓ point: continue only if ΔP ≥ +0.5 and the
  screen (KA: H3; KH: rule_precedence) is not below the control's.
- Lines KA / KH / K5 (M4 K-s1..s3 + K-s4 + K-s5) at ⅓ ½ ⅔ 1; K2 report-only. Rule `lux9b.m6_rules alpha` vs a fresh re-read
  of K-a13 with the Noul `rule_precedence` floor (≥ R − 4). Finalists ≤ 3 in priority KA, KH, K5.

## Done (node A `/data/dev2/runs/9b/m6/`)

- AutoJev wave: attempt 1 (`aj-wave-r1`, sorted-key prompts) refused at conversion; amendment 1; repeat `aj-wave` OK,
  targets `249b1906…`. KA teacher `3f0aabe0…` (AutoJev on S 30,792, own-Lux 91,859).

- `data/m6-split`: S 30,792 rows / 8.77M tokens; production AutoJev coverage 9,239; wave 21,553 rows
  (prompts `728d3c70…`); manifest `a538817d…`. Guard pre-run PASS (`pre/guard.json`).
- `data/m6-kh`: 121,866 rows / 60,359,187 tokens; train `cb15dca2…`, teacher `1f6d0c7d…`; manifest `c8cabcc0…`.
- Exposure: `exposure/m6-kh.json` `656f5456…`, `exposure/x60.json` `14e7c0ca…`; both `groups: []`.
- HS1 `@171e6f0c` fetched into node A's HF cache (train `c90ef316…`, dev `2e9ee9ab…`).

## Next

1. Chains `m6-gpu6` / `m6-gpu7` (mirror `7380a3cbf`): wave 1 KA-s4 + K-s4 (with `ref-ka13` first on GPU7).
2. Early rules for KA-s4 and KH-s4 (`m6/rules/early-*.json`), then second seeds, soups and lines as the chains decide.
3. Rules (CPU): seed rule, α rule per line, finalists → locks → formal runs → successor rule.

## Launch pattern (chain rule)

`bash $L/upload_chain.sh <node> <local chain> /data/dev2/runs/9b/m6/chains/<file>` (scp + size / SHA-256 check), then in a
separate ssh call `bash $L/launch.sh <chain> <remote file> <size> <sha> <SHA>`; liveness `bash $L/alive.sh <chain>` (PID +
`docker ps` names `d2-9b-m6-*`, `dev2-9b-m6-*`, `d2-9b-m6-aj-*`), never `pgrep -f`.

## GPU-hours

AutoJev waves 0.868 (0.424 + 0.444); total 0.868 of 24.

## Process notes

- A filename search on node A (`find /data/dev2 …`) traversed `/data/dev2/private/` and listed training-data snapshot
  paths under the custodian's sealed tree. No file there was opened. From now on searches exclude `/data/dev2/private`.
