# 9B M6 state (resume file)

Updated: 2026-09-30 06:05 UTC+8 (M6 worker; **KA stopped by its early rule; KH-s4 on GPU6, K-s5 on GPU7**)
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

## Running

- Chains `m6-g6` (PID 3537189; step log `logs/m6-gpu6.log`) and `m6-g7` (PID 3537271; `logs/m6-gpu7.log`), mirror
  `e611b96b4`, launched 18:50Z. `ref-ka13` = M4's K ⅓ readout exactly (`m6/readout-ref`).
- Wave 1 done: KA-s4 18:50–21:27Z, K-s4 18:54–21:32Z (preflights PASS; BEST = final checkpoint 1,629 for both).
- **Early rule KA: STOP** (`rules/early-KA-s4.json` `9e3cf68d…`; readout `early-KA-s4/readout.json` `55fe7647…`):

  | ⅓ point (first seed) | T | H3 | H | P | C / N / S | rule_precedence |
  | --- | ---: | ---: | ---: | ---: | --- | ---: |
  | KA-s4 (AutoJev on S) | .8988 | .5731 | .5945 | 73.10 | 800 / 272 / 366 | 272 |
  | K-s4 (own-Lux control) | .9231 | .5654 | .5846 | 73.46 | 800 / 346 / 331 | 346 |

  ΔP −0.36 < +0.5. H3 was higher (+.008; discourse +.010, stance +.013). Typed Score +35, with more level-1 answers
  (76 vs 38; gold 107), but Noul `rule_precedence` fell 74 items to Lux's level. That is the same trade as M5's dose.
  No KA second seed or line.
- Wave 2 running: KH-s4 (GPU6, from 21:39Z) and K-s5 (GPU7, from 21:38Z); ETA ~00:20Z. Budget used 6.32 GPU-h at 21:39Z.

## Next

1. Early rules (`m6/rules/early-KA-s4.json`, later `early-KH-s4.json`), written by the chains.
2. After the lines: `m6/rules.sh` (commit `e7546eb2f`, mirror when needed) → finalists → locks → formal (`formal.sh`
   runner `3277dec9d`), hs1-dev diagnostics, `ship_cal.sh`, successor items 1–7, item 8 hand-off.

## Launch pattern (chain rule)

`bash $L/upload_chain.sh <node> <local chain> /data/dev2/runs/9b/m6/chains/<file>` (scp + size / SHA-256 check), then in a
separate ssh call `bash $L/launch.sh <chain> <remote file> <size> <sha> <SHA>`; liveness `bash $L/alive.sh <chain>` (PID +
`docker ps` names `d2-9b-m6-*`, `dev2-9b-m6-*`, `d2-9b-m6-aj-*`), never `pgrep -f`.

## GPU-hours

AutoJev waves 0.868 (0.424 + 0.444); total 0.868 of 24.

## Process notes

- A filename search on node A (`find /data/dev2 …`) traversed `/data/dev2/private/` and listed training-data snapshot
  paths under the custodian's sealed tree. No file there was opened. From now on searches exclude `/data/dev2/private`.
- The first launch of `m6-gpu6` / `m6-gpu7` (18:48Z, mirror `7380a3cbf`) exited at its first step: this worktree has
  `core.fileMode=false`, so the new wrappers were committed 100644 and `job.sh` could not be executed. No GPU job or run
  directory was created. Fixed by `git update-index --chmod=+x`; relaunched from the new mirror as `m6-g6` / `m6-g7`.
