# 9B M6 state (resume file)

Updated: 2026-09-30 09:40 UTC+8 (continuation worker; **M6 closed: K5-a12 +1.62 [−0.19, +2.41], fails items 1 and 4; no successor**)
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

## Run log

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
- Wave 2 done: KH-s4 21:39–00:18Z (preflights PASS; BEST 1,607), K-s5 21:38–00:11Z (BEST 1,621).
- **Early rule KH: STOP** (`rules/early-KH-s4.json` `488d2b17…`; readout `early-KH-s4/readout.json` `eee0967a…`):

  | ⅓ point (first seed) | T | H3 | H | P | C / N / S | rule_precedence |
  | --- | ---: | ---: | ---: | ---: | --- | ---: |
  | KH-s4 (HS1 F3 + ½ F1) | .9250 | .5432 | .5765 | 73.03 | 800 / 327 / 353 | 327 |
  | K-s4 (control) | .9231 | .5654 | .5846 | 73.46 | 800 / 346 / 331 | 346 |

  ΔP −0.43 < +0.5 and rule_precedence 327 < 346. Score +22, Noul −19, H3 −.022. No KH second seed or line.
  Chain `m6-g6` ended ("chain m6-gpu6 done"); GPU6 idle from 00:26Z.
- `m6-g7`: K5 soup 00:11–00:22Z; K5 line ⅓ / ½ / ⅔, then K2 soup and line (⅓, ½); then it ends (KH stopped).
  GPU-h 11.62 at 00:24Z.
- Continuation commit `e2daf2afc` (mirrored, tree `ddc33f99…`): `chains/m6-htdev2.sh` (HT-DEV v2 diagnostic, runtime
  `bc0a12d70` = the 9B reference's collection code) and `chains/m6-post.sh` (m6-formal.sh, then m6-htdev2.sh).

- `m6-g7` ended 00:58Z ("chain m6-gpu7 done"). **Rules** (`m6/rules.sh e2daf2afc readout-lines`, 01:05Z):
  seed K5 = soup (69.84 ≥ 65.51); K5 α\* = ½ (G\* +.030; ⅓ +.010 < .0225); **finalist K5-a12** (`finalists.json`
  `c500e7f4…`); K2 report-only pick ½. Lock record `lux9b-m6-formal-lock-2026-09-30.md` (`e352dae8f`).
- Chain `m6-post-K5-a12` (mirror `e2daf2afc`, GPU6) ran 01:08–01:30Z, every step exit 0: formal, `hs1-dev`
  (K5-a12 + incumbent), `ship_cal` (**T = 1**: CAL698 worsened CSS-pilot Brier / ECE), T = 1 derivation, HT-DEV v2.
- **Formal:** K5-a12 v3 **69.362** vs released T = 1 67.737: **+1.62 [−0.19, +2.41]**; T +.030 [+.018, +.042], H +.006
  [−.022, +.018]. Items: 1 **FAIL**, 2 pass, 3 pass, 4 **FAIL** (card-eligible mlx −.010 [−.020, −.001]), 5 pass
  (vs Lux1 +3.55 [+1.35, +5.64]), 6 pass, 7 pass. **No successor**; item 8 not reached. HT-DEV v2 TIE (−.007).
  `hs1-dev` unchanged. Record `lux9b-m6-result-2026-09-30.md`.
- Leases GPU6–7 `track=9b-m6 status=idle` (01:33Z). No M6 process or container is running.

## Next

M6 is closed: DEV2.0-9B stands. Nothing left to run. (If relaunched: only integration merges or record follow-ups.)

## Launch pattern (chain rule)

`bash $L/upload_chain.sh <node> <local chain> /data/dev2/runs/9b/m6/chains/<file>` (scp + size / SHA-256 check), then in a
separate ssh call `bash $L/launch.sh <chain> <remote file> <size> <sha> <SHA>`; liveness `bash $L/alive.sh <chain>` (PID +
`docker ps` names `d2-9b-m6-*`, `dev2-9b-m6-*`, `d2-9b-m6-aj-*`), never `pgrep -f`.

## GPU-hours

AutoJev waves 0.868 (0.424 + 0.444); job receipts in total 12.084 (`m6_gpu_hours`), plus the formal collections 0.218
(`GPU-TIME.json`): **≈ 12.30 of 24**.

## Process notes

- A filename search on node A (`find /data/dev2 …`) traversed `/data/dev2/private/` and listed training-data snapshot
  paths under the custodian's sealed tree. No file there was opened. From now on searches exclude `/data/dev2/private`.
- The first launch of `m6-gpu6` / `m6-gpu7` (18:48Z, mirror `7380a3cbf`) exited at its first step: this worktree has
  `core.fileMode=false`, so the new wrappers were committed 100644 and `job.sh` could not be executed. No GPU job or run
  directory was created. Fixed by `git update-index --chmod=+x`; relaunched from the new mirror as `m6-g6` / `m6-g7`.
