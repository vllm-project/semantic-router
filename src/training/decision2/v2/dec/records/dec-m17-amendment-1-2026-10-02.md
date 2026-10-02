# Decoder M17 — amendment 1: the user's Index-first release rule (2026-10-02)

Written ≈02:50Z (10:50 UTC+8), while the four M17 seeds are still training: **before any M17 soup, development readout,
MLX-DEV2 read, formal collection of an M17 point or Index row exists** (the only formal job so far is the LH parity
run `m17-4b-LH`, preregistered and arm-independent). It amends [the prereg](dec-m17-prereg-2026-10-02.md)
(`da770d98a`).

## Why

COORDINATION 2026-10-02 09:55 (USER DECISION, 09:38 / 09:55 UTC+8): **the release rule becomes Index-first**; it
supersedes the 02:05 Index path and successor items 1–8 as release blockers. The note postdates the M17 instruction
(09:40); this worker read it after committing the prereg (02:05Z). It says running workers finish the current
measurement and judge releases by the new rule, not items 1–8, and do not run C1 item 8.

## What changes

1. **Release gate (the only quality blocker):** a frozen M17 candidate may be released iff its private Jev Decision
   Index delta vs **the current `Decision-2.0-Nox-4B` release** is significantly positive: paired bootstrap
   (`v2.eval.ix1.paired_boot`, 2,000 replicates, seed 20261002, scoring cases resampled within each benchmark, both
   runs scored on the same cases through the board weights), **95% CI lower bound > 0** on the full panel.
2. **Integrity checks (blocking, not quality comparisons):** the IX1 parity gate (86 / 86) and, at release, exact
   package parity and the Hub / `trust_remote_code` checks; the row-level Index contamination audit of the candidate's
   TRAIN (the IX1 method, planted controls); **no decision type collapsed** (formal item 3 on the candidate's formal
   run).
3. **References (recorded, never blocking):** development gates 1–7 (MLX-DEV2 included), formal v3 (items 1 / 1'(a)),
   human transfer (2), mlx-diag (4), public 231 (7), 6(b) / 6(b)', the old MLX-DEV. **C1 item 8 is not run.** The
   transfer-only Index delta (without HoVer, When2Call, iSarcasmEval, GSM8K and BPoMP) is recorded privately.
4. **Frozen candidates = both arm soups**, each measured **once** on the Index whatever its development gates say.
   Measurement method = the 4B Index-first worker's (`dec/ops/4bif/ix4b.sh`, branch
   `decision-2-training-4b-indexfirst`), so every 4B candidate is comparable: the candidate's **BF16 copy**
   (`v2.release.bf16_copy`, the weights a release ships; its source fingerprint must equal the soup's model SHA-256)
   restaged onto LH's IX1 package `DEV2.0-4B` `13d42143` (identity, loaded count 4,208,383,488 and `calibration: null`
   checked); IX1 DIAGNOSTIC entries `DEV2.0-4B-LHS10SD-bf16` / `DEV2.0-4B-LHS17SD-bf16` (a small separate commit to
   `v2/eval/ix1/launch.sh`); the 86-request parity gate, then panel-8 (8 shards), dual scoring (`score.sh --size 4B`),
   `family_delta` and the paired bootstraps against IX1's `DEV2.0-4B-LH` run.
5. **Where:** BF16 copy and restage on node F (CPU container of image `decision20-train-fast:host2`); inference on
   node E GPU0–3 and node F GPU2, 3, 6, 7 when idle (the harness's `track=eval-ix1` lease form, purpose naming
   dec-m17; panel-8, kit `87d4650b` and the restaged packages copied through node A, digests checked); scoring and the
   bootstraps on node C (CPU), where the LH run and the suite are. Results stay private (node private dirs and
   `/home/xunliu/code/decision2-program/private/m17/`); the repository records pass / fail, row counts and the parity
   verdict only.
6. **Formal:** both arms are collected on the M17 formal path (as references and for item 3), after the parity run.
   The finalist rule of the prereg no longer limits which arm gets a formal run (≤ 2 points either way).
7. **Anchors:** dropped, except when an arm passes the Index gate and fails only the type-collapse check: then its
   α = ⅓ / ⅔ anchors become frozen candidates (built and read as preregistered, each with one Index run).
8. **Selection and release:** among M17's qualifiers, the largest Index-gain lower bound. Across the tier (the 09:55
   rule) the candidate with the largest lower bound among the 4B candidates on record wins; M17 releases its best
   qualifier to `Decision-2.0-Nox-4B` (then-current `main`, the default card generator, never concurrent with another
   publisher, `main`-unchanged check) only if it has that largest lower bound and no other 4B release is in
   progress; otherwise the verdict goes to the coordinator. If Nox-4B changes before M17's verdict, the bootstrap is
   redone against the new release's IX1 run (CPU only).
9. Budget unchanged (cap 50 GPU-h; ≈ 2.4 GPU-h per Index run). Stop rules unchanged: a failed parity gate or shard
   ends that candidate's Index run (recorded, never rerun).
