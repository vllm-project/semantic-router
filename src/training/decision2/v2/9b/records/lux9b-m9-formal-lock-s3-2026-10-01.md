# 9B Milestone 9 stage-3 formal post-key run, finalist K-a13IB: lock

Frozen 2026-10-02 ≈00:08 UTC+8 (`lock.sh` 16:07:44Z; record pushed `5d0c1c234`, then `status/formal-s3.GO` 16:09:13Z;
formal started 16:09:28Z), before any formal prediction of a stage-3 artifact (`formal-m9/` held only the C0F parity
run). [Preregistration](lux9b-m9-prereg-2026-10-01.md) (`8570a5896`), amendments
[1](lux9b-m9-prereg-amendment-1-2026-10-01.md) (`0b84e0db4`), [2](lux9b-m9-prereg-amendment-2-2026-10-01.md)
(`51c80ddc9`) and [3](lux9b-m9-prereg-amendment-3-2026-10-01.md) (`787abdc54`, stage 3). **Post-key same-panel**
comparison (the v3 labels were accessed earlier in the project); public 231 is a public-subset reproduction.
Development readouts are never release scores. The newest COORDINATION notes were re-read first (newest 2026-10-02
00:10 UTC+8: IB3 not release-safe, naming / card round 2, M13, 27B M6 and the central custodian C1 content recheck
3c7679b0 for IB1-r3 + IB2 + PN1-r2; no rule change for the 9B formal path).

## Why K-a13IB is the only finalist

The official stage-3 rules (`post-a3.sh` KIBX chain, `m9_rules.py` unchanged, 16:06Z): `select/9b-finalists-s3.json`
`4b02f31f…`, typed readout `lines/readout/m9-s3.json` `d312f8e7…`. Gates vs C0 (DEV2.0-9B's weights re-read on the M9
path):

| Point | typed T | C / N / S | RP | HT-DEV v2 Δ | Y1 PN1 clean gold-no Δ | Y3 `hs1-dev` false-yes | MLX-DEV-9B Noul / Choice Δ | retention Δ | Verdict |
| --- | ---: | --- | ---: | --- | --- | ---: | --- | --- | --- |
| C0 | .9250 | 799 / 338 / 343 | 338 | — | (.262) | .152 | — | (.792) | reference |
| **K-a13IB** | **.9344** | 800 / 344 / 351 | 344 | −.006 [−.017, +.005] TIE | −.015 [−.028, −.003] | .134 | −.001 / −.000 | −.000 [−.009, +.009] | **passes every gate** |
| K-a13IBX | .906 | 800 / 319 / 331 | 319 | −.013 [−.022, −.003] TIE | +.032 [+.019, +.045] | .149 | −.001 / +.002 | −.004 [−.014, +.005] | fails: Noul type floor (319 < 326), `rule_precedence` floor (319 < 334), Y1 |

The scratch preview of 13:57Z (`scratch/s3-preview-rules.json`) agrees with the rules output for K-a13IB; it was
never used as the rules output.

## Candidate

| Field | K-a13IB |
| --- | --- |
| Checkpoint (node A) | `/data/dev2/runs/9b/m9/soup/K-a13IB/build/K-a13IB` (16 files, 30 GB, FP32) |
| `model_sha256` (both development manifests) | `4701ba41c70636b5e215cf1f296bc920a9ee39960d4d28a2338e20b2b81e0d91` |
| Weights | uniform FP32 soup [KIB soup, Lux 1.0, Lux 1.0], i.e. ⅓ toward the KIB soup, as K-a13 = [K soup, Lux 1.0, Lux 1.0]; built on node A 13:35Z |
| KIB soup | uniform FP32 soup of K-a13IB-s1 / -s2 / -s3 (BEST = checkpoints 2,083 / 2,081 / 2,084; seeds 20260926 / 1 / 2), content manifest `c9f945ea…` |
| Lux 1.0 member | the zero-step checkpoint of K-a13IB-s1 (`c105bc4f…`), backbone shards and head byte-identical to K-a13's own base `m3/pf-D-s1-zero/run/checkpoint-0000000` (amendment-3 fallback, re-checked by `post-a3.sh`) |
| Training behind | Lux 1.0 full fine-tuning (K-a13's trainer contract) on `kib` `2cd09292…`: 102,172 x60 rows (stratified whole-group cut to 50.0M tokens) + 48,843 IB1-r3 / IB2 rows (10.17M tokens, gold only); 60.44M tokens; own-Lux KL on x60 rows |
| `SHA256SUMS` (written by `lock.sh`) | `soup/K-a13IB/SHA256SUMS` `49c6d9427613…` (16 files; equal to the build's content manifest) |

## Runtime and comparators

- Formal chain `formal-s3` (pid 214241, mirror `787abdc54` = tree `6bdcdcac9196…`, content manifest
  `f02665f2e974…`): `formal.sh` `8adf3cf5faae…`, `formal-chain.sh` `a9ba7a25b29d…`, `m8/lib.sh` `8bb3dbd195dd…`,
  `run_same_panel.sh` `a16d156082c9…`, adapter `v2/dec/adapter-spec-infer-dec.json` `50fb6744947c…`; image
  `sha256:f83b1d10f14d…`; **16,384 tokens**, over-length inputs invalid, no truncation; node A GPU6; source mount =
  the Lux 1.0 package.
- CAL698 per-type temperatures are fitted first (`formal-m9/K-a13IB-cal`); argmax answers and the Noul side of 0.5 do
  not depend on them, so v3 equals the T = 1 run's. T = 1 ships (13:40 policy).
- Autotune cache `formal-m9/triton-cache`: the one copy of the frozen `formal-m3/triton-cache` (source tree
  `af623300d71a…`, copied 07:54Z), shared with the C0F parity run; at lock time 4,785 files, tree `48a2611aa852…`
  (= its tree after C0F's last collection). `formal.sh` records the tree before and after every collection.
- **Bar:** the released T = 1 run I = `release/dev2-8b-t1-derived` (v3 **67.737**). The formal-path parity run of
  DEV2.0-9B's weights on this path (C0F) has 0 answer differences on typed FINAL / CSS15 / public 231 vs I
  (`formal-m9/C0F.parity.json`), so I is the paired bar.
- Comparators (node A, 16K): native Lux1 `eval m1/d1-lux1-autotune-cache` (65.808); Nimble v2 `eval m2/q6-nimble2`;
  the same-renderer control `formal-m3/lux1-16k-shared` (descriptive); I's mlx answers `formal-m4/K-a13-16k-mlx`.

## Reading (prereg "Formal and successor"; evaluated by `lux9b/m9/items.py`, `f1bb2a50f`)

- **Items 1–7 vs I:** (1) `PAIRED-vs-DEV2.0-9B-T1.json` `ci95.low` > 0; (2) its `axis_ci95.H.delta.high` ≥ 0;
  (3) every `types.json` verdict `OK`; (4) `MLX-PAIRED-vs-DEV2.0-9B-T1.json` card-eligible Choice + Noul `ci95.high`
  ≥ 0; (5) vs adopted Lux1 16K `ci95.low` > 0, `H.delta.high` ≥ 0 vs Lux1 and vs Nimble v2, types `OK`; (6) exposure:
  the stage-3 TRAIN ⊂ x60 ∪ IB1-r3 TRAIN ∪ IB2 TRAIN (`exposure/kib-subset.json` `1dfb1843…`: 0 of 151,015 rows
  outside) and both receipts list 0 groups (x60's; `exposure/ib1-ib2-train.json` `9fd9d3db…`); (7)
  `PUBLIC231-vs-DEV2.0-9B-T1.json` not `REGRESSION`.
- **Item 8** (C1 post-key) only if items 1–7 pass, after the custodian's C1 content recheck for IB1-r3 + IB2 (the
  central recheck 3c7679b0; item 8 only for zero exposure under its rule), from a frozen package and a C1 successor
  spec; one attempt.

## Lock facts (`m9/lock.sh`, mirror `624f94027`, node A, CPU; `select/formal-lock-s3.json` `bc4b428c…`)

```json
{
 "rules": {"path": "select/9b-finalists-s3.json", "sha256": "4b02f31f2783a36b3895ff959644b062f1ee4830d66c82774cf8894264b72237"},
 "readout": {"path": "lines/readout/m9-s3.json", "sha256": "d312f8e7de10d2c8e102cd78850253b6c4a2293bc836dd42fd6aac696ba6b496"},
 "finalists": {
  "K-a13IB": {
   "artifact": "/data/dev2/runs/9b/m9/soup/K-a13IB/build/K-a13IB",
   "sha256sums": "49c6d9427613030640827da992b3c0d2b987ccca960875b4a8a5e25429abdc96",
   "files": 16,
   "model_sha256": "4701ba41c70636b5e215cf1f296bc920a9ee39960d4d28a2338e20b2b81e0d91",
   "built": {
    "point": "K-a13IB",
    "members": ["soup/KIB/build/KIB-soup", "arms/pre/m9-KIB-s1-zero/checkpoint-0000000", "arms/pre/m9-KIB-s1-zero/checkpoint-0000000"],
    "arm_artifact": {"content_manifest": "c9f945ea07de6da5ee91b5d66f96904fbfc6b466e42402c7cd65a20b03e98a75", "pulled_utc": "2026-10-01T13:32:04Z"},
    "lux_member": {"content_manifest": "c105bc4f1c2885eee0ed84d4bb0c59128259de3bc68f8bc54f3f2197499dd14d", "pulled_utc": "2026-10-01T13:33:43Z",
                   "weights_equal_to": "/data/dev2/runs/9b/m3/pf-D-s1-zero/run/checkpoint-0000000"},
    "content_manifest": "49c6d9427613030640827da992b3c0d2b987ccca960875b4a8a5e25429abdc96",
    "built_utc": "2026-10-01T13:35:33Z"
   }
  }
 }
}
```

(Member paths are shortened relative to `/data/dev2/runs/9b/m9/`; the node file holds the full paths.)

After this record is pushed, `status/formal-s3.GO` is written; the waiting chain runs `formal.sh` for K-a13IB on GPU6
(≈ 40 min).
