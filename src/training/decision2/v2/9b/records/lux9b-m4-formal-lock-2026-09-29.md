# 9B Milestone 4 formal post-key runs, finalists K-a13, U-a13, KN-a12: lock

Frozen 2026-09-29 (UTC+8) before any formal prediction of a Milestone 4 artifact (`formal-m4` does not exist on node A). **Post-key same-panel**
comparison (the v3 labels were accessed earlier in the project); public 231 is a public-subset reproduction, not the official rank. The finalists
are those of `m4/rules/finalists.json` (node A, 04:16 UTC), confirmed by an independent re-run of `m4_rules.py` from the `3277dec9d` mirror
(`m4/rules/verify-w3/`: seed-K, seed-P, seed-KN and alpha byte-identical to `m4/rules/`, alpha `258835e7c7cc…`; α\*_D = ½ reproduced on
`readout-dline`; 8 / 8 unit tests pass).

## Why these are the finalists (preregistered rules, amendments 3–4)

Development readouts (typed DEV 1,600 + CSS pilot 1,430, CAL698 temperatures) vs the same-runtime Lux 1.0 reference `lux-16k` (proxy 70.82, T .8762,
H3 .5283, Choice / Noul / Score 799 / 272 / 331). Seed rule: the K and KN artifacts are their soups (proxy 67.23 vs seed mean 64.12; 69.37 vs 67.46).
α rule (α\* = smallest eligible α with G ≥ 0.75·G\*): K line G\* .0625 at ½, so **α\* = ⅓** (G .0488); U line G\* .0700 at ½, so **α\* = ⅓** (G .0581);
KN line G\* .0394 at ½, so **α\* = ½**. The D line's α\* = ½ is DW (formally run in Milestone 3, 68.571), so its slot passes on and P fills no slot.
Proxy drop: best pick 76.06 (DW), none ≥ 8 below. Finalists in priority order: K-a13, U-a13, KN-a12.

## Candidates and comparators

| | K-a13 | U-a13 | KN-a12 |
| --- | --- | --- | --- |
| Development (proxy, ΔP vs Lux 95%; T, H3, typed Choice · Noul · Score) | 73.22, +2.40 [+0.71, +5.32]; .9250, .5622, 799 · 338 · 343 | 74.33, +3.51 [+1.76, +6.42]; .9344, .5699, 800 · 341 · 354 | 71.98, +1.16 [−0.74, +4.16]; .9156, .5628, 800 · 294 · 371 |
| Checkpoint (node A `/data/dev2/runs/9b/`) | `m4/K-a13-build/soup` | `m4/U-a13-build/soup` | `m4/KN-a12-build/soup` |
| `model_sha256` (soup driver `console.log`; = CAL698 `checkpoint_sha256`) | `b9d973b3ef555457da2dfa839c45cef3d4a98a47a43d1fd724e77fdec4aa125d` | `8c1ab3719947b098c3afd8f6cd8d61f4cd2ecd84777a32c1d920a037dd2f9396` | `e0dffa0ec943c3f79ac2c13e5f95b79cbf42897ddc565bd4875cb8635ac9674c` |
| Weights (FP32; `members.txt`, `weights.json`) | ⅓ K soup + ⅔ Lux (members 1 : 2); K soup `20dbb8999f21…` = K-s1 / K-s2 / K-s3 SELECT checkpoints 1,624 / 1,420 / 1,424 (seeds 20260926 / 1 / 2) | ⅙ D soup + ⅙ K soup + ⅔ Lux (members 1 : 1 : 4); Milestone 3 D soup `9a1d7db8140d…` | ½ KN soup + ½ Lux (members 1 : 1); KN soup `920dcc726f29…` = KN-s1 / KN-s2 checkpoints 1,238 / 1,409 (seeds 20260926 / 1) |
| Training behind | Lux 1.0 full fine-tuning (backbone 1e-5, head 1e-4, one epoch) on x60 (122,651 rows / 60.2M tokens, XL r2) + 1.0·KL(own Lux ‖ student) on all rows | K as K-a13; D = Milestone 3 arm D (116.1M tokens, own-Lux KL 0.5 on its recipe rows) | as K on xn60 (108,086 rows / 60.1M tokens; no A7q, H1, H8) |
| CAL698 T, Choice / Noul / Score (`m4/NAME-cal/calibration.json` sha256) | 1.420 / 1.037 / .563 (`65297c6d72b8…`) | 1.242 / .939 / .380 (`2ba260a0200f…`) | 1.031 / .811 / **.050** (`f3528fc79568…`) |

- Lux end point `m3/pf-D-s1-zero/run/checkpoint-0000000` (checkpoint-form Lux 1.0, `dc9b795ca75a…`). All three: 7,940,895,744 parameters (backbone
  7,936,684,544 + head 4,211,200, FP32, counted from the safetensors headers; = `formal.sh --loaded-parameters`); 16 files, 31.78 GB, none newer than its build.
- Adapter `v2/dec/adapter-spec-infer-dec.json` (`50fb6744947c…`, `v2.dec.infer_dec`), **16,384 tokens**, over-length inputs invalid, no truncation;
  `infer_dec` refuses a calibration bound to another `model_sha256`. Code: node-A mirror of `3277dec9d` (tree `0f30d20e2df9…`, 2,576 files); `formal.sh`
  `a1ca373d3166…`, `chain-step.sh` `16844687b44a…` and `run_same_panel.sh` `4849283ce01b…` are byte-identical to the local HEAD copies.
- Runtime: the eval runner's default tag `decision20-train-fast:host2` = image `sha256:f83b1d10f14d…` on node A (the image of both comparators and DW);
  `TRITON_CACHE_AUTOTUNING=1` with `formal-m4/triton-cache`, one copy of the frozen `formal-m3/triton-cache` (4,785 files, 104.2 MB, tree sha256
  `af623300d71a…` computed at lock time, not recorded earlier; newest file 2026-09-28 13:38 UTC, during the same-renderer control).
- Comparators (node A, 16,384 tokens): native Lux1 `eval m1/d1-lux1-autotune-cache` (v3 **65.808**, the gating comparator; seal `3ba64e729498…`) and
  the same-renderer control `formal-m3/lux1-16k-shared` (65.231, descriptive; seal `caeafb483702…`).

## Panels, reading and gate

- `formal.sh SHA GPU NAME CHECKPOINT CAL_DIR LABEL`: gold-free smoke (`--max-items 8`), typed FINAL + CSS15 + public 231, then `mlx-diag`, each
  through the eval runner (lease, idle-VRAM check, `GPU-TIME.json`) with the cache tree hash before / after in `NAME-cache.jsonl`; seal before
  scoring; report (revision `m4-NAME-soup`); paired compare vs both comparators; mlx-diag score.
- **Gate (all three):** paired post-key v3 95% lower bound > 0 vs native Lux1 (65.808); human-transfer axis interval not entirely below 0; no type
  collapsed (`python3 -m v2.eval.gates types`).
- Reported: same-renderer paired comparison (vs 65.231), T / H axes, Choice / Noul / Score, typed families, per-task CSS15, public 231 E / S / H,
  typed Brier / ECE, invalid counts, mlx-diag. Disclosed: KN-a12's Score temperature sits on the calibrator's floor (.05, logits ×20; CAL698 Score
  90 / 90 correct), so its typed Score Brier / ECE are expected to be poor; argmax metrics and the gate are unaffected.
- No Hugging Face upload (amendment 1): a passing finalist stays on node A (`m4/NAME-build/soup` + `m4/NAME-cal/`) with a per-file SHA-256 manifest
  written next to it and re-hashed once, and is reported to the coordinator at once with its scored run directory; failing finalists get the same.
- No checkpoint, calibration or limit change after collection; faults are recorded, not retried blindly.
- Budget: 19.3 GPU-h in the node-A Milestone 4 run records (node-B copies in, CPU soups out); three formal runs ≈ 0.55 (DW took 0.18), ≈ 19.9 < 22 cap.

## Launch (node A)

GPU6 / GPU7 leases are `track=9b-clm`, reserved, VRAM 0 at lock time. `chain-step.sh` is committed non-executable, so it runs under `bash` (as in `m4/logs/post-lines1.sh`).
GPU6: one detached driver, K-a13 then KN-a12 (only if K-a13 exits 0). GPU7: U-a13, started only once GPU6's run has created `formal-m4/triton-cache`. Then the type gate per finalist.

```bash
export SHA=3277dec9d81708fa374405ca884043443bab1b49 M4=/data/dev2/runs/9b/m4 F=/data/dev2/runs/9b/formal-m4 D2_9B_GPUS="6 7"
export S=/data/dev2/src/$SHA-src_training_decision2/src/training/decision2 L=/data/dev2/src/$SHA-src_training_decision2/src/training/decision2/v2/9b/lux9b/m4
setsid nohup bash -c 'bash $L/chain-step.sh $SHA 6 formal-gpu6 K-a13 40 -- $L/formal.sh $SHA 6 K-a13 $M4/K-a13-build/soup $M4/K-a13-cal "post-key same-panel" && bash $L/chain-step.sh $SHA 6 formal-gpu6 KN-a12 20 -- $L/formal.sh $SHA 6 KN-a12 $M4/KN-a12-build/soup $M4/KN-a12-cal "post-key same-panel"' > $M4/logs/formal-gpu6.console 2>&1 < /dev/null &  # GPU6 driver
test -d $F/triton-cache && { setsid nohup bash $L/chain-step.sh $SHA 7 formal-gpu7 U-a13 20 -- $L/formal.sh $SHA 7 U-a13 $M4/U-a13-build/soup $M4/U-a13-cal "post-key same-panel" > $M4/logs/formal-gpu7.console 2>&1 < /dev/null & }  # GPU7, once the cache copy exists
for n in K-a13 U-a13 KN-a12; do (cd $S && PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=$S python3 -m v2.eval.gates types --run $F/$n-16k --label $n --output $F/$n-16k.gates/types.json); done  # after all three runs
```

- Outputs: `formal-m4/NAME-smoke`, `NAME-16k` (`SEAL.json`, `REPORT.json`, `PAIRED-vs-Lux1-16K.json`, `PAIRED-vs-Lux1-16K-shared.json`), `NAME-16k-mlx`
  (`mlx-diag.score.json`), `NAME-cache.jsonl`, `NAME-16k.gates/types.json`, `triton-cache.copy.json`; step logs `m4/logs/formal-gpu{6,7}.{log,console}`.
- Post-run: gate = `ci95.low` > 0 and `axis_ci95.H.delta.high` ≥ 0 in `PAIRED-vs-Lux1-16K.json`, and every `types.json` verdict `OK`. Each `output/*.manifest.json`
  must carry the `model_sha256` and calibration `file_sha256` above, `GPU-TIME.json` `image_id` must be `f83b1d10…`, and `triton-cache.copy.json` `source_tree_sha256`
  must be `af623300d71a…`. GPU6 and GPU7 share the cache copy concurrently, so a tree-hash change is reported, not attributed to one run or rerun.
