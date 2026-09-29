# Multilingual diagnostic mlx-diag: the three ~27B open peers on node B (2026-09-29)

Eval & peers track, Milestone 5 (coordinator 14:25 UTC+8: collect mlx-diag for the three 27B peers on node B GPU0–2,
lent by the 9B track). **Development diagnostic, not a release score.** Panel `mlx-diag` v1: 2,275 prompts, prompts
`25fde28f…`, gold `71515a41…` (panel and scorer: `mlx-diag-v1-2026-09-28.md`). Post-key v3 for context: AutoJev-27B
72.133, Eikos-27B 69.290, Jebadiah-27B 65.472.

## How it ran

- **Scored runtime.** Each peer ran exactly as its stored post-key node-B comparator run (`m4/nodeB-kernel/<peer>`),
  with `--panels mlx-diag` and a fresh copy of that run's persisted Triton cache as the only differences (AutoJev
  also moves to the `e9cd637b1` mirror; see Code):
  - image `decision20-train-fast:latest` `sha256:dbe5f32b…`; every receipt records Python 3.12.13, torch
    2.12.0+git6bbd260 (HIP 7.2.53211), Triton 3.7.1, Transformers 5.17.0, FLA 0.5.2, `causal_conv1d` 1.7.0;
  - `--env HIP_FORCE_DEV_KERNARG=1`, `TRITON_CACHE_AUTOTUNING=1`, `TRITON_CACHE_DIR=<run-dir>/triton-cache` (rw mount);
  - adapters, model paths, revisions, mounts and extras as the stored runs:

| Peer | GPU | Adapter (module sha256) | Model (node B) @ revision | Extras / mounts | Pinned trees (pre-check) |
| --- | ---: | --- | --- | --- | --- |
| AutoJev-27B | 0 | `autojev27` (`20756761…`) | `/data/dev2/models/autojev-27b@6f5b557e` @ `6f5b557e` | `model_id=denis-pplx/autojev-27b`, `source=/data/dev2/tools/src/autojev-source`; `--mount /data/dev2/tools/src/autojev-source` | weights `2c5f11e7…` (58 files), source `f7cb68c3…` (34) |
| Eikos-27B (BF16) | 1 | `eikos-27b` (`8dfd7923…`) | `/data/dev2/models/Eikos-27B@103a5647` @ `103a5647` | `model_id=caiovicentino1/Eikos-27B` | weights `db7dd301…` (48 files) |
| Jebadiah-27B | 2 | `jebadiah-27b` (`1b358b50…`) | `/data/dev2/models/jebadiah-27b@c68db2b5` @ `c68db2b5` | `model_id=frontier-infra/jebadiah-27b` | weights `081bd732…` (67 files) |

- **Code.** Exact mirror `e9cd637b1` (subtree) for all three peers; `mirror_to_node.sh --verify` passed (content
  manifest `b412f2f9…`, 1,892 files). AutoJev's own stored run used `33d3aed76` (before the named-lease fix); the
  adapter module sha256 in `e9cd637b1` equals each stored run's `adapter_module_sha256` for all three peers.
- **Prompts.** Gold-free `mlx-diag.prompts.jsonl` sha256 `25fde28f…` is identical on node B and node A and equals the
  registry `prompts_sha256` in `v2/eval/panels.py` (the collector re-checks it at start). No gold was read on node B.
- **CPU pre-checks on node B (all match).** Weight/source trees with `v2.eval.sealed.event3 digest` (mirror `31d0c27e`),
  table above. Frozen post-run caches with `v2/27b/triton_cache.py digest`: AutoJev `12dfb608…` (1,420 files, 11
  autotune entries), Eikos `92fcc0d3…` (2,290, 17), Jebadiah `27437a86…` (649, 8).
- **Caches.** Each of the six runs took its own `cp -a` copy of the peer's persisted cache into `<run-dir>/triton-cache`
  (`triton_cache.py copy`, which refuses a copy whose digest differs from the pinned one; all six matched). After each
  run `triton_cache.py finish`: **no autotune entry added, changed or removed, and no compiled kernel added or changed**.
  The post-run digest differs from the frozen one only because Triton rewrote the absolute child paths of 6 group
  manifests (frozen check passed in all six). The frozen caches were re-hashed after the runs: unchanged.
- **GPUs and leases.** Node B GPU0 AutoJev, GPU1 Eikos, GPU2 Jebadiah, one job at a time per GPU, all three in
  parallel. The 9B track owns them (idle since M4) and lent them to eval at 14:25 UTC+8 as shared leases. The runner
  ran with `--lease-name owner.eval --shared`: it pins the GPU with `ROCR_VISIBLE_DEVICES=<N>` (as in the stored
  runs), so `HIP_VISIBLE_DEVICES` is unset. The `owner` files were not touched (content and mtime 04:09:51Z unchanged).
  Idle check at 07:03:02Z: 0% use, 0% VRAM, no KFD processes, no containers. The chain re-checked VRAM%/use = 0
  before each job, and every `GPU-TIME.json` records `shared: true`, `vram_pct_at_start: 0`. Entries
  `gpu{0,1,2}.lock/owner.eval` held track=eval, purpose, start, expected end 07:45Z and the note "shared lease lent
  by 9b-clm". They were marked `status=released` with `released_utc` 07:07:38Z (GPU0), 07:08:25Z (GPU1) and
  07:07:08Z (GPU2).
- **Chain rule.** One `nohup` chain per GPU, `ops/chain.sh <peer>` (7,118 bytes, sha256 `f1c9e2a2…`): fresh cache
  copy, 20-item smoke, gold-free check, fresh cache copy, full run, gold-free check, release. At launch
  (07:03:12Z): the script was non-empty, three chain processes were running (`pgrep`), and each chain log had
  its first line. The gold-free checker `ops/check_preds.py` (`358c76f7…`) applies the frozen scorer's
  answer-validity rules with a placeholder target (validity depends only on question and answer). It also checks
  the receipt (exit code, rows, image, adapter module, mirror) and each row's input digest. It passed a
  self-test on synthetic rows before launch.
- **Smokes (first 20 prompts: 7 Choice, 4 Noul, 9 Score).** 20/20 rows for each peer, 0 invalid, 0 input-digest
  mismatches. Answer fields: Choice gives an offered label plus the full 18-label probability map; Noul gives a
  probability in [0, 1]; Score gives a value inside the 3-level range plus level probabilities. AutoJev and
  Jebadiah use the same field set; Eikos adds its native projection fields.
- **Scoring.** Predictions and receipts were streamed node B → workstation → node A as `tar | xz -T0 -6`; all 28
  files are sha256-equal on both nodes. Scoring ran on node A with mirror `e32fc4d56`
  (`python3 -m v2.eval.multilingual_panel score --panel /data/dev2/private/panels/mlx-diag-v1`); gold `71515a41…`
  equals the registry. The same scorer re-scores the stored Lux 1.0 predictions to a byte-identical score file
  (`d39a4e68…`).

## Results (2,275 prompts per model, zero invalid or missing answers for every model)

| Model | Type-macro acc. | English | Non-English | Non-EN Choice / Noul / Score | Weakest language | Cross-language agreement |
| --- | ---: | ---: | ---: | --- | --- | ---: |
| AutoJev-27B | 84.2 | 89.2 | 83.4 | 80.6 / 88.7 / 81.0 | ar 75.6 | 0.68 |
| Jebadiah-27B | 83.2 | 86.8 | 82.6 | 77.6 / 85.3 / 84.7 | ar 77.1 | 0.69 |
| Eikos-27B (BF16) | 81.9 | 86.6 | 81.1 | 78.4 / 85.0 / 80.0 | ar 73.1 | 0.62 |
| *Decision 1.0 Lux 9B (v1 record, reference)* | 82.8 | 87.8 | 81.9 | 78.3 / 83.3 / 84.2 | ko 77.0 | 0.65 |

Accuracies in %; chance is 5.6% (Choice), 50% (Noul), 33% (Score). Columns as in the v1 record. The weakest
language is the lowest per-language mean over the types that language appears in. Cross-language agreement is
the share of source items answered identically in all seven languages. Re-deriving Lux 1.0's row from its stored
score reproduces the record exactly.

Findings:

- **Order.** mlx-diag gives AutoJev > Jebadiah > Eikos, while post-key v3 gives AutoJev > Eikos > Jebadiah.
  Jebadiah passes Eikos on Score (non-English 84.7 vs 80.0); their Noul is level (85.3 vs 85.0).
- **Arabic is the weakest language for all three.** Arabic Score (XNLI) is 72.7 / 78.8 / 70.7 and Arabic Choice (MASSIVE)
  78.6 / 75.4 / 75.4 (AutoJev / Jebadiah / Eikos). Next weakest: Russian Score 78.8 for AutoJev and Eikos,
  Korean Noul 80.0 for Jebadiah. Japanese Noul is low for all three (85.0 / 83.0 / 82.0).
- **Loss off English.** Non-English minus English (unrounded accuracies), Choice / Noul / Score: AutoJev-27B -1.2 / -5.3 / -10.9; Jebadiah-27B -1.7 / -2.7 / -8.2; Eikos-27B -2.5 / -4.0 / -9.9.
  Score (XNLI) loses the most, and Choice is nearly language-independent.

## Per-language accuracy (%)

**Choice (MASSIVE, 126 items per language)**

| Model | en | ar | de | es | fr | ja | zh | Non-EN mean | All |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AutoJev-27B | 81.7 | 78.6 | 81.7 | 79.4 | 81.0 | 82.5 | 80.2 | 80.6 | 80.7 |
| Jebadiah-27B | 79.4 | 75.4 | 76.2 | 77.0 | 77.0 | 81.0 | 79.4 | 77.6 | 77.9 |
| Eikos-27B (BF16) | 81.0 | 75.4 | 78.6 | 75.4 | 80.2 | 82.5 | 78.6 | 78.4 | 78.8 |

**Noul (PAWS-X, 100 per language)**

| Model | en | de | es | fr | ja | ko | zh | Non-EN mean | All |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AutoJev-27B | 94.0 | 89.0 | 92.0 | 90.0 | 85.0 | 85.0 | 91.0 | 88.7 | 89.4 |
| Jebadiah-27B | 88.0 | 87.0 | 86.0 | 88.0 | 83.0 | 80.0 | 88.0 | 85.3 | 85.7 |
| Eikos-27B (BF16) | 89.0 | 84.0 | 88.0 | 83.0 | 82.0 | 84.0 | 89.0 | 85.0 | 85.6 |

**Score (XNLI, 99 per language)**

| Model | en | ar | de | es | fr | ru | zh | Non-EN mean | All |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AutoJev-27B | 91.9 | 72.7 | 84.8 | 85.9 | 85.9 | 78.8 | 77.8 | 81.0 | 82.5 |
| Jebadiah-27B | 92.9 | 78.8 | 85.9 | 90.9 | 87.9 | 82.8 | 81.8 | 84.7 | 85.9 |
| Eikos-27B (BF16) | 89.9 | 70.7 | 83.8 | 84.8 | 85.9 | 78.8 | 75.8 | 80.0 | 81.4 |

**Overall per language** (scorer `per_language_mean_accuracy`: mean over the types present in that language)

| Model | en | ar | de | es | fr | ja | ko | ru | zh |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| AutoJev-27B | 89.2 | 75.6 | 85.2 | 85.7 | 85.6 | 83.8 | 85.0 | 78.8 | 83.0 |
| Jebadiah-27B | 86.8 | 77.1 | 83.0 | 84.6 | 84.3 | 82.0 | 80.0 | 82.8 | 83.1 |
| Eikos-27B (BF16) | 86.6 | 73.1 | 82.1 | 82.7 | 83.0 | 82.3 | 84.0 | 78.8 | 81.1 |

Brier (per type, Choice / Noul / Score): AutoJev-27B 0.151 / 0.079 / 0.123; Jebadiah-27B 0.160 / 0.110 / 0.116; Eikos-27B 0.159 / 0.107 / 0.133.

## Invalid answers

0 invalid or missing of 2,275 for every peer (scorer), and 0 in every language cell. The node-B gold-free check
found 0 invalid, 0 missing, 0 duplicate and 0 input-digest mismatches in every run. Jebadiah's native log reports 0
native rejections and 0 over-budget prompts under its 2,048-token render.

## GPU time (runner wall-clock × 1 GPU, smokes included)

| Peer | GPU | Smoke (s) | Full (s) | Full collector (s) | GPU-h |
| --- | ---: | ---: | ---: | ---: | ---: |
| AutoJev-27B | 0 | 61.4 | 197.8 | 191.1 | 0.0720 |
| Jebadiah-27B | 2 | 43.1 | 187.1 | 179.4 | 0.0640 |
| Eikos-27B (BF16) | 1 | 72.9 | 233.7 | 227.2 | 0.0852 |
| **Total** | | 177.4 | 618.7 | | **0.221** |

Total 796.1 GPU-seconds = **0.221 GPU-h** on node B, from 07:03:12Z to 07:08:25Z. Counting the
lease span instead (chain start to release, including the cache copies and checks between jobs) gives 815 GPU-s
= 0.226 GPU-h. CPU only: pre-checks took about 5 s on node B, and scoring took seconds on node A.

## Receipts

| Peer | Predictions sha256 | COLLECT.json sha256 | GPU-TIME.json sha256 | Score JSON sha256 |
| --- | --- | --- | --- | --- |
| AutoJev-27B | `79b02b2801f237f33c74ae4b1b7a5162035802323496b7da280f39d0467f1f42` | `5381be0e781b66210aa5ca41a20d25efa523cc0b4a8b6fa7362659bd857fb566` | `6dcb4b312048af24794beab1d0252d02ff5230f4c00658ec27d1e80977918664` | `d64093e211d98adb90066fddca9171bd260f76a380234a08cbc64ae8fe388e0f` |
| Jebadiah-27B | `a0b58d4ff616011cac8686439b7c20a7a60ab466be530c43abf4b32201108c90` | `e15169e829910eddc0cfd4ab254ba8e30f8cf828ff436113424663424de13651` | `6e90e741d7a3550cc861a3059e75aebd727b9ef669aaebe034d768c78a383287` | `7af59baf81b2d70385c1bdc9b80f36753e8256f1437bb3bc60f9bfb83f77df03` |
| Eikos-27B (BF16) | `c38d8c5053b395a4b24a5957f4706984b3e026f5397a44d572f890f800365cdd` | `693be45c34ead05ff8e84e6e37a1f1b2221b879eaf124c943b33962d56d966e8` | `30ff65949f3d3339083fd5bfb9a1c5fdfd2245228b00b3f4d553750688ff003f` | `ccfcdc3c386834bd702e782ceb8feaa5065f755163e019d19977311d0ab93712` |

| Peer | Smoke predictions sha256 | SMOKE.json sha256 | Cache pinned = copy (each run) | Post-run cache: smoke / full | Autotune entries before → after |
| --- | --- | --- | --- | --- | --- |
| AutoJev-27B | `77e6a56b…` | `0574bfc2…` | `12dfb6088ddd34eb4dd95313e6e99d9e94c085176b57670fa257d71ccfdb36c9` | `226e60a1…` / `7769efb2…` | 11 → 11 (smoke 11 → 11) |
| Jebadiah-27B | `c98d848a…` | `53871e0e…` | `27437a865e5418dba2c30284ebfe64ed28e9b0e070a023dffe3b2dfe6cf684b8` | `dc3af478…` / `40c29ef9…` | 8 → 8 (smoke 8 → 8) |
| Eikos-27B (BF16) | `c12189ae…` | `687c328d…` | `92fcc0d3ee145b40c767b15fbd40e6367b45d092cf4d35b0f636c241069c7d35` | `2ff4744f…` / `9dcebb4e…` | 17 → 17 (smoke 17 → 17) |

Scorer gold `71515a41583e7c4792c7058d45e6b12980f033bc9de2847b15dd3fb7b940b484`; each score file's `predictions_sha256`
equals the predictions file hash above and the collector's `output_sha256`.

## Run directories

- Node B `/data/dev2/runs/eval/m5/mlx-diag-27b/`: `<peer>-smoke/` and `<peer>/` for `autojev27`, `eikos27b` and
  `jebadiah27b`. Each holds the receipts, predictions, logs and its own `triton-cache/` copy with
  `triton-cache.{copy,post}.json`. Next to them sit `<run>.log` (runner), `<run>.ops.log`, `<run>.check.json`
  and `ops/` (`chain.sh`, `check_preds.py`, `precheck.sh`, `selftest.py`, `OPERATIONS.log`, `*.chain.log`,
  `precheck.log`).
- Node A `/data/dev2/runs/eval/m5/mlx-diag-27b/`: `<peer>/` holds `output/mlx-diag.predictions.jsonl`,
  `COLLECT.json`, `GPU-TIME.json`, `logs/mlx-diag.log`, the cache receipts and `mlx-diag.score.json`. `<peer>-smoke/`
  holds `SMOKE.json`, `GPU-TIME.json` and the cache receipts. Also `node-b-checks/` and `scorer-check/lux1.rescore.json`.

## Deviations and notes

- AutoJev ran on mirror `e9cd637b1`, not its stored run's `33d3aed76`, as instructed. The adapter module is
  byte-identical.
- Cache start state differs from the stored runs by design. The stored runs filled an empty per-run cache during
  typed FINAL, CSS15 and public 231 (plus the dev panels). These runs start from that persisted post-run cache, and
  mlx-diag needed no new autotuning. The `TRITON_CACHE_DIR` path in `runtime_env` therefore differs from the stored
  runs.
- The output directories hold only predictions: these adapters write no separate runtime or manifest JSON. Runtime
  identity is in `COLLECT.json` (runtime versions, adapter, mirror) and in each collector log's end line. AutoJev's
  log gives `model_config_sha256` `bacbcbb2…` and 26,086,635,760 loaded parameters, Eikos's gives `be507a9f…`,
  and Jebadiah's gives the sha256 of its bundled runtime files.
- No F1/F2 run was made or scored here; the ~27B track runs those on node B GPU5–7.
