# HF storage for the ~27B successor: BF16 storage revisions of DEV2.0-0.8B / 2B / 4B, stale staging removed, F-b upload STOPPED (2026-09-30)

Coordinator note 2026-09-30 06:10 UTC+8, F-b release path step 1 (release worker, worktree
`vllm-sr-dev2-release-27b`). Times are UTC unless marked. GB = 10^9 bytes, as the Hub counts. Node-side receipts
are copied under [`dev2-bf16-storage-2026-09-30/`](dev2-bf16-storage-2026-09-30/).

## Result

- **Private storage: 61.43 → 50.46 GB of 100 GB** (headroom 38.57 → 49.54 GB). Nothing that the rules protect was
  touched: no training data, panels, eval-artifacts backups, datasets, or repositories outside Decision 2.0.
- **Three storage-only revisions, all private, all in the collection "🎲 Decision 2.0" (order unchanged):**

  | Model | New `main` | Final decision | Replaces | Backbone bytes | Parity (0 answer changes each) |
  | --- | --- | --- | --- | --- | --- |
  | DEV2.0-0.8B | `bede7938a8c209c09f27400b79eed57948d6b75e` | `a34b2486…` | `d4812ac6`, `a97e063f…` | 3.010 → 2.014 GB | typed 1,600, CSS15 6,547, public 231, mlx-diag 2,275; drift ≤ 8.9e-16 |
  | DEV2.0-2B | `a53cf66a0d9d492a84b6617b61e7ce35fcd03af0` | `86129728…` | `a47bdf89`, `743c5b2e…` | 7.527 → 4.782 GB | the same four panels; drift ≤ 8.9e-16 |
  | DEV2.0-4B | `fadbba4ff671b4948fb7530fa6748f522b1ac9e4` | `d6bcdea2…` | `197b70ca`, `4d31cdab…` | 16.823 → 9.685 GB | typed, CSS15, public 231 (formal cache); mlx-diag 2,275 in a no-upload run with the mlx run's cache; drift ≤ 1.9e-14 |

  Parity ran against the T = 1 bindings of each release, before the upload and again on the real download. The drift
  equals the drift of the original FP32 releases, as expected: the copy stores the Linear weights already rounded the
  way the runtime's BF16 autocast rounds them before every matmul.
- **The superseded FP32 backbone blobs are purged** with `rewrite_history=False` (27.360 GB). Commits and refs are
  unchanged, exactly those objects are gone, old weight paths return 403 and every LFS file of `main` is served.
- **Stale staging leftovers are removed** (83.4 MB in six staging repositories, `rewrite_history=False`).
- **F-b does not fit. The upload is STOPPED before the transfer, item 8 and the release.** The exact-parity BF16
  package of F-b is **53.82 GB** of new LFS objects. With 49.54 GB free, it would take usage to **104.28 GB**, over
  the 100 GB cap by 4.28 GB. The ≥ 3 GB headroom rule needs 56.82 GB free, so **7.28 GB is missing**
  (`hf_headroom.sh --min-free-gb 56.82`: exit 1). The Decision 2.0 program cannot free that much itself; see §5.
- **GPU: 0.712 GPU-hours** on node A GPU0 / GPU1 (shared lease `owner.release-bf16`, released). The copies, previews,
  purges and storage work were CPU only.

## 1. Inventory (Hub `usedStorage`, private repositories)

[Before](dev2-bf16-storage-2026-09-30/storage/headroom-before-20260929T221352Z.json) (22:13Z) and
[after](dev2-bf16-storage-2026-09-30/storage/headroom-after-20260929T230014Z.json) (23:00Z), 38 private repositories.

| Repository | Owner | Before | After |
| --- | --- | ---: | ---: |
| DEV2.0-9B | Decision 2.0, released (already BF16) | 17.948 | 17.948 |
| DEV2.0-4B | Decision 2.0, released | 16.854 | 9.716 |
| DEV2.0-2B | Decision 2.0, released | 7.556 | 4.811 |
| DEV2.0-0.8B | Decision 2.0, released | 3.034 | 2.039 |
| DEV2.0-0.6B | Decision 2.0, released (already BF16) | 1.520 | 1.520 |
| DEV2.0-27B | Decision 2.0, released (adapter) | 0.509 | 0.509 |
| `decision-2.0-training-data` | Decision 2.0 dataset (protected) | 3.384 | 3.384 |
| `decision-2.0-eval-artifacts` | Decision 2.0 dataset (protected) | 0.143 | 0.143 |
| 7 `dev2-*staging*` repositories | Decision 2.0 staging | 0.083 | 0.000 |
| Vela-2.0-Encoder-307M-Unified | outside Decision 2.0 | 3.133 | 3.133 (+109 kB by its owner meanwhile) |
| Decision-1.0-Route-0.6B | outside Decision 2.0 | 2.322 | 2.322 |
| 13 Vela datasets | outside Decision 2.0 | 4.564 | 4.564 |
| `semantic-router-cache-triplets`, 5 cache LoRAs, 2 small datasets | outside Decision 2.0 | 0.374 | 0.374 |
| **Total** | | **61.426** | **50.464** |

## 2. Steps and storage after each

| Step | Storage after | Change |
| --- | ---: | ---: |
| start (22:13Z) | 61.426 | |
| stale staging leftovers removed (22:37Z) | 61.342 | −0.083 |
| DEV2.0-2B BF16 revision + FP32 purge | 58.597 | −2.745 (peak +4.782 before the purge) |
| DEV2.0-4B BF16 revision + FP32 purge | 51.459 | −7.138 (peak +9.685) |
| DEV2.0-0.8B BF16 revision + FP32 purge | **50.464** | −0.995 (peak +2.014) |

The three uploads ran before the purges, so the peak was 77.8 GB. The per-step values come from each repository's
own `usedStorage` in the two inventories; with the owner's +109 kB on Vela-2.0 they add up to the measured 50.464.

## 3. BF16 storage revisions

- **Copies** ([`ops/bf16-copy.sh`](dev2-bf16-storage-2026-09-30/ops/bf16-copy.sh), node A, scored image `f83b1d10`,
  CPU, no network; `test_bf16_copy` 8 / 8 first). Each source checkpoint's fingerprint files were first checked
  against the released `MODEL_MANIFEST.json`.

  | Tier | Receipt | Identity (scored → copy) | BF16 / FP32 tensors |
  | --- | --- | --- | --- |
  | 0.8B | `995cdadb…` | `60356482` → `3f02f0e5` | 186 / 134 |
  | 2B | `a5229ef1…` | `073bd1f2` → `32872f29` | 186 / 134 |
  | 4B | `66fe6c02…` | `11b5ca1c` → `8b0a8bc7` | 248 / 178 |

- **Specs** (`specs/dev2-{0p8b,2b,4b}-bf16.json`, made by [`ops/make_bf16.py`](dev2-bf16-storage-2026-09-30/ops/make_bf16.py)):
  each is the size's card-c1 spec with the BF16 checkpoint, its receipt pinned by SHA-256, the copy's identity, the
  new decision and one storage sentence in `runtime_equivalence` (the 9B wording). The package runtime and vendored
  sources stay pinned to the mirror that built the replaced revision (`33de83cea` for 0.8B / 2B, `f8f52c695` for 4B).
  CPU preview builds gave READMEs byte-identical to the released cards.
- **Decisions** (`DEV2.0-{0.8B,2B,4B}.decision.json`, status final, decided by the coordinator under the 06:10 note
  and the 16:05 BF16 directive): the superseded judgement stands unchanged (report, paired comparison, T = 1,
  licence, disclosures, C1 line). Only the identity changes. The C1 baselines of these weights stay valid, because
  the copy computes the same BF16 matmuls.
- **Runs** ([`ops/bf16.sh`](dev2-bf16-storage-2026-09-30/ops/bf16.sh), `release.sh --upload --collect
  --already-collected`, mirror `78592cbb4`): all 17 steps passed for each tier. Examples were bit-identical across
  processes and after the download, the card example ran, and the re-hash covered 30 / 33 / 36 files. Loaded
  parameters are unchanged (753,446,208 / 1,883,930,944 / 4,208,383,488). Readback: private, no card problems. The
  gate sealed 6 / 6 each. Card HTTP 12 / 12, 12 / 12 and 13 / 13 (anonymous 401). Links 14 / 14, 15 / 15 and 17 / 17.
  - The file diff against the replaced revision ([`ops/bf16_diff.py`](dev2-bf16-storage-2026-09-30/ops/bf16_diff.py))
    shows only the converted backbone files (equal to the receipt) and `MODEL_MANIFEST.json`.
  - The collection order is unchanged.

  | Work dir (node A `/data/dev2/runs/release/`) | GPU | Wall | GPU-h |
  | --- | --- | ---: | ---: |
  | `dev2-bf16-4B-mlx-20260929T223257Z` (no upload) | 0 | 247 s | 0.069 |
  | `dev2-bf16-4B-20260929T223705Z` | 0 | 887 s | 0.246 |
  | `dev2-bf16-2B-20260929T223257Z` | 1 | 748 s | 0.208 |
  | `dev2-bf16-0.8B-20260929T224530Z` | 1 | 680 s | 0.189 |

- **Purges** ([`ops/purge_fp32.py`](dev2-bf16-storage-2026-09-30/ops/purge_fp32.py), generalised from the 0.6B M8
  tool). Each plan checked that `main` is the verified BF16 revision, that no branch or tag references a target, and
  that the node-A FP32 checkpoint re-hashes to every target. Targets were 7.527 GB (2B, 2 shards), 16.823 GB (4B, 5)
  and 3.010 GB (0.8B, 1). Every apply check passed.
  - The FP32 weights of older revisions are no longer downloadable from the Hub. Records that cite them rest on the
    per-file SHA-256 and the node-A checkpoints (`dec/m{2,3,4}/formal-candidates/staging-*`).
  - The node-A BF16 copies (`/data/dev2/runs/release/inputs/dev2-{0p8b,2b,4b}-bf16`) are the durable stores of the new
    weights.

## 4. Stale staging leftovers

[`ops/staging_leftovers.py`](dev2-bf16-storage-2026-09-30/ops/staging_leftovers.py) (plan, then apply;
`rewrite_history=False`).

- **Removed:** the last LFS object of each staging repository.
  - The shared Qwen tokenizer `06b95093…` in `dev2-dec-staging`, `dev2-9b-staging` and `dev2-27b-staging`.
  - The 0.6B banner in `dev2-release-staging`.
  - The 0.6B tokenizers in `dev2-staging-06bm6-mxcx` and `dev2-staging-06bm5-z`, 83,394,315 bytes in total.
- **Why removal was safe:** every object is served by a released repository's `main`, except the 0.6B-Z tokenizer
  `aeb13307…`. That one has node copies, and `--allow` re-hashed one of them first.
- **Checks:** commits and refs are unchanged and no LFS object is left. `dev2-release-staging-06bm4` was already
  empty.

## 5. F-b does not fit: STOP

- **Exact package size** ([probe](dev2-bf16-storage-2026-09-30/ops/size_probe.py) on node B's
  `m4b/A1-soup/checkpoint`, safetensors headers):
  - The FP32 backbone is 102.498 GB (25,624,600,064 parameters).
  - The `v2.release.bf16_copy` backbone is **53,797,381,192 bytes**: 496 Linear matrices in BF16, with the
    1,271,398,400-parameter embedding table (5.09 GB) and 353 small tensors (norms, gated-delta `A_log` /
    `dt_bias`, conv filters) kept in FP32.
  - The new decision head adds 21,056,376 bytes. `tokenizer.json` and the banner are already stored in DEV2.0-27B.
  - Total new LFS bytes: **53.82 GB**.
- **A smaller exact package does not exist.** The 27B runtime runs `model.float()` under BF16 autocast
  (`training/model/infer.py`), so the embedding table and norms are used in FP32. An all-BF16 copy (51.25 GB) would
  change the numerics.
- **Arithmetic:** 50.46 GB used + 53.82 GB = 104.28 GB, which is over the cap. Keeping 3 GB of headroom needs 56.82 GB
  free, and 49.54 GB is free, so 7.28 GB is missing. Purging the DEV2.0-27B adapter first (0.49 GB) would break the
  current `main` and still not fit.
- **What stayed undone:** the node B → node A transfer, the C1 post-key guard (item 8), the release and the adapter
  purge. The C1 ledger is untouched, so the 27B tier's one successor slot is still free.
- **Options (user decision):**
  1. **Add private storage** (a paid plan or a storage add-on for the organization). Steps 2–5 can then resume at
     once, taking an estimated 3–4 hours and about 1.5 GPU-hours. A later 27B full-weight update also needs its old and new
     copies side by side (about 108 GB), which is above the free cap even with nothing else stored.
  2. **Free at least 7.28 GB outside Decision 2.0** (delete or make public, the owner's call). Examples:
     Vela-2.0-Encoder-307M-Unified 3.13 + Decision-1.0-Route-0.6B 2.32 + `vela-embedding-training` 1.21 +
     Vela-1.0-Embedding-Data 0.86 = 7.52 GB, which leaves 3.24 GB after the F-b upload.
  3. **Keep DEV2.0-27B on the adapter** until storage is resolved. F-b stays on node B as the hash-recorded
     durable store (`m4b/A1-soup/checkpoint` `5bcfa3d4…`; package `m4b/F-b/package`).

## 6. Commits

`522c28c6c` copy launcher · `78592cbb4` specs, decisions and tools · `6cee2835d` staging tool `--allow` · this record.
