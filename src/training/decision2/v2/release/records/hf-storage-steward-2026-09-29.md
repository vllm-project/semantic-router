# Hugging Face storage steward (2026-09-29)

Times are UTC+8 on 2026-09-29; receipts use UTC (8 hours earlier). GB = 10^9 bytes, as the Hub counts.

## Result

- **Org private storage: 65.30 GB when this work started (04:57), 57.76 GB after (05:15). Headroom is 42.24 GB of
  the 100 GB free-plan cap** (`canPay: false`). This meets the 04:55 policy's ">= 40 GB free" target.
- The coordinator's 04:55 note said 83.90 GB. Two changes since then:
  - 04:51: the decoder track's M4 cleanup deleted 26.16 GB from `dev2-dec-staging`: the 4B N4LKr soup (HOLD) and the
    E8F and X4K seeds.
  - 04:55: the decoder re-uploaded `m3/S2T-soup` (7.56 GB), which put usage at 65.30 GB.
- **One deletion, in one repo.** The three weight objects of `dev2-dec-staging` `m3/S2T-soup/`, 7,535,759,320 bytes.
  They are the staging copy of the released DEV2.0-2B, whose `main` serves the same three files.
  - The deletion used `rewrite_history=False`. The repo's 10 commits and `main` `376c992c` are unchanged.
- **Kept on the Hub (the order rule allows it at 42.24 GB free):**
  - 9B DW: `dev2-9b-staging@8a98dc82` `m3/DW/`, 31.78 GB.
  - 0.6B Z: `dev2-staging-06bm5-z@6b414780`, 2.40 GB.
  - No seed, superseded soup or non-best HOLD soup had stored objects left. The 4B N4LKr soup is already off the
    Hub, and no 27B staging repo exists.
- **Never touched:**
  - The released repos DEV2.0-0.6B / 0.8B / 2B.
  - Every dataset.
  - The private "Decision 2.0" collection, which still holds exactly those three.
- **Headroom command for every track:** `src/training/decision2/v2/common/hf_headroom.sh --node node-a --min-free-gb <upload GB>`
  (§5).

## 1. Inventory (private repositories)

Probe: [`ops/inventory.py.txt`](hf-storage-steward-2026-09-29/ops/inventory.py.txt) (read-only; usedStorage, LFS
objects, trees at every branch and tag, commits). Summaries:
[before](hf-storage-steward-2026-09-29/storage/inventory-before-20260928T205713Z.summary.json) `a6d5831d…`,
[after](hf-storage-steward-2026-09-29/storage/inventory-after-20260928T211530Z.summary.json) `e0d0dc09…`
(per repo and per top-level folder).

| Repository | Kind | Before (04:57) | After (05:15) | Status |
| --- | --- | ---: | ---: | --- |
| `dev2-9b-staging` | staging | 31.78 | 31.78 | kept: 9B current best HOLD (DW) |
| `DEV2.0-2B` | released | 7.56 | 7.56 | never deleted from |
| `dev2-dec-staging` | staging | 7.56 | 0.02 | S2T weights deleted (§2) |
| `decision-2.0-training-data` | dataset | 3.23 | 3.23 | never deleted from |
| `DEV2.0-0.8B` | released | 3.03 | 3.03 | never deleted from |
| `DEV2.0-0.6B` | released | 2.40 | 2.40 | never deleted from |
| `dev2-staging-06bm5-z` | staging | 2.40 | 2.40 | kept: 0.6B current best HOLD (Z) |
| `Decision-1.0-Kai-0.6B-Router-Signals` | model, not staging | 2.32 | 2.32 | out of scope |
| 16 other datasets (Vela, cache, PrivRoute) | dataset | 4.93 | 4.93 | never deleted from |
| `decision-2.0-eval-artifacts` | dataset | 0.08 | 0.08 | never deleted from |
| 7 small models (incl. `dev2-release-staging` 0.58 MB, `dev2-release-staging-06bm4` 0) | model | < 0.01 | < 0.01 | nothing to free |
| **Total (32 private repositories)** | | **65.30** | **57.76** | headroom 42.24 GB |

Top-level folders of the staging repos (stored LFS objects):

- **`dev2-9b-staging`:** `m3/DW/` only. 11 objects, 31.78 GB, pushed 04:19 (FP32 checkpoint, 9 shards).
- **`dev2-dec-staging`, before:** `m3/S2T-soup/` only. 4 objects, 7.56 GB, pushed 04:55.
  - The tree still lists the small files (configs, calibrations) of `m2/{B8F-s1,B8F-s2,E8F-s1,E8F-s2,E8F-s3,E8F-soup,X4K-s1,X4K-s2}`
    and `m3/{E8V-soup,N4J-soup,N4L-soup,N4LKr-soup,N4T-soup}`.
  - Their weight objects were deleted between 02:15 and 04:51 (§4), so those folders have no weight pointers.
- **`dev2-dec-staging`, after:** `m3/S2T-soup/checkpoint/tokenizer.json` only, 19,989,325 bytes.
- **`dev2-staging-06bm5-z`:** the whole repo. 3 objects, 2.40 GB, pushed 02:26.

The only other bytes not at any branch head are 0.46 GB of older revisions in `decision-2.0-training-data`. That is
a dataset, so it is excluded by rule.

## 2. Deletion

Tool: [`ops/free_staging_lfs.py.txt`](hf-storage-steward-2026-09-29/ops/free_staging_lfs.py.txt) `9d9c8a66…`. It ran
from the node-A mirror of `b7de26242` (tree `01611d85…`) in the deletion order of
[`ops/targets.json`](hf-storage-steward-2026-09-29/ops/targets.json) `00663aa2…`.

- [Plan](hf-storage-steward-2026-09-29/storage/plan-20260928T211313Z.receipt.json) `f05544b5…`.
- [Apply](hf-storage-steward-2026-09-29/storage/apply-20260928T211410Z.receipt.json) `6e1cf574…`, deleted
  2026-09-28T21:14:27Z.
- Node-B re-hash: [`nodeb-s2t.hashes.json`](hf-storage-steward-2026-09-29/storage/nodeb-s2t.hashes.json)
  (sha256sum, 21:12:24Z).

| Repo | Path | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| `dev2-dec-staging` | `m3/S2T-soup/checkpoint/backbone/model-00001-of-00002.safetensors` | 3,999,855,496 | `1f083a66d8cdcd02887452b2801efae6af41a6e9529bdc37c962b994721dca08` |
| `dev2-dec-staging` | `m3/S2T-soup/checkpoint/backbone/model-00002-of-00002.safetensors` | 3,527,479,552 | `3a3f291be8d6ac079f1f737ec89092ed3947abd2cd5fd342823c91ce1084af8b` |
| `dev2-dec-staging` | `m3/S2T-soup/checkpoint/decision_head.safetensors` | 8,424,272 | `33b6541bb6636677eb91a4d8e06acd4db81152b11097088840796f11d49c707a` |
| **Total** | | **7,535,759,320** | |

- **Node copies.** Each file has two copies, both re-hashed minutes before the deletion and equal to the Hub SHA-256:
  - node A `/data/dev2/runs/dec/m3/formal-candidates/staging-545a6784/<path>`;
  - node B `/data/dev2/runs/dec/m3/hf-staging/S2T-soup/<path>` (the source of both uploads).
- **Why it was eligible.**
  - It is the staging copy of a released model: DEV2.0-2B@`5ad3e9a3` `main` has the same three SHA-256s
    (`backbone/model-0000{1,2}-of-00002.safetensors`, `decision_head.safetensors`).
  - The 04:10 storage plan schedules exactly this deletion ("2B S2T once released").
  - The 02:45 policy kept a release candidate's staging copy only until release. The 2B release was at ≈04:45, before
    the 04:55 re-upload, and the 04:55 policy supersedes the staging parts of 02:45 / 04:10.
- **Recent-upload rule (pushed 19 minutes before).** Both node copies were re-hashed, and the track records name them:
  - the decoder's incident record `dec-m4-incident-hf-staging-2026-09-29.md` @ `f0c78cc99` (re-uploaded from node B;
    node-B SHA-256 lists; node A's `staging-545a6784`);
  - the 2B release receipt `free-duplicates-20260928T204746Z.receipt.json` @ `73398871f` (node-A paths).
- **Kept: `m3/S2T-soup/checkpoint/tokenizer.json`** (`06b95093…`, 19,989,325 bytes). Kept paths in other repos have
  the same SHA-256: `dev2-9b-staging` (DW), DEV2.0-0.8B, DEV2.0-2B and Decision 1.0 Eos / Sol / Nox / Lux.
- **How rule (b) was applied.** Hub LFS storage is per repository, so the released repo's own copy of the same bytes is
  why the staging copy is redundant; it does not block the deletion. Every other same-SHA file, in the same repo or any
  other org repo at any branch or tag, blocks deletion.
- **Verification** (apply receipt, 20 s after the call):
  - `dev2-dec-staging`: commit list (10) and refs are identical.
  - Exactly the three chosen objects are gone; one object remains.
  - The deleted paths return HTTP 403.
  - DEV2.0-2B serves the three identical files (HTTP 206 on a ranged GET).
  - `dev2-9b-staging` and `dev2-staging-06bm5-z` are unchanged (commits, refs, objects).
  - `hf_headroom.sh` then read 57.76 GB.

## 3. What stays, and the next candidates if space is needed

The order rule keeps the current best HOLD candidate of each tier on the Hub while headroom allows. It does: 42.24 GB is
free after the rest. If a later upload needs more, delete in this order, re-hashing first (neither copy was re-hashed
here because neither was deleted):

1. **9B DW:** `dev2-9b-staging` `m3/DW/`, 31.78 GB. Node A copy `/data/dev2/runs/9b/m3/DW-build/soup`, located by
   exact size and named in `lux9b-m3-formal-lock-DW-2026-09-29.md`. Map repo paths by removing `m3/DW/checkpoint/`.
2. **0.6B Z:** `dev2-staging-06bm5-z`, 2.40 GB. Node A copy `/data/dev2/runs/06b/m1/arms/m5-z-soup-r517/full/best-export`,
   named in `m5-results-2026-09-28.md`.

The 4B N4LKr soup (`dev2-dec-staging` `m3/N4LKr-soup`) has no stored object on the Hub. The decoder's node-B copy
under `/data/dev2/runs/dec/m3/hf-staging/N4LKr-soup` is its durable store.

## 4. Finding: earlier cleanups rewrote commit history

In huggingface_hub 1.33, `permanently_delete_lfs_files` defaults to `rewrite_history=True`. These cleanups called it
without the argument, so each rewrote every commit of the repo it touched:

- decoder M3, 02:15 and 02:32;
- release (2B), 04:47;
- decoder M4, 04:51.

| Repo | Cited revision (records) | Status now |
| --- | --- | --- |
| `dev2-dec-staging` | S2T `545a6784`, N4LKr `784a894f`, E8F `16c0929a` | gone; history now `4e399022` … `f594e7cd` (+ `376c992c`) |
| `dev2-release-staging-06bm4` | `62c61c10` | gone; history now `87e2fa05` … `55394bd8` |
| `dev2-release-staging` (Kai 1.0 dry run) | `5afd8fc8` | gone; history now `00f2d35a`, `545377c0` |

- The 2B release record says "commit history is unchanged". That is not so: the decoder's incident record documents the
  rewrites for its own cleanups, and the table shows the same for the release cleanup.
- The decoder's incident record attributes the loss of the S2T and E8F objects to its own 04:51 call. The release
  worker's receipt shows it deleted them on purpose at 04:47 (`free-duplicates-20260928T204746Z`).
- **Do not re-upload S2T.** DEV2.0-2B@`5ad3e9a3` serves the identical weights, and both nodes keep copies. Cite S2T by
  DEV2.0-2B or by the node-B SHA-256 list.
- This cleanup is the first in the program that preserved history (checked above). The decoder's new
  `ops/m4/m4-free-lfs.py` also passes `rewrite_history=False`.

## 5. Headroom check

`src/training/decision2/v2/common/hf_headroom.sh` (tests `v2.common.tests.test_hf_headroom`; README note in `v2/program/README.md`):

```bash
src/training/decision2/v2/common/hf_headroom.sh --node node-a --min-free-gb 18   # before an ~18 GB upload
src/training/decision2/v2/common/hf_headroom.sh --node node-a --json             # machine-readable
```

It sums the Hub's usedStorage over the org's private repositories (the same measure as the 2B record's probe), prints
usage, headroom and the largest repositories, and exits 1 below `--min-free-gb` (2 on errors). The query runs on the
node, where the token is, and node addresses are redacted from errors.

## 6. Compute and commits

- **GPU:** none. **CPU:** re-hashing 7.5 GB twice on node A (plan and apply) and once on node B, plus Hub API calls.
- **Commits:**
  - `62d6927b7`: `hf_headroom.sh`, its test and the README note;
  - `b7de26242`: ops tools and targets (the mirror used for the run);
  - this record, merged into `xunzhuo/decision-2-training`.
