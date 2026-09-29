# Decoder track — staging-revision citation errata (2026-09-29)

Erratum (2026-09-29, per coordinator 05:25 note). The earlier LFS cleanups of `llm-semantic-router/dev2-dec-staging`
called `permanently_delete_lfs_files` without `rewrite_history=False` (huggingface_hub 1.33 defaults to rewriting
history), so the staging revisions cited in decoder records are no longer part of the repo's history, and their weight
files cannot be downloaded. **From now on, a decoder staging artifact is identified by its per-file SHA-256 list and the
verified node copies, not by a staging revision.** The released repos (DEV2.0-0.8B, DEV2.0-2B) were never rewritten.

Times are UTC on 2026-09-28/29 unless marked UTC+8 (UTC+8 = UTC + 8 h). Correction convention in the amended records:
wrong text is struck through (`~~…~~`) and followed by `[corrected 2026-09-29: …]`. Old revision ids are kept.

## 1. Who deleted what (corrected attribution)

| When (UTC) | Who | Call | Deleted from `dev2-dec-staging` | History | Evidence |
| --- | --- | --- | --- | --- | --- |
| 18:15:39 | decoder M3 | free-lfs-1 | N4T soup, N4J soup (33,667,162,160 B) | rewritten | node B `m3/hf-staging/free-lfs-1.receipt.json` `b49a05a9…` |
| 18:32:46 | decoder M3 | free-lfs-2 | N4L soup, E8V soup, B8F-s1, B8F-s2 (25,895,031,941 B), **including the shared `tokenizer.json` object `06b95093…`** (listed under `m2/B8F-s1`), which every staged folder uses | rewritten | node B `m3/hf-staging/free-lfs-2.receipt.json` `a8b29320…`; remaining 36,705,529,760 B |
| 20:47:46 (04:47 UTC+8) | release worker (2B release) | free-duplicates | **S2T soup** weights + head (3 objects, 7,535,759,320 B) and **E8F soup** weights + head (2 objects, 3,013,820,512 B) | rewritten | [receipt](../../release/records/dev2-2b-release-2026-09-29/storage/free-duplicates-20260928T204746Z.receipt.json); [steward record §4](../../release/records/hf-storage-steward-2026-09-29.md); remaining 26,155,949,928 B |
| 20:51:45 (04:51 UTC+8) | decoder M4 | free-lfs-m4-1 | exactly the requested N4LKr soup, E8F-s1/s2/s3, X4K-s1/s2 (16 objects, 26,155,949,928 B) | rewritten | node B `m4/hf-staging/free-lfs-m4-1.receipt.json` `1bd74dc0…`; remaining 0 B |
| 20:55:46 | decoder M4 | S2T re-upload | none (commit `376c992c`) | kept | node B `m4/hf-staging/S2T-restore.upload.log` |
| 21:14:27 (05:14 UTC+8) | storage steward | free_staging_lfs | S2T weights + head again (7,535,759,320 B) | kept (`rewrite_history=False`) | steward record §2 |
| 23:24:38 | decoder M4 | free-lfs-m4-2 | N4LX soup (16,833,581,080 B), `lost_unrequested: []` | kept | node B `m4/hf-staging/free-lfs-m4-2.receipt.json` `d993ed9e…` |

So the decoder's 20:51Z call deleted only what it requested: the bytes it removed equal the bytes left after the release
worker's 20:47Z call. The S2T and E8F soup weights and heads had already been deleted by the release worker at 20:47Z,
and their `tokenizer.json` (a single object shared by all folders) by the decoder's own M3 cleanup at 18:32Z.

## 2. Hub status of every cited revision (read-only check from node A, 2026-09-29 ≈00:10–00:20 UTC)

`dev2-dec-staging` history now has 12 commits: `4e399022` (initial), `41ea96d1` (batch 1), `30af6c40` (batch 2),
`fdcfba1a` (S2T), `a2233cff`, `bb10187a`, `bac0f142`, `2b990e52` (N4T/N4J/N4L/E8V), `f594e7cd` (N4LKr), `376c992c`
(S2T restore), `5db8791b` (N4LX), `e8656221` (N4XF; `main`).

The pre-rewrite ids are **not in that history**. The Hub API still resolves them by SHA as orphaned commits (tree listing
works; a made-up SHA returns 404), but their weight and head files return HTTP 403. Orphaned commits are not a durable
reference, so treat these revisions as gone. Download status was checked with HEAD requests that follow the redirect;
nothing was downloaded. "Cited in" line numbers refer to the records as of `c62cc0853`, before these errata blocks were
added.

| Revision | Artifact | Cited in | Status | Canonical citation from now on |
| --- | --- | --- | --- | --- |
| `5421582dfed51dd3b19b5686128ed75e0f8525e3` | M2 batch 1 (B8F-s1, E8F-s1, X4K-s1/s2) | dec-m2-batch1-formal-lock:13, dec-m2-0p8b-candidate:30, dec-m3-0p8b-release-support:34, m2-ops/dec-formal-b1.sh.txt:4 (node-A dir name only) | not in history; weights 403 | hash list `batch1.sha256` `74c04b5d…` (§3.6) |
| `16c0929ac0df649d8223483e1adf419d78a647ed` | M2 batch 2 incl. E8F soup | dec-m2-results:175, dec-m2-batch2-formal-lock:23, dec-m2-0p8b-release-candidate:31, dec-m3-0p8b-release-support:16,35,36, incident:18, m2-ops/dec-formal-b2.sh.txt:4, m2-ops/dec-mlx.sh.txt:6 (node-A dir name only) | not in history; weights 403 | E8F soup: §3.3 + DEV2.0-0.8B; seeds: `batch2.sha256` `ab0aa4bc…` (§3.6) |
| `545a6784175ce7a62319abeaec9502d775e586e2` | S2T soup (2B) | dec-m3-results:201, dec-m3-2b-candidate:44, incident:17,28,43 | not in history; weights 403 | §3.1 + DEV2.0-2B@`5ad3e9a3` |
| `784a894fea7ec43f7bded7bf61a12933639987c7` | N4LKr soup (4B HOLD) | dec-m3-results:197,210, incident:18 | not in history; weights 403 | §3.2 |
| `fdcfba1a…`, `f594e7cd…` | S2T / N4LKr after the rewrite | incident:17–18 | in history; weights 403 | same as above |
| `376c992cea89b473ff663d2d1bbbd08d006fc8a8` | S2T restore | incident:36,43, dec-m4-results:67,92 | in history; S2T weights + head 403 (steward, 21:14:27Z); `tokenizer.json` 200 | §3.1 + DEV2.0-2B@`5ad3e9a3` |
| `e86562218928fbb77f8071b45646962f63049fad` | N4XF soup (4B release candidate) | dec-m4-results:83, dec-m4-4b-candidate:54,56 | **live**: `main`, 14 files, all checked LFS files 200 | §3.4 (revision valid; hash list is the identity) |
| `5db8791bfc6b205675aaadc3f7282ac460b631f2` | N4LX soup (4B HOLD) | not cited in records (node B `.commit` only) | in history; weights + head 403 (decoder 23:24Z, history kept); configs/tokenizer 200 | §3.5 |

The released repos serve the same weight SHA-256s as the staged soups (checked the same way):
DEV2.0-2B `main` = `5ad3e9a3cc4865ce0360f4ecce2b345020bfdb38` serves `1f083a66…`, `3a3f291b…`, `33b6541b…`;
DEV2.0-0.8B `main` (`f458c34c…` at check time; released package `0b631a85c19fb573aee34fc68bb413271ebe89f4`) serves
`9db82b84…` and `cf2a2ebd…`.

## 3. Hash lists and verified node copies

All lists are `sha256sum` output on node B under `/data/dev2/runs/dec/`, with paths relative to the listed node-B
directory. On 2026-09-29 ≈00:15 UTC every node-B copy re-hashed OK against its list (`sha256sum -c`), and so did every
node-A copy named below. "Bytes" is the sum over the files in the list.

### 3.1 S2T soup (2B; released as DEV2.0-2B)

Identity = hash list node B `m3/hf-staging/S2T-soup.sha256` (`22e7fe86fba8b925d4930e2691c046a10235dc30ae45b9933560d8e81c9a47dd`),
11 files, 7,555,792,950 B. Node copies: node B `/data/dev2/runs/dec/m3/hf-staging/S2T-soup/m3/S2T-soup`, node A
`/data/dev2/runs/dec/m3/formal-candidates/staging-545a6784/m3/S2T-soup` (node A list `m3/formal-candidates/S2T-soup.nodeB.sha256`,
same bytes). Released: `llm-semantic-router/DEV2.0-2B@5ad3e9a3cc4865ce0360f4ecce2b345020bfdb38`. **Do not re-upload S2T.**

| File (under `m3/S2T-soup/`) | SHA-256 |
| --- | --- |
| `cal698-16k/calibration.json` | `d73e9ce464efcb7fe00def6680a0065488fc6e650c8ea42b479ddef9c04392e7` |
| `cal698-8k-dev/calibration.json` | `f87a162ba947e91d1fe0cddce71aff4594e8c6d5f9cce8b3a3cda28ddc180611` |
| `checkpoint/backbone/config.json` | `071e97d8291168ba712237744c9a60e733acce554e6164322d22550dc96de19b` |
| `checkpoint/backbone/model-00001-of-00002.safetensors` (3,999,855,496 B) | `1f083a66d8cdcd02887452b2801efae6af41a6e9529bdc37c962b994721dca08` |
| `checkpoint/backbone/model-00002-of-00002.safetensors` (3,527,479,552 B) | `3a3f291be8d6ac079f1f737ec89092ed3947abd2cd5fd342823c91ce1084af8b` |
| `checkpoint/backbone/model.safetensors.index.json` | `12b12d9183d20062b4b5ab51d0d0a666ada0a2c53b8bd1b078f3456d5c3db781` |
| `checkpoint/chat_template.jinja` | `273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80` |
| `checkpoint/decision_config.json` | `0b6c3429ee06032739d7386659c332ed7bb62d0b96ccddfcad4fad999e22cd4f` |
| `checkpoint/decision_head.safetensors` (8,424,272 B) | `33b6541bb6636677eb91a4d8e06acd4db81152b11097088840796f11d49c707a` |
| `checkpoint/tokenizer.json` (19,989,325 B) | `06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523` |
| `checkpoint/tokenizer_config.json` | `bee8eba30f0eb4af73c0fe2cd06d0f89b657d7819941c438157ec42f7c80ea87` |

### 3.2 N4LKr soup (4B, HOLD)

Identity = hash list node B `m3/hf-staging/N4LKr-soup.sha256` (`25f4deab542496d97d6c179b83f1f1332b1ee9156a414c74ae0c48306c93119e`),
14 files, 16,853,623,521 B. Node copies: node B `/data/dev2/runs/dec/m3/hf-staging/N4LKr-soup/m3/N4LKr-soup` (its durable
store), node A `/data/dev2/runs/dec/m3/formal-candidates/staging-784a894f/m3/N4LKr-soup`. No stored object on the Hub.

| File (under `m3/N4LKr-soup/`) | SHA-256 |
| --- | --- |
| `cal698-16k/calibration.json` | `ee711d712531d5622e15591eb22234c64cae779da17c04b039c5c95be8f2f546` |
| `cal698-8k-dev/calibration.json` | `05fcb1766713d6fc746d208c7842d0ed6cc994d1b02a6ade8520ca3e51df0645` |
| `checkpoint/backbone/config.json` | `a5ed4156fda05f0f9149c66964d6165916754e7355488a8a07d9b0398acdbdb9` |
| `checkpoint/backbone/model-00001-of-00005.safetensors` | `a1677240107987a2729ddc5a01e25032573f70568e832f87841d7a5753a5f6dc` |
| `checkpoint/backbone/model-00002-of-00005.safetensors` | `c1e4439d60193e6f0887e4679ffaf70835f7c5259cda533513d73b94df79777e` |
| `checkpoint/backbone/model-00003-of-00005.safetensors` | `9ed4eb161a1ee9763be0661a951756f4841ca8201b0b9cc286baa6c580bbb228` |
| `checkpoint/backbone/model-00004-of-00005.safetensors` | `6dc63260e42bf8ceda8994a90dd59cd1990f4101d35e04d1907aa065dac0d1f3` |
| `checkpoint/backbone/model-00005-of-00005.safetensors` | `1623e2ac2f80cc87649c98e2559fa2e9ea12ed49b27f9227006200f39224bec7` |
| `checkpoint/backbone/model.safetensors.index.json` | `1921033b850d1a219406de25ac8a8c13c45048572cd71b4a738246c1f690986c` |
| `checkpoint/chat_template.jinja` | `a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715` |
| `checkpoint/decision_config.json` | `7117bed1f1fe94ce864f93be3cfbf7e8a187da56c2b85e5a46cb25a868d1fe8e` |
| `checkpoint/decision_head.safetensors` | `6375b3bb1a9410d554272689ff35bb1434097097bcd64d7046f695b170fba167` |
| `checkpoint/tokenizer.json` | `06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523` |
| `checkpoint/tokenizer_config.json` | `bee8eba30f0eb4af73c0fe2cd06d0f89b657d7819941c438157ec42f7c80ea87` |

### 3.3 E8F soup (0.8B; released as DEV2.0-0.8B)

Identity = the 9 `m2/E8F-soup/` lines of hash list node B `m2/hf-staging/batch2.sha256`
(`ab0aa4bc00b6c884d982c9a69af2755be51c685eefb4209fdb880d0849dcf8c6`), 3,033,821,530 B. Node copies: node B
`/data/dev2/runs/dec/m2/hf-staging/batch2/m2/E8F-soup`, node A `/data/dev2/runs/dec/m2/formal-candidates/staging-16c0929a/m2/E8F-soup`.
Released: DEV2.0-0.8B `0b631a85c19fb573aee34fc68bb413271ebe89f4` (same weights and head).

| File (under `m2/E8F-soup/`) | SHA-256 |
| --- | --- |
| `cal698-16k/calibration.json` | `9f76867d5be618da315bd860aae689ed0cf2a2a20ea34d5bae485b011877e05c` |
| `cal700/calibration.json` | `4728f1e8f2c882e20ba8c06c02df529d92f31e0fa921c8a322d36d85a0bafad3` |
| `checkpoint/backbone/config.json` | `2b5d7e14da681afa0f2717f4e450c8a4cd05d41a7aff99231578d5aa3a6bad73` |
| `checkpoint/backbone/model.safetensors` (3,009,606,928 B) | `9db82b841878b2f259701e11fdccdfd4b54021b857934304de181e28379f5f1e` |
| `checkpoint/decision_config.json` | `3503b7cb43a1e4994c27916bf35653612a517c249f2ed9458000904a3f8da700` |
| `checkpoint/decision_head.safetensors` (4,213,584 B) | `cf2a2ebdd0b007eca460d90889f129f80a75b2371be54ca1889f4bf15613af6a` |
| `checkpoint/tokenizer.json` | `06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523` |
| `checkpoint/tokenizer_config.json` | `bee8eba30f0eb4af73c0fe2cd06d0f89b657d7819941c438157ec42f7c80ea87` |
| `members.txt` | `a631119c5076f03a3759dd2b4b46c4863acf29d662587efc9c7bd6c96595ccb0` |

### 3.4 N4XF soup (4B release candidate; revision `e8656221` is live)

Identity = hash list node B `m4/hf-staging/N4XF-soup.sha256` (`cea4cc9a2a5997e02ebb55ed3cb653b348f2067855a51b13bbbba7f95da984f3`;
node A `m4/formal-candidates/N4XF-soup.nodeB.sha256`, same bytes), 14 files, 16,853,623,514 B. Node copies: node B
`/data/dev2/runs/dec/m4/hf-staging/N4XF-soup/m4/N4XF-soup`, node A `/data/dev2/runs/dec/m4/formal-candidates/staging-e8656221/m4/N4XF-soup`.
Hub: `dev2-dec-staging@e86562218928fbb77f8071b45646962f63049fad` `m4/N4XF-soup/` (downloadable at check time).

| File (under `m4/N4XF-soup/`) | SHA-256 |
| --- | --- |
| `cal698-16k/calibration.json` | `176a2f0e8f76c7b281c0784ec9e9d2ea17b84aef2e237e2673a062a5a62e421f` |
| `cal698-8k-dev/calibration.json` | `c822fafa3bfc0c95fb5ffe6507e83b4242f5eaa4c014a55b6ec77afb86d801cf` |
| `checkpoint/backbone/config.json` | `a5ed4156fda05f0f9149c66964d6165916754e7355488a8a07d9b0398acdbdb9` |
| `checkpoint/backbone/model-00001-of-00005.safetensors` | `06faf67f333d1136e4cbc583220e98602354b454d3b0fb917bf433a8bf4dbb3f` |
| `checkpoint/backbone/model-00002-of-00005.safetensors` | `abac23bcaad1d5fb2aa73d9d051db970514a4f427b0cff22336b50c5e7aa7302` |
| `checkpoint/backbone/model-00003-of-00005.safetensors` | `1f492a389f49f382b4bbf84686176b51f9621da50d143e436ed61f764a9fa988` |
| `checkpoint/backbone/model-00004-of-00005.safetensors` | `18024da979fb12651442e3a168deb92b4b8e41d41ea8076151df62879ecfa823` |
| `checkpoint/backbone/model-00005-of-00005.safetensors` | `0ad4368c4f04781d30c237c269a9911b9ffb037104a4ee469bb7f56d57e67d15` |
| `checkpoint/backbone/model.safetensors.index.json` | `1921033b850d1a219406de25ac8a8c13c45048572cd71b4a738246c1f690986c` |
| `checkpoint/chat_template.jinja` | `a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715` |
| `checkpoint/decision_config.json` | `6658fbd01bf92030078ad1bbe735a0de77045431055d06c32bc6de8cf7d275d9` |
| `checkpoint/decision_head.safetensors` | `bd324aa69788fa936293ff4959ea1caa8845a4e47589879b0748aed3c3df7c5a` |
| `checkpoint/tokenizer.json` | `06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523` |
| `checkpoint/tokenizer_config.json` | `bee8eba30f0eb4af73c0fe2cd06d0f89b657d7819941c438157ec42f7c80ea87` |

### 3.5 N4LX soup (4B, HOLD; revision `5db8791b` resolves, weights deleted)

Identity = hash list node B `m4/hf-staging/N4LX-soup.sha256` (`f85a820caa2595089b8ab4d0abee99582397160419e0b6cb44e2524a3d1d924a`;
node A `m4/formal-candidates/N4LX-soup.nodeB.sha256`), 14 files, 16,853,624,038 B. Node copies: node B
`/data/dev2/runs/dec/m4/hf-staging/N4LX-soup/m4/N4LX-soup`, node A `/data/dev2/runs/dec/m4/formal-candidates/staging-5db8791b/m4/N4LX-soup`.
Weights and head: `4ef0611e…`, `1b2e18b3…`, `26745407…`, `b8cfee62…`, `240d85a7…` (shards 1–5), head `e87d15b8…` (full values in
the list and in `free-lfs-m4-2.receipt.json`).

### 3.6 M2 batches (E8F seeds, X4K seeds, B8F seeds)

| List (node B `m2/hf-staging/`) | SHA-256 of list | Files | Bytes | Upload revision (gone) | Node copies |
| --- | --- | ---: | ---: | --- | --- |
| `batch1.sha256` (B8F-s1, E8F-s1, X4K-s1, X4K-s2) | `74c04b5dde66fafe3d505ceab1111d7d42ab38e9ee95e8b039b78deb9fddaf41` | 37 | 6,388,624,285 | `5421582d` | node B `/data/dev2/runs/dec/m2/hf-staging/batch1`; node A `/data/dev2/runs/dec/m2/formal-candidates/staging-5421582d` (and `staging-16c0929a`) |
| `batch2.sha256` (B8F-s2, E8F-s2, E8F-s3, E8F soup, E8F-s1 `cal698-16k`) | `ab0aa4bc00b6c884d982c9a69af2755be51c685eefb4209fdb880d0849dcf8c6` | 37 | 12,135,298,259 | `16c0929a` | node B `/data/dev2/runs/dec/m2/hf-staging/batch2`; node A `/data/dev2/runs/dec/m2/formal-candidates/staging-16c0929a` |

Weight and head SHA-256s of the seeds deleted at 20:51Z (also in `free-lfs-m4-1.receipt.json`):

| Folder | Files / bytes on node B | `model.safetensors` or `adapter_model.safetensors` | `decision_head.safetensors` |
| --- | --- | --- | --- |
| `m2/E8F-s1/checkpoint-0001776` | 9 / 3,033,822,934 | `8b332483dc9a86faa4ac51dfb6f31c7c7f3de874bbe14abdaebea70b8bfe147c` | `bab6b40ce900b0f39d830ed86e76b81beabce4fcc71d3c329b73b7346a2efc2f` |
| `m2/E8F-s2/checkpoint-0001520` | 9 / 3,033,822,933 | `4501784c9c010779722668cbdb69f5f0b0b54d24ce46336c7d3e3952affcfa11` | `50f3931967794203a84413fab216507efbe36f312adfde110b28c83d3739e15c` |
| `m2/E8F-s3/checkpoint-0002041` | 9 / 3,033,822,865 | `74f8cda9ea2003434b5c800c13b6c6c4a030247c02816a8c3f80ee7b5292d3da` | `fba71d97fbfec204188df5382717d628846b44d789c271f639cc3190d1ae7817` |
| `m2/X4K-s1/checkpoint-0000315` | 10 / 160,488,456 | `c85edc1f55b0df5c5797ce89f993ed7b495402dbd88b6659d7c3a69272ad76c5` | `c3876307151f429185bc90d2211a34fb4568e68c7a72ea8ce1f7bd685c9bc9b3` |
| `m2/X4K-s2/checkpoint-0000266` | 10 / 160,488,494 | `ebf7a6ab4a29503984360feaf6d97a02ceda74366257675251dd279a5e945af1` | `22f0a750a40adf43974d5a3eae2c1c0508956f2dadf719885fb12c20d9ff994e` |

## 4. Not amended

- `m2-ops/dec-formal-b1.sh.txt`, `dec-formal-b2.sh.txt`, `dec-mlx.sh.txt` use `staging-5421582d` / `staging-16c0929a` only
  as node-A directory names; those directories exist and verify (§3.6).
- DEV2.0-2B@`b2c5d7eac4ef24648cbf9c23c26421f0960ca542` is said in release records to carry the same weights as `5ad3e9a3`;
  not re-checked here.
