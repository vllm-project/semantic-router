# Decoder M4 incident — staging-repo LFS cleanup rewrote history ~~and removed unrequested weights~~ (2026-09-29)

> **Erratum (2026-09-29, per coordinator 05:25 note).** This record wrongly attributed the loss of the S2T soup and E8F
> soup objects to the decoder's 20:51 call.
>
> - **Weights and heads.** The release worker's 2B-release cleanup at 20:47:46 (04:47 UTC+8) deleted them on purpose
>   (receipt [`free-duplicates-20260928T204746Z`](../../release/records/dev2-2b-release-2026-09-29/storage/free-duplicates-20260928T204746Z.receipt.json);
>   [steward record §4](../../release/records/hf-storage-steward-2026-09-29.md)): S2T 3 objects (7,535,759,320 B) and E8F
>   soup 2 objects (3,013,820,512 B). That call also rewrote history.
> - **Tokenizers.** Their `tokenizer.json` is one LFS object (`06b95093…`) shared by every staged folder. The decoder's own
>   M3 cleanup at 18:32 had already deleted it (`free-lfs-2.receipt.json`, listed under `m2/B8F-s1`).
> - **The 20:51 call** deleted exactly the requested 16 objects, 26,155,949,928 B. That equals the bytes left after 20:47
>   (`free-lfs-m4-1.receipt.json` `1bd74dc0…`: remaining 0).
> - **Revisions.** The staging revisions cited here (`545a6784`, `784a894f`, `16c0929a`) are no longer in the repo's
>   history. The Hub still resolves them by SHA as orphaned commits, but their weights return 403.
> - **S2T restore.** The weights of `376c992c` were deleted again by the storage steward at 21:14:27 (history kept) and now
>   return 403. S2T is released as DEV2.0-2B@`5ad3e9a3cc4865ce0360f4ecce2b345020bfdb38`.
> - **Canonical citations** (per-file SHA-256 lists and verified node copies) are in
>   [dec-staging-citation-errata-2026-09-29.md](dec-staging-citation-errata-2026-09-29.md).
>
> Corrections below: struck text `~~…~~` followed by `[corrected 2026-09-29: …]`.

Private repo `llm-semantic-router/dev2-dec-staging`. Times are UTC on 2026-09-28.

## What happened

1. **Storage-policy cleanup (M4).** Before uploading any M4 finalist, I freed the LFS objects of `m3/N4LKr-soup`
   (HOLD), `m2/E8F-s1…s3` and `m2/X4K-s1/s2` (individual seeds) at 20:51 with the M3 tool
   (`records/m3-ops/m3-free-lfs.py.txt`).
   - Every node-B copy was re-hashed first: N4LKr 14 files; M2 batches 37 + 37 files.
   - Deleted 26,155,949,928 bytes, exactly those folders. Receipt `m4/hf-staging/free-lfs-m4-1.receipt.json`
     `1bd74dc0…` on node B.
2. **The tool rewrites history.** It calls `permanently_delete_lfs_files` without `rewrite_history`, and in
   huggingface_hub 1.33 that argument defaults to `True`.
   - Both M3 cleanups (18:15 and 18:32) and this one therefore rewrote every commit of the repo. M3's records
     say history was intact; it was not. [corrected 2026-09-29: the release worker's 20:47 cleanup also rewrote it.]
   - The staging SHAs cited in records no longer exist. For example, S2T `545a6784…` is now `fdcfba1a…`, and
     N4LKr `784a894f…` is now `f594e7cd…`; the other cited commits (for example E8F `16c0929a…`) are gone too.
     [corrected 2026-09-29: they are no longer in the repo history; the Hub still resolves them by SHA as orphaned
     commits, weights 403. Identity by hash list, see the errata record.]
3. ~~**Unrequested weights disappeared.** After the 20:51 call the repo listed no LFS object at all. The weight
   shards, decision heads and tokenizers of `m3/S2T-soup` (the 2B release candidate) and `m2/E8F-soup` were
   gone from every commit, although neither was requested.~~ [corrected 2026-09-29: **S2T and E8F soup objects were
   already gone before the 20:51 call.** The release worker deleted their weight shards and decision heads at 20:47
   (receipt `free-duplicates-20260928T204746Z`). Their `tokenizer.json` (shared object `06b95093…`) had been deleted
   by the decoder's M3 cleanup at 18:32. The 20:51 call deleted only the requested folders (26,155,949,928 B, equal to
   the bytes left after 20:47), so the repo held no LFS object afterwards.] A download at S2T's commit returns 404.
   Small non-LFS files (configs, calibrations) remain.

## Impact

- **The 2B release is not affected.** Release engineering's DEV2.0-2B pipeline had already passed at 19:47
  (package and real HF download verified from the release repo). Node A also keeps the M3 download
  `staging-545a6784` (a node-A directory name; it re-hashes equal to node B's `S2T-soup.sha256` `22e7fe86…`).
- **The DEV2.0-0.8B release is not affected.** Its weights are in the released repo.
- **Records that cite staging commits no longer resolve.** Content stays identifiable by the node-B SHA-256
  lists (`/data/dev2/runs/dec/m3/hf-staging/*.sha256`, `m2/hf-staging/batch*.sha256`). [corrected 2026-09-29: per-list
  SHA-256s, file tables and node copies are in the errata record.]
- **No bytes are lost.** Node B holds every staged folder, hash-verified at 20:50.

## Remediation

- **S2T restored.** `m3/S2T-soup` was re-uploaded from node B at **`dev2-dec-staging@376c992cea89b473ff663d2d1bbbd08d006fc8a8`**
  (20:55). All 11 paths match the node-B list `S2T-soup.sha256`, and a downloaded file re-hashes equal. ~~This
  keeps the policy's "current release candidate's staging copy".~~ [corrected 2026-09-29: S2T was already released
  as DEV2.0-2B, and its staging copy had been deleted on purpose at 20:47. The storage steward deleted the re-uploaded
  weights and head again at 21:14:27 (history kept). `376c992c` is still in the history, but those files return 403.
  S2T must not be re-uploaded.]
- **E8F soup not re-staged.** It is released; the released repo holds it.
- **New tool.** `ops/m4/m4-free-lfs.py` passes `rewrite_history=False` and fails if any object outside the
  requested folders is gone afterwards. The M3 tool must not be used again.
- **For the coordinator and release engineering:** cite S2T staging by content (the SHA-256 list) ~~or by
  `376c992c`~~, not `545a6784`. [corrected 2026-09-29: cite S2T by DEV2.0-2B@`5ad3e9a3cc4865ce0360f4ecce2b345020bfdb38`
  or by hash list `m3/hf-staging/S2T-soup.sha256` (`22e7fe86…`, node B); see the errata record.] Other tracks' cleanups
  should check their tool's `rewrite_history`.
