# Decoder M4 incident — staging-repo LFS cleanup rewrote history and removed unrequested weights (2026-09-29)

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
     say history was intact; it was not.
   - The staging SHAs cited in records no longer exist. For example, S2T `545a6784…` is now `fdcfba1a…`, and
     N4LKr `784a894f…` is now `f594e7cd…`; the other cited commits (for example E8F `16c0929a…`) are gone too.
3. **Unrequested weights disappeared.** After the 20:51 call the repo listed no LFS object at all. The weight
   shards, decision heads and tokenizers of `m3/S2T-soup` (the 2B release candidate) and `m2/E8F-soup` were
   gone from every commit, although neither was requested. A download at S2T's commit returns 404. Small non-LFS
   files (configs, calibrations) remain.

## Impact

- **The 2B release is not affected.** Release engineering's DEV2.0-2B pipeline had already passed at 19:47
  (package and real HF download verified from the release repo). Node A also keeps the M3 download
  `staging-545a6784`.
- **The DEV2.0-0.8B release is not affected.** Its weights are in the released repo.
- **Records that cite staging commits no longer resolve.** Content stays identifiable by the node-B SHA-256
  lists (`/data/dev2/runs/dec/m3/hf-staging/*.sha256`, `m2/hf-staging/batch*.sha256`).
- **No bytes are lost.** Node B holds every staged folder, hash-verified at 20:50.

## Remediation

- **S2T restored.** `m3/S2T-soup` was re-uploaded from node B at **`dev2-dec-staging@376c992cea89b473ff663d2d1bbbd08d006fc8a8`**
  (20:55). All 11 paths match the node-B list `S2T-soup.sha256`, and a downloaded file re-hashes equal. This
  keeps the policy's "current release candidate's staging copy".
- **E8F soup not re-staged.** It is released; the released repo holds it.
- **New tool.** `ops/m4/m4-free-lfs.py` passes `rewrite_history=False` and fails if any object outside the
  requested folders is gone afterwards. The M3 tool must not be used again.
- **For the coordinator and release engineering:** cite S2T staging by content (the SHA-256 list) or by
  `376c992c`, not `545a6784`. Other tracks' cleanups should check their tool's `rewrite_history`.
