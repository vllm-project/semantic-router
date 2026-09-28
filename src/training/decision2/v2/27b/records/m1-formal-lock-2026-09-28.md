# ~27B Milestone 1 formal run lock (before any formal prediction)

Decision under the preregistered rule (`m1-prereg-2026-09-28.md`): no
challenger displaced C0. A1-flac (Qwen3.5-27B, identical recipe) read out
P_dev 67.55 vs C0 68.44 (Δ −0.89) with Choice −8.75 points (791→721/800),
Noul +2.5, Score +6.0 (161→185/400) and CSS-pilot .6203 vs .6178; it fails
the P_dev margin and the Choice floor. Frozen heads read out ≤40.7. A2-flac is
a seed replicate and not eligible for promotion. **C0 takes the formal slot.**

## Candidate and runtime

- Package `llm-semantic-router/DEV2.0-27B`, manifest SHA-256
  `47f1e934062ec5d8f75dc6b74812c4fbfaf2d567f1a099f7f9f5cb2d41ffe782`
  (reassembled on node B: small files from the node A package, model files
  from the BEST368 checkpoint; every manifest-listed hash verified), revision
  `package-sha256:47f1e934…`, base `Qwen/Qwen3.8-27B@1d4bf0f2…` (LFS-verified),
  25,688,227,840 loaded parameters, in-package CAL.
- Native loader inside the package via `publication.package_native_arena`
  functions (`v2/27b/package_collect.py`: one model load, same checks), image
  `sha256:dbe5f32b…`, one MI325X (node B GPU5), no network, **reference
  gated-delta path** (FLA not on `PYTHONPATH`), batch = one prompt with all its
  questions, 4,096-token limit, over-budget questions invalid.

## Order and rules

1. Parity preflight: package-native DEV1,600 and CSS pilot1,430 compared with
   C0's source-run predictions by `publication.panel_parity` (zero category
   changes, p99 drift ≤ .005, maximum ≤ .02). Failure stops the formal run.
2. One formal collection: typed FINAL 1,600 (`e2a4a86b…`), CSS15 6,547
   (`7a527357…`), public231 (`642d3fac…`). One shot; a technical failure is
   recorded and not retried without a new amendment.
3. Adopt into the eval track's runner (integration commit `8e5f22cd8`), gold-free
   seal, then report and paired comparison (5,000 draws, seed 20260927) with
   the adopted AutoJev-27B comparator (seal `4002a7e0…`, reproduces v3 72.310 /
   public231 200).
4. Labels: "post-key same-panel" for v3; public231 is a public-subset rerun,
   not the official JevBench rank. No release or publication decision here.
