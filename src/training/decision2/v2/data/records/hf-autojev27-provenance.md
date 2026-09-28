# AutoJev-27B soft targets — teacher provenance (read before use)

- **Teacher:** `denis-pplx/autojev-27b` at revision `6f5b557e037f5edb25c7dc92dbc6553e5a19c015`
  (Apache-2.0 weights), run through its package-native decision interface (`inference.autojev27`,
  AutoJev source `ee63c151`) on node A, image `f83b1d10…`, FLA kernels, one shared and fully warmed
  Triton autotune cache. It is a third-party decision model and never a Decision 2.0 weight origin.
- **Caveat:** AutoJev's public training pipeline reportedly used SFT data generated with a closed
  OpenAI model, so these targets carry that provenance. Every AutoJev-distilled candidate discloses it
  on its card and in its records and needs a matched own-Lux-target control. Own-Lux targets remain the
  clean default.
- **Permission:** coordinator decision (2026-09-28 18:45 UTC+8; re-qualification ordered 20:15)
  after the runtime passed the M3a qualification v2 (bitwise repeat determinism across processes and
  GPUs, frozen autotune cache, spot check against the eval track's recorded predictions).
- **Files:** `<wave>.targets.jsonl` (`{id, input_sha256, teacher_probs}`, join by id),
  `<wave>.attestation.jsonl` (per row: model id, attested revision, adapter, backend, config / package /
  source hashes, prompt digest, shard, GPU, image), `<wave>.report.json` (hashes, guard, per-shard
  receipts, production repeat check). Only TRAIN rows; no panel, SELECT, CAL or sealed item.
