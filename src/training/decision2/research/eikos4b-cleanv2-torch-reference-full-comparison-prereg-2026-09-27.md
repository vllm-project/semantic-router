# Eikos clean-v2 4B stable-runtime full comparison

Status: prospective. The completed PyTorch-reference repeatability arm passed
two independent full CSS pilot processes with zero answer or probability
drift. This fixed follow-up measures package scores and selected-source versus
merged-package parity on complete, already designated development/public
panels. Its results cannot select a checkpoint or change the runtime.

Freeze the clean-v2 standalone package `SHA256SUMS` SHA-256
`7e005ef609553d973c3a6232840436d1384657a19f5765e42752ebcc04906e39`
and embedded calibration SHA-256
`6b6af1ba82c2fc5111c49a881f64154f1f27ac9b7e09e8859789ae3bb01f3b60`.
The selected source is checkpoint `checkpoint-0232` of base release manifest
SHA-256 `8978a143290508976d8ddffb735416954cca1de779d5524ca4c59d528539d3dd`,
with `NATIVE_BEST.json` SHA-256
`d46a2d8ff09126a22a851fcc988261dd6b14252ac9eaa00046e52fda10abb459`,
adapter weights SHA-256
`31e783d97e70715e67e39b935ed662e6bbc7d185ce848649be9c004c6d51cad7`
and adapter config SHA-256
`82ad5297859fdd2c1d0372aadf47d28da6f662d46f59bb509475012a9710ac71`.
The package provenance must bind that exact source and calibration.

Freeze three complete prompt panels: typed DEV 1,600 SHA-256
`a17ec4b675bbc3da96dba8f31af8f25c9b02cc96ff048fb7de899bdd8b6cf79a`,
CSS pilot 1,430 SHA-256
`598319a429de16c659b59ede0eac3c269939356b0599f4e08d1983e44def3dda`,
and independent JevBench public 231 SHA-256
`642d3fac1b6521fe33df72f9228e4e4e364b7be7ea277893207f97da5bc75ddd`.
For scoring only, freeze DEV gold SHA-256
`c7a8b86bda0d0d6120e572b94dfc756bf10264108554af76307141ae02fbf5dc`,
CSS **pilot** gold SHA-256
`9a7274760dc4ced5ce5219b300974a1cf54c7d5e7e0c7de05d78bb33f2959391`,
and public target SHA-256
`abc17b971d13807a15b3cdb43062f4cd876aad9d7314e72365724904e88b937f`.
The public panel manifest SHA-256 is
`e0e7c67701cf05f996d3b4eac09abf1bb3e6bb363cfe007089acd3644a38ed35`.
Do not read CSS evaluation-task or sealed FINAL labels.

Use one authorized free physical GPU and the exact image digest recorded in
the preceding reference-repeatability execution receipt. Require PyTorch
`2.12.0+git6bbd260`, HIP `7.2.53211`, Transformers `5.17.0`, FLA `0.5.2`,
BF16, native SemIf letter readout, original prompt and option order, and
`--deterministic-algorithms`. Both collectors must set
`--torch-reference-gated-delta` before **either** model load and attest that
the installed Qwen3.5 wrapper changed from the active FLA
`chunk_gated_delta_rule` to its `__wrapped__` Transformers PyTorch reference.
The published package collector SHA-256 is
`9fcbdbbefe8767030d0020cc29bf0150d5b4f1cba348523c8b452beee2f08487`;
the direct parity comparator SHA-256 is
`e3170349b3846cbf4b3094d4a62b8cdb28f886d9723c6eb508e83b758dde622d`.
Record the exact new local source commit, mirror source hashes, image digest,
GPU identity, start/end times and GPU-hours in private receipts.

Execute the stages in this order, with fresh absent outputs and no retry or
threshold change after seeing a result:

1. Run `verify_export --direct-selected --deterministic-algorithms
   --torch-reference-gated-delta` once on
   full DEV and once on full CSS pilot, sequentially. Each invocation loads
   the selected LoRA and the merged package in the **same process** after one
   verified backend switch. Persist all selected-source and merged-package
   answers as separate gold-free prediction files and SHA-bind both to that
   panel's parity report. Require complete original-order IDs, matching input
   digests, valid answers, the pinned source/package/CAL, and the **existing**
   direct-parity gate on **each** panel: zero categorical mismatches, p99
   maximum per-option probability drift at most `0.005`, and worst drift at
   most `0.02`. Bind the two passing reports with `parity_receipt`. Stop the
   arm before scoring if either panel or the combined receipt fails.
2. Run `published_infer --deterministic-algorithms
   --torch-reference-gated-delta` in one fresh process each for complete DEV,
   CSS pilot and public 231, in that order. Require the native manifest to
   bind the frozen package, CAL, input, collector and selected backend, and
   require all DEV 1,600, CSS pilot 1,430 and public 231 answers valid. Score
   DEV with `benchmark.score`, CSS **pilot** with `transfer.score`, and public
   231 with `jev_arena.jevbench_public score` against only the frozen targets
   above. Require DEV `overall.valid_n = 1600`, CSS pilot
   `roles.pilot.valid_items = 1430`, and public `strict_valid = 231` with zero
   renormalized answers. Stop before a later panel if a run, manifest,
   validity or scorer check fails. Record accuracy, calibration and task
   summaries as diagnostics; there is no post hoc accuracy acceptance floor.

Preserve every completed prediction, manifest, parity and score report, logs,
hashes and GPU-hours on pass or failure. A failed parity or validity gate keeps
the clean-v2 first-release candidate on HOLD; a pass only completes these
technical checks. Do not train, search checkpoints, open CSS 15 evaluation or
sealed FINAL gold, change calibration, run official scoring, or publish to HF.
