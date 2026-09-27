# Decision 2.0 27B BEST368: full-panel same-GPU native parity

Status: **technical source/package parity PASS; model release HOLD**. This
gold-free check pairs the previously sealed adapter-preserving package outputs
with one new frozen-source inference pass on each panel's original physical GPU.
It supersedes only the incomplete 20-item same-device diagnosis in the
[package-native development note](adapter-best368-package-native-dev-2026-09-27.md).
It does not reopen the old cross-machine score comparison, train a model, or
consume public or sealed release labels.

## Frozen identity and execution

The inner package manifest remained
`e6ca0f51bf7f27c938f35cbdbc6802d72b10a1d3d33405765fe491c32e25110b`;
its loaded text model fingerprint was
`d9f4990427156a7712325de16f6105659fc00015d44c3e2f9331f52481d350d2`.
Both runs used native CAL file SHA-256
`e4e0d9fda575d807503299bdf7c67828a7c4d3c3b09e9fa1bf3750a51b79db78`,
the same 4,096-token limit, BF16 backbone and FP32 head. Before either run,
the frozen source module hashes, the package/checkpoint 35-entry model file
map, the CAL bytes and prompt bytes matched the prior private manifests. The
source runtime used the six-file scored adapter, preserved separately from the
new package API. The release comparator's signed source is `cdb50b0f7` and
its mirrored archive SHA-256 is
`c2ca0dbd21ac8e6a86cbe56d045e0389ab13ec7df11b89b0d71eef93dbc64779`.

The retained private container inspections establish that each new source run
used the **same physical GPU and exact runtime image** as that panel's earlier
package run, with no concurrent reuse of those two GPUs. The package runs had
exited successfully before source inference started. The original package
predictions were reused byte for byte; no package panel was repeated. Private
plan SHA-256: `0475da0a501b8e02baa85eff8d4353dbd178cc0eb72c23b1e8e9c3545fc9f9b5`;
parity launch SHA-256:
`a7cd101126d7c2331cc8680c77bed6fa2064f6aa600b20d815f5916f3aeea50e`.
The image identity, process/device mapping and executable arguments remain in
those private records.

| Gold-free panel | Prompts SHA-256 | New source predictions SHA-256 | Reused package predictions SHA-256 | Exact complete answer objects | Matched invalid outputs | Scalar p99 / max drift |
| --- | --- | --- | --- | ---: | ---: | ---: |
| Typed DEV1,600 | `a17ec4b675bbc3da96dba8f31af8f25c9b02cc96ff048fb7de899bdd8b6cf79a` | `4d646c6823bf6d8025fee12af7a77222095ff38b0481a3b0ba779fced9d604e2` | `bbe2bcccb79380d385d74ca7ad3ac007d04cee77c281c0af8a0cdb8a7b5e2c12` | 1,600/1,600 | 0 | 0 / 0 |
| CSS pilot1,430 | `598319a429de16c659b59ede0eac3c269939356b0599f4e08d1983e44def3dda` | `650f740cd6e6842f60dd7ace2abcd658f800443f346461a246b177b00823d6ce` | `e157b11f003cebc96877a2e183787ccbb32c0189a1ee7f0f3fdc633ce69005b8` | 1,430/1,430 | 2 | 0 / 0 |

The two CSS invalid answers are the same over-budget answers in source and
package. They remain failures in CSS accuracy. The comparator checked all
3,030 scorer point decisions, 4,800 DEV and 7,758 CSS numeric values. Its
zero-mismatch, p99 ≤0.005 and maximum ≤0.02 thresholds all pass. Private
detail SHA-256 values are `cfad1d2cf743216fe750a5eec45d0e32ce30caaa47d13b44d9c0f8f52c9d44c2`
for DEV and `4da411a51cd4d9ae9414b5f774c54cde3c9267b8563b47d0e804156f52033573`
for CSS. The release-gate-format `native-parity.json` SHA-256 is
`25937c0ea07a9fdbb3ca8e64e856f337437221c83233f44fe26556e5fe53a621`;
the actual `bundle_arena._parity` contract accepted it without threshold
changes. Source prediction sidecars bind panel, model and nested CAL file hash;
the package sidecars additionally bind the package SHA. The comparator now
accepts both native CAL sidecar layouts and rejects absent, incorrect or
contradictory dual CAL bindings.

Source DEV inference used 329.0 GPU-seconds and CSS 331.5 GPU-seconds by
container start/end timestamps, or **660.6 GPU-seconds / 0.184 incremental
GPU-hours**. These are execution durations, not cost or throughput claims;
the package outputs and earlier 20-item diagnosis were reused. The two source
runs exited zero. Local focused parity/package tests passed, and full
`make check` passed with 71 Decision 2.0 training contract tests.

## Interpretation and remaining gates

This proves output preservation for this checkpoint's DEV and CSS pilot prompts
under matched physical GPU, runtime, model, CAL and prompt conditions. The
earlier cross-machine/image historical source outputs still disagree with the
package in scored decisions and probabilities; their scores must **not** be
transferred to a model card. The package-native DEV1,211/1,600 and CSS
pilot845/1,430 results remain the score-bearing development observations. No
sealed JevArena FINAL, independently reviewed authored release set, other
release axes, downloaded model artifact or HF collection entry has been
completed by this parity check. Keep release HOLD.
