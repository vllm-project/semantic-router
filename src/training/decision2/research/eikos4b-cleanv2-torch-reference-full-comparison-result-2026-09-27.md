# Eikos clean-v2 4B stable-runtime full comparison result

The prospective plan is in
`eikos4b-cleanv2-torch-reference-full-comparison-prereg-2026-09-27.md`.
All stages passed under the same pinned package, calibration, source adapter,
image and physical GPU. The selected-source and merged-package models each
used the verified Transformers PyTorch reference gated-delta function before
model load, with deterministic algorithms enabled. No missing or newly
initialized weight warnings occurred.

The same-process selected-source versus merged-package parity covered complete
typed DEV 1,600 and CSS pilot 1,430. Each panel had **zero categorical
mismatches, p99 option-probability drift 0.0 and maximum drift 0.0**, within the
unchanged `0`, `0.005`, `0.02` limits. The combined 3,030-item parity receipt
passed; its SHA-256 is
`f66d253476966979a5be9cb7f4c4511d1a0854e677034b815b6d3f7ff5b9546b`.
The DEV and CSS parity report SHA-256s are, respectively,
`61a28c01133f7d988c2ac9c664e2c19c737cf5d36a36ff5c521c7b52b4ea43d1`
and `ef15622ffce730f70b972ec1ac9662aafda58f2886206c0e3cf20d1c1e5ce1ca`.

The package collector then ran in three fresh processes. The separately
collected DEV and CSS package answers and probabilities exactly matched the
same-process merged outputs, confirming stable output across those processes.
All score reports verified their frozen designated inputs and targets.

| Panel | Validity | Diagnostic score | Prediction SHA-256 |
| --- | --- | --- | --- |
| DEV 1,600 | 1,600/1,600 valid | 1,443/1,600 correct; accuracy 0.901875 | `1596dbcf67ca8ecf1559fb679a1453bd47c07558d8d6176177f24f32c5c8af5b` |
| CSS pilot 1,430 | 1,430/1,430 valid | 788/1,430 correct; micro accuracy 0.551049; median task macro-F1 0.539548 | `83388f3f05340d24bf413d07046c7bde0c4cb353d8937a1934d0e401af6c8779` |
| JevBench public 231 | 231/231 strictly valid; zero renormalized | 195/231 correct; accuracy 0.844156 | `6483a0c786c7a115898be5429c8920193177e320d7bdbb89cb44ff63bfb9a396` |

The direct-parity prediction SHA-256s, in selected-source then merged-package
order, were:

| Panel | Selected-source SHA-256 | Merged-package SHA-256 |
| --- | --- | --- |
| DEV 1,600 | `c0d722d9681562ee7ad2966209e57fcd803594a1f5c0a1f102a5005c6848fe6f` | `0c711315eb7231e09ba4ebfe9e5c6efa8e39332e0fa0d3de9fd91ebed4f571a0` |
| CSS pilot 1,430 | `1677f2af1dedb0ce300f907c8a1cd24d5755ae4bdb9f97177b16028b5a3fcc34` | `04a2c9fcd1c09953a5e66572a77d53a877b0dcf7093cfcc8426d90baa2e56035` |

The DEV, CSS pilot and public score report SHA-256s are
`ccf9472a23ebe221584c810a4fa80e48fd1e46b3c006a10559b053d94aa45c8a`,
`a28fd41ac12eeb3f58007a63918941ef25b444d03c61022eefad658ba2134301`,
and `0aaa5db51a357ed98487b8f048dcd5f3e82d8e244f39b6e6fdd3b6cb9076c635`,
respectively. Five GPU processes used **0.165556 wall GPU-hours** in total;
summed synchronized inference for the three package-only processes was
**0.052334 GPU-hours**. The private execution receipt, including per-process
times, image/GPU identity, manifests, logs and all artifact hashes, has SHA-256
`337ee48a1d5607d7987b3ae64a2f6f4297eac7fda0bbef45253f3767eca9cb0f`.

This passes the frozen source/package parity and designated-panel validity
gates in the repeatable PyTorch reference runtime. Accuracy figures are
diagnostic and were not used to choose a checkpoint or threshold. No CSS 15
evaluation labels, sealed FINAL labels, official scoring, training or HF
publication were involved.
