# Lux1 current-package comparator and frozen runtime — result

Eval & peers track, 2026-09-28; runs R1, D1 and D2 of the Milestone 1
preregistration (amendment A1). **Post-key same-panel** evidence.

## Package and runtime

- Weights: `llm-semantic-router/Decision-1.0-Lux-9B` current `main`
  `cdf4d3ef2dda21518e599fe99ebbe468486b197c`, byte-identical (weights, tokenizer,
  temperature, backbone config) to the runtime-bearing revision
  `bd45a30aee8c84032791c245c70f86dee5389cc8`, whose published `DecisionModel.decide`
  and qualified runtime check are used (bundle manifest `985ade73…`, 7,940,895,744
  deployed parameters, adapter `native-published-v2-overbudget-invalid-v1`).
- Runtime: node A, one MI325X, image `sha256:f83b1d10…` (torch 2.12.0+git6bbd260,
  HIP 7.2.53211, Triton 3.7.1, Transformers 5.17.0, FLA 0.5.2 overlay), runtime
  check `runtime_matches_validated: true`.

## Determinism evidence

| Comparison | Answer-category changes (of 8,778 slots) | Max numeric drift |
| --- | ---: | ---: |
| R1 (node A, GPU6, no persisted autotune) vs D1 (node A, GPU7, autotune persisted) | 0 | 0.0 |
| D2 (node A, GPU6, frozen D1 autotune cache, no new tuning) vs D1 | 0 | 0.0 |
| D1 (node A) vs r4 (node B, 2026-09-28 01:00 UTC) | 43 (typed 7, CSS 36, public 0) | 0.190 |

Host kernel, amdgpu driver (6.19.14.31400000), image package trees and package
bytes are identical on both nodes. FLA's chunked gated-delta kernels select Triton
configurations by timing-based autotuning; node A selects the same configurations
every time (six autotune keys, frozen cache tree SHA-256
`e215f8bd5145181404bae7c502c084033d85c2e767ee4651942e870dcc72a94f`), while node B's
r4 run did not persist its choices. The most likely explanation is that node B's
timing conditions (it hosts other long-running services) selected different
configurations; this was not tested directly because no node B GPU is allocated to
this track. The earlier 0.021 release-example drift is a separate, stale-bundle
comparison and is not needed for current-package parity.

## Frozen runtime and comparator

**Frozen Lux1 runtime:** node A + image `f83b1d10…` + FLA overlay + frozen autotune
cache `e215f8bd…` (bit-identical across GPUs, processes and runs). **Current-package
Lux1 same-panel comparator for the 9B track:** run D1 (D2 identical).

| Metric | Node A frozen runtime (D1 = D2 = R1) | Node B r4 (reference only) |
| --- | ---: | ---: |
| v3 (100·√(T·H)) | **65.808** | 66.268 |
| Typed T / CSS H | 0.77625 / 0.55790 | 0.778125 / 0.56437 |
| Choice / Noul / Score | 711 / 704 / 227 | 712 / 705 / 228 |
| JevBench public 231 (easy/standard/hard) | 183 (48/67/68) | 183 (48/67/68) |
| CSS invalid (native 16,384-token limit) | 4 | 4 |

The 0.46-point cross-node gap is a runtime sensitivity of Lux1 itself, not scorer
noise. 9B candidates must be compared with the node A run on node A, with their own
autotune caches persisted; a cross-node claim would need a node B run with this
frozen cache (proposed for Milestone 2 if a node B GPU slot is granted).

GPU-hours: R1 0.128, D1 0.113, D2 0.109.
