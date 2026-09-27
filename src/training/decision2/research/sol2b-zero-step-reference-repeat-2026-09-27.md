# Sol 2B prospective zero-step reference-runtime check

**Decision: HOLD; stop before optimizer.** This checks whether a fixed
32-item, matched-batch zero-step inference can be compared strictly with the
completed targeted160-to-clean-v2 training control. It does not select a
checkpoint or report model quality. No typed FINAL or CSS15 gold was opened.

The prior [inline soft-replay arm](sol2b-inline-soft-replay-prereg-2026-09-27.md)
stopped before optimization after a BF16 probability mismatch. This follow-up
kept the same FP32-materialized targeted160 source, SELECT700, original
batch-of-two mates and order, fixed 32-ID roster, 8,192-token admission, and
historical checkpoint-zero predictions. It chose the pinned Transformers
PyTorch reference gated-delta implementation prospectively, with deterministic
algorithms enabled before model loading. Source/model, SELECT, roster, and
historical-control SHA-256 values are respectively
`2f4bb061e0881d2d5f29da1cedee8655655ce339bacfa4f6971bcd845add5ae9`,
`32a4352d8ed93ce82430db80175339ad8e4d40c618f2866608fdb6ef5120f2a6`,
`4c247f30d6a879ce20d77acce1a0d9fdbd0d7c9b5a120f6e9b8febd499ad199a`,
and `ceda618a37069832af69375bc5905b28da952f8386be972c88dc6a30671c8cb3`.
The source materialization receipt SHA-256 is
`eae632fba65bc3b208aa3698d2334a88cbd5a0a4b7521ade9398fa3be3aacd1c`.

The signed runner and tests are commit `8b5971422`; runner SHA-256 is
`23be37ac06aba92eeb3562940b80df57c309b5b1dfcb082bd65408f05d394e16`.
Three focused tests and full changed-file `make check` passed locally. The
exact source archive SHA-256 is
`4f1892b5798d8bd17f94994216de646a08a460764b7bd1935d3fbccb5d315531`;
the pinned runtime image digest is
`f83b1d10f14dbe46ea14ee56fd3e5d01849673f3739fed5311c99ba54cbc2d54`.
Both preregistrations fixed zero optimizer updates, at most two independent
processes, a 0.25 GPU-hour limit, zero categorical changes, maximum drift
`1e-4` versus the historical control, and maximum drift `1e-6` between fresh
processes. Historical comparison came first, so its failure required stopping
without the second process.

## Execution and stop evidence

| Attempt | Frozen lock SHA-256 | Observed outcome | Private receipt SHA-256 | GPU wall time |
| --- | --- | --- | --- | ---: |
| r2 | `d5da77fdfa6c5c8c584bc5f9d83e8d275cad225f2ed5fb9ad26384fc74f76f9d` | Pre-inference launcher error: setting `PYTHONPATH` replaced the image's FLA import path; no prediction was produced. | `e10e37f7341fbbb9ffec0d17f9662c49161f1f611c6e6726182a61e85b3b92e1` | 4.748 s = 0.001319 h |
| r3 | `3ba94418affb658c9fb858f17636f06fcac8472f0ecf43edf77205a1003db3d3` | New lock changed only the launcher to preserve the original FLA path. A CPU-only wrapper self-check passed; the first fresh process answered all 32 fixed rows. | Execution `388216b30b98ff15eadf14e2d141626887e31f79aee8a588bf9cddfaa0179229`; stop `e8a54a709962a7834a6cedaa12df240b9f0d2bd74ddf546c121e7d6f97066678` | 25.092 s = 0.006970 h |

The r3 CPU-only wrapper self-check receipt SHA-256 is
`d4841d2f14895955c1f07ab9d7d4c83aee051234137e97f241498711d81ea3c1`.
The first gold-free prediction file SHA-256 is
`cf913af7893ecd149e5a12958742dbd4a1cece6332a6aeaf66e4b4d0ef0c0641`.
All 32 categorical outputs matched the historical control, but the maximum
absolute option-probability drift was **0.0136505365**, exceeding the frozen
`1e-4` limit by more than two orders of magnitude. No second process ran;
therefore cross-process repeatability was **not measured**. Neither attempt
made an optimizer update or read sealed evaluation labels. Combined GPU wall
time was about **0.00829 GPU-hours**, below the 0.25-hour budget; the task
containers exited and the GPU was released.

This result does not establish the specific numerical cause of the historical
drift. It does establish that this reference-runtime output cannot be treated
as an exact zero-step continuation of the completed control under the frozen
threshold. Preserve the old control and its metrics; do not relax the limit,
try another checkpoint, or start the proposed soft-replay treatment under this
protocol. A future causal replay test would need its own prospective common
runtime and comparison design. The existing 2B release status remains HOLD.
