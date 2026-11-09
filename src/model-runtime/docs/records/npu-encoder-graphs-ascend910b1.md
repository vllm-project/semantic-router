# Encoder graphs on NPUs: Ascend 910B1

Record of the encoder graph capture generalized from CUDA to the `npu` device
class, measured on a shared host (the capture is the same bucketed design the
CUDA path already runs; this record pins what it does on torch_npu and what
it buys the encoder families). It closes the measurement asked for by the
graph-capture issue: vela3-fast serving on the NPU with `graphs` enabled at
the same answers as eager.

## Findings

- The vela3 encoder serving path is host-bound: a profiled forward of
  vela3-fast dispatches ~1800 tiny tensor ops per forward (17816
  `empty_tensor`, 3953 `aten::to` and 2622 `aten::copy_` calls across 10
  forwards, 625ms of self CPU time) while the device itself computes a
  fraction of that. Replaying each bucket as an NPUGraph collapses the
  dispatch.
- On the 910B1 (CANN 8.5.1, torch 2.10.0, torch_npu 2.10.0.post4),
  `torch.npu.graph(npu_graph, pool=..., stream=..., capture_error_mode=...)`
  accepts the same keyword surface as `torch.cuda.graph`, and every serve
  captures its buckets without a failure. Back-to-back rounds, graphs on
  (this change) against eager (main `83f8242e7`), same card allocation,
  serving vela3-fast: sequential p50 125.1 / 131.4 ms -> 34.5 ms (3.6x),
  4-way concurrency 12.4 / 11.4 rps -> 40.7 rps (3.3x), 16-way 56.9 / 56.1
  -> 67.6 rps (+19%), 128-way 112.3 / 112.4 -> 140.6 rps (+25%). The gains
  run through both batch shapes: the scheduler's merged batches replay
  through the packed entry, a family's own window batches through the
  padded one.
- Answers are unchanged: a 60-request corpus under 8-way concurrency agrees
  with eager on every choice (60/60, no flips, max |dp| 0.0) — the replay
  is bitwise the eager bucket it was captured from. The padded graph
  computes on the bucket grid through `EncoderBackbone.masked` — whose key
  mask never crosses back to the host — so a replay reads only its static
  input buffers; the backbone's `padded` layout reads the mask on the host
  (`bool(mask.all())`), and a replay would keep that read from its capture.
- The capture is skipped cleanly where the graph APIs are missing:
  `graph_module` resolves the device module's capture surface
  (`torch.cuda`, or `torch.npu` once torch_npu registered it) and returns
  None for CPU, MPS and XPU, so the engine builds no graphs there and the
  `npu` capability reports `graphs` accordingly.

## Setup

- Host: 2-socket Kunpeng 920, 8x Ascend 910B1 64GB (canhazgpu-scheduled, one
  card per serve); CANN 8.5.1, torch 2.10.0, torch_npu 2.10.0.post4;
  `vllm-sr/vela3-fast` at registry pin `7cbefd4b`, profile `exact`, fp32,
  `--device npu:0`, main `83f8242e7`.
- Requests: the vela3 golden requests plus a 30-text corpus x 3 rounds under
  8-way concurrency (accuracy), and sequential / 4-way / 16-way probes plus
  16/32/64/128-way throughput runs (performance), graph replay and eager
  served back-to-back on the same host.
- The engine keeps the capture behind the existing `graphs` engine argument
  (default on, `--no-graphs` to opt out), matching the CUDA path; a capture
  that fails on a device falls back to that bucket's eager forward.

## Implications

- The `npu` device class runs the encoder families' bucketed graph capture:
  sequential encoder latency drops ~3x, which matters most for the
  low-concurrency serving the router actually sees.
- The padded window path of encoder families (vela3 among them) is covered
  by the same buckets through `EncoderGraphs.padded`; grids above the
  buckets or ones that pad too much run the engine's eager padded path.
- Device classes without the graph APIs (CPU, MPS, XPU) are unchanged: no
  graphs are built and the capability reports `graphs: false`.
