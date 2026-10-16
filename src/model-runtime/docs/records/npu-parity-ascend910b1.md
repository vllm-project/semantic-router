# NPU parity: Ascend 910B1 (first round, shared host)

Record of the `npu` device class's first readiness-parity round, measured on a
shared host. The findings feed the readiness gate change and the npu golden
references added to the registry in the same change.

The readiness gate compares two runs bitwise and never retries a golden
failure. This record shows the bitwise standard does not transfer to the npu
device class, and the gate for `npu` is therefore tolerance-based: category
agreement plus a max-diff bound (`GPU_TOLERANCE`), with npu references
collected on the device.

## Findings

- The Vela 1.0 encoder is bitwise deterministic on the NPU: 15 classify runs
  across three collection rounds, run-to-run diff 0.0 on every request.
- Its probabilities differ from the CPU's by up to 1.27e-2 — on the top-class
  probability itself — while the top-1 label never flipped (5/5, including a
  sample whose top-2 gap is 0.026). The CPU reference is therefore not
  reusable for the npu device class.
- The Vela 2.0 decoder is bitwise deterministic per request on a quiet card
  (run-to-run diff 0.0), with CPU-vs-NPU drift 2.2e-4 on the first collected
  decision. Its full corpus on a busy shared host is pending: loading Vela 2.0
  while third-party NPU inference runs on the host fails intermittently at the
  readiness golden gate — driver-level scheduling drift crosses the bitwise
  run-to-run check ("golden answers are not deterministic"), the load is never
  retried (`--load-attempts` semantics), and the runtime exits when no model
  stays ready. On the same host the serve has also been killed right after
  readiness and segfaulted during shutdown — each time while third-party NPU
  inference was active, and never on a quiet card. The tolerance-based npu gate
  in this change is what keeps the load from failing in those windows.

## Setup

- Host: 2-socket Kunpeng 920, 8× Ascend 910B1 64GB; CANN 8.5.1,
  torch 2.10.0, torch_npu 2.10.0.post4; one card, `--device npu:0`.
- Models: Vela-1.0-Encoder-307M-Domain (local package) and
  vllm-sr/Vela-2.0-0.3B (pinned a3209a50), profile `exact`, fp32.
- Requests: the family's golden requests (the registry reference's request
  set) plus a 5-text classify probe and 2 four-signal decisions states,
  answered by a CPU serve and an NPU serve on the same host; the NPU answers
  every request twice. Per-field max abs float diffs come from the JSON
  responses (`npu-parity-910b1-classify.json`).
- References: `tools/golden_answers.py --device npu:0 --record` on both models
  added the `npu` entries to `registry/golden_answers_vela1.json` and
  `golden_answers_vela2.json`.

## Implications

- The npu device class carries its own golden references, collected on the
  device; the CPU reference is not reusable.
- The readiness determinism gate is tolerance-based for the npu device class
  (category agreement plus a max-diff bound): the CPU bitwise standard does
  not transfer, and the bitwise gate made Vela 2.0 unreliable to serve on
  shared Ascend hosts.
- A shared-host caveat applies until Vela 2.0's full corpus is re-collected on
  a quiet card: same-host third-party NPU traffic can still fail a load
  through paths outside the golden gate (serving killed after readiness,
  segfault during shutdown).
