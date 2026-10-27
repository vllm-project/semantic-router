# CUDA: FlashAttention-3 for state-once branched attention

FlashAttention-3 runs the two variable-length attention calls of a state-once
forward (`engines/native/models/branch.py`) on Hopper. It is the
`flash_attention_3` engine argument, off unless an operator sets it. BF16
Decision 3.0 does not consent to approximate kernels: one near tie changed in
the measurement below, and consent would also turn on the Triton fused kernels,
which this change does not.

On every other device, and whenever the argument is off, branched attention
keeps PyTorch's FlashAttention-2 (or the FP32 reference, as before). The CUDA
image does not install `flash_attn_interface`; a process that sets the argument
without it, or on a GPU that is not compute capability 9.x, refuses to load.
Non-causal calls pass `num_splits=1` so FlashAttention-3 does not allocate its
split workspace.

- **Date:** measured 2026-10-25; recorded with this change.
- **Device:** one NVIDIA H200.
- **Package:** `vllm-sr/d3-turbo` (`689b3af1`), BF16.
- **Stack:** CUDA router image `e38f1fe7`, PyTorch 2.10.0+cu128, `flash_attn_3==3.0.1`.
- **Source:** the measurements published on
  [#5470](https://github.com/vllm-project/semantic-router/issues/5470). This
  change did not rerun them.

## Latency

p50 of one request. Short states are unchanged. Attention dominates the long
ones.

| State (input tokens) | FlashAttention-2, p50 | FlashAttention-3, p50 | Speed-up |
| --- | ---: | ---: | ---: |
| ~26,000 tokens, 1 / 16 questions | 3,644 / 3,812 ms | 3,164 / 3,300 ms | 1.15x |
| ~51,000 tokens, 1 / 16 questions | 8,898 / 9,179 ms | 7,028 / 7,182 ms | 1.27–1.28x |
| ≤ 2,000 tokens | unchanged | unchanged | 1.00x |

## Answers

`tools/state_once_parity.py`, the tool's default requests plus 30K and 60K
states, 191 questions. FlashAttention-3 against FlashAttention-2: 190 of 191
decisions identical. The one change was a near tie (margin 0.0068). Largest
absolute probability gap 0.0206, mean 0.0042. For comparison, state-once
against per-row answers on the FlashAttention-2 path: 190 of 191, largest
0.0313, mean 0.0046.

That is below the bar `DtypePolicy` sets for making an approximate kernel the
default (no decision changed, and at least 99% agreement per answer type). The
argument stays off.

## Acceptance checks not rerun here

The acceptance of #5470 asks the record to carry four checks this environment
could not run. `tools/state_once_parity.py` can produce them on an H200 with
`flash_attn_interface` installed:

- Per-answer-type agreement (`by_type` in the tool's report) on the default
  requests and on `--long-states` (about 30K and 60K tokens), against the 99%
  bar.
- The startup golden check passing with `--flash-attention-3`.
- Identical answers with graphs on and with `--no-graphs`.
- Peak memory at 60K tokens with `num_splits=1` (`--long-states --peak-memory`).

```bash
python3 tools/state_once_parity.py --package DIR --output fa3.json --device cuda:0 \
  --flash-attention-3 --long-states --baseline --peak-memory
python3 tools/state_once_parity.py --package DIR --output fa3-eager.json --device cuda:0 \
  --flash-attention-3 --long-states --baseline --no-graphs
```

The 4,000-request sample from `state-once.md` waits until its request file is
in the repository.
