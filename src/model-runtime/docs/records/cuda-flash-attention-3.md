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

- **Date:** latency measured 2026-10-25 on
  [#5470](https://github.com/vllm-project/semantic-router/issues/5470). Answer,
  golden, graph and peak-memory checks measured 2026-10-28 on revision
  `41879a48d`, with two tool workarounds this change now does itself: planning
  `--long-states` with long context, and capping per-row groups at 65,536
  padded tokens. Those checks were not rerun after this commit.
- **Device:** one NVIDIA H200.
- **Package:** `vllm-sr/d3-turbo` (`689b3af1`), BF16.
- **Stack:** CUDA router image `e38f1fe7`, PyTorch 2.10.0+cu128, `flash_attn_3==3.0.1`.

## Latency

p50 of one request, from #5470. Short states are unchanged. Attention dominates
the long ones. The 60K rows there are latency only. That run planned without
long context, so every question over the validated length (32,768 tokens) was
refused and the answer comparison did not include 60K.

| State (input tokens) | FlashAttention-2, p50 | FlashAttention-3, p50 | Speed-up |
| --- | ---: | ---: | ---: |
| ~26,000 tokens, 1 / 16 questions | 3,644 / 3,812 ms | 3,164 / 3,300 ms | 1.15x |
| ~51,000 tokens, 1 / 16 questions | 8,898 / 9,179 ms | 7,028 / 7,182 ms | 1.27–1.28x |
| ≤ 2,000 tokens | unchanged | unchanged | 1.00x |

## Answers

`tools/state_once_parity.py --long-states --row-tokens 65536` on the 2026-10-28
run, planning with long context. 204 questions: the tool's default requests
plus 30K and 60K states. FlashAttention-3 was the selected kernel
(`branch_attention: flash-attn-3`, `numerics: approximate`); every answer
differed from the FlashAttention-2 run. Against that run (`--baseline`): 203 of
204 decisions identical. The one change was thread-64 / `delivery_tone`, a near
tie (margin 0.0068). Largest absolute probability gap 0.040, mean 0.0047. Per
answer type: choice 80/80, noul 62/62, score 61/62 (98.4%).

The answer comparison on #5470 was the default requests and the 30K state only, 191
questions, because the 60K questions were refused. Its 190 of 191, and the gaps
quoted from that issue (largest 0.0206, mean 0.0042; state-once against per-row
on FlashAttention-2, largest 0.0313, mean 0.0046), do not include 60K. The
204-question numbers do.

Score agreement of 98.4% is below the bar `DtypePolicy` sets for making an
approximate kernel the default (no decision changed, and at least 99% agreement
per answer type). The argument stays off.

## Golden check

With `--flash-attention-3`, the startup golden check matched 5/5 against a
FlashAttention-2 `cuda:sm_90` reference recorded with
`tools/golden_answers.py --record`. Without that reference the card says
`unverified`: d3-turbo has no CUDA reference in `golden_answers_decision3.json`.

## Graphs

Graphs on against `--no-graphs`, the same 204 questions. FlashAttention-3: 32
of 204 probability vectors differ (largest absolute gap 0.025, no decision
changed). FlashAttention-2 on the same requests: 33 of 204 differ (largest
0.012, no decision changed), on ticket-16, thread-16 and thread-1/4. The gap
is not specific to FlashAttention-3. `decision3-graphs.md` records bit-equal
graphs for d3-flash on an MI300X, a different package and device.

## Peak memory

`--no-graphs --peak-memory`, with `num_splits=1` on the non-causal call.
FlashAttention-3 matched FlashAttention-2 to the byte on every request. The
60K state with 16 questions peaked at 61.56 GiB, weights included.

## How to rerun

`--long-states` plans with long context and, when `--row-tokens` is omitted,
caps per-row groups at 65,536 padded tokens.

```bash
python3 tools/state_once_parity.py --package DIR --output fa3.json --device cuda:0 \
  --flash-attention-3 --long-states --baseline --peak-memory
python3 tools/state_once_parity.py --package DIR --output fa3-eager.json --device cuda:0 \
  --flash-attention-3 --long-states --baseline --no-graphs
```

The 4,000-request sample from `state-once.md` waits until its request file is
in the repository.
