# CUDA: Vela 2.0 forest graphs

Vela 2.0's decoders run every request as a forest (`models/forest.py`): the
parts as prefix rows, each question as a block. On CUDA the engine now
replays a graph of the forest forward per exact forest shape, captured on the
shape's second use by the same runner as the decoder forward (`fast.py`,
`Graphs.run`). A replay runs the captured kernels on the same shapes, so the
answers are the eager path's bit for bit. ROCm keeps forests eager (below).

- **Date:** 2026-10-10.
- **Device:** one NVIDIA GeForce RTX 4070 Ti SUPER (sm_89, 16 GB), driver
  591.86, under WSL2.
- **Stack:** PyTorch 2.10.0+cu128, Triton 3.6.0, FLA 0.5.2, causal-conv1d 1.7.0.
- **Code:** main of 2026-10-09 with this change.

## Latency

`tools/vela2_bench.py --sides runtime --warmup 0` on 40 prompts: the first 10
questions of MMLU-Pro's test split, each four times in a shuffled order, asked
either one three-way Choice (`--questions`) or the router's signals, once with
graphs and once with `--no-graphs`. Median ms by the occurrence of a prompt:
the first runs eagerly, the second captures its shape, the third and fourth
replay.

| Model | Questions | No graphs | First | Second (capture) | Third and fourth (replay) |
| --- | --- | ---: | ---: | ---: | ---: |
| 0.8B | one Choice | 67.7 | 76.2 | 224.4 | 17.7 |
| 4B | one Choice | 82.0 | 81.1 | 274.6 | 43.5 |
| 0.8B | router signals | 120.1 | 124.0 | 405.7 | 112.8 |
| 4B | router signals | 508.2 | 515.4 | 1,616.5 | 499.1 |

- A one-question request is bound by kernel launches: a replay makes the 0.8B
  3.8x and the 4B 1.9x faster.
- The router's signals pad every block to the widest one, so that request is
  bound by device time and a replay saves 2% to 6%.
- A capture costs about three forwards, once per shape. For a fixed set of
  questions a single-row request's shape is its token count, so shapes repeat
  as often as request lengths do. Forests above the 4,096-token graph cap
  (`MAX_GRAPH_TOKENS`) run eagerly.
- A graph holds far more device memory than its output: 100 one-question
  requests of distinct lengths on the 0.8B left 87 forest graphs and 2,822 MiB
  more in use than before them (26 MiB with graphs off), about 32 MiB a graph
  against 0.5 MiB of output. So captures stop at `MAX_GRAPHS` or once the
  captures have taken `MAX_GRAPH_BYTES` (4 GiB) of device memory or half of
  what was free at the first, whichever is less. Each capture is charged what
  it takes with the allocator's cache emptied before and after it (torch
  empties the cache when a capture begins), so the cache that eager forwards
  grow between captures is not charged to the graphs. With the budget set to
  1 GiB the same run stopped at 22 graphs, charged 1,070 MiB against 980 MiB
  of measured growth. With the card filled to leave 3,071 MiB free, it stopped
  at 30 graphs and 1,357 MiB.
- Counting the device's whole growth since the first capture instead, as the
  budget first did, charged eager forwards to the graphs: a `batching` server
  of Decision 3.0 d3-flash on an MI300X stopped capturing after 16 graphs,
  with the cache 5.8 GiB above the first capture, and one-question requests
  ran eagerly at 38 ms instead of 16 ms from then on.

## Exactness

- `tools/vela2_parity.py --generate 120` on the 0.8B with graphs on: all 120
  requests identical to the packages' engine (max difference 0.0).
- The 4B's answers to 60 generated requests were identical across three passes
  (eager, capture, replay). Its package parity does not fit the 16 GB card
  next to the runtime.
- `tests/test_gpu_forest_graphs.py` runs forests and ordinary batches through
  the engine with graphs on and off and compares them, with eager forwards of
  other shapes between replays and two inputs per shape, so a kept result must
  survive the next replay. Two forests share their tensor shapes and differ
  only in lengths and owners.

## Why ROCm keeps forests eager

On an MI300X in the release image, an eager forward while a forest graph of
eight or more Vela-4B layers is alive faulted the process: a host segfault in
the next synchronize or replay, sometimes a hang. It happened with FLA,
causal-conv1d and the fused kernels replaced by the reference kernels, with
MIOpen off, with rocBLAS instead of hipBLASLt, and with math SDPA, and the HIP
switches `DEBUG_CLR_GRAPH_PACKET_CAPTURE=0`, `HIP_FORCE_DEV_KERNARG` and
`AMD_SERIALIZE_KERNEL=3` did not change it. Random weights in place of the
released ones, fresh allocations and a private graph pool faulted the same way.
The fault is in the ROCm 7.2.3 runtime of that image: under rocgdb, a thread
of `libhsa-runtime64` calls an invalid address while the main thread waits in
`device_synchronize`. `AMD_DIRECT_DISPATCH=0` or `HIP_LAUNCH_BLOCKING=1` avoid
it. The same sequences pass on CUDA and on `rocm/pytorch` with ROCm 10.1.0 and
the same PyTorch 2.12, so ROCm forests can replay once the release image moves
to that ROCm.

## Reproduce

```bash
python3 - <<'PY'
import json, random
# mmlu_pro_prompts.jsonl as built in rocm-mi300x-fused.md
rows = [json.loads(line) for line in open("mmlu_pro_prompts.jsonl")][:10]
order = [row for row in rows for _ in range(4)]
random.Random(0).shuffle(order)
with open("repeated.jsonl", "w") as sink:
    for i, row in enumerate(order):
        sink.write(json.dumps({"id": f"rep-{i}", "text": row["state"]}) + "\n")
json.dump({"route": {"type": "choice", "instructions": "Which kind of task is this request?", "criteria": {
    "code": "Writing or debugging code", "math": "Mathematics", "chat": "Anything else"}}}, open("one_question.json", "w"))
PY
python3 tools/vela2_bench.py --package VELA_DIR --device cuda:0 --sides runtime --prompts repeated.jsonl \
  --warmup 0 [--questions one_question.json] [--no-graphs] --output ARM.json
```
