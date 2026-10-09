# What the fused kernels and graphs buy on ROCm (MI300X)

The native engine's two GPU fast paths, the bit-exact gfx942 fused Triton
kernels and exact-shape HIP graphs, each switched off on their own and
together, on Decision 2.0 and Vela 2.0. The fused kernels compute what the
eager ops compute bit for bit (`tests/test_gpu_fast_path.py` asserts it for
the padded batches Decision 2.0 runs), so only time changes between arms.

- **Date:** 2026-10-09.
- **Device:** one AMD Instinct MI300X (gfx942).
- **Image:** the packages' release base, `vllm/vllm-openai-rocm@sha256:1fd21abe…`:
  PyTorch 2.12.0+git6bbd260, Triton 3.7.1, FLA 0.5.2 and causal-conv1d 1.7.0
  built for gfx942.
- **Code:** main as of #4734.
- **Prompts:** the first 512 questions of MMLU-Pro's test split (MIT,
  revision `b189ec765aa7`), each with one five-way Choice question about its
  subject. The released typed-final panel of the other records is not public.

`tools/gpu_bench.py` with its default 20 runs and `--count 400 --throughput-count 512
--concurrency 1,16`, once per arm: both on (the default), `--no-fused`,
`--no-graphs`, and both off. Times are p50 in ms; throughput is requests per
second at 16 concurrent requests.

| Model | Arm | One request | 16 questions | 128 questions | exact, 16 clients | batching, 16 clients |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Kai-0.6B | both on | 4.49 | 32.7 | 244.3 | 247.5 | 674.5 |
| Kai-0.6B | no fused | 10.21 | 48.8 | 362.4 | 103.3 | 432.7 |
| Kai-0.6B | no graphs | 14.20 | 33.0 | 243.0 | 72.3 | 673.4 |
| Kai-0.6B | both off | 17.93 | 49.8 | 360.8 | 56.9 | 419.5 |
| Eos-0.8B | both on | 5.71 | 38.9 | 283.0 | 190.1 | 538.3 |
| Eos-0.8B | no fused | 10.74 | 49.2 | 351.0 | 99.6 | 427.9 |
| Eos-0.8B | no graphs | 25.46 | 39.1 | 284.8 | 40.0 | 525.7 |
| Eos-0.8B | both off | 31.88 | 50.1 | 347.2 | 31.4 | 417.1 |
| Sol-2B | both on | 7.00 | 58.5 | 432.0 | 152.3 | 359.3 |
| Sol-2B | no fused | 11.27 | 71.7 | 528.4 | 91.7 | 287.8 |
| Sol-2B | no graphs | 24.54 | 58.6 | 432.4 | 40.6 | 356.9 |
| Sol-2B | both off | 31.18 | 72.2 | 532.4 | 32.5 | 283.7 |
| Nox-4B | both on | 12.34 | 130.2 | 991.3 | 84.7 | 160.6 |
| Nox-4B | no fused | 20.12 | 152.4 | 1,179.9 | 51.2 | 133.8 |
| Nox-4B | no graphs | 33.33 | 128.1 | 982.9 | 30.6 | 158.9 |
| Nox-4B | both off | 41.67 | 152.0 | 1,176.8 | 24.2 | 133.3 |
| Lux-9B | both on | 16.87 | 200.0 | 1,507.1 | 60.9 | 101.9 |
| Lux-9B | no fused | 22.87 | 231.2 | 1,782.6 | 44.4 | 87.8 |
| Lux-9B | no graphs | 33.39 | 199.6 | 1,508.8 | 30.8 | 101.6 |
| Lux-9B | both off | 40.84 | 232.3 | 1,784.1 | 24.7 | 86.1 |

Per-arm numbers, with the Vela 2.0 p95s: `rocm-mi300x-fused.json`.

## Decision 2.0 findings

- **The fused kernels still pay with graphs on.** Graphs remove launch
  overhead, but the fused kernels also remove memory traffic: with graphs on,
  turning them off makes one request 36% (Lux) to 127% (Kai) slower, and a
  128-question request 18% (Lux) to 48% (Kai) slower. Exact throughput at 16
  clients falls by 27% (Lux) to 58% (Kai).
- **Graphs pay only for small shapes.** They turn a single request from 2.0x
  (Lux) to 4.5x (Eos) faster, and leave the 16- and 128-question requests
  within 2%, since those shapes (about 5,400 and 43,000 tokens) are above the
  4,096-token graph cap and run eagerly either way.
- **Together** they make one request 2.4x (Lux) to 5.6x (Eos) faster than the
  eager path.
- **Batching** (16 clients) keeps the fused kernels' gain (16% to 56% more
  requests per second) and at most 2.4% from graphs, since coalesced batches
  rarely match a captured shape (Kai: 7 replays against 25 eager forwards).

## Vela 2.0

`tools/vela2_bench.py --sides runtime --tokens 128,512,2048 --requests 40`
with the same four switches (`--no-fused`, `--no-graphs`), one client: the
router's signals as one Vela 2.0 request over a prompt of about N tokens.
p50 in ms.

| Model | Arm | 128 tokens | 512 tokens | 2,048 tokens |
| --- | --- | ---: | ---: | ---: |
| 0.3B | both on | 9.05 | 13.12 | 28.85 |
| 0.3B | no fused | 8.99 | 13.09 | 28.46 |
| 0.3B | no graphs | 9.29 | 13.18 | 28.75 |
| 0.3B | both off | 9.64 | 13.24 | 28.82 |
| 0.8B | both on | 62.49 | 67.51 | 142.88 |
| 0.8B | no fused | 65.90 | 79.70 | 164.95 |
| 0.8B | no graphs | 62.83 | 64.57 | 139.47 |
| 0.8B | both off | 65.96 | 79.45 | 164.10 |
| 4B | both on | 112.77 | 180.63 | 368.17 |
| 4B | no fused | 139.97 | 215.86 | 432.52 |
| 4B | no graphs | 111.77 | 178.92 | 364.31 |
| 4B | both off | 139.46 | 215.25 | 432.10 |
| 9B | both on | 166.12 | 260.87 | 526.60 |
| 9B | no fused | 194.69 | 300.29 | 599.54 |
| 9B | no graphs | 165.30 | 260.79 | 525.94 |
| 9B | both off | 194.81 | 300.05 | 600.29 |

- **Vela 2.0 never replays a graph.** Every arm's receipt shows no capture and
  no replay, on every size, so the graph switch changes nothing and a small
  request costs what an eager forward costs: 112.77 ms for the 4B's 128-token
  request.
- **The fused kernels pay on the decoders.** Turning them off makes the 0.8B,
  4B and 9B 5% to 24% slower. The 0.3B encoder uses only `rotary_half`, and
  its times stay within 1%.

This is the ROCm baseline for porting the kernels to other devices: CUDA
(#4621) and the approximate `max_speed` kernels of `docs/design.md`.

## Reproduce

```bash
python3 - <<'PY'
import json, pyarrow.parquet as pq
# hf download TIGER-Lab/MMLU-Pro --repo-type dataset --revision b189ec765aa7ed75c8acfea42df31fdae71f97be
rows = pq.read_table("data/test-00000-of-00001.parquet", columns=["question"]).column("question").to_pylist()[:512]
q = {"route": {"type": "choice", "instructions": "Which subject is this question about?", "criteria": {
    "stem": "Science, technology, engineering or mathematics", "humanities": "History, philosophy, law or other humanities",
    "social": "Economics, business, psychology or other social sciences", "health": "Health or medicine", "other": "Anything else"}}}
with open("mmlu_pro_prompts.jsonl", "w") as f:
    for i, text in enumerate(rows):
        f.write(json.dumps({"id": f"mmlu-pro-{i}", "state": text, "questions": q}) + "\n")
PY
python3 tools/gpu_bench.py --package PACKAGE_DIR --prompts mmlu_pro_prompts.jsonl --count 400 \
  --throughput-count 512 --concurrency 1,16 [--no-fused] [--no-graphs] --output ARM.json
python3 tools/vela2_bench.py --package VELA_DIR --device rocm:0 --sides runtime \
  --tokens 128,512,2048 --requests 40 [--no-fused] [--no-graphs] --output ARM.json
```
