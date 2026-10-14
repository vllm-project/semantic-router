---
title: Engine arguments
description: Tune batching, precision and shared context with vLLM-style engine arguments.
---

# Engine arguments

The model runtime batches by default, like vLLM: rows of concurrent requests
share forward passes, so a busy deployment answers more requests per second.
Engine arguments tune that batching, the precision the model computes in and
shared context. Set them per deployment; every argument you leave out keeps the
runtime's default.

| Argument | Deployment field | Default | Meaning |
| --- | --- | --- | --- |
| `--max-num-seqs` | `max_num_seqs` | `8192` | Most rows one forward holds, across requests. `1` runs every row alone. |
| `--max-num-batched-tokens` | `max_num_batched_tokens` | `65536` | Padded tokens one forward may hold: its rows times its longest row. |
| `--batch-window-ms` | `batch_window_ms` | `2.0` | How long the scheduler waits after the first request to collect more. |
| `--dtype` | `dtype` | `auto` | Precision the model computes in: `auto`, `float32`, `bfloat16` or `float16`. |
| `--enable-shared-context`, `--no-enable-shared-context` | `enable_shared_context` | the model family's | Compute the shared state of a multi-question decision request once. |

## Batching

Every row of every queued request is planned together, longest first: a
decision request takes one row per question (a request with 10 questions
takes 10 rows), a classify, embeddings or rerank request one per input or
window. A forward holds at most `--max-num-seqs` rows and
`--max-num-batched-tokens` padded tokens; a row longer than the token budget
runs alone. This is vLLM's batching with one difference in what a row is: the
runtime's models answer a row in one forward, where vLLM's sequences take one
decode step after another.

- **`--max-num-seqs`** counts rows, as vLLM's `--max-num-seqs` counts the
  sequences of one step. The default never binds at the default token budget:
  a row costs at least 8 padded tokens, so a 65,536-token forward holds at most
  8,192 rows anyway. Lower it to keep forwards small; `1` runs every row in a
  forward of its own.
- **`--max-num-batched-tokens`** counts padded tokens: the rows of a forward
  times its longest row, rounded up to a multiple of 8. It counts them so also
  for models that lay rows back to back without padding (the Vela 1.0 task
  models on the native engine), for which it overstates what a forward
  computes. Some families keep forwards shorter on their own: Decision 3.0
  sizes a merged forward to about 0.3 s of device time at the measured cost per
  token, and Vela 2.0 0.3B and the Decision 1.0 encoder sizes stay within 2,048
  and 1,024 padded tokens on a CPU.
- **`--batch-window-ms`** is how long the scheduler waits after the first
  request arrives before it plans the queue. vLLM has no such setting: a
  generation request runs one decode step after another, and every step is a
  point where new requests join the running batch. Here a request is one
  forward, so without a window the first request of a burst would run alone
  while the rest queue behind it. A lone request pays the window; `0` plans at
  once and still merges the requests that queue while a forward runs.

A model whose rows answer the same alone or in any batch, such as a Vela 1.0
task model or embedder on a CPU with oneDNN, merges the requests already
queued without waiting for the window, and keeps those forwards within 512
padded tokens so a short request never waits behind a long one. Its answers
never change.

For every other model, the rows beside a row change its padding and GEMM
shapes, so an answer can move by rounding and a near-tie can flip. The
records measure this against the noise the model's own batches already have:

- Decision 2.0 on ROCm: merged rows changed as many decisions as asking the
  same questions alone instead of together does (for example 16 against 18 of
  2,000 for Kai-0.6B), and the Decision Index delta's 95% interval contains 0,
  though Eos-0.8B's estimate leans negative
  ([record](https://github.com/vllm-project/semantic-router/blob/main/src/model-runtime/docs/records/rocm-mi325x-batching.md)).
- Decision 3.0: 99.69% of d3-edge decisions agree with one request per
  forward, and gold accuracy moved from 75.17% to 75.18%; sending a request's
  questions one by one instead of together changes as many
  ([record](https://github.com/vllm-project/semantic-router/blob/main/src/model-runtime/docs/records/decision3-batching.md)).

Responses that ask for `meta` (`"options": {"return_meta": true}`) carry
`numerics`: `exact` for a model whose answers never depend on batching,
`approximate` otherwise. `GET /v1/models` reports each model's `engine_args`
and `numerics`.

## Precision

`--dtype auto` runs a model in the precision it was published in, which is the
precision its readiness check was recorded at:

| Models | `auto` on a GPU | `auto` on a CPU |
| --- | --- | --- |
| Decision 3.0 | BF16 | BF16 |
| Decision 2.0; Decision 1.0 and Vela 2.0 on Qwen3.5 | BF16 | FP32 |
| Vela 1.0 task models, embedders and rerankers, Vela 1.0 Omni, Vela 2.0 0.3B, Decision 1.0 Kai, Lex and Route | FP32 | FP32 |

Another value holds every weight of the backbone in that dtype and computes in
it, with the reference kernels. A value that names the published precision is
the same as `auto`. The runtime still checks at startup that the model answers
its recorded requests the same way twice, but the recorded references apply
only to `auto`, so the model's `golden.status` is `unverified`. `float16` runs
on GPUs only, and the ONNX Runtime engine runs only `auto`.

## Shared context

When a decision request asks many questions about one input, shared context
runs the input once and every question from it, as one packed tree: up to
3.4× faster for 128 questions about one input on Vega-27B. A request shares
only when the tokens it saves outweigh the extra launches on its backbone, so
a request with one or two short questions runs as usual.

Like vLLM's `--enable-prefix-caching`, it is on by default where it is
accuracy-neutral and a switch elsewhere:

- **Decision 2.0: on.** The trees answer exactly as the released runtime's own
  shared-context switch, whose accuracy on every panel stayed within its 95%
  interval of 0 ([record](https://github.com/vllm-project/semantic-router/blob/main/src/model-runtime/docs/records/rocm-mi325x-batching.md)).
- **Decision 1.0 and Vela 2.0 decoder sizes: off.** Their records show
  decisions and span sets changing, so `--enable-shared-context` is an
  explicit choice there.
- Encoders read a request whole and have no shared-context path.

## Faster kernels at the published precision

Where a model's records show that no decision changed, it runs faster kernels
by default, at its published precision:

- **NVIDIA GPUs from Ampere on, with Triton:** Decision 2.0 runs fused
  element-wise kernels, 11% to 48% faster with no decision changed in its
  [record](https://github.com/vllm-project/semantic-router/blob/main/src/model-runtime/docs/records/cuda-fused-approximate.md).
  Vela 2.0 keeps the reference kernels, because they change some of its span
  answers.
- **x86 CPUs with oneDNN:** Vela 2.0 0.3B and Decision 1.0 Kai, Lex and Route
  run merged forwards on pre-packed FP32 copies of their linear layers,
  1.5–1.7× faster with no decision changed. The copy sits next to the weights the
  readiness check runs: 1.32 GB for the three layer stacks of one Decision 1.0
  encoder.

## Set them

For a router-managed model, set the fields on the deployment:

```yaml
global:
  model_catalog:
    deployments:
      decision-lux:
        provider: model_runtime
        artifact: vllm-sr/Decision-2.0-Lux-9B
        device: rocm:0
        max_num_seqs: 256
        batch_window_ms: 1.0
```

`vllm-sr serve` sets them on its default deployment, and `vllm-srun serve`
takes them as flags for every model it serves (a [models file](./reference.md#models-file)
entry can set its own):

```bash
vllm-sr serve vllm-sr/Decision-2.0-Lux-9B --engine --platform rocm --device-ids 0 --max-num-seqs 256
vllm-srun serve vllm-sr/Vela-2.0-0.3B --device cpu --batch-window-ms 0
```

A configuration that still names a removed setting is refused with the
command that rewrites it; see [Migrate](./migrate.md#engine-arguments).

## Reproduce published results

The runtime is built for throughput. Reproducing a model's published numbers is
the job of the model's own repository, which ships the reference code the
numbers were measured with. `--max-num-seqs 1` removes the effect of other
requests, but runs every row alone, which is not how the released runtimes
batched a request's questions. An opt-in deterministic mode, which would give
every row the same answer in any batch, is possible future work.

## Performance without changing answers

These never trade correctness for speed:

- compatible work for the same model and routing stage can share a bundled
  call; one bundle may still require several forwards;
- independent workers can execute in parallel when hardware capacity allows;
  placing multiple workers on one GPU does not add compute capacity;
- cacheable repeated work can reuse recent results while they remain in the
  worker's bounded result cache (`--result-cache-entries`); replicas have
  independent worker caches;
- Decision 3.0 models decode and preprocess a repeated image or video once;
- long inputs are split into windows only when you ask for windowing.
