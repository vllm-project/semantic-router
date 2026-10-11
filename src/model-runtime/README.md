# vLLM Semantic Router model runtime

`vllm-srun` serves the router's models (decision models, classifiers,
embedders and rerankers) behind one HTTP contract. One process can serve
several models. The router manages it for `model_runtime` deployments.
Use `vllm-sr serve ARTIFACT --engine` for a managed instance with a Dashboard
and public System One API. Use `vllm-srun serve` to operate a worker directly,
including its classify, embeddings, rerank and bundle APIs.

Every router image ships it; it is not published to PyPI. To run it on your
own machine, install it from a checkout of the repository, after PyTorch from
the [index that matches your hardware](https://pytorch.org/get-started/locally/)
(CPU, CUDA or ROCm). On ROCm, answers byte-identical to the released model
packages are guaranteed in the router images, which carry the release's own
PyTorch build.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install ./src/model-runtime
vllm-srun serve vllm-sr/Decision-2.0-Kai-0.6B --device cpu --port 8100
curl -s localhost:8100/v1/decisions -H 'content-type: application/json' -d '{
  "state": "Write a Python function that merges two sorted lists.",
  "questions": {
    "domain": {"type": "choice", "instructions": "Which domain is this?",
               "criteria": {"code": "Programming", "math": "Mathematics", "other": "Anything else"}},
    "reasoning": {"type": "noul", "instructions": "Does this need multi-step reasoning?"}
  }
}'
```

| Endpoint | Purpose |
| --- | --- |
| `POST /v1/decisions`, `POST /v1/systemone` | Choice, Noul and Score answers (a superset of System One), plus Set and Span where a model declares them, and `images` and `videos` for models that read them (Decision 3.0) |
| `POST /v1/classify` | Fixed heads: label distributions, label scores and token spans over texts, pairs or grounded answers |
| `POST /v1/embeddings` | OpenAI-compatible embeddings with dimensions and layer exits |
| `POST /v1/rerank` | Pair scores of documents against a query |
| `POST /v1/bundle` | Several surface requests, for one or more models, in one call |
| `GET /v1/models` | Identity, surfaces, heads, limits, placement, profiles, plugins and golden-check status per model |
| `GET /health`, `GET /health/live` | Readiness (gated on golden answers) and liveness |
| `GET /metrics` | Prometheus metrics |

The contract is
[`vllm_srun/api/openapi.yaml`](https://github.com/vllm-project/semantic-router/blob/main/src/model-runtime/vllm_srun/api/openapi.yaml)
and the design is
[`docs/design.md`](https://github.com/vllm-project/semantic-router/blob/main/src/model-runtime/docs/design.md).

## Complete input coverage

Native questions retain their model's ordinary input-fitting behavior by
default. An accepted request or its token usage does not prove that every
supplied part was read. Set `require_full_input: true` on a question when that
guarantee is required. It covers **all** supplied state parts, including context
outside `over`, and cannot be combined with `overflow: truncate`.

A strict question succeeds only when every part fits or its supported windows
cover all tokens and every scored word completely. Otherwise the question
returns `max_length_exceeded` (or `scan_budget_exceeded` beyond the scan limit).
Success includes `input_coverage: "complete"` on its answer. Native Set reports
the proof on `sets.<id>` and every `answers.<id>.<label>` answer; Span reports it
on `answers.<id>`. Errors and ordinary questions omit this proof.

Router tasks that require full input consume this proof before accepting an
answer. Missing proof, including from an older attached worker, is an unknown
result, never evidence that content is clean or grounded. Composed Set requires
the proof from every constituent Noul question. Complete coverage describes
what was read; it does not establish prediction accuracy.

## Built-in models

`vllm-srun models` lists the built-in models with their pinned
revisions. Every package is verified against its manifest or the pinned file
digests before load, and code shipped inside packages is never executed.

## Plugins

Families, engines, accelerators and profiles are entry-point plugins.
[`examples/third_party_plugin`](https://github.com/vllm-project/semantic-router/tree/main/src/model-runtime/examples/third_party_plugin)
is a complete out-of-tree family and engine to start from.

## Development and tests

From a repository checkout, `make model-runtime-install` installs the runtime
in editable mode with CPU PyTorch, every engine and the test extras.

```bash
make model-runtime-install
make model-runtime-test
```

The tests generate tiny random-weight Qwen3 and Qwen3.5 packages, check the
native backbones bit for bit against the Transformers reference on CPU,
validate every response against the OpenAPI contract and run the server over
a Unix socket and TCP. GPU tests are marked `gpu` and skip on CPU hosts.
