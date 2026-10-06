# vLLM Semantic Router model runtime

`vllm_srun` serves the router's models (decision models, classifiers,
embedders and rerankers) behind one HTTP contract. One process can serve
several models. The router manages it for `model_runtime` deployments, and
`vllm-sr serve <hf-model> [<hf-model> ...]` runs it on its own.

```bash
pip install -e "src/model-runtime[test]"
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
| `POST /v1/decisions`, `POST /v1/systemone` | Choice, Noul and Score answers (a superset of System One), plus Set and Span where a model declares them |
| `POST /v1/classify` | Fixed heads: label distributions, label scores and token spans over texts, pairs or grounded answers |
| `POST /v1/embeddings` | OpenAI-compatible embeddings with dimensions and layer exits |
| `POST /v1/rerank` | Pair scores of documents against a query |
| `POST /v1/bundle` | Several surface requests, for one or more models, in one call |
| `GET /v1/models` | Identity, surfaces, heads, limits, placement, profiles, plugins and golden-check status per model |
| `GET /health`, `GET /health/live` | Readiness (gated on golden answers) and liveness |
| `GET /metrics` | Prometheus metrics |

The contract is [`vllm_srun/api/openapi.yaml`](vllm_srun/api/openapi.yaml)
and the design is [`docs/design.md`](docs/design.md).

## Built-in models

`vllm-srun models` lists the built-in models with their pinned
revisions. Every package is verified against its manifest or the pinned file
digests before load, and code shipped inside packages is never executed.

## Plugins

Families, engines, accelerators and profiles are entry-point plugins.
[`examples/third_party_plugin`](examples/third_party_plugin) is a complete
out-of-tree family and engine to start from.

## Tests

```bash
make model-runtime-test
```

The tests generate tiny random-weight Qwen3 and Qwen3.5 packages, check the
native backbones bit for bit against the Transformers reference on CPU,
validate every response against the OpenAPI contract and run the server over
a Unix socket and TCP. GPU tests are marked `gpu` and skip on CPU hosts.
