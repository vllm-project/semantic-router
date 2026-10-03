# vLLM Semantic Router model runtime

`vllm_sr_runtime` serves typed decision models behind one HTTP contract. The
router manages it for `model_runtime` deployments, and `vllm-sr serve
<hf-model>` runs it on its own.

```bash
pip install -e "src/model-runtime[test]"
vllm-sr-runtime serve vllm-sr/Decision-2.0-Kai-0.6B --device cpu --port 8100
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
| `POST /v1/decisions`, `POST /v1/systemone` | Choice, Noul and Score answers (a superset of System One) |
| `GET /v1/models` | Identity, limits, placement, profiles, plugins and golden-check status |
| `GET /health`, `GET /health/live` | Readiness (gated on golden answers) and liveness |
| `GET /metrics` | Prometheus metrics |

The contract is [`vllm_sr_runtime/api/openapi.yaml`](vllm_sr_runtime/api/openapi.yaml)
and the design is [`docs/design.md`](docs/design.md).

## Built-in models

`vllm-sr-runtime models` lists the six Decision 2.0 models with their pinned
revisions. Every package is verified against its manifest before load, and
code shipped inside packages is never executed.

## Tests

```bash
make model-runtime-test
```

The tests generate tiny random-weight Qwen3 and Qwen3.5 packages, check the
native backbones bit for bit against the Transformers reference on CPU,
validate every response against the OpenAPI contract and run the server over
a Unix socket and TCP. GPU tests are marked `gpu` and skip on CPU hosts.
