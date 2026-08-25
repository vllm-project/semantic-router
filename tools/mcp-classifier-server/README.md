# MCP Classifier Examples

This directory contains three standalone MCP servers that implement the domain
classifier contract used by semantic-router. They are examples for local
development and integration testing, not prequalified production services.

Each server exposes:

- `list_categories`, which returns category names and optional descriptions and
  system prompts;
- `classify_text`, which returns a class index, confidence, and optional routing
  hints.

All three support stdio and streamable HTTP. The HTTP endpoint is `/mcp`; the
health endpoint is `/health`.

## Choose an Example

| Server | Method | Extra state | Good for |
|---|---|---|---|
| `server_keyword.py` | regular-expression rules | none | verifying the MCP contract and deterministic rules |
| `server_embedding.py` | nearest examples using Qwen3 embeddings | `training_data.csv` and a Milvus Lite file | experimenting with example-based semantic classification |
| `server_generative.py` | fine-tuned Qwen3 classifier | a local or Hugging Face LoRA checkpoint | evaluating a trained generative classifier |

Choose with measured workload results. The implementation type alone does not
establish accuracy, latency, or memory use.

## Keyword Server

```bash
pip install -r requirements.txt
python server_keyword.py --http --port 8090
curl http://localhost:8090/health
```

Edit `CATEGORIES` in `server_keyword.py` for local rules. `routing_policy.py`
owns the optional backend-model and reasoning hints returned with a class.

## Embedding Server

```bash
pip install -r requirements_embedding.txt
python server_embedding.py \
  --http \
  --port 8091 \
  --device cpu \
  --milvus-uri ./milvus_data.db
```

The server loads `training_data.csv` beside the script and materializes its
embeddings in Milvus Lite. Review the examples, category balance, and database
path before use. Delete or rebuild the database when changing an incompatible
embedding model or training set.

## Generative Server

```bash
pip install -r requirements_generative.txt
python server_generative.py \
  --http \
  --port 8092 \
  --model-path ORGANIZATION/CLASSIFIER_ADAPTER \
  --base-model Qwen/Qwen3-0.6B \
  --device auto
```

`--model-path` accepts a local adapter directory or Hugging Face model ID. The
base model must match the adapter. Validate the label mapping and held-out
metrics supplied with that artifact.

## Connect Semantic Router

Disable the local domain classifier when MCP should be the domain source, then
configure the MCP module:

```yaml
version: v0.3
providers:
  models:
    - name: local/gpt-oss-20b
      provider_model_id: openai/gpt-oss-20b
      backend_refs:
        - provider: vllm
          endpoint: http://127.0.0.1:8000/v1
      control:
        retry: {count: 2, on: [unavailable, timeout]}
        timeout: {request: 60s, stream: 10m}
routing:
  modelCards:
    - name: local/gpt-oss-20b
      capabilities: [chat]
recipes:
  - name: mcp-domain
    routing:
      decisions:
        - name: route
          rules: {}
entrypoints:
  - model_names: [vllm-sr/mcp-domain]
    recipe: mcp-domain
    assignments:
      route:
        models: [{model: local/gpt-oss-20b}]

global:
  model_catalog:
    modules:
      classifier:
        domain:
          enabled: false
        mcp:
          enabled: true
          transport_type: streamable-http
          url: http://localhost:8090/mcp
          tool_name: classify_text
          threshold: 0.6
          timeout_seconds: 30
```

Configure the Model connection, model-free Recipe, and Entrypoint assignments
in the normal Router config. The MCP server supplies classification; it does
not replace the rest of the routing policy. See the
[domain signal guide](../../website/docs/tutorials/signal/learned/domain.md)
for the user-facing configuration model.

## Validate

At minimum, check:

- `/health` and MCP initialization;
- category ordering returned by `list_categories`;
- `classify_text` output for positive, negative, ambiguous, and empty inputs;
- timeout and unavailable-server behavior in the router;
- threshold behavior against a labelled, held-out set.

Do not expose these example servers to an untrusted network without adding the
authentication, transport security, resource controls, and observability
required by your environment.
