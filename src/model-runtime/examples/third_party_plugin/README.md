# Example out-of-tree plugin

A complete, minimal plugin for the vllm-sr model runtime, kept small enough to
read in one sitting. It adds two plugins through Python entry points, the same
way the built-in families and engines register:

| Entry point | Name | What it does |
| --- | --- | --- |
| `vllm_sr_runtime.families` | `example_keywords` | Reads a keyword package, renders text into keyword IDs, and reads label distributions, embeddings and relevance scores out of the engine's hidden states |
| `vllm_sr_runtime.engines` | `example_counts` | Turns keyword IDs into one-hot label vectors on the CPU |

The family never runs the backbone and the engine never reads the package:
that split is what lets a third-party family reuse a built-in engine, or a
third-party engine serve a built-in family.

## Try it

```bash
pip install ./src/model-runtime ./src/model-runtime/examples/third_party_plugin
mkdir -p /tmp/keywords && cat > /tmp/keywords/example_model.json <<'JSON'
{"format": "vllm-sr-example/1", "labels": ["billing", "shipping", "other"],
 "keywords": {"billing": ["refund", "invoice", "charge"], "shipping": ["parcel", "delivery"]}}
JSON
vllm-sr-runtime serve /tmp/keywords --engine example_counts --device cpu --port 8100
```

```bash
curl -s localhost:8100/v1/classify -H 'content-type: application/json' \
  -d '{"input": ["Please refund the invoice", "Where is my parcel?"]}'
curl -s localhost:8100/v1/models | jq '.data[0].plugins[] | select(.name | startswith("example"))'
```

`/v1/models` lists both plugins with their capability descriptors.

## Write your own

1. Subclass `ModelFamily` (`detect`, `verify`, `describe`, `load`) and
   `LoadedModel` (`plan_surface`, `run`, `finish_surface`) from
   `vllm_sr_runtime.plugins.base`. Declare the surfaces you serve and a
   `descriptor()`.
2. Reuse a built-in engine through `ModelSpec`, or subclass `Engine` and
   `EngineModel` (`supports`, `load`, `forward` or `encode`).
3. Register both under the entry-point groups in your `pyproject.toml`.
4. Set `cache_key` on work items whose result depends only on their content,
   so repeated inputs are answered from the result cache, and set
   `fuse_bundled_jobs` when one forward may serve several bundled requests.

The runtime never imports code shipped inside a model package; plugins are
installed code. The runtime's tests install this example through its entry
points (`tests/test_third_party_plugin.py`).
