# Example out-of-tree plugin

A complete, minimal plugin for the vllm-sr model runtime, kept small enough to
read in one sitting. It adds three plugins through Python entry points, the
same way the built-in families, engines and accelerators register:

| Entry point | Name | What it does |
| --- | --- | --- |
| `vllm_srun.families` | `example_keywords` | Reads a keyword package, renders text into keyword IDs, and reads label distributions, embeddings and relevance scores out of the engine's hidden states |
| `vllm_srun.engines` | `example_counts` | Turns keyword IDs into one-hot label vectors on the CPU |
| `vllm_srun.accelerators` | `example_host` | Offers the host CPU as its own device (`--device example_host`); `--device auto` never picks it |

The family never runs the backbone and the engine never reads the package:
that split is what lets a third-party family reuse a built-in engine, or a
third-party engine serve a built-in family. Batching is the runtime's: it
batches the example's requests as it batches every model's, as the engine
arguments (`--max-num-seqs`, `--max-num-batched-tokens`, `--batch-window-ms`)
say.

## Try it

```bash
pip install ./src/model-runtime ./src/model-runtime/examples/third_party_plugin
mkdir -p /tmp/keywords && cat > /tmp/keywords/example_model.json <<'JSON'
{"format": "vllm-sr-example/1", "labels": ["billing", "shipping", "other"],
 "keywords": {"billing": ["refund", "invoice", "charge"], "shipping": ["parcel", "delivery"]}}
JSON
vllm-srun serve /tmp/keywords --engine example_counts --device cpu --port 8100
```

```bash
curl -s localhost:8100/v1/classify -H 'content-type: application/json' \
  -d '{"input": ["Please refund the invoice", "Where is my parcel?"]}'
curl -s localhost:8100/v1/models | jq '.data[0].plugins[] | select(.name | startswith("example"))'
```

`/v1/models` lists the three plugins with their capability descriptors. To run
the same model on the example accelerator, every input in a forward of its
own:

```bash
vllm-srun serve /tmp/keywords --engine example_counts --device example_host \
  --max-num-seqs 1 --port 8100
```

## Write your own

1. Subclass `ModelFamily` (`detect`, `verify`, `describe`, `load`) and
   `LoadedModel` (`plan_surface`, `run`, `finish_surface`) from
   `vllm_srun.plugins.base`, or `DecisionModel` (`plan`, `run`,
   `answer`) from `vllm_srun.plugins.decisions` for `/v1/decisions`.
   Declare the surfaces you serve and a `descriptor()`; name a table of pinned
   models in `builtin_table` and a test-package writer in `fixture_writer` if
   you ship them.
2. Reuse a built-in engine through `ModelSpec`, or subclass `Engine` and
   `EngineModel` (`supports`, `load`, `forward` or `encode`); set
   `auto_priority` if `--engine auto` may try it before the others, and build
   `descriptor()` on `super().descriptor()`, which lists it. An engine runs
   `--dtype auto` only unless it lists more values in `dtypes`.
3. For new hardware, subclass `Accelerator` (`available`, `devices`,
   `torch_device`, `kernels`); set `auto_priority` if `--device auto` may pick
   it.
4. Register each under its entry-point group in your `pyproject.toml`.
5. Set `cache_key` on work items whose result depends only on their content,
   so repeated inputs are answered from the result cache, and set
   `batch_invariant` on the loaded model once a test shows that every row
   answers the same alone or in any batch: queued requests then merge without
   waiting for the batching window.

The runtime never imports code shipped inside a model package; plugins are
installed code. The runtime's tests build this example's wheel, install it
into a fresh directory and discover it through its entry points
(`tests/test_third_party_plugin.py`).
