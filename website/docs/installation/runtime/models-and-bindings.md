---
title: Models and Bindings
description: Declare model resources, assign recipe tasks, and provision immutable model artifacts.
---

# Models and bindings

Use a deployment to describe **where a Router model runs**, and a binding to
specify **what one recipe asks it to do**. A downstream answer model remains a
[`providers.models`](../model-configuration.md) entry.

## Configuration responsibilities

| Surface | Responsibility |
| --- | --- |
| `global.model_catalog.system` | Stable task aliases for the maintained model defaults |
| `global.model_catalog.embeddings` | Embedding catalog, primary model, and embedding options |
| `global.model_catalog.modules` | Consumer enablement, thresholds, mapping defaults, and existing backend settings |
| `global.model_catalog.external` | Named external services and their protocol-related settings |
| `global.model_catalog.deployments` | Local artifact/provider/device/precision/input budget, or external service selection |
| `global.model_catalog.admission` | Capacity and queue policy for the selected deployment/resource |
| `routing.model_bindings` | Recipe-local task contract, adapter, head, and label mapping |

An explicit binding replaces the selected consumer's execution defaults inside
that recipe. It does not change another recipe or require copying all catalog
defaults. A rule also needs to be enabled and reachable for its model to be
prepared. The default public diagnostic API has its own prepared consumers.
Unused named deployments do not trigger downloads.

## Declare one local deployment

Merge this fragment into a canonical v0.3 config with your existing providers
and routing policy:

```yaml
global:
  model_catalog:
    deployments:
      local-pii:
        artifact: models/mmbert32k-pii-detector-merged
        provider: candle
        device: cpu
        precision: native
        input:
          max_tokens: 512
          overflow: reject
    admission:
      local-pii:
        max_concurrency: 2
        max_queue: 8
        queue_timeout_ms: 1000
        on_overflow: shed
routing:
  model_bindings:
    pii_classifier:
      deployment: local-pii
      contract: token_spans.v1
      adapter: mmbert32k
      mapping_path: models/mmbert32k-pii-detector-merged/pii_type_mapping.json
```

The [in-process guide](in-process.md) includes a complete configuration and
request. The [external guide](external.md) shows a named HTTP deployment.

## Fields and compatibility

| Field | Meaning |
| --- | --- |
| `artifact` | Local artifact path, resolved using Router model path rules; required for Candle/ORT |
| `revision` | Snapshot revision for a registered artifact's download |
| `external_model` | Exact `external[].name`; required for HTTP, mutually exclusive with `artifact` |
| `provider` | `candle`, `ort`, or `http` |
| `device`, `precision` | Local execution request; omit for HTTP |
| `input.max_tokens` | Additional token budget; zero/omitted retains the task/model limit except for owned MIGraphX mmBERT embeddings, which require a positive explicit budget |
| `input.overflow` | Requested `reject`, `truncate`, or `window`; the adapter must implement it |
| `contract` | Required result shape for the bound consumer |
| `adapter` | Native architecture adapter or external wire protocol |
| `head` | Separate compatible Candle head directory, or a complete ORT `.onnx` graph |
| `mapping_path` | Task label mapping when that consumer uses a mapping file |

Candle head directories are relative to the Router working directory unless
absolute. ORT graph paths are relative to the artifact directory unless
absolute. An ORT “head” is a complete graph, not a head tensor layered over a
shared encoder. HTTP bindings cannot use local heads. Generic classifier rules
use their explicit ordered `labels`; they reject `mapping_path`.

Current sequence/token adapters support reject or truncate, not automatic
windowed classification. The schema's `window` value does not enable an adapter
that lacks it. Native classification rejects budgets above **512**, even for a
32K-named artifact. HTTP classifier bindings cannot enforce local tokenizer
budgets; leave their `input` unset and configure the service's own limits.

## Task binding names

| Consumer name | Result contract |
| --- | --- |
| `domain_classifier`, `fact_check_classifier`, `feedback_detector`, `modality_detector` | `label_distribution.v1` |
| `prompt_guard` | `label_distribution.v1`; `label_decision.v1` with HTTP chat |
| `pii_classifier`, `hallucination_detector` | `token_spans.v1` |
| `hallucination_explainer` | `text_pair_distribution.v1` |
| `embedding` | `embedding.v1` |
| `complexity` | `score.v1` or `label_distribution.v1`, using an HTTP scorer |
| `classifier.<rule name>` | `label_distribution.v1` with that rule's supported execution type |

The [engine matrix](engines-and-hardware.md#tasks-by-engine) lists which
providers implement each task. Declaring a result contract does not add a
missing provider adapter.

A generic binding must name an existing rule in the same recipe. Names may
contain dots: `classifier.billing.priority` binds the rule named
`billing.priority`. Multiple local rules may use independent models in one
recipe. They require at least two explicit labels; multiclass distributions
are preserved. An LLM rule also retains its instructions and scored JSON
extraction contract.

For a named recipe, put bindings inside that recipe's `routing` block:

```yaml
recipes:
  - name: private-analysis
    routing:
      model_bindings:
        pii_classifier:
          deployment: local-pii
          contract: token_spans.v1
          adapter: mmbert32k
```

Entrypoints and decision references determine whether the recipe is reachable.
See [Models, entrypoints, and serving](../../tutorials/global/models-entrypoints-serving.md)
for the complete recipe structure.

## Provision model files

`mom_registry` associates maintained local paths with Hugging Face repositories
for automatic provisioning. The downloader follows actual default API and
reachable-recipe consumers, including bound artifacts, required heads,
mappings, and provider-specific graphs. It does not download an obsolete
module default that an active explicit binding replaces.

Custom mounted paths do **not** need registry registration. Mount the complete
artifact and let provider preparation validate its files, shapes, labels, and
capabilities. A directory's name or a receipt file is not a compatibility gate.

For repeatable registered downloads, set `revision` to a full commit hash and
use a separate local directory for each simultaneously usable revision. A
complete pinned snapshot with matching standard Hugging Face cache metadata
can be reused offline. A populated directory with a different or unverifiable
revision is rejected before download; choose an empty directory instead of
modifying files that a live or retired generation may still use.

Candle needs the actual checkpoint/config/tokenizer and compatible head data.
ORT needs its selected graph, tokenizer/config files, and the external tensor
files named by that graph. Inline ONNX tensors do not require a fabricated
`.data` file. One artifact directory can retain both Candle and ORT formats.
A separately registered head or mapping repository retains its own default
revision; the deployment revision pins the artifact repository itself.

Sequence and PII preparation checks label identity and order, not only label
count. An all-index vocabulary (`LABEL_0`, `LABEL_1`, …) may use its explicitly
configured index mapping. Semantic labels are not silently renamed, except for
the existing documented aliases of a particular task.

## Validate across authoring tools

```bash
vllm-sr config validate --config config.yaml
```

The Dashboard and generated schema carry the same fields. DSL places
`model_bindings` in the recipe's `ROUTING` block while global deployments stay
in YAML. Helm `configOverride` accepts the complete canonical document. The
Operator maps `spec.config.model_deployments` and `model_admission` to the
catalog and carries `spec.config.routing.model_bindings` into routing.
Validation proves a declaration is coherent; startup proves its actual model
and engine can be prepared.
