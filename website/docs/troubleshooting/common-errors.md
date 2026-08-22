---
title: Common Errors
sidebar_label: Common Errors
---

# Common errors

Start with the first failing component instead of changing several settings at
once:

```bash
vllm-sr status
vllm-sr logs router
vllm-sr logs envoy
vllm-sr validate --config config.yaml
```

The examples below are fragments. Add them to the corresponding section of a
complete canonical configuration and validate the result.
Use [config/config.yaml](https://github.com/vllm-project/semantic-router/blob/main/config/config.yaml)
when you need the exhaustive field context.

## Router cannot load the configuration

### `Failed to create ExtProc server`

This is a top-level startup failure. The useful cause normally appears later on
the same log line or immediately before it.

Check that the file exists and is readable, then run validation outside the
container:

```bash
test -r config.yaml
vllm-sr validate --config config.yaml
```

Follow the field path in the validation error. Do not add missing fields to a
random nested block; canonical fields are location-sensitive.

### `failed to read config file`

The process cannot open the path it received. Check:

- whether `--config` is relative to the current working directory;
- whether the same path exists inside the Router container;
- the file and parent-directory permissions; and
- whether a managed Recipe generated its runtime config in a different
  workspace.

Use `vllm-sr status` to identify the active workspace before inspecting
container mounts.

## Response cache cannot start

### Backend configuration is required

Errors such as these mean `backend_type` selected a backend without its
matching configuration:

```text
milvus configuration is required for Milvus cache backend
qdrant configuration is required for Qdrant cache backend
```

Milvus example:

```yaml
global:
  stores:
    response_cache:
      enabled: true
      backend_type: milvus
      milvus:
        connection:
          host: milvus
          port: 19530
        collection:
          name: response_cache
```

Qdrant example:

```yaml
global:
  stores:
    response_cache:
      enabled: true
      backend_type: qdrant
      qdrant:
        host: qdrant
        port: 6334
        collection_name: response_cache
```

The backend hostname must be resolvable from the Router container or pod, not
only from the host.

### Index or collection is missing

An error ending with `auto-creation is disabled` means the backing service is
reachable but the required index is absent. Choose one operating model:

- provision the index or collection before Router startup; or
- enable development-time creation for that backend.

Redis and Valkey use `development.auto_create_index`; Milvus uses
`development.auto_create_collection`. For example:

```yaml
global:
  stores:
    response_cache:
      enabled: true
      backend_type: redis
      redis:
        # Add the connection, index, and search settings from the runtime example.
        development:
          auto_create_index: true
```

See the checked-in
[response-cache examples](https://github.com/vllm-project/semantic-router/tree/main/config/runtime/response-cache)
for complete backend blocks. Production deployments commonly provision schema
separately and leave automatic creation disabled.

### Cache hits are unexpectedly rare

First confirm that the decision enables the `response_cache` plugin and inspect
`x-vsr-cache-hit` or replay diagnostics. If embeddings are healthy but near
duplicates miss, test a lower similarity threshold on representative traffic:

```yaml
global:
  stores:
    response_cache:
      similarity_threshold: 0.75
```

A per-decision override belongs in the plugin configuration:

```yaml
routing:
  decisions:
    - name: cached-route
      plugins:
        - type: response_cache
          configuration:
            enabled: true
            semantic:
              similarity_threshold: 0.70
```

Lower thresholds increase false-match risk. Evaluate answer equivalence before
rolling them out, and use the management API's response-cache statistics and
test endpoints when diagnosing the backend.

## Responses API store cannot connect

An error such as:

```text
failed to connect to Redis: redis ping failed
```

comes from the Responses API store, not the response cache. Check the separate
service block:

```yaml
global:
  services:
    response_api:
      enabled: true
      store_backend: redis
      redis:
        address: redis:6379
        db: 0
```

Use `store_backend: memory` only for local work where losing stored responses
and conversation chains on restart is acceptable.

## A PII route matches unexpectedly

With debug logging enabled, a matched PII rule includes the denied entity
types:

```text
[Signal Computation] PII rule "<name>" matched: denied_entities=[<types>]
```

If the type is allowed by policy, add it to that signal's allowlist. If the
detector is producing low-confidence false positives, evaluate a higher
threshold:

```yaml
routing:
  signals:
    pii:
      - name: pii-policy
        threshold: 0.90
        pii_types_allowed:
          - GPE
          - ORGANIZATION
```

Changing a privacy threshold changes false-negative risk. Validate it against a
labeled dataset rather than a few hand-written prompts.

## A jailbreak route matches unexpectedly

Contrastive classifier matches use a debug message beginning with:

```text
[Signal Computation] Contrastive jailbreak rule "<name>" matched
```

Other jailbreak classifiers do not emit that exact phrase. Use replay or an
`x-vsr-debug: true` request to inspect matched signals and the selected
decision.

To reduce false positives, evaluate a higher threshold on the affected signal:

```yaml
routing:
  signals:
    jailbreak:
      - name: jailbreak-standard
        threshold: 0.85
```

If a route should not depend on jailbreak detection, remove that condition
from the route. Disabling the classifier globally changes every decision that
uses it.

## MCP category classifier cannot start

These errors identify an incomplete transport:

```text
command is required for stdio transport
URL is required for HTTP transport
```

Use one transport and disable the local domain classifier when MCP should own
category classification.

Stdio example:

```yaml
global:
  model_catalog:
    modules:
      classifier:
        domain:
          enabled: false
        mcp:
          enabled: true
          transport_type: stdio
          command: /app/bin/category-server
          tool_name: classify_text
```

The executable and all arguments must exist inside the Router runtime.

Streamable HTTP example:

```yaml
global:
  model_catalog:
    modules:
      classifier:
        domain:
          enabled: false
        mcp:
          enabled: true
          transport_type: streamable-http
          url: http://mcp-server:8080/mcp
          tool_name: classify_text
```

Test that URL from the Router network. If the server exposes another tool
name, set `tool_name` exactly or omit it to allow discovery of a recognized
classification tool.

## Provider backend has no address

The validation error:

```text
providers.models[<model>].backend_refs requires endpoint or base_url
```

means a backend reference cannot be resolved:

```yaml
providers:
  models:
    - name: local-model
      provider_model_id: local-model
      api_format: openai
      backend_refs:
        - name: local-vllm
          endpoint: 10.0.0.1:8000
          protocol: http
          type: vllm
```

Use `base_url` when the provider requires a complete API root such as
`https://provider.example/v1`. Use a hostname reachable from the Router
network; `localhost` refers to the Router container itself.

See [Container connectivity](./container-connectivity) for end-to-end checks.

## A classifier or embedding model cannot load

Model-load errors vary by implementation but normally include the failed path:

```text
models directory does not exist: <path>
<name> model directory does not exist: <path>
failed to initialize <name> model from <path>: <error>
failed to load pre-trained model <path>: <error>
```

Check the path inside the runtime, not only on the host. A normal local
workspace mounts `models/` at `/app/models`; managed Recipes keep mutable model
state under their workspace and mount it at the same container path.

```yaml
global:
  model_catalog:
    embeddings:
      semantic:
        bert_model_path: /app/models/all-MiniLM-L12-v2
```

Also verify that the artifact format, label mapping, and configured embedding
dimension match the selected implementation.

## Container image has no matching platform

`ImagePullBackOff` with `no matching manifest` means the selected tag or digest
does not contain an image for the node architecture.

Inspect the exact reference from the deployment:

```bash
docker buildx imagetools inspect <registry>/<image>:<tag>
```

If the tag is a multi-platform index, pin its index digest rather than one
architecture's child manifest. If it is single-platform, use a release that
publishes the required architecture or build and publish the image through
your approved pipeline. Do not switch to an unpinned `latest` tag as a
long-term immutability workaround.

## Classification confidence is too low

If requests frequently fall back after domain classification, confirm the
loaded model and category mapping before changing the threshold. Then evaluate
a lower value on labeled traffic:

```yaml
global:
  model_catalog:
    modules:
      classifier:
        domain:
          threshold: 0.50
```

Lowering the threshold may increase wrong-domain routes. Report per-category
precision and recall at the chosen operating point.

## Diagnostic commands

```bash
# Validate the source configuration.
vllm-sr validate --config config.yaml

# Identify the active local stack and component state.
vllm-sr status

# Read component logs without depending on generated container names.
vllm-sr logs router
vllm-sr logs envoy

# Check the public listener and model catalog.
curl -sS http://localhost:8899/v1/models

# Check management health and metrics.
curl -sS http://localhost:8080/health
curl -sS http://localhost:9190/metrics | head
```

If a provider succeeds from the host but fails from the Router, continue with
[Container connectivity](./container-connectivity). For image and artifact
downloads, see [Restricted network environments](./network-tips).
