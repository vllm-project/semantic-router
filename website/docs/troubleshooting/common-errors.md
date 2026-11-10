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
vllm-sr logs envoy        # only with --gateway extproc
vllm-sr config validate --config config.yaml
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
vllm-sr config validate --config config.yaml
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

## Entrypoint / Recipe Validation

`vllm-sr config validate` (or `vllm-sr validate`) includes a repair hint for common multi-recipe wiring errors.

### Unknown recipe

```text
Entrypoint references unknown recipe 'missing-recipe'
Hint: Change this to the name of a recipe defined under recipes.
```

Broken:

```yaml
entrypoints:
  - model_names: [my-model]
    recipe: missing-recipe
recipes:
  - name: production
```

Corrected:

```yaml
entrypoints:
  - model_names: [my-model]
    recipe: production
recipes:
  - name: production
```

### Duplicate recipe name

```text
Duplicate recipe name 'production'
Hint: Rename one recipe so every recipe has a unique name.
```

Give each recipe a distinct `name`, then update entrypoints that refer to the
renamed recipe:

```yaml
recipes:
  - name: production
  - name: staging
entrypoints:
  - model_names: [my-model]
    recipe: production
```

### Default entrypoint name collision {#model-or-reserved-alias-collision}

```text
Entrypoint model 'vllm-sr/auto' is mapped more than once
```

`vllm-sr/auto` publishes the default recipe unless an explicit entrypoint
replaces it. Giving that same name to a named recipe creates a duplicate.

Broken:

```yaml
entrypoints:
  - model_names: [vllm-sr/auto]
    recipe: production
```

Corrected:

```yaml
entrypoints:
  - model_names: [customer-production]
    recipe: production
```

### Entrypoint name collides with a backend model

A public recipe entrypoint must have its own name. The Router rejects an
entrypoint whose name also identifies a backend model, including Fusion,
ReMoM and Flow entrypoints. There is no special dispatch namespace for these
algorithms.

Broken, if `openai/gpt-oss-20b` is a configured backend:

```yaml
entrypoints:
  - model_names: [openai/gpt-oss-20b]
    recipe: flow
```

Corrected, with Flow requests sent to `vllm-sr/flow` and the flow recipe
containing decisions with `algorithm.type: workflows`:

```yaml
entrypoints:
  - model_names: [vllm-sr/flow]
    recipe: flow
```

See the
[entrypoints and recipes tutorial](../tutorials/global/entrypoints-and-recipes.md)
and [recipes tutorial](../tutorials/global/recipes.md) for complete examples.

## A request returns a routing error code

Look the code up in [routing errors](../api/router.md#routing-errors), then
find the Router's own reason in its log under the request's `x-request-id`:

```bash
vllm-sr logs router | grep '<x-request-id>'
```

For `no_route`, the `entrypoint_routing_no_selection` line names the model,
the recipe, and any decision that matched. Fusion, Flow and ReMoM are
selected by decisions inside that recipe; their public names are ordinary
entrypoints. Check the entrypoint mapping and recipe rules first.

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

First confirm that a `response_cache` plugin is enabled for the request, in
`global.plugins` or the decision's plugins, and inspect
`x-vsr-response-path` (`cache` on a hit) or replay diagnostics. If embeddings are healthy but near
duplicates miss, test a lower `min_similarity` on representative traffic:

```yaml
global:
  plugins:
    - type: response_cache
      configuration:
        enabled: true
        semantic:
          match:
            recall: {top_k: 1, min_similarity: 0.75}
```

A decision's own plugin overrides it for that decision:

```yaml
recipes:
  - name: default
    decisions:
      - name: cached-route
        plugins:
          - type: response_cache
            configuration:
              enabled: true
              semantic:
                match:
                  recall: {top_k: 3, min_similarity: 0.70}
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

Preview the request (`POST /api/v1/routing/preview?trace=true`), or send it
with `x-vsr-debug: true` from a caller `global.router.headers.debug` allows:
the `signals` member of `x-vsr-trace` names the `pii` rule that matched.

If the label is allowed by policy, add it to the rule's `ignore_labels`. If the
model produces low-confidence false positives, evaluate a higher `predicate`:

```yaml
recipes:
  - name: default
    signals:
      pii:
        - name: pii-policy
          predicate: {gte: 0.90}
          ignore_labels:
            - DATE_TIME
            - LOCATION
```

Changing a privacy threshold changes false-negative risk. Validate it against a
labeled dataset rather than a few hand-written prompts.

## A jailbreak route matches unexpectedly

Use replay, a routing preview, or an `x-vsr-debug: true` request from a caller
`global.router.headers.debug` allows to inspect the matched signals (the
`signals` member of `x-vsr-trace`) and the selected decision.

To reduce false positives, evaluate a higher `predicate` on the affected rule:

```yaml
recipes:
  - name: default
    signals:
      jailbreak:
        - name: jailbreak-standard
          predicate: {gte: 0.95}
```

If a route should not depend on jailbreak detection, remove that condition
from the route. Changing the rule changes every decision that reads it.

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
          provider: vllm
```

Write a full URL in `endpoint` when the provider requires a complete API root
such as `https://provider.example/v1`. Use a hostname reachable from the Router
network; `localhost` refers to the Router container itself.

The version segment in that API root is accepted for every Provider ID.
Providers whose own operation path already carries the version, such as
`anthropic` and `minimax`, do not repeat it, so both
`https://provider.example/v1` and `https://provider.example` resolve to the
same upstream path.

See [Container connectivity](./container-connectivity) for end-to-end checks.

## A decision or embedding model cannot load

Every model runs in the [model runtime](../model-runtime/overview.md). When a
model cannot load, its feature reports unknown results and the runtime records
why. Check the router's `vsr_model_runtime_ready{deployment="..."}` metric and
the router log, or ask a runtime you run yourself:

```bash
curl -s localhost:8100/v1/models
```

Each model's `status` and `reason` say what failed: a damaged download, a
missing `revision`, a private repository without a token, a device that does
not exist, or a model too large for its device.
[Troubleshooting and FAQ](../model-runtime/troubleshooting.md) lists each
reason and the fix.

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

## Domain confidence is too low

If requests frequently fall back after domain classification, confirm which
model answers the `domains` rules before changing their `predicate`: a rule
without one uses the answering model's calibrated threshold (0.41 on
vela3-fast), or 0.5 with an `uncalibrated_preset` warning. Then evaluate a
lower value on labeled traffic:

```yaml
recipes:
  - name: default
    signals:
      domains:
        - name: math
          predicate: {gte: 0.30}
```

Lowering the predicate may increase wrong-domain routes. Report per-domain
precision and recall at the chosen operating point.

## Diagnostic commands

```bash
# Validate the source configuration.
vllm-sr config validate --config config.yaml

# Identify the active local stack and component state.
vllm-sr status

# Read component logs without depending on generated container names.
vllm-sr logs router
vllm-sr logs envoy        # only with --gateway extproc

# Check the public listener and model catalog.
curl -sS http://localhost:8899/v1/models

# Check management health and metrics.
curl -sS http://localhost:8080/health
curl -sS http://localhost:9190/metrics | head
```

If a provider succeeds from the host but fails from the Router, continue with
[Container connectivity](./container-connectivity). For image and artifact
downloads, see [Restricted network environments](./network-tips).
