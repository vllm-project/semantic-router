# Configuration and recipes

Set `ROUTER_ORIGIN` to the intended stack's management origin. Inspect only the
schema needed for the change:

```bash
vllm-sr config schema --endpoint "$ROUTER_ORIGIN" --section providers.models
vllm-sr config schema --endpoint "$ROUTER_ORIGIN" --surface algorithm:multi_factor
```

Follow section child paths; use expanded/full output only when needed. Installed
help and the Router's advertised API define the supported contract.

## Activate a change

For a new stack, use `config init`, replace placeholders, validate locally and
launch. There is no remote revision to plan against before startup.

For a running stack, derive the candidate from its fresh canonical config:

```bash
vllm-sr config get --format yaml --endpoint "$ROUTER_ORIGIN" > active.yaml
cp active.yaml candidate.yaml
# Edit the intended policy, then:
vllm-sr config validate --config candidate.yaml --endpoint "$ROUTER_ORIGIN"
vllm-sr config plan --config candidate.yaml --mode replace --endpoint "$ROUTER_ORIGIN"
vllm-sr config apply --config candidate.yaml --mode replace --endpoint "$ROUTER_ORIGIN"
```

An old launch file may omit materialized defaults and cause unrelated changes.
Review the plan and choose whole-document replacement or a partial patch
intentionally. `apply` plans again and protects its own write with an ETag; a
separate earlier `plan` does not pin that later call. Use the discovered API's
`If-Match` contract when the reviewed revision must be fixed; there is no CLI
`--etag` option.

For `RESTART_REQUIRED`, use the authorized deployment replacement workflow.
Local Docker supports `serve --config candidate.yaml --replace-active-config`;
ordinary `serve` preserves active edits. Verify readiness, active revision and
representative routed requests after activation. Discover `config versions` and
`config rollback` when recovery is needed.

## Packaged recipes

Use `recipe builtin list`, then inspect `builtin export` or `builtin init`.
Export preserves bundle bytes; initialization binds a selected recipe to actual
providers. Preserve algorithm candidate counts and required capability/quality
evidence. Explicitly excluded decisions create a derivative with reduced coverage.
Keep relative assets valid when moving a derived configuration.

A bundle, recipe name and public entrypoint are different identities.
`vllm-sr/auto` cannot be rebound to a named recipe. Discover the activated binding
and use recipe-specific validate/plan/apply commands for package activation;
`--replace-active-config` does not replace an active recipe package.
Verify both [routing and delivery](https://vllm-sr.ai/install/agent/vllm-sr/references/route-verification.md).
