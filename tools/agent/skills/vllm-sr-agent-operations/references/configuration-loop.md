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

For a new stack, write the configuration as the [main skill](https://vllm-sr.ai/install/agent/vllm-sr/SKILL.md)
does, or start from `config init` and replace its placeholders (its listener
address, `0.0.0.0`, publishes the API on every interface). Validate locally and
launch; there is no remote revision to plan against before startup.

On the Docker target the Router serves `.vllm-sr/runtime-config.yaml`, not
`config.yaml`: editing the source file changes nothing until it is applied.

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

`apply` waits up to 120 s (`--timeout`). A timeout doesn't mean the change
failed: it can activate a moment later, so read `config versions` before
retrying.

A change the running Router can't take (in standalone mode a listener change;
with `--gateway extproc`, also the provider topology) is `restart_required`.
On a local Docker stack, `config apply` saves it and prints `Restart required:
run vllm-sr serve to apply.`; `status` reports it until `serve` applies it.
Elsewhere, use the authorized deployment replacement workflow, such as
`helm upgrade` on Kubernetes. Verify readiness, active revision and representative
routed requests after activation. Discover `config versions` and
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
