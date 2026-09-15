# Configuration and Recipe details

Use these details with the [operations skill](https://vllm-sr.ai/install/agent/vllm-sr/SKILL.md). Start with installed
help and the target Router's discovery; set `ROUTER_ORIGIN` to its actual
management origin. An inference listener is a separate endpoint.

## Discover the smallest contract

```bash
vllm-sr config schema --endpoint "$ROUTER_ORIGIN"
vllm-sr config schema --endpoint "$ROUTER_ORIGIN" --section providers.models
vllm-sr config schema --endpoint "$ROUTER_ORIGIN" --surface algorithm:multi_factor
```

Follow section child paths; request `--expanded` only for a self-contained
section or `--full` for the complete schema. Read `config get` before changing
an existing stack. Use `config init` for a new candidate and replace its starter
placeholders. YAML is the reviewable operator configuration; validate any
compiled Recipe representation through the installed Recipe tooling.

## First launch versus an existing Router

A new stack has no remote revision to plan against. Validate locally, launch
with the chosen stack settings, then discover readiness and the live schema.
Never accidentally plan against another stack on a default port.

For a running Router, preserve unrelated fields and choose replacement of a
complete document or merging an intentional partial patch explicitly:

```bash
vllm-sr config validate --config candidate.yaml --endpoint "$ROUTER_ORIGIN"
vllm-sr config plan --config candidate.yaml --mode replace --endpoint "$ROUTER_ORIGIN"
vllm-sr config apply --config candidate.yaml --mode replace --endpoint "$ROUTER_ORIGIN"
```

`plan` checks the candidate without writing. `apply` plans again and uses its
ETag as a compare-and-swap precondition for that internal plan-to-write interval.
A separate earlier `plan` does not pin a later `apply` to the revision reviewed
then. Refresh and reconcile if active config changed since review; when a pinned
reviewed revision is required, use the discovered API's `If-Match` contract.
There is no CLI `--etag` option. Pass credential variable names through the
supported token option without printing their values.

Provider-backend and listener topology is rendered into Envoy. If the plan
returns `RESTART_REQUIRED`, use the authorized deployment replacement path,
then recheck readiness and active configuration. Local Docker uses
`serve --config candidate.yaml --replace-active-config`; ordinary `serve`
preserves active Dashboard or Recipe edits. That flag does not replace an
active Recipe package: follow its discovered Recipe activation workflow.

For recovery, inspect `config versions` and `config rollback --help`, select the
intended revision, and plan the recovery when supported. Readiness and routed
probes must pass again after recovery.

## Packaged Recipes and entrypoints

Use `recipe builtin list` to discover named Recipes inside verified bundles,
then `builtin export` or `builtin init` as appropriate. Export preserves bundle
bytes and digest. Initialization binds an explicitly selected Recipe to existing
provider identities. Match every remaining decision and retain each algorithm's
minimum candidate count. When the provider config uses relative knowledge-base
assets, write its derivative beside the source config; relocating it also requires
relocating and validating those assets. A documented, authorized `--exclude-decision` creates
a derivative with unsupported request types; keep the original as the baseline.

A bundle, Recipe name, and public model entrypoint are separate identities.
Discover the activated Recipe's published entrypoint from the running Router.
`vllm-sr/auto` is reserved and cannot be rebound to that Recipe. Use
`recipe validate`, `recipe plan`, and `recipe apply` only after checking their
installed help and target contract. Verify the active binding using both
preview and real routed requests; see [evaluation details](https://vllm-sr.ai/install/agent/vllm-sr/references/evaluation-loop.md).
