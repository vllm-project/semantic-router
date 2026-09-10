---
title: Configuration Contract
description: Discover, validate, and extend the canonical vLLM Semantic Router configuration without duplicating schemas.
---

# Configuration Contract

The Go configuration types and routing registries are the source of truth for
the canonical document. A checked-in JSON Schema is generated from that source
and consumed by the Router API, CLI, and Dashboard.

```text
Go config types + routing registries
                 │
                 ├── one checked-in JSON Schema + surface catalog
                 │      ├── embedded in Router builds
                 │      ├── staged into CLI and Dashboard builds
                 │      └── consumed directly by Website and Dashboard UI
                 │
                 └── Router semantic validators
                        └── validate API used before apply
```

This division is intentional:

- the generated schema owns field names, shapes, descriptions, and the
  supported signal, projection, algorithm, and plugin inventories;
- Router validators own cross-field constraints, references, security rules,
  filesystem checks, defaults, and runtime feasibility;
- the Dashboard may add labels or specialized controls, but it merges them
  onto generated fields instead of maintaining another schema;
- migration-only aliases remain explicit in the CLI and never become part of
  the steady-state schema.

## Discover the contract

Start with the compact section and routing-surface index bundled with the CLI:

```bash
vllm-sr config schema
```

Expand only what the current task needs:

```bash
vllm-sr config schema --section global.router.learning
vllm-sr config schema --surface signal:keyword
vllm-sr config schema --surface algorithm:multi_factor
vllm-sr config schema --full
```

Query the exact contract exposed by a running Router:

```bash
vllm-sr config schema \
  --endpoint http://localhost:8080/config/router/schema
```

`--endpoint` works with every progressive option. The Router endpoint is
`GET /config/router/schema`: omitting `view` returns the compact index; use
`view=section&path=...` or `view=surface&kind=...&name=...` to expand one branch,
and `view=full` for the complete JSON Schema.

The Dashboard proxies the deployed Router contract at
`GET /api/router/config/schema`. If that Router endpoint is unavailable, it
falls back to the schema embedded in the Dashboard build and marks the response
with `X-Vllm-Sr-Schema-Source: bundled`. The
**Operate → Platform & Access → Schema Reference** page shows this source and
warns when the deployed and bundled contracts differ. The Website
[Configuration Schema Reference](../api/configuration-schema) visualizes the
current documentation release instead of claiming to represent a deployment.

Every representation has its own `ETag`; use `If-None-Match` when an agent or
editor caches it.

The standard JSON Schema describes the complete canonical document. Its
`x-vllm-sr` section adds routing-specific discovery metadata:

- `signals`: discriminator, YAML collection, runtime observation key,
  decision-reference capability and qualification, and item schema;
- `algorithms`: discriminator, support tier, execution mode, and payload schema;
- `plugins`: discriminator, description, and configuration schema;
- `projections` and `projection_input_types`: supported derived-routing surfaces;
- `global_sections`: canonical global paths used to build management surfaces;
- `schema_endpoint` and `validation.endpoint`: discovery and semantic-validation paths.

## Validate in two stages

JSON Schema validation catches structural errors early. Before applying a
configuration, send the authored YAML to `POST /config/router/validate`:

```json
{
  "yaml": "version: v0.3\nrouting: {}\nglobal: {}\n"
}
```

The Router returns normalized, redacted YAML when the document is valid. It
does not mutate the active configuration. Schema validation alone is not an
apply-time guarantee because it cannot prove references, deployment
connectivity, local assets, or cross-field policy.

Validation has three owners. The generated schema owns document structure;
the Go Router owns routing semantics; CLI or Dashboard deployment code owns
environment-specific checks such as filesystem access and process launch.

Consumer-side checks are allowed only at a boundary they own:

- CLI offline preflight may reproduce a Router diagnostic when the Router is
  not running, but it must derive fields and discriminator values from the
  generated contract. The Router remains authoritative at startup and apply.
- Dashboard forms may check incomplete interaction state before save, but the
  management backend and Router validation decide whether the resulting
  document is valid.
- Migration code may recognize retired names solely to produce canonical
  configuration; those aliases are not accepted steady-state fields.

Do not add consumer field allowlists, copies of signal/algorithm/plugin
inventories, or a consumer-only semantic rule. A rule that determines whether
the Router can run belongs in Go first.

## Agent authoring loop

An automation or deployment agent should:

1. fetch the running Router schema index, falling back to its bundled CLI index;
2. fetch only the relevant section and surface schemas while authoring;
3. use `schema_id`, `contract_version`, and `ETag` as the contract identity;
4. construct the smallest canonical document from schema fields and routing
   surface references;
5. omit the bootstrap-only `setup` block and call the semantic validation endpoint;
6. present validation errors or apply through the normal management workflow.

Agents should never infer a field from an example or send unknown keys when a
schema for the target Router is available.

## Add or change a field

For a steady-state field, update its Go type, YAML tag, and source comment. Add
`jsonschema:"required"` only when presence is structurally mandatory; defaults
and cross-field requirements remain semantic validation. For a discriminated
routing surface, update the matching Go registry as well. Registry coverage is
checked against the reflected Go fields, so generation fails if a signal,
projection, or algorithm payload is added on only one side.

Keep semantic validation in the Router and platform validation beside the
deployment code that owns it. Dashboard presentation metadata may improve a
generated control, but an uncurated new field or global section must still be
editable through the generic schema renderer. Then regenerate and run the
contract checks:

```bash
make config-schema-generate
make config-schema-check
make check
```

The repository tracks exactly one full schema at
`src/semantic-router/pkg/configschema/router-config-v0.3.schema.json`. The
generator also emits a small TypeScript import/typing adapter, but no second
JSON copy. Packaging stages the canonical artifact into wheels and container
images without writing generated files back into the worktree. CI fails when
the canonical artifact or adapter is stale.

`setup.mode` is represented by the product document contract but remains
control-plane metadata. The Dashboard removes it at activation; active Router
documents and calls to `/config/router/validate` must not include it.
