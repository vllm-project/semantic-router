# Built-in Recipes

## Overview

The Router ships a small set of reusable routing Recipes. A Recipe defines how
signals, projections, decisions, algorithms, and plugins work together. It does
not contain provider endpoints, credentials, physical Models, or request-facing
Entrypoints.

## Included family

| Family | Profiles | Model Card |
| --- | --- | --- |
| MoM V1 | Balance, speed, cost, accuracy, and privacy | [MoM V1](latest/mom-v1/README.md) |

## Use a built-in Recipe

```bash
vllm-sr serve
```

Connect Models in the Dashboard, choose a Recipe, then create a Mixture of
Models by assigning configured Models to its decisions. Publishing that
Entrypoint makes the chosen name available through the inference API.

The Dashboard is a client of the Router Management API. Another control plane
can perform the same lifecycle directly.

## Versioning

`latest` is the maintained authoring source bundled into development images.
Each stable Router release binds an immutable `vMAJOR.MINOR` snapshot. Existing
Entrypoints stay pinned to the Recipe revision they selected until a control
plane explicitly updates them.

The Router installs built-in Recipes into every active Namespace as ordinary,
read-only Recipe resources. Installation is content-addressed, audited,
idempotent across replicas, and repeated for Namespaces created later.

## Boundary

- A built-in Recipe never selects or provisions a physical Model.
- Model capability, connection, runtime, pricing, and credential data belong to
  Model resources.
- Decision-to-Model assignment and fallback policy belong to an Entrypoint.
- A control plane must publish an Entrypoint before clients can call it.

## For contributors

Every bundle contains exactly `metadata.yaml`, `config.yaml`, `recipe.dsl`,
`probes.yaml`, and `README.md`. `config.yaml` and `recipe.dsl` are Recipe-only.
The Router image copies only `metadata.yaml` and `config.yaml`; the remaining
files support review and conformance.

See [Recipe authoring and conformance](../CONFORMANCE.md) for validation rules.
