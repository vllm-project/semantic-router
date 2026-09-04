# Built-in model catalog

This directory is the repository source of truth for built-in protocols,
providers, model cards, provider offerings, reasoning behavior, benchmark
definitions, evaluation records, and composite indices.

The source manifest is `catalog.yaml`. Resource files live under `resources/`;
physical model families should use one focused file under `resources/models/`
so a model Day-0 change does not modify an unrelated inventory. Closely related
virtual variants may share a family file. Secrets, operator endpoints, and
request-facing aliases do not belong here.

Run:

```bash
make model-catalog-generate
make model-catalog-check
```

Generation validates the resource graph and rewrites the committed Router,
CLI, Dashboard, and website projections. Do not edit those projections by
hand. Ordinary user YAML never includes the catalog release, digest, or default
index identity; those are embedded build metadata.

## Resource ownership

- `protocols.yaml`: supported operations and their wire paths, including
  inference creation and model inventory discovery.
- `providers.yaml`: Provider IDs, protocol compatibility, auth defaults,
  non-secret request-header defaults, reasoning transport, support tier,
  conformance, and presentation metadata. Credential-bearing headers are
  forbidden here.
- `models/`: intrinsic model or virtual-model facts, one file per family.
- `offerings.yaml`: provider/model pairings, provider model IDs, restrictions,
  and dated pricing.
- `reasoning-families.yaml`: reusable request projections for reasoning knobs.
- `benchmarks.yaml`: versioned benchmark and metric definitions.
- `evaluations.yaml`: redistributable built-in measurements and evidence.
- `indices.yaml`: auditable normalization, weights, and missing-data policy.

Missing evaluation evidence stays missing. Never insert a guessed zero,
parameter-size proxy, or copied third-party result whose redistribution terms
are unknown.

See the [Day-0 support guide](../../website/docs/community/model-provider-day-0-support.md)
for the end-to-end contribution workflow.
