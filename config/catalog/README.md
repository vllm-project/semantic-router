# Built-in model catalog

This directory is the repository source of truth for built-in protocols,
providers and their native model mappings, model cards, reasoning behavior,
benchmark definitions, evaluation records, and composite indices.

The source manifest is `catalog.yaml`. Resource files live under `resources/`;
physical model families use one focused file under `resources/models/single/`
so a model Day-0 change does not modify an unrelated inventory. Router recipes
and their logical entrypoints live separately under `resources/models/virtual/`.
Secrets, operator endpoints, and request-facing aliases do not belong here.

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

- `protocols.yaml`: supported operations and their wire paths, including the
  default protocol base path used when an endpoint does not supply an API root.
  A configured `base_url` path replaces this default base path; the operation
  suffix is then appended exactly once.
- `providers/`: one file per Provider ID, including protocol compatibility,
  auth defaults, provider-native model IDs, per-provider restrictions/pricing,
  non-secret request-header defaults, reasoning transport, support tier,
  conformance, and presentation metadata. Each provider owns its `models[]`
  mappings; credential-bearing headers are forbidden here.
- `models/single/`: intrinsic facts for one deployable model family.
- `models/virtual/`: recipe-backed logical model identities and role contracts.
- `reasoning-families.yaml`: reusable request projections for reasoning knobs.
- `benchmarks.yaml`: versioned benchmark and metric definitions.
- `evaluations/single/`: redistributable measurements for physical models.
- `evaluations/virtual/`: recipe-run measurements for virtual models.
- `indices.yaml`: auditable normalization, weights, and missing-data policy.

Missing evaluation evidence stays missing. Never insert a guessed zero,
parameter-size proxy, or copied third-party result whose redistribution terms
are unknown. Two available records for the same model and versioned metric are
rejected instead of choosing a hidden winner; revise the evaluation identity or
resolve the conflicting evidence explicitly.

The generator creates exactly five default-index slots for every Model Card and
every selectable reasoning effort: MMLU-Pro, GPQA Diamond, Humanity's Last Exam
without tools, SWE-bench Verified, and Terminal-Bench 2.1. A slot links only to
an exact model/effort/profile measurement; otherwise it is emitted as
`missing`. A vendor-published score with an unspecified effort stays on a
separate `published` row and is never copied into `low`, `medium`, `high`, or
another selectable effort.

A physical Model Card represents one canonical upstream model identity. Date
snapshots, cloud aliases, quantizations, and serving-engine packaging do not
become duplicate cards: provider-specific names belong in that provider's
`models[]`, while runtime or quantization details belong in an evaluation
subject. A distinct checkpoint only becomes a new card when the publisher
treats it as a separately selectable model with materially different behavior.

Every active physical Model Card must be reachable through at least one
provider-owned mapping. A card may therefore appear under several providers
without duplicating its intrinsic identity. Virtual recipes are materialized
from packaged assets and keep their own evaluation directory.

The `reasoning` capability and `reasoning_family` serve different purposes. A
card can truthfully advertise reasoning even when vLLM Semantic Router has not
yet verified a configurable reasoning projection for that family. Only attach a
built-in reasoning family when its user-facing levels and wire transport are
implemented and tested; otherwise the model remains usable without inventing a
toggle.

Virtual-model `recommended_pool` entries are suggestions, not foreign keys.
They may name catalog-backed models or operator-defined models that only exist
in a deployment configuration.

See the [Day-0 support guide](../../website/docs/community/model-provider-day-0-support.md)
for the end-to-end contribution workflow.
