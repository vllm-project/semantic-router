# Built-in model catalog

This directory is the repository source of truth for built-in protocols,
providers and their native model mappings, model cards, reasoning behavior,
benchmark definitions, evaluation records, and composite indices.

The source manifest is `catalog.yaml`. Resource files live under `resources/`;
physical models are grouped into one focused file per creator under
`resources/models/single/`, and their measurements use the matching creator
file under `resources/evaluations/single/`. Router recipes and their logical
entrypoints live separately under `resources/models/virtual/`, with recipe-run
measurements under `resources/evaluations/virtual/`. Secrets, operator
endpoints, and request-facing aliases do not belong here.

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
- `providers/`: one file per stable Provider ID, including the runtime serving
  contract: protocol compatibility, auth defaults, provider-native model IDs,
  per-provider restrictions/pricing, non-secret request-header defaults,
  reasoning transport, support tier, conformance, and presentation metadata.
  The optional repository-owned `presentation.featured` flag curates the
  default Dashboard picker without removing any provider from search or the
  runtime registry. Each provider owns its `models[]` mappings;
  every mapping explicitly classifies the creator-to-serving-channel
  relationship as `first_party`, `managed_cloud`, `gateway`, or `self_hosted`.
  Credential-bearing headers are forbidden here.
- `models/single/`: intrinsic facts for physical models, grouped by creator.
- `models/virtual/`: recipe-backed logical model identities and role contracts.
- `reasoning-families.yaml`: reusable request projections for reasoning knobs.
- `benchmarks.yaml`: versioned benchmark and metric definitions.
- `evaluations/single/`: exact benchmark measurements for physical models,
  grouped by creator.
- `evaluations/virtual/`: recipe-run measurements for virtual models.
- `indices.yaml`: auditable normalization, weights, and missing-data policy.

Missing evaluation evidence stays missing. Never insert a guessed zero or a
parameter-size proxy. Two available records for the same model, effort,
versioned benchmark profile, and metric are rejected instead of choosing a
hidden winner; revise the evaluation identity or resolve the conflicting
evidence explicitly.

The generator creates exactly five default-index slots for every Model Card and
every selectable reasoning effort: MMLU-Pro, GPQA Diamond, Humanity's Last Exam
without tools, SWE-bench Verified, and Terminal-Bench 2.1. A slot links only to
an exact model/effort/profile measurement; otherwise it is emitted as
`missing`. A vendor-published score with an unspecified effort stays on a
separate `unspecified` row and is never copied into `low`, `medium`, `high`, or
another selectable effort.

Evaluation admission and selectable-effort completeness are separate facts.
Every physical card must have at least five distinct benchmarks in one exact
model/effort/provenance evidence bucket. A reasoning family may expose
additional real runtime levels whose effort-specific measurements have not been
published; those
levels retain explicit `missing` slots. The catalog audit reports selectable
levels as complete, partial, or unmeasured and can enforce them with a stricter
opt-in gate, but neither generation nor the Hub copies a score across levels.
For a card without `reasoning_family`, labels such as `enabled`, `disabled`,
`default`, and `unspecified` describe the published run condition only; they do
not create a user-configurable selector.

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

The built-in physical inventory is curated at the creator-company level. The
current baseline contains 83 physical cards from 22 mainstream creators and
five separately stored virtual cards. For each creator, prefer roughly the
latest three generations or representative product lines over accumulating a
shallow long tail of lesser-known creators. This policy is about Model Cards,
not serving endpoints: the 60 `ProviderDefinition` resources remain broad so
Add Model and handwritten custom models can use a known runtime contract even
when that provider has no curated built-in model mapping. `ModelCard.publisher`
is the creator; a `ProviderDefinition` is the runtime API contract for the
cloud, gateway, or runtime serving it; and a binding's `relationship` states
how that serving channel relates to the creator. This prevents a gateway from
being mistaken for the model publisher without overloading provider category
or support tier.
The repository-only `catalog.yaml.inventory.physical` policy records the
creator allowlist, reviewed current representative model IDs, and minimum
depth. Generation rejects unlisted physical creators, missing or stale
representatives, or a creator that falls below that depth. The policy
is not emitted into runtime snapshots or exposed in user configuration; whether
a candidate is mainstream and which recent lines are representative remains a
review decision rather than a mechanical release-date ranking.

## User configuration boundary

Catalog adoption is additive within the existing v0.3 hierarchy:

- `providers.models[].catalog` optionally selects a canonical built-in Model
  Card, while `providers.models[].name` remains the request-facing alias.
- `backend_refs[].provider` selects the stable runtime Provider ID. The
  provider's `models[]` mapping connects the canonical card to a native model
  ID when that provider has a built-in mapping.
- A catalog-backed model materializes its card and reasoning family
  automatically. An intentional `routing.modelCards` override uses the
  canonical `catalog` value as its `name`.
- A custom vLLM, SGLang, private, or newly released model omits `catalog`. It
  can remain a minimal binding or supply a handwritten Model Card, custom
  reasoning behavior, and evaluations.

Catalog release versions, digests, internal index identities, generated
defaults, and binding relationship classifications never belong in normal
user YAML. Users select a Provider ID but do not declare or override the
repository-owned relationship.

The `reasoning` capability and `reasoning_family` serve different purposes. A
card can truthfully advertise reasoning even when vLLM Semantic Router has not
yet verified a configurable reasoning projection for that family. Only attach a
built-in reasoning family when its user-facing levels and wire transport are
implemented and tested; otherwise the model remains usable without inventing a
toggle.

Most reasoning families have one control axis. A family may additionally set
`activation_parameter` when a model has an independent on/off switch as well as
an effort ladder. For example, Qwen3.8 uses `enable_thinking` for activation and
`reasoning_effort` for `low`, `medium`, or `xhigh`; `none` is not fabricated as
an effort level. Provider bindings still own whether those controls travel as
chat-template kwargs, top-level fields, or a provider-native object.

Virtual-model `recommended_pool` entries are suggestions, not foreign keys.
They may name catalog-backed models or operator-defined models that only exist
in a deployment configuration.

Model Hub is a catalog, not an overall model ranking. The generated product
views may compare only one selected benchmark version, profile, and metric.
Every bar is one exact model-and-reasoning-effort record and labels that effort
explicitly; missing records are omitted rather than treated as zero. Internal
index resources remain available to routing code, but they do not create a
public composite leaderboard.

See the [Day-0 support guide](../../website/docs/community/model-provider-day-0-support.md)
for the end-to-end contribution workflow.
