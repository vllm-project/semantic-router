---
sidebar_position: 10
title: Upgrade and rollback
description: Migrate Router manifests, upgrade durable state, and retain a tested rollback path.
---

# Upgrade and rollback

Router releases, Router manifests, the Management HTTP API, and database schemas
are separate versioned contracts. Upgrade each one deliberately instead of inferring
compatibility from an image tag.

| Contract | Version signal | Compatibility rule |
| --- | --- | --- |
| Router release | image, chart, or Python package version | Pin the release and artifact digest. |
| Router manifest | top-level `version` | The current runtime accepts the strict `v0.3` contract. |
| Management HTTP API | path and media type | Clients select both explicitly; it evolves independently. |
| Durable persistence | schema migration revision | Migration must finish before new replicas become ready. |

## Manifest migration

Serving processes have one strict parser and no fallback reader. When a target
release removes or renames a field, convert the retained source offline:

```bash
vllm-sr config migrate \
  --config ./router-previous.yaml \
  --output ./router-current.yaml
```

The current converter rewrites only the documented v0.3 changes:

| Previous field | Current v0.3 field |
| --- | --- |
| `providers.models[].reliability.retry_count` | `providers.models[].control.retry.count` |
| `providers.models[].reliability.retry_on` | `providers.models[].control.retry.on` |
| `pricing.prompt_per_1m` | `pricing.input_cost_per_million_tokens` |
| `pricing.completion_per_1m` | `pricing.output_cost_per_million_tokens` |
| `pricing.cached_input_per_1m` | `pricing.cache_read_cost_per_million_tokens` |
| `pricing.cache_write_per_1m` | `pricing.cache_write_cost_per_million_tokens` |
| per-Model `pricing.currency` | `global.billing.currency` |
| `global.router.config_source: file` | field removed; file authority is the default when no Management store exists |
| disabled or empty `global.router.skip_processing` | field removed |

Earlier `reliability` load-balancing, health-check, and outlier-ejection fields
were accepted but never controlled Router dispatch. The converter removes them,
reports the number removed, and does not invent replacement behavior. Existing
`backend_refs[].weight` values remain unchanged.

The four rates become quoted decimal strings. Empty static consumer API-key,
`authz`, and `ratelimit` blocks are removed. Non-empty blocks stop conversion;
define equivalent users, keys, access policies, and quota policies through the
Management API before removing them from the retained source.

`global.router.config_source` no longer chooses runtime authority. With no
Management store, the v0.3 file is authoritative. With a Management store, the file
seeds an empty store once and PostgreSQL is authoritative thereafter. The converter
therefore removes `config_source: file` but rejects `config_source: kubernetes`:
export that desired state, use it as the file bootstrap for an empty Management
store, then apply later changes explicitly through the versioned Management API.

`global.router.skip_processing` is also removed. An empty or disabled block is
safe to strip. An enabled block stops conversion because a caller-controlled bypass
would evade Router authentication, authorization, and quota. Move operational
traffic to authenticated health or Management endpoints instead of recreating the
bypass.

The command rejects duplicate YAML keys, unknown or lossy constructs, plaintext
secrets outside the supported provider `backend_refs[].api_key` field, and an
existing output unless `--force` is present. It reports every blocker with a source
path, writes atomically with owner-only permissions, and runs strict validation
before publishing the output. Prefer `api_key_env` for shared manifests. Converter
packages are not imported by `serve` or `validate`.

Provider connection data, including the existing mutually exclusive `api_key` and
`api_key_env` backend inputs, `routing.modelCards`, `recipes[].routing`,
`entrypoints[].model_names`, assignments, and safe environment references retain
their public v0.3 meaning. The converter never invents generated resource IDs,
backend IDs, revisions, or catalog digests.

Existing top-level default routing keeps its automatic request names. When no
explicit Entrypoint claims one of them, omission of
`global.router.auto_model_names` means `vllm-sr/auto`, `auto`, and `MoM` (or the
configured `auto_model_name`). A present list replaces those defaults; a present
empty list disables implicit names. An explicit Entrypoint that claims an automatic
name remains the sole owner of that name. The converter preserves these fields and
does not rewrite one form into another.

## Future manifest versions

A `v0.3` binary does not accept a future `version: v0.4` document or fields that
only v0.4 defines. A v0.4 release must publish its schema and an explicit offline
v0.3-to-v0.4 converter. The serving process still reads one target version; it does
not maintain permanent dual readers.

Additive optional fields may be introduced only when their absence preserves the
documented v0.3 behavior. A field removal, rename, semantic reinterpretation, or
default change requires a new manifest version unless the release explicitly lists
the break and supplies the converter.

## File-authoritative upgrade

1. Pin the current image digest and retain the exact source manifest.
2. Run `vllm-sr config migrate` into a new file.
3. Review Model control, pricing, Recipes, Entrypoints, assignments, and secret
   references.
4. Validate the output with the target CLI.
5. Start a replacement Router with the converted manifest.
6. Test discovery, non-streaming inference, streaming inference, tools, images, and
   every published Mixture-of-Models.
7. Move traffic only after readiness and those checks succeed.

Rollback restores the previous image and its matching retained manifest together.
Never mount two manifest versions into one serving process.

## Durable-state upgrade

When `global.stores.management` is configured:

1. Back up PostgreSQL and record the active routing and policy publication revisions,
   release digests, and required keyring versions.
2. Convert and validate the immutable bootstrap manifest offline.
3. Run the target release's forward-only schema migration Job.
4. Confirm the schema revision before allowing new replicas to become ready.
5. Roll out replicas gradually and verify routing publication, provider credentials,
   authenticated discovery, authorization, quota admission, actual settlement, usage,
   and audit.
6. Move traffic only after every live replica acknowledges the same publication.

The runtime store is a rebuildable hot projection and global-counter authority, not
the desired-state migration source. Never hand-edit it during an upgrade. Rollback
without a database restore is allowed only when the previous release declares the
resulting schema readable; otherwise restore the PostgreSQL backup and matching
keyring versions first.

## Management API versioning

Manifest versions and Management HTTP API versions evolve independently. Clients
select the API path and media type explicitly:

```http
Accept: application/vnd.vllm-semantic-router.management.v1+json
Content-Type: application/vnd.vllm-semantic-router.management.v1+json
```

Within `/management/v1`, compatible evolution is limited to additive response
fields and optional request fields with documented defaults. Clients ignore unknown
response fields at every object nesting level while continuing to validate required
fields and known-field types; the server rejects unknown request fields. Removing or
renaming a field, changing its meaning or default, or altering a lifecycle requires
`/management/v2`, a new media type, and a separately published OpenAPI document.
Clients never negotiate by Router release number or silently fall back.

## Acceptance and rollback record

Record:

- release tags and resolved image digests;
- source and target manifest checksums;
- converter summary and review owner;
- schema migration and database backup revisions, when applicable;
- active routing and policy publication revisions;
- discovery, inference, streaming, tools, image, quota, usage, and audit evidence; and
- the exact rollback image, manifest, backup, and keyring set.

Keep the previous artifacts until the rollback window closes. A successful process
start is necessary, but it is not sufficient evidence of a safe upgrade.
