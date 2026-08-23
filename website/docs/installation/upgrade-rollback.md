---
sidebar_position: 10
title: Upgrade and rollback
description: Migrate Router manifests, upgrade managed state, and retain a tested rollback path.
---

# Upgrade and rollback

Router releases, Router configuration manifests, the Management HTTP API, and
managed database schemas are separate versioned contracts. Upgrade each contract
deliberately instead of inferring compatibility from an image tag.

| Contract | Version signal | Compatibility rule |
| --- | --- | --- |
| Router release | image, chart, or Python package version | Pin the release and artifact digest used by each deployment. |
| Router manifest | top-level `version` | The v0.4 runtime accepts only `v0.4`. |
| Management HTTP API | path and media type | Clients select both explicitly; it evolves independently from the manifest. |
| Managed persistence | schema migration revision | A migration job must complete before new Router replicas become ready. |

## v0.3 to v0.4 manifest migration

The v0.4 runtime has one schema and does not contain a v0.3 reader. Convert a
v0.3 manifest offline before upgrading the Router:

```bash
vllm-sr config migrate \
  --config ./router-v0.3.yaml \
  --output ./router-v0.4.yaml
```

The output path defaults to `<source-stem>.v0.4.yaml`. An existing output is
never replaced unless `--force` is present, and the source file can never be the
output. The converter:

1. accepts exactly `version: v0.3`;
2. rejects duplicate YAML keys and unknown or lossy constructs;
3. separates Model connection data from semantic Model cards;
4. turns decision `modelRefs` into Entrypoint assignments;
5. preserves supported global services and Recipe documents;
6. validates the complete v0.4 schema and cross-resource references; and
7. writes the result atomically with owner-only permissions.

The converter reports every detected blocker together, using stable issue codes,
source paths, and corrective guidance. It never produces a partially valid file.

### Secret handling

Only environment references may cross the migration boundary. For example:

```yaml
# v0.3 source
api_key_env: OPENAI_API_KEY
```

becomes a named v0.4 credential reference backed by the same environment
variable. Plaintext keys, tokens, passwords, secrets, and authorization headers
cause migration to fail. Their values are never repeated in terminal output.
Move each secret into the deployment secret store, replace the source value with
an environment reference, and rerun the command.

### Resource translation

| v0.3 source | v0.4 result |
| --- | --- |
| `providers.models[]` | `models[]` connection, provider, runtime, and pricing fields |
| `routing.modelCards[]` | matching `models[].card` semantic metadata |
| top-level `routing` | a named default Recipe document |
| `recipes[].routing` | a named Recipe document |
| decision `modelRefs` | Entrypoint decision assignments |
| `model_names` | Entrypoint name plus aliases |
| environment-backed backend authentication | named `global.services.backend_credentials` reference |

The migration stops when a value has no exact v0.4 meaning. Examples include
plaintext credentials, backend transport overrides that are not represented by
the v0.4 provider contract, embedded algorithm model selection, model-bound
Recipes with no request-facing Entrypoint, conflicting global model selection,
or orphan Model cards. Correct the named source path before retrying.

## Standalone upgrade

1. Pin the current image digest and retain the v0.3 source manifest.
2. Run `vllm-sr config migrate` without `--force`.
3. Review the generated Models, Recipes, Entrypoints, assignments, pricing, and
   environment credential names.
4. Supply every referenced environment secret to the new deployment.
5. Validate the generated file with the target v0.4 CLI.
6. Start a replacement Router with the v0.4 manifest and run authenticated
   discovery, non-streaming inference, streaming inference, tool-call, and quota
   checks.
7. Move traffic only after readiness and those checks succeed.

Do not mount the v0.3 and v0.4 manifests into the same runtime and do not depend
on a fallback parser. A standalone rollback restores the pinned previous Router
release together with its retained v0.3 manifest.

## Managed upgrade

Managed deployments add durable control-plane state to the manifest procedure:

1. Back up PostgreSQL and record the active routing and policy publication
   revisions, image digests, and required keyring versions.
2. Convert and review the immutable bootstrap manifest offline.
3. Run the release's schema migration job with the target Router image.
4. Confirm the migration revision before allowing new replicas to become ready.
5. Roll out replicas gradually and verify policy projection, credential loading,
   authenticated discovery, admission, settlement, usage ingestion, and audit.
6. Move traffic only after every replica acknowledges the same valid publication.

Valkey is a rebuildable serving projection, not the migration authority. Never
hand-edit it as an upgrade step. A managed rollback is supported only when the
previous Router release accepts the resulting database revision. Otherwise,
restore the recorded PostgreSQL backup and matching keyring versions before
starting the previous image.

There is intentionally no v0.4-to-v0.3 down-converter.

## Management API versioning

Manifest versions and Management HTTP API versions evolve independently. A v0.4
manifest does not imply a particular future Management API version.

Clients must select the API path and media type explicitly:

```http
Accept: application/vnd.vllm-semantic-router.management.v1+json
Content-Type: application/vnd.vllm-semantic-router.management.v1+json
```

Within `/management/v1`, compatibility changes are limited to additive response
fields and optional request fields with documented defaults. Clients must ignore
unknown response fields, but the server rejects unknown request fields. A change
that removes or renames a field, changes its meaning or default, or alters a
resource lifecycle uses `/management/v2`, a new media type, and a separately
published OpenAPI document. Clients never negotiate by Router release number or
silently fall back to another Management API version.

## Acceptance and rollback record

Before completing any upgrade, record:

- release tags and resolved image digests;
- source and target manifest checksums;
- converter success summary and manual review owner;
- database migration and backup revisions, when managed;
- active routing and policy publication revisions;
- smoke-test evidence for discovery, inference, streaming, tools, quota, usage,
  and audit; and
- the exact rollback image, manifest, database backup, and keyring set.

Keep the previous artifacts until the rollback window closes. Treat a successful
process start as necessary but not sufficient evidence of a safe upgrade.
