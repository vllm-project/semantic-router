---
title: Configuration Management
description: How a configuration change activates on a running Router, what it rebuilds, how a rejected change is reported, and how to list and roll back versions.
---

# Configuration Management

Every way of changing a running Router's configuration goes through one
lifecycle: editing the file it watches, the management API, the dashboard, and
a Kubernetes resource. A change that fails any step is rejected and the version
that serves keeps serving. A change that passes every step takes the next
version number and replaces the previous version at once.

## How a change activates

The Router turns the document into an immutable snapshot and takes it through
these stages:

| Stage | What happens |
| --- | --- |
| `parse` | The document is read in the canonical layout. |
| `compile` | Listeners, providers, recipes, entrypoints and global settings become typed resources that refer to each other by name. A duplicate name or a reference to nothing is rejected here. |
| `validate` | The candidate is checked against the running Router: settings that need a restart, features the gateway mode lacks, and the model artifacts it needs. |
| `warm` | Models are prepared and the routing pipeline is built. In standalone mode the upstream connection pools are built, and health-checked backends finish their first check, before any request uses them. |
| `activate` | The new snapshot replaces the old one in one step. |

Requests in flight when a version activates finish on the version they started
on, and the old version is released when the last of them ends. In standalone
mode a request is served by one version from routing through its fallback
chain to the last byte of the response. Every routed response names the
version that served it in the `x-vsr-config-version` header.

If the watched file changes again before a candidate activates, the candidate
is dropped as superseded and the newer document goes through the lifecycle
instead.

## What a change rebuilds

The Router rebuilds only what a change reaches:

- A change to a provider's backends or to a listener keeps the loaded
  classifiers and embedding models.
- In standalone mode, a change to recipes or signals keeps the upstream
  connection pools, and a changed provider keeps the pools of its backends that
  did not change.
- In standalone mode the Router binds its listeners at startup. Adding or
  removing a listener, or changing a listener's `address`, `port`, `timeout` or
  `tls`, is rejected with `restart_required`; restart the Router to apply it.
  Listener `api_keys` change without a restart.

## Change a local stack

On the docker target, `vllm-sr serve` copies `config.yaml` into the active
document, `.vllm-sr/runtime-config.yaml`, which is the file the Router
watches. Editing `config.yaml` afterwards changes nothing until you apply it:

```bash
vllm-sr config plan --config config.yaml    # check the change against the running Router
vllm-sr config apply --config config.yaml   # activate it
```

`vllm-sr config apply` waits up to 120 seconds for the change to activate,
which covers loading another model; pass `--timeout` to wait longer. A timeout
doesn't cancel the change: `vllm-sr config versions` shows whether it
activated.

A change that needs a restart, such as a listener's new port, is saved instead,
as the Dashboard saves one: `config apply` answers "Restart required: run
`vllm-sr serve` to apply.", `vllm-sr status` reports the saved change, and the
next `vllm-sr serve` applies it.

To serve `config.yaml` instead of the active document, Dashboard edits
included, restart with:

```bash
vllm-sr serve --config config.yaml --replace-active-config
```

## Versions

Versions count activations. The first configuration a Router serves is version
1, and each activation takes the next number. A Router that restarts on the
newest recorded document keeps its version; any other document takes the next
one. A rollback activates a new version too.

| Where | What it shows |
| --- | --- |
| `x-vsr-config-version` response header | The version that served the request. |
| `GET /api/v1/config` | The serving version and document hash, in the `x-vsr-config-version` and `x-vsr-config-hash` headers. |
| `GET /api/v1/config/hash` | `active_version`, and `last_rejection` once a change was rejected. |
| `llm_config_active_version` and `llm_config_active_info` | The serving version, and that version and its hash as labels. |

## Rejected changes

A rejected change leaves the serving version in place. Its reasons say where it
failed: each has a `stage`, a `code`, a `path` into the document when one is
known, and a `message`.

| Code | Meaning |
| --- | --- |
| `invalid_document` | The document does not parse or is not in the canonical layout. |
| `duplicate_name` | Two resources of one kind share a name. |
| `unresolved_reference` | A resource refers to one that does not exist. |
| `invalid_resource` | A resource is invalid. |
| `restart_required` | The change needs a restart, such as a standalone listener's new address or port. |
| `unsupported` | The change needs a feature the gateway mode lacks; the message names the `--gateway` mode that has it. |
| `artifact_unavailable` | A model artifact the change needs is not available. |
| `model_unavailable` | A model the change needs could not be prepared. |
| `build_failed` | The routing pipeline or the upstream layer could not be built. |
| `warmup_failed` | A component failed to warm up. |
| `activation_failed` | The new version could not take over. |
| `shutting_down` | The Router is shutting down. |
| `canceled` | The change was canceled before it finished. |

A configuration mutation through the management API waits for the result. It
returns `config_version` when the change activated. When the change is
rejected it returns `status: activation_failed` with `activation.reasons`; the
document stays persisted, so correct it with `PUT` or roll back. Both are
checked against the version that serves, not against the rejected document, so
they work even when it does not parse. `GET /api/v1/config/hash` keeps the last
rejection.
`llm_config_updates_total{result="failed"}` counts rejections by source and
stage, and `llm_config_last_rejection_timestamp_seconds` records when the last
one happened.

The management audit (`GET /api/v1/observability/audit`) records every
activation, rejection and superseded change as `config.activate`,
`config.reject` and `config.supersede`, with the version, the document hash,
the source and the management request that caused it.

## History and rollback

The Router records every activation in a history of the last 10 versions
(`-config-history-limit` changes it). Every configuration file has its own
history, so Routers whose files share a directory never mix theirs:

- A workspace's `config.yaml` keeps it in `.vllm-sr/config-backups` beside the
  file, where the dashboard keeps its backups too.
- Any other file keeps it in a directory named after the file inside
  `.vllm-sr/config-backups`.
- `VLLM_SR_CONFIG_BACKUP_DIR` moves it.

List the history, newest first, and roll back to a recorded version:

```bash
vllm-sr config versions
vllm-sr config rollback 3
```

`vllm-sr config rollback` takes a version number or the timestamp of an older
backup. Through the API, list the history with `GET /api/v1/config/versions`
and post the version to `POST /api/v1/config/rollback`. Like every
configuration change, a rollback must send the persisted document's `ETag` in
`If-Match`, or it is refused with `428`
([Read and change router configuration](../api/apiserver#read-and-change-router-configuration)):

```bash
etag=$(curl -s -o /dev/null -D - http://localhost:8080/api/v1/config \
  -H "Authorization: Bearer ${VSR_MGMT_TOKEN}" | awk 'tolower($1) == "etag:" {print $2}' | tr -d '\r')
curl -X POST http://localhost:8080/api/v1/config/rollback \
  -H "Authorization: Bearer ${VSR_MGMT_TOKEN}" \
  -H "If-Match: ${etag}" \
  -H "Content-Type: application/json" \
  -d '{"config_version": 3}'
```

A rollback never changes the recorded version: it writes the recorded document
and activates it as a new version whose `rollback_of` names the restored one.
Versions that came from a Kubernetes resource record no document; restore those
at their source.

On Kubernetes, the Helm chart keeps a single replica's history on the models
volume, so it survives the rollout that a ConfigMap change requires. With
several replicas, autoscaling or no persistence, each Pod keeps its own history
for as long as it runs; restore an earlier document through the ConfigMap
there.
