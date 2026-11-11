---
title: Data and Storage
description: Choose backing services for cache, memory, replay, response history, and vector data.
---

# Data and Storage

Semantic Router can run without an external vector database, but several
optional capabilities persist or share data. Choose stores by capability,
durability, scale, and data-handling policy rather than by product name alone.

## What may be stored

| Capability | Typical data | Why it is stored |
| --- | --- | --- |
| Response cache | Request representation and prior response | Reuse a previous answer for an equivalent request. |
| Agentic memory | Embedded memories and metadata | Retrieve relevant information across turns or sessions. |
| Vector store | Documents, chunks, embeddings, and file metadata | Support retrieval and uploaded knowledge. |
| Router Replay | Route metadata and, when enabled, bounded request and response bodies | Inspect and evaluate routing behavior. |
| Response API | Recent response records | Retrieve generated responses through the management surface. |

These stores have different privacy and retention implications. A local model
route is not end-to-end private if its prompts or responses are written to an
unapproved shared store.

## Available guides

### Valkey

[Valkey Agentic Memory](valkey-memory) describes a Redis-compatible,
single-service option with vector search. It is useful for smaller deployments
or teams that already operate Valkey.

### Qdrant

[Qdrant](qdrant) covers Docker and Kubernetes deployment plus Router bindings.
It can back semantic cache, memory, vector-store, and replay configurations.

### Milvus

[Milvus](milvus) covers persistent vector storage, Kubernetes deployment,
monitoring, migration, and recovery. It is suited to larger or shared vector
collections, but has more operational dependencies than a single-service
store.

### In-memory stores

In-memory backends are useful for local experiments and short-lived caches.
They do not provide restart durability or cross-replica sharing. Do not infer
that ephemeral means insensitive: content remains readable within the running
process and may still appear in logs or diagnostics.

## Selection questions

Before enabling a store, decide:

1. Which Router capability will use it?
2. Must data survive a Router restart?
3. Must several Router replicas share the same state?
4. What request or response content can be captured?
5. What are the retention, deletion, encryption, and access requirements?
6. Who backs up and restores the service?

Then configure the shared store under `global.stores` or the relevant service
block, and enable only the route plugins that need it. See
[Stores and Tools](../tutorials/global/stores-and-tools) for the canonical
configuration surface.

## Files on the Router's disk {#router-local-files}

Two features keep files on the Router's own disk rather than in a service:

| Feature | Setting | When unset |
| --- | --- | --- |
| Vector store uploads | `global.stores.vector_store.file_storage_dir` | `/var/lib/vsr/data` |
| Router Flow tool state, `file` backend | `global.integrations.looper.flow.state.file.directory` | The user cache directory |

`/var/lib/vsr` is the Router's data directory, the one every deployment below
makes writable. Keep local storage paths under it:

```yaml title="vector-store-files"
global:
  stores:
    vector_store:
      enabled: true
      backend_type: memory
      file_storage_dir: /var/lib/vsr/data
```

At startup, before it downloads any model, the Router checks that it can write
every directory its configuration uses. When it cannot, it stops with an error
that names the setting:

```text
check local storage: global.stores.vector_store.file_storage_dir: /var/lib/vsr/data is not writable (/var/lib: read-only file system); mount a writable volume at that path, or set global.stores.vector_store.file_storage_dir to a writable directory
```

A configuration update that uses such a directory is refused with the reason
`storage_unavailable`, and the running configuration stays active.

### Helm

The chart runs the Router on a read-only root filesystem and mounts a writable
volume at `/var/lib/vsr`. By default the volume is an `emptyDir`, which a new
Pod starts empty: uploaded files are lost when the Pod is replaced, even when
`metadata_store: postgres` keeps their records. To keep them, put the data
directory on a PersistentVolumeClaim:

```yaml title="values.yaml"
data:
  persistence:
    enabled: true
    size: 20Gi
```

`storageClassName`, `accessMode`, `annotations` and `existingClaim` work as
they do for the model volume, `persistence`. Each Router replica has its own
data directory unless the claim is `ReadWriteMany`, so with several replicas a
file uploaded through one replica cannot be read through another. A volume you
mount at `/var/lib/vsr` with `extraVolumes` and `extraVolumeMounts` replaces
the chart's.

The Router's other writable state lives on the model volume, `/app/models`
(downloaded models, runtime caches and a single replica's configuration
history), and on `/tmp` (runtime sockets, startup status and the default Flow
state directory).

### Local Docker stack

`vllm-sr serve` runs the Router on a writable filesystem, so the defaults work,
but `/var/lib/vsr` is inside the Router container and a new container starts
without it. To keep uploaded files across restarts, point `file_storage_dir`
under `/app/.vllm-sr`, the stack's state directory on the host, as the
`vela-amd` recipe does with `/app/.vllm-sr/files`.
