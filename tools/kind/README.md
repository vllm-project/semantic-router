# Local Kind Cluster

This directory generates a two-node [kind](https://kind.sigs.k8s.io/) cluster
for local semantic-router development. The worker mounts the repository's
`models/` directory at `/mnt/models`, and host port `30080` is mapped to the
control-plane node.

## Prerequisites

Install Docker, kind, `kubectl`, and `envsubst`. Ensure Docker has enough memory
for a control-plane node, worker node, router, and any local model service you
intend to run.

## Create the Cluster

From the repository root:

```bash
./tools/kind/generate-kind-config.sh
kind create cluster --config tools/kind/kind-config.yaml
kubectl cluster-info --context kind-semantic-router-cluster
```

The generator resolves the repository root, creates `models/` if needed, and
writes the ignored `tools/kind/kind-config.yaml` from the checked-in template.
Regenerate it after moving the worktree or changing the template.

This directory creates only the cluster. Choose a maintained deployment path
from the [installation documentation](../../website/docs/installation/installation.md)
instead of applying `deploy/kubernetes/` as one aggregate Kustomize target;
that directory contains several independent examples.

## Load Local Images

When a deployment uses an image that exists only in the local Docker daemon,
load it into the cluster before applying the workload:

```bash
kind load docker-image IMAGE:TAG --name semantic-router-cluster
```

Set the workload's image pull policy accordingly.

## Inspect the Mount

```bash
docker exec semantic-router-cluster-worker ls -la /mnt/models
kubectl get nodes -o wide
```

The host mount is a development convenience, not a production persistence
design. A workload must be scheduled on the worker and explicitly mount a
volume that maps to `/mnt/models` before it can see those files.

## Recreate or Remove

```bash
kind delete cluster --name semantic-router-cluster
./tools/kind/generate-kind-config.sh
kind create cluster --config tools/kind/kind-config.yaml
```

Change `kind-config.yaml.template` if you need different ports, mounts, or node
resources; do not commit the generated configuration.
