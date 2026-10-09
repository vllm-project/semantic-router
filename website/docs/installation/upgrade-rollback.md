---
sidebar_position: 10
---

# Upgrade and Rollback

This runbook covers how to upgrade, pin, and roll back each release surface of
the vLLM Semantic Router in a production environment.

---

## Release Channels

| Channel | Tag pattern | Updated on | Use case |
|---------|-------------|------------|----------|
| **Versioned** | `v0.4.0` / `0.4.0` | Tagged releases only | Production release identifier; verify and pin a digest where immutability is required |
| **Nightly** | `nightly-YYYYMMDD` | Date-stamped builds | Pre-release testing |
| **Latest** | `latest` | Affected image changes on `main` + releases | Development only |

:::tip Recommendation
Use a **versioned** release in production, then record the resolved artifact
digest. A tag is a readable release identifier; only a verified digest is an
immutable reference. Find published releases on the [GitHub Releases
page](https://github.com/vllm-project/semantic-router/releases).
:::

---

## Prerequisites

- `helm` ≥ 3.14 when using `--reset-then-reuse-values`
- `kubectl` configured for your target cluster
- `pip` ≥ 22 (for Python CLI)
- `docker` or `podman` (for direct image operations)

---

## 1. Checking Your Current Version

### Helm release

```bash
helm list -n vllm-semantic-router-system
helm history semantic-router -n vllm-semantic-router-system
```

The `CHART` column shows the chart version (e.g. `semantic-router-0.2.0`) and
`APP VERSION` shows the image tag that chart deployed.

### Running container image

```bash
# Get the image tag currently used by the Router deployment
kubectl get deployment -n vllm-semantic-router-system \
  -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.spec.template.spec.containers[0].image}{"\n"}{end}'
```

### Python CLI

```bash
vllm-sr --version
pip show vllm-sr
```

---

## 2. Upgrading

### 2a. Helm chart upgrade

Always upgrade to a specific version. Never rely on `latest` in production.

```bash
# Pull the chart metadata first (optional but useful to verify it exists)
helm show chart oci://ghcr.io/vllm-project/charts/semantic-router --version 0.4.0

# Upgrade to a specific version
# --reset-then-reuse-values (Helm ≥ 3.14) resets to the new chart's defaults
# first, then re-applies your previous overrides on top. Review the resulting
# manifests because renamed or incompatible values still require migration.
helm upgrade semantic-router \
  oci://ghcr.io/vllm-project/charts/semantic-router \
  --version 0.4.0 \
  --namespace vllm-semantic-router-system \
  --reset-then-reuse-values \
  --wait \
  --timeout 10m
```

From a source checkout, `make helm-upgrade-version CHART_VERSION=0.4.0`
uses the same versioned chart with the configured cluster context.

:::caution Review values before every chart upgrade
`--reuse-values` skips new chart defaults and can break when a release adds
required values. `--reset-then-reuse-values` (Helm ≥ 3.14) starts from the new
defaults, but it cannot migrate renamed, removed, or incompatible values. Read
the release notes and render or diff the proposed manifests before applying
them. If you are on Helm < 3.14, supply a reviewed values file explicitly with
`-f your-values.yaml`.
:::

:::warning The chart serves standalone mode by default
From the first release with standalone mode
([#4623](https://github.com/vllm-project/semantic-router/issues/4623)), the
chart sets `gateway.mode: standalone`: the Router serves the OpenAI-compatible
API on its own listeners and no longer serves ext_proc on port 50051. If Envoy
Gateway, Agent Router, Istio, KServe, llm-d or another gateway calls the Router
over ext_proc, add `--set gateway.mode=extproc` (or `gateway: {mode: extproc}`
in your values file) to the upgrade. An upgrade whose live config still has the
old default listeners `grpc-50051` and `http-8080` fails to render in
standalone mode, before anything changes. `helm rollback` restores the previous
release and its mode.
:::

Verify after upgrade:

```bash
helm status semantic-router -n vllm-semantic-router-system
kubectl rollout status deployment/semantic-router -n vllm-semantic-router-system
```

### 2b. Docker image upgrade (non-Helm deployments)

Find the latest version on the [GitHub Releases page](https://github.com/vllm-project/semantic-router/releases), then:

```bash
# Pull by version tag (substitute podman for docker if using podman)
docker pull ghcr.io/vllm-project/semantic-router/vllm-sr:v0.4.0

# Read the multi-architecture index digest, not a platform-specific manifest.
DIGEST=$(docker buildx imagetools inspect \
  ghcr.io/vllm-project/semantic-router/vllm-sr:v0.4.0 \
  --format '{{.Manifest.Digest}}')
echo "Use digest: ${DIGEST}"
```

From a source checkout, `make docker-pull-release DOCKER_TAG=v0.4.0`
pulls the full set of production release images.

For Kubernetes manifests, pin to the digest, not the tag:

```yaml
image: ghcr.io/vllm-project/semantic-router/vllm-sr@sha256:<digest>
```

Published versioned images for a full release:

| Image | Typical owner |
|-------|---------------|
| `ghcr.io/vllm-project/semantic-router/vllm-sr:v0.4.0` | Router image for `vllm-sr serve`, Helm and the Operator (CPU) |
| `ghcr.io/vllm-project/semantic-router/vllm-sr-cuda:v0.4.0` | The Router image for NVIDIA GPUs |
| `ghcr.io/vllm-project/semantic-router/vllm-sr-rocm:v0.4.0` | The Router image for AMD GPUs |
| `ghcr.io/vllm-project/semantic-router/dashboard:v0.4.0` | Dashboard backend/frontend image |
| `ghcr.io/vllm-project/semantic-router/operator:v0.4.0` | Kubernetes operator image |
| `ghcr.io/vllm-project/semantic-router/operator-bundle:v0.4.0` | Operator bundle image |

Image repositories do not necessarily publish identical release channels.

Releases up to v0.4.0 also published `extproc` and `extproc-rocm`, the
Kubernetes Router images. From the first release with standalone mode one
image family serves every launcher; for that release `vllm-sr` is also
published as `extproc` and `vllm-sr-rocm` as `extproc-rocm`, with the same
digests, so pinned manifests keep working while you move them to `vllm-sr`.
Verify the exact tag or digest in GHCR before adding a platform-specific image
to a production manifest.

### 2c. Python CLI upgrade

```bash
pip install --upgrade vllm-sr==0.4.0
vllm-sr --version    # verify
```

To upgrade to the latest stable release:

```bash
pip install --upgrade vllm-sr
```

#### Fleet Simulator is removed

The Fleet Simulator (`vllm-sr-sim`) is no longer part of Semantic Router:
releases publish neither its package nor its image, and the CLI never starts
it. Installed copies keep working as they are. If an earlier `vllm-sr serve`
started its sidecar automatically, remove that container once (substitute
`podman` if that was the runtime used):

```bash
docker container inspect vllm-sr-sim-container \
  --format '{{.Name}}\t{{.Config.Image}}\t{{.State.Status}}'
docker stop vllm-sr-sim-container
docker rm vllm-sr-sim-container
```

---

## 3. Rollback

### 3a. Helm rollback (fastest path)

Helm stores the release values and manifests for each revision. A rollback
creates a new rollout; nodes may still need to pull an older image, so wait for
workload readiness before treating it as complete.

```bash
# View history
helm history semantic-router -n vllm-semantic-router-system

# Roll back to the previous revision
helm rollback semantic-router -n vllm-semantic-router-system --wait

# Roll back to a specific revision number (e.g. revision 3)
helm rollback semantic-router 3 -n vllm-semantic-router-system --wait

# Verify
helm status semantic-router -n vllm-semantic-router-system
kubectl rollout status deployment/semantic-router -n vllm-semantic-router-system
```

If Helm history is unavailable, install an older chart only with the values
saved and tested for that release:

```bash
helm upgrade semantic-router \
  oci://ghcr.io/vllm-project/charts/semantic-router \
  --version 0.2.0 \
  --namespace vllm-semantic-router-system \
  -f values-0.2.0.yaml \
  --wait
```

### 3b. Docker / Kubernetes manifest rollback

If you are managing Kubernetes manifests directly (without Helm), roll back the
Deployment to the previous revision using the built-in rollout history:

```bash
# View rollout history
kubectl rollout history deployment/semantic-router -n vllm-semantic-router-system

# Undo the last rollout
kubectl rollout undo deployment/semantic-router -n vllm-semantic-router-system

# Undo to a specific revision
kubectl rollout undo deployment/semantic-router \
  --to-revision=3 -n vllm-semantic-router-system

# Verify
kubectl rollout status deployment/semantic-router -n vllm-semantic-router-system
```

If using pinned image digests, update your manifest to the previous image digest
and `kubectl apply`.

### 3c. Python CLI rollback

```bash
pip install vllm-sr==0.2.0
vllm-sr --version
```

---

## 4. Version Pinning Reference

### Helm values file

Create a `values-production.yaml` that explicitly pins image tags:

```yaml
image:
  tag: "v0.4.0"   # readable release tag; use a digest when immutability is required
  pullPolicy: IfNotPresent
```

Then deploy with:

```bash
helm upgrade semantic-router \
  oci://ghcr.io/vllm-project/charts/semantic-router \
  --version 0.4.0 \
  -f values-production.yaml \
  --namespace vllm-semantic-router-system
```

---

## 5. Nightly Builds

Nightly images use `nightly-YYYYMMDD`; nightly chart versions use
`0.0.0-nightly.YYYYMMDD`. They are intended for pre-release testing only, and
older dates may no longer be retained. Discover an available date before
pinning it:

```bash
# Requires the oras CLI. Inspect both repositories because image and chart
# retention can differ.
oras repo tags ghcr.io/vllm-project/semantic-router/vllm-sr \
  | grep -E '^nightly-[0-9]{8}$' | sort -V | tail
oras repo tags ghcr.io/vllm-project/charts/semantic-router \
  | grep -E '^0\.0\.0-nightly\.[0-9]{8}$' | sort -V | tail
```

Choose a date that exists in both lists, then verify the exact artifacts before
deploying them:

```bash
export NIGHTLY_DATE=<available-YYYYMMDD>

docker pull \
  "ghcr.io/vllm-project/semantic-router/vllm-sr:nightly-${NIGHTLY_DATE}"

helm show chart oci://ghcr.io/vllm-project/charts/semantic-router \
  --version "0.0.0-nightly.${NIGHTLY_DATE}"

helm install semantic-router \
  oci://ghcr.io/vllm-project/charts/semantic-router \
  --version "0.0.0-nightly.${NIGHTLY_DATE}" \
  --namespace vllm-semantic-router-system --create-namespace
```

Nightly builds are not automatically promoted to a versioned release. Use them
for pre-release validation, not as an unpinned production channel.

---

## 6. Troubleshooting

### Helm: `Error: chart not found`

```bash
# List available versions in the OCI registry (requires oras CLI)
oras repo tags ghcr.io/vllm-project/charts/semantic-router

# Verify a specific version exists before installing
helm show chart oci://ghcr.io/vllm-project/charts/semantic-router --version 0.4.0
```

### Helm: release is in a broken state after failed upgrade

```bash
helm rollback semantic-router -n vllm-semantic-router-system --wait
# If rollback also fails due to a bad state, force-reinstall:
helm uninstall semantic-router -n vllm-semantic-router-system
helm install semantic-router \
  oci://ghcr.io/vllm-project/charts/semantic-router \
  --version <last-known-good> \
  -f your-values.yaml \
  --namespace vllm-semantic-router-system --create-namespace
```

### Kubernetes: `ImagePullBackOff` after upgrade

The image tag may not exist yet (release still publishing) or the pull secret
is missing. Check:

```bash
kubectl describe pod -n vllm-semantic-router-system <pod-name>
# Look for "ErrImagePull" and the exact tag that failed
```

If the tag genuinely does not exist, roll back while the release completes:

```bash
helm rollback semantic-router -n vllm-semantic-router-system
```
