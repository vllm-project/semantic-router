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
| **Versioned** | `v0.3.0` / `0.3.0` | Tagged releases only | Production release identifier; verify and pin a digest where immutability is required |
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
# Get the image tag currently used by the extproc deployment
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
helm show chart oci://ghcr.io/vllm-project/charts/semantic-router --version 0.3.0

# Upgrade to a specific version
# --reset-then-reuse-values (Helm ≥ 3.14) resets to the new chart's defaults
# first, then re-applies your previous overrides on top. Review the resulting
# manifests because renamed or incompatible values still require migration.
helm upgrade semantic-router \
  oci://ghcr.io/vllm-project/charts/semantic-router \
  --version 0.3.0 \
  --namespace vllm-semantic-router-system \
  --reset-then-reuse-values \
  --wait \
  --timeout 10m
```

:::caution Review values before every chart upgrade
`--reuse-values` skips new chart defaults and can break when a release adds
required values. `--reset-then-reuse-values` (Helm ≥ 3.14) starts from the new
defaults, but it cannot migrate renamed, removed, or incompatible values. Read
the release notes and render or diff the proposed manifests before applying
them. If you are on Helm < 3.14, supply a reviewed values file explicitly with
`-f your-values.yaml`.
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
docker pull ghcr.io/vllm-project/semantic-router/extproc:v0.3.0
docker pull ghcr.io/vllm-project/semantic-router/vllm-sr:v0.3.0

# Read the multi-architecture index digest, not a platform-specific manifest.
DIGEST=$(docker buildx imagetools inspect \
  ghcr.io/vllm-project/semantic-router/extproc:v0.3.0 \
  --format '{{.Manifest.Digest}}')
echo "Use digest: ${DIGEST}"
```

For Kubernetes manifests, pin to the digest, not the tag:

```yaml
image: ghcr.io/vllm-project/semantic-router/extproc@sha256:<digest>
```

Published versioned images for a full release:

| Image | Typical owner |
|-------|---------------|
| `ghcr.io/vllm-project/semantic-router/extproc:v0.3.0` | Router ExtProc runtime |
| `ghcr.io/vllm-project/semantic-router/extproc-rocm:v0.3.0` | ROCm router ExtProc runtime |
| `ghcr.io/vllm-project/semantic-router/vllm-sr:v0.3.0` | Local/runtime CLI image |
| `ghcr.io/vllm-project/semantic-router/vllm-sr-rocm:v0.3.0` | ROCm local/runtime CLI image |
| `ghcr.io/vllm-project/semantic-router/dashboard:v0.3.0` | Dashboard backend/frontend image |
| `ghcr.io/vllm-project/semantic-router/operator:v0.3.0` | Kubernetes operator image |
| `ghcr.io/vllm-project/semantic-router/operator-bundle:v0.3.0` | Operator bundle image |

Image repositories do not necessarily publish identical release channels.
Verify the exact tag or digest in GHCR before adding a platform-specific image
to a production manifest.

### 2c. Python CLI upgrade

```bash
pip install --upgrade vllm-sr==0.3.0
vllm-sr --version    # verify
```

To upgrade to the latest stable release:

```bash
pip install --upgrade vllm-sr
```

#### One-time cleanup for the former Fleet Simulator sidecar

Current releases do not build or start Fleet Simulator as part of the
`vllm-sr serve` lifecycle, and `vllm-sr stop` intentionally does not manage a
standalone simulator. When upgrading from a release where `vllm-sr serve`
automatically started the old sidecar, first inspect the exact legacy container
(substitute `podman` if that was the runtime used):

```bash
docker container inspect vllm-sr-sim-container \
  --format '{{.Name}}\t{{.Config.Image}}\t{{.State.Status}}'
```

Only when deployment history confirms that this exact container is the old
automatically managed sidecar, remove it once:

```bash
docker stop vllm-sr-sim-container
docker rm vllm-sr-sim-container
```

Do not remove a Fleet Simulator instance started explicitly with the standalone
package, standalone Make targets, or a custom deployment. Those instances are
independent of the Router runtime and remain supported.

### 2d. Fleet simulator Python package upgrade

`vllm-sr-sim` is a separate PyPI package with its own release cadence. Inspect
the published versions, then pin one that matches your environment. Include
`--pre` when selecting a development release:

```bash
python -m pip index versions --pre vllm-sr-sim
pip install --upgrade --pre vllm-sr-sim==<published-version>
```

Fleet Simulator has an independent version stream. Pin its package version
separately from the Router release.

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
  tag: "v0.3.0"   # readable release tag; use a digest when immutability is required
  pullPolicy: IfNotPresent
```

Then deploy with:

```bash
helm upgrade semantic-router \
  oci://ghcr.io/vllm-project/charts/semantic-router \
  --version 0.3.0 \
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
helm show chart oci://ghcr.io/vllm-project/charts/semantic-router --version 0.3.0
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
