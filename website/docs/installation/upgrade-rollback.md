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
| **Versioned** | `v0.3.0` / `0.3.0` | Tagged releases only | Production — immutable, recommended |
| **Nightly** | `nightly-20260115` | Daily at 02:00 UTC | Pre-release testing |
| **Latest** | `latest` | Every push to `main` + releases | Development only |

:::tip Recommendation
Always use a **versioned** tag in production. It is immutable — the digest
never changes for a given version tag. Find the latest release on the
[GitHub Releases page](https://github.com/vllm-project/semantic-router/releases).
:::

---

## Prerequisites

- `helm` ≥ 3.14 (for Helm OCI operations)
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
# first, then re-applies your previous overrides on top. This is safer than
# --reuse-values alone, which breaks if the new chart adds new required values.
helm upgrade semantic-router \
  oci://ghcr.io/vllm-project/charts/semantic-router \
  --version 0.3.0 \
  --namespace vllm-semantic-router-system \
  --reset-then-reuse-values \
  --wait \
  --timeout 10m
```

:::caution Use `--reset-then-reuse-values`, not `--reuse-values`, for cross-version upgrades
`--reuse-values` only merges your old stored values and skips new chart defaults,
which causes template errors when a new chart version introduces new required
values. `--reset-then-reuse-values` (Helm ≥ 3.14) resets to the new defaults
first, then re-applies your overrides — it is always safe to use.
If you are on Helm < 3.14, supply your configuration explicitly with `-f your-values.yaml` instead.
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

# Get the immutable digest for maximum pinning stability
DIGEST=$(docker inspect --format='{{index .RepoDigests 0}}' \
  ghcr.io/vllm-project/semantic-router/extproc:v0.3.0)
echo "Use digest: ${DIGEST}"
```

For Kubernetes manifests, pin to the digest, not the tag:

```yaml
image: ghcr.io/vllm-project/semantic-router/extproc@sha256:<digest>
```

### 2c. Python CLI upgrade

```bash
pip install --upgrade vllm-sr==0.3.0
vllm-sr --version    # verify
```

To upgrade to the latest stable release:

```bash
pip install --upgrade vllm-sr
```

---

## 3. Rollback

### 3a. Helm rollback (fastest path)

Helm keeps a history of every deployed revision. Rolling back requires no
re-download and takes effect immediately.

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

Alternatively, roll back by re-installing an older chart version:

```bash
helm upgrade semantic-router \
  oci://ghcr.io/vllm-project/charts/semantic-router \
  --version 0.2.0 \
  --namespace vllm-semantic-router-system \
  --reset-then-reuse-values \
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

### Makefile variables

When building or deploying locally via `make`, override these variables to
target a specific release instead of `latest`:

```bash
# Use a specific image tag for all docker-* targets
make docker-build-extproc DOCKER_TAG=v0.3.0

# Pull all production images at a specific version
make docker-pull-release DOCKER_TAG=v0.3.0

# Install/upgrade the Helm chart at a pinned chart version
make helm-upgrade-version CHART_VERSION=0.3.0
```

### Helm values file (recommended for long-running environments)

Create a `values-production.yaml` that explicitly pins image tags:

```yaml
image:
  tag: "v0.3.0"   # pin to an immutable release tag
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

Nightly images and charts are built every day at 02:00 UTC and tagged
`nightly-YYYYMMDD`. They are intended for pre-release testing only.

```bash
# Pull the nightly image built on a specific date
docker pull ghcr.io/vllm-project/semantic-router/vllm-sr:nightly-20260115

# Install the nightly Helm chart
helm install semantic-router \
  oci://ghcr.io/vllm-project/charts/semantic-router \
  --version 0.0.0-nightly.20260115 \
  --namespace vllm-semantic-router-system --create-namespace
```

Nightly builds are **not** automatically promoted to a versioned release. Promotion
happens only via a tagged release.

---

## 6. Promotion Policy

```
nightly-YYYYMMDD  ──→  (manual QA + CI green)  ──→  v0.3.0
```

A nightly build is promoted to a release by:

1. Verifying all CI checks pass on the candidate commit.
2. Bumping version fields in `src/vllm-sr/pyproject.toml` and `candle-binding/Cargo.toml` to the target version.
3. Pushing a `v<version>` tag — this triggers `docker-release.yml`, `helm-publish.yml`, `pypi-publish.yml`, `publish-crate.yml`, and `release.yml` simultaneously.
4. The `release.yml` workflow validates all surfaces are consistent before the GitHub Release is created.

There is no automated gating from nightly → release; that decision is made by
the release owner.

---

## 7. Troubleshooting

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
