---
sidebar_position: 10
---

# 升级与回滚（Upgrade and Rollback）

本手册覆盖了在生产环境中，如何对 vLLM Semantic Router 的各个发布面（release surface）进行升级、版本固定（pin）与回滚。

---

## 发布渠道（Release Channels）

| 渠道 | Tag 模式 | 更新频率 | 适用场景 |
|---------|-------------|------------|----------|
| **版本化** | `v0.3.0` / `0.3.0` | 仅打 tag 的正式发布 | 生产环境 — 不可变，推荐 |
| **夜间构建** | `nightly-20260115` | 每日 02:00 UTC | 预发布测试 |
| **Latest** | `latest` | 每次 push 到 `main` + 正式发布 | 仅用于开发 |

::::tip 建议
生产环境务必使用 **版本化** tag。它是不可变的：同一个版本 tag 的 digest 永远不会变化。你可以在 [GitHub Releases](https://github.com/vllm-project/semantic-router/releases) 查看最新发布版本。
::::

---

## 前置条件

- `helm` ≥ 3.14（用于 Helm OCI 操作）
- 已为目标集群配置 `kubectl`
- `pip` ≥ 22（用于 Python CLI）
- `docker` 或 `podman`（用于直接镜像操作）

---

## 1. 查看当前版本

### Helm release

```bash
helm list -n vllm-semantic-router-system
helm history semantic-router -n vllm-semantic-router-system
```

`CHART` 列显示 chart 版本（例如 `semantic-router-0.2.0`），`APP VERSION` 显示该 chart 部署的镜像 tag。

### 运行中的容器镜像

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

## 2. 升级

### 2a. Helm chart 升级

生产环境升级务必指定具体版本，切勿依赖 `latest`。

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

::::caution 跨版本升级请使用 `--reset-then-reuse-values`，而不是 `--reuse-values`
`--reuse-values` 只会合并旧版本存储的 values，并跳过新 chart 的默认值；当新 chart 引入新的必填字段时，这会导致模板渲染错误。`--reset-then-reuse-values`（Helm ≥ 3.14）会先重置为新版本默认值，再把你的覆盖项叠加上去——总是安全。

如果你使用 Helm < 3.14，请改用 `-f your-values.yaml` 显式提供配置。
::::

升级后验证：

```bash
helm status semantic-router -n vllm-semantic-router-system
kubectl rollout status deployment/semantic-router -n vllm-semantic-router-system
```

### 2b. Docker 镜像升级（非 Helm 部署）

在 [GitHub Releases](https://github.com/vllm-project/semantic-router/releases) 找到最新版本后：

```bash
# Pull by version tag (substitute podman for docker if using podman)
docker pull ghcr.io/vllm-project/semantic-router/extproc:v0.3.0
docker pull ghcr.io/vllm-project/semantic-router/vllm-sr:v0.3.0
docker pull ghcr.io/vllm-project/semantic-router/anthropic-shim:v0.3.0

# Get the immutable digest for maximum pinning stability
DIGEST=$(docker inspect --format='{{index .RepoDigests 0}}' \
  ghcr.io/vllm-project/semantic-router/extproc:v0.3.0)
echo "Use digest: ${DIGEST}"
```

对于 Kubernetes manifests，建议固定到 digest，而不是 tag：

```yaml
image: ghcr.io/vllm-project/semantic-router/extproc@sha256:<digest>
```

一次完整版本发布通常会包含这些镜像：

| 镜像 | 典型责任方 |
|-------|---------------|
| `ghcr.io/vllm-project/semantic-router/extproc:v0.3.0` | Router ExtProc 运行时 |
| `ghcr.io/vllm-project/semantic-router/extproc-rocm:v0.3.0` | ROCm Router ExtProc 运行时 |
| `ghcr.io/vllm-project/semantic-router/vllm-sr:v0.3.0` | 本地/运行时 CLI 镜像 |
| `ghcr.io/vllm-project/semantic-router/vllm-sr-rocm:v0.3.0` | ROCm 本地/运行时 CLI 镜像 |
| `ghcr.io/vllm-project/semantic-router/anthropic-shim:v0.3.0` | Anthropic 兼容 API shim 镜像 |
| `ghcr.io/vllm-project/semantic-router/dashboard:v0.3.0` | Dashboard 后端/前端镜像 |
| `ghcr.io/vllm-project/semantic-router/llm-katan:v0.3.0` | Fleet simulation 服务镜像 |
| `ghcr.io/vllm-project/semantic-router/operator:v0.3.0` | Kubernetes operator 镜像 |
| `ghcr.io/vllm-project/semantic-router/operator-bundle:v0.3.0` | Operator bundle 镜像 |

### 2c. Python CLI 升级

```bash
pip install --upgrade vllm-sr==0.3.0
vllm-sr --version    # verify
```

升级到最新稳定版：

```bash
pip install --upgrade vllm-sr
```

### 2d. Fleet simulator Python 包升级

`vllm-sr-sim` 是一个独立 PyPI 包，发布节奏与主仓库不同。若你依赖 simulator CLI 或 dashboard sidecar 包数据，请显式固定版本：

```bash
pip install --upgrade vllm-sr-sim==0.1.0
```

Fleet simulator 的发布使用独立的 `vllm-sr-sim-v<version>` tag 流与 `pypi-publish-vllm-sr-sim.yml` workflow；它不会随着 router 的 `v<version>` tag 一起发布。

---

## 3. 回滚

### 3a. Helm 回滚（最快路径）

Helm 会保留每次部署的 revision 历史。回滚不需要重新下载，可立即生效。

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

也可以通过重新安装旧 chart 版本来回滚：

```bash
helm upgrade semantic-router \
  oci://ghcr.io/vllm-project/charts/semantic-router \
  --version 0.2.0 \
  --namespace vllm-semantic-router-system \
  --reset-then-reuse-values \
  --wait
```

### 3b. Docker / Kubernetes manifest 回滚

若你直接管理 Kubernetes manifests（不使用 Helm），可通过 rollout history 回滚到之前的 revision：

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

如果你使用了镜像 digest 固定，请将 manifest 更新到上一个 digest 后再 `kubectl apply`。

### 3c. Python CLI 回滚

```bash
pip install vllm-sr==0.2.0
vllm-sr --version
```

---

## 4. 版本固定（Pinning）参考

### Makefile 变量

在本地通过 `make` 构建或部署时，可覆盖这些变量以指定版本，而不是使用 `latest`：

```bash
# Use a specific image tag for all docker-* targets
make docker-build-extproc DOCKER_TAG=v0.3.0

# Pull all production images at a specific version
make docker-pull-release DOCKER_TAG=v0.3.0

# Install/upgrade the Helm chart at a pinned chart version
make helm-upgrade-version CHART_VERSION=0.3.0
```

### Helm values 文件（推荐用于长期运行环境）

创建一个 `values-production.yaml` 显式固定镜像 tag：

```yaml
image:
  tag: "v0.3.0"   # pin to an immutable release tag
  pullPolicy: IfNotPresent
```

然后部署：

```bash
helm upgrade semantic-router \
  oci://ghcr.io/vllm-project/charts/semantic-router \
  --version 0.3.0 \
  -f values-production.yaml \
  --namespace vllm-semantic-router-system
```

---

## 5. 夜间构建（Nightly Builds）

夜间镜像与 chart 每天 02:00 UTC 构建一次，并打上 `nightly-YYYYMMDD` tag。它们仅用于预发布测试。

```bash
# Pull the nightly image built on a specific date
docker pull ghcr.io/vllm-project/semantic-router/vllm-sr:nightly-20260115

# Install the nightly Helm chart
helm install semantic-router \
  oci://ghcr.io/vllm-project/charts/semantic-router \
  --version 0.0.0-nightly.20260115 \
  --namespace vllm-semantic-router-system --create-namespace
```

夜间构建 **不会** 自动晋升为版本化发布；只有通过打 tag 的正式发布才会晋升。

---

## 6. 晋升策略（Promotion Policy）

```
nightly-YYYYMMDD  ──→  (manual QA + CI green)  ──→  v0.3.0
```

夜间构建晋升为正式发布需要：

1. 确认候选 commit 的所有 CI 检查通过。
2. 将 `src/vllm-sr/pyproject.toml` 与 `candle-binding/Cargo.toml` 的版本字段 bump 到目标版本。
3. 推送 `v<version>` tag —— 会同时触发 `docker-release.yml`、`helm-publish.yml`、`pypi-publish.yml`、`publish-crate.yml` 与 `release.yml`。
4. `release.yml` workflow 会先验证各个发布面版本一致，然后再创建 GitHub Release。

Fleet simulator 包通过 bump `src/fleet-sim/pyproject.toml` 并推送 `vllm-sr-sim-v<version>` tag 来晋升，触发 `pypi-publish-vllm-sr-sim.yml`。

nightly → release 没有自动化 gating；是否晋升由 release owner 决策。

---

## 7. 故障排查（Troubleshooting）

### Helm：`Error: chart not found`

```bash
# List available versions in the OCI registry (requires oras CLI)
oras repo tags ghcr.io/vllm-project/charts/semantic-router

# Verify a specific version exists before installing
helm show chart oci://ghcr.io/vllm-project/charts/semantic-router --version 0.3.0
```

### Helm：升级失败后 release 状态异常

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

### Kubernetes：升级后出现 `ImagePullBackOff`

镜像 tag 可能尚未发布完成（release 仍在发布），或缺少 pull secret。请检查：

```bash
kubectl describe pod -n vllm-semantic-router-system <pod-name>
# Look for "ErrImagePull" and the exact tag that failed
```

如果确认 tag 还不存在，建议在 release 完成前先回滚：

```bash
helm rollback semantic-router -n vllm-semantic-router-system
```

