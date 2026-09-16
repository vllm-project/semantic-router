---
sidebar_position: 10
translation:
  source_commit: "33349fdab9ad294da19ebd11588f8adbe8771b4a"
  source_file: "docs/installation/upgrade-rollback.md"
  outdated: false
---

# 升级与回滚

本运行手册介绍如何在生产环境中升级、固定和回滚 vLLM Semantic Router 的每个发行表面。

---

## 发行通道

| 通道 | 标签模式 | 更新时机 | 用途 |
|---------|-------------|------------|----------|
| **Versioned** | `v0.3.0` / `0.3.0` | 仅带标签的发行 | 生产发行标识符；在需要不可变时验证并固定 digest |
| **Nightly** | `nightly-YYYYMMDD` | 带日期戳的构建 | 预发行测试 |
| **Latest** | `latest` | `main` 上受影响的镜像变更 + 发行 | 仅用于开发 |

:::tip 建议
在生产中使用 **versioned** 发行，然后记录已解析的产物 digest。标签是可读的发行标识符；只有经过验证的 digest 才是不可变引用。已发布的发行见 [GitHub Releases 页面](https://github.com/vllm-project/semantic-router/releases)。
:::

---

## 前置条件

- 使用 `--reset-then-reuse-values` 时需要 `helm` ≥ 3.14
- 已为你的目标集群配置 `kubectl`
- `pip` ≥ 22（用于 Python CLI）
- `docker` 或 `podman`（用于直接镜像操作）

---

## 1. 检查当前版本

### Helm release

```bash
helm list -n vllm-semantic-router-system
helm history semantic-router -n vllm-semantic-router-system
```

`CHART` 列显示 chart 版本（例如 `semantic-router-0.2.0`），`APP VERSION` 显示该 chart 部署的镜像标签。

### 正在运行的容器镜像

```bash
# 获取 extproc deployment 当前使用的镜像标签
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

始终升级到特定版本。在生产中永远不要依赖 `latest`。

```bash
# 先拉取 chart 元数据（可选，但有助于验证它存在）
helm show chart oci://ghcr.io/vllm-project/charts/semantic-router --version 0.3.0

# 升级到特定版本
# --reset-then-reuse-values（Helm ≥ 3.14）先重置为新 chart 的默认值，
# 然后在其上重新应用你先前的覆盖。复核结果清单，因为重命名
# 或不兼容的 values 仍需要迁移。
helm upgrade semantic-router \
  oci://ghcr.io/vllm-project/charts/semantic-router \
  --version 0.3.0 \
  --namespace vllm-semantic-router-system \
  --reset-then-reuse-values \
  --wait \
  --timeout 10m
```

:::caution 每次 chart 升级前都要复核 values
`--reuse-values` 会跳过新的 chart 默认值，并可能在发行添加必需 values 时失败。`--reset-then-reuse-values`（Helm ≥ 3.14）从新默认值开始，但不能迁移已重命名、已删除或不兼容的 values。阅读发行说明，并在应用之前渲染或 diff 拟议清单。如果你使用 Helm < 3.14，请用 `-f your-values.yaml` 显式提供经过复核的 values 文件。
:::

升级后验证：

```bash
helm status semantic-router -n vllm-semantic-router-system
kubectl rollout status deployment/semantic-router -n vllm-semantic-router-system
```

### 2b. Docker 镜像升级（非 Helm 部署）

在 [GitHub Releases 页面](https://github.com/vllm-project/semantic-router/releases)查找最新版本，然后：

```bash
# 按版本标签拉取（如果使用 podman，将 docker 替换为 podman）
docker pull ghcr.io/vllm-project/semantic-router/extproc:v0.3.0
docker pull ghcr.io/vllm-project/semantic-router/vllm-sr:v0.3.0

# 读取多架构索引 digest，而不是平台特定清单。
DIGEST=$(docker buildx imagetools inspect \
  ghcr.io/vllm-project/semantic-router/extproc:v0.3.0 \
  --format '{{.Manifest.Digest}}')
echo "Use digest: ${DIGEST}"
```

对于 Kubernetes 清单，固定到 digest，而不是标签：

```yaml
image: ghcr.io/vllm-project/semantic-router/extproc@sha256:<digest>
```

完整发行的已发布版本化镜像：

| 镜像 | 典型所有者 |
|-------|---------------|
| `ghcr.io/vllm-project/semantic-router/extproc:v0.3.0` | Router ExtProc 运行时 |
| `ghcr.io/vllm-project/semantic-router/extproc-rocm:v0.3.0` | ROCm router ExtProc 运行时 |
| `ghcr.io/vllm-project/semantic-router/vllm-sr:v0.3.0` | 本地/运行时 CLI 镜像 |
| `ghcr.io/vllm-project/semantic-router/vllm-sr-rocm:v0.3.0` | ROCm 本地/运行时 CLI 镜像 |
| `ghcr.io/vllm-project/semantic-router/dashboard:v0.3.0` | 控制面板后端/前端镜像 |
| `ghcr.io/vllm-project/semantic-router/operator:v0.3.0` | Kubernetes operator 镜像 |
| `ghcr.io/vllm-project/semantic-router/operator-bundle:v0.3.0` | Operator bundle 镜像 |

镜像仓库不一定发布相同的发行通道。在将平台特定镜像添加到生产清单之前，先在 GHCR 中验证精确的标签或 digest。

### 2c. Python CLI 升级

```bash
pip install --upgrade vllm-sr==0.3.0
vllm-sr --version    # 验证
```

升级到最新稳定发行：

```bash
pip install --upgrade vllm-sr
```

#### 对先前 Fleet Simulator sidecar 的一次性清理

当前发行不会在 `vllm-sr serve` 生命周期中构建或启动 Fleet Simulator，并且 `vllm-sr stop` 有意不管理独立模拟器。从会自动启动旧 sidecar 的发行升级时，先检查该精确的遗留容器（如果当时使用的运行时是 `podman`，请替换）：

```bash
docker container inspect vllm-sr-sim-container \
  --format '{{.Name}}\t{{.Config.Image}}\t{{.State.Status}}'
```

仅当部署历史确认此精确容器是旧的自动管理 sidecar 时，才一次性移除它：

```bash
docker stop vllm-sr-sim-container
docker rm vllm-sr-sim-container
```

不要移除用独立包、独立 Make 目标或自定义部署显式启动的 Fleet Simulator 实例。这些实例独立于 Router 运行时，并且仍然受支持。

### 2d. Fleet simulator Python 包升级

`vllm-sr-sim` 是一个单独的 PyPI 包，有自己的发行节奏。检查已发布的版本，然后固定一个与你的环境匹配的版本。选择开发发行时包含 `--pre`：

```bash
python -m pip index versions --pre vllm-sr-sim
pip install --upgrade --pre vllm-sr-sim==<published-version>
```

Fleet Simulator 有独立的版本流。将其包版本与 Router 发行分开固定。

---

## 3. 回滚

### 3a. Helm 回滚（最快路径）

Helm 为每个 revision 存储发行 values 和清单。回滚会创建一次新的发布；节点可能仍需要拉取较旧的镜像，因此在将其视为完成之前，等待工作负载就绪。

```bash
# 查看历史
helm history semantic-router -n vllm-semantic-router-system

# 回滚到上一个 revision
helm rollback semantic-router -n vllm-semantic-router-system --wait

# 回滚到特定 revision 编号（例如 revision 3）
helm rollback semantic-router 3 -n vllm-semantic-router-system --wait

# 验证
helm status semantic-router -n vllm-semantic-router-system
kubectl rollout status deployment/semantic-router -n vllm-semantic-router-system
```

如果 Helm 历史不可用，仅用为该发行保存并测试过的 values 安装较旧的 chart：

```bash
helm upgrade semantic-router \
  oci://ghcr.io/vllm-project/charts/semantic-router \
  --version 0.2.0 \
  --namespace vllm-semantic-router-system \
  -f values-0.2.0.yaml \
  --wait
```

### 3b. Docker / Kubernetes 清单回滚

如果直接管理 Kubernetes 清单（不使用 Helm），使用内置发布历史将 Deployment 回滚到上一个 revision：

```bash
# 查看发布历史
kubectl rollout history deployment/semantic-router -n vllm-semantic-router-system

# 撤销上一次发布
kubectl rollout undo deployment/semantic-router -n vllm-semantic-router-system

# 撤销到特定 revision
kubectl rollout undo deployment/semantic-router \
  --to-revision=3 -n vllm-semantic-router-system

# 验证
kubectl rollout status deployment/semantic-router -n vllm-semantic-router-system
```

如果使用固定的镜像 digest，将清单更新为先前的镜像 digest 并 `kubectl apply`。

### 3c. Python CLI 回滚

```bash
pip install vllm-sr==0.2.0
vllm-sr --version
```

---

## 4. 版本固定参考

### Helm values 文件

创建一个显式固定镜像标签的 `values-production.yaml`：

```yaml
image:
  tag: "v0.3.0"   # 可读的发行标签；在需要不可变时使用 digest
  pullPolicy: IfNotPresent
```

然后用以下方式部署：

```bash
helm upgrade semantic-router \
  oci://ghcr.io/vllm-project/charts/semantic-router \
  --version 0.3.0 \
  -f values-production.yaml \
  --namespace vllm-semantic-router-system
```

---

## 5. Nightly 构建

Nightly 镜像使用 `nightly-YYYYMMDD`；nightly chart 版本使用 `0.0.0-nightly.YYYYMMDD`。它们仅用于预发行测试，较旧的日期可能不再保留。在固定之前先发现可用日期：

```bash
# 需要 oras CLI。检查两个仓库，因为镜像和 chart
# 的保留可能不同。
oras repo tags ghcr.io/vllm-project/semantic-router/vllm-sr \
  | grep -E '^nightly-[0-9]{8}$' | sort -V | tail
oras repo tags ghcr.io/vllm-project/charts/semantic-router \
  | grep -E '^0\.0\.0-nightly\.[0-9]{8}$' | sort -V | tail
```

选择两个列表中都存在的日期，然后在部署之前验证精确产物：

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

Nightly 构建不会自动晋升为带版本的发行。将它们用于预发行验证，而不是作为未固定的生产通道。

---

## 6. 故障排查

### Helm：`Error: chart not found`

```bash
# 列出 OCI 注册表中的可用版本（需要 oras CLI）
oras repo tags ghcr.io/vllm-project/charts/semantic-router

# 在安装之前验证特定版本存在
helm show chart oci://ghcr.io/vllm-project/charts/semantic-router --version 0.3.0
```

### Helm：失败升级后 release 处于损坏状态

```bash
helm rollback semantic-router -n vllm-semantic-router-system --wait
# 如果回滚也因不良状态失败，强制重新安装：
helm uninstall semantic-router -n vllm-semantic-router-system
helm install semantic-router \
  oci://ghcr.io/vllm-project/charts/semantic-router \
  --version <last-known-good> \
  -f your-values.yaml \
  --namespace vllm-semantic-router-system --create-namespace
```

### Kubernetes：升级后出现 `ImagePullBackOff`

镜像标签可能尚不存在（发行仍在发布），或缺少 pull secret。检查：

```bash
kubectl describe pod -n vllm-semantic-router-system <pod-name>
# 查找 "ErrImagePull" 以及失败的精确标签
```

如果标签确实不存在，在发行完成期间回滚：

```bash
helm rollback semantic-router -n vllm-semantic-router-system
```
