---
translation:
  source_commit: "545f5e3"
  source_file: "docs/installation/installation.md"
  outdated: true
is_mtpe: true
sidebar_position: 2
---

# 安装

本指南将帮助您安装和运行 vLLM Semantic Router。Router 完全在 CPU 上运行，推理不需要 GPU。

## 系统要求

:::note[注意]
无需 GPU - Router 使用优化的 BERT 模型在 CPU 上高效运行。
:::

**要求：**

- **Python**: 3.10 或更高版本
- **容器运行时**: Docker 或 Podman（运行 Router 容器所需）

## 快速开始

### 1. 使用一键安装脚本（macOS/Linux）

```bash
curl -fsSL https://vllm-semantic-router.com/install.sh | bash
```

这个安装器会：

- 检测 Python 3.10 或更高版本
- 将 `vllm-sr` 安装到 `~/.local/share/vllm-sr`
- 将 launcher 写入 `~/.local/bin/vllm-sr`
- 默认为 `vllm-sr serve` 准备 Docker 或 Podman，除非您显式跳过
- 默认自动执行一次 `vllm-sr serve`，并在可能时打开 dashboard
- 如果无法自动打开浏览器，会打印 dashboard 地址和远程服务器访问提示

常用变体：

```bash
# 只安装 CLI
curl -fsSL https://vllm-semantic-router.com/install.sh | bash -s -- --mode cli

# 将本地 serve 固定为 Podman
curl -fsSL https://vllm-semantic-router.com/install.sh | bash -s -- --runtime podman

# 强制首次启动走 AMD / ROCm 路径
curl -fsSL https://vllm-semantic-router.com/install.sh | bash -s -- --platform amd

# 安装但不自动启动 serve + dashboard
curl -fsSL https://vllm-semantic-router.com/install.sh | bash -s -- --no-launch

# 跳过 runtime bootstrap，只执行用户态安装
curl -fsSL https://vllm-semantic-router.com/install.sh | bash -s -- --runtime skip
```

如果 `~/.local/bin` 还不在您的 `PATH` 中，安装器会打印需要添加的 `export` 行。

Windows 用户请使用下面的手动 PyPI 安装流程。

### 2. 手动通过 PyPI 安装

```bash
# 创建虚拟环境（推荐）
python -m venv vsr
source vsr/bin/activate  # Windows 上: vsr\Scripts\activate

# 从 PyPI 安装
pip install vllm-sr
```

验证安装：

```bash
vllm-sr --version
```

### 3. 之后再次启动 `vllm-sr`

```bash
vllm-sr serve
```

如果您没有使用 `--no-launch`，安装器已经自动帮您跑过一次 `vllm-sr serve`。

如果当前目录还没有 `config.yaml`，`vllm-sr serve` 会自动 bootstrap 一个最小 setup config，并以 setup mode 启动 dashboard。

Router 将：

- 自动下载所需的 ML 模型（约 1.5GB，一次性）
- 在端口 8700 上启动 dashboard
- 激活后在端口 8888 上启动 Envoy Proxy
- 激活后启动 Semantic Router 服务
- 在端口 9190 上启用 metrics

### 4. 打开 Dashboard

在浏览器中打开 [http://localhost:8700](http://localhost:8700)。

如果您是在远程服务器上运行安装器，且浏览器没有自动打开，请使用安装器打印出的 URL 和 SSH tunnel 提示访问 dashboard。

首次使用时：

1. 先配置一个或多个模型。
2. 选择 routing preset，或保留 single-model baseline。
3. 激活生成的配置。

激活后，`config.yaml` 会写入当前目录，Router 会退出 setup mode。

### 5. 测试 Router

```bash
curl http://localhost:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MoM",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### 6. 可选：通过 CLI 打开 Dashboard

```bash
vllm-sr dashboard
```

## 常用命令

```bash
# 查看日志
vllm-sr logs router        # Router 日志
vllm-sr logs envoy         # Envoy 日志
vllm-sr logs router -f     # 跟踪日志

# 检查状态
vllm-sr status

# 停止 Router
vllm-sr stop
```

## 高级配置

### YAML-first 工作流

如果您更倾向于直接编辑 YAML，而不是使用 dashboard setup flow：

```bash
# 在当前目录生成一个精简的高级样板
vllm-sr init

# 启动前校验它
vllm-sr validate config.yaml
```

`vllm-sr init` 是可选的。它会为 YAML-first 用户生成 advanced sample 和 `.vllm-sr/router-defaults.yaml`。`router-defaults.yaml` 保存的是高级 runtime defaults，不是首次进入 dashboard 时必须编辑的文件。

### HuggingFace 设置

启动前设置环境变量：

```bash
export HF_ENDPOINT=https://huggingface.co  # 或镜像：https://hf-mirror.com
export HF_TOKEN=your_token_here            # 仅针对 gated models
export HF_HOME=/path/to/cache              # 自定义缓存目录

vllm-sr serve
```

### 自定义选项

```bash
# 使用自定义配置文件
vllm-sr serve --config my-config.yaml

# 使用自定义 Docker 镜像
vllm-sr serve --image ghcr.io/vllm-project/semantic-router/vllm-sr:latest

# 控制镜像拉取策略
vllm-sr serve --image-pull-policy always
```

## Kubernetes 部署

在 Kubernetes 或 OpenShift 上进行生产部署时，请使用 **Kubernetes Operator**：

### 使用 Operator 快速开始

```bash
# 克隆仓库
git clone https://github.com/vllm-project/semantic-router
cd semantic-router/deploy/operator

# 安装 CRDs 和 operator
make install
make deploy IMG=ghcr.io/vllm-project/semantic-router-operator:latest

# 部署一个 semantic router 实例
kubectl apply -f config/samples/vllm_v1alpha1_semanticrouter.yaml
```

**优势：**

- ✅ 使用 Kubernetes CRDs 进行声明式配置
- ✅ 自动检测平台（OpenShift/Kubernetes）
- ✅ 内置高可用性和扩展能力
- ✅ 集成监控和可观测性
- ✅ 生命周期管理和升级

详情请参阅 **[Kubernetes Operator 指南](k8s/operator)**。

### 其他 Kubernetes 部署选项

- **[Istio 集成](k8s/istio.md)** - 服务网格部署
- **[AI Gateway](k8s/ai-gateway.md)** - Gateway API 集成
- **[生产环境堆栈](k8s/production-stack.md)** - 完整的生产环境设置
- **[Dynamo](k8s/dynamo.md)** - 动态配置管理

## Docker Compose

用于本地开发和测试：

- **[Docker Compose](docker-compose.md)** - 快速本地部署

## 下一步

- **[配置指南](configuration.md)** - 高级路由和信号配置
- **[API 文档](../api/router.md)** - 完整 API 参考
- **[教程](../tutorials/intelligent-route/keyword-routing.md)** - 通过示例学习

## 获取帮助

- **Issues**: [GitHub Issues](https://github.com/vllm-project/semantic-router/issues)
- **社区**: 加入 vLLM Slack 中的 `#semantic-router` 频道
- **文档**: [vllm-semantic-router.com](https://vllm-semantic-router.com/)
