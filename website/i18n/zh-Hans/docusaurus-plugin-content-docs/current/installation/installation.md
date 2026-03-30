---
sidebar_position: 2
translation:
  source_commit: "0ee41b5f"
  source_file: "docs/installation/installation.md"
  outdated: false
---

# 安装

本指南帮助您安装并运行 vLLM Semantic Router。路由器完全在 CPU 上运行，推理侧**不需要 GPU**。

## 系统要求

:::note
无需 GPU——路由器在 CPU 上使用优化的 BERT 模型高效运行。
:::

**要求：**

- **Python**：3.10 或更高
- **容器运行时**：Docker（运行路由器容器所必需）

## 快速开始

### 1. 一行安装脚本（macOS/Linux）

```bash
curl -fsSL https://vllm-semantic-router.com/install.sh | bash
```

安装脚本会：

- 检测 Python 3.10 或更新版本
- 将 `vllm-sr` 安装到 `~/.local/share/vllm-sr`
- 在 `~/.local/bin/vllm-sr` 写入启动器
- 除非您选择退出，否则为 `vllm-sr serve` 准备 Docker
- 在可能的情况下自动启动 `vllm-sr serve` 并打开控制台
- 若无法打开浏览器，则打印控制台访问方式与远程服务器提示

常用变体：

```bash
# 仅安装 CLI
curl -fsSL https://vllm-semantic-router.com/install.sh | bash -s -- --mode cli

# 跳过运行时引导，稍后自行启动 Docker
curl -fsSL https://vllm-semantic-router.com/install.sh | bash -s -- --runtime skip

# 首次启动强制走 AMD/ROCm 路径
curl -fsSL https://vllm-semantic-router.com/install.sh | bash -s -- --platform amd

# 安装但不自动启动 serve 与控制台
curl -fsSL https://vllm-semantic-router.com/install.sh | bash -s -- --no-launch

# 跳过运行时引导，仅保留用户态安装步骤
curl -fsSL https://vllm-semantic-router.com/install.sh | bash -s -- --runtime skip
```

若 `~/.local/bin` 尚未在 `PATH` 中，安装脚本会打印需要添加的 `export` 行。

### 智能体安装（OpenClaw 及类似智能体）

当需要由本地智能体代您执行安装时，请使用智能体流程。仍使用同一受支持的安装器，但应让智能体使用 **no-launch** 路径，以免在交接时自动启动 `vllm-sr serve` 或打开浏览器。

长期推荐：从本仓库发布或安装由仓库维护的 `openclaw-vsr-bridge` skill（`dashboard/backend/skillpacks/openclaw-vsr-bridge/SKILL.md`）。在注册表条目就绪前，可使用下列托管提示文件，使首页与智能体文案保持简短。

托管提示文件：

- 仅 CLI 提示：[https://vllm-semantic-router.com/install/agent/vllm-sr-cli.md](https://vllm-semantic-router.com/install/agent/vllm-sr-cli.md)
- 安装并桥接提示：[https://vllm-semantic-router.com/install/agent/openclaw-vsr-bridge.md](https://vllm-semantic-router.com/install/agent/openclaw-vsr-bridge.md)

仅 CLI 提示：

```text
Fetch and follow https://vllm-semantic-router.com/install/agent/vllm-sr-cli.md.
```

安装并桥接提示：

```text
Fetch and follow https://vllm-semantic-router.com/install/agent/openclaw-vsr-bridge.md.
```

托管 Markdown 仍指向同一受支持安装器。OpenClaw 的后续交接仍通过 bridge skill 或 `vllm-sr config import --from openclaw` 完成，而非第二套安装器。

Windows 用户请使用下文手动 PyPI 流程。

### 2. 手动 PyPI 安装

```bash
# 建议创建虚拟环境
python -m venv vsr
source vsr/bin/activate  # Windows：vsr\Scripts\activate

# 从 PyPI 安装
pip install vllm-sr
```

验证安装：

```bash
vllm-sr --version
```

### 3. 稍后重启 `vllm-sr`

```bash
vllm-sr serve
```

若未使用 `--no-launch`，安装脚本已为您运行过一次 `vllm-sr serve`。

若当前目录尚无 `config.yaml`，`vllm-sr serve` 会引导生成最小配置并以**设置模式**启动控制台。

路由器将：

- 自动下载所需 ML 模型（约 1.5GB，一次性）
- 在端口 8700 启动控制台
- 在端口 8810 启动 `vllm-sr-sim` sidecar
- 激活后在端口 8888 启动 Envoy 代理
- 激活后启动语义路由器服务
- 在端口 9190 暴露指标

### 4. 打开控制台

在浏览器中打开 [http://localhost:8700](http://localhost:8700)。

若在远程服务器上运行安装脚本且浏览器未自动打开，请使用安装脚本打印的 URL 与 SSH 隧道说明。

首次运行设置：

1. 配置一个或多个模型。
2. 选择路由预设或保留单模型基线。
3. 激活生成的配置。

激活后，会在当前目录写入 `config.yaml`，路由器退出设置模式。

### 5. 测试路由器

```bash
curl http://localhost:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MoM",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### 6.（可选）从 CLI 打开控制台

```bash
vllm-sr dashboard
```

## 常用命令

```bash
# 查看日志
vllm-sr logs router        # 路由器日志
vllm-sr logs envoy         # Envoy 日志
vllm-sr logs simulator     # Fleet 模拟器 sidecar 日志
vllm-sr logs router -f     # 跟踪日志

# 查看状态
vllm-sr status             # 含模拟器 sidecar 状态

# 停止路由器
vllm-sr stop
```

## 高级配置

### 以 YAML 为先的工作流

若希望直接编辑 YAML 而非使用控制台设置流程：

```bash
# 在 serve 之前校验 canonical 配置
vllm-sr validate config.yaml
```

v0.3 已移除 `vllm-sr init`。请直接按 canonical 布局 `version/listeners/providers/routing/global` 创建 `config.yaml`，用 `vllm-sr config migrate --config old-config.yaml` 迁移旧文件，或使用 `vllm-sr config import --from openclaw` 导入受支持的 OpenClaw 模型提供商。

### HuggingFace 设置

启动前设置环境变量：

```bash
export HF_ENDPOINT=https://huggingface.co  # 或镜像：https://hf-mirror.com
export HF_TOKEN=your_token_here            # 仅门控模型需要
export HF_HOME=/path/to/cache              # 自定义缓存目录

vllm-sr serve
```

### 自定义选项

```bash
# 指定配置文件
vllm-sr serve --config my-config.yaml

# 指定 Docker 镜像
vllm-sr serve --image ghcr.io/vllm-project/semantic-router/vllm-sr:latest

# 控制镜像拉取策略
vllm-sr serve --image-pull-policy always
```

## Kubernetes 部署

在生产环境使用 Kubernetes 或 OpenShift 时，请使用 **Kubernetes Operator**：

### 使用 Operator 快速开始

```bash
git clone https://github.com/vllm-project/semantic-router
cd semantic-router/deploy/operator

make install
make deploy IMG=ghcr.io/vllm-project/semantic-router-operator:latest

kubectl apply -f config/samples/vllm_v1alpha1_semanticrouter.yaml
```

**优势：**

- 使用 Kubernetes CRD 的声明式配置
- 自动检测平台（OpenShift/Kubernetes）
- 内置高可用与扩缩
- 集成监控与可观测
- 生命周期管理与升级

完整说明见 **[Kubernetes Operator 指南](k8s/operator)**。

### 其他 Kubernetes 部署选项

- **[Istio 集成](k8s/istio)** — 服务网格部署
- **[AI Gateway](k8s/ai-gateway)** — Gateway API 集成
- **[Production Stack](k8s/production-stack)** — 完整生产环境
- **[Dynamo](k8s/dynamo)** — 动态配置管理

## Docker Compose

本地开发与测试：

- **[Docker Compose](docker-compose)** — 快速本地部署

## 下一步

- **[配置指南](configuration)** — 高级路由与信号配置
- **[Kubernetes Operator](k8s/operator)** — 生产级 Kubernetes 部署
- **[API 文档](../api/router)** — 完整 API 参考
- **[教程](../tutorials/signal/overview)** — 示例学习

## 获取帮助

- **Issue**：[GitHub Issues](https://github.com/vllm-project/semantic-router/issues)
- **社区**：加入 vLLM Slack 的 `#semantic-router` 频道
- **文档**：[vllm-semantic-router.com](https://vllm-semantic-router.com/)
