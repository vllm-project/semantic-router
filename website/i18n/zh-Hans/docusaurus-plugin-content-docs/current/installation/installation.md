---
sidebar_position: 2
translation:
  source_commit: "0ee41b5f"
  source_file: "docs/installation/installation.md"
  outdated: false
---

# 快速开始

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
- 将最新开发版 `vllm-sr` 安装到 `~/.local/share/vllm-sr`
- 在 `~/.local/bin/vllm-sr` 写入启动器
- 除非您选择退出，否则为 `vllm-sr serve` 准备 Docker
- 在可能的情况下自动启动 `vllm-sr serve` 并打开控制台
- 若无法打开浏览器，则打印控制台访问方式与远程服务器提示

若 `~/.local/bin` 尚未在 `PATH` 中，安装脚本会打印需要添加的 `export` 行。

若您需要最新稳定版，请运行：

```bash
curl -fsSL https://vllm-semantic-router.com/install.sh | bash -s -- --channel stable
```

Windows 用户请使用下文手动 PyPI 流程。

### 2. 手动 PyPI 安装

```bash
# 建议创建虚拟环境
python -m venv vsr
source vsr/bin/activate  # Windows：vsr\Scripts\activate

# 安装最新开发版
pip install --pre vllm-sr

# 若需要最新稳定版
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

# 设置路由器日志级别
vllm-sr serve --log-level debug

# 指定 Docker 镜像
vllm-sr serve --image ghcr.io/vllm-project/semantic-router/vllm-sr:latest

# 控制镜像拉取策略
vllm-sr serve --image-pull-policy always
```

## Docker Compose

本地开发与测试：

- **[Docker Compose](docker-compose)** — 快速本地部署

## 下一步

- **[Install with Operator](k8s/operator)** — 使用 Operator 在 Kubernetes 或 OpenShift 上部署
- **[配置指南](configuration)** — 高级路由与信号配置
- **[API 文档](../api/router)** — 完整 API 参考
- **[教程](../tutorials/signal/overview)** — 示例学习

## 获取帮助

- **Issue**：[GitHub Issues](https://github.com/vllm-project/semantic-router/issues)
- **社区**：加入 vLLM Slack 的 `#semantic-router` 频道
- **文档**：[vllm-semantic-router.com](https://vllm-semantic-router.com/)
