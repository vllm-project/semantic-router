---
translation:
  source_commit: "a0e504f"
  source_file: "docs/installation/installation.md"
  outdated: false
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
- **Docker**: 运行 Router 容器所需

## 快速开始

### 1. 安装 vLLM Semantic Router

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

### 2. 启动 `vllm-sr`

```bash
vllm-sr serve
```

如果当前目录还没有 `config.yaml`，`vllm-sr serve` 会自动 bootstrap 一个最小工作区，并以 setup mode 启动 dashboard。

Router 将：

- 自动下载所需的 ML 模型（约 1.5GB，一次性）
- 在端口 8700 上启动 dashboard
- 激活后在端口 8888 上启动 Envoy Proxy
- 激活后启动 Semantic Router 服务
- 在端口 9190 上启用 metrics

### 3. 打开 Dashboard

在浏览器中打开 [http://localhost:8700](http://localhost:8700)。

首次使用时：

1. 先配置一个或多个模型。
2. 选择 routing preset，或保留 single-model baseline。
3. 激活生成的配置。

激活后，`config.yaml` 会写入当前目录，Router 会退出 setup mode。

### 4. 可选：通过 CLI 打开 Dashboard

```bash
vllm-sr dashboard
```

### 5. 测试 Router

```bash
curl http://localhost:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MoM",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
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

## 下一步

- **[配置指南](configuration.md)** - 高级路由和信号配置
- **[API 文档](../api/router.md)** - 完整 API 参考
- **[教程](../tutorials/intelligent-route/keyword-routing.md)** - 通过示例学习

## 获取帮助

- **Issues**: [GitHub Issues](https://github.com/vllm-project/semantic-router/issues)
- **社区**: 加入 vLLM Slack 中的 `#semantic-router` 频道
- **文档**: [vllm-semantic-router.com](https://vllm-semantic-router.com/)
