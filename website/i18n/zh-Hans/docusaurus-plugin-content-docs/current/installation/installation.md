---
translation:
  source_commit: "a0e504f"
  source_file: "docs/installation/installation.md"
  outdated: true
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

### 2. 初始化配置

```bash
# 在当前目录创建 config.yaml
vllm-sr init
```

这将创建一个带有默认设置的 `config.yaml` 文件。

### 3. 配置后端

编辑生成的 `config.yaml` 以配置您的模型和后端 endpoint：

```yaml
providers:
  # 模型配置
  models:
    - name: "qwen/qwen3-1.8b"           # 模型名称
      endpoints:
        - name: "my_vllm"
          weight: 1
          endpoint: "localhost:8000"    # 域名或 IP:端口
          protocol: "http"              # http 或 https
      access_key: "your-token-here"     # 可选：用于身份验证

  # 回退的默认模型
  default_model: "qwen/qwen3-1.8b"
```

**配置选项：**

- **endpoint**: 带有端口的域名或 IP 地址（例如 `localhost:8000`、`api.openai.com`）
- **protocol**: `http` 或 `https`
- **access_key**: 可选的身份验证 token（Bearer token）
- **weight**: 负载均衡权重（默认：1）

**示例：本地 vLLM**

```yaml
providers:
  models:
    - name: "qwen/qwen3-1.8b"
      endpoints:
        - name: "local_vllm"
          weight: 1
          endpoint: "localhost:8000"
          protocol: "http"
  default_model: "qwen/qwen3-1.8b"
```

**示例：带有 HTTPS 的外部 API**

```yaml
providers:
  models:
    - name: "openai/gpt-4"
      endpoints:
        - name: "openai_api"
          weight: 1
          endpoint: "api.openai.com"
          protocol: "https"
      access_key: "sk-xxxxxx"
  default_model: "openai/gpt-4"
```

### 4. 启动 Router

```bash
vllm-sr serve
```

Router 将：

- 自动下载所需的 ML 模型（约 1.5GB，一次性）
- 在端口 8888 上启动 Envoy Proxy
- 启动 Semantic Router 服务
- 在端口 9190 上启用 metrics

### 5. 测试 Router

```bash
curl http://localhost:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MoM",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### 6. 启动 Dashboard

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
