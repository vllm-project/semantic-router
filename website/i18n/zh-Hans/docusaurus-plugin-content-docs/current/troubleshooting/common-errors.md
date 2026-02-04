---
title: 常见错误
sidebar_label: 常见错误
translation:
  source_commit: "1d1439a"
  source_file: "docs/troubleshooting/common-errors.md"
  outdated: false
---

# 常见错误及解决方法

本指南提供了运行 vLLM Semantic Router 时可能遇到的常见 log 消息和错误的快速参考。每个部分都将错误模式映射到其根本原因和配置修复方法。

:::tip
使用本页末尾的[快速诊断命令](#快速诊断命令)可以快速识别问题。
:::

## 配置加载错误

### 创建 ExtProc 服务器失败

**Log 模式：**

```
Failed to create ExtProc server: <error>
```

**原因及修复：**

| 原因                   | 修复方法                                                 |
| ----------------------- | --------------------------------------------------- |
| 配置路径无效     | 验证 `--config` 标志指向存在的 YAML 文件 |
| YAML 语法错误       | 使用 `yq` 或在线验证器验证 YAML         |
| 缺少必填字段 | 检查所有必填字段是否存在               |

```bash
# 验证配置路径
./router --config /app/config/config.yaml
```

---

### 读取配置文件失败

**Log 模式：**

```
failed to read config file: <error>
```

**修复方法：**

- 验证文件存在：`ls -la config/config.yaml`
- 检查权限：`chmod 644 config/config.yaml`
- 确保路径是绝对路径或正确的相对路径

> 参见代码：[cmd/main.go](https://github.com/vllm-project/semantic-router/blob/main/src/semantic-router/cmd/main.go)。

---

## Cache 和存储错误

### 需要 Milvus 配置路径

**Log 模式：**

```
milvus config path is required
```

**修复方法：** 使用 Milvus 后端时设置 `backend_config_path`：

```yaml
semantic_cache:
  enabled: true
  backend_type: "milvus"
  backend_config_path: "config/milvus.yaml" # ← 添加此项
```

---

### 索引不存在且禁用了自动创建

**Log 模式：**

```
index <name> does not exist and auto-creation is disabled
```

**修复方法：** 在 Redis/Milvus 配置中启用自动创建：

```yaml
# 在 config/redis.yaml 中
index:
  auto_create: true # ← 启用此项
```

---

### Redis 存储尚未实现

**日志模式：**

```
redis store not yet implemented
```

**注意：** Redis 响应存储尚不可用。请改用 `memory` 或 `milvus`：

```yaml
semantic_cache:
  backend_type: "memory" # 或 "milvus"
```

> 参见代码：[pkg/cache](https://github.com/vllm-project/semantic-router/tree/main/src/semantic-router/pkg/cache) 和 [pkg/responsestore](https://github.com/vllm-project/semantic-router/tree/main/src/semantic-router/pkg/responsestore)。

---

## PII 和安全错误

### PII 策略违规

**Log 模式：**

```
PII policy violation for decision <name>: denied PII types [<types>]
```

**修复方法：**

1. **允许该 PII 类型**（如果应该被允许）：

```yaml
plugins:
  - type: "pii"
    configuration:
      pii_types_allowed:
        - "LOCATION" # 在此添加被拒绝的类型
```

1. **提高阈值**（如果是误报）：

```yaml
classifier:
  pii_model:
    threshold: 0.95 # 从默认的 0.9 提高
```

---

### 检测到 Jailbreak 攻击

**日志模式：**

```
Jailbreak detected: type=<type>, confidence=<score>
```

**修复方法：**

1. **提高阈值** 以减少误报：

```yaml
prompt_guard:
  threshold: 0.8 # 从默认的 0.7 提高
```

1. **为特定 decision 禁用**：

```yaml
decisions:
  - name: "internal_decision"
    jailbreak_enabled: false
```

> 参见代码：[pii/policy.go](https://github.com/vllm-project/semantic-router/blob/main/src/semantic-router/pkg/utils/pii/policy.go) 和 [req_filter_jailbreak.go](https://github.com/vllm-project/semantic-router/blob/main/src/semantic-router/pkg/extproc/req_filter_jailbreak.go)。

---

## MCP 客户端错误

### 必须指定命令或 URL

**日志模式：**

```
either command or URL must be specified
```

**修复方法：** 指定传输配置：

```yaml
# 对于 stdio 传输
mcp_clients:
  my_client:
    transport_type: "stdio"
    command: "/path/to/mcp-server"

# 对于 HTTP 传输
mcp_clients:
  my_client:
    transport_type: "streamable-http"
    url: "http://localhost:8080"
```

---

### stdio 传输需要命令

**日志模式：**

```
command is required for stdio transport
```

**修复方法：** 为 stdio 传输添加命令：

```yaml
mcp_clients:
  my_client:
    transport_type: "stdio"
    command: "python"
    args: ["-m", "my_mcp_server"]
```

> 参见代码：[pkg/mcp/factory.go](https://github.com/vllm-project/semantic-router/blob/main/src/semantic-router/pkg/mcp/factory.go)。

---

## Endpoint 错误

### 地址格式无效

**Log 模式：**

```
invalid endpoint address: <address>
```

**修复方法：**

| 错误格式                  | 正确格式                              |
| ---------------------- | ------------------------------------ |
| `http://10.0.0.1:8000` | `10.0.0.1`（地址）+ `8000`（端口） |
| `vllm.example.com`     | 改用 IP 地址               |
| `10.0.0.1:8000`        | 分开地址和端口字段     |

```yaml
vllm_endpoints:
  - name: "endpoint1"
    address: "10.0.0.1" # 仅 IP，不含协议/端口
    port: 8000 # 端口单独设置
```

> 参见：[config.yaml#vllm_endpoints](https://github.com/vllm-project/semantic-router/blob/main/config/config.yaml#L43-L51) 和 [pkg/extproc](https://github.com/vllm-project/semantic-router/tree/main/src/semantic-router/pkg/extproc)。

---

## 模型加载错误

### 找不到模型

**日志模式：**

```
failed to load model: <path>
```

**修复方法：**

- 验证模型路径存在
- 检查模型是否已下载：`ls -la models/`
- 确保路径在容器内可访问

```yaml
bert_model:
  model_id: /app/models/all-MiniLM-L12-v2 # 在容器中使用绝对路径
```

---

## 性能问题

### Cache 命中率低

**症状：** Cache 很少返回 hit，后端延迟高

**修复方法：** 降低 similarity threshold：

```yaml
semantic_cache:
  similarity_threshold: 0.75 # 从默认的 0.8 降低

# 或按 decision 设置
plugins:
  - type: "semantic-cache"
    configuration:
      similarity_threshold: 0.70
```

---

### 分类置信度过低

**症状：** 许多查询落入 "other" 类别

**修复方法：** 降低类别阈值：

```yaml
classifier:
  category_model:
    threshold: 0.5 # 从默认的 0.6 降低
```

---

## 快速诊断命令

```bash
# 检查配置语法
yq eval '.' config/config.yaml

# 测试 endpoint 连通性
curl -s http://<address>:<port>/health

# 检查模型文件
ls -la models/

# 查看最近 log
docker logs semantic-router --tail 100

# 检查指标
curl -s http://localhost:9190/metrics | grep semantic_router
```
