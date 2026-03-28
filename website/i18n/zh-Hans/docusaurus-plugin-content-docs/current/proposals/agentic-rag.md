---
translation:
  source_commit: "f123732c"
  source_file: "docs/proposals/agentic-rag.md"
  outdated: false
---

# OpenAI RAG 集成

本指南说明如何在 Semantic Router 中使用 OpenAI 的 File Store 与 Vector Store API 实现 RAG（检索增强生成），流程参考 [OpenAI Responses API cookbook](https://cookbook.openai.com/examples/rag_on_pdfs_using_file_search)。

## 概述

OpenAI RAG 后端对接 OpenAI 的 File Store 与 Vector Store API，提供一等公民的 RAG 体验。支持两种工作流模式：

1. **直接检索模式**（默认）：使用向量库搜索 API 同步检索
2. **基于工具的模式**：在请求中加入 `file_search` 工具（Responses API 工作流）

## 架构

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────┐
│     Semantic Router                 │
│  ┌───────────────────────────────┐  │
│  │      RAG Plugin               │  │
│  │  ┌─────────────────────────┐  │  │
│  │  │  OpenAI RAG Backend     │  │  │
│  │  └──────┬──────────────────┘  │  │
│  └─────────┼──────────────────── ┘  │
└────────────┼─────────────────────── ┘
             │
             ▼
┌─────────────────────────────────────┐
│      OpenAI API                     │
│  ┌──────────────┐  ┌──────────────┐ │
│  │ File Store   │  │Vector Store  │ │
│  │   API        │  │   API        │ │
│  └──────────────┘  └──────────────┘ │
└─────────────────────────────────────┘
```

## 前置条件

1. 具备 File Store 与 Vector Store API 访问权限的 OpenAI API Key
2. 文件已上传至 OpenAI File Store
3. 已创建向量库并关联文件

## 配置

### 基础配置

在决策配置中加入 OpenAI RAG 后端：

```yaml
decisions:
  - name: rag-openai-decision
    signals:
      - type: keyword
        keywords: ["research", "document", "knowledge"]
    plugins:
      rag:
        enabled: true
        backend: "openai"
        backend_config:
          vector_store_id: "vs_abc123"  # 你的向量库 ID
          api_key: "${OPENAI_API_KEY}"  # 或使用环境变量
          max_num_results: 10
          workflow_mode: "direct_search"  # 或 "tool_based"
```

### 高级配置

```yaml
rag:
  enabled: true
  backend: "openai"
  similarity_threshold: 0.7
  top_k: 10
  max_context_length: 5000
  injection_mode: "tool_role"  # 或 "system_prompt"
  on_failure: "skip"  # 或 "warn" 或 "block"
  cache_results: true
  cache_ttl_seconds: 3600
  backend_config:
    vector_store_id: "vs_abc123"
    api_key: "${OPENAI_API_KEY}"
    base_url: "https://api.openai.com/v1"  # 可选，默认 OpenAI
    max_num_results: 10
    file_ids:  # 可选：限制检索文件
      - "file-123"
      - "file-456"
    filter:  # 可选：元数据过滤
      category: "research"
      published_date: "2024-01-01"
    workflow_mode: "direct_search"  # 或 "tool_based"
    timeout_seconds: 30
```

## 工作流模式

### 1. 直接检索模式（默认）

使用向量库搜索 API 同步检索，在发往 LLM 之前注入上下文。

**适用**：需要立即注入上下文并希望控制检索流程时。

**示例**：

```yaml
backend_config:
  workflow_mode: "direct_search"
  vector_store_id: "vs_abc123"
```

**流程**：

1. 用户发送查询
2. RAG 插件调用向量库搜索 API
3. 将检索到的上下文注入请求
4. 将带上下文的请求发给 LLM

### 2. 基于工具的模式（Responses API）

向请求添加 `file_search` 工具，由 LLM 自动调用，结果出现在响应标注中。

**适用**：使用 Responses API 且希望由 LLM 决定何时检索时。

**示例**：

```yaml
backend_config:
  workflow_mode: "tool_based"
  vector_store_id: "vs_abc123"
```

**流程**：

1. 用户发送查询
2. RAG 插件向请求添加 `file_search` 工具
3. 请求发往 LLM
4. LLM 调用 `file_search` 工具
5. 结果出现在响应标注中

## 使用示例

### 示例 1：基础 RAG 查询

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-VSR-Selected-Decision: rag-openai-decision" \
  -d '{
    "model": "gpt-4o-mini",
    "messages": [
      {
        "role": "user",
        "content": "What is Deep Research?"
      }
    ]
  }'
```

### 示例 2：Responses API 与 file_search 工具

```bash
curl -X POST http://localhost:8080/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4o-mini",
    "input": "What is Deep Research?",
    "tools": [
      {
        "type": "file_search",
        "file_search": {
          "vector_store_ids": ["vs_abc123"],
          "max_num_results": 5
        }
      }
    ]
  }'
```

### 示例 3：Python 客户端

```python
import requests

# 直接检索模式
response = requests.post(
    "http://localhost:8080/v1/chat/completions",
    headers={
        "Content-Type": "application/json",
        "X-VSR-Selected-Decision": "rag-openai-decision"
    },
    json={
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "user", "content": "What is Deep Research?"}
        ]
    }
)

result = response.json()
print(result["choices"][0]["message"]["content"])
```

## File Store 操作

OpenAI RAG 后端包含用于管理文件的 File Store 客户端：

### 上传文件

```go
import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/openai"

client := openai.NewFileStoreClient("https://api.openai.com/v1", apiKey)
file, err := client.UploadFile(ctx, fileReader, "document.pdf", "assistants")
```

### 创建向量库

```go
vectorStoreClient := openai.NewVectorStoreClient("https://api.openai.com/v1", apiKey)
store, err := vectorStoreClient.CreateVectorStore(ctx, &openai.CreateVectorStoreRequest{
    Name:    "my-vector-store",
    FileIDs: []string{"file-123", "file-456"},
})
```

### 将文件关联到向量库

```go
_, err := vectorStoreClient.CreateVectorStoreFile(ctx, "vs_abc123", "file-123")
```

## 测试

### 单元测试

运行 OpenAI RAG 相关单元测试：

```bash
cd src/semantic-router
go test ./pkg/openai/... -v
go test ./pkg/extproc/req_filter_rag_openai_test.go -v
```

### E2E 测试

基于 OpenAI cookbook 的 E2E：

```bash
# Python E2E
python e2e/testing/08-rag-openai-test.py --base-url http://localhost:8080

# Go E2E（需要 Kubernetes 集群）
make e2e-test E2E_TESTS=rag-openai
```

### OpenAI API 校验测试套件

校验测试确保 OpenAI API 实现（Files、Vector Stores、Search）与上游兼容，改编自 [openai-python/tests](https://github.com/openai/openai-python/tree/main/tests)。在设置 `OPENAI_API_KEY` 时运行。

**Python E2E（对真实 API 做契约校验）：**

```bash
# 在仓库根目录；若未设置 OPENAI_API_KEY 则跳过全部测试
OPENAI_API_KEY=sk-... python e2e/testing/09-openai-api-validation-test.py --verbose

# 可选：覆盖 API base URL
OPENAI_BASE_URL=https://api.openai.com/v1 OPENAI_API_KEY=sk-... python e2e/testing/09-openai-api-validation-test.py
```

**Go 集成（pkg/openai 客户端对真实 API）：**

```bash
cd src/semantic-router
# 未设置 OPENAI_API_KEY 时跳过测试
OPENAI_API_KEY=sk-... go test -tags=openai_validation ./pkg/openai -v
```

覆盖：Files（list、upload、get、delete）、Vector Stores（list、create、get、update、delete）、Vector Store Files（list）、Vector Store Search（响应 schema）。

## 监控与可观测性

OpenAI RAG 后端暴露以下指标：

- `rag_retrieval_attempts_total{backend="openai", decision="...", status="success|error"}`
- `rag_retrieval_latency_seconds{backend="openai", decision="..."}`
- `rag_similarity_score{backend="openai", decision="..."}`
- `rag_context_length_chars{backend="openai", decision="..."}`
- `rag_cache_hits_total{backend="openai"}`
- `rag_cache_misses_total{backend="openai"}`

### 追踪

创建以下 OpenTelemetry span：

- `semantic_router.rag.retrieval` — RAG 检索
- `semantic_router.rag.context_injection` — 上下文注入

## 错误处理

RAG 插件支持三种失败模式：

- **skip**（默认）：无上下文继续，记录警告
- **warn**：带警告头继续
- **block**：返回错误响应（503）

```yaml
rag:
  on_failure: "skip"  # 或 "warn" 或 "block"
```

## 最佳实践

1. **同步工作流用直接检索**：需要立即注入上下文时
2. **Responses API 用基于工具的模式**：希望由 LLM 控制检索时
3. **缓存结果**：对频繁查询启用缓存
4. **设置合理超时**：按向量库规模配置 `timeout_seconds`
5. **筛选结果**：用 `file_ids` 或 `filter` 缩小检索范围
6. **关注指标**：跟踪检索延迟与相似度分数

## 故障排查

### 无结果

- 确认向量库 ID 正确
- 确认文件已关联到向量库
- 确认文件处理完成（查看 `file_counts.completed`）

### 高延迟

- 降低 `max_num_results`
- 启用结果缓存
- 使用 `file_ids` 限制检索范围

### 认证错误

- 检查 API Key
- 确认 Key 有权访问 File Store 与 Vector Store API
- 确认自定义端点 URL 正确

## 参考

- [OpenAI Responses API Cookbook - RAG on PDFs](https://cookbook.openai.com/examples/rag_on_pdfs_using_file_search)
- [OpenAI File Store API 文档](https://platform.openai.com/docs/api-reference/files)
- [OpenAI Vector Store API 文档](https://platform.openai.com/docs/api-reference/vector-stores)
