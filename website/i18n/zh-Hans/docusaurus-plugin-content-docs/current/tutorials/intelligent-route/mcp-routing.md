---
translation:
  source_commit: "5694873"
  source_file: "docs/tutorials/intelligent-route/mcp-routing.md"
  outdated: false
---

# 基于 MCP 的路由 (MCP Based Routing)

本指南向您展示如何使用模型上下文协议 (Model Context Protocol, MCP) 实现自定义分类逻辑。MCP 路由允许您集成外部服务、LLM 或自定义业务逻辑来进行分类决策，同时保持数据私密且路由逻辑可扩展。

## 关键优势

- **基准/高准确度**：使用强大的 LLM（GPT-4, Claude）进行带有上下文学习 (In-context Learning) 的分类
- **可扩展性**：轻松集成自定义分类逻辑，无需修改路由代码
- **隐私性**：将分类逻辑和数据保留在您自己的基础设施中
- **灵活性**：将 LLM 推理与业务规则、用户上下文和外部数据相结合

## 它解决了什么问题？

内置分类器受限于预定义模型和逻辑。MCP 路由实现了：

- **LLM 驱动的分类**：使用 GPT-4/Claude 进行复杂、细微的分类
- **上下文学习**：提供示例和上下文以提高分类准确性
- **自定义业务逻辑**：根据用户层级、时间、地点、历史记录实施路由规则
- **外部数据集成**：在分类过程中查询数据库、API、特性标志 (Feature Flags)
- **快速实验**：无需重新部署路由即可更新分类逻辑

## 何时使用

- **高准确度要求**：LLM 分类优于 BERT/嵌入模型的场景
- **复杂领域**：需要超出关键词/嵌入匹配的细微理解
- **自定义业务规则**（用户层级、A/B 测试、基于时间的路由）
- **私密/敏感数据**：分类过程必须保留在您的基础设施中
- **快速迭代**：无需更改代码即可更新分类逻辑

## 配置

在您的 `config.yaml` 中配置 MCP 分类器：

```yaml
classifier:
  # 禁用内置分类器
  category_model:
    model_id: ""
  
  # 启用 MCP 分类器
  mcp_category_model:
    enabled: true
    transport_type: "http"
    url: "http://localhost:8090/mcp"
    threshold: 0.6
    timeout_seconds: 30
    # tool_name: "classify_text"  # 可选：如果未指定则自动发现

categories: []  # 类别从 MCP 服务器加载

default_model: openai/gpt-oss-20b

vllm_endpoints:
  - name: endpoint1
    address: 127.0.0.1
    port: 8000
    weight: 1

model_config:
  openai/gpt-oss-20b:
    reasoning_family: gpt-oss
    preferred_endpoints: [endpoint1]
```

## 工作原理

1. **启动**：路由连接到 MCP 服务器并调用 `list_categories` 工具
2. **类别加载**：MCP 返回类别、系统提示词和描述
3. **分类**：对于每个请求，路由调用 `classify_text` 工具
4. **路由**：MCP 响应包含类别、模型和推理设置

### MCP 响应格式

**list_categories**:

```json
{
  "categories": ["math", "science", "technology"],
  "category_system_prompts": {
    "math": "你是一位数学专家...",
    "science": "你是一位科学专家..."
  },
  "category_descriptions": {
    "math": "数学和计算查询",
    "science": "科学概念和查询"
  }
}
```

**classify_text**:

```json
{
  "class": 3,
  "confidence": 0.85,
  "model": "openai/gpt-oss-20b",
  "use_reasoning": true
}
```

## MCP 服务器示例

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class ClassifyRequest(BaseModel):
    text: str

@app.post("/mcp/list_categories")
def list_categories():
    return {
        "categories": ["math", "science", "general"],
        "category_system_prompts": {
            "math": "你是一位数学专家。",
            "science": "你是一位科学专家。",
            "general": "你是一位乐于助人的助手。"
        }
    }

@app.post("/mcp/classify_text")
def classify_text(request: ClassifyRequest):
    # 自定义分类逻辑
    if "equation" in request.text or "solve" in request.text:
        return {
            "class": 0,  # math
            "confidence": 0.9,
            "model": "openai/gpt-oss-20b",
            "use_reasoning": True
        }
    return {
        "class": 2,  # general
        "confidence": 0.7,
        "model": "openai/gpt-oss-20b",
        "use_reasoning": False
    }
```

## 请求示例

```bash
# 数学查询（MCP 决定路由）
curl -X POST http://localhost:8801/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MoM",
    "messages": [{"role": "user", "content": "解方程：2x + 5 = 15"}]
  }'
```

## 效益

- **自定义逻辑**：实施特定领域的分类规则
- **动态路由**：MCP 为每个查询决定模型和推理设置
- **集中控制**：在外部服务中管理路由逻辑
- **可扩展性**：分类能力可独立于路由进行扩展
- **集成**：连接到现有的机器学习基础设施

## 真实世界用例

### 1. 复杂领域分类（高准确度）

**问题**：细微的法律/医疗查询需要比 BERT/嵌入模型更好的准确度
**解决方案**：MCP 使用带有上下文示例的 GPT-4 进行分类
**影响**：准确度达 98%（BERT 约 85%），成为质量比较的基准

### 2. 专有分类逻辑（私有）

**问题**：分类逻辑包含商业机密，不能使用外部服务
**解决方案**：MCP 服务器在私有 VPC 中运行，将所有逻辑和数据保留在内部
**影响**：完整的数据隐私，无外部 API 调用

### 3. 自定义业务规则（可扩展）

**问题**：需要根据用户层级、地点、时间、A/B 测试进行路由
**解决方案**：MCP 将 LLM 分类与数据库查询和业务逻辑相结合
**影响**：无需修改路由代码即可实现灵活路由

### 4. 快速实验（可扩展）

**问题**：数据科学团队需要每天测试新的分类方法
**解决方案**：独立更新 MCP 服务器，无需更改路由
**影响**：部署新分类逻辑只需几分钟，而非数天

### 5. 多租户平台 (可扩展 + 私有)

**问题**：每个客户需要自定义分类，数据必须隔离
**解决方案**：MCP 加载租户特定的模型/规则，执行数据隔离
**影响**：支持 1000+ 租户的自定义逻辑，完整的数据隐私

### 6. 混合方法（高准确度 + 可扩展）

**问题**：边缘情况需要 LLM 的准确度，常见查询需要快速路由
**解决方案**：MCP 对常见模式使用缓存响应，对新颖查询使用 LLM
**影响**：95% 的缓存命中率，长尾请求享有 LLM 级的准确度

## 高级 MCP 服务器示例

### 上下文感知分类 (Context-Aware Classification)

```python
@app.post("/mcp/classify_text")
def classify_text(request: ClassifyRequest, user_id: str = Header(None)):
    # 检查用户历史记录
    user_history = get_user_history(user_id)

    # 根据上下文调整分类
    if user_history.is_premium:
        return {
            "class": 0,
            "confidence": 0.95,
            "model": "openai/gpt-4",  # 高级模型
            "use_reasoning": True
        }

    # 免费层级使用快速模型
    return {
        "class": 0,
        "confidence": 0.85,
        "model": "openai/gpt-oss-20b",
        "use_reasoning": False
    }
```

### 基于时间的路由

```python
@app.post("/mcp/classify_text")
def classify_text(request: ClassifyRequest):
    current_hour = datetime.now().hour

    # 高峰时段：使用缓存响应
    if 9 <= current_hour <= 17:
        return {
            "class": get_cached_category(request.text),
            "confidence": 0.9,
            "model": "fast-model",
            "use_reasoning": False
        }

    # 非高峰时段：启用推理
    return {
        "class": classify_with_ml(request.text),
        "confidence": 0.95,
        "model": "reasoning-model",
        "use_reasoning": True
    }
```

### 基于风险的路由

```python
@app.post("/mcp/classify_text")
def classify_text(request: ClassifyRequest):
    # 计算风险评分
    risk_score = calculate_risk(request.text)

    if risk_score > 0.8:
        # 高风险：人工审核
        return {
            "class": 999,  # 特殊类别
            "confidence": 1.0,
            "model": "human-review-queue",
            "use_reasoning": False
        }

    # 正常路由
    return standard_classification(request.text)
```

## 与内置分类器的对比

| 特性 | 内置 | MCP |
|---------|----------|-----|
| 自定义模型 | ❌ | ✅ |
| 业务逻辑 | ❌ | ✅ |
| 动态更新 | ❌ | ✅ |
| 用户上下文 | ❌ | ✅ |
| A/B 测试 | ❌ | ✅ |
| 外部 API | ❌ | ✅ |
| 延迟 | 5-50ms | 50-200ms |
| 复杂度 | 低 | 高 |

## 性能注意事项

- **延迟**：MCP 为每个请求增加 50-200ms（网络 + 分类）
- **缓存**：为重复查询缓存 MCP 响应
- **超时**：设置适当的超时（默认 30s）
- **回退**：配置当 MCP 不可用时的默认模型
- **监控**：跟踪 MCP 延迟和错误率

## 参考

完整配置请参见 [config-mcp-classifier.yaml](https://github.com/vllm-project/semantic-router/blob/main/config/intelligent-routing/out-tree/config-mcp-classifier.yaml)。
