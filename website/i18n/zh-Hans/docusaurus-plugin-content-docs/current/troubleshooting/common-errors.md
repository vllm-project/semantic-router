---
title: 常见错误
sidebar_label: 常见错误
translation:
  source_commit: "e56591a9cb24f073bf159927e87116ba6d278741"
  source_file: "docs/troubleshooting/common-errors.md"
  outdated: true
---

# 常见错误

先从第一个失败的组件入手，不要同时改多项设置：

```bash
vllm-sr status
vllm-sr logs router
vllm-sr logs envoy        # 仅限 --gateway extproc
vllm-sr config validate --config config.yaml
```

下面的示例是片段。把它们加到完整规范配置的对应章节，并校验结果。
需要穷尽字段上下文时，使用 [config/config.yaml](https://github.com/vllm-project/semantic-router/blob/main/config/config.yaml)。

## Router 无法加载配置

### `Failed to create ExtProc server`

这是顶层启动失败。有用的原因通常出现在同一日志行的后半段，或紧挨着它的上一行。

确认文件存在且可读，然后在容器外运行校验：

```bash
test -r config.yaml
vllm-sr config validate --config config.yaml
```

按校验错误中的字段路径排查。不要把缺失字段加到随意的嵌套块里；规范字段对位置敏感。

### `failed to read config file`

进程打不开它收到的路径。检查：

- `--config` 是否相对于当前工作目录；
- Router 容器内是否存在同一路径；
- 文件和父目录权限；以及
- 受管配方是否在另一个工作区生成了运行时配置。

检查容器挂载前，先用 `vllm-sr status` 确认活动工作区。

## 入口 / 配方校验 {#entrypoint--recipe-validation}

`vllm-sr config validate`（或 `vllm-sr validate`）会为常见的多配方接线错误给出修复提示。

### 未知配方 {#unknown-recipe}

```text
Entrypoint references unknown recipe 'missing-recipe'
Hint: Change this to the name of a recipe defined under recipes.
```

错误写法：

```yaml
entrypoints:
  - model_names: [my-model]
    recipe: missing-recipe
recipes:
  - name: production
```

修正后：

```yaml
entrypoints:
  - model_names: [my-model]
    recipe: production
recipes:
  - name: production
```

### 配方名重复 {#duplicate-recipe-name}

```text
Duplicate recipe name 'production'
Hint: Rename one recipe so every recipe has a unique name.
```

为每个配方设置不同的 `name`，再更新引用被改名配方的 entrypoint：

```yaml
recipes:
  - name: production
  - name: staging
entrypoints:
  - model_names: [my-model]
    recipe: production
```

### 模型或保留别名冲突 {#model-or-reserved-alias-collision}

```text
Entrypoint model 'vllm-sr/auto' conflicts with a configured model or reserved alias
Hint: Use a distinct entrypoint model name; do not reuse a configured model or
reserved alias such as vllm-sr/auto.
```

错误写法：

```yaml
entrypoints:
  - model_names: [vllm-sr/auto]
    recipe: production
```

修正后：

```yaml
entrypoints:
  - model_names: [customer-production]
    recipe: production
```

### 入口名与后端模型冲突 {#looper-alias-names-a-model}

公开 recipe 入口必须使用独立名字。与后端模型同名的入口会被拒绝，
Fusion、ReMoM 和 Flow 也遵循同一规则，没有单独的隐式分发名字。

如果 `openai/gpt-oss-20b` 是后端模型，请为工作流声明不同的入口：

```yaml
entrypoints:
  - model_names: [vllm-sr/flow]
    recipe: flow
```

将 `algorithm.type: workflows` 的决策放在 `flow` recipe 中。

## 请求返回路由错误码 {#a-request-returns-a-routing-error-code}

先在[路由错误](../api/router.md#routing-errors)中查到该原因码，再按请求的 `x-request-id` 在 Router 日志中找到 Router 自己的原因：

```bash
vllm-sr logs router | grep '<x-request-id>'
```

对于 `no_route`，`entrypoint_routing_no_selection` 日志行会给出模型、配方以及已匹配的决策。Fusion、Flow 和 ReMoM 由 recipe 中的决策选择；先检查 entrypoint 映射和 recipe 规则。

## 响应缓存无法启动

### 必须配置后端

类似下面的错误表示 `backend_type` 选了后端，却没有配套配置：

```text
milvus configuration is required for Milvus cache backend
qdrant configuration is required for Qdrant cache backend
```

Milvus 示例：

```yaml
global:
  stores:
    response_cache:
      enabled: true
      backend_type: milvus
      milvus:
        connection:
          host: milvus
          port: 19530
        collection:
          name: response_cache
```

Qdrant 示例：

```yaml
global:
  stores:
    response_cache:
      enabled: true
      backend_type: qdrant
      qdrant:
        host: qdrant
        port: 6334
        collection_name: response_cache
```

后端主机名必须能从 Router 容器或 Pod 解析，而不仅是从宿主机解析。

### 缺少索引或集合

以 `auto-creation is disabled` 结尾的错误表示后端服务可达，但所需索引不存在。选择一种运作方式：

- 在 Router 启动前预先创建索引或集合；或
- 为该后端启用开发时自动创建。

Redis 和 Valkey 使用 `development.auto_create_index`；Milvus 使用 `development.auto_create_collection`。例如：

```yaml
global:
  stores:
    response_cache:
      enabled: true
      backend_type: redis
      redis:
        # 从运行时示例补充连接、索引和搜索设置。
        development:
          auto_create_index: true
```

完整后端块见仓库中的
[响应缓存示例](https://github.com/vllm-project/semantic-router/tree/main/config/runtime/response-cache)。生产部署通常单独供给 schema，并关闭自动创建。

### 缓存命中异常偏少

先确认决策启用了 `response_cache` 插件，并检查 `x-vsr-cache-hit` 或回放诊断。若嵌入正常但近似重复仍未命中，可在代表性流量上测试更低的相似度阈值：

```yaml
global:
  stores:
    response_cache:
      similarity_threshold: 0.75
```

按决策覆盖应写在插件配置中：

```yaml
routing:
  decisions:
    - name: cached-route
      plugins:
        - type: response_cache
          configuration:
            enabled: true
            semantic:
              similarity_threshold: 0.70
```

降低阈值会增加误匹配风险。上线前先评估答案是否等价，诊断后端时使用管理 API 的响应缓存统计和测试端点。

## Responses API 存储无法连接

类似：

```text
failed to connect to Redis: redis ping failed
```

这类错误来自 Responses API 存储，而不是响应缓存。检查单独的服务块：

```yaml
global:
  services:
    response_api:
      enabled: true
      store_backend: redis
      redis:
        address: redis:6379
        db: 0
```

仅在本地工作、且可接受重启后丢失已存储响应和会话链时，才使用 `store_backend: memory`。

## PII 路由意外匹配

启用调试日志后，匹配的 PII 规则会包含被拒绝的实体类型：

```text
[Signal Computation] PII rule "<name>" matched: denied_entities=[<types>]
```

若该类型按策略应被允许，把它加入该信号的允许名单。若检测器产生低置信度误报，评估更高阈值：

```yaml
routing:
  signals:
    pii:
      - name: pii-policy
        threshold: 0.90
        pii_types_allowed:
          - GPE
          - ORGANIZATION
```

更改隐私阈值会改变漏报风险。请用带标签的数据集验证，而不是几条手写提示词。

## 越狱路由意外匹配

对比分类器匹配使用以下开头的调试消息：

```text
[Signal Computation] Contrastive jailbreak rule "<name>" matched
```

其他越狱分类器不会发出这句原文。用回放或带 `x-vsr-debug: true` 的请求检查匹配信号和所选决策。

要减少误报，对受影响信号评估更高阈值：

```yaml
routing:
  signals:
    jailbreak:
      - name: jailbreak-standard
        threshold: 0.85
```

若某条路由不应依赖越狱检测，从该路由中去掉该条件。全局关闭分类器会改变所有使用它的决策。

## MCP 类别分类器无法启动

这些错误表示传输配置不完整：

```text
command is required for stdio transport
URL is required for HTTP transport
```

使用一种传输，并在应由 MCP 负责类别分类时禁用本地领域分类器。

Stdio 示例：

```yaml
global:
  model_catalog:
    modules:
      classifier:
        domain:
          enabled: false
        mcp:
          enabled: true
          transport_type: stdio
          command: /app/bin/category-server
          tool_name: classify_text
```

可执行文件及其全部参数必须存在于 Router 运行时内。

Streamable HTTP 示例：

```yaml
global:
  model_catalog:
    modules:
      classifier:
        domain:
          enabled: false
        mcp:
          enabled: true
          transport_type: streamable-http
          url: http://mcp-server:8080/mcp
          tool_name: classify_text
```

从 Router 网络测试该 URL。若服务器暴露的是另一个工具名，精确设置 `tool_name`，或省略它以允许发现已识别的分类工具。

## 提供方后端没有地址

校验错误：

```text
providers.models[<model>].backend_refs requires endpoint or base_url
```

表示后端引用无法解析：

```yaml
providers:
  models:
    - name: local-model
      provider_model_id: local-model
      api_format: openai
      backend_refs:
        - name: local-vllm
          endpoint: 10.0.0.1:8000
          protocol: http
          provider: vllm
```

当提供方需要完整 API 根（例如 `https://provider.example/v1`）时，使用 `base_url`。使用 Router 网络可达的主机名；`localhost` 指向 Router 容器自身。

该 API 根中的版本段对每个 Provider ID 都可接受。自身操作路径已带版本的提供方（例如 `anthropic` 和 `minimax`）不会重复它，因此 `https://provider.example/v1` 和 `https://provider.example` 解析到同一上游路径。

端到端检查见[容器连通性](./container-connectivity)。

## 分类器或嵌入模型无法加载

每个模型都运行在[模型运行时](model-runtime/overview.md)中。模型无法加载时，依赖它的功能返回未知结果，运行时会记录原因。查看 Router 的 `vsr_model_runtime_ready{deployment="..."}` 指标和 Router 日志，或直接询问你自己运行的运行时：

```bash
curl -s localhost:8100/v1/models
```

每个模型的 `status` 和 `reason` 说明失败原因：下载损坏、缺少 `revision`、私有仓库没有令牌、设备不存在，或模型超出设备容量。[故障排查与 FAQ](model-runtime/troubleshooting.md) 列出了每种原因及修复方法。

## 容器镜像没有匹配的平台

带 `no matching manifest` 的 `ImagePullBackOff` 表示所选标签或摘要不包含该节点架构的镜像。

从部署中检查确切引用：

```bash
docker buildx imagetools inspect <registry>/<image>:<tag>
```

若标签是多平台索引，固定其索引摘要，而不是某一架构的子清单。若是单平台，使用发布了所需架构的发行版，或通过已批准的流水线构建并发布镜像。不要把未固定的 `latest` 标签当作长期不可变替代方案。

## 分类置信度过低

若请求在领域分类后频繁回退，先确认已加载的模型和类别映射，再改阈值。然后在带标签流量上评估更低的值：

```yaml
global:
  model_catalog:
    modules:
      classifier:
        domain:
          threshold: 0.50
```

降低阈值可能增加错误领域路由。请按所选工作点报告各类别的精确率和召回率。

## 诊断命令

```bash
# Validate the source configuration.
vllm-sr config validate --config config.yaml

# Identify the active local stack and component state.
vllm-sr status

# Read component logs without depending on generated container names.
vllm-sr logs router
vllm-sr logs envoy        # 仅限 --gateway extproc

# Check the public listener and model catalog.
curl -sS http://localhost:8899/v1/models

# Check management health and metrics.
curl -sS http://localhost:8080/health
curl -sS http://localhost:9190/metrics | head
```

若提供方在宿主机上成功、在 Router 中失败，继续看[容器连通性](./container-connectivity)。镜像和产物下载见[受限网络环境](./network-tips)。
