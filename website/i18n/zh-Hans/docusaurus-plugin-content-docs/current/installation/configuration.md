---
sidebar_position: 4
translation:
  source_commit: "baa07413"
  source_file: "docs/installation/configuration.md"
  outdated: false
---

# 配置

Semantic Router v0.3 在本地 CLI、控制台、Helm 与 Operator 之间使用**同一套 canonical YAML 契约**：

```yaml
version:
listeners:
providers:
routing:
global:
```

背景说明见 [统一配置契约 v0.3](../proposals/unified-config-contract-v0-3)。本页为**如何实际使用该契约**的操作指南。

## Canonical 契约

- `version`：schema 版本。请使用 `v0.3`。
- `listeners`：路由器监听端口与超时等。
- `providers`：部署绑定与提供商默认值。
- `routing`：路由语义。
- `global`：稀疏运行时覆盖。若某字段省略，则使用路由器内置默认。

## 各节的职责划分

- `routing` 为 **DSL 所拥有** 的表面。
  - `routing.modelCards`
  - `routing.modelCards[].loras`
  - `routing.signals`
  - `routing.projections`（分区及派生路由输出）
  - `routing.decisions`
- `providers` 拥有部署与默认选择元数据。
  - `defaults`
  - `models`
  - `providers.defaults` 存放 `default_model`、`reasoning_families`、`default_reasoning_effort`
  - `providers.models[*]` 存放 `provider_model_id`、`backend_refs`、`pricing`、`api_format`、`external_model_ids`
- `global` 拥有路由器级运行时覆盖。
  - `global.router` 聚合路由引擎控制项（如配置来源选择、route-cache、模型选择默认等）
  - `global.router.config_source` 选择运行时配置来自 canonical YAML 文件（`file`）还是进程内 Kubernetes CRD 协调（`kubernetes`）
  - `global.services` 聚合共享 API 与控制面服务，如 `response_api`、`router_replay`、`observability`、`authz`、`ratelimit`
  - `global.stores` 聚合有存储支撑的服务，如 `semantic_cache`、`memory`、`vector_store`
- `global.integrations` 聚合辅助运行时集成，如 `tools`、`looper`
- `global.model_catalog` 聚合路由器持有的模型资产，如嵌入、系统模型、外部模型、可复用分类器与模型支撑模块
- `global.model_catalog.embeddings.semantic.embedding_config.top_k` 限制打分后路由要输出的嵌入规则条数上限；内置默认为 `1`
- `global.model_catalog.classifiers[]` 为启动时加载的分类器包（如分类体系分类器）的可复用注册表
- `global.model_catalog.modules` 聚合能力模块，如 `prompt_guard`、`classifier`、`hallucination_mitigation`

## Canonical 示例

```yaml
version: v0.3

listeners:
  - name: http-8899
    address: 0.0.0.0
    port: 8899
    timeout: 300s

providers:
  defaults:
    default_model: qwen3-8b
    reasoning_families:
      qwen3:
        type: chat_template_kwargs
        parameter: enable_thinking
    default_reasoning_effort: medium
  models:
    - name: qwen3-8b
      reasoning_family: qwen3
      provider_model_id: qwen3-8b
      backend_refs:
        - name: primary
          endpoint: host.docker.internal:8000
          protocol: http
          weight: 100
          api_key_env: OPENAI_API_KEY

routing:
  modelCards:
    - name: qwen3-8b
      modality: text
      capabilities: [chat, reasoning]
      loras:
        - name: math-adapter
          description: Adapter used for symbolic math and proof-style prompts.

  signals:
    keywords:
      - name: math_terms
        operator: OR
        keywords: ["algebra", "calculus"]
    structure:
      - name: many_questions
        feature:
          type: count
          source:
            type: regex
            pattern: '[?？]'
        predicate:
          gte: 3
    embeddings:
      - name: technical_support
        threshold: 0.75
        candidates: ["installation guide", "troubleshooting steps"]
      - name: account_management
        threshold: 0.72
        candidates: ["billing information", "subscription management"]

  projections:
    partitions:
      - name: support_intents
        semantics: exclusive
        temperature: 0.3
        members: [technical_support, account_management]
        default: technical_support
    scores:
      - name: request_difficulty
        method: weighted_sum
        inputs:
          - type: embedding
            name: technical_support
            weight: 0.18
            value_source: confidence
          - type: context
            name: long_context
            weight: 0.18
          - type: structure
            name: many_questions
            weight: 0.12
    mappings:
      - name: request_band
        source: request_difficulty
        method: threshold_bands
        outputs:
          - name: support_fast
            lt: 0.25
          - name: support_escalated
            gte: 0.25

  decisions:
    - name: support_route
      description: Route support requests that need an escalated answer
      priority: 100
      rules:
        operator: AND
        conditions:
          - type: embedding
            name: technical_support
          - type: projection
            name: support_escalated
      modelRefs:
        - model: qwen3-8b
          use_reasoning: true
          lora_name: math-adapter

global:
  router:
    config_source: file
  services:
    observability:
      metrics:
        enabled: true
```

对于 `routing.signals.structure`，`feature.type: density` 现使用**内置多语言文本单位**归一化：每个 CJK 字符计为一个单位，连续拉丁字母与数字串计为一个单位，标点忽略，从而使同一 density 规则在英文、中文与混写提示下行为一致，且无需单独的 `normalize_by` 字段。

## 仓库中的配置资产

仓库将**详尽的 canonical 参考配置**与**可复用路由片段**分开：

- `config/config.yaml`：详尽 canonical 参考配置
- `config/signal/`：可复用的 `routing.signals` 片段
- `config/decision/`：可复用的 `routing.decisions` 规则形状片段
- `config/algorithm/`：可复用的 `decision.algorithm` 片段
- `config/plugin/`：可复用的路由插件片段

`config/decision/` 按布尔情形组织：`single/`、`and/`、`or/`、`not/`、`composite/`。  
`config/algorithm/` 按路由策略族组织：`looper/` 与 `selection/`。  
`config/plugin/` 按每个插件或可复用包单独目录组织。  
仓库在 `go test ./pkg/config/...` 中强制该片段目录，因此路由表面变更须同步更新 `config/` 树。

最新教程遵循同一分类：

- `tutorials/signal/overview` 以及 `tutorials/signal/heuristic/`、`tutorials/signal/learned/` 对应 `config/signal/`
- `tutorials/decision/` 对应 `config/decision/`
- `tutorials/algorithm/` 对应 `config/algorithm/`，每种算法一页
- `tutorials/plugin/` 对应 `config/plugin/`，每种插件一页
- `tutorials/global/` 对应 `global:` 下的稀疏路由器级覆盖

与仓库相关的运行时与测试台资产现位于 `config/` 之外：

- `deploy/examples/runtime/semantic-cache/`
- `deploy/examples/runtime/response-api/`
- `deploy/examples/runtime/tools/`
- `e2e/config/`
- `deploy/local/envoy.yaml`

仅测试用 ONNX 绑定资产位于 `e2e/config/onnx-binding/`。

上述目录为支持资产，**不是**面向用户的主配置契约。手写配置请从 `config/config.yaml` 或上述片段目录开始。本仓库中，详尽参考配置将 `global.integrations.tools.tools_db_path` 指向 `deploy/examples/runtime/tools/tools_db.json` 以供本地开发。

`config/config.yaml` 不再只是示例。仓库将其作为**详尽公开契约参考**强制执行：

- `go test ./pkg/config/...` 检查其与 canonical schema 及路由表面目录一致
- `make agent-lint` 在 lint 层运行同一参考配置契约检查，合入前即可拦截配置/schema 漂移
- 维护中的 `deploy/` 与 `e2e/` 路由器配置资产亦按同一 canonical 契约校验，避免示例与测试配置退回旧版稳态字段

## 投影工作流

当原始信号目录本身不足以支撑决策时，使用 `routing.projections`：

1. `routing.signals` 定义可复用检测器。
2. `routing.projections.partitions` 在互斥域或嵌入族内解析唯一胜者。
3. `routing.projections.scores` 将学习与启发式信号组合为加权分数。
4. `routing.projections.mappings` 将分数映射为具名路由带。
5. `routing.decisions[*].rules.conditions[*]` 可用 `type: projection` 引用这些带。

控制台镜像同一契约：

- `Config -> Projections` 编辑分区、分数与映射
- `Config -> Decisions` 可用条件类型 `projection` 引用映射输出
- `DSL -> Visual` 直接管理 `PROJECTION partition`、`PROJECTION score`、`PROJECTION mapping` 实体

专题教程见 [Projections](../tutorials/projection/overview)。端到端维护示例：

- [`deploy/recipes/balance.yaml`](https://github.com/vllm-project/semantic-router/blob/main/deploy/recipes/balance.yaml)
- [`deploy/recipes/balance.dsl`](https://github.com/vllm-project/semantic-router/blob/main/deploy/recipes/balance.dsl)

## 如何使用

### Python CLI

直接使用 canonical YAML。

```bash
vllm-sr serve --config config.yaml
```

若需先迁移旧配置：

```bash
vllm-sr config migrate --config old-config.yaml
vllm-sr validate config.yaml
```

v0.3 已移除 `vllm-sr init`。稳态文件为 `config.yaml`。本仓库默认详尽参考文件为 [`config/config.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/config.yaml)。

### 本地路由器 / YAML 优先

本地 Docker 或直接开发路由器时，以 canonical 形式手写 `config.yaml`，serve 前校验：

```bash
vllm-sr validate config.yaml
vllm-sr serve --config config.yaml
```

若只需覆盖少量运行时默认，将其写在 `global:` 下，其余保持未设置。

### 控制台 / 引导

若需从 URL 导入或直接编辑完整 canonical YAML，请使用控制台。

- 引导远程导入接受完整的 `version/listeners/providers/routing/global` 文件
- 配置页编辑同一 canonical 契约
- DSL 编辑器可导入同一 YAML，但**仅将 `routing` 反编译为 DSL**
- 决策中的 `modelRefs` 可带 `lora_name`，名称解析到 `routing.modelCards[].loras`

### Helm

Helm values 在 `config` 下镜像同一 canonical 契约。

```yaml
config:
  version: v0.3
  providers:
    defaults:
      default_model: qwen3-8b
    models:
      - name: qwen3-8b
        provider_model_id: qwen3-8b
        backend_refs:
          - name: primary
            endpoint: semantic-router-vllm.default.svc.cluster.local:8000
            protocol: http
  routing:
    modelCards:
      - name: qwen3-8b
```

然后照常 install 或 upgrade：

```bash
helm upgrade --install semantic-router deploy/helm/semantic-router -f values.yaml
```

### Operator

Operator 保持相同逻辑契约，但包在 CRD 内：

- `spec.config.providers`
- `spec.config.routing`
- `spec.config.global`

`spec.vllmEndpoints` 仍是 Kubernetes 原生后端发现适配器。控制器在渲染路由器配置时，将该数据投影到 canonical `providers.models[].backend_refs[]` 与 `routing.modelCards`（含声明的 `loras`）。

详见 [Kubernetes Operator](./k8s/operator)。

### DSL

DSL **仅**拥有 `routing` 表面。

- 编写 `MODEL`、`SIGNAL`、`ROUTE`
- 编译为路由片段
- `providers` 与 `global` 保留在 YAML 中

DSL 编译器输出：

```yaml
routing:
  modelCards:
  signals:
  decisions:
```

**不会**发出 `listeners`、`providers` 或 `global`。

## 导入与迁移

### 引导远程导入

设置向导可从 URL 导入完整 canonical YAML 并应用完整配置（含 `providers`、`routing`、`global`）。

### DSL 导入

DSL 编辑器可导入：

- 完整路由器配置 YAML
- 仅路由的 YAML 片段

两种情况下，**仅 `routing` 节**会被反编译为 DSL。

### 迁移旧配置

对较旧的扁平或混排配置使用 CLI：

```bash
vllm-sr config migrate --config old-config.yaml
```

可迁移的旧形态包括：

- 顶层 `signals`、扁平 `keyword_rules`/`categories`/其他信号块，以及 `decisions`
- 顶层 `model_config`
- 顶层 `vllm_endpoints` 与 `provider_profiles`
- `providers.models[].endpoints`
- 内联 `access_key`

并收敛为 canonical `providers`/`routing`/`global`。

### 导入 OpenClaw 模型提供商

若已有含受支持 OpenAI 兼容端点的 `openclaw.json`，希望由 VSR 接管模型路由并将 OpenClaw 重写为指向首个 VSR 监听器：

```bash
vllm-sr config import --from openclaw --source openclaw.json --target config.yaml
```

省略 `--source` 时，导入器依次检查 `OPENCLAW_CONFIG_PATH`、`./openclaw.json`、`~/.openclaw/openclaw.json`。

## 按环境的快速指南

### Python CLI

1. 以 canonical 形式编写 `config.yaml`。
2. 运行 `vllm-sr validate config.yaml`。
3. 运行 `vllm-sr serve --config config.yaml`。

### 本地路由器

1. 提供商级默认放在 `providers.defaults`，部署绑定放在 `providers.models[].backend_refs[]`。
2. 路由语义放在 `routing.modelCards`/`signals`/`decisions`。
3. 仅将实际需要的运行时覆盖放在 `global.router`/`services`/`stores`/`integrations`/`model_catalog`，模型支撑模块放在 `global.model_catalog.modules`。
4. 仅当进程内 `IntelligentPool` / `IntelligentRoute` 控制器为事实来源时使用 `global.router.config_source: kubernetes`。本地、CLI、控制台、Helm 与 Operator 编写的 canonical YAML 通常保持 `file`。

### Helm

1. 将相同 canonical 配置放在 `values.yaml -> config`。
2. 使用 `helm upgrade --install ... -f values.yaml`。
3. 将 Helm 视为部署封装，而非第二套配置 schema。

### Operator

1. 可移植配置放在 `spec.config`。
2. 仅在需要 Kubernetes 原生后端发现时使用 `spec.vllmEndpoints`。
3. 预期 Operator 从该适配层渲染 canonical 路由器配置。

### DSL

1. 对 `routing.modelCards`、`routing.signals`、`routing.decisions` 使用 DSL。
2. 仍可导入完整 YAML 文件，但只有 `routing` 会反编译为 DSL。
3. 端点、API 密钥、监听器与 `global` 保留在 YAML。
4. 可复用路由片段现位于 `config/signal/`、`config/decision/`、`config/algorithm/`、`config/plugin/`。
