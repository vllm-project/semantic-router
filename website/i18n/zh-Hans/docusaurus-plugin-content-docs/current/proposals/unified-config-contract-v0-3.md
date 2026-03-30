---
translation:
  source_commit: "baa07413"
  source_file: "docs/proposals/unified-config-contract-v0-3.md"
  outdated: false
---

# 统一配置契约 v0.3

Issue: [#1505](https://github.com/vllm-project/semantic-router/issues/1505)

---

## 以前

在 v0.3 之前，仓库里存在多份部分重叠的配置契约：

- 路由器运行时消费扁平的 Go 配置
- Python CLI 使用自己的嵌套 YAML 及合并/默认逻辑
- 控制面板与引导流程导入 YAML，但仍假设旧版顶层 `signals` 与 `decisions`
- Helm 与 Operator 各自以不同方式翻译配置
- DSL 将路由语义与旧版 `BACKEND`、`GLOBAL` 预期混在一起

这带来三个长期问题：

1. 同一概念需要在多个 schema 层反复编辑。
2. 端点、API Key 与模型语义被混在一起。
3. 运行时默认依赖 `router-defaults.yaml` 等外部模板，难以推理与替换。

## 旧模型的问题

### CLI 与路由器漂移

Python CLI 与 Go 路由器没有单一的 schema 所有者。用户可能通过 CLI、控制面板或 Kubernetes 构建配置，仍会遇到结构不一致。

### 模型语义与部署绑定纠缠

逻辑模型同时承载：

- 语义路由身份
- 端点绑定
- API Key
- 提供商模型 ID

导致难以复用。若多个逻辑模型指向同一后端，配置仍会重复后端细节。

### DSL 范围过宽

DSL 适合表达路由语义，但旧版 `BACKEND` 与 `GLOBAL` 块使其看起来也像部署与运行时状态的编写入口，在本地、面板与 Kubernetes 工作流下不可持续。

## v0.3 契约

v0.3 定义唯一 canonical 配置：

```yaml
version:
listeners:
providers:
routing:
global:
```

### 各节含义

- `providers`：部署绑定与提供商默认值
- `routing`：语义路由图
- `global`：稀疏的路由器级运行时覆盖

### DSL 边界

DSL 仅拥有：

- `routing.modelCards`
- `routing.signals`
- 用于信号协调与派生路由输出的 `routing.projections`
- `routing.decisions`

不再拥有端点、API Key、监听器或路由器全局运行时设置。

### 部署绑定拆分

模型语义与部署绑定被显式分离：

- `routing.modelCards` 承载语义目录数据（规模、上下文窗口、描述、能力等）
- `routing.modelCards[].loras` 承载每个逻辑模型的 canonical LoRA 适配器目录
- `providers.defaults` 承载提供商级默认，如 `default_model`、`reasoning_families`、`default_reasoning_effort`
- `providers.models` 直接承载各模型的访问绑定
- 每个 `providers.models[].backend_refs[]` 项自带传输与鉴权字段，如 `endpoint`、`base_url`、`protocol`、`auth_header`、`auth_prefix`、`api_key`、`api_key_env`
- `routing.decisions[].modelRefs[].lora_name` 与对应 `routing.modelCards[].loras` 条目解析，故 `lora_name` 现为受支持的路由契约的一部分，而非仅运行时逃生舱

## 全局默认

路由器级默认由路由器自身持有，不再依赖第二份用户维护的默认文件。

- 路由器提供类型化的内置默认
- `global:` 仅覆盖需要修改的字段
- `global.router` 聚合路由引擎控制项，含 `config_source`
- `global.services` 聚合共享 API 与运行时服务
- `global.stores` 聚合有存储支撑的服务
- `global.integrations` 聚合辅助运行时集成
- `global.model_catalog` 聚合路由器持有的模型资产，含 `embeddings`、`system`、`external`、`classifiers`、`modules`，以及如 `embedding_config.top_k` 等嵌入回退旋钮
- `global.model_catalog.modules` 存放路由器自有模块设置，如 `prompt_compression`、`prompt_guard`、`classifier`、`hallucination_mitigation`、`feedback_detector`、`modality_detector`
- 省略的字段保留内置默认

这使本地、面板、Helm 与 Operator 在相同基线上一致。

## 各入口如何统一

### 引导导入

远程引导导入可拉取并应用完整 canonical YAML，保留「一份远程 YAML 即可配置整台路由器」的初衷。

### DSL 导入

DSL 导入仍接受完整路由器配置 YAML，但仅将 `routing` 节反编译为 DSL。静态部署与全局运行时设置保留在 YAML 中。

路由器解析器对稳态运行时配置**仅**接受 canonical v0.3 YAML。旧版混排布局须先经显式迁移。

进程内 CRD 协调路径也通过 `global.router.config_source: kubernetes` 回到同一 canonical 解析器，而不再维护单独的稳态运行时布局。

### 仓库配置资产

仓库不再在 `config/intelligent-routing/` 等目录下提供大型完整示例树，而是：

- `config/config.yaml` 为详尽的 canonical 参考配置
- `config/signal/`、`config/decision/`、`config/algorithm/`、`config/plugin/` 存放可复用的路由片段
- `config/decision/` 按布尔规则形状组织（`single`、`and`、`or`、`not`、`composite`）
- `config/algorithm/` 按路由策略族组织（`looper`、`selection`）
- 最新 `docs/tutorials/` 源码树与 `signal/decision/algorithm/plugin/global` 对齐，旧教程树已从活跃文档面移除
- 运行时支持示例如 `deploy/examples/runtime/semantic-cache/`、`response-api/`、`tools/` 保持独立，因其不属于面向用户的配置契约
- 仅测试台使用的清单位于 `e2e/config/`
- `go test ./pkg/config/...` 与 `make agent-lint` 约束 `config/config.yaml` 与公开配置契约对齐且保持详尽

### Helm 与 Operator

Helm values 与 Operator 配置对齐到相同 canonical 概念，而非另造稳态路由器 schema。

## 迁移路径

旧配置可用：

```bash
vllm-sr config migrate --config old-config.yaml
```

该命令将旧配置重写为 canonical 的 `providers`/`routing`/`global`。

覆盖混排嵌套布局与旧版扁平运行时布局（如顶层 `keyword_rules`、`model_config`、`vllm_endpoints`、`provider_profiles`）。

清理过程中已移除 `vllm-sr init`。canonical `config.yaml` 现为唯一预期用户编写的稳态文件。

## 结果

仓库现在只有一套公开的配置叙事：

- 完整路由器配置位于 canonical YAML
- DSL 是该配置的**路由语义**视图
- 部署绑定位于 `providers.defaults` 与 `providers.models[]`
- 运行时覆盖位于 `global.router`、`global.services`、`global.stores`、`global.integrations`、`global.model_catalog`，模型相关模块在 `global.model_catalog.modules`
- 结构信号 `density` 使用单一内置多语言归一化路径，不再暴露按规则的 `normalize_by` 选择
- `global.router.config_source` 是文件配置与 Kubernetes CRD 协调之间的 canonical 开关
- 内置默认由路由器持有
- 仓库样例资产按 `signal/decision/algorithm/plugin` 片段组织，而非并行完整配置示例

这消除了旧有 CLI/路由器/面板/Helm/Operator 漂移，使各环境共享同一契约。
