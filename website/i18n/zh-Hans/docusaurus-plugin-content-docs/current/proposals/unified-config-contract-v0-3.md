---
title: 统一配置契约 v0.3
description: 记录路由器、CLI、仪表盘、Helm、operator 和 DSL 共享的已实现配置契约。
created: 2026-03-17
status: Implemented
translation:
  source_commit: "9156d5bc1ed9edff626b95a2b8260a77cb1712c5"
  source_file: "docs/proposals/unified-config-contract-v0-3.md"
  outdated: false
---

> **状态：** 已实现 · **创建日期：** 2026-03-17

## 问题 {#problem}

路由器、CLI、仪表盘、Helm chart、operator 和 DSL 先前解释重叠的配置形态。一个表面接受的文件，在另一表面可能需要翻译或未文档化的默认值。模型身份也与部署端点和凭证混在一起。

## 已实现契约 {#implemented-contract}

公开配置有八个顶层部分：

```yaml
version:
listeners:
providers:
evaluation:
routing:
entrypoints:
recipes:
global:
```

| 部分 | 职责 |
| --- | --- |
| `version` | 选择配置契约。 |
| `listeners` | 定义面向请求和管理的监听器。 |
| `providers` | 将逻辑模型名称绑定到提供商标识符和端点。 |
| `evaluation` | 可选定义运营方自有基准、指数 DAG 和模型关联记录。 |
| `routing` | 定义默认模型卡片、信号、投影、决策、算法和插件。 |
| `entrypoints` | 将面向请求的模型名称映射到默认配置文件或命名配方。 |
| `recipes` | 定义共享提供商和全局基础设施的额外隔离路由配置文件。 |
| `global` | 保存路由器范围的服务、存储、集成、模型模块和稀疏运行时覆盖。 |

未知或已退役形态应以清晰的校验错误失败，而不是在运行时悄悄翻译。

## 提供商与模型边界 {#provider-and-model-boundary}

`providers.defaults` 拥有默认提供商行为和默认模型。
`providers.models[].backend_refs[]` 拥有物理后端绑定。
`providers.models[].api_format` 只拥有上游线格式，从不选择 Provider。Router 拥有的监听器配置中的物理模型必须声明显式后端 Provider；仅元数据的外部网关配置和内置虚拟模型可以保持无后端。
本地 CLI serve 默认使用 standalone 前端，拒绝缺少后端的物理 Chat 模型；`--gateway extproc` 选择 Envoy 传输。仅元数据的外部网关配置使用相应网关集成，原生 System One 模型则引用 model-runtime deployment。
`providers.models[].pricing` 拥有成本感知选择和账务使用的可选部署成本元数据。定价不属于路由模型卡片。

`evaluation` 拥有可选运营方基准定义、指数 DAG 和测量记录。每个 `evaluation.records[].model` 引用一个规范 Model Card 身份，因此可复用评分语义和模型证据有一个顶层所有者，而不嵌入路由元数据。

`routing.modelCards` 描述面向路由的模型身份。可选 `routing.modelCards[].loras` 声明决策可以用 `lora_name` 选择的 LoRA adapter。信号和决策引用逻辑模型名称，而不是端点或凭证。

## 路由与 DSL 边界 {#routing-and-dsl-boundary}

路由拥有：

- 模型卡片；
- 命名信号和投影；
- 决策、候选 `modelRefs`、算法和插件；
- 路由本地输出和适配策略。

算法可以声明 `minimum_candidates` 作为可移植配方契约。无模型资产可以带着该声明并使用空 `modelRefs`；具体入口绑定必须满足它，并且请求时资格过滤器必须在选择或多模型执行开始前保持它。

结构化请求控制在信号边界仍是事实。例如，对话信号暴露协议是要求还是禁止工具执行，投影将这些事实与文本派生观察调和，决策消费面向策略的结果输出。

顶层 `entrypoints` 选择默认路由配置文件或顶层 `recipes` 中的命名项；它们不嵌套在 `routing` 内。

DSL 是路由语义的编写视图。它不拥有提供商凭证、监听器、评估定义或记录、存储或全局运行时服务。导入和导出保持同一规范路由文档，而不是发明另一稳态 schema。

分类器后端失败以 `Unknown` 进入决策评估。`NOT` 保留该状态，而 `AND` 和 `OR` 使用 CEL 风格的短路语义。决策用根级 `rules.on_unknown: no_match|match|fail_request` 解析终端 `Unknown`；省略则保留现有按族兼容行为。

## 入口点与多配方路由 {#entrypoints-and-multi-recipe-routing}

`entrypoints[]` 将请求模型名称映射到顶层路由或一个命名配方。`recipes[]` 包含复用同一提供商清单和全局运行时的隔离路由配置文件。

这使公开 API 保持稳定，同时允许一个进程中共存多种路由策略。入口点在信号和决策运行之前解析配方。

## 默认值与配置来源 {#defaults-and-configuration-source}

内置默认值位于路由器中。`global.router.config_source` 选择基于文件的配置或 Kubernetes CRD 调和。外部模板不得在校验后应用隐藏默认值。

内置模型资源位于 `global.model_catalog.deployments`，`system.decision_model` 选择默认判断部署。模块保存任务策略，global bindings 提供共享默认绑定，recipe 的 `routing.model_bindings` 可为任务选择其他资源或契约。`candle`、`ort`、`openvino` 及 `variant` 等原生执行字段已退役，参见[模型运行时迁移](../model-runtime/migrate.md)。

领域、复杂度、安全、PII 等消费者共用此边界。外部分类后端仍声明 `protocol`、`contract`、`model`、`deadline_ms`；`model` 是外部目录中的精确名称。接受多种响应契约的消费者必须显式指定契约。连接器响应体上限由其 `max_response_bytes` 管理。

命名 recipe 不继承顶层 `routing` 的 signals、decisions、strategy 或 fallback，各自解析内置默认值。没有显式默认 recipe 入口时，`vllm-sr/auto` 指向顶层 routing；显式 `recipe: default` 映射替换这一名称。Replay 数据保留策略归属 `global.services.router_replay`，可由决策插件覆盖。

仪表盘、Helm chart 和 operator 可以帮助用户编写或传输配置，但得到的文档仍使用同一契约。

## 仓库来源 {#repository-sources}

`config/config.yaml` 是详尽的规范参考配置。可复用示例位于：

- `config/fragments/signal/`；
- `config/fragments/decision/`；
- `config/fragments/algorithm/`；以及
- `config/fragments/plugin/`。

运行时部署示例与路由片段保持分开。契约测试和 `make check` 保持参考配置、schema、示例和公开文档对齐。

## 迁移 {#migration}

使用 `vllm-sr config migrate --config old-config.yaml` 转换受支持的旧布局。审阅结果，通过部署的密钥机制解析凭证，并在服务前校验。

`vllm-sr init` 已移除。规范 YAML 是稳态配置来源；交互式或图形编写工具必须导出同一文档。

## 范围与非目标 {#scope-and-non-goals}

该契约统一配置所有权。它不要求每个编写表面都以一种形式暴露每个高级字段，也不使 DSL 成为部署语言的替代。

## 参考资料 {#references}

- [当前配置指南](../installation/configuration)
- [配置工作流](../installation/configuration-workflows)
- [信号、决策与模型选择](../overview/signal-driven-decisions)
- [虚拟模型](../tutorials/global/entrypoints-and-recipes)
- [相关议题 #1505](https://github.com/vllm-project/semantic-router/issues/1505)
