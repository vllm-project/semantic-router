---
translation:
  source_commit: "baa07413"
  source_file: "docs/tutorials/signal/overview.md"
  outdated: true
---

# 信号（Signal）

## 概览

`signal/` 是 `routing` 的**检测层**。

信号在 `routing.signals` 下定义命名检测器；决策在 `routing.decisions` 中引用这些名称，使检测可复用、路由逻辑可读。跨信号协调与派生路由档位现位于 `routing.projections`：`routing.projections.partitions` 是独占领域或嵌入分区的运行时载体；决策可用 `type: projection` 引用 `routing.projections.mappings` 的输出。DSL 编写中对应 `PROJECTION partition ...` 以及 `PROJECTION score ...` / `PROJECTION mapping ...` 块。完整投影工作流、规范 YAML、控制台路径与 DSL 示例见 [Projections](../projection/overview)。

本教程组直接映射 `config/signal/` 下的片段树，文档按提取方式组织：

- `heuristic/`：请求形态、词法、身份与轻量检测器信号
- `learned/`：依赖嵌入或分类器、使用路由器自有模型资产或维护检测模块的信号

## 主要优势

- 同一检测器可被多条决策复用。
- 检测逻辑与路由结果分离。
- 一条路由可组合词法、策略、语义与安全输入。
- 信号名成为稳定策略构件，配置审查更容易。

## 解决什么问题？

没有信号层时，每条决策都要内联检测逻辑，导致重复、策略难审计，并把「检测到了什么」与「应该做什么」混在一起。

信号把请求理解变成命名目录，供路由图其余部分组合。

## 何时使用

在以下情况使用 `signal/`：

- 多条路由需要同一检测器
- 同一决策树要混合多种检测方式
- 需要在检测、决策逻辑、算法与插件之间划清边界
- 希望配置片段清晰映射到 `config/signal/`

## 配置

在规范 v0.3 YAML 中，信号位于 `routing.signals`：

```yaml
routing:
  signals:
    keywords:
      - name: urgent_keywords
        operator: OR
        keywords: ["urgent", "asap"]
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
    mappings:
      - name: request_band
        source: request_difficulty
        method: threshold_bands
        outputs:
          - name: support_escalated
            gte: 0.25
```

最新信号文档仍覆盖 `config/signal/` 下各族，但按两级分类组织，便于看清运行时成本与依赖模型。

### 启发式信号

这类信号来自显式规则、请求形态或轻量检测，**不依赖**路由器自有分类模型。

| 信号族 | 片段目录 | 用途 | 文档 |
| ------ | -------- | ---- | ---- |
| `authz` | `config/signal/authz/` | 按身份、角色或租户策略路由 | [Authz](./heuristic/authz) |
| `context` | `config/signal/context/` | 按有效 token 窗口需求路由 | [Context](./heuristic/context) |
| `keyword` | `config/signal/keyword/` | 词法或 BM25 风格匹配 | [Keyword](./heuristic/keyword) |
| `language` | `config/signal/language/` | 按检测到的请求语言路由 | [Language](./heuristic/language) |
| `structure` | `config/signal/structure/` | 按请求形态（如问题数量、有序工作流标记）路由 | [Structure](./heuristic/structure) |

### 学习型信号

这类信号使用嵌入或分类模型，通常依赖 `global.model_catalog` 资产或模块配置。

| 信号族 | 片段目录 | 用途 | 文档 |
| ------ | -------- | ---- | ---- |
| `complexity` | `config/signal/complexity/` | 检测难/易推理流量 | [Complexity](./learned/complexity) |
| `domain` | `config/signal/domain/` | 请求主题族分类 | [Domain](./learned/domain) |
| `embedding` | `config/signal/embedding/` | 语义相似度匹配 | [Embedding](./learned/embedding) |
| `modality` | `config/signal/modality/` | 纯文本、图像生成或混合输出模式 | [Modality](./learned/modality) |
| `fact-check` | `config/signal/fact-check/` | 需证据核验的提示 | [Fact Check](./learned/fact-check) |
| `jailbreak` | `config/signal/jailbreak/` | 提示注入或越狱企图 | [Jailbreak](./learned/jailbreak) |
| `pii` | `config/signal/pii/` | 敏感个人数据 | [PII](./learned/pii) |
| `preference` | `config/signal/preference/` | 推断响应风格偏好 | [Preference](./learned/preference) |
| `kb` | `config/signal/kb/` | 将知识库标签或分组绑定为命名路由信号 | [Knowledge Base](./learned/kb) |
| `user-feedback` | `config/signal/user-feedback/` | 纠正或升级类反馈 | [User Feedback](./learned/user-feedback) |

请遵守：

- 信号命名且可复用
- 信号只做检测；路由结果归属 `decision/`
- 分区与派生档位放在 `routing.projections`，不要塞回 `routing.signals`
- 模型选择分离，归属 `algorithm/`
- 路由侧行为分离，归属 `plugin/`

## 下一步

- 需要 `PROJECTION partition`、加权分数聚合或命名档位时，阅读 [Projections](../projection/overview)。
- 完整公开约定见 [`config/config.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/config.yaml)。
- 仓库内真实策略可参考维护的 `balance` 资产：
  - [`deploy/recipes/balance.yaml`](https://github.com/vllm-project/semantic-router/blob/main/deploy/recipes/balance.yaml)
  - [`deploy/recipes/balance.dsl`](https://github.com/vllm-project/semantic-router/blob/main/deploy/recipes/balance.dsl)
