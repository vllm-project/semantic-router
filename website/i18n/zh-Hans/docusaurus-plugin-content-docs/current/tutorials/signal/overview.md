---
translation:
  source_commit: "faf639ff0c8622c4dff8165b89aa304b7dd93762"
  source_file: "docs/tutorials/signal/overview.md"
  outdated: false
---

# 信号 {#signal}

## 概览 {#overview}

信号把请求事实变成决策可复用的名称。例如，信号可以识别长提示、工具密集的对话、某种语言，或可能的提示注入。决策再根据该信号是否匹配来选择动作。

需要在决策运行前把多个信号组合成分数、分区或路由档位时，使用[投影](../projection/overview)。

## 主要优势 {#key-advantages}

- 同一检测器可被多条决策复用。
- 检测逻辑与路由结果分离。
- 一条路由可组合词法、策略、语义与安全输入。
- 信号名成为稳定策略构件，配置审查更容易。

## 解决什么问题？ {#what-problem-does-it-solve}

没有信号层时，每条决策都要内联检测逻辑。这会造成重复、路由策略难审计，并把「检测到了什么」与「应该做什么」混在一起。

信号把请求理解变成命名目录，供路由图其余部分组合。

## 何时使用 {#when-to-use}

在以下情况使用信号：

- 多条路由需要同一检测器
- 同一决策树要混合多种检测方式
- 需要在检测、决策逻辑、算法与插件之间划清边界
- 希望调检测而不改写路由结果

## 配置 {#configuration}

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

按需要检测的事实类型选择信号。

### 启发式信号 {#heuristic-signals}

这类信号使用显式规则、请求形态、身份或轻量检测器，不需要通用分类器模型。

| 信号 | 用于 |
| ------ | --------- |
| [Authz](./heuristic/authz) | 按可信身份、角色或租户策略路由 |
| [Conversation](./heuristic/conversation) | 检测多轮、工具密集或智能体式请求结构 |
| [Context](./heuristic/context) | 按有效上下文窗口需求路由 |
| [Event](./heuristic/event) | 按类型、严重级别、动作码或紧急程度检测结构化事件 |
| [Input Modality](./heuristic/input-modality) | 检测存在哪些输入模态（文本、图像、音频、视频） |
| [Keyword](./heuristic/keyword) | 匹配显式词、短语、BM25 词项或 n-gram |
| [Language](./heuristic/language) | 按检测到的请求语言路由 |
| [Metadata](./heuristic/metadata) | 使用有界、调用方提供的应用提示 |
| [Structure](./heuristic/structure) | 检测提示中的计数、密度与有序标记 |

### 学习型信号 {#learned-signals}

这类信号使用嵌入、分类器或已配置的检测模型。选择远程提供方前，请阅读各页的数据处理说明。

| 信号 | 用于 |
| ------ | --------- |
| [Classifier](./learned/classifier) | 暴露自定义本地分类器或外部 LLM 的标签 |
| [Complexity](./learned/complexity) | 估计 easy、medium 或 hard 推理流量 |
| [Domain](./learned/domain) | 对请求主题分类 |
| [Embedding](./learned/embedding) | 用代表性示例匹配语义意图 |
| [Modality](./learned/modality) | 分类文本、图像生成或混合输出意图 |
| [Fact Check](./learned/fact-check) | 检测可能需要证据核验的提示 |
| [Hallucination](./learned/hallucination) | 对照给定依据上下文检查模型回答 |
| [Jailbreak](./learned/jailbreak) | 检测提示注入或越狱企图 |
| [Safety](./learned/safety) | 检测不安全内容及可选风险类别 |
| [PII](./learned/pii) | 检测敏感个人数据 |
| [Preference](./learned/preference) | 推断响应风格偏好 |
| [Reask](./learned/reask) | 检测近期对话历史中的重复提问 |
| [Knowledge Base](./learned/kb) | 匹配可复用示例集中的标签或分组 |
| [User Feedback](./learned/user-feedback) | 检测纠正、不满或升级反馈 |

请遵守：

- 信号命名且可复用
- 信号只做检测；路由结果归属 `decision/`
- 分区与派生路由档位放在 `routing.projections`，不要塞回 `routing.signals`
- 模型选择分离，归属 `algorithm/`
- 路由侧行为分离，归属 `plugin/`

## 下一步 {#next-steps}

- 需要 `PROJECTION partition`、加权分数聚合或命名路由档位时，阅读[投影](../projection/overview)。
- 完整公开约定见 [`config/config.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/config.yaml)。
- 完整路由策略见 `balance` 配方：
  - [`config/recipes/balance/config.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/recipes/balance/config.yaml)
  - [`config/recipes/balance/recipe.dsl`](https://github.com/vllm-project/semantic-router/blob/main/config/recipes/balance/recipe.dsl)

信号可按族检查请求文本、对话历史、图像、调用方元数据或可信身份。学习型信号可能把这些数据发给已配置的远程分类器或嵌入提供方。作为策略关卡使用前，请阅读各族页面上的依赖与数据说明。
