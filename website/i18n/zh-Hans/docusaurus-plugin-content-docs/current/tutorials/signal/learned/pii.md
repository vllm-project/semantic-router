---
translation:
  source_commit: "a65e60e035f593b80c0a9c1963c34a53abe90444"
  source_file: "docs/tutorials/signal/learned/pii.md"
  outdated: false
---

# 个人身份信息信号 {#pii-signal}

## 概览 {#overview}

`pii` 检测请求中的敏感个人数据。在 `routing.signals.pii` 下定义 PII 规则。

它使用通过 `global.model_catalog.system.pii_classifier` 配置的 PII 检测器。

## 主要优势 {#key-advantages}

- 隐私敏感路由显式化。
- 决策可在流量到达后端前拦截、降级或隔离风险流量。
- 支持低风险标识类型的允许列表。
- 隐私策略可在路由与插件间复用。

## 解决什么问题？ {#what-problem-does-it-solve}

没有专用 PII 信号时，隐私敏感流量可能在检测前到达错误模型或插件栈。临时过滤器也使策略更难审计。

`pii` 将个人数据检测变成可复用路由输入。

## 何时使用 {#when-to-use}

在以下情况使用 `pii`：

- 提示可能含受监管或敏感个人数据
- 部分 PII 类型可接受，其他必须触发更安全路由
- 隐私敏感流量需要不同插件或后端
- 路由策略依赖早期 PII 检测

## 配置 {#configuration}

```yaml
routing:
  signals:
    pii:
      - name: restricted_pii
        threshold: 0.85
        include_history: true
        pii_types_allowed:
          - EMAIL_ADDRESS
        description: Sensitive prompts where only low-risk identifiers may pass through.
```

`pii_types_allowed` 为空时，任意检测到的 PII 都可能使信号匹配。

## 完整的本地扫描 {#complete-local-scans}

隐式本地 Vela PII 默认逐项扫描最多 32,768 个 token（含特殊 token）的文本。
每次前向计算最多处理 512 个 token，相邻窗口重叠 255 个内容 token。
窗口由模型 tokenizer 确定；覆盖范围不依赖字符估算或窗口边界处的重新分词。

Candle 和 ORT 保留原始 UTF-8 偏移，按周围上下文为每个 token 选择一次观测，
最后统一解码 BIO 实体。重叠不会重复计算输入用量或实体置信度。
这保证已准入 token 的完整覆盖，不保证检测准确率，也不等于单次 32K 前向的质量。

显式模块预算、后端、窗口或配方绑定保留各自策略。例如，配置
`input: {max_tokens: 8192, overflow: reject}` 的部署仍会拒绝超限输入。
显式启用窗口的示例：

```yaml
global:
  model_catalog:
    modules:
      classifier:
        pii:
          use_mmbert_32k: true
          max_sequence_length: 32768  # 完整文本预算，含特殊 token。
          window: {size: 512, overlap: 255}
```

使用命名绑定时，在部署中声明 `input.overflow: window` 和正整数
`input.max_tokens`；此限制替代模块预算。窗口大小与重叠仍由同一个 `window` 块提供。
不支持的适配器、缺少窗口参数或超过已加载模型容量的限制都会报错。
窗口大小包含 tokenizer 的特殊 token，重叠只计算内容 token。

文本超过文档限制或任一窗口失败时，会返回分类器错误，不会将部分扫描报告为成功。
既有 `on_error` 和决策 `rules.on_unknown` 策略决定路由结果。
远程后端及显式截断配置保留下文所述的部分结果语义。

## 远程后端 {#remote-backend-token_spansv1}

没有 `backend` 时，PII 检测保持本地模型。远程 PII 分类器使用共享 backend 块：`model` 命名 `global.model_catalog.external[]` 中带 `model_role: classification` 的条目，协议是 `http_classify`，约定是 `token_spans.v1`。服务接收 `{"inputs": "<request text>"}`，并回答实体片段：其 `start`/`end` 是该精确字符串中的 Unicode 码点偏移，`label` 来自已配置的 PII 映射，`score` 在 `[0, 1]` 内，以及片段 `text`，必须等于它指向的切片。HuggingFace token 分类拼写 `entity_group` 与 `word` 作为别名接受。裸 JSON 片段列表或信封 `{"spans": [...], "truncated_at": n, "model": "..."}` 都有效；信封的 `model` 若存在，必须等于目录条目的 `llm_model_name`。

当片段超出文本、与自身重叠、携带未知或范围外标签、分数越界、别名值冲突，或正文不是片段列表时，Router 拒绝整个响应，而不是部分接受。已声明的 `truncated_at` 保留截止前的片段，并将其余内容标记为未打分。被拒绝或部分响应对 PII 规则的影响由 `on_error` 决定：`allow`（默认）把未读内容当作未匹配，`block` 将其匹配为 `classification_error`，因此未核验文本不能当作干净通过。

PII 映射不能将 `classification_error` 声明为实体标签。带 `B-`、`I-` 或 `E-` 前缀的别名（含叠放前缀）也被保留，并在任一映射方向的映射加载时被拒绝。

在两种策略下，声明截止前返回的片段都是真实检测。即使其余内容从未被读取，因其中之一匹配的规则仍是真正匹配，因此使用 `rules.on_unknown: no_match` 的决策仍会看到它；只有仅因扫描失败才存在的匹配对决策引擎才是未知。检测 API 在其实体旁用 `scan_incomplete: true` 表达同一含义，因此截断后的 `has_pii: false` 应读作「已读部分中没有」，而不是干净扫描。

```yaml
global:
  model_catalog:
    external:
      - name: pii-service
        model_role: classification
        llm_endpoint:
          address: pii-spans.default.svc
          port: 8080
        llm_model_name: pii-spans-v1
    modules:
      classifier:
        pii:
          backend:
            protocol: http_classify
            contract: token_spans.v1
            model: pii-service
            deadline_ms: 5000
          on_error: block
```

无论片段来自本地模型还是远程后端，`PIIDetected`、`PIIEntities`、`MatchedPIIRules` 与掩码文本都相同；重叠与嵌套片段在掩码前合并。

## 依赖与限制 {#dependencies-and-limitations}

PII 分类器处理提示与可选历史。它是路由控制，不能替代脱敏、加密、访问控制或数据防泄漏。按实体类型校准阈值。完整示例见：
[`config/fragments/signal/pii/strict.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/pii/strict.yaml)。
