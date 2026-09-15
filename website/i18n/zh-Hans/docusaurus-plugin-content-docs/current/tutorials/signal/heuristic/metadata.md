---
translation:
  source_commit: "e56591a9cb24f073bf159927e87116ba6d278741"
  source_file: "docs/tutorials/signal/heuristic/metadata.md"
  outdated: false
---

# 元数据信号 {#metadata-signal}

## 概览 {#overview}

`metadata` 匹配调用方在请求元数据中提供的有界字符串值。面向确定性应用提示，例如同意、队列或工作负载类别。

元数据是不受信任的输入。授权与已认证身份仍使用 `authz` 信号和可信请求头。

## 主要优势 {#key-advantages}

- 按显式应用上下文路由，无需从提示文本推断
- 将不受信任的提示与已认证身份分开
- 支持可复用的命名规则，可做精确、集合成员或存在性测试

## 解决什么问题？ {#what-problem-does-it-solve}

有些路由事实不属于提示。调用方可能知道请求属于金丝雀队列，或远程处理同意已被拒绝。Metadata 信号把这些事实暴露给决策，而不把它们传给模型选择器。

## 何时使用 {#when-to-use}

将 metadata 信号用于非权威应用提示。不要用它们授予权限、绕过防护或确立用户身份。

## 配置 {#configuration}

```yaml
routing:
  signals:
    metadata:
      - name: consent-denied
        key: consent
        predicate:
          equals: denied
      - name: canary-cohort
        key: cohort
        predicate:
          in: [beta, canary]
      - name: has-workload-class
        key: workload_class
        predicate:
          exists: true
```

必须恰好一个谓词比较器。请求元数据值是字符串，在决策匹配前求值。规则名与键必须已去空白。请求最多接受 32 个条目、128 字节的键和 1024 字节的值。

Chat Completions、Anthropic Messages、`/api/v1/diagnostics/classify/intent` 与 `/api/v1/routing/preview` 都接受同一顶层字符串映射：

```json
{
  "metadata": {
    "consent": "denied",
    "cohort": "canary"
  }
}
```

## 依赖与限制 {#dependencies-and-limitations}

元数据由调用方控制，不会转发给模型选择器。它不得授予特权或绕过安全策略；可信身份请使用 `authz`。完整示例见：
[`config/fragments/signal/metadata/routing-hints.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/metadata/routing-hints.yaml)。
