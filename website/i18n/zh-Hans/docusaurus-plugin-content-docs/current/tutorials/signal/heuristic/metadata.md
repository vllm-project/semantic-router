---
translation:
  source_commit: "9716b569"
  source_file: "docs/tutorials/signal/heuristic/metadata.md"
  outdated: false
---

# 元数据信号

## 概览

`metadata` 匹配调用方在请求元数据中提供的、有明确上限的字符串值。它适用于确定性的应用提示，例如同意状态、群组或工作负载类别。

元数据是不受信任的输入。授权和经过身份验证的身份仍使用 `authz` 信号和受信任的请求头。

## 主要优势

- 根据显式的应用上下文进行路由，无需从提示文本中推断
- 将不受信任的提示与经过身份验证的身份分离
- 支持可复用的命名规则，并提供精确匹配、集合成员关系或存在性测试

## 解决什么问题？

有些路由事实不应放在提示中。调用方可能知道某个请求属于金丝雀群组，或远程处理同意已被拒绝。元数据信号将这些事实提供给决策，而无需把它们传递给模型选择器。

## 何时使用

将元数据信号用于不具权威性的应用提示。不要用它们授予权限、绕过防护机制或确立用户身份。

## 配置

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

必须且只能指定一个谓词比较器。请求元数据值均为字符串，并在决策匹配前求值。规则名称和键必须去除首尾空白。请求最多接受 32 个条目，键最多为 128 字节，值最多为 1024 字节。

Chat Completions、Anthropic Messages、`/api/v1/classify/intent` 和 `/api/v1/eval` 均接受相同的顶层字符串映射：

```json
{
  "metadata": {
    "consent": "denied",
    "cohort": "canary"
  }
}
```
