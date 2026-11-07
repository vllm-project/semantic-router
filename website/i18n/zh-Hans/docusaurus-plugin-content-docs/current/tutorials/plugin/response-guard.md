---
is_mtpe: true
translation:
  source_commit: "72b3d17c5fdcaa506e603b1697fdb2d4b6318a6e"
  source_file: "docs/tutorials/plugin/response-guard.md"
  outdated: false
---

# 响应守卫

## 概览

`response_guard` 是一个路由局部插件，按它所检查的信号拦截或标注模型的回答。每条规则像决策规则一样按名称引用信号，条件成立时执行它的动作。

## 主要优势

- 一个插件覆盖对回答的所有检查：输出中的越狱内容、上下文不支持的说法，以及任何响应阶段的 `systemone` 问题。
- 检测器和阈值归信号所有；规则只引用名称，一次检测服务所有读取它的规则。
- 请求上的条件（例如事实核查结果）可以先于代价较高的回答检查：`AND` 在第一个为假的条件处停止。

## 它解决什么问题？

路由在模型回答之前选定模型，因此决策无法依据模型说了什么来行动。`response_guard` 是回答之后的那一步：它在回答上读取响应阶段信号，在请求上读取请求阶段信号，据此拒绝、标记或放行回答。

## 何时使用

- 携带越狱内容的回答不能到达客户端
- 有依据或带工具的路由需要标记上下文不支持的回答
- 路由需要记录对回答的判定而不修改回答

## 配置

在 `routing.signals` 下声明信号，并在 `routing.decisions[].plugins` 下添加插件：

```yaml
routing:
  signals:
    jailbreak:
      - name: unsafe_completion
        direction: response
        threshold: 0.9
    fact_check:
      - name: needs_fact_check
        description: Prompts whose answers are worth checking against their context.
    hallucination:
      - name: ungrounded_claims
        description: Answers with claims their grounding context does not support.
  decisions:
    - name: grounded_support
      # ...rules, modelRefs...
      plugins:
        - type: response_guard
          configuration:
            rules:
              - when: {type: jailbreak, name: unsafe_completion}
                action: block
              - when:
                  operator: AND
                  conditions:
                    - {type: fact_check, name: needs_fact_check}
                    - {type: hallucination, name: ungrounded_claims}
                action: annotate
                details: true
            on_unknown: annotate
```

| 字段 | 含义 |
|---|---|
| `rules[].when` | 按名称引用信号的条件树，写法同 `decisions[].rules`。`direction: response` 的越狱规则、幻觉规则和响应阶段的 `systemone` 规则读取回答；其他信号读取请求。树的节点上不接受 `on_error` 和 `on_unknown`，由守卫的 `on_unknown` 决定。 |
| `rules[].action` | `block` 以 HTTP 403 拒绝回答；`annotate` 交付回答，并在 `x-vsr-response-warnings` 中写入该规则的警告码；`none` 只记录结果。 |
| `rules[].details` | 配合 `annotate`，在回答开头再写一条说明。幻觉说明会列出不被支持的片段。 |
| `on_unknown` | 规则读取的响应阶段信号无法读取时（检测器失败或超时、回答没有文本，或没有可对照的依据上下文）该规则怎么做：`block`、`annotate`（默认）或 `none`。 |

规则按顺序读取，第一条拦截的规则结束读取。响应阶段信号只在规则读到它时才求值，并且无论多少规则读取它，每个回答只求值一次。

规则添加的警告码有 `hallucination`（回答无法检查时为 `unverified_factual`）、`response_jailbreak`，其他规则为 `response_guard`。被拦截的回答会替换为一条说明发现了什么的消息，而不会说明某项检查失败。

依据上下文是 [context 插件](./context.md)检索到的文档，以及请求自身的工具结果。

流式回答只有在字节已发给客户端后才完整存在，因此它的规则会被读取，并以 `enforcement: not_enforced_streaming` 记录在 [Router Replay](./router-replay) 中，但不会执行任何动作。规则必须拦截时，请使用非流式响应。

`POST /api/v1/plugins/response_guard/preview` 在给定的问题、上下文和回答上读取某个活动绑定的守卫；`mode: probe` 会调用已配置的模型，但不生成回答，也不持久化。

仍使用 `response_jailbreak` 或 `hallucination` 插件的配置会在加载时失败并给出一行提示，`vllm-sr config migrate` 会把它们改写为基于这些插件所检查信号的 `response_guard` 规则。

完整示例见
[`config/fragments/plugin/response-guard/strict.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/plugin/response-guard/strict.yaml)
和
[`config/fragments/plugin/response-guard/grounded-answers.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/plugin/response-guard/grounded-answers.yaml)。
