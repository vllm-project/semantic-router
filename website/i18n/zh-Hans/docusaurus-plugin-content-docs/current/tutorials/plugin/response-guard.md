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

在 `recipes[].signals` 下声明信号，并在 `recipes[].decisions[].plugins` 下添加插件：

```yaml
recipes:
  - name: default
    signals:
      jailbreak:
        - name: unsafe_completion
          stage: response
          predicate: {gte: 0.9}
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
| `rules[].when` | 按名称引用信号的条件树，写法同 `decisions[].rules`。`stage: response` 的越狱、安全和 `systemone` 规则，以及所有幻觉和拒答规则读取回答；其他信号读取请求。树的节点上不接受 `on_error` 和 `on_unknown`，由守卫的 `on_unknown` 决定。 |
| `rules[].action` | `block` 以 HTTP 403 拒绝回答，流式回答则以流的错误事件拒绝；`annotate` 交付回答，并在 `x-vsr-warnings` 中写入该规则的警告码；`none` 只记录结果。 |
| `rules[].details` | 配合 `annotate`，在回答开头再写一条说明。幻觉说明会列出不被支持的片段。 |
| `on_unknown` | 规则读取的响应阶段信号无法读取时（检测器失败或超时、回答没有文本，或没有可对照的依据上下文）该规则怎么做：`block`、`annotate`（默认）或 `none`。 |
| `enabled` | `false` 关闭决策从 `global.plugins` 或其配方插件继承的守卫，此时不要求 `rules`。 |

规则按顺序读取，第一条拦截的规则结束读取。响应阶段信号只在规则读到它时才求值，并且无论多少规则读取它，每个回答只求值一次。请求具体模型时绕过配方信号，因此不运行任何守卫，包括 `global.plugins` 中的守卫。

规则添加的警告码有 `hallucination`（回答无法检查时为 `unverified_factual`）、`response_jailbreak`，其他规则为 `response_guard`。被拦截的回答会替换为一条说明发现了什么的消息，而不会说明某项检查失败。

依据上下文是 [context 插件](./context.md)检索到的文档，以及请求自身的工具结果。

`POST /api/v1/plugins/response_guard/preview` 在给定的问题、上下文和回答上读取某个活动绑定的守卫；`mode: probe` 会调用已配置的模型，但不生成回答，也不持久化。

仍使用 `response_jailbreak` 或 `hallucination` 插件的配置会在加载时失败并给出一行提示，`vllm-sr config migrate` 会把它们改写为基于这些插件所检查信号的 `response_guard` 规则。

完整示例见
[`config/fragments/plugin/response-guard/strict.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/plugin/response-guard/strict.yaml)
和
[`config/fragments/plugin/response-guard/grounded-answers.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/plugin/response-guard/grounded-answers.yaml)。

## 流式回答 {#streamed-answers}

能拒绝或改写回答的守卫（含 `block` 规则、带 `details` 的 `annotate` 规则，或 `on_unknown: block`）同样作用于流式回答。这类决策的客户端请求流式响应时（Chat Completions、Responses 或 Messages 上的 `stream: true`），Router 会扣住模型的流，直到流结束、守卫读完整个回答。客户端随后收到以下之一：

- 模型流出的原始事件，包括 usage 块和终止事件，守卫的警告码写入 `x-vsr-warnings`；
- 对带 `details` 的规则，回答开头带有守卫说明的回答事件；
- 缓冲回答的 403 所携带的拒绝消息，以该协议的流错误形式发送：Chat 的 `error` 块、Responses 的 `response.failed` 事件或 Messages 的 `error` 事件。

在 Envoy 之后，Router 只为这些请求要求缓冲响应体；独立运行时，Router 自行缓冲响应体。代价是首 token 时间：模型完成之前客户端收不到任何内容，因此长回答必须在客户端和所有代理的空闲超时之内完成。被扣住的回答上限为 64 MiB，即 Router 接受任何模型响应的大小，Envoy 自身的缓冲上限先于它生效。更大的回答不经读取即以同样的流错误拒绝；没有得到完整回答就结束的流只发送它的失败信息：守卫没有读过的内容不会到达客户端。

规则只做不带 `details` 的 `annotate` 或只记录（`none`）的守卫保持真正的流式传输：客户端在生成过程中即收到回答，守卫在流结束后读取它并把结果记录在 [Router Replay](./router-replay) 中，但它的警告码无法进入 `x-vsr-warnings`，因为响应头已随第一个 token 发出。

Router Replay 用 `stream: buffered` 或 `stream: unbuffered` 记录流式回答的结果。`x-vsr-trace` 以 `response_guard` 携带守卫的结果，被扣住的流还带有 `buffered`；`llm_response_guard_buffered_total` 按决策和结果统计被扣住的流：`passed`、`annotated`、`blocked`、`over_limit` 或 `incomplete`。

## 缓存的回答 {#cached-answers}

同一决策上的 [`response_cache`](./response-cache.md) 只保存守卫放行的回答：被拦截的回答，以及只在 `x-vsr-warnings` 中被标记的回答，都不会存入缓存。缓存命中无需调用模型即可应答，因此守卫不会对它运行。精确命中是同一个请求，守卫对它的判定仍然成立。语义命中返回守卫为先前一个相似请求放行的回答，不会针对新请求求值守卫中依赖请求本身的条件，例如 `keyword` 或 `fact_check` 条件。在这一点重要的场景下，请为该决策的缓存使用 `mode: exact`。

修改守卫会让该决策的缓存重新开始。修改守卫读取的信号（例如越狱规则的 `predicate`）则不会：请在该决策的 `response_cache` 配置中为 `revision.policy_revision` 设置新值。见[配合响应守卫](./response-cache.md#with-a-response-guard)。
