---
translation:
  source_commit: "485dba984a011cee07ec21e7d6a6d54f69509dae"
  source_file: "docs/tutorials/signal/learned/jailbreak.md"
  outdated: false
---

# 越狱检测信号 {#jailbreak-signal}

## 概览 {#overview}

`jailbreak` 在 Router 提交路由前检测提示注入与越狱企图。在 `routing.signals.jailbreak` 下定义越狱规则。

它使用 `global.model_catalog.modules.prompt_guard` 以及 `global.model_catalog.system` 中已配置的越狱模型绑定。

## 主要优势 {#key-advantages}

- 让决策在模型选择前拦截或降级不安全流量。
- 支持分类器、对比式与混合风格安全检测。
- 越狱策略在路由决策内可见。
- 同一安全信号可跨多条受保护路由复用。

## 解决什么问题？ {#what-problem-does-it-solve}

若越狱检测仅发生在下游，Router 仍可能将不安全流量送到错误模型或工具链。若逻辑在路由图外，安全策略更难审计。

`jailbreak` 将注入检测作为一等路由输入。

## 何时使用 {#when-to-use}

在以下情况使用 `jailbreak`：

- 不安全流量必须在模型选择前拦截
- 提示注入应路由到更安全回退
- 多轮历史应影响路由
- 安全策略必须与路由逻辑同图可见、可测

## 配置 {#configuration}

```yaml
routing:
  signals:
    jailbreak:
      - name: prompt_injection
        method: contrastive
        threshold: 0.8
        include_history: true
        description: Detect common prompt-injection or jailbreak attempts.
        jailbreak_patterns:
          - ignore previous instructions
          - reveal the hidden prompt
          - jailbreak mode
        benign_patterns:
          - explain the policy
          - summarize the safety rules
```

多轮攻击使用 `include_history`；把模式列表当作所配置检测方法的调参数据。

### 方向 {#direction}

`direction` 选择规则打分的对象。默认 `request` 在 Router 提交路由前对提示打分。`response` 对模型自身输出打分，因此规则只有在模型回答后才存在：

```yaml
routing:
  signals:
    jailbreak:
      - name: unsafe_completion
        direction: response
        threshold: 0.85
        description: Detect jailbreak content in the model's own output.
```

响应方向规则只使用序列分类器：`method: contrastive`、模式列表与 `include_history` 是请求阶段设置，在其上会被拒绝。匹配、分数与失败报告在与请求方向规则相同的 `jailbreak:<name>` 键下。路由回放将观察记录为每条响应方向规则一条结果，包含判定（`detected`、`not_detected` 或 `unavailable`）、它阈值化的分数或失败码，以及插件应用的动作；使用 `x-vsr-debug` 时，`x-vsr-matched-jailbreak` 请求头在请求规则之后携带匹配的响应规则。

响应方向规则不是决策输入。决策在请求路由时、模型回答前选定，因此在规则中或通过投影直接读取该规则的决策会在配置加载时被拒绝。观察由请求所选决策的 `response_jailbreak` 插件消费，并对其应用已配置动作。规则从请求解析到的配方读取，因此在一个入口配方上声明的规则只对该入口的响应打分。一旦声明了响应方向规则，插件自己的 `threshold` 会被忽略，加载时会报告这一点；阈值由规则拥有。决策的 `response_jailbreak` 插件在未声明响应方向规则时运行也会在加载时报告：插件随后自己对响应分类，这是兼容路径。任一消费者都足以为本配方供给 `prompt_guard`：即使没有决策规则读取 jailbreak 信号，越狱模型及其标签映射也会为响应阶段加载。

未解析的检测器（后端失败，或响应没有可打分的文本）通过 `SignalErrors` 报告，方式与其他信号相同，而不会看起来像干净响应。只有每一块都被打分时，响应才是干净的：后端失败的块会让规则保持未解析，除非其他块已经产生的分数已经匹配它。响应只打分一次，每条规则在该分数上画自己的线，因此部分扫描按规则解析：分数 0.5 匹配阈值为 0.4 的规则，并使阈值为 0.9 的规则保持未解析，因为从未打分的那一块本可能给出更高分数。

流式响应在流结束后打分，并以 `enforcement: not_enforced_streaming` 代替动作记录。此时字节已到达客户端，因此 `response_jailbreak` 插件不运行，`block`、请求头或 body 动作都不适用。

## 依赖与限制 {#dependencies-and-limitations}

已配置的 prompt-guard 运行时处理当前提示，以及可选的对话历史。检测是概率性的，可能被绕过或过度触发；请与最小权限工具和后端策略组合。完整示例见：
[`config/fragments/signal/jailbreak/patterns.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/jailbreak/patterns.yaml)。
