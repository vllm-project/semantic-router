---
is_mtpe: false
translation:
  source_commit: "118fb72eb67a0e7193697e350d7b494dc016631e"
  source_file: "docs/tutorials/signal/application/safety.md"
  outdated: false
---

# 安全信号 {#safety-signal}

## 概述 {#overview}

`safety` 检测请求或模型答案中的有害内容，可选地按指定的危害类别检测。没有 `labels` 的规则提出 `harm` [预设](../../../system-one/reference/presets#preset-catalog)问题：“Is this request harmful?”，选项为 `safe` 和 `unsafe`。带 `labels` 的规则询问 34 个[危害类别](../../../system-one/reference/presets#harm-categories)中哪些适用，并匹配它列出的类别。

## 主要优势 {#key-advantages}

- 把内容风险与提示攻击分开；后者由 `jailbreak` 检测。
- 可以按具体危害类别路由，而不仅仅是安全或不安全。
- 在路由前检查请求，在路由后检查答案。

## 解决什么问题？ {#what-problem-does-it-solve}

内容风险和提示攻击需要不同的标签。对有害话题的善意讨论可以是安全的，而有害的请求不一定试图覆盖任何指令。`safety` 读取内容风险；`jailbreak` 读取攻击。

## 何时使用 {#when-to-use}

宽泛的内容策略使用不带 labels 的规则；只有部分类别应触发某个决策时，使用带 labels 的规则。在规则开始拒绝流量之前，请在你的应用所用的语言和输入长度上验证 predicate。

## 配置 {#configuration}

```yaml
recipes:
  - name: default
    signals:
      safety:
        - name: unsafe-content
        - name: violent-or-criminal
          labels:
            - Violence
            - Criminal Planning/Confessions
            - Guns and Illegal Weapons
            - S1
            - S2
        - name: unsafe-answer
          stage: response
          predicate: {gte: 0.6}
    decisions:
      - name: handle-content-risk
        priority: 300
        rules:
          operator: AND
          on_unknown: fail_request
          conditions:
            - {type: safety, name: unsafe-content}
        modelRefs:
          - model: safety-capable-model
```

| 字段 | 含义 |
| --- | --- |
| `name` | 规则名称，供 `{type: safety, name: ...}` 条件和防护规则使用 |
| `labels` | 规则匹配的危害类别；不写时，规则匹配不安全的请求 |
| `predicate` | 不带 labels 时，`unsafe` 的最低概率：请求阶段默认为模型的校准阈值（vela3-fast 上为 0.52），否则为 0.5。带 labels 时，列出类别的最低概率；不写时以模型自己的选择为准 |
| `stage` | `request`（默认）或 `response`，后者读取模型的答案 |
| `deployment`、`timeout_ms` | 回答问题的部署，以及答案的截止时间 |

每个阶段中，所有不带 labels 的规则共享一个问题，所有带 labels 的规则共享 34 类别问题；每条规则应用自己的 labels 和 predicate。阳性答案可能来自正处于危机中寻求帮助的人：它并不意味着恶意。请在决策中选择响应策略，并让提示攻击决策与内容风险决策并存，明确设置它们的优先级。

问题读取完整输入，输入超过一次前向计算的长度时分窗口读取。与 `jailbreak` 一样，它也会读取 `global.router.routing_input` 从路由文本中移除的内容。模型没有给出答案，或没能读完完整输入时，规则为未知，由决策的 `rules.on_unknown` 决定：上面的 `fail_request` 会返回 HTTP 503。

### 危害类别 {#harm-categories}

这些类别是 vela3-fast 训练时使用的 21 个 AEGIS 2.0 类别和 13 个 MLCommons 危害代码；[参考](../../../system-one/reference/presets#harm-categories)列出了全部类别。vela3-fast 用一个训练过的 set 问题回答全部 34 个类别。其他模型在同一次调用中零样本回答同样的标签，每个类别一个是/否问题。提到某个话题并不自动等于有害；请用你自己的数据测量你依赖的类别。

### 在响应阶段 {#at-the-response-stage}

`stage: response` 规则针对模型的答案提问。只有 [`response_guard`](../../plugin/response-guard) 读取它；读取它的决策会在加载时被拒绝。

### 回答问题的模型 {#the-model-that-answers}

除非 `safety` 绑定或规则的 `deployment` 指定了其他部署，否则由决策模型回答所有安全规则；见[选择部署](../../../system-one/deploy/deployments)。vela3-fast、Vela 3.0 的 d3 规格和 Vela 2.0 有安全/不安全问题的训练形式。

完整示例见 [`config/fragments/signal/safety/content-safety.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/safety/content-safety.yaml)。
