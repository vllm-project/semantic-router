---
is_mtpe: false
translation:
  source_commit: "118fb72eb67a0e7193697e350d7b494dc016631e"
  source_file: "docs/tutorials/signal/application/user-feedback.md"
  outdated: false
---

# 用户反馈信号 {#user-feedback-signal}

## 概述 {#overview}

`user_feedbacks` 读取当前用户轮次对上一个答案的反馈。它提出 `feedback` [预设](../../../system-one/reference/presets#preset-catalog)问题：“What feedback does this user turn give about the previous answer?”，有五个选项：`SAT`、`NEED_CLARIFICATION`、`WRONG_ANSWER`、`WANT_DIFFERENT` 和 `NO_FEEDBACK`。

## 主要优势 {#key-advantages}

- 把纠正和抱怨变成路由事实。
- 只在用户表示答案失败时才升级到更强的模型。
- 区分反馈与普通的追问（`NO_FEEDBACK`）。

## 解决什么问题？ {#what-problem-does-it-solve}

用户说“这不对”或“这不管用”，就是在告诉 Router 上一个模型不够好。没有这个信号时，下一轮仍会交给同一个模型；有了它，决策就可以升级。

## 何时使用 {#when-to-use}

在以下情况使用 `user_feedbacks`：

- 用户纠正后，下一轮应交给更强的模型；
- 用户要求澄清时，应给出更详细的答案；
- 你按路由记录满意或不满意的情况。

## 配置 {#configuration}

```yaml
recipes:
  - name: default
    signals:
      user_feedbacks:
        - name: wrong_answer
          description: The user says the previous answer is wrong.
          labels: [WRONG_ANSWER]
        - name: need_clarification
          labels: [NEED_CLARIFICATION]
        - name: unhappy
          labels: [WRONG_ANSWER, WANT_DIFFERENT]
    decisions:
      - name: escalate-after-correction
        priority: 300
        rules:
          operator: AND
          conditions:
            - {type: user_feedback, name: wrong_answer}
        modelRefs:
          - model: strong-model
```

| 字段 | 含义 |
| --- | --- |
| `name` | 规则名称，供 `{type: user_feedback, name: ...}` 条件使用 |
| `labels` | 规则匹配的反馈选项；没有 `labels` 的规则本身不匹配任何内容，由它的条件用 `label` 指定一个 |
| `predicate` | 选中选项的最低概率；默认为模型的校准阈值（vela3-fast 上为 0.47），否则为 0.5 |
| `deployment`、`timeout_ms` | 回答问题的部署，以及答案的截止时间 |

规则的名称不决定它的反馈：请在 `labels` 中写明。`NO_FEEDBACK` 答案只匹配列出它的规则。

Router 只在一个有文本、且紧跟在助手答案之后的用户轮次上提出这个问题，问题会连同这个答案一起读取该轮次。第一轮、工具结果续写和助手预填都不会提出它。对话后期出现的新问题仍可能被读成反馈；在让反馈改变所选模型之前，请验证这种情况。

### 回答问题的模型 {#the-model-that-answers}

除非 `user_feedback` 绑定或规则的 `deployment` 指定了其他部署，否则由决策模型回答所有反馈规则；见[选择部署](../../../system-one/deploy/deployments)。vela3-fast、Vela 3.0 的 d3 规格和 Vela 2.0 有该问题的训练形式。

完整示例见 [`config/fragments/signal/user-feedback/escalation.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/user-feedback/escalation.yaml)。
