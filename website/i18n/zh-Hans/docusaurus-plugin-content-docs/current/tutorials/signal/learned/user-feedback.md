---
translation:
  source_commit: "371e7010ff1b89b4d2a3f4d9ffc020e6574f0452"
  source_file: "docs/tutorials/signal/learned/user-feedback.md"
  outdated: false
---

# 用户反馈信号 {#user-feedback-signal}

## 概览 {#overview}

`user-feedback` 从对话中检测纠正、不满或升级反馈。在 `routing.signals.user_feedbacks` 下定义其标签。

该族为学习型：依赖 `global.model_catalog.modules.feedback_detector` 下配置的反馈检测器。将 `feedback_mapping_path` 留空，使检测器从模型的 `config.json` 读取 `id2label`；仅在映射文件的索引顺序与模型头一致时才设置它。

## 主要优势 {#key-advantages}

- 用户表示答案错误或不清时，Router 可作出反应。
- 升级行为在路由决策内可见。
- 帮助后续轮次切换到更强模型或更安全插件。
- 同一反馈检测器可跨多条路由复用。

## 解决什么问题？ {#what-problem-does-it-solve}

后续轮次往往与首轮需要不同路由。若 Router 忽略用户反馈，用户表示失败后仍可能重复同一弱路径。

`user-feedback` 在路由图中直接暴露不满与纠正信号。

## 何时使用 {#when-to-use}

在以下情况使用 `user-feedback`：

- 后续纠正应升级到更强模型
- 负面反馈应触发更详尽或更安全的处理
- Router 对「答案错误」与「需要澄清」应不同反应
- 对话状态比原始领域更重要

## 配置 {#configuration}

```yaml
routing:
  signals:
    user_feedbacks:
      - name: wrong_answer
        description: User indicates the current answer is incorrect.
      - name: need_clarification
        description: User asks for a clearer or more detailed follow-up.
```

定义决策将消费的反馈标签，再由学习检测器决定每轮匹配哪一条。

## 依赖与限制 {#dependencies-and-limitations}

反馈检测器处理对话文本，可能把引用或假设中的抱怨误判为真实反馈。请在后续流量上评估它，并保留正常回退路径。

检测器不够确信、低于已配置 `threshold` 的预测，会报告为 `satisfied`，并附带模型对该类的自身概率。该数字常常远低于阈值，因为模型把概率质量放在了阈值拒绝的类上。把这一对读作不确定，而不是用户满意的证据。完整示例见：
[`config/fragments/signal/user-feedback/escalation.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/user-feedback/escalation.yaml)。
