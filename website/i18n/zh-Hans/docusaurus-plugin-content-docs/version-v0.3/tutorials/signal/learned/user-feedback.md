---
translation:
  source_commit: "043cee97"
  source_file: "docs/tutorials/signal/learned/user-feedback.md"
  outdated: false
---

# User Feedback 信号

## 概览

`user-feedback` 从对话中检测纠正、不满或升级反馈。映射到 `config/signal/user-feedback/`，在 `routing.signals.user_feedbacks` 中声明。

该族为学习型：依赖 `global.model_catalog.modules.feedback_detector` 配置的反馈检测器。

## 主要优势

- 用户表示答案错误或不清时路由器可作出反应。
- 升级行为在路由决策内可见。
- 帮助后续轮次切换到更强模型或更安全插件。
- 同一反馈检测器可跨多条路由复用。

## 解决什么问题？

后续轮次往往与首轮需要不同路由。若忽略用户反馈，用户表示失败后仍可能重复弱路径。

`user-feedback` 在路由图中直接暴露不满与纠正信号。

## 何时使用

在以下情况使用 `user-feedback`：

- 后续纠正应升级到更强模型
- 负面反馈应触发更详尽或更安全的处理
- 路由器对「答案错误」与「需要澄清」应不同反应
- 对话状态比原始领域更重要

## 配置

源片段族：`config/signal/user-feedback/`

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
