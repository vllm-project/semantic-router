---
translation:
  source_commit: "9156d5bc1ed9edff626b95a2b8260a77cb1712c5"
  source_file: "docs/tutorials/signal/learned/safety.md"
  outdated: false
---

# Safety 信号

`safety` 判断内容是否不安全；`jailbreak` 判断是否试图改变模型的指令执行。内容有害并不一定属于提示词攻击，两类信号应按应用策略组合。

## 配置

```yaml
routing:
  signals:
    safety:
      - name: unsafe_content
        threshold: 0.5
```

默认判断模型是 Vela 2.0 0.3B。显式任务绑定可选择其他兼容判断部署，或 Vela 1.0 Safety、Shield 等专用模型。可执行的任务能力不代表在应用流量上的准确率；切换模型后重新评估阈值、漏检与误报。

## 选择 Vela Shield {#select-vela-shield}

```yaml
global:
  model_catalog:
    system:
      safety: models/Vela-1.0-Encoder-307M-Shield
```

Safety 与 Shield 都输出 `safe`/`unsafe`，但独立训练，不应直接假设阈值可以通用。模型路径与绑定的完整配置见[安全模型指南](../../../model-runtime/guides/safety)。

## 风险类别

原生或组合判断任务可以识别风险类别；专用 Hazard 使用与其产物绑定的十二个标签阈值。内置 Hazard deployment 必须显式绑定，目录中存在它不会自动启用级联或加载模型。具体标签、operating point 与规则方式见[安全模型指南](../../../model-runtime/guides/safety)。

模型未读完整输入、未就绪或失败不能视为已证明安全。需要拒绝未知结果时，在决策中配置相应的 `rules.on_unknown`；按代表性数据验证完整策略。
