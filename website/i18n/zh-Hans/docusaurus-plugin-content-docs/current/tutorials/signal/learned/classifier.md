---
translation:
  source_commit: "9716b569"
  source_file: "docs/tutorials/signal/learned/classifier.md"
  outdated: false
---

# 分类器信号

## 概览

`classifier` 公开来自本地原生序列分类器或已配置外部 LLM 的可复用标签分数。决策使用必需的数值谓词测试已声明的标签。

对于各自的领域，专用的领域、PII、越狱、事实核查、KB 和偏好信号仍是首选接口。

## 主要优势

- 无需添加领域逻辑即可集成任意序列分类头
- 将 LLM 分类器限制为使用已声明的标签和确定性 JSON 输出
- 计算一次标签映射，供多个决策使用不同分数进行门控

## 解决什么问题？

有些训练完成的分类器不属于内置信号分类体系。分类器信号提供一个狭窄的标签-分数接口，同时维持事实提取与决策控制逻辑之间的分离。

## 何时使用

将此信号用于真正可复用的分类头或通过提示驱动的 LLM 标签器。对于参考短语相似度，优先使用嵌入/KB 信号；对于响应风格路由，优先使用偏好信号。

## 配置

```yaml
routing:
  signals:
    classifiers:
      - name: phishing
        type: local
        model_path: models/phishing-email
        labels: [BENIGN, PHISHING]
        use_cpu: true

  decisions:
    - name: phishing-local
      rules:
        operator: AND
        conditions:
          - type: classifier
            name: phishing
            label: PHISHING
            predicate:
              gte: 0.5
            on_error: no_match
      modelRefs:
        - model: local-small
          use_reasoning: false
```

LLM 分类器引用一个命名的 `global.model_catalog.external` 条目，并添加 `instructions`。运行时固定温度、输出 schema、token 上限和精确标签验证。分类器叶节点是唯一接受 `on_error` 的决策谓词；失败会在 eval/replay 诊断中公开有明确上限的 `classifier_evaluation_failed` 代码。

本地分类器使用 `model_path`；`models/` 下的路径会参与常规模型注册表/下载流程。每个进程仅支持一个进程全局的二分类本地通用分类器，其决策谓词针对胜出标签的置信度使用 `gte: 0.5` 或更高阈值。更改其模型或标签顺序需要重启路由器，以免正在进行的配置重载替换进程全局的原生状态。尝试进行此类变更的管理 API 更新会返回 `RESTART_REQUIRED`，而不是应用不完整的运行时快照。
