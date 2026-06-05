---
translation:
  source_commit: "043cee97"
  source_file: "docs/tutorials/global/safety-models-and-policy.md"
  outdated: false
---

# 安全、模型与策略

## 概览

本页介绍 `global:` 内的共享运行时模型与策略块。

这些设置定义共享安全行为、共享运行时模型设置与全路由器策略默认。

## 主要优势

- 将共享策略与路由局部安全插件分离。
- 集中内置分类器与嵌入模型覆盖。
- 使 authz、限流与选择默认一致。
- 为覆盖系统模型绑定提供单一入口。

## 解决什么问题？

路由器依赖共享运行时模型与共享策略默认，且不绑定单条路由。若这些设置散落在路由中，行为难推理且难安全变更。

这些 `global:` 块将共享模型与策略覆盖集中在一层。

## 何时使用

在以下情况使用这些块：

- 内置安全与分类模型需要共享运行时设置
- 信号或算法层依赖共享嵌入或外部模型设置
- authz 或限流应对整台路由器生效
- 某系统能力应绑定到不同内部模型

## 配置

### Prompt Guard 与分类器

```yaml
global:
  model_catalog:
    modules:
      prompt_guard:
        model_ref: prompt_guard
        use_mmbert_32k: true
      classifier:
        domain:
          model_ref: domain_classifier
          threshold: 0.5
          use_mmbert_32k: true
```

### 嵌入与外部模型

```yaml
global:
  model_catalog:
    embeddings:
      semantic:
        mmbert_model_path: models/mom-embedding-ultra
        use_cpu: true
```

### Authz 与限流

```yaml
global:
  services:
    authz:
      enabled: true
    ratelimit:
      enabled: true
```

### 模型选择与 Looper 默认

```yaml
global:
  router:
    model_selection:
      enabled: true
  integrations:
    looper:
      enabled: true
```

### 系统模型

```yaml
global:
  model_catalog:
    system:
      prompt_guard: models/mmbert32k-jailbreak-detector-merged
      domain_classifier: models/mmbert32k-intent-classifier-merged
```
