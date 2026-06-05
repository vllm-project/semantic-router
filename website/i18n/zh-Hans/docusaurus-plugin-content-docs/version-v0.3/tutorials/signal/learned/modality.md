---
translation:
  source_commit: "71b0e522"
  source_file: "docs/tutorials/signal/learned/modality.md"
  outdated: false
---

# Modality 信号

## 概览

`modality` 检测请求应停留在文本生成、切换到图像生成，或两者兼顾。映射到 `config/signal/modality/`，在 `routing.signals.modality` 中声明。

该族现归入「学习型」，因为典型部署依赖路由器自带的 `modality_detector` 模块分类输出模式，即使路由结果仍像简单请求形态决策。

## 主要优势

- 图像生成路由与纯文本路由分离。
- 多模态流量在 `routing.decisions` 中可见。
- 避免在每个决策规则里混入模态检查。
- 从简单形态路由扩展到检测器支持的多模态分类。

## 解决什么问题？

文本聊天、图像生成与混合工作流常共享入口，但不应共享同一路径。没有模态信号时，路由逻辑脆弱且重复。

`modality` 将输出模式暴露为命名路由输入。

## 何时使用

在以下情况使用 `modality`：

- 路由器同时服务自回归与扩散类后端
- 部分路由只接受图像生成提示
- 多模态处理必须在路由图中显式
- 需要稳定信号名如 `AR`、`DIFFUSION` 或 `BOTH`

## 配置

源片段族：`config/signal/modality/`

```yaml
routing:
  signals:
    modality:
      - name: AR
        description: Text-only autoregressive requests.
      - name: DIFFUSION
        description: Requests that should route into image generation flows.
      - name: BOTH
        description: Requests that need both text and image generation behavior.
```

规则名与期望决策引用的路由行为对齐。维护配置中，支撑检测器通常通过 `global.model_catalog.modules.modality_detector` 配置。
