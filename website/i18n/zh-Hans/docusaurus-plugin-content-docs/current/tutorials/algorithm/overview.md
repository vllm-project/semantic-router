---
translation:
  source_commit: "043cee97"
  source_file: "docs/tutorials/algorithm/overview.md"
  outdated: true
---

# 算法（Algorithm）

## 概览

最新算法教程与 `config/algorithm/` 下的片段目录一致。

算法仅在**决策已匹配**且 `modelRefs` 暴露多个候选模型之后才有意义；路由器随后用 `decision.algorithm` 选择或协调这些候选。

## 主要优势

- 将路由资格与模型选择策略分离。
- 一条决策可保留多个候选模型，而不内联排序逻辑。
- 支持单模型排序与多模型编排。
- 与仓库片段树一致：`config/algorithm/selection/` 与 `config/algorithm/looper/` 下每种算法一页教程。

## 解决什么问题？

路由匹配后仍需在候选模型间**有原则地**选择。没有算法层时，团队要么硬编码胜者，要么在路由间重复排序逻辑。

算法使匹配后的选择策略显式且可复用。

## 何时使用

在以下情况使用 `algorithm/`：

- `modelRefs` 含多个候选
- 路由策略依赖延迟、反馈、语义契合或在线探索
- 一条决策应编排多个模型而非只选一个
- 希望模型选择演进而不改决策规则本身

## 配置

在规范 v0.3 YAML 中，算法位于每条已匹配决策内：

```yaml
routing:
  decisions:
    - name: computer-science-reasoning
      rules:
        operator: AND
        conditions:
          - type: domain
            name: "computer science"
      modelRefs:
        - model: qwen2.5:7b
        - model: qwen3:14b
      algorithm:
        type: router_dc
        router_dc:
          temperature: 0.07
```

仓库为每种算法保留一页教程。

### 选择类算法

- [Automix](./selection/automix)
- [Elo](./selection/elo)
- [GMT Router](./selection/gmtrouter)
- [Hybrid](./selection/hybrid)
- [KMeans](./selection/kmeans)
- [KNN](./selection/knn)
- [Latency Aware](./selection/latency-aware)
- [RL Driven](./selection/rl-driven)
- [Router DC](./selection/router-dc)
- [Static](./selection/static)
- [SVM](./selection/svm)

### 循环类算法

- [Confidence](./looper/confidence)
- [Ratings](./looper/ratings)
- [ReMoM](./looper/remom)
