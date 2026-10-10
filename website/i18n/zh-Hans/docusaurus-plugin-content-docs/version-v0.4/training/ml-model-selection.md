---
title: 基于 ML 的模型选择
sidebar_label: ML 模型选择
translation:
  source_commit: "e56591a9cb24f073bf159927e87116ba6d278741"
  source_file: "docs/training/ml-model-selection.md"
  outdated: false
---

# 基于 ML 的模型选择 {#ml-based-model-selection}

基于 ML 的选择学习决策候选池中哪个模型最适合某个请求。当历史评测数据比固定优先级顺序或少量手写规则携带更多信息时，它很有用。

选择器在路由决策匹配之后运行。它只能从该决策的 `modelRefs` 中选择；它不会发现、部署或认证 provider 模型。

## 可用选择器 {#available-selectors}

| 选择器 | 如何选择 | 在以下情况考虑 |
|----------|----------------|------------------|
| [KNN](/zh-Hans/docs/tutorials/algorithm/selection/knn) | 在相似查询上组合记录的质量和速度 | 相似请求往往偏好同一模型，且可追溯性很重要 |
| [KMeans](/zh-Hans/docs/tutorials/algorithm/selection/kmeans) | 将请求映射到已学习的簇 | 工作负载形成稳定簇，且查找成本很重要 |
| [SVM](/zh-Hans/docs/tutorials/algorithm/selection/svm) | 使用已学习的决策边界 | 候选模型在特征空间中干净分离 |
| [MLP](/zh-Hans/docs/tutorials/algorithm/selection/mlp) | 用神经网络为候选打分 | 你有足够数据训练非线性选择器，并能运维其运行时依赖 |

没有普遍最好的选择器。将每个候选与简单基线比较，例如固定默认、随机选择，以及同一留出数据集上的最佳单模型。

## 训练前准备 {#before-you-train}

你需要：

- 两个或更多 OpenAI 兼容模型端点
- 代表性查询和标准答案，或其他可辩护的评分方法
- 足够的重复覆盖，以便在重要工作负载切片上评测每个候选模型
- 训练和在线推理时相同的嵌入模型
- 密钥、速率限制、成本以及模型响应保留计划

基准测试将每个所选查询发送到多个 provider 端点，并存储其响应、质量分数和延迟。当提示词或响应包含用户数据时，将输出视为敏感。

## 控制面板工作流 {#dashboard-workflow}

在控制面板中打开 `/ml-setup` 以运行引导工作流：

1. 上传模型端点 YAML 文件和查询 JSONL 文件。
2. 对候选模型做基准测试。
3. 训练一个或多个选择器。
4. 定义决策并下载配置片段。

Benchmark 和 Train 步骤会在控制面板的 ML 数据目录下产生数据和模型产物。

:::caution 当前配置导出

生成的 `ml-model-selection-values.yaml` 是迁移片段，不是独立的规范 Router 配置。其 `config.model_selection`、`config.strategy` 和 `config.decisions` 字段必须经过评审，并映射到完整配置中的 `global.router.model_selection`、`global.router.strategy` 和 `routing.decisions`。添加所需的 listeners 和 providers，然后在部署前运行 `vllm-sr config validate --config ...`。

:::

## 命令行工作流 {#command-line-workflow}

### 1. 安装训练依赖 {#1-install-the-training-dependencies}

```bash
cd src/training/model_selection/ml_model_selection
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### 2. 准备评测查询 {#2-prepare-evaluation-queries}

每行使用一个 JSON 对象。`ground_truth` 是为模型输出打分所必需的；`category` 可选，但对切片评测有用。

```jsonl
{"query":"What is the derivative of x^2?","ground_truth":"2x","category":"math","metric":"MATH"}
{"query":"Which city is the capital of France?","ground_truth":"B","category":"other","metric":"em_mc","choices":"A) London B) Paris C) Berlin D) Rome"}
```

选择与任务匹配的指标。基准支持精确/包含匹配、选择题提取、GSM8K 和 MATH 答案提取、文本 F1 以及代码评测。检查基准输出，而不是假设一个指标适合每个领域。

### 3. 描述候选端点 {#3-describe-the-candidate-endpoints}

将凭据放在环境变量中。不要把字面 API 密钥写进 YAML 文件。

```yaml
models:
  - name: local-small
    endpoint: http://localhost:8000/v1
  - name: hosted-model
    endpoint: https://provider.example/v1
    api_key: ${PROVIDER_API_KEY}
```

### 4. 对每个候选做基准测试 {#4-benchmark-every-candidate}

```bash
python benchmark.py \
  --queries queries.jsonl \
  --model-config models.yaml \
  --output benchmark-output.jsonl \
  --concurrency 4
```

从低并发开始。仅在确认每个端点都能维持请求速率，且 provider 速率限制没有扭曲延迟测量后，再提高并发。

### 5. 训练选择器 {#5-train-selectors}

```bash
python train.py \
  --data-file benchmark-output.jsonl \
  --output-dir models
```

默认情况下脚本训练 KNN、KMeans、SVM 和 MLP 产物。使用 `--algorithm knn|kmeans|svm|mlp` 训练单个选择器，或在 PyTorch 依赖不可用时使用 `--skip-mlp`。训练接受 `cpu`、`cuda` 或 `mps`。当前 Router 决策工厂在 CPU 上运行已加载的 MLP 产物；其 `device` 字段为兼容性接受，但未接到选择上。

输出目录包含 `knn_model.json`、`kmeans_model.json`、`svm_model.json` 和 `mlp_model.json` 等 JSON 产物。其内容特定于被基准测试的模型名、嵌入模型和特征布局。

### 6. 配置 Router {#6-configure-the-router}

将选择器设置合并进完整规范配置。此示例是片段；完整文件仍需要 listeners、providers 以及决策使用的任何信号。

```yaml
global:
  router:
    model_selection:
      ml:
        models_path: /models/selection
        model_type: qwen3
        embedding_dim: 1024
        knn:
          k: 5
          pretrained_path: /models/selection/knn_model.json

routing:
  decisions:
    - name: math
      description: Route math requests with the trained KNN selector.
      priority: 100
      rules:
        operator: AND
        conditions:
          - type: domain
            name: math
      algorithm:
        type: knn
      modelRefs:
        - model: local-small
        - model: hosted-model
```

配置的 `model_type` 和 `embedding_dim` 选择 ML 选择器使用的嵌入特征空间，必须与训练产物匹配。其他选择算法和嵌入消费者继续使用默认语义嵌入配置。若省略 `model_type`，ML 选择器也为向后兼容使用该默认值。`modelRefs` 中的模型名必须与基准数据中记录的名称以及已配置的 provider 别名匹配。

```bash
vllm-sr config validate --config config.yaml
```

## 上线前评测 {#evaluate-before-rollout}

使用未用于拟合或调优选择器的留出数据集。报告：

- 按工作负载切片的回答质量指标
- 所选模型分布
- 端到端延迟和 provider 成本
- 相对选择最佳已评测响应的 oracle 的 regret
- 与固定默认、最佳单模型和随机基线的比较
- 失败、超时和排除的样本

将数据集修订、源提交、模型修订、嵌入模型、硬件、命令和原始报告与任何头条结果一起发布。没有这些来源的百分比或 QPS 不能描述预期生产性能。

## 常见问题 {#common-problems}

### 找不到产物 {#artifact-not-found}

从 Router 运行时内部检查 `models_path` 和每个 `pretrained_path`，而不仅仅在主机上。按配置使用的同一路径挂载或打包文件。

### 嵌入维度不匹配 {#embedding-dimension-mismatch}

训练和推理使用同一嵌入模型和特征布局，并将 `embedding_dim` 设为导出产物的维度。

### 留出质量差 {#poor-held-out-quality}

确认模型名和标签匹配，检查类别和领域覆盖，查找训练/测试泄漏，并与简单基线比较。更复杂的选择器无法补偿不具代表性的基准数据。

## 参考 {#references}

- [模型训练概览](./training-overview)
- [当前模型目录](./model-catalog)
- [模型性能评测](./model-performance-eval)
- [训练源和完整 CLI 选项](https://github.com/vllm-project/semantic-router/tree/main/src/training/model_selection/ml_model_selection)
- [FusionFactory (arXiv:2507.10540)](https://arxiv.org/abs/2507.10540)
- [Avengers-Pro (arXiv:2508.12631)](https://arxiv.org/abs/2508.12631)
