---
translation:
  source_commit: "cdd0ca1"
  source_file: "docs/training/ml-model-selection.md"
  outdated: true
---

# 基于 ML 的模型选择

本文档介绍 Semantic Router 中基于机器学习的模型选择技术的配置方式和实验结果。

## 概述

基于 ML 的模型选择通过机器学习算法，根据查询特征和历史性能数据，将请求路由到最合适的 LLM。在基准测试中，ML 路由的质量分数比随机选择高出 13%-45%。

### 支持的算法

| 算法 | 描述 | 适用场景 |
|------|------|---------|
| **KNN**（K-Nearest Neighbors，K 近邻） | 基于相似查询的质量加权投票 | 高精度，多样化查询类型 |
| **KMeans** | 基于聚类的路由，优化效率 | 快速推理，均衡负载 |
| **SVM**（Support Vector Machine，支持向量机） | RBF 核决策边界 | 领域边界清晰的场景 |

### 参考论文

- [FusionFactory (arXiv:2507.10540)](https://arxiv.org/abs/2507.10540) — 基于 LLM 路由器的查询级融合
- [Avengers-Pro (arXiv:2508.12631)](https://arxiv.org/abs/2508.12631) — 性能-效率优化路由

## 配置

### 基础配置

在 `config.yaml` 中启用基于 ML 的模型选择：

```yaml
# 启用 ML 模型选择
model_selection:
  ml:
    enabled: true
    models_path: ".cache/ml-models"  # 训练好的模型文件路径

# 查询表示的嵌入模型
embedding_models:
  qwen3_model_path: "models/mom-embedding-pro"  # Qwen3-Embedding-0.6B
```

### 按决策类型配置算法

为不同的决策类型配置不同的算法：

```yaml
decisions:
  # 数学查询 - 使用 KNN 进行质量加权选择
  - name: "math_decision"
    description: "Mathematics and quantitative reasoning"
    priority: 100
    rules:
      operator: "AND"
      conditions:
        - type: "domain"
          name: "math"
    algorithm:
      type: "knn"
      knn:
        k: 5
    modelRefs:
      - model: "llama-3.2-1b"
      - model: "llama-3.2-3b"
      - model: "mistral-7b"

  # 编程查询 - 使用 SVM 实现清晰边界
  - name: "code_decision"
    description: "Programming and software development"
    priority: 100
    rules:
      operator: "AND"
      conditions:
        - type: "domain"
          name: "computer science"
    algorithm:
      type: "svm"
      svm:
        kernel: "rbf"
        gamma: 1.0
    modelRefs:
      - model: "codellama-7b"
      - model: "llama-3.2-3b"
      - model: "mistral-7b"

  # 通用查询 - 使用 KMeans 追求效率
  - name: "general_decision"
    description: "General knowledge queries"
    priority: 50
    rules:
      operator: "AND"
      conditions:
        - type: "domain"
          name: "other"
    algorithm:
      type: "kmeans"
      kmeans:
        num_clusters: 8
    modelRefs:
      - model: "llama-3.2-1b"
      - model: "llama-3.2-3b"
      - model: "mistral-7b"
```

### 算法参数

#### KNN 参数

```yaml
algorithm:
  type: "knn"
  knn:
    k: 5  # 邻居数量（默认：5）
```

#### KMeans 参数

```yaml
algorithm:
  type: "kmeans"
  kmeans:
    num_clusters: 8  # 聚类数量（默认：8）
```

#### SVM 参数

```yaml
algorithm:
  type: "svm"
  svm:
    kernel: "rbf"   # 核函数类型：rbf、linear（默认：rbf）
    gamma: 1.0      # RBF 核的 gamma 值（默认：1.0）
```

## 实验结果

### 基准测试设置

- **测试查询**：109 条跨多个领域的查询
- **评估模型**：4 个 LLM（codellama-7b、llama-3.2-1b、llama-3.2-3b、mistral-7b）
- **嵌入模型**：Qwen3-Embedding-0.6B（1024 维）
- **验证数据**：带有真实性能评分的基准查询

### 性能对比

| 策略 | 平均质量 | 平均延迟 | 最佳模型命中率 |
|------|---------|---------|--------------|
| **Oracle（理论最优）** | 0.495 | 10.57s | 100.0% |
| **KMEANS 选择** | 0.252 | 20.23s | 23.9% |
| 始终使用 llama-3.2-3b | 0.242 | 25.08s | 15.6% |
| **SVM 选择** | 0.233 | 25.83s | 14.7% |
| 始终使用 mistral-7b | 0.215 | 70.08s | 13.8% |
| 始终使用 llama-3.2-1b | 0.212 | 3.65s | 26.6% |
| **KNN 选择** | 0.196 | 36.62s | 13.8% |
| 随机选择 | 0.174 | 40.12s | 9.2% |
| 始终使用 codellama-7b | 0.161 | 53.78s | 4.6% |

### ML 路由相对随机选择的提升

| 算法 | 质量提升 | 最佳模型选择率 |
|------|---------|--------------|
| **KMEANS** | **+45.5%** | 提高 2.6 倍 |
| **SVM** | **+34.4%** | 提高 1.6 倍 |
| **KNN** | **+13.1%** | 提高 1.5 倍 |

### 关键发现

1. **所有 ML 方法均优于随机选择**，质量分数全面领先
2. **KMEANS 质量最优**，比随机选择提升 45%，延迟也可控
3. **SVM 性能均衡**，提升 34%，决策边界清晰
4. **KNN 模型选择多样化**，根据查询相似度调用不同模型

## 架构

```text
┌─────────────────────────────────────────────────────────────────────┐
│                         在线推理                                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  请求 (model="auto")                                                │
│       ↓                                                             │
│  生成查询 embedding (Qwen3, 1024 维)                                       │
│       ↓                                                             │
│  添加类别 One-Hot (14 维) → 1038 维特征向量                            │
│       ↓                                                             │
│  决策引擎 → 按领域匹配决策                                             │
│       ↓                                                             │
│  加载 ML 选择器 (从 JSON 加载 KNN/KMeans/SVM)                         │
│       ↓                                                             │
│  执行推理 → 选择最佳模型                                               │
│       ↓                                                             │
│  路由到选中的 LLM 端点                                                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## 训练自有模型

**离线训练与在线推理：**

- **离线训练**：使用 **Python** + scikit-learn 完成 KNN、KMeans 和 SVM 的训练
- **在线推理**：使用 **Rust** 中的 [Linfa](https://github.com/rust-ml/linfa)（通过 `ml-binding`）完成

训练阶段用 Python 和 scikit-learn，方便实验迭代。生产推理用 Rust 和 Linfa，保证低延迟。

### 前置条件

```bash
cd src/training/ml_model_selection
pip install -r requirements.txt
```

### 方式 1：下载预训练模型

```bash
python download_model.py \
  --output-dir ../../../.cache/ml-models \
  --repo-id abdallah1008/semantic-router-ml-models
```

### 方式 2：使用 HuggingFace 上的预基准测试数据进行训练

我们在 HuggingFace 上提供了可直接使用的基准测试数据：

**HuggingFace 数据集：** [abdallah1008/ml-selection-benchmark-data](https://huggingface.co/datasets/abdallah1008/ml-selection-benchmark-data)

| 文件 | 描述 |
|------|------|
| `benchmark_training_data.jsonl` | 4 个模型（codellama-7b、llama-3.2-1b、llama-3.2-3b、mistral-7b）的预基准测试数据 |
| `validation_benchmark_with_gt.jsonl` | 带真实值的验证数据，用于测试 |

```bash
# 下载基准测试数据
huggingface-cli download abdallah1008/ml-selection-benchmark-data \
  --repo-type dataset \
  --local-dir .cache/ml-models

# 使用预基准测试数据直接训练
python train.py \
  --data-file .cache/ml-models/benchmark_training_data.jsonl \
  --output-dir models/
```

这是最快的入门方式，不用自己跑 LLM 基准测试。

### 方式 3：使用自有数据训练

#### 步骤 1：准备输入数据（JSONL 格式）

创建一个包含查询的 JSONL 文件，每行必须包含 `query` 和 `category` 字段：

```jsonl
{"query": "What is the derivative of x^2?", "category": "math", "ground_truth": "2x"}
{"query": "Write a Python function to sort a list", "category": "computer science", "ground_truth": "def sort(lst): return sorted(lst)"}
{"query": "Explain photosynthesis", "category": "biology", "ground_truth": "Process where plants convert sunlight to energy"}
{"query": "What are the legal requirements for a contract?", "category": "law"}
```

**必填字段：**

| 字段 | 类型 | 描述 |
|------|------|------|
| `query` | string | 输入的查询文本 |
| `category` | string | 领域类别（参见 [VSR 类别](#vsr-categories)） |
| `ground_truth` | string | 期望的答案（用于计算性能/质量分数） |

**推荐字段（用于准确的性能评分）：**

| 字段 | 类型 | 描述 |
|------|------|------|
| `metric` | string | 评估方法 — 决定性能的计算方式 |
| `choices` | string | 用于多选题 — 触发多选题评估 |

**可选字段：**

| 字段 | 类型 | 描述 |
|------|------|------|
| `task_name` | string | 任务标识符，用于日志和追踪（如 "mmlu"、"gsm8k"） |

**关于 Metric 字段**

如果不指定 `metric`，基准测试默认使用 **CEM（条件精确匹配）**，这可能无法准确评分：

- 数学问题（使用 `metric: "GSM8K"` 或 `metric: "MATH"`）
- 多选题（使用 `metric: "em_mc"` 或包含 `choices`）
- 代码生成（使用 `metric: "code_eval"`）

为获得最佳结果，请始终为问题类型指定合适的 `metric`。

**多选题**

对于多选题，包含 `choices`（选项内容的字符串）并将 `ground_truth` 设为正确答案字母：

```jsonl
{"query": "What is the capital of France?\nA) London\nB) Paris\nC) Berlin\nD) Rome", "category": "other", "ground_truth": "B", "choices": "London,Paris,Berlin,Rome"}
```

基准测试脚本会：

1. 通过 `choices` 字段或 `metric: "em_mc"` 检测多选题
2. 从模型响应中提取答案字母（A/B/C/D）
3. 与 `ground_truth`（正确字母）比对

**评估指标**

`metric` 字段控制性能的计算方式：

| 指标 | 描述 | ground_truth 示例 |
|------|------|-------------------|
| `em_mc` | 多选题 — 提取字母 | `"B"` |
| `GSM8K` | 数学 — 提取 `####` 后的数字 | `"explanation #### 42"` |
| `MATH` | LaTeX 数学 — 从 `\boxed{}` 中提取 | `"\\boxed{2x+1}"` |
| `f1_score` | 文本重叠 F1 分数 | `"Paris is the capital"` |
| `code_eval` | 运行代码断言 | `"['assert func(1)==2']"` |
| （默认） | CEM — 包含匹配 | `"Paris"` |

**训练必须包含 Ground Truth**

训练 ML 模型选择必须有 `ground_truth` 字段。没有它，系统无法判断哪个模型在每条查询上表现更好。训练过程会将每个 LLM 的响应与 `ground_truth` 比对，算出性能分数。

#### 步骤 2：配置 LLM 端点（models.yaml）

创建 `models.yaml` 文件来配置 LLM 端点及认证信息：

```yaml
models:
  # 本地 Ollama 模型（无需认证）
  - name: llama-3.2-1b
    endpoint: http://localhost:11434/v1

  - name: llama-3.2-3b
    endpoint: http://localhost:11434/v1

  # OpenAI，使用环境变量中的 API 密钥
  - name: gpt-4
    endpoint: https://api.openai.com/v1
    api_key: ${OPENAI_API_KEY}
    max_tokens: 2048
    temperature: 0.0

  # HuggingFace，使用 Token
  - name: mistral-7b-hf
    endpoint: https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2
    api_key: ${HF_TOKEN}
    headers:
      Authorization: "Bearer ${HF_TOKEN}"

  # 自定义 API，使用 Bearer Token
  - name: custom-llm
    endpoint: https://api.custom.com/v1
    api_key: ${CUSTOM_API_KEY}
    headers:
      Authorization: "Bearer ${CUSTOM_API_KEY}"
      X-Custom-Header: "value"
    max_tokens: 1024
    temperature: 0.1

  # vLLM 自托管
  - name: codellama-7b
    endpoint: http://vllm-server:8000/v1
    # 本地 vLLM 无需认证
```

#### 步骤 3：运行基准测试

基准测试脚本会将每个查询发送到所有已配置的 LLM 并测量：

**性能（质量分数 0-1）：**

| 查询类型 | 评分方法 |
|---------|---------|
| **多选题**（A/B/C/D） | 选项与 `ground_truth` 精确匹配 |
| **数值/数学** | 解析并比较数字（基于容差） |
| **文本/代码** | 模型响应与 `ground_truth` 之间的 F1 分数 |
| **精确匹配** | 精确匹配为 1.0，否则为 0.0 |

**延迟（响应时间）：**

- 从请求发送到响应接收的时间（秒）
- 包含网络延迟 + 模型推理时间
- 用于效率加权：`speed_factor = 1 / (1 + latency)`

**输出格式：**

基准测试为每个（查询，模型）对生成一条 JSONL 记录：

```jsonl
{"query": "What is 2+2?", "category": "math", "model_name": "llama-3.2-1b", "response": "4", "ground_truth": "4", "performance": 1.0, "response_time": 0.523}
{"query": "What is 2+2?", "category": "math", "model_name": "llama-3.2-3b", "response": "The answer is 4", "ground_truth": "4", "performance": 0.85, "response_time": 1.234}
{"query": "What is 2+2?", "category": "math", "model_name": "mistral-7b", "response": "2+2=4", "ground_truth": "4", "performance": 0.92, "response_time": 2.156}
```

**运行基准测试：**

```bash
# 使用模型配置文件（推荐）
python benchmark.py \
  --queries your_queries.jsonl \
  --model-config models.yaml \
  --output benchmark_output.jsonl \
  --concurrency 4 \
  --limit 500  # 可选：限制查询数量用于测试

# 或使用简单模型列表（所有模型同一端点）
python benchmark.py \
  --queries your_queries.jsonl \
  --models llama-3.2-1b,llama-3.2-3b,mistral-7b \
  --endpoint http://localhost:11434/v1 \
  --output benchmark_output.jsonl
```

**benchmark.py 参数：**

| 参数 | 默认值 | 描述 |
|------|-------|------|
| `--queries` | （必填） | 输入 JSONL 文件路径 |
| `--model-config` | None | models.yaml 的路径 |
| `--models` | None | 逗号分隔的模型名称（替代 --model-config） |
| `--endpoint` | `http://localhost:8000/v1` | API 端点（配合 --models 使用） |
| `--output` | `benchmark_output.jsonl` | 输出文件路径 |
| `--concurrency` | `4` | 并行请求 LLM 的数量 |
| `--limit` | None | 限制处理的查询数量 |
| `--max-tokens` | `1024` | LLM 响应的最大 token 数 |
| `--temperature` | `0.0` | 生成温度（0.0 = 确定性输出） |

**并发参数**

`--concurrency` 参数控制并行发送到 LLM 的请求数：

- **较高值**（8-16）：基准测试更快，但可能压垮本地模型
- **较低值**（1-2）：较慢但对资源受限环境更安全
- **推荐值**：从 4 开始，如果 LLM 服务器能承受再增加

对于单 GPU 上的 Ollama，使用 `--concurrency 2-4`。对于云 API（OpenAI、HuggingFace），可以使用 `--concurrency 8-16`。

#### 步骤 4：训练 ML 模型

```bash
python train.py \
  --data-file benchmark_output.jsonl \
  --output-dir models/
```

### train.py 参数

| 参数 | 默认值 | 描述 |
|------|-------|------|
| `--data-file` | （必填） | JSONL 基准测试数据路径 |
| `--output-dir` | `models/` | 训练好的模型 JSON 文件保存目录 |
| `--embedding-model` | `qwen3` | 嵌入模型：`qwen3`、`gte`、`mpnet`、`e5`、`bge` |
| `--cache-dir` | `.cache/` | 嵌入缓存目录 |
| `--knn-k` | `5` | KNN 邻居数 |
| `--kmeans-clusters` | `8` | KMeans 聚类数 |
| `--svm-kernel` | `rbf` | SVM 核函数：`rbf`、`linear` |
| `--svm-gamma` | `1.0` | RBF 核的 gamma 值 |
| `--quality-weight` | `0.9` | 质量与速度权重（0=速度优先，1=质量优先） |
| `--batch-size` | `32` | 嵌入生成的批大小 |
| `--device` | `cpu` | 设备：`cpu`、`cuda`、`mps` |
| `--limit` | None | 限制训练样本数 |

**示例：**

```bash
# 使用自定义 KNN k 值训练
python train.py \
  --data-file benchmark.jsonl \
  --output-dir models/ \
  --knn-k 7

# 使用少量样本训练（用于测试）
python train.py \
  --data-file benchmark.jsonl \
  --output-dir models/ \
  --limit 1000

# 使用 GPU 加速训练
python train.py \
  --data-file benchmark.jsonl \
  --output-dir models/ \
  --device cuda \
  --batch-size 64

# 使用自定义算法参数训练
python train.py \
  --data-file benchmark.jsonl \
  --output-dir models/ \
  --knn-k 10 \
  --kmeans-clusters 12 \
  --svm-kernel rbf \
  --svm-gamma 0.5 \
  --quality-weight 0.85
```

### VSR 类别 {#vsr-categories}

系统支持 14 个领域类别，使用精确名称（带空格，不用下划线）：

```text
biology, business, chemistry, computer science, economics, engineering,
health, history, law, math, other, philosophy, physics, psychology
```

### 验证训练好的模型

运行 Go 验证脚本以确认 ML 路由的收益：

```bash
cd src/training/ml_model_selection

# 设置库路径（WSL/Linux）
export LD_LIBRARY_PATH=$PWD/../../../candle-binding/target/release:$PWD/../../../ml-binding/target/release:$LD_LIBRARY_PATH

# 运行验证
go run validate.go --qwen3-model /path/to/Qwen3-Embedding-0.6B
```

## 模型文件

训练好的模型以 JSON 文件存储：

| 文件 | 算法 | 大小 |
|------|------|------|
| `knn_model.json` | K 近邻 | ~2-10 MB |
| `kmeans_model.json` | KMeans 聚类 | ~50 KB |
| `svm_model.json` | 支持向量机 | ~1-5 MB |

这些文件从 HuggingFace 下载或在训练过程中生成：

- **模型**：[abdallah1008/semantic-router-ml-models](https://huggingface.co/abdallah1008/semantic-router-ml-models)
- **基准测试数据**：[abdallah1008/ml-selection-benchmark-data](https://huggingface.co/datasets/abdallah1008/ml-selection-benchmark-data)

## 最佳实践

### 算法选择指南

| 场景 | 推荐算法 | 原因 |
|------|---------|------|
| **质量优先任务** | KNN (k=5) | 质量加权投票能提供最高精度 |
| **高吞吐系统** | KMeans | 聚类查找速度快，延迟低 |
| **领域特定路由** | SVM | 领域之间的决策边界清晰 |
| **通用场景** | KMEANS | 质量和速度的最佳平衡 |

### 超参数调优

1. **KNN k 值**：从 k=5 开始，增大可使决策更平滑
2. **KMeans 聚类数**：匹配不同查询模式的数量（通常 8-16）
3. **SVM gamma**：对归一化嵌入使用 1.0，根据数据分布调整

### 特征向量组成

ML 模型使用 1038 维特征向量：

- **1024 维**：Qwen3 语义 embedding
- **14 维**：类别 One-Hot 编码（VSR 领域类别）

```text
特征向量 = [embedding(1024)] ⊕ [category_one_hot(14)]
```

## 故障排查

### 模型加载失败

```text
Error: pretrained model not found
```

从 HuggingFace 下载模型：

```bash
cd src/training/ml_model_selection
python download_model.py --output-dir ../../../.cache/ml-models
```

### 选择精度低

1. 确保嵌入模型与训练时一致（Qwen3-Embedding-0.6B）
2. 检查类别分类是否正常工作
3. 确认配置中的模型名称与训练数据一致

### 维度不匹配

```text
Error: embedding dimension mismatch
```

确保训练和推理使用相同的嵌入模型（Qwen3 输出 1024 维）。

## 后续步骤

- [训练概览](/docs/training/training-overview) — 通用训练文档
- [模型性能评估](/docs/training/model-performance-eval) — 详细性能指标
