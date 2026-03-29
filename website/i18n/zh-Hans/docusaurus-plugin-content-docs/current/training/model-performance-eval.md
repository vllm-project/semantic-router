---
translation:
  source_commit: "485c74ba"
  source_file: "docs/training/model-performance-eval.md"
  outdated: false
---

# 模型性能评估

## 为什么要做评估？

评估让路由决策有据可依。通过在 MMLU-Pro 上衡量各类别准确率（并用 ARC 做快速 sanity check），你可以：

- 为每个决策选择合适的模型，并在 `decisions.modelRefs` 中配置
- 根据整体表现选择合理的 `default_model`
- 判断 CoT 提示是否值得付出延迟/成本
- 在模型、提示词或参数变更时捕获回退
- 让 CI 与发布流程可复现、可审计

简而言之，评估把主观感受变成可度量信号，从而提升路由器的质量、成本效益与可靠性。

---

本文说明如何通过兼容 vLLM 的 OpenAI 端点自动评估模型（MMLU-Pro 与 ARC Challenge）、生成基于性能的路由配置，并更新配置中的 `categories.model_scores`。

相关代码见 [/src/training/model_eval](https://github.com/vllm-project/semantic-router/tree/main/src/training/model_eval)。

### 端到端流程概览

#### 1）评估模型

- 各类别准确率
- ARC Challenge：总体准确率

#### 2）可视化结果

- 各类别准确率的柱状图/热力图

![Bar](/img/bar.png)
![Heatmap](/img/heatmap.png)

#### 3）生成更新后的 config.yaml

- 为每个类别创建带 `modelRefs` 的决策
- 将 `default_model` 设为平均表现最佳者
- 保留或应用决策级推理相关设置

## 1. 前置条件

- 已启动、兼容 vLLM 的 OpenAI 端点，正在提供你的模型
  - 端点 URL 形如 http://localhost:8000/v1
  - 若端点需要 API Key，请按需配置

  ```bash
  # 终端 1
  vllm serve microsoft/phi-4 --port 11434 --served_model_name phi4

  # 终端 2
  vllm serve Qwen/Qwen3-0.6B --port 11435 --served_model_name qwen3-0.6B
  ```

- 评估脚本依赖的 Python 包：
  - 仓库根目录的 matplotlib：见 [requirements.txt](https://github.com/vllm-project/semantic-router/blob/main/requirements.txt)
  - `/src/training/model_eval` 下的依赖：见 [requirements.txt](https://github.com/vllm-project/semantic-router/blob/main/src/training/model_eval/requirements.txt)

  ```bash
  # 本指南后续均在此目录操作
  cd /src/training/model_eval
  pip install -r requirements.txt
  ```

**重要配置要求：**

vLLM 命令中的 `--served-model-name` **必须与** `config/config.yaml` 里的模型名称**完全一致**：

```yaml
# config/config.yaml 必须与上述 --served-model-name 一致
providers:
  models:
    - name: "phi4"            # 与 --served_model_name phi4 一致
      provider_model_id: "phi4"
      backend_refs:
        - name: "endpoint1"
          endpoint: "127.0.0.1:11434"
          protocol: "http"
    - name: "qwen3-0.6B"      # 与 --served_model_name qwen3-0.6B 一致
      provider_model_id: "qwen3-0.6B"
      backend_refs:
        - name: "endpoint2"
          endpoint: "127.0.0.1:11435"
          protocol: "http"
  defaults:
    default_model: "phi4"

routing:
  modelCards:
    - name: "phi4"
    - name: "qwen3-0.6B"
```

**可选提示：**

- 若计划直接使用生成后的配置，请确保 `config/config.yaml` 在 `providers.models[]` 中包含已部署模型名，并在 `routing.modelCards` 中有对应的语义目录。

## 2. 在 MMLU-Pro 上评估

脚本见 [mmul_pro_vllm_eval.py](https://github.com/vllm-project/semantic-router/blob/main/src/training/model_eval/mmlu_pro_vllm_eval.py)。

### 用法示例

```bash
# 评估少量模型、每类少量样本、直接提示
python mmlu_pro_vllm_eval.py \
  --endpoint http://localhost:11434/v1 \
  --models phi4 \
  --samples-per-category 10

python mmlu_pro_vllm_eval.py \
  --endpoint http://localhost:11435/v1 \
  --models qwen3-0.6B \
  --samples-per-category 10

# 使用 CoT 评估（结果保存在 *_cot 目录下）
python mmlu_pro_vllm_eval.py \
  --endpoint http://localhost:11435/v1 \
  --models qwen3-0.6B \
  --samples-per-category 10
  --use-cot 

# 若已正确配置 Semantic Router，可一次性评估
python mmlu_pro_vllm_eval.py \
  --endpoint http://localhost:8801/v1 \
  --models qwen3-0.6B, phi4 \
  --samples-per-category
  # --use-cot # 若使用 CoT 请取消本行注释
```

### 主要参数

- **--endpoint**：vLLM OpenAI URL（默认 http://localhost:8000/v1）
- **--models**：空格分隔列表，或单个逗号分隔字符串；若省略，脚本会向端点的 /models 查询
- **--categories**：仅评估指定类别；若省略则使用数据集中全部类别
- **--samples-per-category**：每类题目数量上限（用于快速试跑）
- **--use-cot**：启用思维链（Chain-of-Thought）变体；结果保存在不同子目录后缀（_cot 与 _direct）
- **--concurrent-requests**：并发请求数以提升吞吐
- **--output-dir**：结果保存目录（默认 results）
- **--max-tokens**、**--temperature**、**--seed**：生成与可复现性相关参数

### 每个模型的输出

- **results/Model_Name_(direct|cot)/**
  - **detailed_results.csv**：每题一行，含 is_correct 与 category
  - **analysis.json**：overall_accuracy、category_accuracy 映射、avg_response_time、计数等
  - **summary.json**：精简指标
- **mmlu_pro_vllm_eval.txt**：提示与回答日志（调试/检查）

**说明**

- **模型命名**：文件夹名会将斜杠替换为下划线；例如 gemma3:27b → gemma3:27b_direct 目录。
- 类别准确率仅统计成功请求；失败请求不计入。

## 3. 在 ARC Challenge 上评估（可选，总体 sanity check）

脚本见 [arc_challenge_vllm_eval.py](https://github.com/vllm-project/semantic-router/blob/main/src/training/model_eval/arc_challenge_vllm_eval.py)。

### 用法示例

``` bash
python arc_challenge_vllm_eval.py \
  --endpoint http://localhost:8801/v1\
  --models qwen3-0.6B,phi4
  --output-dir arc_results
```

### 主要参数

- **--samples**：抽样题目总数（默认 20）；本脚本中 ARC 不按类别划分
- 其余参数与 **MMLU-Pro** 脚本一致

### 每个模型的输出

- **results/Model_Name_(direct|cot)/**
  - **detailed_results.csv**：每题一行，含 is_correct 与 category
  - **analysis.json**：overall_accuracy、avg_response_time
  - **summary.json**：精简指标
- **arc_challenge_vllm_eval.txt**：提示与回答日志（调试/检查）

**说明**

- ARC 结果不会直接写入 `categories[].model_scores`，但有助于发现回退。

## 4. 可视化各类别表现

脚本见 [plot_category_accuracies.py](https://github.com/vllm-project/semantic-router/blob/main/src/training/model_eval/plot_category_accuracies.py)。

### 用法示例

```bash
# 使用 results/ 生成柱状图
python plot_category_accuracies.py \
  --results-dir results \
  --plot-type bar \
  --output-file results/bar.png

# 使用 results/ 生成热力图
python plot_category_accuracies.py \
  --results-dir results \
  --plot-type heatmap \
  --output-file results/heatmap.png

# 使用示例数据生成示例图
python src/training/model_eval/plot_category_accuracies.py \
  --sample-data \
  --plot-type heatmap \
  --output-file results/category_accuracies.png
```

### 主要参数

- **--results-dir**：`analysis.json` 所在目录
- **--plot-type**：bar 或 heatmap
- **--output-file**：输出图片路径（默认 model_eval/category_accuracies.png）
- **--sample-data**：若无真实结果，生成假数据用于预览图表

### 行为说明

- 查找所有 `results/**/analysis.json`，按模型聚合 `analysis["category_accuracy"]`
- 增加 Overall 列，表示跨类别平均
- 生成图表以便快速对比模型/类别表现

**说明**

- `direct` 与 `cot` 作为不同变体，标签会追加 `:direct` 或 `:cot`；图例为简洁会隐藏 `:direct`。

## 5. 生成基于性能的路由配置

脚本见 [result_to_config.py](https://github.com/vllm-project/semantic-router/blob/main/src/training/model_eval/result_to_config.py)。

### 用法示例

```bash
# 使用 results/ 生成新配置文件（不覆盖已有文件）
python src/training/model_eval/result_to_config.py \
  --results-dir results \
  --output-file config/config.eval.yaml

# 修改语义缓存相似度阈值
python src/training/model_eval/result_to_config.py \
  --results-dir results \
  --output-file config/config.eval.yaml \
  --similarity-threshold 0.85

# 从指定子目录生成
python src/training/model_eval/result_to_config.py \
  --results-dir results/mmlu_run_2025_09_10 \
  --output-file config/config.eval.yaml
```

### 主要参数

- **--results-dir**：包含 `analysis.json` 的目录
- **--output-file**：目标配置路径（默认 `config/config.eval.yaml`）
- **--similarity-threshold**：写入生成配置中的语义缓存阈值
- **--backend-endpoint**：生成 `providers.models[].backend_refs[]` 时使用的端点
- **--backend-protocol**：生成 `backend_refs` 时使用的协议
- **--backend-type**：生成 `backend_refs` 时使用的后端类型
- **--api-format**：生成模型时使用的 `providers.models[].api_format`
- **--provider-name**：用于 `providers.models[].external_model_ids` 的键名

### 行为说明

- 读取全部 `analysis.json`，提取 `analysis["category_accuracy"]`
- 将 `direct` 与 `cot` 变体合并为每个基础模型一条逻辑目录项
- 构造 canonical v0.3 脚手架：
  - **providers.defaults.default_model**：跨类别平均表现最佳者
  - **providers.models[]**：每个已评估逻辑模型的部署绑定
  - **routing.modelCards[]**：路由决策使用的逻辑模型目录
  - **routing.signals.domains[]**：每个 MMLU-Pro 类别一条域信号，含排序后的 `model_scores`
  - **routing.decisions**：留空，便于你单独组合路由策略
  - **global**：语义缓存、工具、BERT 嵌入、提示防护与分类器等模块的稀疏覆盖
- 若存在特殊 `auto` 占位模型则省略

### Schema 对齐要点

- **providers.defaults.default_model**：跨类别平均最佳
- **providers.models[]**：每个已评估逻辑模型一条生成后端绑定
- **routing.modelCards[]**：每个已评估基础模型一条逻辑项
- **routing.signals.domains[].name**：MMLU-Pro 类别字符串
- **routing.signals.domains[].model_scores**：带 score 与 `use_reasoning` 的排序模型列表
- **global**：稀疏运行时覆盖，非路由器默认的完整拷贝

**说明**

- 本脚本仅适用于 **MMLU_Pro** 评估结果。
- 默认输出为 `config/config.eval.yaml`，避免覆盖仓库中的完整参考文件 `config/config.yaml`。
- 生成文件为 canonical 脚手架，上线前仍需检查 `listeners`、`providers.models[].backend_refs[]` 及各类运行时覆盖。
- 若生产配置包含**环境相关项（多端点权重、密钥、定价、策略等）**，请将生成的 `providers`/`routing`/`global` 段落合并进对应部署配置，而非整体替换。

### config.eval.yaml 示例

更多配置说明见 [configuration](https://vllm-semantic-router.com/docs/installation/configuration)。

```yaml
version: v0.3
listeners: []

providers:
  defaults:
    default_reasoning_effort: medium
    default_model: phi4
  models:
    - name: phi4
      provider_model_id: phi4
      api_format: openai
      external_model_ids:
        openai: phi4
      backend_refs:
        - name: phi4-backend
          endpoint: 127.0.0.1:11434
          protocol: http
          type: chat
          weight: 1
    - name: qwen3-0.6B
      provider_model_id: qwen3-0.6B
      api_format: openai
      external_model_ids:
        openai: qwen3-0.6B
      backend_refs:
        - name: qwen3-0.6B-backend
          endpoint: 127.0.0.1:11435
          protocol: http
          type: chat
          weight: 1

routing:
  modelCards:
    - name: phi4
      description: Generated from MMLU-Pro evaluation results for category-aware routing.
      quality_score: 0.81
      capabilities: [chat]
      tags: [generated, mmlu-pro]
      modality: ar
    - name: qwen3-0.6B
      description: Generated from MMLU-Pro evaluation results for category-aware routing.
      quality_score: 0.77
      capabilities: [chat]
      tags: [generated, mmlu-pro]
      modality: ar
  signals:
    domains:
      - name: business
        description: MMLU-Pro category generated from evaluation results.
        mmlu_categories: [business]
        model_scores:
          - model: phi4
            score: 0.88
            use_reasoning: false
          - model: qwen3-0.6B
            score: 0.75
            use_reasoning: false
      - name: law
        description: MMLU-Pro category generated from evaluation results.
        mmlu_categories: [law]
        model_scores:
          - model: phi4
            score: 0.84
            use_reasoning: false
          - model: qwen3-0.6B
            score: 0.73
            use_reasoning: false
  decisions: []

global:
  stores:
    semantic_cache:
      enabled: true
      similarity_threshold: 0.85
      max_entries: 1000
      ttl_seconds: 3600
  integrations:
    tools:
      enabled: true
      top_k: 3
      similarity_threshold: 0.2
      tools_db_path: deploy/examples/runtime/tools/tools_db.json
      fallback_to_empty: true
  model_catalog:
    embeddings:
      semantic:
        mmbert_model_path: models/mom-embedding-ultra
        use_cpu: true
        embedding_config:
          model_type: mmbert
          preload_embeddings: true
          target_dimension: 768
          target_layer: 22
          top_k: 1
          min_score_threshold: 0.5
    modules:
      prompt_guard:
        enabled: true
        model_id: models/mmbert32k-jailbreak-detector-merged
        threshold: 0.7
        use_cpu: true
        use_mmbert_32k: true
        jailbreak_mapping_path: models/mmbert32k-jailbreak-detector-merged/jailbreak_type_mapping.json
      classifier:
        domain:
          model_id: models/mmbert32k-intent-classifier-merged
          threshold: 0.5
          use_cpu: true
          use_mmbert_32k: true
          category_mapping_path: models/mmbert32k-intent-classifier-merged/category_mapping.json
          fallback_category: other
        pii:
          model_id: models/mmbert32k-pii-detector-merged
          threshold: 0.9
          use_cpu: true
          use_mmbert_32k: true
          pii_mapping_path: models/mmbert32k-pii-detector-merged/pii_type_mapping.json
```

该输出便于 diff 与合并：

- `providers.models[]` 携带端点/协议绑定，便于提升为可运行配置。
- `routing.modelCards[]` 与 `routing.signals.domains[]` 承载评估得到的路由语义。
- `routing.decisions` 为空，因为评估步骤无法推断完整生产策略。
