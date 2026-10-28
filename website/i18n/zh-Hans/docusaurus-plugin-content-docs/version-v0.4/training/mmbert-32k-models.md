---
title: 训练 Vela Embedding 和 Reranker
sidebar_label: Embedding 和 Reranking
translation:
  source_commit: "915ddf56e0335e2046c38aa17c4aec6233908039"
  source_file: "docs/training/mmbert-32k-models.md"
  outdated: false
---

# 训练 Vela Embedding 和 Reranker {#train-vela-embedding-and-reranker}

使用 Vela Embedding 高效找到相关文档，再用 Vela Reranker 改善少量候选的排序。两者均从共享的 [Vela Encoder](https://huggingface.co/llm-semantic-router/Vela-1.0-Encoder-307M)适配而来，并支持选择编码器深度和输出维度。

直接使用已发布模型，请参阅[嵌入和重排序](../installation/runtime/embeddings.md)。

## Embedding：双编码器 {#embedding-model-bi-encoder}

Embedding 分别将查询和文档编码为归一化向量。预先计算文档向量，再用余弦相似度或点积与查询比较。

使用查询与正例文档，以及有效的负例训练。如果应用还需要比较请求、对相关文本分组或识别重复提问，可以加入语义相似度或释义样本。覆盖索引实际使用的语言和领域。

## Reranker：交叉编码器 {#reranking-model-cross-encoder}

Reranker 联合读取查询和候选文档，输出相关性分数。它用于检索系统已返回的候选。

训练时提供完整候选列表和相关性标注。从检索系统挖掘难负例，有助于区分看似相关但不正确的结果。未标注相关性的文档应与已确认负例分开。分数用于排序，不代表文档正确的概率。

## 选择深度、维度和上下文 {#choose-depth-dimension-and-context}

| 设置 | 可选值 | 权衡 |
| --- | --- | --- |
| 编码器深度 | 3、6、11、22 层 | 更少层数减少编码器计算 |
| 维度 | 64、128、256、512、768 | 更小的嵌入向量降低索引存储和比较成本 |
| 输入上限 | 最多 32,768 tokens | 更长输入需要更多内存和时间 |

训练会监督全部 20 种深度与维度组合。Reranker 为每种组合提供已训练的评分 head；降低维度不会跳过编码器层。评测时选择实际准备部署的组合。

Embedding 的查询和索引文档必须使用相同模型 revision、深度和维度。改变这些设置后重建索引。Reranker 的预算包含查询、文档和特殊 tokens 的总长度。

## 准备训练 {#prepare-a-training-run}

在仓库 checkout 中创建独立环境，安装适合加速器的 PyTorch 构建及[训练依赖](https://github.com/vllm-project/semantic-router/blob/main/src/training/model_embeddings/mmbert_32k/requirements.txt)。ROCm 使用 PyTorch 的 `cuda` 设备名称。

选择一种起点：

- **新任务：**下载 Vela Encoder，从完整 checkpoint 初始化任务。
- **继续任务：**下载 Vela Embedding 或 Reranker，设置 `initialization: "continued_task"`，保留编码器和 head，重新初始化优化器。
- **恢复中断：**使用 `--resume` 恢复优化器和训练进度。

分别准备训练集、开发集、采样计划和训练配置。[训练参考](https://github.com/vllm-project/semantic-router/tree/main/src/training/model_embeddings/mmbert_32k#train-a-new-task-from-a-standard-base)提供 JSON 格式、损失函数、可选教师监督和配置字段。

以下命令假定文件已准备在 `/data/retrieval`。将 `VELA_TRAIN_CONFIG_SHA256` 设置为配置文件的 SHA-256：

```bash
export PYTHONPATH="$PWD"

python -m src.training.model_embeddings.mmbert_32k.newbase_training \
  --config /data/retrieval/task.json \
  --config-sha256 "${VELA_TRAIN_CONFIG_SHA256:?Set the configuration SHA-256}" \
  --output /data/retrieval/run --device cuda
```

配置中选择 `task: "embedding"` 或 `task: "reranker"`，并设置数据、输入预算、训练步数和评测间隔。先用小预算检查首轮开发集结果，再延长训练。

## 评测 checkpoint {#evaluate-the-trained-checkpoint}

使用与基线相同的开发集评测已保存的 checkpoint：

```bash
python -m src.training.model_embeddings.mmbert_32k.newbase_scoring \
  --model /data/retrieval/run/step-100 --task embedding \
  --known-dev /data/retrieval/validation --split validation \
  --output /data/retrieval/evaluation --device cpu --token-budget 32768
```

把 checkpoint 路径替换成实际保存的步骤；Reranker 使用 `--task reranker`。先根据开发集选择 checkpoint，再在独立测试集上做最终对比。

Embedding 需要覆盖检索、相似度和多语言迁移。Reranker 使用固定候选列表，对比 nDCG 等排序指标。分别报告短输入、长输入，以及各部署深度与维度的延迟和内存。

完整 MMTEB 分数需要运行所选基准的全部任务，并使用一致评测协议。任务子集可以诊断短板，但不能得出总体分数或榜单排名。

## 部署结果 {#deploy-the-result}

通过[本地模型绑定](../installation/runtime/in-process.md)选择 checkpoint 和引擎，再通过[路由预览](../installation/runtime/lifecycle-diagnostics.md)检查深度、维度和输入预算。ONNX 部署需要从同一组训练权重导出的图。

## 早期 mmBERT 流程 {#earlier-mmbert-workflows}

原始基座、Embedding 和 Reranker 配方保留在[工作流 README](https://github.com/vllm-project/semantic-router/tree/main/src/training/model_embeddings/mmbert_32k)。历史配置用于早期 mmBERT 流程；新的 Vela 任务使用上面的 Vela 起始模型。
