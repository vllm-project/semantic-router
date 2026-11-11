---
is_mtpe: false
title: 训练 Vela Embedding
sidebar_label: Embedding
translation:
  source_commit: "118fb72eb67a0e7193697e350d7b494dc016631e"
  source_file: "docs/training/mmbert-32k-models.md"
  outdated: false
---

# 训练 Vela Embedding {#train-vela-embedding}

Vela Embedding 为 Router 的召回功能查找候选：响应缓存、记忆和 RAG。它基于共享的 [Vela Encoder](https://huggingface.co/vllm-sr/Vela-1.0-Encoder-307M) 适配而来，支持选择编码器深度和输出维度。嵌入只用于召回；召回找到的内容由决策模型评判。

不经训练直接使用已发布的模型，请参阅[嵌入](../model-runtime/guides/embeddings.md)。

## Embedding 模型：双编码器 {#embedding-model-bi-encoder}

Embedding 模型把查询和文档分别编码为归一化向量。预先计算文档向量，然后用余弦相似度或点积把查询向量与你的索引比较。

用查询-正例对和有用的负例文档训练。如果你的应用还要比较请求、对相关文本分组或识别重复问题，再加入语义相似度或改写示例。包括你的索引要服务的语言和领域。

## 选择深度、维度和上下文 {#choose-depth-dimension-and-context}

| 设置 | 可选值 | 取舍 |
| --- | --- | --- |
| 编码器深度 | 3、6、11、22 层 | 层数越少，编码器计算越少 |
| 维度 | 64、128、256、512、768 | 向量越小，索引存储和比较成本越低 |
| 输入上限 | 最多 32,768 个 token | 输入越长，需要的内存和时间越多 |

训练会监督全部 20 种深度/维度组合。请评测你打算部署的组合。查询和已索引文档要使用相同的模型 revision、深度和维度，这些设置改变时要重建索引。

## 准备训练 {#prepare-a-training-run}

在仓库检出目录中，为这个训练器创建锁定的 uv 环境：

```bash
uv sync --locked --project src/training/model_embeddings/mmbert_32k
```

在 Linux 上，默认的 PyPI torch wheel 是 CUDA 构建，在 macOS 上是平台默认构建。发布运行仍使用训练器旁记录的、按摘要固定的镜像。ROCm 使用 PyTorch 的 `cuda` 设备名。[requirements.txt](https://github.com/vllm-project/semantic-router/blob/main/src/training/model_embeddings/mmbert_32k/requirements.txt) 供使用 pip 安装的用户保留。

选择一个起点：

- **新任务**：下载 Vela Encoder，并从这个完整 checkpoint 初始化任务。
- **继续一个任务**：下载 Vela Embedding 并选择 `initialization: "continued_task"`。这会保留训练好的编码器，同时从新的优化器开始。
- **恢复中断的运行**：用 `--resume` 恢复它的优化器和进度。

准备独立的训练和开发语料、采样计划和训练配置。[训练流程参考](https://github.com/vllm-project/semantic-router/tree/main/src/training/model_embeddings/mmbert_32k#train-a-new-task-from-a-standard-base)给出了 JSON 格式、支持的损失函数、可选的教师监督和配置字段。

下面的命令假设这些文件已放在 `/data/retrieval` 下。把 `VELA_TRAIN_CONFIG_SHA256` 设为配置文件的 SHA-256：

```bash
export PYTHONPATH="$PWD"

uv run --locked --project src/training/model_embeddings/mmbert_32k \
  python -m src.training.model_embeddings.mmbert_32k.newbase_training \
  --config /data/retrieval/task.json \
  --config-sha256 "${VELA_TRAIN_CONFIG_SHA256:?Set the configuration SHA-256}" \
  --output /data/retrieval/run --device cuda
```

配置选择 `task: "embedding"`、数据集、输入预算、训练步数和评测间隔。先用较小的预算，在扩大运行规模前检查最初的开发结果。

## 评测训练好的 checkpoint {#evaluate-the-trained-checkpoint}

用与基线相同的开发语料评测保存的 checkpoint：

```bash
export PYTHONPATH="$PWD"

uv run --locked --project src/training/model_embeddings/mmbert_32k \
  python -m src.training.model_embeddings.mmbert_32k.newbase_scoring \
  --model /data/retrieval/run/step-100 --task embedding \
  --known-dev /data/retrieval/validation --split validation \
  --output /data/retrieval/evaluation --device cpu --token-budget 32768
```

把 checkpoint 路径换成你的运行保存的某一步。用开发结果选择 checkpoint，再在留出数据上做一次单独的最终比较。比较检索、相似度和多语言迁移，并分别测量短输入和长输入，以及每种部署深度/维度下的延迟和内存。

完整的 MMTEB 结果需要完整的选定基准和匹配的评测协议。任务子集可以诊断差距，但不能确立整体基准得分或排行榜名次。

## 部署结果 {#deploy-the-result}

把 checkpoint 声明为嵌入部署，并在召回使用它的地方指定它（[嵌入](../model-runtime/guides/embeddings.md)）；用之前模型建立索引的存储必须重新建立索引。在把流量交给它之前，用真实请求测试召回。

## 早期 mmBERT 流程 {#earlier-mmbert-workflows}

最初的基础模型和嵌入流程保留在[流程 README](https://github.com/vllm-project/semantic-router/tree/main/src/training/model_embeddings/mmbert_32k) 中。其中签入的历史配置可以复现早期的 mmBERT 流程。新的训练请使用上面的 Vela 起点。
