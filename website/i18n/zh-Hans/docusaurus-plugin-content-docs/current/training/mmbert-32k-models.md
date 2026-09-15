---
title: mmBERT-32K 基础、嵌入与重排序
sidebar_label: mmBERT-32K 模型
translation:
  source_commit: "e8c4109fd4151ad0c7c0163c8ead375bef882ddf"
  source_file: "docs/training/mmbert-32k-models.md"
  outdated: false
---

# mmBERT-32K 基础、嵌入与重排序 {#mmbert-32k-foundation-embedding-and-reranking}

三个模型构成一个渐进的文本检索家族：

```text
mmBERT base encoder
  -> 32K YaRN masked-language continuation
     -> bi-encoder embedding training
     -> cross-encoder reranking training
```

基础提供长多语言表示。Embedder 通过独立编码输入使检索高效。Reranker 通过联合阅读每个查询-文档对，在小候选集上花费更多计算。

## 共享基础 {#shared-foundation}

[`mmbert-32k-yarn`](https://huggingface.co/llm-semantic-router/mmbert-32k-yarn)
是 ModernBERT 家族编码器，有 22 层 Transformer、隐藏大小 768，以及约 3.07 亿参数。它使用 256K 词表，并用 YaRN 旋转位置缩放将基础 8192 token 位置范围扩展到 32768 token。

已核对工作流在 9 个 CC-100 语言流上继续掩码语言模型训练。它用显式文档边界打包完整 32K 序列，并应用 30% token 掩码。生产配置使用 30774 个序列、1 个 epoch、学习率 `1e-5`、BF16、每设备 batch 1，以及梯度累积 16。

此阶段改变编码器的语言表示和长上下文行为；它不添加路由头。

## 嵌入模型：bi-encoder {#embedding-model-bi-encoder}

[`mmbert-embed-32k-2d-matryoshka`](https://huggingface.co/llm-semantic-router/mmbert-embed-32k-2d-matryoshka)
将 32K 基础用作 bi-encoder：

```text
query -----------------> shared encoder -> normalized query vector
document --------------> shared encoder -> normalized document vector
                                            cosine/dot-product similarity
```

因为两个输入独立编码，文档向量可以预计算并大规模搜索。

“2D Matryoshka” 表示损失监督两个轴：

- **嵌入维度：** 一个训练好的向量可以截断到 768、512、256、128 或 64 维。
- **编码器深度：** 中间层接收有用的检索监督，从而支持可选层表示。

已核对训练路径将 BGE-M3 检索样本与可选 AllNLI 样本结合，并使用由 Sentence Transformers `Matryoshka2dLoss` 包装的 multiple-negatives ranking 目标。生产配置使用 32K 最大长度、1 个 epoch、batch 16 加累积 2、学习率 `2e-5`、BF16，以及用于语义相似度评测的 STS-B。

## 重排序模型：cross-encoder {#reranking-model-cross-encoder}

[`mmbert-rerank-32k-2d-matryoshka`](https://huggingface.co/llm-semantic-router/mmbert-rerank-32k-2d-matryoshka)
一起阅读查询和候选：

```text
[query, candidate] -> shared 32K encoder -> CLS representation -> relevance score
```

联合注意力比 bi-encoder 搜索更昂贵，但可以建模查询与候选之间的细粒度交互。在初始检索之后使用它，而不是覆盖整个语料。

该模型在第 3、6、11 和 22 层以及 768、512、256、128 和 64 维附加打分头。这创建 20 个层/维度头。训练在所有头上平均二元相关性损失，使用 BGE-M3 查询-正例-负例记录。生产配置每个查询使用 3 个负例、1 个 epoch、batch 16 加累积 2、学习率 `2e-5`、BF16 和梯度检查点。

这些头使早期层分数可训练；训练器本身仍计算所有隐藏状态。若运行时想要延迟节省，必须单独实现提前终止。

## 运行已核对配置 {#run-the-checked-configurations}

安装家族依赖并设置本地数据和输出路径：

```bash
python -m pip install --requirement \
  src/training/model_embeddings/mmbert_32k/requirements.txt

export PYTHONPATH="$PWD/src"
export MMBERT32K_FOUNDATION_DATA=/path/to/tokenized-cc100-32k
export MMBERT32K_FOUNDATION_OUTPUT=/path/to/mmbert-32k-yarn
export MMBERT32K_BGE_DATA=/path/to/bge-m3-data
export MMBERT32K_EMBEDDER_OUTPUT=/path/to/mmbert-embed-32k-2d
export MMBERT32K_RERANKER_OUTPUT=/path/to/mmbert-rerank-32k-2d
```

在开始大规模运行前检查每个已解析命令：

```bash
python -m training.model_embeddings.mmbert_32k \
  --config src/training/model_embeddings/mmbert_32k/configs/foundation.json \
  --stage train --print-command

python -m training.model_embeddings.mmbert_32k \
  --config src/training/model_embeddings/mmbert_32k/configs/embedder.json \
  --print-command

python -m training.model_embeddings.mmbert_32k \
  --config src/training/model_embeddings/mmbert_32k/configs/reranker.json \
  --print-command
```

去掉 `--print-command` 即可运行。先用 `--stage prepare` 准备基础数据集；使用[工作流 README](https://github.com/vllm-project/semantic-router/tree/main/src/training/model_embeddings/mmbert_32k) 中的精确准备和数据完整性流程。

## 发布前评测 {#evaluate-before-release}

对于基础，在留出语料上验证长上下文加载和掩码语言损失。对于 embedder，在每个支持维度和所选层上报告检索 Recall@k 和语义相似度。对于 reranker，为全部 20 个头报告排序指标和分类质量，而不仅仅是最大的最终层头。

还要在你打算部署的精确层、维度、序列长度和 batch 大小上测量延迟和内存。“Matryoshka” 提供选择；它不会让每个选择同样准确。
