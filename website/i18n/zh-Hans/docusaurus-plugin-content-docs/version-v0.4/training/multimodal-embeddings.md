---
title: 多模态嵌入模型
sidebar_label: 多模态嵌入
translation:
  source_commit: "e8c4109fd4151ad0c7c0163c8ead375bef882ddf"
  source_file: "docs/training/multimodal-embeddings.md"
  outdated: false
---

# 多模态嵌入模型 {#multimodal-embedding-models}

多模态 embedder 将文本、图像和音频映射到共享向量空间。对齐后，Router 可以用与文本检索相同的相似度操作跨模态比较输入。

当内存和延迟是主要约束时选择小模型。当表示容量、长文本和更强的视觉/音频塔值得更大服务占用时选择大模型。

## 架构比较 {#architecture-comparison}

| 组件 | Small | Large |
| --- | --- | --- |
| 文本塔 | `all-MiniLM-L6-v2` | mmBERT-32K 2D Matryoshka embedder |
| 图像塔 | SigLIP base，patch 16，512 像素输入 | SigLIP2 SO400M，patch 14，384 像素输入 |
| 音频塔 | Whisper tiny 编码器 | Whisper medium 编码器 |
| 跨模态层 | 两层 Transformer 融合 | 独立塔投影到同一空间 |
| 输出 | 规范化 384 维向量 | 规范化 768 维向量 |
| 文本上限 | 已核对生产配置中为 128 token | 32768 token |
| 主要损失 | 由 Matryoshka 包装的对比对齐 | 缓存的 multiple-negatives ranking 损失 |

两个模型都使用模态专用编码器，因为像素、波形和文本 token 需要不同的前端。投影和对齐训练使得到的表示可比较。

## 小模型 {#small-model}

[`multi-modal-embed-small`](https://huggingface.co/llm-semantic-router/multi-modal-embed-small)
将紧凑的预训练塔与两层融合 Transformer 结合。其规范化 384 维输出在 32、64、128、256 和 384 维受监督，因此部署可以在向量大小和质量之间权衡。

训练分阶段进行，避免一次动摇每个编码器：

1. 在预训练塔冻结时训练投影和融合层。
2. 解冻所选上层以做部分适配。
3. 对齐表示稳定后，微调完整图文路径。
4. 使用缓存的 Whisper 输入特征继续音文对齐。

已核对的 Stage 1 配置使用缓存的 LLaVA-CC3M 图文样本、6 个 epoch、每进程 batch 64、学习率 `1e-4`、混合精度、温度 `0.07` 和 Matryoshka 对比损失。Stage 2、4 和 5–7 会更改哪些塔可训练以及采样哪对模态；启动前先检查阶段配置。

从仓库根目录运行：

```bash
python -m pip install --requirement \
  src/training/model_embeddings/multimodal/small/requirements.txt

export PYTHONPATH="$PWD/src"
export MM_EMBED_SMALL_TRAIN_CACHE=/path/to/cache/train
export MM_EMBED_SMALL_VAL_CACHE=/path/to/cache/validation
export MM_EMBED_SMALL_OUTPUT_DIR=/path/to/output

python -m training.model_embeddings.multimodal.small.train \
  --config src/training/model_embeddings/multimodal/small/configs/production.yaml \
  --print-config
```

检查已解析路径和阶段后，去掉 `--print-config`，并使用适合你环境的分布式启动器。`--max-steps 2` 提供短加速器冒烟运行。

## 大模型 {#large-model}

[`multi-modal-embed-large`](https://huggingface.co/llm-semantic-router/multi-modal-embed-large)
使用长上下文 mmBERT embedder、更大的 SigLIP2 视觉塔和更大的 Whisper 音频塔。每个塔被投影到共享的 768 维空间。与小型融合架构不同，生产三编码器保持模态编码独立，因此缓存嵌入和成对对比训练保持直接。

原始样本被预处理为已校验的张量分片。训练按顺序加载这些分片，带有有界预取，并使用缓存的 multiple-negatives ranking 损失：匹配对是正例，有效 batch 中的其他样本是负例，配置的难负例使边界更有信息。

生产配置使用 10 个 epoch、每设备 batch 12、梯度累积 8、学习率 `1e-5`、BF16、损失缩放 20，以及 50% 难负例比例。

```bash
python -m pip install --requirement \
  src/training/model_embeddings/multimodal/large/requirements.txt

export PYTHONPATH="$PWD/src"
export MM_EMBED_LARGE_OUTPUT_DIR=/path/to/output
export MM_EMBED_LARGE_TRAIN_CACHE=/path/to/cache/train
export MM_EMBED_LARGE_VAL_CACHE=/path/to/cache/validation

python -m training.model_embeddings.multimodal.large.train \
  --config src/training/model_embeddings/multimodal/large/configs/production.yaml
```

[工作流 README](https://github.com/vllm-project/semantic-router/tree/main/src/training/model_embeddings/multimodal/large)
记录预处理、冒烟配置、评测和打包。

## 评测跨模态质量 {#evaluate-cross-modal-quality}

评测你打算服务的每一对模态：text-image、image-text、text-audio 和 audio-text。报告两个方向的 Recall@k，以及按领域和按语言的切片。对于小模型，在每个支持的 Matryoshka 维度上重复评测。若应用依赖同模态检索，也要测试它；跨模态对齐不会自动证明强的 image-image 或 audio-audio 检索。

保持与产物一致的预处理契约。图像缩放、音频特征提取、文本截断、规范化和输出维度是模型的一部分，不是可互换的服务细节。
