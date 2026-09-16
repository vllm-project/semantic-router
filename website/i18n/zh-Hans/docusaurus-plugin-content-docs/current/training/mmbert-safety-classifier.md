---
title: 训练 Vela Safety 和 Hazard
sidebar_label: Safety 和 Hazard
translation:
  source_commit: "915ddf56e0335e2046c38aa17c4aec6233908039"
  source_file: "docs/training/mmbert-safety-classifier.md"
  outdated: false
---

# 训练 Vela Safety 和 Hazard {#train-vela-safety-and-hazard}

根据应用的内容策略适配 Vela Safety 和 Hazard。**Safety** 预测 `safe` 或 `unsafe`；**Hazard** 识别风险类别，帮助选择合适的响应。提示词注入和越狱检测使用独立的 Guard 模型。

直接使用已发布模型，请参阅[安全模型](../installation/runtime/safety.md)。本页介绍训练兼容 checkpoint 的流程。

## 定义输出 {#define-the-outputs}

两个模型均使用 Vela Encoder 基座和序列分类 head。

| 模型 | 输出 | 训练目标 |
| --- | --- | --- |
| Safety | 两类 softmax：`safe`、`unsafe` | 交叉熵 |
| Hazard | 十二个独立 sigmoid 分数 | 带 mask 的二元交叉熵 |

Hazard 使用以下标签顺序：

```text
violence, criminal_activity, sexual_content, child_exploitation,
hate, harassment_abuse, regulated_substances, weapons,
self_harm, privacy, specialized_advice, misinformation
```

一个请求可以具有多个风险，也可以没有风险。Hazard 训练应包含安全负例。某类别尚未经过标注时，将对应 `label_mask` 设置为 0，表示未知，不应当作不存在。

## 准备训练和评测数据 {#prepare-training-and-evaluation-data}

可以从 AEGIS 和 CultureGuard 的[数据构建工具](https://github.com/vllm-project/semantic-router/blob/main/src/training/model_classifier/vela-applications.md#data-preparation)开始，或使用自己的请求标注。按内容策略复核来源标签：提及敏感主题、引用威胁和请求有害行为需要不同判断。

同时包含安全的教育、预防、支持性请求和不安全样本。相关文档、对话和翻译应处于同一分区，分别准备训练集、开发集和最终测试集。

各任务需要带标签映射的 `contract.json`。Safety 行包含单个 `label`；Hazard 行包含有序的 `targets` 和 `label_mask` 数组。[配方参考](https://github.com/vllm-project/semantic-router/blob/main/src/training/model_classifier/vela-applications.md)提供完整格式和已复核数据的准入命令。

## 从 Vela Encoder 训练 {#train-from-vela-encoder}

使用独立环境，安装 Transformers 4.57.6 和适合平台的 PyTorch 构建。[训练参考](https://github.com/vllm-project/semantic-router/blob/main/src/training/model_classifier/vela-applications.md#initialization-and-training)列出依赖。ROCm 使用 PyTorch 的 `cuda` 设备 API。

下载固定 revision 的 Vela Encoder 到 `/models/vela-base`，将 `VELA_BASE_REVISION` 设置为该 revision。以下命令假设契约和数据已准备完毕。

训练 Safety：

```bash
python -m src.training.model_classifier.sequence_repair.train \
  --method full --fresh-head \
  --base /models/vela-base --base-id llm-semantic-router/Vela-1.0-Encoder-307M \
  --base-revision "${VELA_BASE_REVISION:?Set the downloaded revision}" \
  --contract /data/safety/contract.json \
  --train /data/safety/train.jsonl --dev /data/safety/dev.jsonl \
  --output /data/safety/run \
  --steps 600 --batch-size 4 --accumulate 4 \
  --max-length 32768 --microbatch-token-budget 32768 \
  --learning-rate 0.00001 --head-learning-rate 0.0001 \
  --eval-every 200 --evaluation-dtype float32 \
  --selection binary-fp-budget-recall --positive-label unsafe \
  --selection-false-positive-budget 0.1
```

使用多标签训练器训练 Hazard：

```bash
python -m src.training.model_classifier.safety_classifier.train_vela_hazard \
  --method full --fresh-head \
  --base /models/vela-base --base-id llm-semantic-router/Vela-1.0-Encoder-307M \
  --base-revision "${VELA_BASE_REVISION:?Set the downloaded revision}" \
  --contract /data/hazard/contract.json \
  --train /data/hazard/train.jsonl --dev /data/hazard/dev.jsonl \
  --output /data/hazard/run \
  --steps 600 --batch-size 4 --accumulate 4 \
  --max-length 32768 --microbatch-token-budget 32768 \
  --learning-rate 0.00001 --head-learning-rate 0.0001 \
  --eval-every 200 --evaluation-dtype float32 \
  --selection fp-budget-macro-f1 --selection-false-positive-budget 0.05
```

以上预算和学习率仅为示例。在选择 checkpoint 前，先确定应用允许的误报率。完整训练会保存 `best-model` 和 `last-model` 目录。继续兼容的 Vela 任务 checkpoint 时，将其作为 base，并省略 `--fresh-head`。

## 评测风险和长输入 {#evaluate-risks-and-long-inputs}

Safety 需要测量不安全请求召回率和安全请求误报率。Hazard 需要测量每类 precision、recall、average precision，以及安全请求触发任意类别的比例。

使用开发集选择阈值，再固定阈值完成最终测试。分别报告语言和类别，避免大数据源掩盖弱项。

同时测试短请求和 8K、16K、32K 文档，将相关内容放在不同位置，并加入正常引用。预算包含特殊 tokens；训练记录超长样本，评测拒绝溢出。完整上下文和窗口扫描向模型提供的上下文不同，需要分别评测。

## 导出并接入模型 {#export-and-connect-the-models}

使用[序列导出工具](https://github.com/vllm-project/semantic-router/tree/main/src/training/model_classifier/sequence_repair#freeze-then-evaluate-the-independent-test)，指定 `--method full`、选中的 checkpoint，以及 `--runtime-task safety` 或 `--runtime-task hazard`。它保留标签顺序，输出权重、tokenizer 和 runtime 映射。ONNX 部署需要从同一组权重导出的图。

通过[本地绑定](../installation/runtime/in-process.md)或支持的[外部服务](../installation/runtime/external.md)配置模型。[Safety 信号指南](/docs/tutorials/signal/learned/safety)说明二元风险规则和类别条件。在带类别条件的 Safety 规则中，Safety 达到阈值后才运行 Hazard。

使用[路由预览](../installation/runtime/lifecycle-diagnostics.md)，在实际服务配置下验证分数、决策、错误和延迟。

## 早期 Safety 模型 {#earlier-safety-models}

早期 mmBERT Safety adapter 和九类 Hazard head 保留在[历史训练流程](https://github.com/vllm-project/semantic-router/tree/main/src/training/model_classifier/safety_classifier)。其标签契约不同于 Vela 的十二类独立 Hazard 输出。
