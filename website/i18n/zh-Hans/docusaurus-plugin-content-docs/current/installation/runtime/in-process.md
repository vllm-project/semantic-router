---
title: 进程内模型
description: 选择本地引擎和硬件，配置并运行分类器。
translation:
  source_commit: "dc7f402642a8b8ecec8218e2086a4c6f186ea406"
  source_file: "docs/installation/runtime/in-process.md"
  outdated: false
---

如果希望在本地推理且不另建模型服务，可以在 Router 进程内运行模型。先按照[安装指南](/zh-Hans/docs/installation/)安装 CLI 和匹配的镜像。

## 选择引擎和模型 {#choose-an-engine-and-model}

| 引擎 | 硬件 | 模型格式 |
| --- | --- | --- |
| Candle | CPU（`cpu`）、NVIDIA（`cuda:0`）、Apple Metal（`metal:0`） | 兼容的模型权重；精度为 `native` 或 `fp32` |
| ONNX Runtime | CPU（`cpu`） | ONNX 图；使用 `precision: native` |
| ONNX Runtime with MIGraphX | AMD GPU（`migraphx:N`） | 兼容的 ONNX 图；精度为 `native` 或 `fp16` |
| ML 和 NLP 引擎 | CPU | 已训练的选择器或关键词匹配配置 |

Candle 支持 GPU 索引 0。BERT、已合并的 BERT LoRA 和 LoRA token 模型不支持 Metal。ORT 接受 CPU 和 MIGraphX 设备；上表中的 NVIDIA 和 Metal 选项使用 Candle。现有 OpenVINO 主嵌入模型集成仍采用单独的平台配置，不提供 deployment provider 或缓存/分窗 API。

| 模型家族 | 支持的用途 |
| --- | --- |
| ModernBERT / mmBERT | 序列分类和 token 分类；mmBERT 文本嵌入 |
| BERT 和已合并的 BERT LoRA | 序列分类和 token 分类；BERT 文本嵌入 |
| DeBERTa | 序列分类 |
| 专用幻觉检测和 NLI 模型 | 使用 Candle 检查内容依据和句对关系 |
| Qwen3 和 Gemma 嵌入模型 | 使用 Candle 生成文本嵌入 |
| 兼容的多模态模型 | 模型实际提供的文本、图像和音频编码器 |
| MLP、KNN、K-means、SVM | 在 CPU 上根据嵌入特征选择模型 |
| BM25 和 N-gram | 在 CPU 上匹配关键词 |
| TextRank、TF-IDF 和启发式方法 | 在 Go 中执行提示词压缩和规则 |

ORT 支持导出的 mmBERT 分类图，以及 mmBERT 或多模态嵌入图。本地分类器最多接受 **512 个 token**，包含特殊 token；嵌入模型的上限由模型决定。分类器需要针对任务训练的分类头和标签，例如领域分类、提示词防护、PII、事实核查、反馈或输出模态。Qwen3/Gemma 嵌入模型不提供本地生成式分类功能。

其他本地功能的配置见[嵌入模型](embeddings.md)、[安全模型](safety.md)、[MLP 选择](/zh-Hans/docs/tutorials/algorithm/selection/mlp)和[关键词信号](/zh-Hans/docs/tutorials/signal/heuristic/keyword)。

## 配置分类器 {#configure-a-classifier}

下面的示例在 CPU 上运行自定义邮件分类器。开始前请准备：

- 将完整且兼容的模型文件放到 `models/email-classifier`。
- 将 `BENIGN` 和 `PHISHING` 替换为模型的标签，并保持训练时的顺序。
- 将回答模型的地址替换为 Router 可以访问的地址。

**deployment** 指定模型文件和引擎。**binding** 将该部署连接到 `email-risk` 分类规则。将以下内容保存为 `config.yaml`：

```yaml
version: v0.3
listeners:
  - name: http
    address: 0.0.0.0
    port: 8899
providers:
  defaults:
    model: answer-model
  models:
    - name: answer-model
      backend_refs:
        - name: answer
          endpoint: 127.0.0.1:8000
          protocol: http
routing:
  model_bindings:
    classifier.email-risk:
      deployment: email-risk-cpu
      contract: label_distribution.v1
      adapter: auto
  signals:
    classifiers:
      - name: email-risk
        type: local
        labels: [BENIGN, PHISHING]
  decisions:
    - name: inspect-email
      priority: 100
      rules:
        operator: AND
        on_unknown: fail_request
        conditions:
          - type: classifier
            name: email-risk
            label: PHISHING
            predicate:
              gte: 0.8
      modelRefs:
        - model: answer-model
global:
  model_catalog:
    deployments:
      email-risk-cpu:
        artifact: models/email-classifier
        provider: candle
        device: cpu
        precision: native
        input:
          max_tokens: 512
          overflow: reject
```

## 启动并测试 {#start-and-test}

```bash
vllm-sr config validate --config config.yaml
vllm-sr serve --config config.yaml
curl -sS http://localhost:8899/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"auto","messages":[{"role":"user","content":"Review this email requesting a password reset."}]}'
```

当分类器的钓鱼邮件分数达到 0.8 时，这条规则匹配。示例将匹配和不匹配的请求都发送给同一个回答模型；修改决策中的模型或插件，即可执行自己的策略。

## 更换引擎或模型 {#change-the-engine-or-model}

对于导出的 mmBERT ONNX 模型，将部署改为 `provider: ort`，让 `artifact` 指向完整的 ONNX 目录，并在 binding 中设置 `adapter: mmbert` 和 `head: onnx/model.onnx`。设备从上表中选择。

自定义模型目录不需要注册表条目。目录应包含权重或 ONNX 图、分词器、配置、标签以及图引用的外部张量文件。LoRA 模型需要完整的已合并权重，不能只提供 adapter 增量。已注册模型由正常的 serve 流程下载。替换正在使用的模型时，固定 `revision` 并使用新目录。

Binding 仅对一个配方生效。将它放入对应配方的 `routing` 块，即可更换该配方的模型而不影响其他配方。完整字段见[配置参考](/zh-Hans/docs/api/configuration-schema)。

从源码构建时，先运行 `make vllm-sr-dev`，再为 serve 命令添加 `--image-pull-policy never`。
