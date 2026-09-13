---
title: 嵌入模型
description: 为语义路由、缓存和向量存储配置本地或远程嵌入模型。
translation:
  source_commit: "dc7f402642a8b8ecec8218e2086a4c6f186ea406"
  source_file: "docs/installation/runtime/embeddings.md"
  outdated: false
---

嵌入模型用于语义匹配、缓存、记忆和向量存储。选择本地模型或兼容 OpenAI API 的文本嵌入服务，将下面相应的片段合入现有 `config.yaml`。

## 本地嵌入 {#local-embeddings}

本例选择维护的 mmBERT 模型，使用第 22 层和 768 维向量：

```yaml
global:
  model_catalog:
    embeddings:
      semantic:
        embedding_config:
          model_type: mmbert
          preload_embeddings: true
          target_dimension: 768
          target_layer: 22
        mmbert_model_path: models/mmbert-embed-32k-2d-matryoshka
```

使用匹配的 Candle 或 ORT 镜像，并准备模型所需文件。其他模型家族和设备见[进程内模型](in-process.md)。嵌入信号沿用现有候选文本和阈值。

### AMD GPU {#amd-gpu}

AMD serve 默认让语义嵌入在 CPU 上运行。如需在 AMD GPU 上运行 mmBERT 嵌入，在上述本地配置中添加以下部署和 binding：

```yaml
global:
  model_catalog:
    deployments:
      local-embedding:
        artifact: models/mmbert-embed-32k-2d-matryoshka
        provider: ort
        device: migraphx:0
        precision: native
        input:
          max_tokens: 1024
          overflow: reject
routing:
  model_bindings:
    embedding:
      deployment: local-embedding
      contract: embedding.v1
      adapter: mmbert
```

使用维护的 ROCm 镜像和模型的 ONNX 导出文件。GPU 嵌入必须设置正数 `max_tokens`；根据工作负载和模型选择预算。本例拒绝超过 1024 个 token 的输入。更大的预算会增加准备和推理成本。分类任务另有 512 个 token 的上限。

## 远程嵌入 {#remote-embeddings}

在 Router 环境中设置服务密钥：

```bash
export EMBEDDING_API_KEY="<provider-key>"
```

```yaml
global:
  model_catalog:
    embeddings:
      semantic:
        embedding_config:
          backend: openai_compatible
          model_type: remote
          preload_embeddings: false
          target_dimension: 1536
        endpoint:
          base_url: https://embedding.example.com/v1
          model: text-embedding-model
          api_key_env: EMBEDDING_API_KEY
          timeout_seconds: 10
          max_retries: 2
          max_response_bytes: 16777216
          dimensions: 1536
```

将 URL、模型和维度替换为服务提供方的值。如果基础 URL 尚未以 `/embeddings` 结尾，Router 会添加该后缀。两处维度设置必须一致。认证使用 bearer token；默认响应大小上限为 16 MiB。

远程嵌入只支持文本，不提供本地分词器分窗、层选择、图像或音频编码。需要这些功能的配置必须使用兼容的本地模型。远程服务会接收到待嵌入的文本。

## 匹配使用方的要求 {#match-the-consumers-requirements}

| 使用方 | 更换模型前需要检查 |
| --- | --- |
| 语义信号和模型选择器 | 匹配阈值和训练时的嵌入空间 |
| 向量存储和持久化缓存 | 已存索引的维度和模型 revision |
| 内存 mmBERT 缓存 | 必须提供第 6 层、256 维向量 |
| 记忆 | 配置的维度；mmBERT 默认为 256，多模态模型默认为 384 |
| 响应缓存和 RAG 分窗 | 本地分词器分窗支持 |
| 图像或音频功能 | 本地模型包含所需编码器 |

嵌入空间改变后，应重建已存向量，即使新模型的输出维度相同。ORT 导出文件必须包含已启用功能使用的每一层。Router 在启动时对这些层预热；层缺失或无效会阻止配置激活。

## 启动并检查 {#start-and-inspect}

```bash
vllm-sr config validate --config config.yaml
vllm-sr serve --config config.yaml
curl -fsS http://localhost:8080/startup-status | jq '.embedding_provider'
```

启动时会检查服务或本地模型以及向量维度。要测试具体输入，使用 `POST /api/v1/diagnostics/embeddings`；请求示例见 [API 参考](/zh-Hans/docs/api/apiserver)。状态报告会隐藏凭据。
