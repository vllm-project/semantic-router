---
title: 嵌入模型
description: 使用 Vela 进行语义路由和检索，或连接远程嵌入服务。
translation:
  source_commit: "8506d548e0217bb14ef45aeba21875c0013e16ee"
  source_file: "docs/installation/runtime/embeddings.md"
  outdated: true
---

嵌入模型将文本转换为向量，用于语义匹配、检索、缓存和记忆。Vela Embedding 是默认的本地模型。选择下方的 CPU 配置、[AMD GPU](#amd-gpu) 或[远程服务](#remote-embeddings)，将配置片段合入现有 Router 配置。

## 本地嵌入 {#local-embeddings}

选择 Vela 的完整表示：第 22 层、768 维。

```yaml
global:
  model_catalog:
    embeddings:
      semantic:
        mmbert_model_path: models/Vela-1.0-Encoder-307M-Embedding
        embedding_config:
          model_type: mmbert
          preload_embeddings: true
          target_layer: 22
          target_dimension: 768
```

CPU 镜像使用 Candle 推理。`mmbert` 指定兼容的推理架构，模型路径选择 Vela。正常的 serve 流程会下载已注册的模型。

### AMD GPU {#amd-gpu}

在 AMD 上运行 Vela Embedding，需要增加显式 ROCm deployment 和 binding：

```yaml
global:
  model_catalog:
    deployments:
      local-embedding:
        artifact: models/Vela-1.0-Encoder-307M-Embedding
        revision: 1e57cebf5a7b7fec6e6973f05bbca97c5cca4436
        provider: ort
        device: rocm:0
        precision: native
        custom_ops_profile: ck_flash_attention
        input:
          max_tokens: 32768
          overflow: reject
routing:
  model_bindings:
    embedding:
      deployment: local-embedding
      contract: embedding.v1
      adapter: mmbert
      head: onnx/model_fa.onnx
```

运行 `vllm-sr serve --platform amd --config config.yaml`。AMD 镜像必须包含 ROCm execution provider 和 CK 算子库。发布的图支持最长 32,768 tokens，包含特殊 token；此 GPU deployment 会拒绝 CPU 回退。

保留下载的 ONNX 配套图和外部权重，启用的功能可能使用不同嵌入层。包含分类器和重排的完整配置见 [Vela AMD 配方](https://github.com/vllm-project/semantic-router/blob/main/config/recipes/vela-amd/README.md)，其完整分类链路的输入上限为 8K。

## 测试嵌入 {#test-an-embedding}

启动 Router 后检查就绪状态，生成两个向量：

```bash
curl -fsS http://localhost:8080/ready
curl -fsS http://localhost:8080/api/v1/diagnostics/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"texts":["How do I reset my password?","I forgot my login password."],"model":"mmbert","target_layer":22,"dimension":768}' \
  | jq '{total_count, total_processing_time_ms, embeddings: [.embeddings[] | {dimension, model_used, processing_time_ms}]}'
```

预期得到两个 768 维结果及实测处理时间。通过 [Route Preview](lifecycle-diagnostics.md#inspect-the-executed-path) 查看语义信号如何影响路由。[API 参考](../../api/apiserver.md)还提供相似度请求。

## 输入策略 {#input-policy}

语义路由默认使用代表性文本采样。需要嵌入完整路由文本时，设置：

```yaml
global:
  model_catalog:
    embeddings:
      semantic:
        embedding_config:
          full_context: true
```

此设置适用于语义 embedding 信号和使用 embedding 的本地 Complexity。独立的远程 Complexity 服务保留自己的输入策略。提示词压缩仍会生效，除非该信号被配置为豁免。

Deployment 的 `input.max_tokens` 仍限制可接受的输入。增加该上限不会自动开启 `full_context`。选择适合模型和时延目标的预算；长输入容量本身不代表检索准确率。

Vela 提供第 3、6、11、22 层和 64、128、256、512、768 维表示。较小表示可以降低开销；更改 `target_layer` 或 `target_dimension` 前先评估检索质量。

## 远程嵌入 {#remote-embeddings}

在 Router 环境中设置服务密钥：

```bash
export EMBEDDING_API_KEY="<provider-key>"
```

配置服务模型和向量维度：

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

将 URL、模型名和维度替换为服务提供的值，两处维度必须一致。Router 使用 bearer 认证调用 `/embeddings`。服务会接收需要嵌入的文本。

远程嵌入支持文本。依赖本地 tokenizer 窗口、层选择、图像或音频编码的功能需要兼容的本地模型。

## 更换模型时保持向量空间一致 {#change-a-model-without-mixing-vector-spaces}

更换嵌入权重、层或维度后，应重新索引已存储文档。维度相同不代表两个向量空间兼容。持久缓存和记忆按表示隔离；向量存储的空间发生变化时需要重新摄取文档。

| 功能 | 更换表示前检查 |
| --- | --- |
| 语义信号和模型选择器 | 重新检查阈值与选择器兼容性 |
| 向量存储、记忆和持久缓存 | 匹配维度并重新索引受影响数据 |
| 内存 mmBERT 缓存 | 保留第 6 层、256 维表示 |
| 响应缓存和 RAG 窗口 | 保留本地 tokenizer 窗口能力 |

ONNX 部署必须包含启用功能所需的全部层，缺少相应层会阻止启动。见[故障排查](lifecycle-diagnostics.md)。
