---
title: 调优与验证 Recipe
description: 用真实请求改善路由质量、时延、成本和 Agent 连续性。
translation:
  source_commit: "68519cf498b94c947cf4c7cd9a2800a41276dea6"
  source_file: "docs/benchmarking/agent-evaluation-loop.md"
  outdated: false
---

# 调优与验证 Recipe {#tune-and-verify-a-recipe}

好的 recipe 能为请求选择合适的处理路径，并在时延、成本和安全要求内交付更好的结果。本指南介绍如何结合真实 Preview 请求、路由响应和会话轨迹来改进配方。

开始前需要一个运行中的部署和可访问的模型端点。新部署请先阅读 [Agent 安装指南](../installation/agent)。配置与 Preview 使用管理地址，真实模型请求使用推理地址。

## 选择目标并保留基线 {#choose-an-objective-and-a-baseline}

选择一个可测量的目标，例如减少不必要的推理调用、改善高风险问题的回答、加快工具调用轮次，或减少 Agent 运行中的模型切换。在调整阈值前，先确定质量底线以及可接受的时延或成本。

导出内置配方包或保存当前配置，记录生效的 recipe、运行时镜像、模型版本和模型分配。保留原始测试用例，以便变更前后运行相同请求。

为每个 decision 和兜底路径准备一组有代表性的小数据集，覆盖普通请求、边界情况、多语言改写、长输入、引用的指令和多轮对话。加入反例：医学定义不一定需要走个性化治疗建议的路径，被引用的攻击文本也不应自动被当作指令。

为部分请求提供相同的背景信息，但要求完成不同任务，并单独变化简洁或详细的回答风格。这能检验难度信号是否识别了实际工作，而非主题或回答长度。报告覆盖率时，将相关改写和工具变体视为同一组用例。

## 为每一层明确职责 {#give-each-part-a-clear-job}

| 层 | 用途 |
| --- | --- |
| Signals | 识别语义、风险、难度和明确的请求约束。 |
| Projections | 将证据组合成 decision 可以使用的分数或类别。 |
| Decisions | 选择少量具有不同处理行为的路径。 |
| Algorithms | 在每条路径内选择或协调模型。 |
| Plugins | 应用检索、工具、缓存和数据处理策略。 |
| Models | 在声明的能力和上下文限制内执行请求。 |

Decision 名称保持简短，例如 `simple`、`medium` 和 `reasoning`。处理行为不同时再增加 decision，不必为每个主题或语言单独增加一条。

对必需工具、结构化输出等明确事实使用 heuristic。语义判断则使用 learned signals：Embedding 识别意图，Complexity 判断难度，Domain 配合 FactCheck 识别需要谨慎处理的问题，Feedback 配合 Reask 识别需要纠正的回答。单个信号过于宽泛时，通过 projection 组合证据。主题本身不等于难度；FactCheck 预测核实需求，并不核验陈述真假。

检查未知信号如何影响每个 decision，尤其是使用 `NOT` 的条件。分类器失败不应成为选择更便宜路径的依据。调整阈值前，先比较简单与困难请求的信号分数。如果两者高度重叠，需要改进任务示例或模型；把所有不确定的请求升级为推理会增加用量，但不能证明区分能力得到改善。

加入关键词条件也不保证减少推理：被使用的信号族可能在计算 decision 前并行执行，需要测量真实请求的成本。

测试回答恢复时，应包含之前的 assistant 回复：缺少这段历史时，Feedback 路由会跳过推理。直接修正指令、自包含的错误报告与 Feedback 命中应分开测试，并用调整语气、格式等普通修改请求作对照。协作路由需要区分多执行者授权、关于 Agent 的讨论以及仅使用一位独立执行者的请求。工作流负责内部阶段，用户不必说出每个阶段才能请求协作。

检查多语言示例是否在原型压缩后仍被保留。规则级 `prototype_scoring` 配置随 Recipe 一起发布；省略时继承全局设置。使用 `enabled: false` 保留全部去重候选，并通过 `best_weight` 和 `top_m` 指定组合评分方式。替换基线前先测量实际效果。

## 校验并预览候选配方 {#validate-and-preview-the-candidate}

编辑前先发现运行中的配置契约：

```bash
vllm-sr config schema --endpoint "$ROUTER_ORIGIN" \
  --surface algorithm:multi_factor
vllm-sr config validate --config candidate.yaml \
  --endpoint "$ROUTER_ORIGIN"
vllm-sr config plan --config candidate.yaml \
  --endpoint "$ROUTER_ORIGIN"
```

可热更新的变更使用 `vllm-sr config apply`。如果计划返回 `RESTART_REQUIRED`，则通过部署流程激活。对于已授权替换的本地实例，运行 `vllm-sr serve --config candidate.yaml --replace-active-config`。测试前确认就绪状态和生效版本。

将 `ENTRYPOINT` 设为正在评测的公开入口，然后预览数据集中的一个用例：

```bash
vllm-sr route preview \
  --endpoint "$ROUTER_ORIGIN" --model "$ENTRYPOINT" \
  --prompt 'Give a brief definition of a readiness probe.' \
  --trace --json --timeout 300
```

重组已保存的信号值有助于隔离规则变更，但不属于新的 Preview 结果。应先精确重现基线的 decision 和 heuristic 命中，再用真实请求测试候选版本。

Preview 会执行已配置的分类器和 embedding，不调用后端生成。检查命中的信号、projection 结果、decision、algorithm、选择状态和错误。多轮用例需要保留完整 messages 和工具字段；CLI 无法表达请求形状时，使用已发现的 Preview HTTP schema。

分别报告策略覆盖和部署覆盖。Decision 正确但没有合格后端，说明容量或模型分配存在问题。即时响应不需要选择模型，多模型计划仍需实际执行。这些结果都不能当作成功的后端调用。

## 验证交付和应用效果 {#verify-delivery-and-the-application-result}

通过推理监听器发送相同请求：

```bash
vllm-sr route probe \
  --config candidate.yaml \
  --base-url "$INFERENCE_BASE_URL" --model "$ENTRYPOINT" \
  --prompt 'Give a brief definition of a readiness probe.' \
  --expect-recipe "$RECIPE" --expect-decision "$DECISION" \
  --expect-selected-model "$SELECTED_MODEL" \
  --expect-response-model "$RESPONSE_MODEL" \
  --timeout 300
```

期望值来自测试用例和已验证的后端响应标识。选中模型的响应头与上游响应中的 model 是两个独立断言。仅当后端不提供稳定标识时，才省略响应 model 断言。

除了路由，还要检查完整输出。HTTP 200 如果只有推理过程、空答案或被截断的响应，不算成功交付。完成预算应覆盖真实输入，并为最终答案留出空间。请求省略 token 限制时，配置了 `request_params.default_max_tokens` 就使用 decision 的默认值，否则使用后端默认值。

比较模型选择策略时，至少分配两个合格且可访问的模型。只有一个候选模型的测试可以验证交付，但无法衡量模型间的选择效果。

将冷启动与预热后的中位数、p95 时延分开测量，同时比较路由质量、回答质量、分类器计算、后端调用、token 用量和成本。多模型算法需要确认预期的不同 worker 和最终阶段确实执行。

## 测试检索、风险处理和 Agent 连续性 {#test-retrieval-risk-handling-and-agent-continuity}

**检索。** 将 [RAG 与神经重排](../tutorials/plugin/rag#neural-reranking) 配置到有真实知识库的路径。先检索较多候选，再使用 `rag.rerank` 保留最相关文档。检查文档标识、检索覆盖、排序、基于证据的回答和新增时延。Preview 选择插件，真实路由请求执行插件。先验证单模型路径，再考虑为多模型工作流的每个阶段添加检索。

**风险处理。** [Guard](../tutorials/signal/learned/jailbreak) 检测提示词攻击；[Safety 和 Hazard](../tutorials/signal/learned/safety) 识别内容风险及类别。设计拒绝规则前，需要对照测试有害协助、求助和正常分析。PII 可以选择受限的模型池，但不会自动脱敏，也不能证明供应商的数据保留策略。

**Agent 连续性。** 配合稳定的 session 和 conversation 标识使用 [Router Learning protection](../tutorials/learning/protection)。测试完整工具循环、连续追问、明确纠错、后端失败、decision 变化和新对话。对比 `apply`、`observe` 和 `bypass`：观测到的保持模型建议与真正的 hold 不同。联合检查选中的后端、路由响应头、Replay API 和 Dashboard。在不同 recipe 中复用同一个 session ID，验证隔离性。保持策略不能保留已经不符合候选要求的模型。

初始化内置 recipe 时，会开启 conversation protection，并关闭在线 adaptation，同时保留基础配置中已有的设置。protection 保持模型需要客户端提供稳定标识。在评估应用中的实际结果后再开启 adaptation。

当 protection 参与模型选择时，Preview 可能返回 `execution_required`。它会评估请求信号，但不会推进对话状态；实际是否保持或切换模型，需要通过路由请求及其 Replay 记录验证。

激活下一个候选版本前，先检查 Replay 和 Dashboard。默认的内存 Replay 存储会在配置重载和进程重启时清空，应提前保存比较所需的轨迹。

## 保留确实改善目标的变更 {#keep-changes-that-improve-the-objective}

用同一数据集运行基线和候选版本，按语言、输入长度、用例和会话阶段检查回归。候选版本改善目标且没有违反质量底线和硬约束时保留它，否则恢复基线并保存证据。

更广泛的路由工作负载可通过 `vllm-sr benchmark catalog` 发现。完整模型或虚拟模型比较，可使用 `vllm-sr benchmark intelligence list` 和 `plan --help` 了解固定测试集。虚拟模型必须通过真实端点评测，不能用成员模型分数拼出结果。部分测试有助于调优，但不能代表完整测试集得分。

[Agent 调优参考](https://vllm-sr.ai/install/agent/vllm-sr/references/recipe-tuning.md) 提供可复用的检查步骤。原始评测输出保留在 Git 之外，凭据放在 `--token-env` 或 `--api-key-env` 指定名称的环境变量中。
