---
translation:
  source_commit: "eb9f384f"
  source_file: "docs/proposals/Prism-153key.md"
  outdated: false
---

# [提案] PRISM：面向 vLLM-SR 模型选择的 153 键合法性层

**Issue：** #1422  
**作者：** Mossaab Souaissa — MSIA Systems  
**里程碑：** v0.3 — Themis  
**参考：** https://doi.org/10.5281/zenodo.18750029  
**白皮书：** https://github.com/user-attachments/files/25750911/PRISM-Vllm-SR-whitepaper-COMPLET-EN.pdf

---

## 1. 背景与动机

vLLM-SR 回答：**哪个模型最适合该请求？**  
PRISM 回答：**所选模型是否有资格回答这一具体查询？**

二者互补，而非重复。

### 「撒谎模型」问题

若无结构约束，任意模型都可回答任意查询——即便超出训练领域。这会产生**自信幻觉**：模型在不擅长的领域仍以高置信度即兴作答。

当前 vLLM-SR 流水线在模型选定后**没有**合法性校验。PRISM 在**不修改**既有路由逻辑的前提下增加该层。

### 设计原则：增量扩展、不破坏兼容

PRISM 作为现有组件的**可选扩展**集成：

- 若某模型配置中无 `prism.enabled` → 行为**不变**
- 若决策中无 `type: "prism-execution"` 插件块 → **跳过** Key 3
- 若请求时尚未就绪 PRISM 注册表 → 回退到 `"general"` → 标准 vLLM-SR 路由

| PRISM 组件 | 集成点 | 类型 |
|----------------|------------------|------|
| Key 1 — QUALIFICATION | `config.yaml` 中的 `model_config` | `prism.enabled: true` + 启动时自动发现 |
| Key 2 — CLASSIFICATION | `req_filter_classification.go` | 新求值块 — 进程内 `candle-binding` |
| Key 3 — EXECUTION | `req_filter_prism_execution.go` | 新 ExtProc 过滤器 — 沿用 `req_filter_jailbreak.go` 模式 |
| 153-Registry | `pkg/registry/prism_registry.go` | 内存存储，启动时异步填充 Key 1 |

**本 PR 范围：仅 hybrid 模式**（Key 1 + Key 2 + Key 3）。`fine_filter` 与 `coarse_filter` 见第 9 节，为后续变体。

---

## 2. 架构概览

（ASCII 流程图与英文版一致，见英文 `docs/proposals/Prism-153key.md`。）

要点：**PHASE 1** 增加 `runPrismClassification()`；**PHASE 3** 增加 `filterCandidatesByPrism()` 与重路由循环；**PHASE 4** 增加 `runPrismExecution()`。启动时 `NewPrismRegistry` 后台执行 `qualifyAllModels()`，就绪前 Key 2 将域置为 `"general"`。

---

## 3. Key 1 — QUALIFICATION（启动时异步自动发现）

### 3.1 原则

运维人员**仅**在 `model_config` 中声明 `prism.enabled: true`。

`NewPrismRegistry()` 在后台 goroutine 中对每个启用模型发送 Key 1 资格提示词，解析 JSON 自声明并写入 153-Registry。路由器**立即**启动；goroutine 完成后 PRISM 才对后续请求生效。

初始化期间：`IsReady()` 为 `false` → `runPrismClassification()` 置 `ctx.PrismDomain = "general"` → 无 PRISM 的标准路由。

### 3.2 config.yaml — 最小人类声明

（YAML 示例与英文版相同。）

### 3.3 Key 1 资格提示词

（英文提示词正文与英文版相同，用于与模型交互。）

**HTTP 调用：** `POST http://{VLLMEndpoint.Address}:{VLLMEndpoint.Port}/v1/chat/completions`  
**鉴权：** `Authorization: Bearer {ModelParams.AccessKey}`（若设置 AccessKey）  
**超时：** 每模型 30 秒  
**失败或 domain 为 "general"/"unknown"：** 该模型不进入注册表并打警告日志。

### 3.4 Go 结构体

（`PrismModelConfig`、`PrismThresholds`、`PrismConfig` 等定义与英文版代码块一致。）

### 3.5 153-Registry — 异步初始化的内存存储

（`RegistryEntry`、`PrismRegistry` 及方法与英文版一致。）

### 3.6 分数档位

| 分数 | 级别 | 含义 |
|-------|-------|---------|
| 0.00 - 0.30 | 浅显 | 通识，无专精 |
| 0.31 - 0.60 | 基础 | 有专业词汇，深度有限 |
| 0.61 - 0.90 | 确认 | 具备流程与方法的实际专精 |
| 0.91 - 1.00 | 精通 | 标准、公式与复杂案例 |

### 3.7 经验分规则

（`init` / `ACCEPTED` / `REFUSED` / `clamp` / `status` 与英文版相同。）

---

## 4. Key 2 — CLASSIFICATION（`req_filter_classification.go` 中新块）

### 4.1 原则

Key 2 是 `req_filter_classification.go` 中的新求值块，进程内使用 `candle-binding` 的 `candle.GetEmbedding()`。Key 2 **不回答问题**，只做意图分类，结果写入 `RequestContext` 供 Key 3 在 PHASE 4 使用。

### 4.2 RequestContext 扩展

（`PrismDomain`、`PrismConfidence` 等字段与英文版一致。）

### 4.3 嵌入 API

（`GetEmbedding` 签名与英文版一致。）

### 4.4 域嵌入缓存

在 `qualifyAllModels()` 结束时**预计算**各域嵌入并存入 `RegistryEntry.DomainEmbedding`，避免每请求重算。运行时 Key 2 仅计算**查询**的 1 次嵌入，与缓存的域嵌入做余弦相似度比较。

### 4.5 回退规则

注册表未就绪、`GetEmbedding` 出错、最佳相似度低于 `ConfidenceStrict`、未知域 → 一律 `ctx.PrismDomain = "general"`，不抛错、不阻塞请求。

### 4.6 置信度档位与 Key 3 行为

（与英文版表格一致：&lt;0.40 跳过 Key 3；0.40–0.69 严格档；0.70–0.89 灵活档；≥0.90 自动接受等。）

### 4.7 关键词提取

（`extractKeywords` 与英文版一致。）

---

## 5. Key 3 — EXECUTION（新 ExtProc 过滤器）

### 5.1 原则

`runPrismExecution()` 位于 `req_filter_prism_execution.go`，模式与 `req_filter_jailbreak.go` **一致**。强制所选模型按 Key 1 在启动时固定的域自证合法性，而非运行时随意声称。

### 5.2 参考模式

（`runJailbreakFilter` 与 Key 3 实现代码块与英文版一致。）

### 5.3 共享辅助

（`req_filter_prism_helpers.go` 与英文版一致。）

### 5.4 重路由循环

（`processor_req_body.go` 中循环与英文版一致。）

### 5.5 全局 REFUSED 与 HTTP 响应

`ctx.Blocked = true` 与越狱路径相同，沿用现有 `buildResponse()`。

---

## 6. filterCandidatesByPrism — 模型选择前的预过滤

（逻辑与英文版 `filterCandidatesByPrism` 一致：域为 general 或未就绪时不过滤；无 PRISM 合格候选时退回全量候选，**绝不**因 PRISM 单独阻塞标准路由。）

---

## 7. PRISM 全局配置

（YAML `prism:` 块与英文版相同；注释说明工业场景默认阈值，通用场景可降低 `confidence_strict` 与 `expertise_min_score`。）

---

## 8. 新建/修改文件清单

（文件列表与英文版第 8 节一致。）

---

## 9. 三种集成模式（后续变体）

| 模式 | 激活的 Key | 范围 |
|------|------------|------|
| `hybrid` | Key 1 + 2 + 3 | **本 PR**，合法性最强 |
| `coarse_filter` | Key 1 + 2 | 未来：仅路由前过滤 |
| `fine_filter` | Key 1 + 3 | 未来：仅路由后校验 |

---

## 10. 待确认问题（@HuaminChen、@Xunzhuo）

1. `selectModelFromCandidates` 是否包含 `decision` 第三参数。  
2. 153-Registry 仅内存是否可接受 v0.3，抑或需 Redis/SQLite。  
3. Key 2 域嵌入与 `candle-binding` 并发/CGo 一致性。  
4. 是否提供 `config.recipe-prism-general.yaml` 降低通用场景阈值。  
5. `UnregisteredPolicy` 默认 `passthrough` vs `refuse` 的产品取舍。

---

## 11. 性能说明

Key 2 每请求一次 `GetEmbedding` + 与 N 个预缓存域嵌入比对；工业场景 N 常为 3–10，开销可忽略。域数量很大时可按查询哈希做嵌入相似度缓存（后续优化）。

---

## 12. 参考

- PRISM 白皮书与 Zenodo DOI、Issue #1422、Draft PR #1425 链接与英文版相同。
