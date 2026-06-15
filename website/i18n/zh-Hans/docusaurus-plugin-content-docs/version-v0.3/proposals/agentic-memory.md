---
translation:
  source_commit: "f123732c"
  source_file: "docs/proposals/agentic-memory.md"
  outdated: false
---

# 智能体记忆（Agentic Memory）

## 执行摘要

本文描述 Semantic Router 中**智能体记忆**的**概念验证（POC）**。智能体记忆使 AI 智能体能够**跨会话记住信息**，从而提供连续性与个性化。

> **POC 范围：** 本文为概念验证，非生产级设计。目标是验证核心记忆流（检索 → 注入 → 提取 → 存储）在可接受准确度下可行。生产加固（错误处理、扩展、监控）不在范围内。

### 核心能力

| 能力 | 说明 |
|------------|-------------|
| **记忆检索** | 基于嵌入的检索与简单预过滤 |
| **记忆写入** | 基于 LLM 的事实与流程提取 |
| **跨会话持久化** | 记忆存于 Milvus（重启可保留；生产级备份/高可用未验证） |
| **用户隔离** | 按 `user_id` 划分（见下表） |

> **用户隔离与 Milvus 性能说明：**
> 
> | 方式 | POC | 生产（1 万+ 用户） |
> |----------|-----|-------------------------|
> | **简单过滤** | 检索后按 `user_id` 过滤 | 退化：先搜全库再过滤 |
> | **分区键** | POC 过重 | 物理隔离，每用户 O(log N) |
> | **标量索引** | POC 过重 | 对 `user_id` 建索引以加速过滤 |
> 
> **POC：** 使用简单元数据过滤（测试足够）。  
> **生产：** 在 Milvus schema 中将 `user_id` 配为分区键或标量索引字段。

### 关键设计原则

1. **简单预过滤** 决定是否检索记忆  
2. 利用历史 **上下文窗口** 对查询消歧  
3. **LLM 提取事实** 并在保存时分类  
4. 对检索结果做 **基于阈值的过滤**

### POC 明确假设

| 假设 | 含义 | 若错误的风险 |
|------------|-------------|---------------|
| LLM 提取基本准确 | 可能存入错误事实 | 记忆污染（可用 Forget API 修复） |
| 0.6 相似度阈值为起点 | 可能需调参 | 可依检索质量日志调整 |
| Milvus 可用且已配置 | 宕机则功能关闭 | 优雅降级（不崩溃） |
| 嵌入模型输出 384 维向量 | 须与 Milvus schema 一致 | 启动失败（可检测） |
| 可通过 Response API 链获得历史 | 上下文所需 | 无历史则跳过记忆 |

---

## 目录

1. [问题陈述](#1-problem-statement)
2. [架构概览](#2-architecture-overview)
3. [记忆类型](#3-memory-types)
4. [流水线集成](#4-pipeline-integration)
5. [记忆检索](#5-memory-retrieval)
6. [记忆写入](#6-memory-saving)
7. [记忆操作](#7-memory-operations)
8. [数据结构](#8-data-structures)
9. [API 扩展](#9-api-extension)
10. [配置](#10-configuration)
11. [失败模式与回退（POC）](#11-failure-modes-and-fallbacks-poc)
12. [成功标准（POC）](#12-success-criteria-poc)
13. [实现计划](#13-implementation-plan)
14. [后续增强](#14-future-enhancements)

---

## 1. 问题陈述 {#1-problem-statement}

### 现状

Response API 通过 `previous_response_id` 提供会话链，但**跨会话知识会丢失**：

```
Session A (March 15):
  User: "My budget for the Hawaii trip is $10,000"
  → Saved in session chain

Session B (March 20) - NEW SESSION:
  User: "What's my budget for the trip?"
  → No previous_response_id → Knowledge LOST ❌
```

### 目标状态

使用智能体记忆时：

```
Session A (March 15):
  User: "My budget for the Hawaii trip is $10,000"
  → Extracted and saved to Milvus

Session B (March 20) - NEW SESSION:
  User: "What's my budget for the trip?"
  → Pre-filter: memory-relevant ✓
  → Search Milvus → Found: "budget for Hawaii is $10K"
  → Inject into LLM context
  → Assistant: "Your budget for the Hawaii trip is $10,000!" ✅
```

---

## 2. 架构概览 {#2-architecture-overview}

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      AGENTIC MEMORY ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│                         ExtProc Pipeline                                │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │  Request → Fact? → Tool? → Security → Cache → MEMORY → LLM       │   │
│  │              │       │                          ↑↓               │   │
│  │              └───────┴──── signals used ────────┘                │   │
│  │                                                                  │   │
│  │  Response ← [extract & store] ←─────────────────┘                │   │
│  │                                                                  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                          │                              │
│                    ┌─────────────────────┴─────────────────────┐        │
│                    │                                           │        │ 
│          ┌─────────▼─────────┐                    ┌────────────▼───┐    │ 
│          │ Memory Retrieval  │                    │ Memory Saving  │    │
│          │  (request phase)  │                    │(response phase)│    │
│          ├───────────────────┤                    ├────────────────┤    │
│          │ 1. Check signals  │                    │ 1. LLM extract │    │
│          │    (Fact? Tool?)  │                    │ 2. Classify    │    │
│          │ 2. Build context  │                    │ 3. Deduplicate │    │
│          │ 3. Milvus search  │                    │ 4. Store       │    │
│          │ 4. Inject to LLM  │                    │                │    │
│          └─────────┬─────────┘                    └────────┬───────┘    │
│                    │                                       │            │
│                    │         ┌──────────────┐              │            │
│                    └────────►│    Milvus    │◄─────────────┘            │ 
│                              └──────────────┘                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 组件职责

| 组件 | 职责 | 位置 |
|-----------|---------------|----------|
| **Memory Filter** | 决策 + 检索 + 注入 | `pkg/extproc/req_filter_memory.go` |
| **Memory Extractor** | 基于 LLM 的事实提取 | `pkg/memory/extractor.go`（新建） |
| **Memory Store** | 存储接口 | `pkg/memory/store.go` |
| **Milvus Store** | 向量库后端 | `pkg/memory/milvus_store.go` |
| **Existing Classifiers** | Fact/Tool 信号（复用） | `pkg/extproc/processor_req_body.go` |

### 存储架构

[Issue #808](https://github.com/vllm-project/semantic-router/issues/808) 提出了多层存储架构。本文**分阶段**实现：

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    STORAGE ARCHITECTURE (Phased)                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  PHASE 1 (MVP)                                                  │    │
│  │  ┌─────────────────────────────────────────────────────────┐    │    │
│  │  │  Milvus (Vector Index)                                  │    │    │
│  │  │  • Semantic search over memories                        │    │    │
│  │  │  • Embedding storage                                    │    │    │
│  │  │  • Content + metadata                                   │    │    │
│  │  └─────────────────────────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  PHASE 2 (Performance)                                          │    │
│  │  ┌─────────────────────────────────────────────────────────┐    │    │
│  │  │  Redis (Hot Cache)                                      │    │    │
│  │  │  • Fast metadata lookup                                 │    │    │
│  │  │  • Recently accessed memories                           │    │    │
│  │  │  • TTL/expiration support                               │    │    │
│  │  └─────────────────────────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  PHASE 3+ (If Needed)                                           │    │
│  │  ┌───────────────────────┐  ┌───────────────────────┐           │    │
│  │  │  Graph Store (Neo4j)  │  │  Time-Series Index    │           │    │
│  │  │  • Memory links       │  │  • Temporal queries   │           │    │
│  │  │  • Relationships      │  │  • Decay scoring      │           │    │
│  │  └───────────────────────┘  └───────────────────────┘           │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

| 层 | 用途 | 何时需要 | 状态 |
|-------|---------|-------------|--------|
| **Milvus** | 语义向量检索 | 核心能力 | MVP |
| **Redis** | 热缓存、快速访问、TTL | 性能优化 | Phase 2 |
| **Graph (Neo4j)** | 记忆关联 | 多跳推理查询 | 按需 |
| **Time-Series** | 时序查询、衰减 | 按时间的重要性打分 | 按需 |

> **设计决策：** 先从仅 Milvus 开始。其余层按**实证需求**增加，而非臆测。`Store` 接口抽象存储，后续可换后端而不改检索/写入逻辑。

---

## 3. 记忆类型 {#3-memory-types}

| 类型 | 用途 | 示例 | 状态 |
|------|---------|---------|--------|
| **Semantic** | 事实、偏好、知识 | "User's budget for Hawaii is $10,000" | MVP |
| **Procedural** | 步骤、流程 | "To deploy payment-service: run npm build, then docker push" | MVP |
| **Episodic** | 会话摘要、过往事件 | "On Dec 29 2024, user planned Hawaii vacation with $10K budget" | MVP（受限） |
| **Reflective** | 自省、经验教训 | "Previous budget response was incomplete - user prefers detailed breakdowns" | 未来 |

> **情景记忆（MVP 限制）：** 未实现会话结束检测。情景记忆仅在 LLM 提取**显式**产出摘要式内容时创建。可靠的会话结束触发推迟到 Phase 2。
>
> **反思记忆：** 自省与经验教训。不在本 POC 范围内。见 [附录 A](#appendix-a-reflective-memory)。

### 记忆向量空间

记忆按**内容/主题**聚类，而非按类型。类型是元数据：

```
┌────────────────────────────────────────────────────────────────────────┐
│                      MEMORY VECTOR SPACE                               │
│                                                                        │
│     ┌─────────────────┐                    ┌─────────────────┐         │
│     │  BUDGET/MONEY   │                    │   DEPLOYMENT    │         │
│     │    CLUSTER      │                    │    CLUSTER      │         │
│     │                 │                    │                 │         │
│     │ ● budget=$10K   │                    │ ● npm build     │         │
│     │   (semantic)    │                    │   (procedural)  │         │
│     │ ● cost=$5K      │                    │ ● docker push   │         │
│     │   (semantic)    │                    │   (procedural)  │         │
│     └─────────────────┘                    └─────────────────┘         │
│                                                                        │
│  ● = memory with type as metadata                                      │
│  Query matches content → type comes from matched memory                │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

### Response API 与智能体记忆：何时有价值？

**关键区分：** 当存在 `previous_response_id` 时，Response API 已把**完整对话历史**发给 LLM。智能体记忆的价值在于**跨会话**上下文。

```
┌─────────────────────────────────────────────────────────────────────────┐
│           RESPONSE API vs. AGENTIC MEMORY: CONTEXT SOURCES              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  SAME SESSION (has previous_response_id):                               │
│  ─────────────────────────────────────────                              │
│    Response API provides:                                               │
│      └── Full conversation chain (all turns) → sent to LLM              │
│                                                                         │
│    Agentic Memory:                                                      │
│      └── STILL VALUABLE - current session may not have the answer       │
│      └── Example: 100 turns planning vacation, but budget never said    │
│      └── Days ago: "I have 10K spare, is that enough for a week in      │
│          Thailand?" → LLM extracts: "User has $10K budget for trip"     │
│      └── Now: "What's my budget?" → answer in memory, not this chain    │
│                                                                         │
│  NEW SESSION (no previous_response_id):                                 │
│  ──────────────────────────────────────                                 │
│    Response API provides:                                               │
│      └── Nothing (no chain to follow)                                   │
│                                                                         │
│    Agentic Memory:                                                      │
│      └── ADDS VALUE - retrieves cross-session context                   │
│      └── "What was my Hawaii budget?" → finds fact from March session   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

> **设计决策：** 记忆检索在**两种**场景下都有价值——新会话（无链）与已有会话（查询可能引用其他会话）。预过滤通过时**始终**检索。
>
> **已知冗余：** 若答案已在当前链中，仍会检索记忆（浪费约 10–30ms）。若不语义理解查询，无法廉价判断「答案是否已在历史中」。POC 接受该开销。
>
> **Phase 2 方案：** [上下文压缩](#context-compression-high-priority) 可正确处理——不再由 Response API 发送全量历史，而发送压缩摘要 + 最近轮次 + 相关记忆。摘要在汇总时提取事实，从而消除冗余。

---

## 4. 流水线集成 {#4-pipeline-integration}

### 当前流水线（main 分支）

```
1. Response API Translation
2. Parse Request
3. Fact-Check Classification
4. Tool Detection
5. Decision & Model Selection
6. Security Checks
7. PII Detection
8. Semantic Cache Check
9. Model Routing → LLM
```

### 集成智能体记忆后的增强流水线

```
REQUEST PHASE:
─────────────
1.  Response API Translation
2.  Parse Request
3.  Fact-Check Classification        ──┐
4.  Tool Detection                     ├── Existing signals
5.  Decision & Model Selection       ──┘
6.  Security Checks
7.  PII Detection
8.  Semantic Cache Check ───► if HIT → return cached
9.  🆕 Memory Decision: 
    └── if (NOT Fact) AND (NOT Tool) AND (NOT Greeting) → continue
    └── else → skip to step 12
10. 🆕 Build context + rewrite query          [~1-5ms]
11. 🆕 Search Milvus, inject memories         [~10-30ms]
12. Model Routing → LLM

RESPONSE PHASE:
──────────────
13. Parse LLM Response
14. Cache Update
15. 🆕 Memory Extraction (async goroutine, if auto_store enabled)
    └── Runs in background, does NOT add latency to response
16. Response API Translation
17. Return to Client
```

> **第 10 步说明：** 查询改写策略（上下文前缀、LLM 改写、HyDE）见 [附录 C](#appendix-c-query-rewriting-for-memory-search)。

---

## 5. 记忆检索 {#5-memory-retrieval}

### 流程

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      MEMORY RETRIEVAL FLOW                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. MEMORY DECISION (reuse existing pipeline signals)                   │
│     ──────────────────────────────────────────────────                  │
│                                                                         │
│     Pipeline already classified:                                        │
│     ├── ctx.IsFact       (Fact-Check classifier)                        │
│     ├── ctx.RequiresTool (Tool Detection)                               │
│     └── isGreeting(query) (simple pattern)                              │
│                                                                         │
│     Decision:                                                           │
│     ├── Fact query?     → SKIP (general knowledge)                      │
│     ├── Tool query?     → SKIP (tool provides answer)                   │
│     ├── Greeting?       → SKIP (no context needed)                      │
│     └── Otherwise       → SEARCH MEMORY                                 │
│                                                                         │
│  2. BUILD CONTEXT + REWRITE QUERY                                       │
│     ─────────────────────────────                                       │
│     History: ["Planning vacation", "Hawaii sounds nice"]                │
│     Query: "How much?"                                                  │
│                                                                         │
│     Option A (MVP): Context prepend                                     │
│     → "How much? Hawaii vacation planning"                              │
│                                                                         │
│     Option B (v1): LLM rewrite                                          │
│     → "What is the budget for the Hawaii vacation?"                     │
│                                                                         │
│  3. MILVUS SEARCH                                                       │
│     ─────────────                                                       │
│     Embed context → Search with user_id filter → Top-k results          │
│                                                                         │
│  4. THRESHOLD FILTER                                                    │
│     ────────────────                                                    │
│     Keep only results with similarity > 0.6                             │
│     ⚠️ Threshold is configurable; 0.6 is starting value, tune via logs  │
│                                                                         │
│  5. INJECT INTO LLM CONTEXT                                             │
│     ────────────────────────                                            │
│     Add as system message: "User's relevant context: ..."               │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 实现

#### MemoryFilter 结构体

```go
// pkg/extproc/req_filter_memory.go

type MemoryFilter struct {
    store memory.Store  // Interface - can be MilvusStore or InMemoryStore
}

func NewMemoryFilter(store memory.Store) *MemoryFilter {
    return &MemoryFilter{store: store}
}
```

> **说明：** `store` 为第 8 节的 `Store` 接口，而非具体实现。运行时通常为生产环境 `MilvusStore` 或测试用 `InMemoryStore`。

#### 记忆决策（复用现有流水线）

> **已知限制：** `IsFact` 分类器面向**通识**事实核查（如「法国首都是哪里？」）。可能将**个人事实**问题（如「我的预算是多少？」）误判为 fact，从而跳过记忆。
>
> **POC 缓解：** 增加**个人指代**检测。若查询含人称代词（"my", "I", "me"），则覆盖 `IsFact`，仍检索记忆。
>
> **未来：** 重训或增强 fact 分类器以区分通识与个人事实。

```go
// pkg/extproc/req_filter_memory.go

// shouldSearchMemory decides if query should trigger memory search
// Reuses existing pipeline classification signals with personal-fact override
func shouldSearchMemory(ctx *RequestContext, query string) bool {
    // Check for personal indicators (overrides IsFact for personal questions)
    hasPersonalIndicator := containsPersonalPronoun(query)
    
    // 1. Fact query → skip UNLESS it contains personal pronouns
    if ctx.IsFact && !hasPersonalIndicator {
        logging.Debug("Memory: Skipping - general fact query")
        return false
    }
    
    // 2. Tool required → skip (tool provides answer)
    if ctx.RequiresTool {
        logging.Debug("Memory: Skipping - tool query")
        return false
    }
    
    // 3. Greeting/social → skip (no context needed)
    if isGreeting(query) {
        logging.Debug("Memory: Skipping - greeting")
        return false
    }
    
    // 4. Default: search memory (conservative - don't miss context)
    return true
}

func containsPersonalPronoun(query string) bool {
    // Simple check for personal context indicators
    personalPatterns := regexp.MustCompile(`(?i)\b(my|i|me|mine|i'm|i've|i'll)\b`)
    return personalPatterns.MatchString(query)
}

func isGreeting(query string) bool {
    // Match greetings that are ONLY greetings, not "Hi, what's my budget?"
    lower := strings.ToLower(strings.TrimSpace(query))
    
    // Short greetings only (< 20 chars and matches pattern)
    if len(lower) > 20 {
        return false
    }
    
    greetings := []string{
        `^(hi|hello|hey|howdy)[\s\!\.\,]*$`,
        `^(hi|hello|hey)[\s\,]*(there)?[\s\!\.\,]*$`,
        `^(thanks|thank you|thx)[\s\!\.\,]*$`,
        `^(bye|goodbye|see you)[\s\!\.\,]*$`,
        `^(ok|okay|sure|yes|no)[\s\!\.\,]*$`,
    }
    for _, p := range greetings {
        if regexp.MustCompile(p).MatchString(lower) {
            return true
        }
    }
    return false
}
```

#### 上下文构建

```go
// buildSearchQuery builds an effective search query from history + current query
// MVP: context prepend, v1: LLM rewrite for vague queries
func buildSearchQuery(history []Message, query string) string {
    // If query is self-contained, use as-is
    if isSelfContained(query) {
        return query
    }
    
    // MVP: Simple context prepend
    context := summarizeHistory(history)
    return query + " " + context
    
    // v1 (future): LLM rewrite for vague queries
    // if isVague(query) {
    //     return rewriteWithLLM(history, query)
    // }
}

func isSelfContained(query string) bool {
    // Self-contained: "What's my budget for the Hawaii trip?"
    // NOT self-contained: "How much?", "And that one?", "What about it?"
    
    vaguePatterns := []string{`^how much\??$`, `^what about`, `^and that`, `^this one`}
    for _, p := range vaguePatterns {
        if regexp.MustCompile(`(?i)`+p).MatchString(query) {
            return false
        }
    }
    return len(query) > 20 // Short queries are often vague
}

func summarizeHistory(history []Message) string {
    // Extract key terms from last 3 user messages
    var terms []string
    count := 0
    for i := len(history) - 1; i >= 0 && count < 3; i-- {
        if history[i].Role == "user" {
            terms = append(terms, extractKeyTerms(history[i].Content))
            count++
        }
    }
    return strings.Join(terms, " ")
}

// v1: LLM-based query rewriting (future enhancement)
func rewriteWithLLM(history []Message, query string) string {
    prompt := fmt.Sprintf(`Conversation context: %s
    
Rewrite this vague query to be self-contained: "%s"
Return ONLY the rewritten query.`, summarizeHistory(history), query)
    
    // Call LLM endpoint
    resp, _ := http.Post(llmEndpoint+"/v1/chat/completions", ...)
    return parseResponse(resp)
    // "how much?" → "What is the budget for the Hawaii vacation?"
}
```

#### 完整检索

```go
// pkg/extproc/req_filter_memory.go

func (f *MemoryFilter) RetrieveMemories(
    ctx context.Context,
    query string,
    userID string,
    history []Message,
) ([]*memory.RetrieveResult, error) {
    
    // 1. Memory decision (skip if fact/tool/greeting)
    if !shouldSearchMemory(ctx, query) {
        logging.Debug("Memory: Skipping - not memory-relevant")
        return nil, nil
    }
    
    // 2. Build search query (context prepend or LLM rewrite)
    searchQuery := buildSearchQuery(history, query)
    
    // 3. Search Milvus
    results, err := f.store.Retrieve(ctx, memory.RetrieveOptions{
        Query:     searchQuery,
        UserID:    userID,
        Limit:     5,
        Threshold: 0.6,
    })
    if err != nil {
        return nil, err
    }
    
    logging.Infof("Memory: Retrieved %d memories", len(results))
    return results, nil
}

// InjectMemories adds memories to the LLM request
func (f *MemoryFilter) InjectMemories(
    requestBody []byte,
    memories []*memory.RetrieveResult,
) ([]byte, error) {
    if len(memories) == 0 {
        return requestBody, nil
    }
    
    // Format memories as context
    var sb strings.Builder
    sb.WriteString("## User's Relevant Context\n\n")
    for _, mem := range memories {
        sb.WriteString(fmt.Sprintf("- %s\n", mem.Memory.Content))
    }
    
    // Add as system message
    return injectSystemMessage(requestBody, sb.String())
}
```

---

## 6. 记忆写入 {#6-memory-saving}

### 触发条件

记忆提取由三类事件触发：

| 触发 | 说明 | 状态 |
|---------|-------------|--------|
| **每 N 轮** | 每 10 轮提取一次 | MVP |
| **会话结束** | 会话结束时生成情景摘要 | 未来 |
| **上下文漂移** | 主题显著变化时提取 | 未来 |

> **说明：** 会话结束与漂移检测需额外实现。MVP 仅依赖「每 N 轮」触发。

### 流程

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       MEMORY SAVING FLOW                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  TRIGGERS:                                                              │
│  ─────────                                                              │
│  ├── Every N turns (e.g., 10)      ← MVP                                │
│  ├── End of session                ← Future (needs detection)           │
│  └── Context drift detected        ← Future (needs detection)           │
│                                                                         │
│  Runs: Async (background) - no user latency                             │
│                                                                         │
│  1. GET BATCH                                                           │
│     ─────────                                                           │
│     Get last 10-15 turns from session                                   │
│                                                                         │
│  2. LLM EXTRACTION                                                      │
│     ──────────────                                                      │
│     Prompt: "Extract important facts. Include context.                  │
│              Return JSON: [{type, content}, ...]"                       │
│                                                                         │
│     LLM returns:                                                        │
│       [{"type": "semantic", "content": "budget for Hawaii is $10K"}]    │
│                                                                         │
│  3. DEDUPLICATION                                                       │
│     ─────────────                                                       │
│     For each extracted fact:                                            │
│     - Embed content                                                     │
│     - Search existing memories (same user, same type)                   │
│     - If similarity > 0.9: UPDATE existing (merge/replace)              │
│     - If similarity 0.7-0.9: CREATE new (gray zone, conservative)       │
│     - If similarity < 0.7: CREATE new                                   │
│                                                                         │
│     Example:                                                            │
│       Existing: "User's budget for Hawaii is $10,000"                   │
│       New:      "User's budget is now $15,000"                          │
│       → Similarity ~0.92 → UPDATE existing with new value               │
│                                                                         │
│  4. STORE IN MILVUS                                                     │
│     ───────────────                                                     │
│     Memory { id, type, content, embedding, user_id, created_at }        │
│                                                                         │
│  5. SESSION END (future): Create episodic summary                       │
│     ─────────────────────────────────────────────                       │
│     "On Dec 29, user planned Hawaii vacation with $10K budget"          │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

> **关于 `user_id`：** 此处指**已登录用户**（经认证的身份），而非当前会话中的匿名会话用户；具体映射需在 semantic router agent 侧配置。

### 实现

```go
// pkg/memory/extractor.go

type MemoryExtractor struct {
    store       memory.Store  // Interface - can be MilvusStore or InMemoryStore
    llmEndpoint string        // LLM endpoint for fact extraction
    batchSize   int           // Extract every N turns (default: 10)
    turnCounts  map[string]int
    mu          sync.Mutex
}

// ProcessResponse extracts and stores memories (runs async)
// 
// Triggers (MVP: only first one implemented):
//   - Every N turns (e.g., 10)       ← MVP
//   - End of session                 ← Future: needs session end detection
//   - Context drift detected         ← Future: needs drift detection
//
func (e *MemoryExtractor) ProcessResponse(
    ctx context.Context,
    sessionID string,
    userID string,
    history []Message,
) error {
    e.mu.Lock()
    e.turnCounts[sessionID]++
    turnCount := e.turnCounts[sessionID]
    e.mu.Unlock()
    
    // MVP: Only extract every N turns
    // Future: Also trigger on session end or context drift
    if turnCount % e.batchSize != 0 {
        return nil
    }
    
    // Get recent batch
    batchStart := max(0, len(history) - e.batchSize - 5)
    batch := history[batchStart:]
    
    // LLM extraction
    extracted, err := e.extractWithLLM(batch)
    if err != nil {
        return err
    }
    
    // Store with deduplication
    for _, fact := range extracted {
        existing, similarity := e.findSimilar(ctx, userID, fact.Content, fact.Type)
        
        if similarity > 0.9 && existing != nil {
            // Very similar → UPDATE existing memory
            existing.Content = fact.Content  // Use newer content
            existing.UpdatedAt = time.Now()
            if err := e.store.Update(ctx, existing.ID, existing); err != nil {
                logging.Warnf("Failed to update memory: %v", err)
            }
            continue
        }
        
        // similarity < 0.9 → CREATE new memory
        mem := &Memory{
            ID:        generateID("mem"),
            Type:      fact.Type,
            Content:   fact.Content,
            UserID:    userID,
            Source:    "conversation",
            CreatedAt: time.Now(),
        }
        
        if err := e.store.Store(ctx, mem); err != nil {
            logging.Warnf("Failed to store memory: %v", err)
        }
    }
    
    return nil
}

// findSimilar searches for existing similar memories
func (e *MemoryExtractor) findSimilar(
    ctx context.Context,
    userID string,
    content string,
    memType MemoryType,
) (*Memory, float32) {
    results, err := e.store.Retrieve(ctx, memory.RetrieveOptions{
        Query:     content,
        UserID:    userID,
        Types:     []MemoryType{memType},
        Limit:     1,
        Threshold: 0.7,  // Only consider reasonably similar
    })
    if err != nil || len(results) == 0 {
        return nil, 0
    }
    return results[0].Memory, results[0].Score
}

// extractWithLLM uses LLM to extract facts
// 
// ⚠️ POC Limitation: LLM extraction is best-effort. Failures are logged but do not
// block the response. Incorrect extractions may occur.
//
// Future: Self-correcting memory (see Section 14 - Future Enhancements):
//   - Track memory usage (access_count, last_accessed)
//   - Score memories based on usage + age + retrieval feedback
//   - Periodically prune low-score, unused memories
//   - Detect contradictions → auto-merge or flag for resolution
//
func (e *MemoryExtractor) extractWithLLM(messages []Message) ([]ExtractedFact, error) {
    prompt := `Extract important information from these messages.

IMPORTANT: Include CONTEXT for each fact.

For each piece of information:
- Type: "semantic" (facts, preferences) or "procedural" (instructions, how-to)
- Content: The fact WITH its context

BAD:  {"type": "semantic", "content": "budget is $10,000"}
GOOD: {"type": "semantic", "content": "budget for Hawaii vacation is $10,000"}

Messages:
` + formatMessages(messages) + `

Return JSON array (empty if nothing to remember):
[{"type": "semantic|procedural", "content": "fact with context"}]`

    // Call LLM with timeout
    ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
    defer cancel()
    
    reqBody := map[string]interface{}{
        "model": "qwen3",
        "messages": []map[string]string{
            {"role": "user", "content": prompt},
        },
    }
    jsonBody, _ := json.Marshal(reqBody)
    
    req, _ := http.NewRequestWithContext(ctx, "POST",
        e.llmEndpoint+"/v1/chat/completions",
        bytes.NewReader(jsonBody))
    req.Header.Set("Content-Type", "application/json")
    
    resp, err := http.DefaultClient.Do(req)
    if err != nil {
        logging.Warnf("Memory extraction LLM call failed: %v", err)
        return nil, err  // Caller handles gracefully
    }
    defer resp.Body.Close()
    
    if resp.StatusCode != 200 {
        logging.Warnf("Memory extraction LLM returned %d", resp.StatusCode)
        return nil, fmt.Errorf("LLM returned %d", resp.StatusCode)
    }
    
    facts, err := parseExtractedFacts(resp.Body)
    if err != nil {
        // JSON parse error - LLM returned malformed output
        logging.Warnf("Memory extraction parse failed: %v", err)
        return nil, err  // Skip this batch, don't store garbage
    }
    
    return facts, nil
}
```

---

## 7. 记忆操作 {#7-memory-operations}

可对记忆执行的全部操作，由 `Store` 接口实现（见 [第 8 节](#8-data-structures)）。

| 操作 | 说明 | 触发 | 接口方法 | 状态 |
|-----------|-------------|---------|------------------|--------|
| **Store** | 将新记忆写入 Milvus | 自动（LLM 提取）或显式 API | `Store()` | MVP |
| **Retrieve** | 语义检索相关记忆 | 查询时自动 | `Retrieve()` | MVP |
| **Update** | 修改已有记忆内容 | 去重或显式 API | `Update()` | MVP |
| **Forget** | 按 ID 删除单条记忆 | 显式 API | `Forget()` | MVP |
| **ForgetByScope** | 按用户/项目删除全部 | 显式 API | `ForgetByScope()` | MVP |
| **Consolidate** | 合并相关记忆为摘要 | 定时/达阈值 | `Consolidate()` | 未来 |
| **Reflect** | 从记忆模式生成洞察 | 智能体发起 | `Reflect()` | 未来 |

### Forget 操作

```go
// Forget single memory
DELETE /v1/memory/{memory_id}

// Forget all memories for a user
DELETE /v1/memory?user_id=user_123

// Forget all memories for a project
DELETE /v1/memory?user_id=user_123&project_id=project_abc
```

**用例：**

- 用户要求「忘掉关于 X 的内容」
- GDPR/隐私合规（被遗忘权）
- 清除过时信息

### 未来：Consolidate

将多条相关记忆合并为一条摘要：

```
Before:
  - "Budget for Hawaii is $10,000"
  - "Added $2,000 to Hawaii budget"
  - "Final Hawaii budget is $12,000"

After consolidation:
  - "Hawaii trip budget: $12,000 (updated from initial $10,000)"
```

**触发方式：** 记忆条数超阈值、定时后台任务、会话结束。

### 未来：Reflect

通过分析记忆模式生成洞察：

```
Input: All memories for user_123 about "deployment"

Output (Insight):
  - "User frequently deploys payment-service (12 times)"
  - "Common issue: port conflicts"
  - "Preferred approach: docker-compose"
```

**用例：** 智能体可基于模式主动提供帮助。

---

## 8. 数据结构 {#8-data-structures}

### Memory

```go
// pkg/memory/types.go

type MemoryType string

const (
    MemoryTypeEpisodic   MemoryType = "episodic"
    MemoryTypeSemantic   MemoryType = "semantic"
    MemoryTypeProcedural MemoryType = "procedural"
)

type Memory struct {
    ID          string         `json:"id"`
    Type        MemoryType     `json:"type"`
    Content     string         `json:"content"`
    Embedding   []float32      `json:"-"`
    UserID      string         `json:"user_id"`
    ProjectID   string         `json:"project_id,omitempty"`
    Source      string         `json:"source,omitempty"`
    CreatedAt   time.Time      `json:"created_at"`
    AccessCount int            `json:"access_count"`
    Importance  float32        `json:"importance"`
}
```

### Store 接口

```go
// pkg/memory/store.go

type Store interface {
    // MVP Operations
    Store(ctx context.Context, memory *Memory) error                         // Save new memory
    Retrieve(ctx context.Context, opts RetrieveOptions) ([]*RetrieveResult, error) // Semantic search
    Get(ctx context.Context, id string) (*Memory, error)                     // Get by ID
    Update(ctx context.Context, id string, memory *Memory) error             // Modify existing
    Forget(ctx context.Context, id string) error                             // Delete by ID
    ForgetByScope(ctx context.Context, scope MemoryScope) error              // Delete by scope
    
    // Utility
    IsEnabled() bool
    Close() error
    
    // Future Operations (not yet implemented)
    // Consolidate(ctx context.Context, memoryIDs []string) (*Memory, error)  // Merge memories
    // Reflect(ctx context.Context, scope MemoryScope) ([]*Insight, error)    // Generate insights
}
```

---

## 9. API 扩展 {#9-api-extension}

### 请求（已有）

```go
// pkg/responseapi/types.go

type ResponseAPIRequest struct {
    // ... existing fields ...
    MemoryConfig  *MemoryConfig  `json:"memory_config,omitempty"`
    MemoryContext *MemoryContext `json:"memory_context,omitempty"`
}

type MemoryConfig struct {
    Enabled             bool     `json:"enabled"`
    MemoryTypes         []string `json:"memory_types,omitempty"`
    RetrievalLimit      int      `json:"retrieval_limit,omitempty"`
    SimilarityThreshold float32  `json:"similarity_threshold,omitempty"`
    AutoStore           bool     `json:"auto_store,omitempty"`
}

type MemoryContext struct {
    UserID    string `json:"user_id"`
    ProjectID string `json:"project_id,omitempty"`
}
```

### 请求示例

```json
{
    "model": "qwen3",
    "input": "What's my budget for the trip?",
    "previous_response_id": "resp_abc123",
    "memory_config": {
        "enabled": true,
        "auto_store": true
    },
    "memory_context": {
        "user_id": "user_456"
    }
}
```

---

## 10. 配置 {#10-configuration}

```yaml
# config.yaml
memory:
  enabled: true
  auto_store: true  # Enable automatic fact extraction
  
  milvus:
    address: "milvus:19530"
    collection: "agentic_memory"
    dimension: 384             # Must match embedding model output
  
  # Embedding model for memory
  embedding:
    model: "all-MiniLM-L6-v2"   # 384-dim, optimized for semantic similarity
    dimension: 384
  
  # Retrieval settings
  default_retrieval_limit: 5
  default_similarity_threshold: 0.6   # Tunable; start conservative
  
  # Extraction runs every N conversation turns
  extraction_batch_size: 10

# External models for memory LLM features
# Query rewriting and fact extraction are enabled by adding external_models
external_models:
  - llm_provider: "vllm"
    model_role: "memory_rewrite"      # Enables query rewriting
    llm_endpoint:
      address: "qwen"
      port: 8000
    llm_model_name: "qwen3"
    llm_timeout_seconds: 30
    max_tokens: 100
    temperature: 0.1
  - llm_provider: "vllm"
    model_role: "memory_extraction"   # Enables fact extraction
    llm_endpoint:
      address: "qwen"
      port: 8000
    llm_model_name: "qwen3"
    llm_timeout_seconds: 30
    max_tokens: 500
    temperature: 0.1
```

### 配置说明

| 参数 | 取值 | 理由 |
|-----------|-------|-----------|
| `dimension: 384` | 固定 | 须与 all-MiniLM-L6-v2 输出一致 |
| `default_similarity_threshold: 0.6` | 起始值 | 依检索质量日志调参 |
| `extraction_batch_size: 10` | 默认 | 新鲜度与 LLM 成本平衡 |
| `llm_timeout_seconds: 30` | 默认 | 防止提取无限阻塞 |

> **嵌入模型选择：**
> 
> | 模型 | 维度 | 优点 | 缺点 |
> |-------|-----------|------|------|
> | **all-MiniLM-L6-v2**（POC 选用） | 384 | 语义相似度更好、措辞容错高，适合记忆检索与去重 | 需单独加载模型 |
> | Qwen3-Embedding-0.6B（已有） | 1024 | 语义缓存已加载，无额外内存 | 对措辞更敏感，可能漏检相近记忆 |
>
> **记忆为何用 384 维？** 较低维度更能捕获高层语义，对具体数字、姓名等细节不敏感，有利于：
>
> - **检索**：「我的预算是多少？」与「夏威夷行程预算是 $10K」即使措辞不同也能匹配
> - **去重**：「预算 $10K」与「预算现为 $15K」可识别为同一主题（更新数值）
> - **跨会话**：不同会话间措辞自然不同
>
> **替代方案：** 可复用 Qwen3-Embedding（1024 维）以避免加载第二模型；代价是匹配略严，可能增加假阴性。

---

## 11. 失败模式与回退（POC） {#11-failure-modes-and-fallbacks-poc}

本节明确记录各组件失败时的行为。POC 范围内优先**优雅降级**，而非复杂恢复。

| 失败 | 检测 | 行为 | 日志 |
|---------|-----------|----------|---------|
| **Milvus 不可用** | Store 初始化连接错误 | 本会话关闭记忆功能 | `ERROR: Milvus unavailable, memory disabled` |
| **Milvus 检索超时** | 上下文 deadline 超时 | 跳过记忆注入，无记忆继续 | `WARN: Memory search timeout, skipping` |
| **嵌入生成失败** | candle-binding 报错 | 本请求跳过记忆 | `WARN: Embedding failed, skipping memory` |
| **LLM 提取失败** | HTTP 错误或超时 | 跳过提取，不保存记忆 | `WARN: Extraction failed, batch skipped` |
| **LLM 返回非法 JSON** | 解析错误 | 跳过提取，不保存记忆 | `WARN: Extraction parse failed` |
| **无历史** | `ctx.ConversationHistory` 为空 | 仅用查询检索（无上下文前缀） | `DEBUG: No history, query-only search` |
| **阈值过高** | 返回 0 条 | 不注入记忆 | `DEBUG: No memories above threshold` |
| **阈值过低** | 大量无关结果 | 噪声上下文（POC 可接受） | `DEBUG: Retrieved N memories` |

### 优雅降级原则

> **即使记忆失败，请求也必须成功。** 记忆是增强能力，不是硬依赖。所有记忆操作均包在错误处理中：记录日志并继续。

```go
// Example: Memory retrieval with fallback
memories, err := memoryFilter.RetrieveMemories(ctx, query, userID, history)
if err != nil {
    logging.Warnf("Memory retrieval failed: %v", err)
    memories = nil  // Continue without memories
}
// Proceed with request (memories may be nil/empty)
```

---

## 12. 成功标准（POC） {#12-success-criteria-poc}

### 功能标准

| 标准 | 如何验证 | 通过条件 |
|-----------|-----------------|----------------|
| 跨会话检索 | 会话 A 存事实，会话 B 查询 | 事实被检索并注入 |
| 用户隔离 | 用户 A 存事实，用户 B 查询 | 用户 B 看不到用户 A 的事实 |
| 优雅降级 | 停止 Milvus 后发请求 | 请求成功（无记忆） |
| 提取执行 | 对话后查日志 | 出现 `Memory: Stored N facts` |

### 质量标准（POC 后测量）

| 指标 | 目标 | 测量方式 |
|--------|--------|----------------|
| 检索相关性 | 多数注入记忆相关 | 人工抽查 50 条 |
| 提取准确性 | 多数提取事实正确 | 人工抽查 50 条 |
| 延迟影响 | P50 增加 &lt;50ms | 开关记忆对比 |

> **POC 范围：** 仅验证功能标准。质量指标在 POC 后测量，用于调阈值与改进提取提示词。

---

## 13. 实现计划 {#13-implementation-plan}

### Phase 1：检索

| 任务 | 文件 |
|------|-------|
| 记忆决策（复用 Fact/Tool 信号） | `pkg/extproc/req_filter_memory.go` |
| 从历史构建上下文 | `pkg/extproc/req_filter_memory.go` |
| Milvus 检索 + 阈值过滤 | `pkg/memory/milvus_store.go` |
| 向请求注入记忆 | `pkg/extproc/req_filter_memory.go` |
| 接入请求阶段 | `pkg/extproc/processor_req_body.go` |

### Phase 2：写入

| 任务 | 文件 |
|------|-------|
| 创建 MemoryExtractor | `pkg/memory/extractor.go` |
| 基于 LLM 的事实提取 | `pkg/memory/extractor.go` |
| 去重逻辑 | `pkg/memory/extractor.go` |
| 接入响应阶段（异步） | `pkg/extproc/processor_res_body.go` |

### Phase 3：测试与调参

| 任务 | 说明 |
|------|-------------|
| 单元测试 | 记忆决策、提取、检索 |
| 集成测试 | 端到端流程 |
| 阈值调参 | 依结果调整相似度阈值 |

---

## 14. 后续增强 {#14-future-enhancements}

### 上下文压缩（高优先级） {#context-compression-high-priority}

**问题：** Response API 当前向 LLM 发送**全部**对话历史。200 轮会话意味着每次请求数千 token——成本高且可能触顶上下文。

**方案：** 用两类输出替换旧消息：

| 输出 | 用途 | 存储 | 替代 |
|--------|---------|---------|----------|
| **Facts** | 长期记忆 | Milvus | （第 6 节已有） |
| **Current state** | 会话上下文 | Redis | 旧消息 |

> **要点：** 当前状态应为**结构化**（非散文摘要），便于知识图谱：
>
> ```json
> {"topic": "Hawaii vacation", "budget": "$10K", "decisions": ["fly direct"], "open": ["which hotel?"]}
> ```

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    CONTEXT COMPRESSION FLOW                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  BACKGROUND (every 10 turns):                                           │
│    1. Extract facts (reuse Section 6) → save to Milvus                  │
│    2. Build current state (structured JSON) → save to Redis             │
│                                                                         │
│  ON REQUEST (turn N):                                                   │
│    Context = [current state from Redis]   ← replaces old messages       │
│            + [raw last 5 turns]           ← recent context              │
│            + [relevant memories]          ← cross-session (Milvus)      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**实现改动：**

| 文件 | 修改 |
|------|--------|
| `pkg/responseapi/translator.go` | 用当前状态 + 最近若干轮取代全量历史 |
| `pkg/responseapi/context_manager.go` | 新建：管理当前状态 |
| Redis 配置 | 当前状态带 TTL 存储 |

**LLM 收到的上下文（替代全量历史）：**

```
Context sent to LLM:
  1. Current state (structured JSON from Redis)  ~100 tokens
  2. Last 5 raw messages                         ~400 tokens
  3. Relevant memories from Milvus               ~150 tokens
  ─────────────────────────────────────────────
  Total: ~650 tokens (vs 10K for full history)
```

**与智能体记忆的协同：**

- 第 6 节的事实提取在压缩过程中运行 → 写入 Milvus
- 当前状态替代旧消息 → 减少 token
- 结构化格式 → 便于未来接 KG

**收益：**

- Token 可控（成本可预测）
- 上下文质量更好（结构化状态优于全量历史）
- **KG-ready**：结构化状态可直接映射图节点/边
- 可支撑超长会话（1000+ 轮）

---

### 写入触发

| 能力 | 说明 | 做法 |
|---------|-------------|----------|
| **会话结束检测** | 会话结束时触发提取 | 超时 / 显式信号 / API |
| **上下文漂移检测** | 主题显著变化时触发 | 轮次间嵌入相似度 |

### 存储层

| 能力 | 说明 | 优先级 |
|---------|-------------|----------|
| **Redis 热缓存** | Milvus 前的快速访问层 | 高 |
| **TTL 与过期** | 自动删除旧记忆（Redis 原生） | 高 |

### 高级能力

| 能力 | 说明 | 优先级 |
|---------|-------------|----------|
| **自纠正记忆** | 跟踪使用、按访问/年龄打分、自动剪枝低分记忆 | 高 |
| **矛盾检测** | 检测冲突事实、自动合并或标记 | 高 |
| **按类型路由检索** | 按 semantic/procedural/episodic 检索 | 中 |
| **每用户配额** | 限制存储上限 | 中 |
| **图存储** | 多跳查询的记忆关系 | 按需 |
| **时序索引** | 时序查询与衰减打分 | 按需 |
| **并发处理** | 同用户多会话加锁 | 中 |

### 已知 POC 限制（明确推迟）

| 限制 | 影响 | 为何可接受 |
|------------|--------|----------------|
| **无并发控制** | 同用户多会话竞态 | POC 少见；生产修复 |
| **无记忆上限** | 重度用户可能无限积累 | Phase 3 加配额 |
| **未测备份恢复** | Milvus 盘故障可能丢数据 | 基础持久化可用；生产验证 HA |
| **无智能更新** | 更正可能产生重复 | 最新优先；Forget API 可用 |
| **无对抗防御** | 提示注入可能污染记忆 | POC 信任输入；后续过滤 |

---

---

## 附录

### 附录 A：反思记忆 {#appendix-a-reflective-memory}

**状态：** 未来扩展 — 不在本 POC 范围内。

基于过往交互的自省与经验教训。参见 [Reflexion 论文](https://arxiv.org/abs/2303.11366)。

**存储内容：**

- 错误或不优回答带来的洞察
- 对回答风格的偏好学习
- 改善后续交互的模式

**示例：**

- 「上次部署步骤有误——下次应先核对 k8s 版本」
- 「用户偏好技术内容用要点列表而非长段落」
- 「预算类问题应给分项，而非只给总额」

**为何未来再做：** 需要评估回答质量并生成自反思，依赖核心记忆基础设施之上。

---

### 附录 B：文件树

```
pkg/
├── extproc/
│   ├── processor_req_body.go     (EXTEND) Integrate retrieval
│   ├── processor_res_body.go     (EXTEND) Integrate extraction
│   └── req_filter_memory.go      (EXTEND) Pre-filter, retrieval, injection
│
├── memory/
│   ├── extractor.go              (NEW) LLM-based fact extraction
│   ├── store.go                  (existing) Store interface
│   ├── milvus_store.go           (existing) Milvus implementation
│   └── types.go                  (existing) Memory types
│
├── responseapi/
│   └── types.go                  (existing) MemoryConfig, MemoryContext
│
└── config/
    └── config.go                 (EXTEND) Add extraction config
```

---

### 附录 C：记忆检索的查询改写 {#appendix-c-query-rewriting-for-memory-search}

检索记忆时，「how much?」等模糊查询需要上下文才有效。本附录说明查询改写策略。

#### 问题

```
History: ["Planning Hawaii vacation", "Looking at hotels"]
Query: "How much?"
→ Direct search for "How much?" won't find "Hawaii budget is $10,000"
```

#### 方案 1：上下文前缀（MVP）

简单拼接，无 LLM 调用，延迟约 0ms。

```go
func buildSearchQuery(history []Message, query string) string {
    context := extractKeyTerms(history)  // "Hawaii vacation planning"
    return query + " " + context         // "How much? Hawaii vacation planning"
}
```

**优点：** 快、简单  
**缺点：** 可能混入无关词

#### 方案 2：LLM 查询改写

用 LLM 将查询改写为自包含问句，延迟约 100–200ms。

```go
func rewriteQuery(history []Message, query string) string {
    prompt := `Given conversation about: %s
    Rewrite this query to be self-contained: "%s"
    Return ONLY the rewritten query.`
    return llm.Complete(fmt.Sprintf(prompt, summarize(history), query))
}
// "How much?" → "What is the budget for the Hawaii vacation?"
```

**优点：** 自然、嵌入匹配更好  
**缺点：** LLM 延迟与成本

#### 方案 3：HyDE（假设文档嵌入）

生成假设答案，对其嵌入而非对查询嵌入。

**HyDE 解决的问题：**

```
Query: "What's the cost?"           → embeds as QUESTION style
Stored: "Budget is $10,000"         → embeds as STATEMENT style
Result: Low similarity (style mismatch)

With HyDE:
Query → LLM generates: "The cost is approximately $10,000"
This embeds as STATEMENT style → matches stored memory!
```

```go
func hydeRewrite(query string, history []Message) string {
    prompt := `Based on this conversation: %s
    Write a short factual answer to: "%s"`
    return llm.Complete(fmt.Sprintf(prompt, summarize(history), query))
}
// "How much?" → "The budget for the Hawaii trip is approximately $10,000"
```

**优点：** 检索质量最好（弥合问句/文档风格差异）  
**缺点：** 延迟最高（~200ms）、LLM 成本

#### 建议

| 阶段 | 方案 | 适用 |
|-------|----------|----------|
| **MVP** | 上下文前缀 | 默认全部查询 |
| **v1** | LLM 改写 | 模糊查询（"how much?", "and that?"） |
| **v2** | HyDE | **观测到**问句风格检索分数低之后 |

> **说明：** HyDE 是依据线上表现做的优化，不是预测。当你发现相关记忆存在但检索不到时再上。

#### 参考文献

**Query Rewriting:**

1. **HyDE** - [Precise Zero-Shot Dense Retrieval without Relevance Labels](https://arxiv.org/abs/2212.10496) (Gao et al., 2022) - Style bridging (question → document style)
2. **RRR** - [Query Rewriting for Retrieval-Augmented LLMs](https://arxiv.org/abs/2305.14283) (Ma et al., 2023) - Trainable rewriter with RL, handles conversational context

**Agentic Memory (from [Issue #808](https://github.com/vllm-project/semantic-router/issues/808)):**

5. **MemGPT** - [Towards LLMs as Operating Systems](https://arxiv.org/abs/2310.08560) (Packer et al., 2023)
6. **Generative Agents** - [Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442) (Park et al., 2023)
7. **Reflexion** - [Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366) (Shinn et al., 2023)
8. **Voyager** - [An Open-Ended Embodied Agent with LLMs](https://arxiv.org/abs/2305.16291) (Wang et al., 2023)

---

*文档作者：[Yehudit Kerido, Marina Koushnir]*  
*最后更新：2025 年 12 月*  
*状态：POC 设计 - v3（已评审修订）*  
*基于：[Issue #808 - Explore Agentic Memory in Response API](https://github.com/vllm-project/semantic-router/issues/808)*
