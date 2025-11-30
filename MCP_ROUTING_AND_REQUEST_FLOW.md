# Understanding MCP Routing and Request Flow in Semantic Router

## Overview

This document explains:
1. What MCP (Model Context Protocol) routing is and its purpose
2. How `mcp_classifier.go` implements MCP classification
3. The complete request flow from client query to LLM response

---

## Part 1: MCP Routing Overview

### What is MCP (Model Context Protocol)?

**Model Context Protocol (MCP)** is an open protocol that allows the semantic router to externalize its classification logic to remote services. Instead of using built-in classification models (like ModernBERT or candle-based classifiers), the router can delegate classification decisions to external MCP servers.

### Key Architecture

```
User Query
    ↓
Semantic Router (Go)
    ↓
MCP Classifier (mcp_classifier.go)
    ↓
MCP Client (stdio or HTTP transport)
    ↓
External MCP Server (Python/Any Language)
    ↓
Classification Logic (regex/embeddings/generative model)
    ↓
Returns: {class, confidence, model, use_reasoning, probabilities}
```

### Purpose of MCP Routing

1. **Flexible Classification Logic**
   - Use any classification approach: regex, ML models, embeddings, generative models
   - Swap classification logic without modifying router code
   - Experiment with different strategies easily

2. **Dynamic Category Discovery**
   - Categories loaded at runtime from MCP server
   - No rebuild/restart needed to add categories
   - Supports hot-reloading

3. **Intelligent Routing Decisions**
   - **Model selection**: Which LLM to use (e.g., "gpt-oss-20b" vs "deepseek-coder")
   - **Reasoning control**: Enable/disable chain-of-thought for complex queries
   - **Confidence-based fallback**: Low confidence → higher-quality models

4. **Per-Category System Prompts**
   - Math: "You are a mathematics expert. Show step-by-step solutions..."
   - Code: "You are a coding expert. Include practical examples..."
   - Each category gets specialized prompts

5. **Scalability & Distribution**
   - Classification runs on separate servers
   - Independent scaling
   - Load balancing across multiple MCP instances

### MCP vs Built-in Classification

| Feature | Built-in (ModernBERT) | MCP Classification |
|---------|----------------------|-------------------|
| **Location** | Embedded in router | External service |
| **Flexibility** | Fixed at compile time | Swappable at runtime |
| **Categories** | Static configuration | Dynamic discovery |
| **Latency** | Very low (~10ms) | Higher (~50-200ms) |
| **Routing Logic** | Fixed in config | MCP server decides |
| **System Prompts** | Config file | Per-category from MCP |
| **Updates** | Require restart | Hot-reload possible |

---

## Part 2: MCP Classifier Implementation

### File: `src/semantic-router/pkg/classification/mcp_classifier.go`

#### Key Components

**1. MCPCategoryClassifier Struct** (lines 65-69)
```go
type MCPCategoryClassifier struct {
    client   mcpclient.MCPClient  // MCP client (HTTP or stdio)
    toolName string               // Classification tool name
    config   *config.RouterConfig // Router configuration
}
```

**2. Initialization** (lines 72-123)
- Creates MCP client with configured transport (HTTP/stdio)
- Connects to MCP server
- Auto-discovers classification tools:
  - Searches for: `classify_text`, `classify`, `categorize`, `categorize_text`
  - Or tools with "classif" in name/description
- Falls back to explicitly configured `tool_name`

**3. Classification Methods**

**a) Classify** (lines 186-231)
- Basic classification without probabilities
- Input: text string
- Output: `{class: int, confidence: float64}`

**b) ClassifyWithProbabilities** (lines 234-281)
- Full probability distribution
- Input: text + `with_probabilities: true`
- Output: `{class, confidence, probabilities[]}`
- Used for entropy-based reasoning decisions

**c) ListCategories** (lines 284-341)
- Calls MCP server's `list_categories` tool
- Returns: `{categories[], category_system_prompts{}, category_descriptions{}}`
- Builds mapping for index-to-name translation

**4. Entropy-Based Reasoning Decision** (lines 404-561)

The `classifyCategoryWithEntropyMCP` method implements intelligent routing:

```go
func (c *Classifier) classifyCategoryWithEntropyMCP(text string) (
    string,                      // category name
    float64,                     // confidence
    entropy.ReasoningDecision,   // reasoning decision
    error
)
```

Process:
1. Calls `ClassifyWithProbabilities` → gets full distribution
2. Calculates Shannon entropy from probabilities
3. Uses entropy to decide:
   - Which category (highest probability)
   - Whether to use reasoning (high entropy = uncertain = use reasoning)
   - Confidence level for routing
4. Records metrics for observability
5. Returns category + reasoning decision

### Transport Layers

**HTTP Transport** (`http_client.go`)
- RESTful HTTP/JSON-RPC
- Best for: Production, distributed systems
- Example: `http://localhost:8090/mcp`

**Stdio Transport** (`stdio_client.go`)
- Standard input/output communication
- Best for: Local development, embedded scenarios
- Launches subprocess: `python server.py`

### Required MCP Tools

MCP servers must implement two tools:

**Tool 1: `list_categories`**
```json
{
  "categories": ["math", "science", "technology", "history", "general"],
  "category_system_prompts": {
    "math": "You are a mathematics expert...",
    "science": "You are a science expert..."
  },
  "category_descriptions": {
    "math": "Mathematical and computational queries",
    "science": "Scientific concepts"
  }
}
```

**Tool 2: `classify_text`**
```json
{
  "class": 0,
  "confidence": 0.92,
  "model": "openai/gpt-oss-20b",
  "use_reasoning": false,
  "probabilities": [0.92, 0.03, 0.02, 0.02, 0.01],
  "entropy": 0.45
}
```

### Configuration

In `config.yaml`:
```yaml
classifier:
  mcp_category_model:
    enabled: true
    transport_type: "http"  # or "stdio"
    url: "http://localhost:8090/mcp"
    threshold: 0.6
    timeout_seconds: 30
    # For stdio:
    # command: "python"
    # args: ["server.py"]

categories: []  # Loaded dynamically from MCP server
```

### Example MCP Servers

The repository includes three reference implementations in `examples/mcp-classifier-server/`:

1. **Regex-Based** (`server_keyword.py`)
   - Pattern matching with regex
   - Fast (~1-5ms)
   - Simple routing logic

2. **Embedding-Based** (`server_embedding.py`)
   - Qwen3-Embedding-0.6B model
   - FAISS vector database
   - Higher accuracy (~50-100ms)

3. **Generative Model** (`server_generative.py`)
   - Fine-tuned Qwen3-0.6B with LoRA
   - True softmax probabilities
   - Highest accuracy (70-85%)
   - Shannon entropy calculation

---

## Part 3: Complete Request Flow

### Flow Overview

```
Client → Envoy Proxy → ExtProc Handler → Classification →
Security Checks → Cache → Routing → vLLM → Response
```

### Detailed Step-by-Step Flow

#### 1. Entry Point: Envoy Proxy

**File:** `config/envoy.yaml:1-120`

- Client sends request to `0.0.0.0:8801`
- Envoy HTTP connection manager receives request
- All requests match `prefix: "/"` route → `vllm_dynamic_cluster`
- Request intercepted by `ext_proc` filter before backend

**ExtProc Configuration:**
- Service: `127.0.0.1:50051` (gRPC)
- Processing mode: Request headers (SEND), Request body (BUFFERED), Response headers (SEND), Response body (BUFFERED)
- Timeout: 300s for long LLM requests

#### 2. ExtProc Handler: Request Processing Stream

**File:** `src/semantic-router/pkg/extproc/processor_core.go:17-123`

**Process Loop:**
1. Request Headers → `handleRequestHeaders`
2. Request Body → `handleRequestBody`
3. Response Headers → `handleResponseHeaders`
4. Response Body → `handleResponseBody`

**Initialization:**
- Creates `RequestContext` to maintain state
- Stores headers, timing, classification results, routing decisions

#### 3. Request Headers Phase

**File:** `src/semantic-router/pkg/extproc/processor_req_header.go:49-134`

**`handleRequestHeaders` Process:**

1. **Timing & Tracing** (lines 52-72)
   - Records `ctx.StartTime = time.Now()`
   - Extracts OpenTelemetry trace context
   - Starts span `tracing.SpanRequestReceived`

2. **Header Extraction** (lines 75-89)
   - Stores all headers in `ctx.Headers` map
   - Captures `X-Request-ID` for correlation
   - Stores method and path

3. **Streaming Detection** (lines 104-109)
   - Checks `Accept: text/event-stream`
   - Sets `ctx.ExpectStreamingResponse = true` for SSE

4. **Special Routes** (lines 112-115)
   - `GET /v1/models` → Returns model list directly
   - Bypasses normal routing

5. **Response:** Returns `CONTINUE` to proceed to body phase

#### 4. Request Body Phase: Core Classification & Routing

**File:** `src/semantic-router/pkg/extproc/processor_req_body.go:20-206`

**`handleRequestBody` Process:**

##### 4.1 Request Parsing (lines 24-62)
```go
ctx.ProcessingStartTime = time.Now()  // Start routing timer
ctx.OriginalRequestBody = v.RequestBody.GetBody()
```

- Extracts `stream` parameter
- Parses OpenAI request
- Extracts original model name
- Records metric: `metrics.RecordModelRequest(originalModel)`
- Extracts user content and messages

##### 4.2 Decision Evaluation & Model Selection

**File:** `src/semantic-router/pkg/extproc/req_filter_classification.go:10-114`

**For Auto Models Only:**
- Checks `r.Config.IsAutoModelName(originalModel)`
- Non-auto models skip classification

**Decision Engine Evaluation:**

**File:** `src/semantic-router/pkg/classification/classifier.go:585-625`

**`EvaluateDecisionWithEngine`:**

1. **Rule Evaluation** (line 594)
   - `EvaluateAllRules(text)` → Returns matched rules by type

   **Rule Types** (`classifier.go:541-583`):
   - **Keyword Rules**: Pattern matching in text
   - **Embedding Rules**: Semantic similarity using BERT embeddings
   - **Domain Rules**: Category classification (math, physics, code, etc.)

2. **Decision Engine Processing**

   **File:** `src/semantic-router/pkg/decision/engine.go:62-102`

   **`EvaluateDecisions`:**
   - Iterates through all configured decisions
   - Evaluates rule combination with AND/OR logic

   **`evaluateRuleCombination` (lines 119-177):**
   - Checks if conditions match
   - Supports AND (all conditions) or OR (any condition)
   - Calculates confidence as ratio of matched conditions

   **`selectBestDecision` (lines 179-204):**
   - Sorts by confidence or priority based on strategy
   - Returns best match

3. **Model Selection**
   - Extracts model from decision's `ModelRefs[0]`
   - Uses LoRA name if specified, otherwise base model
   - Determines reasoning mode from `UseReasoning` config

**Returns:** `(decisionName, confidence, reasoningDecision, selectedModel)`

##### 4.3 Security Checks

**File:** `src/semantic-router/pkg/extproc/req_filter_jailbreak.go:16-93`

**`performSecurityChecks`:**

1. **Jailbreak Detection**
   - Checks if enabled for decision
   - Gets decision-specific threshold
   - Calls `Classifier.AnalyzeContentForJailbreakWithThreshold`

   **File:** `src/semantic-router/pkg/classification/classifier.go:478-514`
   - Uses ModernBERT or linear classifier
   - Returns `hasJailbreak, detections, error`

2. **Blocking Response**
   - If jailbreak detected: Returns 400 error immediately
   - Records metric: `metrics.RecordRequestError(..., "jailbreak_block")`
   - **Request processing STOPS here**

##### 4.4 PII Detection & Policy Check

**File:** `src/semantic-router/pkg/extproc/req_filter_pii.go:17-115`

**`performPIIDetection`:**

1. **PII Detection**
   - Checks if enabled for decision
   - Calls `Classifier.DetectPIIInContent(allContent)`

   **File:** `src/semantic-router/pkg/classification/classifier.go:948-974`
   - Uses ModernBERT token classifier or LoRA model
   - Detects: PERSON, EMAIL, PHONE, SSN, CREDIT_CARD, etc.
   - Returns list of detected PII types

2. **Policy Check**
   - Calls `PIIChecker.CheckPolicy(decisionName, detectedPII)`
   - Checks if decision allows detected PII types
   - If violated: Returns error immediately
   - **Request processing STOPS here**

##### 4.5 Semantic Cache Lookup

**File:** `src/semantic-router/pkg/extproc/req_filter_cache.go:15-87`

**`handleCaching`:**

1. **Cache Check**
   - Extracts query and model
   - Checks if cache enabled for decision
   - Gets decision-specific similarity threshold

2. **Cache Lookup**
   - Calls `Cache.FindSimilarWithThreshold(model, query, threshold)`
   - Uses embedding similarity to find cached responses
   - On cache **HIT**:
     - Sets `ctx.VSRCacheHit = true`
     - Returns immediate response with cached data
     - **Request processing STOPS here (skips LLM call)**

3. **Cache MISS**
   - Stores pending request: `Cache.AddPendingRequest(...)`
   - Will be updated with response later

##### 4.6 Model Routing & Request Modification

**File:** `src/semantic-router/pkg/extproc/processor_req_body.go:92-206`

**For Auto Models:**

**`handleAutoModelRouting`:**

1. **Model Selection Validation**
   - If `matchedModel == originalModel`: No routing needed, CONTINUE

2. **Routing Decision Recording**
   - Records decision in tracing span
   - Tracks VSR decision metadata
   - Records metric: `metrics.RecordModelRouting(originalModel, matchedModel)`

3. **Endpoint Selection**

   **File:** `src/semantic-router/pkg/config/helper.go:228-248`

   **`SelectBestEndpointAddressForModel`:**
   - Gets endpoints from model's `preferred_endpoints` config
   - Selects endpoint with highest weight
   - Returns `"address:port"` string

4. **Request Body Modification**

   **File:** `src/semantic-router/pkg/extproc/processor_req_body.go:232-263`

   **`modifyRequestBodyForAutoRouting`:**
   - Changes model field: `openAIRequest.Model = matchedModel`
   - Serializes with stream parameter preserved
   - Sets reasoning mode if configured
   - Adds decision-specific system prompt

5. **Response Creation**

   **File:** `src/semantic-router/pkg/extproc/processor_req_body.go:265-323`

   **`createRoutingResponse`:**
   - Creates body mutation with modified request
   - Sets routing headers:
     - `x-vsr-destination-endpoint`: Target vLLM endpoint address:port
     - `x-selected-model`: Selected model name
   - Applies decision-specific header mutations
   - Removes `content-length` (will be recalculated)

6. **Tool Selection**
   - Filters available tools based on query similarity
   - Modifies tools array in request if enabled

**For Specified Models:**
- Selects endpoint for specified model
- Creates response with endpoint header only
- No body modification needed

#### 5. Request Forwarding to vLLM

**Envoy Configuration:** `config/envoy.yaml:100-114`

**Dynamic Cluster Routing:**

1. **ExtProc Response Applied:**
   - Envoy receives:
     - `Status: CONTINUE`
     - `BodyMutation`: Modified request with new model
     - `HeaderMutation`: Added headers including `x-vsr-destination-endpoint`

2. **Original Destination Cluster:**
   ```yaml
   type: ORIGINAL_DST
   lb_policy: CLUSTER_PROVIDED
   original_dst_lb_config:
     use_http_header: true
     http_header_name: "x-vsr-destination-endpoint"
   ```
   - Envoy reads `x-vsr-destination-endpoint` header
   - Dynamically routes to that endpoint
   - No static cluster configuration needed

3. **Request Sent:**
   - Modified request with selected model → vLLM endpoint
   - Includes system prompt and reasoning mode
   - Stream parameter preserved for SSE

#### 6. Response Headers Phase

**File:** `src/semantic-router/pkg/extproc/processor_res_header.go:16-187`

**`handleResponseHeaders`:**

1. **Status Code Detection**
   - Extracts `:status` pseudo-header
   - Records errors for non-2xx:
     - `upstream_5xx` for 500+ status
     - `upstream_4xx` for 400+ status

2. **Streaming Detection**
   - Checks `Content-Type: text/event-stream`
   - Sets `ctx.IsStreamingResponse = true` for SSE

3. **TTFT Measurement**
   - **Non-streaming:** Records Time To First Token on headers arrival
   - **Streaming:** Defers TTFT to first body chunk
   - Records: `metrics.RecordModelTTFT(ctx.RequestModel, ttft)`

4. **VSR Decision Headers**
   - Adds custom response headers:
     - `x-vsr-selected-category`: Domain classification (e.g., "math")
     - `x-vsr-selected-decision`: Decision engine result
     - `x-vsr-selected-reasoning`: Reasoning mode ("on"/"off")
     - `x-vsr-selected-model`: Final model selected
     - `x-vsr-injected-system-prompt`: System prompt added ("true"/"false")

5. **Streaming Mode Override**
   - If streaming detected:
     - Sets `response_body_mode: STREAMED` dynamically
     - Allows ExtProc to receive SSE chunks for TTFT

#### 7. Response Body Phase

**File:** `src/semantic-router/pkg/extproc/processor_res_body.go:14-129`

**`handleResponseBody`:**

1. **Streaming Response Handling**
   - **First SSE chunk:**
     - Records TTFT: `metrics.RecordModelTTFT(ctx.RequestModel, ttft)`
     - Calculates from `ctx.ProcessingStartTime`
   - **Subsequent chunks:**
     - Returns CONTINUE immediately
     - Chunks streamed directly to client

2. **Non-Streaming Response Processing**

   **Token Extraction:**
   ```go
   var parsed openai.ChatCompletion
   json.Unmarshal(responseBody, &parsed)
   promptTokens := int(parsed.Usage.PromptTokens)
   completionTokens := int(parsed.Usage.CompletionTokens)
   ```

   **Metrics Recording:**
   - `RecordModelTokensDetailed(model, promptTokens, completionTokens)`
   - `RecordModelCompletionLatency(model, latency)`
   - `RecordModelTPOT(model, timePerToken)`
     - TPOT = Time Per Output Token = latency / completionTokens
   - `RecordModelCost(model, currency, costAmount)`
     - Cost = (promptTokens × promptRate + completionTokens × completionRate) / 1M

   **Usage Logging:**
   ```json
   {
     "event": "llm_usage",
     "request_id": "...",
     "model": "selected-model",
     "prompt_tokens": 150,
     "completion_tokens": 300,
     "total_tokens": 450,
     "completion_latency_ms": 2500,
     "cost": 0.0045,
     "currency": "USD"
   }
   ```

3. **Cache Update**
   - Updates pending cache entry with response
   - `Cache.UpdateWithResponse(requestID, responseBody)`
   - Future similar queries will hit cache

4. **Response**
   - Returns CONTINUE
   - Response body passed through unmodified to client

#### 8. Final Response to Client

**Envoy Proxy:**
- Receives CONTINUE from ExtProc
- Forwards response to client with:
  - Original vLLM response body
  - VSR decision headers
  - Standard HTTP headers

**Client Receives:**
```http
HTTP/1.1 200 OK
Content-Type: application/json
x-vsr-selected-category: mathematics
x-vsr-selected-decision: math_decision
x-vsr-selected-reasoning: on
x-vsr-selected-model: qwen-72b-chat
x-vsr-injected-system-prompt: true

{
  "id": "chatcmpl-...",
  "model": "qwen-72b-chat",
  "choices": [...],
  "usage": {
    "prompt_tokens": 150,
    "completion_tokens": 300,
    "total_tokens": 450
  }
}
```

---

## Visual Flow Diagram

```
Client Request
    ↓
[Envoy Proxy :8801]
    ↓ (ext_proc gRPC)
[ExtProc Server :50051]
    ↓
┌─────────────────────────────────────────────────────────┐
│ Request Headers Phase                                   │
│ - Extract headers, request ID, streaming detection      │
│ - Initialize tracing context                            │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│ Request Body Phase                                      │
│                                                          │
│ 1. Parse OpenAI Request                                │
│    ├─ Extract model, messages, parameters              │
│    └─ Extract user content                             │
│                                                          │
│ 2. Decision Evaluation (Auto Models Only)              │
│    ├─ Evaluate Keyword Rules                           │
│    ├─ Evaluate Embedding Rules                         │
│    ├─ Evaluate Domain Rules (Category Classification)  │
│    ├─ Decision Engine: Combine rules with AND/OR       │
│    └─ Select Model from Decision's ModelRefs           │
│                                                          │
│ 3. Security Checks                                     │
│    ├─ Jailbreak Detection → BLOCK if detected         │
│    └─ PII Detection → BLOCK if policy violated        │
│                                                          │
│ 4. Semantic Cache Lookup                               │
│    └─ Cache HIT → Return cached response (STOP)       │
│                                                          │
│ 5. Model Routing                                       │
│    ├─ Select vLLM Endpoint (weighted selection)        │
│    ├─ Modify Request Body (change model)              │
│    ├─ Add System Prompt (decision-specific)           │
│    ├─ Set Reasoning Mode                               │
│    └─ Set Headers: x-vsr-destination-endpoint         │
└─────────────────────────────────────────────────────────┘
    ↓
[Envoy: Original Destination Cluster]
    ↓ (routes to x-vsr-destination-endpoint)
[vLLM Endpoint: selected-model:8000]
    ↓ (LLM inference)
[vLLM Response]
    ↓
[Envoy Proxy]
    ↓ (ext_proc gRPC)
┌─────────────────────────────────────────────────────────┐
│ Response Headers Phase                                  │
│ - Detect status code (record errors)                   │
│ - Detect streaming (text/event-stream)                 │
│ - Record TTFT (non-streaming only)                     │
│ - Add VSR decision headers                             │
└─────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────┐
│ Response Body Phase                                     │
│                                                          │
│ Streaming:                                              │
│ ├─ First chunk: Record TTFT                            │
│ └─ Pass through all chunks                             │
│                                                          │
│ Non-Streaming:                                          │
│ ├─ Parse token usage                                   │
│ ├─ Record metrics (tokens, latency, TPOT, cost)       │
│ └─ Update semantic cache                               │
└─────────────────────────────────────────────────────────┘
    ↓
[Envoy Proxy]
    ↓
Client Receives Response
```

---

## Summary: Key Design Patterns

1. **External Processing Pattern**
   - Envoy delegates to external gRPC service
   - Enables complex routing without Envoy recompilation

2. **Dynamic Routing**
   - Uses ORIGINAL_DST cluster with HTTP header
   - Routes to dynamically selected backends

3. **Streaming Support**
   - Special handling for SSE responses
   - Deferred TTFT measurement, chunk pass-through

4. **Circuit Breaker Pattern**
   - Security checks can immediately terminate requests
   - Prevents wasted LLM calls

5. **Semantic Cache**
   - Embedding-based similarity search
   - Reduces LLM calls for similar queries

6. **Decision Engine**
   - Flexible rule combination (AND/OR)
   - Complex routing based on multiple signals

7. **Observable Architecture**
   - Comprehensive metrics, tracing, structured logging
   - Every stage tracked

---

## Key Files Reference

### MCP Implementation
- `src/semantic-router/pkg/classification/mcp_classifier.go` - MCP classifier
- `src/semantic-router/pkg/mcp/http_client.go` - HTTP transport
- `src/semantic-router/pkg/mcp/stdio_client.go` - Stdio transport
- `examples/mcp-classifier-server/` - Example MCP servers

### Request Flow
- `config/envoy.yaml` - Envoy configuration
- `src/semantic-router/pkg/extproc/processor_core.go` - Main processing loop
- `src/semantic-router/pkg/extproc/processor_req_header.go` - Request headers phase
- `src/semantic-router/pkg/extproc/processor_req_body.go` - Request body phase
- `src/semantic-router/pkg/extproc/req_filter_classification.go` - Classification
- `src/semantic-router/pkg/extproc/req_filter_jailbreak.go` - Security checks
- `src/semantic-router/pkg/extproc/req_filter_pii.go` - PII detection
- `src/semantic-router/pkg/extproc/req_filter_cache.go` - Semantic cache
- `src/semantic-router/pkg/extproc/processor_res_header.go` - Response headers phase
- `src/semantic-router/pkg/extproc/processor_res_body.go` - Response body phase
- `src/semantic-router/pkg/decision/engine.go` - Decision engine
- `src/semantic-router/pkg/classification/classifier.go` - Core classification logic
