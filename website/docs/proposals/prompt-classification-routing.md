# Prompt Classification Routing

**Related Issues:** [#313](https://github.com/vllm-project/semantic-router/issues/313), [#200](https://github.com/vllm-project/semantic-router/issues/200)

This proposal introduces a **unified content scanning and routing framework** that extends the vLLM Semantic Router with three complementary signal sources:

1. **Keyword-Based Routing** - Deterministic, fast, Boolean logic for exact term matching
2. **Regex Content Scanning** - Pattern-based detection for safety, compliance, and structured data
3. **Embedding Similarity Scanning** - Semantic concept detection robust to paraphrasing

All three signals integrate with the existing **BERT-based classification** through a **Signal Fusion Layer**, providing users with a powerful, flexible routing control plane while maintaining backward compatibility with the current architecture.

## Key Design Principles

- **Complementary, Not Replacement**: Augment existing BERT classification rather than replacing it
- **Dual Execution Paths**: Support both in-tree (low-latency) and out-of-tree via MCP (high-flexibility) modes
- **Policy-Driven Fusion**: Allow users to compose signals using Boolean expressions, thresholds, and weighted rules
- **Performance-Conscious**: Provide fast paths for common cases while supporting complex scenarios
- **Security-First**: ReDoS protection, input validation, and comprehensive audit logging

## Problem Statement & Motivation

### Current Limitations

The vLLM Semantic Router currently relies exclusively on **ModernBERT classification** for semantic category detection. While powerful, this approach has several limitations:

#### From Issue #313: No Deterministic Routing

**Problem:** Cannot route queries based on specific keywords or technology terms

- Query: "How to secure a Kubernetes cluster with RBAC?"
- Current: Must run ML inference (~20-30ms) → Classify as "computer science" → Route to general models
- Desired: Match keywords `["kubernetes", "k8s", "RBAC"]` → Route directly to `[k8s-expert, devops-model]`

**Impact:**

- Unnecessary latency (~20-30ms) for queries that could be routed deterministically in ~1-2ms
- Less precise routing (category "computer science" is too broad)
- Cannot leverage domain knowledge (e.g., "CVE-" patterns always go to security models)
- No Boolean logic for complex matching (e.g., "Kubernetes AND security" vs "Kubernetes OR Docker")

#### No Semantic Concept Detection Beyond Categories

**Problem:** Cannot detect presence of specific concepts/topics within a query

- Cannot route based on "multi-step reasoning" concept detection
- Cannot detect domain-specific intents like "sentiment analysis" or "code generation"
- Embedding similarity is used for caching but not for routing decisions

### Use Cases

#### Use Case 1: Technology-Specific Routing (Issue #313)

**Scenario:** Enterprise AI gateway routing to specialized infrastructure models

```yaml
# Desired Configuration
keyword_routing:
  rules:
    - name: "kubernetes-infrastructure"
      keywords: ["kubernetes", "k8s", "kubectl", "helm"]
      operator: "OR"
      models: ["k8s-expert", "devops-model"]
      priority: 100
```

**Benefits:**

- Deterministic routing in ~1-2ms vs ~20-30ms for ML inference
- Precise model selection based on domain expertise
- Easy to update and maintain without ML retraining

#### Use Case 2: Security-Critical Pattern Detection

**Scenario:** Prevent data exfiltration and compliance violations

```yaml
regex_scanning:
  rules:
    - name: "ssn-detection"
      pattern: '\b\d{3}-\d{2}-\d{4}\b'
      action: "block"
      response: "Cannot process queries containing SSN patterns"
    
    - name: "cve-routing"
      pattern: 'CVE-\d{4}-\d{4,7}'
      action: "route"
      models: ["security-hardened-model"]
```

**Benefits:**

- Guaranteed blocking of PII/sensitive patterns (no ML false negatives)
- Compliance audit trail
- Sub-millisecond detection

#### Use Case 3: Semantic Intent Detection

**Scenario:** Route queries requiring multi-step reasoning

```yaml
embedding_similarity:
  concepts:
    - name: "multi-step-reasoning"
      keywords:
        - "step-by-step"
        - "break down the problem"
        - "analyze systematically"
      threshold: 0.75
      action: "boost_category"
      category: "reasoning"
```

**Benefits:**

- Robust to paraphrasing ("explain thoroughly" → similar to "step-by-step")
- Can detect semantic presence without exact word matches
- Complements BERT classification with fine-grained intent detection

## Proposed Solution Architecture

### High-Level System Design

import ZoomableMermaid from '@site/src/components/ZoomableMermaid';

<ZoomableMermaid title="System Architecture Overview" defaultZoom={5.5}>
{`graph TD
    A[Envoy External Processor<br/>semantic-router ExtProc] --> B[Request Handler<br/>handleModelRouting]
    
    B --> C{Execution Path}
    
    C -->|In-Tree<br/>Low Latency| D[In-Tree Signal Providers]
    C -->|Out-of-Tree<br/>High Flexibility| E[MCP Services]
    
    D --> D1[Keyword Matcher<br/>~1-2ms]
    D --> D2[Regex Scanner<br/>~2-5ms]
    D --> D3[Embedding Similarity<br/>~5-10ms]
    D --> D4[BERT Classifier<br/>~20-30ms]
    
    E --> E1[MCP Keyword Scanner]
    E --> E2[MCP Similarity Scorer]
    
    D1 --> F[Signal Fusion Layer<br/>Policy Evaluation]
    D2 --> F
    D3 --> F
    D4 --> F
    E1 --> F
    E2 --> F
    
    F --> G{Fusion Decision}
    
    G -->|Block| H[Return 403<br/>Safety Violation]
    G -->|Route| I[Model Selection<br/>from Candidates]
    G -->|Boost| J[Apply Category<br/>Weights]
    G -->|Fallthrough| K[Use BERT<br/>Category]
    
    I --> L[Endpoint Selection]
    J --> L
    K --> L
    
    L --> M[Forward to<br/>vLLM Backend]
    
    style D1 fill:#e1f5ff
    style D2 fill:#e1f5ff
    style D3 fill:#e1f5ff
    style D4 fill:#e1f5ff
    style E1 fill:#fff9c4
    style E2 fill:#fff9c4
    style F fill:#c8e6c9
    style H fill:#ffcdd2
    style M fill:#c8e6c9`}
</ZoomableMermaid>

### Component Breakdown

#### In-Tree Signal Providers (Low-Latency Path)

The in-tree path provides four core signal providers that run directly within the router process for minimal latency:

**A. Keyword Matcher**

The Keyword Matcher performs fast, deterministic matching of exact terms or phrases within queries.

**How it Works:**

- Maintains a collection of keyword rules, each containing a list of terms to match
- Scans incoming queries for the presence of these keywords
- Supports Boolean operators (AND/OR) to combine multiple keywords
- Can be case-sensitive or case-insensitive
- Returns matched rules along with their associated candidate models

**Characteristics:**

- **Performance:** ~1-2ms for dozens of rules with hundreds of keywords
- **Use Case:** Technology terms (kubernetes, SQL), product names, domain-specific vocabulary
- **Complexity:** O(n×m) where n=rules, m=keywords per rule
- **Limitations:** No fuzzy matching, no regex patterns, exact term matching only

**Example Use:** Route queries containing "kubernetes" or "k8s" to infrastructure expert models.

**B. Regex Scanner**

The Regex Scanner uses regular expression patterns to detect structured data and specific patterns within queries.

**How it Works:**

- Compiles regex patterns at startup using RE2 engine (guaranteed linear-time matching)
- Scans query content against all patterns
- Each pattern can specify an action (block, route, or log)
- Returns matches with associated actions

**Characteristics:**

- **Performance:** ~2-5ms for dozens of patterns
- **Use Case:** PII patterns (SSN, credit cards), CVE IDs, email addresses, structured data
- **Safety:** RE2 engine prevents catastrophic backtracking (ReDoS protection)
- **Limitations:** Best for fewer than 100 patterns; for larger rule sets, use MCP with Hyperscan

**Example Use:** Detect and block Social Security Numbers, route CVE IDs to security models.

**C. Embedding Similarity Scanner**

The Embedding Similarity Scanner detects semantic concepts and intents that may be expressed in different ways.

**How it Works:**

- Reuses the existing BERT embedder from the router
- Pre-computes embeddings for concept keywords at startup
- Embeds the incoming query once
- Computes cosine similarity between query embedding and each concept's keyword embeddings
- Aggregates similarities (mean, max, or any threshold)
- Returns concepts that exceed their configured similarity thresholds

**Characteristics:**

- **Performance:** ~5-10ms (one-time embedding + fast cosine similarity)
- **Use Case:** Semantic intent detection (multi-step reasoning, code generation, sentiment analysis)
- **Advantages:** Robust to paraphrasing and word choice variations
- **Limitations:** Requires threshold calibration; less interpretable than keyword/regex

**Example Use:** Detect "multi-step reasoning" requests even when phrased as "explain thoroughly" or "walk me through".

**D. BERT Classifier (Existing)**

The existing BERT-based classifier remains a core signal provider, now treated as an equal peer to the new scanning methods.

**How it Works:**

- Uses ModernBERT model to classify queries into semantic categories
- Returns category name and confidence score
- Categories mapped to model pools with scoring

**Characteristics:**

- **Performance:** ~20-30ms
- **Use Case:** Broad semantic categorization (computer science, reasoning, biology, etc.)
- **Advantages:** Well-established, handles nuanced semantic understanding
- **Role:** Serves as both a signal and a fallback when other signals don't match

#### Out-of-Tree Signal Providers (MCP Path)

MCP (Model Context Protocol) servers run as separate processes or services, providing flexibility and scalability at the cost of modest added latency.

**A. MCP Keyword Scanner**

External keyword scanning service that can handle massive rule sets and specialized matching engines.

**Capabilities:**

- **Aho-Corasick Algorithm**: Efficiently searches for thousands to tens of thousands of literal keywords simultaneously
- **Hyperscan Engine**: Handles tens of thousands to hundreds of thousands of complex regex patterns with compiled pattern databases
- **Custom Matching Logic**: Domain-specific algorithms (e.g., SQL injection detection, code analysis)

**Benefits:**

- Hot-reload rule sets without router restart
- Scale to massive pattern databases (100K+ patterns)
- Offload CPU-intensive matching to dedicated services
- Independent versioning and lifecycle management
- A/B test different rule configurations

**Tradeoffs:**

- Added network latency (~2-5ms for localhost/cluster-local)
- Additional operational complexity
- Requires separate deployment and monitoring

**B. MCP Similarity Scorer**

External semantic similarity service with customizable embedding models and advanced capabilities.

**Capabilities:**

- **Custom Embedding Models**: Domain-tuned SBERT, Embedding Gemma, multilingual models
- **GPU Batching**: Batch multiple requests for higher throughput
- **Vector Database Integration**: Use Milvus, Qdrant, or other vector DBs for large-scale concept search
- **Fine-Tuned Models**: Deploy models specifically trained for your domain

**Benefits:**

- Bring your own embedding model
- Domain-specific fine-tuning for better accuracy
- Advanced aggregation strategies
- Multilingual support
- Scale embedding inference independently

**Tradeoffs:**

- Higher latency than in-tree (~10-20ms additional)
- Requires GPU resources for optimal performance
- More complex deployment architecture

#### Signal Fusion Layer

The Signal Fusion Layer is the decision-making engine that combines all signals (keyword, regex, embedding similarity, and BERT) into actionable routing decisions.

**How it Works:**

1. **Gather Signals**: Collect results from all active signal providers (in-tree and MCP)
2. **Evaluate Policy Rules**: Process rules in priority order (highest first)
3. **Match Conditions**: Evaluate Boolean expressions that reference signal results
4. **Execute Actions**: Perform the action of the first matching rule
5. **Return Decision**: Block, route to specific models, boost categories, or fallthrough to BERT

**Policy Types:**

**1. Block Actions**

- Immediately reject requests that violate safety or compliance rules
- Example: Block all queries containing SSN patterns

**2. Route Actions**

- Directly route to specific model candidates based on signal matches
- Example: Route Kubernetes queries to k8s-expert models

**3. Boost Actions**

- Apply weight multipliers to BERT categories based on signal presence
- Example: Boost "reasoning" category weight by 1.5x when multi-step reasoning is detected

**4. Fallthrough Actions**

- Use standard BERT classification when no specific rules match
- Acts as the default catch-all

**Policy Evaluation:**

- **Priority-Based**: Rules evaluated from highest to lowest priority (200 → 0)
- **Short-Circuit**: First matching rule wins, no further evaluation
- **Boolean Expressions**: Combine multiple signal conditions with AND, OR, NOT
- **Flexible Comparisons**: Support `==`, `!=`, `>`, `<`, `>=`, `<=` for numeric thresholds

**Expression Capabilities:**

- Reference keyword matches: `keyword.kubernetes-infrastructure.matched`
- Check similarity scores: `similarity.multi-step-reasoning.score > 0.75`
- Use BERT results: `bert.category == 'computer science'`
- Combine signals: `keyword.security.matched && bert.category == 'security'`

## Configuration Schema

The content scanning framework is configured through several interconnected configuration files that define rules, patterns, concepts, and policies.

### Top-Level Configuration

The main router configuration extends with a new `content_scanning` section that controls:

**Framework Control:**

- Enable/disable the entire content scanning system
- Default action when no rules match (fallthrough to BERT or block)
- Audit logging toggle

**In-Tree Providers:**

- **Keyword Matching:** Enable/disable, path to rules file
- **Regex Scanning:** Enable/disable, path to patterns file, choice of regex engine (RE2 recommended)
- **Embedding Similarity:** Enable/disable, path to concepts file, default similarity threshold

**MCP Providers (Optional):**

- **Keyword Scanner:** Endpoint URL, authentication, rule set version ID, timeout
- **Similarity Scorer:** Endpoint URL, authentication, concept set version ID, timeout

**Fusion Policy:**

- Path to fusion policy file
- Default action behavior
- Audit logging configuration

### Keyword Rules Configuration

Keyword rules define exact term matching for deterministic routing:

**Per Rule:**

- **Name:** Unique identifier for the rule
- **Description:** Human-readable explanation
- **Keywords:** List of terms to match (e.g., "kubernetes", "k8s", "kubectl")
- **Operator:** Boolean logic (OR = any keyword, AND = all keywords)
- **Case Sensitivity:** Whether to match case-sensitively
- **Candidate Models:** List of models to route to when matched
- **Priority:** Numeric priority for conflict resolution (higher = evaluated first)

**Example Rules:**

- Kubernetes infrastructure (OR operator, case-insensitive)
- Database operations (OR operator, case-insensitive)
- Security critical terms (OR operator, case-sensitive for CVE IDs)

### Regex Patterns Configuration

Regex patterns define structured data detection and safety checks:

**Per Pattern:**

- **Name:** Unique identifier
- **Description:** What the pattern detects
- **Pattern:** Regular expression (RE2 syntax)
- **Action:** What to do on match (block, route, log)
- **Block Message:** Error message if action is block
- **Candidate Models:** Models to route to if action is route
- **Priority:** Numeric priority (higher = evaluated first)

**Example Patterns:**

- SSN detection (block action, high priority)
- Credit card detection (block action, high priority)
- CVE ID routing (route action to security models)
- Email detection (log action for audit)

### Embedding Similarity Concepts Configuration

Concepts define semantic intents that may be expressed in various ways:

**Per Concept:**

- **Name:** Unique identifier
- **Description:** What intent this detects
- **Keywords:** Reference phrases that represent the concept
- **Threshold:** Minimum similarity score to match (0.0-1.0)
- **Aggregate Method:** How to combine keyword similarities (mean, max, any)
- **Action:** What to do on match (boost_category, route)
- **Category/Models:** Target category to boost or models to route to
- **Boost Weight:** Multiplier for category boosting

**Example Concepts:**

- Multi-step reasoning (mean aggregation, boost reasoning category by 1.5x)
- Code generation (max aggregation, route to code models)
- Sentiment analysis (mean aggregation, route to NLP specialists)

### Fusion Policy Configuration

Fusion policies combine all signals into routing decisions:

**Policy Structure:**

- Rules evaluated in priority order (200 → 0)
- First matching rule wins (short-circuit evaluation)

**Per Rule:**

- **Name:** Unique identifier
- **Condition:** Boolean expression referencing signals
- **Action:** Decision type (block, route, boost_category, fallthrough)
- **Priority:** Numeric priority (200=safety, 150=routing, 100=boost, 50=consensus, 0=default)
- **Models/Category:** Target for route or boost actions
- **Message:** Block message if action is block

**Priority Levels:**

- **200:** Safety blocks (SSN, credit cards, PII)
- **150:** High-confidence routing overrides (keyword + regex matches)
- **100:** Category boosting (embedding similarity signals)
- **50:** Consensus requirements (multiple signals must agree)
- **0:** Default fallthrough to BERT

**Expression Language:**

- Reference signals: `keyword.<rule>.matched`, `regex.<pattern>.matched`, `similarity.<concept>.score`
- Boolean operators: `&&` (AND), `||` (OR), `!` (NOT)
- Comparisons: `==`, `!=`, `>`, `<`, `>=`, `<=`
- BERT results: `bert.category`, `bert.confidence`

## Integration with Existing Router

### Request Processing Flow

The content scanning framework integrates seamlessly into the existing router's request handling flow:

**Integration Point:** The `handleModelRouting()` function in the request handler

**Processing Steps:**

1. **Check if Content Scanning is Enabled**
   - If disabled, use existing BERT-only routing (backward compatible)
   - If enabled, proceed with signal gathering

2. **Gather Signals in Parallel**
   - Launch concurrent signal providers (keyword, regex, embedding, BERT)
   - Each provider runs independently to minimize latency
   - MCP providers called with timeout protection
   - BERT classification always runs as a fallback option

3. **Evaluate Fusion Policy**
   - Collect all signal results into a unified input structure
   - Pass to Signal Fusion Layer for policy evaluation
   - Policy engine processes rules in priority order
   - First matching rule determines the action

4. **Handle Fusion Decision**
   - **Block Decision:** Immediately return 403 error with explanation
   - **Route Decision:** Select best model from candidate list
   - **Boost Decision:** Apply weight multipliers to BERT categories, then classify
   - **Fallthrough Decision:** Use standard BERT classification

5. **Continue Normal Flow**
   - Selected model passed to endpoint selection
   - Request modified with new model and routing headers
   - Forwarded to appropriate vLLM backend

**Key Design Principles:**

- Non-blocking parallel execution for minimum latency
- Graceful degradation if components fail
- Comprehensive observability at each step
- Backward compatible with existing routing logic

### Backward Compatibility

**Guarantee:** Existing deployments continue to work without changes.

- **Default behavior:** `content_scanning.enabled: false` → Uses existing BERT-only routing
- **Opt-in model:** Users explicitly enable content scanning in configuration
- **Fallthrough policy:** If no scanning rules match, system falls back to BERT classification
- **Configuration validation:** Invalid scanning configs are rejected at startup with clear error messages
