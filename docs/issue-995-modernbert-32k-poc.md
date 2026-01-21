# ModernBERT-base-32k Integration POC: Complete Design Document

## Executive Summary

This document describes a **Proof of Concept** for integrating ModernBERT-base-32k into the Semantic Router to extend context window from 512 to 32,768 tokens for signal extraction. This replaces the original plan to pre-train a base model, as ModernBERT-base-32k already exists with RoPE support up to 32K tokens.

> **⚠️ POC Scope:** This is a proof of concept to validate the technical feasibility of:
> - Loading and using ModernBERT-base-32k model
> - LoRA adapter compatibility with ModernBERT-base-32k
> - Processing sequences up to 32K tokens
> - Chunking logic for sequences > 32K tokens
> - Backward compatibility considerations
> 
> Production hardening (error handling, scaling, monitoring) is out of scope.

### Core Capabilities

| Capability | Description |
|------------|-------------|
| **Extended Context Window** | Base model supports 32,768 tokens (64x increase from 512) |
| **ModernBERT Integration** | Use existing `llm-semantic-router/modernbert-base-32k` model |
| **RoPE Support** | Model uses RoPE (Rotary Position Embeddings) for long context |
| **Automatic Chunking** | Sequences > 32K tokens are automatically chunked |
| **LoRA Compatibility** | Test existing LoRA adapters or retrain if needed |

### Key Design Principles

1. **Use existing model** - No pre-training needed, ModernBERT-base-32k already exists
2. **Replace BERT-base** - Use ModernBERT-base-32k as base for signal extraction
3. **32K target** - Support up to 32K tokens (not configurable)
4. **Chunking for overflow** - Automatically chunk sequences > 32K tokens
5. **Minimal changes** - Leverage existing ModernBERT infrastructure

### Explicit Assumptions (POC)

| Assumption | Implication | Risk if Wrong |
|------------|-------------|---------------|
| ModernBERT-base-32k loads with existing code | May need adjustments to loading code | Low - infrastructure exists |
| LoRA adapters work with ModernBERT-base-32k | May need retraining | Medium - retraining takes time |
| 32K context is sufficient | May need larger for some use cases | Low - can extend later |
| Chunking works well for signal extraction | May lose context across chunks | Medium - need to test |
| ~16GB GPU memory available | May need CPU fallback | Medium - performance impact |

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Architecture Overview](#2-architecture-overview)
3. [Model Integration](#3-model-integration)
4. [Chunking Strategy](#4-chunking-strategy)
5. [LoRA Adapter Compatibility](#5-lora-adapter-compatibility)
6. [Pipeline Integration](#6-pipeline-integration)
7. [Implementation Details](#7-implementation-details)
8. [Data Structures](#8-data-structures)
9. [Configuration](#9-configuration)
10. [Failure Modes and Fallbacks](#10-failure-modes-and-fallbacks-poc)
11. [Success Criteria](#11-success-criteria-poc)
12. [Implementation Plan](#12-implementation-plan)
13. [Known POC Limitations](#13-known-poc-limitations-explicitly-deferred)
14. [Future Enhancements](#14-future-enhancements)

---

## 1. Problem Statement

### Current State

The Semantic Router uses BERT-base models with a **512-token context window** for signal extraction (domain classification, PII detection, jailbreak detection). This limitation causes:

```
Long conversation (2000 tokens):
  User: [1500 tokens of context] "What's my budget for the Hawaii trip?"
  → Truncated to 512 tokens
  → Context lost: "Hawaii trip" mentioned earlier is cut off
  → Signal extraction quality degrades ❌
```

### Desired State

With ModernBERT-base-32k:

```
Long conversation (2000 tokens):
  User: [1500 tokens of context] "What's my budget for the Hawaii trip?"
  → Full 2000 tokens processed
  → Context preserved: "Hawaii trip" information available
  → Signal extraction quality maintained ✅

Very long conversation (50000 tokens):
  → Automatically chunked into 32K token pieces
  → Each chunk processed separately
  → Results aggregated
```

### Use Cases

| Use Case | Current (512) | With 32K | Benefit |
|----------|---------------|----------|---------|
| **Long conversations** | Truncated, context lost | Full context preserved | Better signal extraction |
| **Document analysis** | Only first 512 tokens | Up to 32K tokens | More complete analysis |
| **Multi-turn reasoning** | Limited history | Extended history | Better context understanding |
| **Signal extraction** | May miss signals in later tokens | Captures signals across full window | Improved accuracy |
| **Very long documents** | Only first 512 tokens | Chunked and processed | Complete document coverage |

**Example Scenarios:**

**Scenario 1: Long Conversation with Context**
```
Current (512 tokens):
  User: [500 tokens of conversation history] "What's my budget for the Hawaii trip?"
  → Only last 12 tokens processed: "What's my budget for the Hawaii trip?"
  → Context lost: "Hawaii trip" mentioned earlier is cut off
  → Domain classification may fail ❌

With ModernBERT-32k:
  User: [500 tokens of conversation history] "What's my budget for the Hawaii trip?"
  → Full 512 tokens processed
  → Context preserved: "Hawaii trip" information available
  → Domain classification accurate ✅
```

**Scenario 2: Very Long Document**
```
Current (512 tokens):
  Document: 50,000 tokens of technical documentation
  → Only first 512 tokens analyzed
  → PII detection misses information in later sections ❌

With ModernBERT-32k:
  Document: 50,000 tokens
  → Chunked: [0-32K], [32K-50K]
  → Each chunk analyzed separately
  → PII detection covers entire document ✅
```

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│              MODERNBERT-BASE-32K INTEGRATION ARCHITECTURE               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│                         Model Loading Pipeline                           │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │  HuggingFace: llm-semantic-router/modernbert-base-32k           │   │
│  │       │                                                          │   │
│  │       ▼                                                          │   │
│  │  TraditionalModernBertClassifier::load_from_directory()          │   │
│  │       │                                                          │   │
│  │       ▼                                                          │   │
│  │  ModernBERT-base-32k Model                                       │   │
│  │  - Context window: 32,768 tokens                                 │   │
│  │  - RoPE support (already implemented)                            │   │
│  │  - 149M parameters                                               │   │
│  │       │                                                          │   │
│  │       ▼                                                          │   │
│  │  LoRA Adapters (test compatibility)                             │   │
│  │  - Domain classifier LoRA                                        │   │
│  │  - PII detector LoRA                                             │   │
│  │  - Jailbreak detector LoRA                                       │   │
│  │                                                                  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│                         Request Processing Pipeline                      │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │  Input Text                                                      │   │
│  │       │                                                          │   │
│  │       ▼                                                          │   │
│  │  Tokenize & Check Length                                         │   │
│  │       │                                                          │   │
│  │       ├─→ ≤ 32K tokens: Process directly                         │   │
│  │       │                                                          │   │
│  │       └─→ > 32K tokens: Chunk into 32K pieces                   │   │
│  │              │                                                   │   │
│  │              ▼                                                   │   │
│  │         Process each chunk                                       │   │
│  │              │                                                   │   │
│  │              ▼                                                   │   │
│  │         Aggregate results                                        │   │
│  │       │                                                          │   │
│  │       ▼                                                          │   │
│  │  ModernBERT-base-32k + LoRA                                      │   │
│  │  - Domain classification                                         │   │
│  │  - PII detection                                                 │   │
│  │  - Jailbreak detection                                           │   │
│  │       │                                                          │   │
│  │       ▼                                                          │   │
│  │  Signal Extraction Results                                       │   │
│  │                                                                  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Component Responsibilities

| Component | Responsibility | Location |
|-----------|---------------|----------|
| **Model Loader** | Load ModernBERT-base-32k from HuggingFace | `candle-binding/src/model_architectures/traditional/modernbert.rs` |
| **Chunking Logic** | Split sequences > 32K into chunks | New module or extend existing |
| **LoRA Adapter** | Apply LoRA adapters to ModernBERT | Existing LoRA infrastructure |
| **Signal Extractor** | Extract signals using extended context | `pkg/extproc/processor_req_body.go` |

---

## 3. Model Integration

### 3.1 Model Information

**Model**: `llm-semantic-router/modernbert-base-32k`
- **HuggingFace**: https://huggingface.co/llm-semantic-router/modernbert-base-32k
- **Context Length**: 32,768 tokens
- **Parameters**: 149M
- **Architecture**: ModernBERT with RoPE
- **Base Model**: ModernBERT-base (extended from 8K to 32K using YaRN)

### 3.2 Loading Strategy

The codebase already has `TraditionalModernBertClassifier` that can load ModernBERT models:

```rust
// Existing code in modernbert.rs
TraditionalModernBertClassifier::load_from_directory(
    model_path: &str,
    use_cpu: bool,
) -> Result<Self>
```

**Steps:**
1. Download model from HuggingFace (or use cached version)
2. Use `load_from_directory()` to load the model
3. Verify model loads correctly with 32K context support
4. Test basic inference

### 3.3 Code Changes Required

| File | Change | Complexity |
|------|--------|------------|
| `candle-binding/src/model_architectures/traditional/modernbert.rs` | Test loading 32K model | Low |
| Model registry | Update to use ModernBERT-base-32k | Low |
| Configuration | Update model paths and context length | Low |

---

## 4. Chunking Strategy

### 4.1 Chunking Algorithm

For sequences longer than 32K tokens, we need to split them into chunks:

```
Input: 50,000 tokens
Chunk 1: tokens 0-32,767 (32,768 tokens)
Chunk 2: tokens 32,768-50,000 (17,232 tokens)
```

### 4.2 Chunking Approaches

#### Option 1: Simple Sequential Chunking
- Split at token boundaries
- Process each chunk independently
- Aggregate results (e.g., take max confidence, union of PII detections)

**Pros:** Simple, fast  
**Cons:** May lose context at chunk boundaries

#### Option 2: Overlapping Chunks
- Split with overlap (e.g., 512 tokens overlap)
- Process each chunk
- Deduplicate results at boundaries

**Pros:** Preserves context at boundaries  
**Cons:** More processing, need deduplication

#### Option 3: Sliding Window
- Use sliding window approach
- More sophisticated but complex

**Pros:** Best context preservation  
**Cons:** Most complex, slowest

### 4.3 Recommended Approach: Simple Sequential with Overlap

For POC, use **Option 2 (Overlapping Chunks)** with small overlap:

```
Input: 50,000 tokens
Chunk 1: tokens 0-32,767 (32,768 tokens)
Chunk 2: tokens 32,640-50,000 (17,360 tokens)  // 128 token overlap
```

**Rationale:**
- Simple to implement
- Preserves some context at boundaries
- Good balance between complexity and quality

### 4.4 Result Aggregation

Different signal types need different aggregation:

| Signal Type | Aggregation Strategy |
|-------------|---------------------|
| **Domain Classification** | Take highest confidence across chunks |
| **PII Detection** | Union of all detected PII (deduplicate by position) |
| **Jailbreak Detection** | Take max (if any chunk detects jailbreak, flag it) |

### 4.5 Code Changes Required

| File | Change | Complexity |
|------|--------|------------|
| New module: `pkg/extproc/chunking.go` | Implement chunking logic | Medium |
| `pkg/extproc/processor_req_body.go` | Integrate chunking before processing | Low |
| Signal extraction logic | Aggregate results from chunks | Medium |

---

## 5. LoRA Adapter Compatibility

### 5.1 Current LoRA Adapters

| Adapter | Base Model | Purpose |
|---------|------------|---------|
| `lora_intent_classifier_bert-base-uncased_model` | BERT-base | Domain classification |
| `lora_pii_detector_bert-base-uncased_model` | BERT-base | PII detection |
| `lora_jailbreak_classifier_bert-base-uncased_model` | BERT-base | Jailbreak detection |

### 5.2 Compatibility Check

**Key Factors:**
- **Hidden size**: BERT-base = 768, ModernBERT-base = 768 ✅ (compatible)
- **Architecture**: Both transformer-based ✅ (likely compatible)
- **Position embeddings**: BERT uses absolute, ModernBERT uses RoPE ⚠️ (may need testing)

### 5.3 Testing Strategy

1. **Load ModernBERT-base-32k**
2. **Load existing LoRA adapters**
3. **Test inference** on sample data
4. **Compare accuracy** with BERT-base + LoRA baseline

### 5.4 If Incompatible

If LoRA adapters don't work:
- **Option A**: Retrain LoRA adapters on ModernBERT-base-32k
- **Option B**: Fine-tune ModernBERT-base-32k directly (no LoRA)

**Recommendation**: Test first, retrain only if needed.

---

## 6. Pipeline Integration

### 6.1 Integration Points

The ModernBERT-base-32k integration affects the following pipeline stages:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    EXTPROC PIPELINE INTEGRATION                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Request → Fact? → Tool? → Security → Cache → SIGNAL EXTRACTION → LLM  │
│              │       │                          ↑                       │
│              └───────┴──── signals used ──────┘                       │
│                                                                         │
│  Response ← [results] ←──────────────────────────────────┘              │
│                                                                         │
│  Signal Extraction (Modified):                                          │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │                                                                  │   │
│  │  1. Tokenize input text                                         │   │
│  │  2. Check length:                                                │   │
│  │     - ≤ 32K: Process directly with ModernBERT-32k               │   │
│  │     - > 32K: Chunk → Process each chunk → Aggregate results     │   │
│  │  3. Extract signals:                                             │   │
│  │     - Domain classification                                      │   │
│  │     - PII detection                                              │   │
│  │     - Jailbreak detection                                        │   │
│  │  4. Return aggregated results                                    │   │
│  │                                                                  │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Signal Extraction Flow

| Stage | Current (BERT 512) | With ModernBERT-32k |
|-------|-------------------|---------------------|
| **Tokenization** | Tokenize full text | Tokenize full text |
| **Length Check** | Truncate to 512 | Check if > 32K, chunk if needed |
| **Model Processing** | BERT-base (512) | ModernBERT-32k (up to 32K) |
| **Result Aggregation** | Single result | Aggregate if chunked |
| **Signal Output** | Domain, PII, Jailbreak | Domain, PII, Jailbreak |

### 6.3 Backward Compatibility

| Aspect | Compatibility | Notes |
|--------|--------------|-------|
| **Short sequences (< 512)** | ✅ Fully compatible | Processed normally, no changes |
| **Medium sequences (512-32K)** | ✅ Improved | More context available |
| **Long sequences (> 32K)** | ✅ New capability | Chunking enabled |
| **API contracts** | ✅ No breaking changes | Same output format |
| **Configuration** | ⚠️ Model path changes | Update config.yaml |

---

## 7. Implementation Details

### 7.1 Model Loading (Rust)

#### File: `candle-binding/src/model_architectures/traditional/modernbert.rs`

**Test loading ModernBERT-base-32k:**

```rust
// Test function to verify 32K model loads
pub fn test_load_modernbert_32k(model_path: &str) -> Result<()> {
    let classifier = TraditionalModernBertClassifier::load_from_directory(
        model_path,
        true, // use_cpu
    )?;
    
    // Verify config has 32K context
    assert!(classifier.config.max_position_embeddings >= 32768);
    
    Ok(())
}
```

### 7.2 Chunking Implementation (Go)

#### File: `src/semantic-router/pkg/extproc/chunking.go` (new)

```go
package extproc

const (
    MaxContextLength = 32768
    ChunkOverlap     = 128  // Overlap between chunks
)

// ChunkText splits text into chunks of max MaxContextLength tokens
func ChunkText(tokens []int32, maxLength int) [][]int32 {
    if len(tokens) <= maxLength {
        return [][]int32{tokens}
    }
    
    var chunks [][]int32
    start := 0
    
    for start < len(tokens) {
        end := start + maxLength
        if end > len(tokens) {
            end = len(tokens)
        }
        
        chunk := tokens[start:end]
        chunks = append(chunks, chunk)
        
        // Move start with overlap
        start = end - ChunkOverlap
        if start >= len(tokens) {
            break
        }
    }
    
    return chunks
}
```

### 7.3 Integration in Request Processing

#### File: `src/semantic-router/pkg/extproc/processor_req_body.go`

**Add chunking before signal extraction:**

```go
func (p *RequestBodyProcessor) extractSignals(ctx *RequestContext) error {
    // ... existing code ...
    
    // Tokenize input
    tokens := tokenize(text)
    
    // Check if chunking needed
    if len(tokens) > MaxContextLength {
        chunks := ChunkText(tokens, MaxContextLength)
        
        var allResults []SignalResult
        for _, chunk := range chunks {
            results := processChunk(chunk)
            allResults = append(allResults, results...)
        }
        
        // Aggregate results
        aggregated := aggregateResults(allResults)
        return aggregated
    } else {
        // Process normally
        return processNormally(tokens)
    }
}
```

### 7.4 Registry Updates

#### File: `src/semantic-router/pkg/config/registry.go`

**Update model specifications:**

```go
var DefaultModelRegistry = []ModelSpec{
    // Domain/Intent Classification
    {
        LocalPath:        "models/mom-domain-classifier",
        RepoID:           "llm-semantic-router/modernbert-base-32k",  // Updated
        // ... existing fields ...
        MaxContextLength: 32768,  // Updated from 512
        // ... existing fields ...
    },
    
    // PII Detection
    {
        LocalPath:        "models/mom-pii-classifier",
        RepoID:           "llm-semantic-router/modernbert-base-32k",  // Updated
        // ... existing fields ...
        MaxContextLength: 32768,  // Updated from 512
        // ... existing fields ...
    },
    
    // Jailbreak Detection
    {
        LocalPath:        "models/mom-jailbreak-classifier",
        RepoID:           "llm-semantic-router/modernbert-base-32k",  // Updated
        // ... existing fields ...
        MaxContextLength: 32768,  // Updated from 512
        // ... existing fields ...
    },
}
```

---

## 8. Data Structures

### 7.1 Chunking Result

```go
type ChunkResult struct {
    ChunkIndex    int
    Tokens        []int32
    StartPosition int
    EndPosition   int
    Signals       SignalResult
}

type AggregatedResult struct {
    DomainClassification *DomainResult
    PIIDetections        []PIIDetection
    JailbreakDetected    bool
    Confidence           float64
}
```

### 7.2 Signal Result (Extended)

```go
type SignalResult struct {
    Domain        string
    PII           []PIIDetection
    Jailbreak     bool
    Confidence    float64
    ChunkIndex    int  // New: which chunk this came from
}
```

---

## 9. Configuration

### 8.1 Model Configuration

```yaml
# config.yaml
classifier:
  category_model:
    model_id: "llm-semantic-router/modernbert-base-32k"
    use_modernbert: true
    max_context_length: 32768
    threshold: 0.6
    use_cpu: true
    # Chunking configuration
    chunking:
      enabled: true
      max_chunk_size: 32768
      overlap: 128
```

### 8.2 Chunking Configuration

```yaml
chunking:
  enabled: true
  max_chunk_size: 32768
  overlap: 128
  aggregation_strategy: "max_confidence"  # or "union", "majority"
```

---

## 10. Failure Modes and Fallbacks (POC)

| Failure Mode | Detection | Fallback | Impact |
|--------------|-----------|----------|--------|
| **Model loading fails** | Error during load | Fall back to BERT-base | Feature disabled |
| **LoRA incompatibility** | Low accuracy or errors | Retrain LoRA adapters | Additional work needed |
| **Memory OOM** | Out of memory error | Reduce chunk size or use CPU | Slower processing |
| **Chunking loses context** | Poor accuracy on chunked text | Increase overlap or use sliding window | More processing time |
| **32K too slow** | Latency > threshold | Reduce to 16K or 8K | Less context |

---

## 11. Success Criteria (POC)

### 10.1 Technical Success

- [ ] **Model loads successfully** - ModernBERT-base-32k loads without errors
- [ ] **32K context works** - Model processes sequences up to 32K tokens
- [ ] **Chunking works** - Sequences > 32K are properly chunked and processed
- [ ] **LoRA adapters work** - Existing adapters function correctly (or retrained if needed)
- [ ] **Results aggregated** - Chunk results are properly aggregated

### 10.2 Performance Success

- [ ] **Signal extraction accuracy** - Maintained or improved on long sequences
- [ ] **Short sequence performance** - No significant degradation on 512-token sequences
- [ ] **Latency** - Acceptable for 32K tokens (< 200ms target)
- [ ] **Memory usage** - Within available resources (~16GB GPU or CPU fallback)

### 10.3 Compatibility Success

- [ ] **LoRA adapters compatible** - Work without modification OR retrained successfully
- [ ] **No breaking changes** - Existing workflows continue to work
- [ ] **Model registry updated** - Metadata reflects new context window

### 10.4 Metrics to Track

| Metric | Baseline (BERT 512) | Target (ModernBERT 32K) | Measurement |
|--------|---------------------|-------------------------|-------------|
| **Domain classification accuracy (512 tokens)** | ~0.95 | ≥ 0.90 | Test set |
| **Domain classification accuracy (16K tokens)** | N/A | ≥ 0.85 | Test set |
| **PII detection F1 (512 tokens)** | ~0.90 | ≥ 0.85 | Test set |
| **PII detection F1 (16K tokens)** | N/A | ≥ 0.80 | Test set |
| **Jailbreak detection accuracy (512 tokens)** | ~0.95 | ≥ 0.90 | Test set |
| **Jailbreak detection accuracy (16K tokens)** | N/A | ≥ 0.85 | Test set |
| **Inference latency (512 tokens)** | ~10ms | ≤ 15ms | Benchmark |
| **Inference latency (16K tokens)** | N/A | ≤ 150ms | Benchmark |
| **Inference latency (32K tokens)** | N/A | ≤ 300ms | Benchmark |
| **Memory usage (32K tokens)** | N/A | ≤ 16GB GPU or CPU fallback | Benchmark |

### 10.5 Additional Evaluation Metrics

Based on project manager feedback, the following metrics are critical for production deployment:

#### 10.5.1 Chunking Threshold Analysis

**Goal:** Identify the optimal chunking threshold that balances latency vs. accuracy under high concurrency.

| Metric | Description | Measurement |
|--------|-------------|-------------|
| **Latency vs. Context Length** | Measure inference latency across different context lengths (1K, 4K, 8K, 16K, 32K tokens) | Benchmark with varying context lengths |
| **Concurrency Impact** | Measure latency degradation under concurrent requests (1, 10, 50, 100 concurrent requests) | Load testing with concurrent requests |
| **Accuracy vs. Chunking** | Compare accuracy of full context vs. chunked processing for sequences > 32K | Test set with long sequences |
| **Pivot Point Identification** | Identify context length where chunking becomes beneficial (latency/accuracy trade-off) | Empirical analysis of latency curves |

**Deliverables:**
- Latency curves for different context lengths under various concurrency levels
- Chunking threshold recommendation (e.g., "chunk at 24K tokens when > 20 concurrent requests")
- Heuristics for dynamic chunking based on system load

#### 10.5.2 CPU/GPU Selection Optimization

**Goal:** Identify optimal device selection (CPU vs. GPU) based on context length and kernel performance.

| Metric | Description | Measurement |
|--------|-------------|-------------|
| **GPU Latency by Context Length** | Measure GPU inference latency across context lengths (1K, 8K, 16K, 32K) | GPU benchmark |
| **CPU Latency by Context Length** | Measure CPU inference latency across context lengths (1K, 8K, 16K, 32K) | CPU benchmark |
| **GPU Memory Usage** | Track GPU memory consumption by context length | GPU memory profiling |
| **CPU Memory Usage** | Track CPU memory consumption by context length | CPU memory profiling |
| **Kernel Performance** | Compare different kernel implementations (if available) | Kernel-specific benchmarks |
| **Device Selection Heuristics** | Identify context length thresholds for CPU vs. GPU selection | Empirical analysis |

**Deliverables:**
- Performance comparison matrix (CPU vs. GPU by context length)
- Device selection heuristics (e.g., "use GPU for > 8K tokens, CPU for < 4K tokens")
- Resource utilization profiles for both devices

#### 10.5.3 Retrieval Accuracy for Long Context

**Goal:** Evaluate retrieval accuracy degradation with long context and identify mitigation strategies.

**Model Card Baseline (from [HuggingFace model card](https://huggingface.co/llm-semantic-router/modernbert-base-32k)):**

According to the model card, the model shows significant retrieval accuracy degradation at longer distances:

**Distance-Based Retrieval Accuracy:**
| Distance | Accuracy |
|----------|----------|
| 64 tokens | 80% |
| 128 tokens | 60% |
| 256 tokens | 73% |
| 512 tokens | **27%** |
| 1024 tokens | 40% |
| 2048 tokens | 47% |
| 4096 tokens | 53% |
| 8192 tokens | **20%** |

**Position Accuracy:**
- Position Accuracy (early): **90%**
- Position Accuracy (late): **71%**

**Passkey Retrieval:**
- Passkey Retrieval @ 512 tokens: **60%**
- Passkey Retrieval @ 1K tokens: **50%**

**Key Limitations (from model card):**
1. **Passkey Retrieval at Long Distances**: The model struggles with needle-in-haystack retrieval beyond ~1K tokens. This is a fundamental limitation of MLM architectures.
2. **Context Utilization**: At very long contexts (16K+), the model may not always benefit from additional context for MLM predictions.

| Metric | Description | Measurement |
|--------|-------------|-------------|
| **Retrieval Accuracy by Position** | Measure accuracy for information at different positions in context (beginning, middle, end) | Test set with labeled positions |
| **Retrieval Accuracy by Context Length** | Compare accuracy across context lengths (512, 1K, 4K, 8K, 16K, 32K tokens) | Production dataset evaluation |
| **Distance-Based Retrieval** | Measure accuracy at specific distances (64, 128, 256, 512, 1K, 2K, 4K, 8K tokens) | Test set matching model card evaluation |
| **Long Context Degradation** | Quantify accuracy degradation for sequences > 16K tokens | Comparison with model card baseline |
| **Production Dataset Evaluation** | Test on real-world production datasets with long contexts | Production dataset testing |
| **Mitigation Strategies** | Evaluate techniques to improve long-context retrieval (e.g., attention mechanisms, chunking strategies) | Comparative analysis |

**Deliverables:**
- Retrieval accuracy curves by context length and position
- Distance-based retrieval accuracy results (matching model card format)
- Production dataset evaluation results
- Recommendations for mitigating accuracy degradation
- Comparison with model card reported accuracy (baseline validation)

---

## 12. Implementation Plan

### Phase 1: Model Integration

**Goal:** Load and test ModernBERT-base-32k model.

**Tasks:**
1. [ ] Download `llm-semantic-router/modernbert-base-32k` from HuggingFace
2. [ ] Test loading with `TraditionalModernBertClassifier::load_from_directory()`
3. [ ] Verify 32K context support (check config)
4. [ ] Test basic inference on sample sequences (512, 1K, 8K, 16K, 32K tokens)
5. [ ] Document any issues or needed adjustments

**Deliverables:**
- Model loads successfully from HuggingFace
- Basic inference works on sample sequences
- Test results documented (latency, memory usage, accuracy)
- Any code adjustments documented

### Phase 2: LoRA Compatibility Testing

**Goal:** Verify existing LoRA adapters work with ModernBERT-base-32k.

**Tasks:**
1. [ ] Load ModernBERT-base-32k
2. [ ] Load existing LoRA adapters:
   - [ ] Domain classifier LoRA
   - [ ] PII detector LoRA
   - [ ] Jailbreak detector LoRA
3. [ ] Test inference with LoRA adapters
4. [ ] Compare accuracy with BERT-base baseline
5. [ ] Document compatibility results

**Deliverables:**
- Compatibility test results (accuracy comparison with baseline)
- Performance metrics (latency, memory usage)
- If incompatible: detailed retraining plan
- Decision document: compatible vs. retrain needed

### Phase 3: Chunking Implementation

**Goal:** Implement chunking logic for sequences > 32K.

**Tasks:**
1. [ ] Create `chunking.go` module
2. [ ] Implement chunking algorithm with overlap
3. [ ] Integrate chunking into request processing
4. [ ] Implement result aggregation:
   - [ ] Domain classification aggregation
   - [ ] PII detection aggregation (union with deduplication)
   - [ ] Jailbreak detection aggregation (max)
5. [ ] Test chunking on various sequence lengths

**Deliverables:**
- Chunking module implemented (`chunking.go`)
- Integration complete in request processing pipeline
- Unit tests for chunking logic
- Integration tests for chunked processing
- Tests passing with various sequence lengths

### Phase 4: Registry & Configuration Updates

**Goal:** Update model registry and configuration.

**Tasks:**
1. [ ] Update `registry.go` with ModernBERT-base-32k
2. [ ] Update `MaxContextLength` to 32768
3. [ ] Update `config.yaml` with new model paths
4. [ ] Add chunking configuration
5. [ ] Test configuration loading

**Deliverables:**
- Registry updated with ModernBERT-base-32k specifications
- Configuration files updated (`config.yaml`)
- Configuration validation tests passing
- Backward compatibility verified (old configs still work)

### Phase 5: Testing & Validation

**Goal:** Comprehensive testing of the integrated solution.

**Testing Strategy:**

| Test Category | Approach | Success Criteria |
|--------------|----------|------------------|
| **Unit Tests** | Test chunking, aggregation functions in isolation | All unit tests pass |
| **Integration Tests** | Test model loading, inference, chunking pipeline | Integration tests pass |
| **Performance Tests** | Benchmark latency, memory for various lengths | Meet target metrics |
| **Accuracy Tests** | Compare signal extraction accuracy vs. baseline | Maintain or improve accuracy |
| **Edge Case Tests** | Test boundary conditions, error scenarios | Graceful handling |

**Tasks:**
1. [ ] **Backward Compatibility Testing**
   - [ ] Test with 512-token sequences
   - [ ] Verify accuracy maintained (≥ 0.90 for domain, ≥ 0.85 for PII)
   - [ ] Test existing workflows continue to work
   - [ ] Verify no breaking changes
   
2. [ ] **Extended Context Testing**
   - [ ] Test with various lengths (1K, 8K, 16K, 32K tokens)
   - [ ] Test chunking with sequences > 32K (35K, 50K, 100K tokens)
   - [ ] Verify chunking preserves context (overlap works)
   - [ ] Test with information at different positions (beginning, middle, end)
   
3. [ ] **Performance Benchmarking**
   - [ ] Measure latency for different lengths (512, 1K, 8K, 16K, 32K)
   - [ ] Measure memory usage (GPU and CPU)
   - [ ] Compare with baseline (BERT-base 512)
   - [ ] Document performance characteristics
   
4. [ ] **Signal Extraction Testing**
   - [ ] Test domain classification on long sequences
   - [ ] Test PII detection on long sequences (various PII types)
   - [ ] Test jailbreak detection on long sequences
   - [ ] Test with information at different positions (early, middle, late)
   - [ ] Test chunked sequences (verify aggregation works)
   
5. [ ] **End-to-End Integration**
   - [ ] Test full pipeline (request → chunking → processing → aggregation → response)
   - [ ] Test error handling (model load failure, OOM, invalid input)
   - [ ] Test edge cases (empty input, very short input, exactly 32K tokens)
   - [ ] Test concurrent requests
   
6. [ ] **LoRA Adapter Testing** (if retrained)
   - [ ] Test each adapter (domain, PII, jailbreak)
   - [ ] Compare accuracy with original adapters
   - [ ] Test performance impact

### Phase 6: Advanced Evaluation Metrics

**Goal:** Collect empirical data for production deployment decisions.

**Tasks:**

1. [ ] **Chunking Threshold Analysis**
   - [ ] Measure inference latency across context lengths (1K, 4K, 8K, 16K, 24K, 32K tokens)
   - [ ] Test latency under concurrent requests (1, 10, 50, 100 concurrent requests)
   - [ ] Compare accuracy of full context vs. chunked processing for sequences > 32K
   - [ ] Identify pivot point where chunking becomes beneficial (latency/accuracy trade-off)
   - [ ] Document latency curves and chunking threshold recommendations
   - [ ] Create heuristics for dynamic chunking based on system load

2. [ ] **CPU/GPU Selection Optimization**
   - [ ] Benchmark GPU inference latency across context lengths (1K, 8K, 16K, 32K tokens)
   - [ ] Benchmark CPU inference latency across context lengths (1K, 8K, 16K, 32K tokens)
   - [ ] Profile GPU memory consumption by context length
   - [ ] Profile CPU memory consumption by context length
   - [ ] Compare different kernel implementations (if available)
   - [ ] Identify context length thresholds for CPU vs. GPU selection
   - [ ] Create device selection heuristics based on context length and system load

3. [ ] **Retrieval Accuracy for Long Context**
   - [ ] Measure retrieval accuracy for information at different positions (beginning, middle, end)
   - [ ] Compare accuracy across context lengths (512, 1K, 4K, 8K, 16K, 32K tokens)
   - [ ] Quantify accuracy degradation for sequences > 16K tokens
   - [ ] Evaluate on production datasets with long contexts
   - [ ] Compare results with model card reported accuracy
   - [ ] Test mitigation strategies (e.g., attention mechanisms, chunking strategies)
   - [ ] Document recommendations for improving long-context retrieval

**Deliverables:**
- **Chunking Threshold Report:**
  - Latency curves for different context lengths under various concurrency levels
  - Chunking threshold recommendation (e.g., "chunk at 24K tokens when > 20 concurrent requests")
  - Heuristics for dynamic chunking based on system load
  
- **CPU/GPU Selection Report:**
  - Performance comparison matrix (CPU vs. GPU by context length)
  - Device selection heuristics (e.g., "use GPU for > 8K tokens, CPU for < 4K tokens")
  - Resource utilization profiles for both devices
  
- **Retrieval Accuracy Report:**
  - Retrieval accuracy curves by context length and position
  - Production dataset evaluation results
  - Recommendations for mitigating accuracy degradation
  - Comparison with model card reported accuracy

**Deliverables:**
- Comprehensive test results report
- Performance benchmarks (latency, memory, accuracy)
- Comparison with baseline (BERT-base 512)
- Validation that success criteria are met
- Edge case testing results
- End-to-end integration test results

### Phase 7: LoRA Retraining (If Needed)

**Goal:** Retrain LoRA adapters if incompatible.

**Tasks:**
1. [ ] Prepare training dataset
2. [ ] Retrain domain classifier LoRA
3. [ ] Retrain PII detector LoRA
4. [ ] Retrain jailbreak detector LoRA
5. [ ] Validate retrained adapters

**Deliverables:**
- Retrained LoRA adapters (if needed)
- Training logs and metrics
- Validation results (accuracy, latency)
- Comparison with original adapters
- Deployment-ready adapters


---

## 13. Known POC Limitations (Explicitly Deferred)

| Limitation | Impact | Why Acceptable |
|------------|--------|----------------|
| **No production error handling** | Basic error handling only | POC focuses on technical feasibility |
| **No monitoring/metrics** | Limited observability | Manual testing sufficient for POC |
| **No performance optimization** | May be slower than optimized version | Validate feasibility first, optimize later |
| **Chunking overlap fixed** | 128 token overlap may not be optimal | Can tune in production |
| **No adaptive chunking** | Fixed 32K chunk size | Simple approach for POC validation |
| **No result caching** | Chunks reprocessed on each request | Acceptable for POC scale |
| **Limited memory management** | Assumes sufficient GPU/CPU memory | POC testing on controlled environment |
| **No backward compatibility testing** | Limited testing of existing workflows | Focus on new capability validation |

---

## 14. Future Enhancements

### Short-term (Post-POC)

| Feature | Description | Priority |
|---------|-------------|----------|
| **Smart chunking** | Overlap based on sentence boundaries | Medium |
| **Result caching** | Cache chunk results to avoid reprocessing | Medium |
| **Dynamic chunk size** | Adjust chunk size based on available memory | Low |
| **Production error handling** | Comprehensive error handling and recovery | High |
| **Performance optimization** | Optimize inference latency and memory usage | High |
| **Monitoring and metrics** | Add observability for production use | High |

### Long-term

| Feature | Description | Priority |
|---------|-------------|----------|
| **Larger context windows** | Extend beyond 32K if needed | Low |
| **Multi-model ensemble** | Use multiple models for better accuracy | Low |
| **Adaptive chunking** | Adjust chunking strategy based on content | Low |
| **Hybrid processing** | Combine short and long context models | Low |

---

## Appendices

### Appendix A: Model Architecture Details

**ModernBERT-base-32k Architecture:**

| Component | Specification |
|-----------|--------------|
| **Base Model** | ModernBERT-base |
| **Parameters** | 149M |
| **Hidden Size** | 768 |
| **Layers** | 12 |
| **Attention Heads** | 12 |
| **Context Length** | 32,768 tokens (extended from 8K) |
| **Position Embeddings** | RoPE (Rotary Position Embeddings) |
| **Extension Method** | YaRN (Yet another RoPE extensioN) |

**RoPE Configuration:**
- Base frequency: 10,000
- Scaling factor: Applied via YaRN
- Supports up to 32K tokens with maintained quality

**Comparison with BERT-base:**

| Feature | BERT-base | ModernBERT-base-32k |
|---------|-----------|---------------------|
| **Context Window** | 512 tokens | 32,768 tokens (64x) |
| **Position Embeddings** | Absolute | RoPE (relative) |
| **Parameters** | 110M | 149M |
| **Architecture** | Transformer | ModernBERT (improved) |

### Appendix B: Chunking Examples

**Example 1: Simple Sequential Chunking**
```
Input: 50,000 tokens
Chunk 1: [0, 32767] (32,768 tokens)
Chunk 2: [32640, 50000] (17,360 tokens, 128 overlap)

Processing:
  - Chunk 1 processed independently
  - Chunk 2 processed independently (with 128 token overlap)
  - Results aggregated
```

**Example 2: Result Aggregation - Domain Classification**
```
Chunk 1: Domain="technology" (confidence=0.85)
Chunk 2: Domain="technology" (confidence=0.90)
Result: Domain="technology" (confidence=0.90)  // Max confidence
```

**Example 3: Result Aggregation - PII Detection**
```
Chunk 1: PII=[email1@example.com, phone1: +1-555-0100]
Chunk 2: PII=[email2@example.com]
Result: PII=[email1@example.com, phone1: +1-555-0100, email2@example.com]  // Union
```

**Example 4: Result Aggregation - Jailbreak Detection**
```
Chunk 1: Jailbreak=false (confidence=0.1)
Chunk 2: Jailbreak=true (confidence=0.95)
Result: Jailbreak=true  // Max (any detection flags it)
```

**Example 5: Overlap Boundary Handling**
```
Input: 35,000 tokens
Chunk 1: [0, 32767] (32,768 tokens)
Chunk 2: [32640, 35000] (23,360 tokens, 128 overlap)

Overlap region: [32640, 32767] (128 tokens)
  - Processed in both chunks
  - Deduplicated in aggregation
  - Ensures context continuity at boundary
```

### Appendix C: File Tree

```
candle-binding/src/model_architectures/traditional/
├── modernbert.rs                    (EXTEND) Test 32K model loading
│   └── TraditionalModernBertClassifier::load_from_directory()
│       └── Verify 32K context support

src/semantic-router/pkg/extproc/
├── chunking.go                      (NEW) Chunking logic
│   ├── ChunkText() - Split tokens into chunks
│   ├── ChunkWithOverlap() - Chunk with overlap
│   └── Constants: MaxContextLength, ChunkOverlap
├── processor_req_body.go            (MODIFY) Integrate chunking
│   └── extractSignals() - Add chunking before processing
├── aggregation.go                   (NEW) Result aggregation
│   ├── AggregateDomainResults() - Max confidence
│   ├── AggregatePIIDetections() - Union with deduplication
│   └── AggregateJailbreakResults() - Max detection
└── processor_res_body.go            (NO CHANGE) Response processing

src/semantic-router/pkg/config/
└── registry.go                      (MODIFY) Update to ModernBERT-base-32k
    └── DefaultModelRegistry - Update model specs

config/
└── config.yaml                     (MODIFY) Update model paths and chunking config
    └── classifier.category_model - ModernBERT-32k config

docs/
└── issue-995-modernbert-32k-poc.md (THIS FILE)
```

### Appendix D: API Changes

**No Breaking API Changes:**

The integration maintains full backward compatibility with existing APIs:

| API Endpoint | Changes | Notes |
|--------------|---------|-------|
| `/v1/classify` | None | Same request/response format |
| Signal extraction | Internal only | No API changes |
| Configuration | Model paths updated | Backward compatible config |

**Configuration Changes:**

```yaml
# Before (BERT-base)
classifier:
  category_model:
    model_id: "bert-base-uncased"
    max_context_length: 512

# After (ModernBERT-32k)
classifier:
  category_model:
    model_id: "llm-semantic-router/modernbert-base-32k"
    max_context_length: 32768
```

**Internal Changes Only:**
- Model loading logic
- Chunking implementation
- Result aggregation
- No external API contract changes

### Appendix E: Testing Data

**Test Sequences:**

| Test Case | Length | Description | Expected Result |
|-----------|--------|-------------|-----------------|
| **Short** | 100 tokens | Normal conversation | Process normally, no chunking |
| **Medium** | 1,000 tokens | Long conversation | Process normally, no chunking |
| **Long** | 16,000 tokens | Very long conversation | Process normally, no chunking |
| **Extended** | 32,000 tokens | Maximum context | Process normally, no chunking |
| **Chunked** | 50,000 tokens | Requires chunking | Chunk into 2 pieces, aggregate |
| **Very Long** | 100,000 tokens | Multiple chunks | Chunk into 4 pieces, aggregate |

**Test Scenarios:**

1. **Domain Classification:**
   - Short: "What's the weather?" → Technology
   - Long: [15K tokens of conversation] "What's the weather?" → Technology
   - Chunked: [50K tokens] Domain mentioned in middle → Correct domain

2. **PII Detection:**
   - Short: "My email is user@example.com" → Detected
   - Long: [15K tokens] PII at end → Detected
   - Chunked: [50K tokens] PII in different chunks → All detected

3. **Jailbreak Detection:**
   - Short: "Ignore previous instructions" → Detected
   - Long: [15K tokens] Jailbreak at beginning → Detected
   - Chunked: [50K tokens] Jailbreak in middle chunk → Detected

### Appendix F: References

1. **ModernBERT-base-32k**: https://huggingface.co/llm-semantic-router/modernbert-base-32k
2. **ModernBERT Paper**: [Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder](https://arxiv.org/abs/2406.07528)
3. **YaRN Paper**: [YaRN: Yet another RoPE extensioN](https://arxiv.org/abs/2309.00071)
4. **RoPE Paper**: [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
5. **Issue #995**: [Pre-train base model with longer context window](https://github.com/vllm-project/semantic-router/issues/995)

---

*Document Author: @henschwartz*  
*Last Updated: 2026-01-20*  
*Status: POC DESIGN - v1*  
*Based on: [Issue #995 - Pre-train base model with longer context window](https://github.com/vllm-project/semantic-router/issues/995)*
