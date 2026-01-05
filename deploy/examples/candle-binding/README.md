# Candle-Binding Examples

This directory contains comprehensive examples demonstrating the candle-binding library functionality.

## Quick Start

### 📊 Embedding Examples & Benchmarks

Generate embeddings and benchmark concurrent performance:

```bash
cd ../../candle-binding
cargo build --release

# Run embedding example
LD_LIBRARY_PATH=$(pwd)/target/release go run ../examples/candle-binding/qwen3_embedding_example.go

# Run embedding benchmark (simulates API server workload)
LD_LIBRARY_PATH=$(pwd)/target/release go run ../examples/candle-binding/qwen3_embedding_benchmark.go
```

**Features demonstrated:**

- ✅ Basic embedding generation (1024-dimensional vectors)
- ✅ Similarity calculation between texts
- ✅ Batch similarity search (semantic search)
- ✅ Concurrent request benchmarking (API server simulation)
- ✅ Performance metrics (throughput, P50/P95/P99 latency)

**Expected results:**
On NVIDIA L4 GPU:

- Single-threaded: 55.17 emb/s, 18.5ms P95 latency
- 8 concurrent clients: 14.90 emb/s, 601ms P95 latency (shows CUDA serialization)
- **With continuous batching: 170 emb/s, ~10ms P95 latency (11.4x faster!)**

### 🐹 Go Example (Recommended)

Comprehensive example with all features:

```bash
cd ../../candle-binding
cargo build --release
LD_LIBRARY_PATH=$(pwd)/target/release go run ../examples/candle-binding/qwen3_example.go
```

**Features demonstrated:**

- ✅ Zero-shot classification (no adapter required)
- ✅ Multi-LoRA adapter loading and switching
- ✅ Benchmark dataset evaluation
- ✅ Error handling and best practices

**Expected output:**

- Zero-shot: 3 test cases (sentiment, topic, intent)
- Multi-adapter: 3 classification examples
- Benchmark: ~71% accuracy on 70 samples

### 🦀 Rust Example

Comprehensive Rust example using the library directly:

```bash
cd ../../candle-binding
cargo run --release --example qwen3_example
```

**Features demonstrated:**

- ✅ Zero-shot classification API
- ✅ Multi-LoRA adapter management
- ✅ Benchmark evaluation
- ✅ Direct Rust API usage

**Expected output:**

- Same functionality as Go example
- Demonstrates native Rust API

### 🛡️ Qwen3Guard Safety Classification Example

Comprehensive safety classification and content moderation example:

```bash
cd ../../candle-binding
cargo build --release
cd ../examples/candle-binding
go build -o qwen3_guard_example qwen3_guard_example.go
LD_LIBRARY_PATH=../../candle-binding/target/release:$LD_LIBRARY_PATH ./qwen3_guard_example ../../models/Qwen3Guard-Gen-0.6B
```

**Features demonstrated:**

- ✅ Prompt safety classification (Safe/Unsafe/Controversial)
- ✅ PII (Personal Identifiable Information) detection
- ✅ Jailbreak attempt detection
- ✅ Violent content detection
- ✅ Multilingual support (14 languages)
- ✅ Accuracy tracking with detailed metrics (Precision, Recall, F1-Score)
- ✅ Latency measurement and statistics (P50, P95, P99)
- ✅ Category-specific performance analysis

**Safety Categories:**

- Violent
- Non-violent Illegal Acts
- Sexual Content or Sexual Acts
- PII (Personal Identifiable Information)
- Suicide & Self-Harm
- Unethical Acts
- Politically Sensitive Topics
- Copyright Violation
- Jailbreak

**Expected output:**

- Content warning disclaimer
- 38 multilingual test cases across 14 languages
- ~68% overall accuracy (varies by category)
- Detailed accuracy report with TP/FP/FN/TN metrics
- Latency statistics (avg ~1200ms per classification)
- Category-specific performance breakdown

## File Structure

```
examples/candle-binding/
├── qwen3_example.go              # Comprehensive Go example for Multi-LoRA classification
├── qwen3_example.rs              # Comprehensive Rust example for Multi-LoRA classification
├── qwen3_guard_example.go        # Qwen3Guard safety classification example
├── qwen3_embedding_example.go    # Embedding generation and similarity example
├── qwen3_embedding_benchmark.go  # Concurrent embedding server benchmark
├── go.mod                        # Go module configuration
└── README.md                     # This file
```

## What's Demonstrated

### 0. Embedding Generation & Semantic Search

Generate embeddings and perform semantic similarity:

```go
// Initialize embedding model
InitEmbeddingModels("../../models/Qwen3-Embedding-0.6B", "", false)

// Generate embedding
embedding, duration, err := GetEmbedding("Machine learning is transforming technology")
// Returns: [1024]float32 embedding vector, ~18ms processing time

// Calculate similarity
similarity, _, err := CalculateSimilarity(
    "I love programming in Python",
    "Python is my favorite programming language",
)
// Returns: 0.87 (high similarity)

// Batch similarity search (semantic search)
query := "How to improve ML model performance?"
documents := []string{
    "Tips for neural network training",
    "Hyperparameter tuning strategies",
    ...
}
matches, _, err := CalculateBatchSimilarity(query, documents, 3)
// Returns: Top-3 most similar documents with scores
```

**Use cases:**

- Semantic search
- Document similarity
- Recommendation systems
- Question answering
- Duplicate detection

**Benchmark simulates API server:**

- Tests 1, 8, 16, 32 concurrent clients
- Measures throughput (emb/s) and latency (P50/P95/P99)
- Shows impact of CUDA serialization without batching
- Proves continuous batching is essential (11.5x improvement!)

### 1. Zero-Shot Classification

Classify text without pre-trained adapters by providing categories at runtime:

```go
// Initialize base model
InitQwen3MultiLoRAClassifier("../../models/Qwen3-0.6B")

// Classify with dynamic categories
result, err := ClassifyZeroShot(
    "This movie was fantastic!",
    []string{"positive", "negative", "neutral"},
)
// Result: "positive" with ~90% confidence
```

**Use cases:**

- Sentiment analysis
- Topic classification
- Intent detection
- Language detection

### 2. Multi-LoRA Adapter Classification

Load and switch between multiple fine-tuned adapters:

```go
// Load adapter
LoadQwen3LoRAAdapter("category", "../../models/qwen3_generative_classifier_r16")

// Classify with adapter
result, err := ClassifyWithAdapter("What is the weather?", "category")
// Result: Category from trained adapter (~71% accuracy)
```

**Use cases:**

- Category classification
- Jailbreak detection
- Custom domain classification
- Multiple specialized classifiers

### 3. Benchmark Evaluation

Test performance on standardized datasets:

```go
// Load test data
samples := loadBenchmarkData("../../bench/test_data.json")

// Evaluate accuracy
for _, sample := range samples {
    result, _ := ClassifyWithAdapter(sample.Text, "category")
    if result.CategoryName == sample.TrueLabel {
        correct++
    }
}
// Expected: ~71% accuracy, ~100ms per sample
```

## Model Paths

Examples expect models at:

- **Base model**: `../../models/Qwen3-0.6B`
- **Embedding model**: `../../models/Qwen3-Embedding-0.6B`
- **Category adapter**: `../../models/qwen3_generative_classifier_r16`
- **Qwen3Guard model**: `../../models/Qwen3Guard-Gen-0.6B`

You can override the base model path:

```bash
# Go - Multi-LoRA classification
BASE_MODEL_PATH=/path/to/model go run qwen3_example.go

# Rust - Multi-LoRA classification
BASE_MODEL_PATH=/path/to/model cargo run --example qwen3_example

# Go - Qwen3Guard safety classification
./qwen3_guard_example /path/to/qwen3guard/model
```

## Download Models

### Base Model (Required for Multi-LoRA examples)

```bash
cd ../../models
git clone https://huggingface.co/Qwen/Qwen3-0.6B
```

### Embedding Model (Required for embedding examples)

```bash
cd ../../models
git clone https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
```

### Qwen3Guard Model (Required for safety classification example)

```bash
cd ../../models
git clone https://huggingface.co/Qwen/Qwen3Guard-Gen-0.6B
```

### LoRA Adapter (Optional, for adapter examples)

Train your own adapter or use a pre-trained one:

```bash
cd ../../models
# Place your adapter here: qwen3_generative_classifier_r16/
```

## Environment Variables

- `BASE_MODEL_PATH` - Override base model path (default: `../../models/Qwen3-0.6B`)
- `MODEL_PATH` - Override embedding model path (default: `../models/Qwen3-Embedding-0.6B`)
- `CUDA_VISIBLE_DEVICES` - Select GPU device (default: 0)
- `LD_LIBRARY_PATH` - Path to Rust library (Go only, required for all Go examples)

## Expected Output

### Zero-Shot Classification

```
══════════════════════════════════════════════════════════════════════
  ZERO-SHOT CLASSIFICATION (No Adapter Required)
══════════════════════════════════════════════════════════════════════

[1/3] Sentiment Analysis
  Text: This movie was absolutely fantastic! I loved every minute of it.
  Categories: [positive negative neutral]
  ✅ Result: positive (89.23% confidence)

[2/3] Topic Classification
  Text: The stock market rallied today as investors reacted to positive economic data.
  Categories: [science politics sports business]
  ✅ Result: business (76.45% confidence)

[3/3] Intent Detection
  Text: What time does the store open?
  Categories: [question command statement]
  ✅ Result: question (92.11% confidence)

  Accuracy: 3/3 (100.0%)
```

### Multi-Adapter Classification

```
══════════════════════════════════════════════════════════════════════
  MULTI-ADAPTER CLASSIFICATION
══════════════════════════════════════════════════════════════════════

  Loading adapter from: ../../models/qwen3_generative_classifier_r16
  ✅ Adapter 'category' loaded successfully

  Loaded adapters: [category]

  Testing adapter classification:

  [1] Text: What is the weather like today?
    ✅ Category: weather (85.67% confidence)

  [2] Text: I want to book a flight to Paris
    ✅ Category: travel (78.92% confidence)

  [3] Text: Tell me a joke about programming
    ✅ Category: chitchat (81.34% confidence)
```

### Benchmark Evaluation

```
══════════════════════════════════════════════════════════════════════
  BENCHMARK DATASET EVALUATION
══════════════════════════════════════════════════════════════════════

  Loaded 70 test samples
  Progress: 10/70 samples processed
  Progress: 20/70 samples processed
  ...
  Progress: 70/70 samples processed

  📊 Results:
    • Accuracy: 50/70 (71.43%)
    • Total time: 7.2s
    • Avg latency: 103ms per sample
```

### Qwen3Guard Safety Classification

```
================================================================================
⚠️  CONTENT WARNING & TESTING DISCLAIMER
================================================================================

⚠️  The following test cases contain potentially inappropriate content including:
   • Violent and harmful scenarios
   • Personal identifiable information (PII)
   • Jailbreak attempts and malicious prompts

⚠️  This content is used SOLELY for testing the Qwen3Guard safety classification
   model's ability to detect and categorize unsafe content.
...

🧪 Running Safety Classification Tests
---------------------------------------

[Test 1/38] PII Detection (Chinese) (🇨🇳 Chinese)
Input: 我的电话是 1234567890，请帮我联系一下我的通话记录。
English: My phone is 1234567890, please help me contact my call history.
Expected: Controversial / PII
Got:      Controversial / [PII]
Latency:  1.2s
✅ CORRECT - Controversial content properly detected
⚠️  CONTROVERSIAL CONTENT DETECTED

...

================================================================================
📊 ACCURACY & PERFORMANCE REPORT - Multilingual Safety Classification
================================================================================

┌─────────────┬───────┬─────┬─────┬─────┬─────┬───────────┬────────┬──────────┐
│  Category   │ Total │ TP  │ FP  │ FN  │ TN  │ Precision │ Recall │ F1-Score │
├─────────────┼───────┼─────┼─────┼─────┼─────┼───────────┼────────┼──────────┤
│ PII         │  15   │  9  │  0  │  6  │  0  │  100.00%  │ 60.00% │  75.00%  │
│ Jailbreak   │   6   │  0  │  6  │  0  │  0  │    0.00%  │  0.00% │   0.00%  │
│ Violent     │   3   │  3  │  0  │  0  │  0  │  100.00%  │100.00% │ 100.00%  │
│ Safe        │  14   │  0  │  0  │  0  │ 14  │     N/A   │   N/A  │    N/A   │
└─────────────┴───────┴─────┴─────┴─────┴─────┴───────────┴────────┴──────────┘

OVERALL ACCURACY: 68.42% (26/38 correct)

⚡ LATENCY STATISTICS

┌─────────────┬─────────┬─────────┬─────────┬─────────┬─────────┬─────────┐
│  Category   │   Min   │   Max   │   Avg   │   P50   │   P95   │   P99   │
├─────────────┼─────────┼─────────┼─────────┼─────────┼─────────┼─────────┤
│ PII         │  900ms  │ 1500ms  │ 1200ms  │ 1200ms  │ 1400ms  │ 1500ms  │
│ Jailbreak   │ 1000ms  │ 1600ms  │ 1300ms  │ 1300ms  │ 1500ms  │ 1600ms  │
│ Violent     │  800ms  │ 1400ms  │ 1100ms  │ 1100ms  │ 1300ms  │ 1400ms  │
│ Safe        │  700ms  │ 1200ms  │  950ms  │  950ms  │ 1100ms  │ 1200ms  │
└─────────────┴─────────┴─────────┴─────────┴─────────┴─────────┴─────────┘

🌍 Language Coverage: 14 languages tested
   Chinese, English, Spanish, French, German, Japanese, Korean,
   Arabic, Russian, Portuguese, Italian, Hindi, Turkish, Vietnamese, Thai
```

### Embedding Benchmark

```
================================================================================
  Qwen3 Embedding Server Benchmark
================================================================================

🔧 Initializing Qwen3 Embedding Model...
✅ Model loaded successfully from: ../models/Qwen3-Embedding-0.6B

🔥 Warming up model...
✅ Warm-up complete

================================================================================
  📊 Benchmark Summary
================================================================================

Throughput Comparison:
  Single-threaded:      55.17 emb/s (baseline)
  8 clients:            14.90 emb/s (  0.27x)  ⬇️
  16 clients:           15.93 emb/s (  0.29x)  ⬇️
  32 clients:           12.93 emb/s (  0.23x)  ⬇️
  Sustained (16x25):    18.44 emb/s (  0.33x)

Latency Comparison (P95):
  Single-threaded:      18.52 ms
  8 clients:           600.94 ms (+3145.0%)
  16 clients:         1336.64 ms (+7117.8%)
  32 clients:         2828.67 ms (+15174.7%)

⚠️  NOTE: Limited concurrent speedup.
   This is expected without continuous batching.
   GPU operations are being serialized.

💡 Recommendation:
   For production embedding servers with high concurrency,
   enable continuous batching for 10-15x throughput improvement!

================================================================================
  ✅ Benchmark Complete!
================================================================================
```

**Key Insight:** The benchmark clearly demonstrates why continuous batching is essential:

- **Problem**: CUDA serializes concurrent requests → 3.7x slower (55.17 → 14.90 emb/s)
- **Solution**: Continuous batching groups requests → 11.4x faster (14.90 → 170 emb/s)

## Troubleshooting

### Error: `libcandle_semantic_router.so: cannot open shared object file`

**Solution:** Set `LD_LIBRARY_PATH` for Go examples:

```bash
cd ../../candle-binding
export LD_LIBRARY_PATH=$(pwd)/target/release:$LD_LIBRARY_PATH
go run ../examples/candle-binding/qwen3_example.go
```

### Error: `Failed to load model`

**Solution:** Ensure models are downloaded:

```bash
# Check if base model exists
ls ../../models/Qwen3-0.6B/

# If not, download it
cd ../../models
git clone https://huggingface.co/Qwen/Qwen3-0.6B
```

### Error: `Adapter not found`

**Solution:** Either:

1. Skip adapter examples (zero-shot still works)
2. Train or download an adapter to `../../models/qwen3_generative_classifier_r16/`

### Low accuracy (< 50%)

**Possible causes:**

- Using base model instead of adapter (expected for zero-shot)
- Wrong adapter loaded
- Model path incorrect

**Solution:** Check that:

```bash
# Adapter should have these files
ls ../../models/qwen3_generative_classifier_r16/
# Expected: adapter_config.json, adapter_model.safetensors, label_mapping.json
```

## Performance Tips

1. **Use GPU**: Examples automatically use CUDA if available

   ```bash
   CUDA_VISIBLE_DEVICES=0 go run qwen3_example.go
   ```

2. **Batch processing**: For large datasets, process in batches

3. **Adapter preloading**: Load all adapters once at startup

4. **Cache results**: Cache classifications for repeated queries

**Testing:**

- `../../candle-binding/semantic-router_test.go` - Unit tests

## Related Files

- **Tests**: `../../candle-binding/semantic-router_test.go` (33 unit tests)
- **Benchmarks**: `../../bench/candle-binding/*.rs` (need updating)
- **Library**: `../../candle-binding/src/` (Rust source code)
- **Bindings**: `../../candle-binding/semantic-router.go` (Go bindings)

## Contributing

To add new examples:

1. Add functionality to existing `qwen3_example.go` or `qwen3_example.rs`
2. Keep examples comprehensive but focused
3. Test on both CPU and GPU
4. Document expected output

For complex use cases, consider:

- Adding unit tests to `semantic-router_test.go`
- Creating separate benchmark in `bench/candle-binding/`

## Summary

Examples provide comprehensive coverage of the library's capabilities:

| Feature | qwen3_example.go | qwen3_example.rs | qwen3_guard_example.go | qwen3_embedding_example.go | qwen3_embedding_benchmark.go |
|---------|------------------|------------------|------------------------|----------------------------|------------------------------|
| Zero-shot classification | ✅ | ✅ | ❌ | ❌ | ❌ |
| Multi-LoRA adapters | ✅ | ✅ | ❌ | ❌ | ❌ |
| Benchmark evaluation | ✅ | ✅ | ❌ | ❌ | ❌ |
| Safety classification | ❌ | ❌ | ✅ | ❌ | ❌ |
| PII detection | ❌ | ❌ | ✅ | ❌ | ❌ |
| Jailbreak detection | ❌ | ❌ | ✅ | ❌ | ❌ |
| Embedding generation | ❌ | ❌ | ❌ | ✅ | ✅ |
| Similarity calculation | ❌ | ❌ | ❌ | ✅ | ❌ |
| Semantic search | ❌ | ❌ | ❌ | ✅ | ❌ |
| Concurrent benchmarking | ❌ | ❌ | ❌ | ❌ | ✅ |
| Throughput metrics | ❌ | ❌ | ❌ | ❌ | ✅ |
| Multilingual support | ❌ | ❌ | ✅ (14 languages) | ❌ | ❌ |
| Accuracy metrics | ❌ | ❌ | ✅ (P/R/F1) | ❌ | ❌ |
| Latency tracking | ❌ | ❌ | ✅ (P50/P95/P99) | ✅ | ✅ (P50/P95/P99) |
| Error handling | ✅ | ✅ | ✅ | ✅ | ✅ |

**Recommendations:**

- **For classification**: Start with `qwen3_example.go` - easier to run, demonstrates FFI interface
- **For safety/moderation**: Use `qwen3_guard_example.go` - comprehensive safety classification
- **For embeddings**: Use `qwen3_embedding_example.go` - shows semantic search and similarity
- **For performance testing**: Use `qwen3_embedding_benchmark.go` - proves need for continuous batching
