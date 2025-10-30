# Candle-Binding Examples

This directory contains comprehensive examples demonstrating the candle-binding library functionality.

## Quick Start

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
├── qwen3_example.go         # Comprehensive Go example for Multi-LoRA classification
├── qwen3_example.rs         # Comprehensive Rust example for Multi-LoRA classification
├── qwen3_guard_example.go   # Qwen3Guard safety classification example
├── go.mod                   # Go module configuration
└── README.md                # This file
```

## What's Demonstrated

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
- `CUDA_VISIBLE_DEVICES` - Select GPU device (default: 0)
- `LD_LIBRARY_PATH` - Path to Rust library (Go only)

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

## Documentation

For detailed documentation, see:

- `../../candle-binding/ZERO_SHOT_CLASSIFICATION.md` - Zero-shot guide
- `../../candle-binding/BASE_MODEL_EXPLAINED.md` - Base model concepts
- `../../candle-binding/MULTI_ADAPTER_IMPLEMENTATION.md` - Multi-LoRA architecture
- `../../candle-binding/QUICK_START_ZERO_SHOT.md` - Quick start guide
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

| Feature | qwen3_example.go | qwen3_example.rs | qwen3_guard_example.go |
|---------|------------------|------------------|------------------------|
| Zero-shot classification | ✅ | ✅ | ❌ |
| Multi-LoRA adapters | ✅ | ✅ | ❌ |
| Benchmark evaluation | ✅ | ✅ | ❌ |
| Safety classification | ❌ | ❌ | ✅ |
| PII detection | ❌ | ❌ | ✅ |
| Jailbreak detection | ❌ | ❌ | ✅ |
| Multilingual support | ❌ | ❌ | ✅ (14 languages) |
| Accuracy metrics | ❌ | ❌ | ✅ (P/R/F1) |
| Latency tracking | ❌ | ❌ | ✅ (P50/P95/P99) |
| Error handling | ✅ | ✅ | ✅ |

**Recommendations:**

- **For classification**: Start with `qwen3_example.go` - easier to run, demonstrates FFI interface
- **For safety/moderation**: Use `qwen3_guard_example.go` - comprehensive safety classification
