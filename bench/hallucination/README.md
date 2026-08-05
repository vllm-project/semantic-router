# Hallucination Detection Benchmark

E2E evaluation of hallucination detection through the semantic router.

## Quick Start

```bash
# 1. Start vLLM (if not running)
docker run -d --gpus all -p 8083:8000 vllm/vllm-openai:latest \
    vllm serve --model Qwen/Qwen2.5-14B-Instruct-AWQ

# 2. Start semantic router with hallucination config
cd /path/to/semantic-router
export LD_LIBRARY_PATH=$PWD/candle-binding/target/release
./bin/router -config=bench/hallucination/config-7b.yaml

# 3. Start Envoy
make run-envoy

# 4. Run benchmark
python3 -m bench.hallucination.evaluate \
    --endpoint http://localhost:8801 \
    --dataset halueval \
    --max-samples 50

# Or use the Makefile target:
make bench-hallucination MAX_SAMPLES=50
```

## Using the Large Model

The large model (`lettucedect-large-modernbert-en-v1`, 395M params) provides better detection accuracy than the base model.

### Step 1: Download the Large Model

```bash
cd /path/to/semantic-router

# Download from HuggingFace
hf download KRLabsOrg/lettucedect-large-modernbert-en-v1 \
    --local-dir models/lettucedect-large-modernbert-en-v1
```

### Step 2: Update Config

Edit `bench/hallucination/config-7b.yaml`:

```yaml
global:
  model_catalog:
    modules:
      hallucination_mitigation:
        detector:
          model_id: "models/lettucedect-large-modernbert-en-v1"  # Use large model
          threshold: 0.5
          use_cpu: true  # Set to false for GPU
```

### Step 3: Restart Router

```bash
# Kill existing router
pkill -f "router.*config"

# Start with updated config
export LD_LIBRARY_PATH=$PWD/candle-binding/target/release
./bin/router -config=bench/hallucination/config-7b.yaml
```

## Supported Models

| Model | Params | HuggingFace ID |
|-------|--------|----------------|
| Base | 149M | `KRLabsOrg/lettucedect-base-modernbert-en-v1` |
| Large | 395M | `KRLabsOrg/lettucedect-large-modernbert-en-v1` |

Both use `ModernBertForTokenClassification` architecture supported by candle-binding.

## Config Reference

Key settings in `config-7b.yaml`:

```yaml
providers:
  models:
    - name: Qwen/Qwen2.5-14B-Instruct-AWQ
      backend_refs:
        - endpoint: 127.0.0.1:8083
          protocol: http
          type: chat

global:
  model_catalog:
    modules:
      prompt_guard:
        enabled: true
      hallucination_mitigation:
        enabled: true
        detector:
          model_id: "models/lettucedect-large-modernbert-en-v1"
          threshold: 0.5
          use_cpu: true
        on_hallucination_detected: warn  # or block
```

### Multi-Level Filtering Parameters

The following multi-stage filtering parameters help to reduce false positives while maintaining high detection accuracy:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_span_length` | int | 1 | **Length-based filtering**: Minimum number of tokens in a detected span to report. Filters out single-token spans which are often false positives (e.g., common words, articles). Higher values = fewer false positives but may miss short hallucinations. |
| `min_span_confidence` | float | 0.0 | **Confidence-based filtering**: Minimum confidence score (0.0-1.0) required for a span to be reported. Spans below this threshold are discarded. Higher values = higher precision but lower recall. |
| `context_window_size` | int | 50 | **Context extraction**: Number of characters of surrounding context to include when reporting flagged spans. Used for debugging and understanding why a span was flagged. Does not affect detection. |
| `enable_nli_filtering` | bool | true | **NLI-based false positive reduction**: Enable Natural Language Inference filtering to reduce false positives. When enabled, detected spans are checked for entailment with the context. Highly entailed spans (paraphrases, inferences) are filtered out. |
| `nli_entailment_threshold` | float | 0.75 | **NLI filtering threshold**: Entailment score above which spans are considered false positives and filtered out. Higher values = stricter filtering (fewer spans removed). Only used when `enable_nli_filtering=true`. Range: 0.0 (filter nothing) to 1.0 (filter only perfect entailment). |

**Example: Conservative vs Sensitive Configuration**

```yaml
# Conservative: High precision, fewer false positives
min_span_length: 3
min_span_confidence: 0.7
enable_nli_filtering: true
nli_entailment_threshold: 0.7

# Sensitive: High recall, catch more hallucinations
min_span_length: 1
min_span_confidence: 0.4
enable_nli_filtering: true
nli_entailment_threshold: 0.65
```

## Datasets

| Dataset | Command |
|---------|---------|
| HaluEval | `--dataset halueval` |
| Custom | `--dataset /path/to/data.jsonl` |

## Standalone Detector Comparison (no router required)

Two scripts evaluate detectors directly, outside the router stack:

- `evaluate_detectors.py` — any detector supported by the `lettucedetect`
  library (encoder or generative via an OpenAI-compatible endpoint)
- `evaluate_halugate.py` — the halugate sentinel + detector pipeline
  (public models, paper serialization, optional threshold sweep)

```bash
pip install lettucedetect datasets
python3 -m bench.hallucination.evaluate_detectors \
    --detector llm:KRLabsOrg/lettucedect-v2-qwen-2b \
    --base-url http://localhost:8077/v1 --dataset halueval --max-samples 1000
python3 -m bench.hallucination.evaluate_halugate --max-samples 1000 --sweep
```

Example-level results on all 10,000 HaluEval QA samples (2026-07; HaluEval is
an evaluation-only benchmark — neither model saw it in training):

| Detector | Params | Config | Precision | Recall | F1 | Accuracy | p50 |
|----------|--------|--------|-----------|--------|----|----------|-----|
| `KRLabsOrg/lettucedect-v2-qwen-2b` (vLLM-served) | 2B | native | 0.962 | 0.769 | 0.855 | 0.869 | ~116ms |
| sentinel + `modernbert-base-32k-haldetect-combined` | 149M | best swept (thr 0.3, span 1) | 0.999 | 0.502 | 0.668 | 0.750 | ~25ms |
| same | 149M | production (thr 0.82, span 2) | 1.000 | 0.394 | 0.565 | 0.697 | ~25ms |

Notes: the "best swept" row selects the threshold on the evaluation data itself
(an oracle upper bound, deliberately generous to the baseline); the generative
detector runs untuned at its native decision point. Baseline latency is the
sentinel + detector pipeline as timed by `evaluate_halugate.py` (sentinel-skipped
samples contribute sentinel-only time); the sentinel passes ~100% of this
all-factual dataset (correct behavior — its efficiency win applies to mixed
workloads). Both serializations (router format and paper format) were tested;
the difference is <2 F1 points. The two models make a natural two-tier pairing:
the 149M encoder pipeline is ~4.5x faster at near-perfect precision, the 2B
generative detector catches roughly 1.5x the hallucinations and types each span.

## Output

Results saved to `bench/hallucination/results/` with:

- Precision, Recall, F1 (when ground truth available)
- Latency metrics (avg, p50, p99)
- Per-sample detection results

### Two-Stage Pipeline Efficiency Metrics

The benchmark tracks the computational savings from the two-stage detection pipeline:

```
⚡ Two-Stage Pipeline Efficiency:
----------------------------------------
  Fact-check needed:     65/100 queries
  Detection skipped:     35/100 queries
  Avg context length:    4500 chars
  Estimated detect time: 6500.00 ms (if all ran)
  Actual detect time:    4225.00 ms
  Time saved:            2275.00 ms
  Efficiency gain:       35.0%

  💡 Pre-filtering skipped 35.0% of requests,
     saving 2275ms of detection compute.
```

This demonstrates the value of the HaluGate Sentinel pre-classifier:

- **O(1) filtering** before **O(n) detection** (n = context length)
- Non-factual queries (creative, opinion, brainstorming) skip expensive token classification
- Critical for RAG applications with large contexts (8K+ tokens)
