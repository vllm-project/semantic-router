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
./bin/router -config=bench/hallucination_bench/config.yaml

# 3. Start Envoy
func-e run --config-path config/envoy.yaml

# 4. Run benchmark
python -m bench.hallucination_bench.evaluate \
    --endpoint http://localhost:8801 \
    --dataset halueval \
    --max-samples 50
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

Edit `bench/hallucination_bench/config.yaml`:

```yaml
hallucination_mitigation:
  enabled: true
  
  hallucination_model:
    model_id: "models/lettucedect-large-modernbert-en-v1"  # Use large model
    threshold: 0.5
    use_cpu: true  # Set to false for GPU
```

### Step 3: Restart Router

```bash
# Kill existing router
pkill -f "router.*config.yaml"

# Start with updated config
export LD_LIBRARY_PATH=$PWD/candle-binding/target/release
./bin/router -config=bench/hallucination_bench/config.yaml
```

## Supported Models

| Model | Params | HuggingFace ID |
|-------|--------|----------------|
| Base | 149M | `KRLabsOrg/lettucedect-base-modernbert-en-v1` |
| Large | 395M | `KRLabsOrg/lettucedect-large-modernbert-en-v1` |

Both use `ModernBertForTokenClassification` architecture supported by candle-binding.

## Config Reference

Key settings in `config.yaml`:

```yaml
# vLLM endpoint
vllm_endpoints:
  - name: "vllm-qwen"
    address: "127.0.0.1"
    port: 8083

# Hallucination detection
hallucination_mitigation:
  enabled: true
  hallucination_model:
    model_id: "models/lettucedect-large-modernbert-en-v1"
    threshold: 0.5
    use_cpu: true
  on_hallucination_detected: "warn"  # or "block"
```

## Datasets

| Dataset | Command |
|---------|---------|
| HaluEval | `--dataset halueval` |
| Custom | `--dataset /path/to/data.jsonl` |

## Output

Results saved to `bench/hallucination_bench/results/` with:

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
