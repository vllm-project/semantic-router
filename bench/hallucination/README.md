# Hallucination Detection Benchmark

This benchmark measures hallucination detection through the router or directly
against a detector. It records accuracy metrics when labels are available,
latency, and per-sample decisions. Run artifacts are written under
`bench/hallucination/results/` and are not treated as repository baselines.

## Router evaluation

The router path exercises the complete request flow: Envoy, semantic routing,
the selected model backend, and the hallucination plugin.

### Prerequisites

- An OpenAI-compatible model server that matches the provider in
  `config-7b.yaml`. The checked-in configuration expects
  `Qwen/Qwen2.5-14B-Instruct-AWQ` at `127.0.0.1:8083`.
- The router and its classifier models built or downloaded through the normal
  repository workflow.
- Envoy listening on port `8801`.

Start the router and Envoy in separate terminals:

```bash
make run-router CONFIG_FILE=bench/hallucination/config-7b.yaml
make run-envoy
```

Then run a small evaluation:

```bash
python3 -m bench.hallucination.evaluate \
  --endpoint http://localhost:8801 \
  --dataset halueval \
  --max-samples 50 \
  --output-dir bench/hallucination/results
```

The equivalent repository target is:

```bash
make bench-hallucination MAX_SAMPLES=50
```

Use `--dataset /path/to/data.jsonl` for a custom JSONL dataset. Run
`python3 -m bench.hallucination.evaluate --help` for the complete option list.

## Configuration

The benchmark configuration uses the v0.3 provider model:

```yaml
providers:
  models:
    - name: Qwen/Qwen2.5-14B-Instruct-AWQ
      backend_refs:
        - name: vllm-general
          endpoint: 127.0.0.1:8083
          protocol: http
          type: chat
```

Classifier modules belong in the global model catalog. A prompt-guard module,
for example, has this shape:

```yaml
global:
  model_catalog:
    modules:
      prompt_guard:
        enabled: true
        model_ref: prompt_guard
        model_id: models/mmbert32k-jailbreak-detector-merged
```

Hallucination detector defaults use the same catalog:

```yaml
global:
  model_catalog:
    modules:
      hallucination_mitigation:
        enabled: true
        detector:
          model_ref: hallucination_detector
          model_id: models/mom-halugate-detector
          threshold: 0.82
          min_span_length: 2
          min_span_confidence: 0.6
          context_window_size: 50
          enable_nli_filtering: true
          nli_entailment_threshold: 0.75
```

Each routing decision enables the plugin and chooses how flagged responses are
handled. See `config-7b.yaml` for the complete, runnable example.

### Detector controls

| Setting | Default | Purpose |
| --- | ---: | --- |
| `threshold` | `0.82` | Minimum detector score for a candidate hallucination. |
| `min_span_length` | `2` | Ignore detected spans shorter than this token count. |
| `min_span_confidence` | `0.6` | Ignore spans whose confidence is below this value. |
| `context_window_size` | `50` | Include this many surrounding characters in diagnostic context. |
| `enable_nli_filtering` | `true` | Remove candidates that are entailed by the supplied context. |
| `nli_entailment_threshold` | `0.75` | Treat candidates above this entailment score as supported. |

Raise the span thresholds when false positives are more costly; lower them when
recall is more important. Any comparison should record the complete
configuration, model revision, dataset revision, and endpoint settings.

## Standalone detector comparison

`evaluate_detectors.py` evaluates supported detectors without starting the
router. Install its Python dependencies, then provide one or more detector
specifications:

```bash
python3 -m pip install lettucedetect datasets
python3 -m bench.hallucination.evaluate_detectors \
  --detector llm:KRLabsOrg/lettucedect-v2-qwen-2b \
  --base-url http://localhost:8077/v1 \
  --dataset halueval \
  --max-samples 1000
```

Use `transformer:<path>` instead of `llm:<model>` for a locally loaded encoder.
The script writes metrics and per-sample rows to the results directory.

## Agent trajectories

An agent can finish with a correct answer after an earlier step misreported a
tool result. `evaluate_trajectories.py` runs a detector on every assistant step
of a trajectory instead of only the last answer.

`testdata/agent_trajectories.json` holds a few hand-written trajectories. Each
one has the user `request` and ordered `steps`. A `tool` step records the tool
`name`, its `arguments` and its `output`. An `assistant` step records the
message `text` and its gold `unsupported_spans`, where an empty list marks the
message as supported:

```json
{
  "role": "assistant",
  "text": "All six parser tests pass, so I'll move on to the changelog.",
  "unsupported_spans": [
    {"start": 0, "end": 25, "text": "All six parser tests pass", "label": "contradicted"}
  ]
}
```

Spans follow the router's `token_spans.v1` contract. Offsets are Unicode code
points into the message, `text` must equal that slice, `label` comes from the
hallucination label set (`SUPPORTED` and `O` are rejected), and a repeated
`(label, start, end)` is an error. The loader also rejects unknown keys, so a
misspelled field fails instead of turning a step into a supported one.

Each assistant step is checked with the request as the question and, as the
context, the tool outputs before that step, joined with blank lines the way the
router's response stage joins tool results. A later passing test run therefore
cannot make an earlier false claim look supported.

```bash
python3 -m pip install lettucedetect
python3 -m bench.hallucination.evaluate_trajectories \
  --detector transformer:KRLabsOrg/lettucedect-base-modernbert-en-v1
```

`--detector` accepts the same specifications as `evaluate_detectors.py`. The
script prints one line per step and writes step-level and character-level
metrics to the results directory. The examples show that a detector runs on
every step; they are too few to compare detectors. The loader and scoring tests
need no model:

```bash
python3 -m pytest -q bench/hallucination/test_trajectories.py
```

## Reading the output

The router evaluator reports:

- precision, recall, F1, and accuracy when the dataset provides labels;
- average, median, and tail latency;
- the detector decision and available trace data for each sample.

Treat results as run-specific. Compare runs only when their dataset, models,
configuration, hardware, and service topology are equivalent.
