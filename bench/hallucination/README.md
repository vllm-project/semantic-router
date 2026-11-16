# Hallucination Detection Benchmark

This benchmark measures hallucination detection through the router or directly
against a detector. It records accuracy metrics when labels are available,
latency, and per-sample decisions. Run artifacts are written under
`bench/hallucination/results/` and are not treated as repository baselines.

## Router evaluation

The router path exercises the complete request flow: Envoy, semantic routing,
the selected model backend, and the response guard that reads the
hallucination check.

### Prerequisites

- An OpenAI-compatible model server that matches the provider in
  `config-7b.yaml`. The checked-in configuration expects
  `Qwen/Qwen2.5-14B-Instruct-AWQ` at `127.0.0.1:8083`.
- The router and its decision model built or downloaded through the normal
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

The benchmark configuration is a v0.5 document. Its provider is the model
server from the prerequisites:

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

The decision model answers the hallucination check, as it answers every
signal. The recipe declares a `fact_check` rule that marks questions that need
facts, and a `hallucination` rule that marks answers their grounding context
does not support:

```yaml
recipes:
  - name: default
    signals:
      fact_check:
        - name: needs_fact_check
          description: Query contains factual claims that should be verified against context.
      hallucination:
        - name: ungrounded_claims
          description: Answers with claims their grounding context does not support.
```

Each decision adds a `response_guard` plugin that reads both rules and chooses
how a flagged answer is handled: `annotate` adds a note and lets the answer
through, and `block` refuses it.

```yaml
plugins:
  - type: response_guard
    configuration:
      rules:
        - when:
            operator: AND
            conditions:
              - type: fact_check
                name: needs_fact_check
              - type: hallucination
                name: ungrounded_claims
          action: annotate
      on_unknown: annotate
```

See `config-7b.yaml` for the complete, runnable example.

### Rule controls

| Field | Purpose |
| --- | --- |
| `predicate` | The least probability of an unsupported span; by default the model's own span thresholds decide. |
| `deployment`, `timeout_ms` | The deployment that answers instead of the decision model, and the answer's deadline. |

Raise `predicate` when false positives are more costly; lower it when recall
is more important. Any comparison should record the complete
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

`testdata/agent_trajectories.json` holds a few trajectories, most of them
hand-written. Each one has the user `request` and ordered `steps`. A `tool` step
records the tool `name`, its `arguments` and its `output`. An `assistant` step
records the message `text` and its gold `unsupported_spans`, where an empty list
marks the message as supported:

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

A step taken before any tool has run is checked against an empty context, so a
claim such as "the tests pass" made before running them is unsupported, even
though a detector may accept it as plausible. The router does not check such a
response: with no tool output there is no grounding context, so its
`hallucination` rule is unknown and the guard's `on_unknown` decides.

Tool output gets long quickly. A detector with a short input limit, such as the
ModernBERT detector `models/mom-halugate-detector` (512 tokens), keeps the
question and the answer, and cuts the context from the end until the input
fits. Tool outputs are joined oldest first, so the most recent output is the
first to go. `--context-window 512` applies the same cut before calling the
detector. It cuts only where a context token ends, so it never keeps part of a
token. Every row records `context_chars` and `context_chars_seen`, and sets
`truncated` when the step saw only part of its context; `truncated_steps` in
the metrics counts the outcomes of those steps. A miss caused by the cut then no
longer looks like a detector miss. The router's hallucination question reads
the request, context and answer whole, up to the decision model's scan budget;
past it the rule is unknown.

Besides step-level and character-level scores, the script reports where each
trajectory first goes wrong: whether the detector flagged the earliest
unsupported step, and if not, how many assistant steps later it first flagged an
unsupported one. A flag on a supported step does not count as a catch, and a
flag before the first unsupported step is reported as a false alarm.
`mean_steps_late` averages that delay over the trajectories the detector caught.

```bash
python3 -m pip install lettucedetect
python3 -m bench.hallucination.evaluate_trajectories \
  --detector transformer:KRLabsOrg/lettucedect-base-modernbert-en-v1 \
  --context-window 512
```

`--detector` accepts the same specifications as `evaluate_detectors.py`.
`--context-window` counts tokens with the transformer detector's tokenizer; pass
`--tokenizer` for an `llm:` detector. The script prints one line per step and
per trajectory, and writes step-level, character-level and first-false-step
metrics to the results directory. The examples show that a detector runs on
every step; they are too few to compare detectors. The loader and scoring tests
need no model. When `transformers` is installed, one more test downloads the
ModernBERT tokenizer to check the cut against real token counts:

```bash
python3 -m pytest -q bench/hallucination/test_trajectories.py
```

The `flask-async-commit-misreported` trajectory combines the `grep` and
`git_log` samples for `pallets__flask-4074` from the test split of the
[LettuceDetect Grounded Hallucination Dataset](https://huggingface.co/datasets/KRLabsOrg/lettucedetect-code-hallucination)
by KRLabsOrg (Kovács et al., [GroundLM 2026](https://aclanthology.org/2026.groundlm-1.44/), [arXiv:2607.00895](https://arxiv.org/abs/2607.00895)),
revision `866a7c5392c3cf87e4fbc2b3808815d524f54331`, used under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The request joins the
two samples' questions, each tool output is the text inside the sample's code
fence, and the answers and spans are unchanged. The dataset's tool-output
samples are single-step, so they are a source for more multi-step trajectories.

## Reading the output

The router evaluator reports:

- precision, recall, F1, and accuracy when the dataset provides labels;
- average, median, and tail latency;
- the detector decision and available trace data for each sample.

Treat results as run-specific. Compare runs only when their dataset, models,
configuration, hardware, and service topology are equivalent.
