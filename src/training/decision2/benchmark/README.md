# Typed Decision Bench 2 (programmatic track)

This is a small, reproducible **programmatically verifiable** track for
Choice, Noul, and Score decisions. Each item's answer comes from a deterministic
oracle over its JSON state. It is intended to expose specific reasoning and
robustness failures during Decision 2.0 development; it is not a substitute
for human-labeled deployment data or an externally maintained benchmark.

## Families and split

The split boundary is the **generator family**, not a random row split. All
variants of a group stay together. Never train or tune on the `final` families
after freezing the model and evaluation adapters.

| Split | Family | Main target | Type |
| --- | --- | --- | --- |
| dev | `attribute_gate` | Conjunction, exclusion, no eligible option | Choice |
| dev | `rule_precedence` | Conflicting matching rules, numeric priority | Noul |
| dev | `set_reconciliation` | Requested/verified/blocked set comparison | Score |
| dev | `transition_table` | Guarded state transition | Choice |
| final | `constraint_competition` | Hard constraints plus tie-breaking | Choice |
| final | `exception_stack` | Negated facts, conjunctive exceptions, conflicts | Noul |
| final | `evidence_join` | Two-key evidence join and explicit insufficient evidence | Choice + Noul |
| final | `resource_ledger` | Ordered event replay and clamped state | Score |

Each generator instance produces four records: an anchor, a fact
counterfactual with a changed gold answer, an order permutation with unchanged
answer, and a consistent opaque-identifier relabeling with unchanged answer.
The JSONL has `group_id` and `pairs` containing stable pair IDs, relations, and
roles. Order perturbations only reorder fields that are semantically unordered;
the ledger still uses explicit `tick` order. Generated option and entity codes
are random and their criterion text does not identify the winner. The oracle
uses the state facts, not a template answer string.

## Generate

Run these commands **on the experiment host**. The generator itself needs only
Python 3.10+ and the standard library.

```bash
python3 -m benchmark.generate \
  --split dev --seed decision2-public-dev-v1 --groups-per-family 100 \
  --output runs/dev.gold.jsonl --prompts-output runs/dev.prompts.jsonl
```

For a final run, freeze the generator code, model revisions, and inference
adapters first. Create fresh private entropy and retain it securely to replay
the exact items. The seed **contents** never enter the JSONL; its SHA-256
commitment does.

```bash
python3 -c 'import secrets; open("runs/final.seed", "wb").write(secrets.token_bytes(32))'
chmod 600 runs/final.seed
python3 -m benchmark.generate \
  --split final --seed-file runs/final.seed --groups-per-family 100 \
  --output runs/final.gold.jsonl --prompts-output runs/final.prompts.jsonl
```

`--overwrite` is required to replace existing outputs. Keep the final seed,
gold file, and prompts private until all registered models have completed.
The final split rejects a literal `--seed` and requires at least 32 bytes from
`--seed-file`. Do not reuse a seed or fit on final feedback. A new code version,
seed, or selection rule creates a new benchmark instance and needs a new report.

## JSONL contract

The private file has one object per line:

```json
{
  "schema_version": "typed-decision-bench/1",
  "id": "td_...",
  "split": "dev",
  "family": "attribute_gate",
  "group_id": "g_...",
  "pairs": [{"id": "p_...", "relation": "counterfactual", "role": "anchor"}],
  "state": {},
  "questions": {"decision": {"type": "choice", "instructions": "...", "criteria": {}}},
  "gold": {"decision": {"type": "choice", "value": "...", "semantic_value": "...", "label_to_semantic": {}}},
  "provenance": {
    "source": "programmatic_synthetic",
    "suite_version": "0.1.0",
    "generator_family": "attribute_gate",
    "generator_version": 1,
    "python_version": "...",
    "rng": "python_random_MT19937",
    "instance_index": 0,
    "variant": "base",
    "oracle": "benchmark.generate.oracle:attribute_gate",
    "counterfactual_edit": "...",
    "seed_commitment_sha256": "...",
    "generator_code_sha256": "...",
    "payload_sha256": "..."
  }
}
```

The separate prompts file has **only** `id`, `state`, and `questions`. An
adapter must send only `state` and `questions` to the model; `id` is for
matching the result. Never send `gold`, split/family, pair metadata, or
provenance. For Jev-style APIs, the `questions` map uses native `choice`,
`noul`, and `score` shapes. The Score criteria are ordered arrays.

Write one standardized prediction per prompt:

```json
{
  "id": "td_...",
  "source_input_sha256": "SHA-256 of the exact state/questions payload",
  "answers": {
    "decision": {
      "type": "choice",
      "choice": "offered_label",
      "probabilities": {"offered_label": 0.9, "other_label": 0.1}
    }
  },
  "latency_ms": 42.0,
  "usage": {"input_tokens": 100, "output_tokens": 10},
  "cost_usd": 0.00001
}
```

The question keys in `answers` must match exactly. For Noul, supply `noul` as
a probability in `[0,1]`; Boolean-only outputs are invalid. Every prediction
must carry `source_input_sha256`, computed from UTF-8 JSON of exactly
`{"state": state, "questions": questions}` with insertion order preserved,
Unicode unescaped, and no spaces. The scorer rejects missing or mismatched
input hashes to prevent scoring a response to another prompt under the same ID.
For Choice, `choice` must be an offered label. For Score, `score` may be fractional in the
range `0..K-1`; if `probabilities` are supplied, its value must equal their
probability-weighted level mean within 0.06. Probability maps for Choice and
Score are optional for point accuracy but required for calibration and
selective metrics. They may omit exactly zero-probability options; all
unmentioned options are filled with zero. The listed probabilities must sum
to one within 0.02 and may not name unknown options.
For accepted Choice and Score maps, the scorer divides each option by the
original sum before computing Brier, NLL, ECE, and selective confidence.
The original map still controls answer consistency, and point accuracy uses
the returned answer. Noul uses its supplied probability.
Use a common adapter and prompt budget for all open models; preserve the exact
model revision and inference settings in the run manifest.

## Score

```bash
python3 -m benchmark.score \
  --gold runs/dev.gold.jsonl --predictions runs/model.dev.predictions.jsonl \
  --model-id org/model --model-revision exact-sha --backend vllm \
  --output runs/model.dev.report.json
```

The report includes all-item and valid-answer accuracy, family-macro
accuracy, type/family slices, Brier, negative log likelihood, 10-bin ECE,
Score mean absolute error, confidence threshold coverage, pair relation
consistency and joint accuracy, insufficient-evidence precision/recall,
evidence-question consistency, invalid-format reasons, latency percentiles,
usage, and cost **when supplied**. Missing or invalid answers count as wrong
in `accuracy_all`. Score exact level uses the highest-probability level when a
distribution exists; without one, it uses the nearest integer. Noul at exactly
0.5 is uncertain and does not receive point credit. The API's separate
`confidence` field is deliberately ignored: its semantics differ from a
correctness probability.

`typed-decision-report/2` adds `metric_policy` and
`option_probability_sum_abs_delta` under `overall`, each `by_type`, and each
`by_family` summary. The diagnostic counts accepted Choice/Score maps only
and records `n`, mean, p50/p95/p99, max, and counts above 0.001 and 0.01
before normalization. Old probability scores must be regenerated from the
same frozen prediction JSONL before comparing them with v2 scores.

Latency, token counts, and cost come from the prediction file and are not
independently verified. Compare latency under identical load, warmup,
concurrency, and network conditions; report hosted API and local inference
settings separately. Compare costs only with explicit pricing or measured
hardware cost assumptions. Preserve raw responses alongside normalized
predictions for audit, outside the training dataset.

## Verification

Lightweight generator and scorer correctness checks (no model inference):

```bash
python3 -m unittest discover -s benchmark/tests -v
python3 -m py_compile benchmark/generate.py benchmark/score.py
```

## Limits

The tasks are synthetic and use compact JSON rules. Success does not establish
real-world semantic judgment, multilingual robustness, calibration under human
disagreement, or general superiority to Jev or another model. The final
generator **code** is visible in this repository; withholding a high-entropy
seed prevents exact item memorization but cannot prevent fitting to the
generator grammar. For publishable model-family claims, pair this track with
independent public/human-labeled datasets, preregistered model revisions and
metrics, and a genuinely sealed external set. ECE is unstable for small slices;
the deterministic gold labels do not represent empirical event probabilities.
