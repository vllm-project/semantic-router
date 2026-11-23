# recipe-lab

Closed-loop experiments for semantic-router routing recipes (RSI workbench).

This is the first slice of the [recipe-lab umbrella](https://github.com/vllm-project/semantic-router/issues/5404):
the misroute attribution report ([#5489](https://github.com/vllm-project/semantic-router/issues/5489)).
It turns "routing accuracy is X%" into a prioritized, executable fix queue:
every wrong outcome on a labeled dataset is classified into an actionable
error class, and the class distribution decides where the later recipe-lab
phases spend effort.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
```

## Commands

| command | purpose |
|---|---|
| `plan` | load and validate an experiment spec, print it normalized |
| `collect-replays` | pull `router_replay` records from the router management API and join them against the labeled case list (`--unlabeled` emits raw diagnostic rows) |
| `attribute` | run the E1 / decision_mismatch / E2 / E3 misroute attribution over labeled records |

Worked examples with a runnable spec and records live in
[`examples/`](examples/):

```bash
# validate and preview the experiment contract (runs nothing)
recipe-lab plan --spec examples/spec.example.yaml

# pull replay records from a running router and join them to labels
recipe-lab collect-replays --url http://127.0.0.1:8080 \
  --cases examples/cases.example.jsonl --output attribution-input.jsonl

# classify every wrong outcome and write the report
recipe-lab attribute --spec examples/spec.example.yaml \
  --records examples/records.example.jsonl --output-dir out/
```

`attribute` writes `attribution.jsonl` (one row per case; the file path plus
its content hash is the downstream reference - later recipe-lab phases cite
it as `attribution_ref`) and `report.md` (class distribution, E1 breakdowns
per signal family and per failing signal, decision confusion pairs, policy
deviation and per-domain E3 candidates).

## Record format (attribution input)

One JSON object per line (the `router_replay` adapter emits this shape):

```json
{"case_id": "c1", "truth": "math", "matched_decision": "code", "routed": "local/model-b",
 "family": "d3", "answer_ok": false}
```

- `answer_ok`: `true` / `false` / `null` (`null` = no evidence; by the
  evidence-first rule unknown correctness is never counted as an error).
- `matched_decision` is the decision the router actually matched (renamed
  from the earlier `pred`; reading old records with a top-level `pred` field
  still works).
- `truth` is either the hand-labeled expected decision for that prompt (a
  string), or an object `{"signals": {...}}` carrying per-signal
  ground-truth facts: each fact is a label string (`"domain": "code"`), a
  number (`"difficulty": 3`), or a numeric predicate (`"difficulty": {gte: 3}`).
  Facts drive both the oracle policy's rule evaluation and the E1 signal
  judgment.
- Signal evidence (`signal_values` / `signal_confidences` /
  `decision_tier`) is carried verbatim from the replay record. Keys are
  `<type>:<name>` (e.g. `decision:difficulty`); choice-type decision signals
  store one probability per choice under `decision:<name>:<choice>`, while
  score-type signals keep the numeric score under `decision:<name>`. When
  reading a signal, values fall back per key from `signal_values` to
  `signal_confidences` (matching the router's own lookup).

## Error classes

| class | meaning |
|---|---|
| `correct` | routed as intended and answered correctly |
| `E1_signal` | a truth signal contradicts the record's observed signal evidence (needs signal-level truth: truth = `{"signals": ...}`) |
| `decision_mismatch` | matched decision != expected decision, judged without usable signal-level truth (fallback; with fully verified signal truth a decision miss is `E2_policy`) |
| `E2_policy` | decision or routed model deviates from the oracle-policy expectation, answer wrong |
| `E3_capability` | routing exactly as intended, answer still wrong |
| `suboptimal_routing` | policy deviation but still a correct answer (counted, never an error) |
| `no_evidence` | answer correctness unknown; never an error |
| `policy_gap` | truth (label or signal facts) not covered by the declared oracle policy |

## Oracle policy

The oracle policy is declared in the experiment spec under
`attribution.oracle_policy`; attribution never recomputes it from router
logic. Two forms are supported:

```yaml
attribution:
  oracle_policy:
    # form 1 (legacy): decision label -> intended model
    math: local/model-math
    code: local/model-code

    # form 2: signal-fact rules, evaluated top-down; first match wins
    rules:
      - when: {domain: code, difficulty: {gte: 3}}
        expect: {decision: domain_code_complex, models: [m/pro, m/max]}
      - when: {domain: code}
        expect: {decision: domain_code}
    default: {decision: simple_general}
```

String truths resolve through the label mapping; `{"signals": ...}` truths
evaluate the rules against the truth facts (unmatched and default-less cases
become `policy_gap`). A truth whose signals contradict the record's observed
signal evidence is `E1_signal` (the failing signal and its observed value are
recorded on the row).

## Signal family

`family` on attribution rows is derived from the case's own signal evidence
key type: `classifier:` -> `vela_classify`, `decision:` -> `vela_decisions`,
`projection:` -> `d3`; other key type segments surface verbatim. Records
without signal evidence keep whatever `family` tag the input record carries
(e.g. from `collect-replays --family`).

## Scope of this slice

The replay-tier loop, answer bank, proposers and the ledger arrive with
[sub-issue 2](https://github.com/vllm-project/semantic-router/issues/5490);
the live tier with
[sub-issue 3](https://github.com/vllm-project/semantic-router/issues/5491).
This slice owns the contract fields the attribution needs (objective, frozen
workload identity, oracle policy); the remaining experiment/budget/candidate
contract surface lands with sub-issue 2.

## Tests

```bash
pytest
```
