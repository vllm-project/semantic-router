# Feedback Router Recipe

This recipe turns dissatisfaction recovery into a first-class routing policy. It keeps cheap clarification on the local Qwen lane, sends normal repair and retry work to Gemini, and escalates durable or verification-sensitive failures to GPT-5.4.

The maintained assets live here:

- `feedback-router.yaml`
- `feedback-router.dsl`
- `feedback.probes.yaml`

The policy mirrors the heuristics described in `.augment/feedback.md`, but expresses them with router-native primitives:

- `signals` capture explicit dissatisfaction, repeated-question reasks, code failures, verification pressure, and high-stakes domains.
- `projections` combine those signals into reusable recovery and verification bands.
- `decisions` map low-cost clarification, mid-cost repair, and premium recovery lanes to concrete models.
- `plugins` inject lane-specific recovery prompts and stable debug headers.

## Design Goals

- Keep ordinary clarification follow-ups on `qwen/qwen3.5-rocm`.
- Route single-turn dissatisfaction or one-turn reasks to `google/gemini-3.1-pro`.
- Route `lookback_turns: 2` persistent reasks to `openai/gpt5.4`, because at that point the cheaper recovery lane has likely already failed.
- Escalate verification-sensitive legal, health, and fact-check retries directly to `openai/gpt5.4`.
- Treat code failures as a separate lane from general answer repair.
- Make the `reask` signal unambiguous by returning only the maximum matched lookback tier. If `persistently_dissatisfied` matches, `likely_dissatisfied` is suppressed.

## Route Order

| Priority | Decision | Target model | Reasoning | Purpose |
|---|---|---|---|---|
| `240` | `feedback_verified_recovery` | `openai/gpt5.4` | high | Evidence-sensitive or high-stakes correction |
| `230` | `feedback_persistent_code_recovery` | `openai/gpt5.4` | high | Same coding failure repeated across two prior user turns |
| `220` | `feedback_code_recovery` | `google/gemini-3.1-pro` | medium | Single-turn code repair, tracebacks, execution failures |
| `210` | `feedback_persistent_recovery` | `openai/gpt5.4` | high | Same non-code question repeated across two prior user turns |
| `200` | `feedback_general_recovery` | `google/gemini-3.1-pro` | medium | General dissatisfaction or one-turn reask |
| `180` | `feedback_need_clarification` | `qwen/qwen3.5-rocm` | off | Cheap clarification and restatement |

The intentional cost split is:

- `lookback_turns: 1` implies likely dissatisfaction and stays on the Gemini repair tier.
- `lookback_turns: 2` implies persistent dissatisfaction and upgrades to GPT-5.4.

That makes the recipe visible in both behavior and cost: stronger evidence of failure buys a more expensive model.

## Cost Profile

Pricing is configured in `providers.models[].pricing`, which is what the router uses for cost accounting and Insights.

| Model | Prompt / 1M | Completion / 1M | Role |
|---|---|---|---|
| `qwen/qwen3.5-rocm` | `$0.00` | `$0.00` | Cheap clarification and fallback lane |
| `google/gemini-3.1-pro` | `$0.48` | `$1.92` | Mid-cost repair and retry lane |
| `openai/gpt5.4` | `$1.20` | `$4.80` | Premium verified and persistent recovery lane |

These are routing economics examples, not vendor billing quotes.

## Signal Strategy

### Explicit dissatisfaction

- `user_feedback("wrong_answer")`
- `user_feedback("need_clarification")`

### Implicit dissatisfaction

- `reask("likely_dissatisfied")` with `lookback_turns: 1`
- `reask("persistently_dissatisfied")` with `lookback_turns: 2`
- `keyword("frustration_feedback_markers")`
- `keyword("code_error_markers")`

### Verification pressure

- `fact_check("needs_fact_check")`
- `keyword("verification_markers")`
- `domain("health")`
- `domain("law")`

### Projections

- `feedback_recovery_pressure`
- `verification_pressure`
- `feedback_recovery_band`
- `verification_band`

The `feedback_recovery_pressure` score deliberately weights `persistently_dissatisfied` above `likely_dissatisfied`, because a two-turn streak is stronger evidence that the cheap or mid-cost lane already failed.

## Keyword Ordering

The keyword runtime is first-match, not multi-match. That means a generic rule placed too early can shadow a more specific one later in the list.

- `verification_markers` must stay ahead of `frustration_feedback_markers`, or factual verification retries like `Verify this with sources ...` will be misread as generic dissatisfaction.
- `code_error_markers` must stay ahead of `frustration_feedback_markers`, or traceback-heavy repair prompts with words like `wrong` can lose their code-specific signal.

The maintained recipe therefore orders keyword rules from most specific to most generic:

1. `verification_markers`
2. `code_error_markers`
3. `clarification_feedback_markers`
4. `frustration_feedback_markers`

## Reask Semantics

`reask` is history-aware and only looks at prior user turns from the same conversation. It compares the current user turn against the most recent prior user turns, newest first.

- `likely_dissatisfied` matches when the latest prior user turn is semantically similar enough.
- `persistently_dissatisfied` matches when the latest two prior user turns both clear the threshold.
- If both tiers would match, the router only returns the maximum matched lookback tier.

That last rule matters for recipe design: persistent retries should not also light up the cheaper one-turn lane and create ambiguous eval output.

## Maintained Probe Strategy

`feedback.probes.yaml` now mixes two probe shapes:

- `query`: the manifest's single-turn probe field for explicit dissatisfaction, code repair, verified recovery, and clarification.
- `messages`: multi-turn chat-completions payloads for persistent reask escalation.

The calibration loop converts those probe shapes into the live `/api/v1/eval` payload contract:

- single-turn probe `query` -> request body `{"text": "..."}`
- multi-turn probe `messages` -> request body `{"messages": [...]}`

This matters for two reasons:

- single-turn probes cannot exercise `reask("persistently_dissatisfied")`; the signal needs prior user turns
- hand-written eval calls should use `text`, not `query`, or the router will reject the request as empty input

## Maintained Test Queries

These are the maintained probe families in `feedback.probes.yaml`.

### `feedback_verified_recovery`

- `Your previous legal answer was wrong. Verify this with sources: can a landlord enter my apartment without notice in California?`
- `你刚才的医疗建议不对，请核实并给出处：发烧到多少度需要去急诊？`

### `feedback_code_recovery`

- `That fix is still wrong. I now get TypeError: list indices must be integers, not str in this Python function. Repair it.`
- `That answer was wrong. My React component still crashes during render and the previous fix did not solve it. Repair the code path.`

### `feedback_persistent_code_recovery`

```json
{
  "messages": [
    {"role": "user", "content": "Fix this same Python function. It crashes with TypeError: list indices must be integers, not str."},
    {"role": "assistant", "content": "Try indexing the list with the dictionary key."},
    {"role": "user", "content": "That is still wrong. Fix this same Python function. It crashes with TypeError: list indices must be integers, not str."},
    {"role": "assistant", "content": "Wrap the list in dict() and try again."},
    {"role": "user", "content": "Still wrong. Fix this same Python function. It crashes with TypeError: list indices must be integers, not str."}
  ]
}
```

### `feedback_general_recovery`

- `That answer was wrong. Explain inflation vs recession again in plain English with a cleaner structure.`
- `That answer was wrong. Explain compound interest vs simple interest again in plain English.`

### `feedback_persistent_recovery`

```json
{
  "messages": [
    {"role": "user", "content": "Explain inflation vs recession in plain English."},
    {"role": "assistant", "content": "Inflation means prices rise over time."},
    {"role": "user", "content": "That was still wrong. Explain inflation vs recession in plain English."},
    {"role": "assistant", "content": "Inflation is when goods become more expensive."},
    {"role": "user", "content": "Still wrong. Explain inflation vs recession in plain English."}
  ]
}
```

```json
{
  "messages": [
    {"role": "user", "content": "请问如何证明勾股定理？"},
    {"role": "assistant", "content": "可以用直角三角形面积法来解释。"},
    {"role": "user", "content": "请问如何证明勾股定理？"},
    {"role": "assistant", "content": "也可以从相似三角形出发。"},
    {"role": "user", "content": "请问如何证明勾股定理？"}
  ]
}
```

### `feedback_need_clarification`

- `Please explain that more clearly and give one simple example.`
- `讲清楚一点，再举一个简单例子。`

## Validation Commands

Repo-local config and DSL checks:

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT/src/semantic-router"
go test ./pkg/config ./pkg/dsl ./pkg/services ./pkg/apiserver ./pkg/classification
go run ./cmd/dsl decompile \
  -o ../../deploy/recipes/feedback/feedback-router.dsl \
  ../../deploy/recipes/feedback/feedback-router.yaml
go run ./cmd/dsl validate ../../deploy/recipes/feedback/feedback-router.dsl
```

Probe-driven local validation:

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"
python3 tools/agent/scripts/router_calibration_loop.py validate \
  --yaml deploy/recipes/feedback/feedback-router.yaml
```

Live router probe evaluation:

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
ROUTER_URL="http://<router-host>:8080"
cd "$REPO_ROOT"
python3 tools/agent/scripts/router_calibration_loop.py eval \
  --router-url "$ROUTER_URL" \
  --probes deploy/recipes/feedback/feedback.probes.yaml
```

Direct single-turn eval example:

```bash
curl -s http://<router-host>:8080/api/v1/eval \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "Your previous answer was wrong. Verify this with sources: who wrote The Iliad and roughly when was it composed?"
  }'
```

Direct multi-turn eval example:

```bash
curl -s http://<router-host>:8080/api/v1/eval \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [
      {"role": "user", "content": "Explain inflation vs recession in plain English."},
      {"role": "assistant", "content": "Inflation means prices rise over time."},
      {"role": "user", "content": "That was still wrong. Explain inflation vs recession in plain English."},
      {"role": "assistant", "content": "Inflation is when goods become more expensive."},
      {"role": "user", "content": "Still wrong. Explain inflation vs recession in plain English."}
    ]
  }'
```

If the endpoint is wired correctly, the eval response should expose only `reask:["persistently_dissatisfied"]` for that payload, not both reask tiers.

Expected progression for a same-question retry sequence is:

| Conversation state | Expected route | Expected model |
|---|---|---|
| 1st ask, no dissatisfaction evidence yet | cheap lane or no escalation | `qwen/qwen3.5-rocm` |
| 2nd ask with one prior matching user turn | `feedback_general_recovery` | `google/gemini-3.1-pro` |
| 3rd ask with two prior matching user turns | `feedback_persistent_recovery` | `openai/gpt5.4` |

The same progression applies to code repair, except the second and third steps become `feedback_code_recovery` then `feedback_persistent_code_recovery`.
