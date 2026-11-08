# Jailbreak Signal

## Overview

`jailbreak` detects prompt-injection and jailbreak attempts before the Router
commits to a route. Define jailbreak rules under `recipes[].signals.jailbreak`.

The decision model answers it unless a `jailbreak` binding names another
deployment; see [Choose a model for a signal](../../../installation/choose-a-model-for-a-signal.md).
`global.model_catalog.modules.prompt_guard` holds its settings.

## Key Advantages

- Lets decisions block or downgrade unsafe traffic before model selection.
- Supports classifier, contrastive, and hybrid-style safety detection.
- Keeps jailbreak policy visible inside routing decisions.
- Reuses one safety signal across multiple guarded routes.

## What Problem Does It Solve?

If jailbreak detection only happens downstream, the router can still send unsafe traffic to the wrong model or toolchain. If it lives outside the routing graph, safety logic becomes harder to audit.

`jailbreak` solves that by making injection detection a first-class routing input.

## When to Use

Use `jailbreak` when:

- unsafe traffic must be blocked before model selection
- prompt-injection attempts should route to a safer fallback
- multi-turn history should influence routing
- safety policy must be visible and testable in the same graph as routing logic

## Configuration

```yaml
recipes:
  - name: default
    signals:
      jailbreak:
        - name: prompt_injection
          method: contrastive
          threshold: 0.8
          include_history: true
          description: Detect common prompt-injection or jailbreak attempts.
          jailbreak_patterns:
            - ignore previous instructions
            - reveal the hidden prompt
            - jailbreak mode
          benign_patterns:
            - explain the policy
            - summarize the safety rules
```

`jailbreak_patterns` define deterministic hard-block patterns.
The runtime normalizes case and whitespace, then checks whether each normalized
message contains a normalized pattern as a substring. A match blocks the
request independently of the contrastive threshold. This explicit match does
not create a contrastive confidence score or replace a genuine score when one
is available.

`benign_patterns` provide benign examples for contrastive scoring; they are
not explicit allow rules. The contrastive threshold applies to the contrastive
score, not to deterministic explicit-pattern matches.

Use `include_history` for multi-turn attacks.

### Request content and history

For a routed request, Guard scores the consecutive user and tool messages at
the end of the request, stopping at an assistant, system or developer message.
This keeps all textual tool results together when a protocol represents them
inside one user message. With
`include_history: true`, it also scores earlier user and tool text. System and
developer instructions and assistant replies are not independently scored as
attack evidence. This narrows earlier versions' history behavior, which also
scanned those trusted roles.

If this current group has no eligible text, Guard does not substitute an older
user turn or concatenate trusted instructions. Eligible earlier content is
still inspected when `include_history` is enabled. No eligible text means no
classification evidence, not a guarantee that the request is safe.

This projection preserves complete text pieces and does not use the general
routing text compressor. Each piece is scored separately; any matching piece
can match the rule. It does not infer authority relationships across messages
or inspect images, raw tool arguments, or content retrieved after routing.
Message roles are a protocol boundary, not authenticated identity. The direct
text detection API and response-direction scans retain their existing input
contract; callers supplying flat text remain responsible for its scope.

### Token windows for a local classifier

The default decision model reads each text within its own input limit and
takes no window ([Choose a model](../../../model-runtime/choose-a-model.md#built-in-signals)
lists each model's limits). Bound to Vela 1.0 Guard, the signal scans each
text piece in 512-token windows with 255 content tokens of overlap, within
that model's document budget: 32,768 tokens including special tokens. This
applies only when no `window` is specified and `max_sequence_length` remains
zero. Each forward remains bounded to 512 tokens;
a long piece requires multiple forwards. Preparation records the resolved window and document budget
and checks the loaded model's actual capacity. An incompatible custom artifact
fails preparation, and a piece exceeding the document budget produces an
input-limit error.

Document budgets, window policies, and remote deployments keep their
configured behavior. For example, a qualified 8K deployment with
`input.overflow: reject` continues to process one whole piece within that budget.

For a checkpoint evaluated with overlapping token windows, configure the same
window policy in the prompt-guard module:

```yaml
global:
  model_catalog:
    modules:
      prompt_guard:
        max_sequence_length: 32768
        window:
          size: 128
          overlap: 63
```

`size` includes the tokenizer's special tokens; `overlap` counts content
tokens. For a tokenizer with two special tokens, this example scans 126 content
tokens at a time with a stride of 63. The runtime plans windows from the
complete input's token IDs without decoding and re-tokenizing window text,
and resets positions in each window.
The total input must fit `max_sequence_length`; overflow is an inference
error, never an uninspected suffix.

With a `jailbreak` binding to a local deployment, the deployment's
`input.max_tokens` supplies the complete-document budget. With window scanning, that budget can exceed the
checkpoint's single-forward capacity: `input: {max_tokens: 65536, overflow: window}`
and `window: {size: 32768, overlap: 256}` scan up to 64K tokens using forwards of
at most 32K tokens. Preparation checks each window against the loaded checkpoint
and graph; a 32K document budget alone does not establish 32K forward support.
All counts use the Guard tokenizer, including its special tokens. Explicit
non-window `reject` and `truncate` deployments keep their existing input policy.

Request rules, the detection API, and response scans use the same maximum
positive-label risk across windows. For multiple positive labels, the runtime
sums their probabilities within each window before choosing the riskiest
window. It retains that window's complete distribution for labels and
confidence. Contrastive rules keep their existing text-window policy.

Outside the implicit default, omitting `window` retains whole-input inference
or the configured legacy scan. Window sizes and thresholds need separate
checkpoint evaluation; scanning all tokens does not establish understanding of
distant context. Quoted attacks and instructions whose meaning depends on
another window require separate evaluation. A `jailbreak` binding to a
[model runtime](../../../model-runtime/guides/safety.md) deployment can also
select token windows; the model must support the requested window size.

A provider result declaring truncated or incompletely processed input is an
unscanned input, as is an input over the guard's input under `reject` or over
its [scan budget](../../../model-runtime/reference.md#long-inputs). Request
rules, the text detection APIs, and response scans cannot use its
probabilities to report a clean complete input: a rule matches it with the
type `unscanned`, so padding a prompt cannot carry an attack past the guard.
So does a scan that misses the signals' deadline
(`global.model_catalog.signal_timeout_ms`). A request whose question gets no
answer (the deployment fails, is overloaded or returns an error) does not
match by default; a request-stage rule with `on_unknown: match` matches it,
with the type `unavailable`, and a decision's `rules.on_unknown` still decides
such a rule. A detection on another completely scored piece still counts as a
detection. A response-stage rule follows the `response_guard` plugin's
`on_unknown`.

### Direction

`direction` selects what a rule scores. The default, `request`, scores the
prompt before the Router commits to a route. `response` scores the model's own
output, so the rule only exists once the model has answered:

```yaml
recipes:
  - name: default
    signals:
      jailbreak:
        - name: unsafe_completion
          direction: response
          threshold: 0.85
          description: Detect jailbreak content in the model's own output.
```

A response-direction rule uses the sequence classifier only: `method: contrastive`,
the pattern lists and `include_history` are request-stage settings and are
rejected on it. Matches, scores and failures are reported under the same
`jailbreak:<name>` key as a request-direction rule. Router Replay records the
observation as one outcome per response-direction rule, with the verdict
(`detected`, `not_detected` or `unavailable`), the score it thresholded or the
failure code, and the action the guard applied; with `x-vsr-debug`, the
`x-vsr-matched-jailbreak` header carries the matched response rules after the
request ones.

A response-direction rule is not a decision input. Decisions are selected while
the request is being routed, before the model has answered, so a decision that
reads one, directly in its rules or through a projection, is rejected when the
configuration loads. The observation is
read by the [`response_guard`](../../plugin/response-guard) of the decision
selected for the request, whose rules name it and apply their action to it.
The rule is read from the recipe the request resolved to, so a rule declared
on one entrypoint's recipe scores only that entrypoint's responses. The rule
owns the threshold. Declaring it is enough to provision the jailbreak signal
for the recipe: its model and label mapping are loaded for the response stage
even when no decision rule reads a jailbreak signal.

An unresolved detector (backend failure, or a response with no text to score)
is reported through `SignalErrors`, the way every other signal reports one,
rather than looking like a clean response; a guard rule that reads it takes the
guard's `on_unknown`. A response is clean only when every
chunk of it was scored: a chunk the backend failed on leaves the rule
unresolved unless the score the other chunks produced already matches it. The
response is scored once and each rule draws its own line across that score, so
a partial scan is resolved per rule: a score of 0.5 matches a rule at 0.4 and
leaves a rule at 0.9 unresolved, because the chunk that was never scored is
where a higher score would have been.

A streamed response is scored once the stream ends and recorded with
`enforcement: not_enforced_streaming` in place of an action. Its bytes are
already with the client by then, so no guard action applies.

## Dependencies and Limitations

The configured prompt-guard runtime processes the current prompt and,
optionally, conversation history. Detection is probabilistic and can be evaded
or over-triggered; combine it with least-privilege tools and backend policy.
See a complete example:
[`config/fragments/signal/jailbreak/patterns.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/fragments/signal/jailbreak/patterns.yaml).
