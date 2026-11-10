# Signal

## Overview

Signals turn request facts into names that routing decisions can reuse. A
signal can identify a long prompt, a tool-heavy conversation, a language, a
subject area or a likely prompt-injection attempt; a decision then chooses
what to do when it matches.

Signals come in three layers:

- **The `systemone` primitive** asks the decision model a question you write
  in plain language.
- **Application signals** are eleven ready-made questions, each compiled
  into one `systemone` question of the Router's own
  ([presets](../../system-one/get-started/concepts#presets)).
- **Heuristic signals** are deterministic code: keywords, language, request
  shape, identity and metadata.

Use [Projections](../projection/overview) when you need to combine several
signals into a score, partition or routing band before a decision runs.

## Key Advantages

- Reuses one detector across many decisions.
- Keeps detection separate from route outcomes.
- Lets one route combine lexical, structural, policy and model-judged inputs.
- Makes configuration reviews easier: signal names are stable policy
  building blocks.

## What Problem Does It Solve?

Without a signal layer, every decision inlines its own detection. That
duplicates logic, makes route policies harder to audit, and mixes "what did we
detect?" with "what should we do?". Signals turn request understanding into a
named catalog that the rest of the routing graph composes.

## When to Use

Use signals when:

- more than one route needs the same detector;
- you want to mix detection methods in one decision tree;
- you need a clean boundary between detection, decisions, algorithms and
  plugins;
- you want to tune detection without rewriting route outcomes.

## Configuration

Signals live under `recipes[].signals`, one list per signal type:

```yaml
recipes:
  - name: default
    signals:
      keywords:
        - name: urgent_keywords
          operator: OR
          keywords: ["urgent", "asap"]
      domains:
        - name: math
          labels: [math]
      jailbreak:
        - name: prompt_attack
      systemone:
        - name: support_intent
          question:
            type: choice
            instructions: What does the user need?
            choices:
              - key: technical_support
                description: Installation, errors or troubleshooting
              - key: account_management
                description: Billing information or subscription changes
              - key: other
                description: Anything else
          labels: [technical_support]
    decisions:
      - name: urgent-support
        priority: 200
        rules:
          operator: AND
          conditions:
            - {type: systemone, name: support_intent}
            - {type: keyword, name: urgent_keywords}
        modelRefs:
          - model: support-model
```

A decision's condition names the rule by its condition type and name:
`{type: domain, name: math}`. Choose a signal by the kind of fact you need.

### The System One Primitive

| Signal | Condition type | Use it to |
| ------ | -------------- | --------- |
| [System One](./systemone) | `systemone` | ask the decision model your own `choice`, `noul`, `score`, `set` or `span` question |

### Application Signals

Each application signal asks one preset of the decision model, unless a
binding or the rule names another deployment. Rules take the preset's
canonical question and override only what their type allows.

| Signal | Condition type | Use it to |
| ------ | -------------- | --------- |
| [Domain](./application/domain) | `domain` | route on the request's subject area, one of 14 |
| [Fact Check](./application/fact-check) | `fact_check` | detect requests whose answer relies on facts that should be checked |
| [User Feedback](./application/user-feedback) | `user_feedback` | read what a turn says about the previous answer |
| [Jailbreak](./application/jailbreak) | `jailbreak` | detect prompt-injection and jailbreak attempts, in the request or the answer |
| [Safety](./application/safety) | `safety` | detect harmful content, optionally in named harm categories |
| [PII](./application/pii) | `pii` | locate personal data in the request |
| [Hallucination](./application/hallucination) | `hallucination` | find claims in the answer that its context does not support |
| [Refusal](./application/refusal) | `refusal` | detect answers that decline the request |
| [Turn Relation](./application/turn-relation) | `turn_relation` | tell a follow-up from a re-ask, a topic shift or a return |
| [Complexity](./application/complexity) | `complexity` | rate a request easy, medium or hard |
| [Modality](./application/modality) | `modality` | read which output the request asks for: text, an image, speech, video, ... |

### Heuristic Signals

These signals use explicit rules, request shape, identity or bounded
metadata. They ask no model.

| Signal | Condition type | Use it to |
| ------ | -------------- | --------- |
| [Authz](./heuristic/authz) | `authz` | route from trusted identity, role or tenant policy |
| [Context](./heuristic/context) | `context` | route by the request's estimated token count |
| [Conversation](./heuristic/conversation) | `conversation` | detect multi-turn, tool-heavy or agentic request structure |
| [Input Modality](./heuristic/input-modality) | `input_modality` | detect which input media (text, image, audio, video) are present |
| [Keyword](./heuristic/keyword) | `keyword` | match words, phrases, BM25 terms or n-grams, or count how often they occur |
| [Language](./heuristic/language) | `language` | route by detected request language |
| [Metadata](./heuristic/metadata) | `metadata` | use bounded, caller-provided application hints |

Keep these rules in mind:

- keep signals named and reusable;
- keep signals detection-only; routing outcomes belong in decisions;
- keep partitions and derived routing bands in `recipes[].projections`, not
  inside `recipes[].signals`;
- keep model choice in the decision's algorithm, and route-side behavior in
  its plugins.

## Choose the User Text Signals Read

Signals read the last user message that has text. Some clients put text there
that is not the user's question, and `global.router.routing_input` tells the
Router to leave it out:

```yaml
global:
  router:
    routing_input:
      # A user message whose text matches one of these RE2 patterns is never
      # routing text.
      skip_user_messages_matching:
        - '(?s)^\s*OpenClaw runtime context:.*End OpenClaw runtime context\.\s*$'
      # Spans between these literal markers are cut from the routing text.
      strip_blocks:
        - start: '<relevant-memories>'
          end: '</relevant-memories>'
        - start: '<recalled-memories>'
          end: '</recalled-memories>'
```

- OpenClaw 2026.10.4 and later end every turn, including tool-result turns,
  with a separate user message that holds only runtime metadata. Without the
  skip pattern, every request routes on that metadata.
- OpenClaw memory plugins prepend recalled memories to the user's prompt on
  every turn: memory-lancedb-pro wraps them in `<relevant-memories>`, mem0's
  plugin in `<relevant-memories>` or `<recalled-memories>`. Stripping the blocks
  keeps them from drowning out the question and spares the decision model from
  reading them.

The routing text is the last user message that is not skipped and still has
text after stripping. It takes the place of the last user message for:

- every routing signal, and the conversation facts they read;
- the semantic-cache lookup query;
- tool selection;
- the RAG query;
- memory: the retrieval query, the history its query rewriting reads, and the
  user message it stores;
- the query that model-selection algorithms read;
- the request the hallucination check reads;
- Replay's prompt;
- the routing preview, classify and eval APIs.

A skipped message is not counted as a user turn, so a runtime-context turn
after a tool result leaves the request a tool-loop turn, and user-feedback and
turn-relation signals do not fire on it. The request sent to the model is
unchanged, and context-length estimates still count all of it.

Jailbreak, PII and safety signals keep reading skipped messages and stripped
blocks along with the routing text, and tool-result checks are unaffected, so
a client cannot hide content from them inside a fake memory block or a fake
runtime-context turn. Every other signal, keyword rules included, reads only
the routing text.

- Patterns use Go RE2 syntax and match anywhere in the text unless anchored with
  `^` and `$`. A pattern that matches empty text is rejected when the config
  loads.
- Only a message whose content is all text can be skipped. A message that also
  carries images, audio or tool results always routes.
- Markers are literal and case-sensitive. Blocks are cut left to right and do
  not nest: a block ends at the first end marker after its start marker. A start
  marker with no end marker after it stays in the routing text.
- When every user message is skipped, routing proceeds as for a request without
  user text.
- The semantic cache compares the routing text, but skipped messages and
  stripped blocks must still match before a cached answer is reused.
- `llm_routing_input_removed_total` counts skipped messages and stripped blocks,
  and Replay's route diagnostics record both counts per request. The removed
  text is never logged.

## See What Matched

`POST /api/v1/routing/preview` evaluates every signal of the active recipe for
a request and reports the matched rules and the decision, without calling a
model. On a routed response, the `signals` member of `x-vsr-trace` lists the
matched rules as `type:name`, such as `domain:math` or `pii:contact`, for a
caller `global.router.headers.debug` allows; a `response_guard` adds its codes,
such as `hallucination`, to `x-vsr-warnings`. See
[VSR routing headers](../../api/headers).

## Next Steps

- Read [System One](../../system-one/overview) for how the decision model
  answers every model-judged signal, and which deployment answers each one.
- Read [Projections](../projection/overview) when you need a partition,
  weighted score aggregation or named routing bands.
- Start from [`config/config.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/config.yaml)
  for the exhaustive public contract.

Signals can read request text, conversation history, images, caller metadata
or trusted identity depending on the type. Model-judged signals send what they
read to the deployment that answers them; [what leaves the host](../../system-one/deploy/deployments#what-leaves-the-host)
lists the parts each one sends.
