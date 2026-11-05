# Signal

## Overview

Signals turn request facts into names that routing decisions can reuse. For
example, a signal can identify a long prompt, a tool-heavy conversation, a
language, or a likely prompt-injection attempt. A decision then chooses what to
do when that signal matches.

Use [Projections](../projection/overview) when you need to combine several
signals into a score, partition, or routing band before a decision runs.

## Key Advantages

- Reuses one detector across multiple decisions.
- Keeps detection logic separate from route outcomes.
- Lets one route combine lexical, policy, semantic, and safety inputs.
- Makes config reviews easier because signal names become stable policy building blocks.

## What Problem Does It Solve?

Without a signal layer, every decision has to inline its own detection logic. That creates duplication, makes route policies harder to audit, and mixes "what did we detect?" with "what should we do?".

Signals solve that by turning request understanding into a named catalog that the rest of the routing graph can compose.

## When to Use

Use signals when:

- more than one route needs the same detector
- you want to mix different detection methods in one decision tree
- you need a clean boundary between detection, decision logic, algorithms, and plugins
- you want to tune detection without rewriting route outcomes

## Configuration

In canonical v0.3 YAML, signals live under `routing.signals`:

```yaml
routing:
  signals:
    keywords:
      - name: urgent_keywords
        operator: OR
        keywords: ["urgent", "asap"]
    embeddings:
      - name: technical_support
        threshold: 0.75
        candidates: ["installation guide", "troubleshooting steps"]
      - name: account_management
        threshold: 0.72
        candidates: ["billing information", "subscription management"]
  projections:
    partitions:
      - name: support_intents
        semantics: exclusive
        temperature: 0.3
        members: [technical_support, account_management]
        default: technical_support
    scores:
      - name: request_difficulty
        method: weighted_sum
        inputs:
          - type: embedding
            name: technical_support
            weight: 0.18
            value_source: confidence
    mappings:
      - name: request_band
        source: request_difficulty
        method: threshold_bands
        outputs:
          - name: support_escalated
            gte: 0.25
```

Choose a signal by the kind of fact you need to detect.

### Heuristic Signals

These signals use explicit rules, request shape, identity, or lightweight
detectors. They do not require a general-purpose classifier model.

| Signal | Use it to |
| ------ | --------- |
| [Action](./heuristic/action) | route by the operation a request asks for, such as explain, fix, or refactor |
| [Authz](./heuristic/authz) | route from trusted identity, role, or tenant policy |
| [Conversation](./heuristic/conversation) | detect multi-turn, tool-heavy, or agentic request structure |
| [Context](./heuristic/context) | route by effective context-window needs |
| [Event](./heuristic/event) | detect structured events by type, severity, action code, or urgency |
| [Input Modality](./heuristic/input-modality) | detect which input modalities (text, image, audio, video) are present |
| [Keyword](./heuristic/keyword) | match explicit words, phrases, BM25 terms, or n-grams |
| [Language](./heuristic/language) | route by detected request language |
| [Metadata](./heuristic/metadata) | use bounded, caller-provided application hints |
| [Structure](./heuristic/structure) | detect counts, density, and ordered markers in a prompt |

### Learned Signals

These signals use embeddings, classifiers, or configured detector models. Check
each page's data-handling notes before choosing a remote provider.

| Signal | Use it to |
| ------ | --------- |
| [Classifier](./learned/classifier) | expose labels from a custom local classifier or external LLM |
| [Complexity](./learned/complexity) | estimate easy, medium, or hard reasoning traffic |
| [System One](./learned/systemone) | ask typed questions with the selected decision model |
| [Domain](./learned/domain) | classify the request topic |
| [Embedding](./learned/embedding) | match semantic intent from representative examples |
| [Modality](./learned/modality) | classify text, image-generation, or mixed output intent |
| [Fact Check](./learned/fact-check) | detect prompts that may need evidence verification |
| [Hallucination](./learned/hallucination) | check the model's answer against the grounding context it was given |
| [Jailbreak](./learned/jailbreak) | detect prompt-injection or jailbreak attempts |
| [Safety](./learned/safety) | detect unsafe content and optional risk categories |
| [PII](./learned/pii) | detect sensitive personal data |
| [Preference](./learned/preference) | infer response-style preferences |
| [Reask](./learned/reask) | detect a repeated question in recent conversation history |
| [Knowledge Base](./learned/kb) | match labels or groups from a reusable exemplar set |
| [User Feedback](./learned/user-feedback) | detect correction, dissatisfaction, or escalation feedback |

Keep these rules in mind:

- keep signals named and reusable
- keep signals detection-only; routing outcomes belong in `decision/`
- keep partitions and derived routing bands in `routing.projections`, not back inside `routing.signals`
- keep model choice separate; that belongs in `algorithm/`
- keep route-side behavior separate; that belongs in `plugin/`

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
  keeps them from drowning out the question and spares learned signals from
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
- the question the hallucination check reads;
- Replay's prompt;
- the routing preview, classify and eval APIs.

A skipped message is not counted as a user turn, so a runtime-context turn
after a tool result leaves the request a tool-loop turn, and user-feedback and
re-ask signals do not fire on it. The request sent to the model is unchanged,
and context-length estimates still count all of it.

Jailbreak, PII and content-safety signals keep scanning skipped messages and
stripped blocks along with the routing text, and tool-result checks are
unaffected, so a client cannot hide content from them inside a fake memory block
or a fake runtime-context turn. Every other signal, keyword rules included,
reads only the routing text.

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

## Next Steps

- Read [Projections](../projection/overview) when you need `PROJECTION partition`, weighted score aggregation, or named routing bands.
- Start from [`config/config.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/config.yaml) for the exhaustive public contract.
- See the `balance` recipe for a complete routing strategy:
  - [`config/recipes/balance/config.yaml`](https://github.com/vllm-project/semantic-router/blob/main/config/recipes/balance/config.yaml)
  - [`config/recipes/balance/recipe.dsl`](https://github.com/vllm-project/semantic-router/blob/main/config/recipes/balance/recipe.dsl)

Signals can inspect request text, conversation history, images, caller
metadata, or trusted identity depending on the family. Learned signals may send
that data to a configured remote classifier or embedding provider. Review the
dependency and data notes on each family page before using it as a policy gate.
