---
title: System One reference
sidebar_label: Reference
description: The preset catalog, the support matrix, calibrated thresholds, the configuration fields that name System One, and the load-time codes.
---

# System One reference

## Preset catalog

Every application signal, judge and check asks one of these questions. The
wording and options are the canonical, zero-shot definition; a family with a
trained binding is asked the trained form.

| Preset | Type | Options | Stage and input | Reads | Asked by |
| --- | --- | --- | --- | --- | --- |
| `domain` | choice | 14 subject areas | request | The current turn, truncated | `domains` |
| `fact_check` | choice | `FACT_CHECK_NEEDED`, `NO_FACT_CHECK_NEEDED` | request | The current turn, truncated | `fact_check` |
| `feedback` | choice | `SAT`, `NEED_CLARIFICATION`, `WRONG_ANSWER`, `WANT_DIFFERENT`, `NO_FEEDBACK` | request | The current turn, truncated, and the answer it replies to | `user_feedbacks` |
| `jailbreak` | choice | `benign`, `jailbreak` | request or response | The whole input; the answer at the response stage | `jailbreak` |
| `harm` | choice | `safe`, `unsafe` | request or response | The whole input; the answer at the response stage | `safety` without `labels` |
| `harm_categories` | set | The 34 [harm categories](#harm-categories) | request or response | The whole input | `safety` with `labels` |
| `pii` | span | 11 [personal data labels](#personal-data-labels) | request | The whole input, in windows | `pii`, `masking` |
| `hallucination` | span | `unsupported` | response | The answer and its grounding context | `hallucination` |
| `refusal` | choice | `compliance`, `refusal` | response | The request and the answer | `refusal`, `escalate` `accept.refusal` |
| `turn_relation` | choice | `continue`, `reask`, `shift`, `return` | request | The current turn and the last 4 turns | `turn_relation` |
| `complexity` | choice | `easy`, `medium`, `hard` | request | The current turn, truncated | `complexity` |
| `modality` | choice | `text`, `image`, `text_image`, `image_edit`, `speech`, `audio`, `video`, `video_edit` | request | The current turn, truncated | `modality` |
| `relevance` | noul | | request | The request and a candidate | `context` sources and compression |
| `same_request` | noul | | request | The request and a cached request | `response_cache` |
| `tool_rank` | choice | The candidate tools | request | The request | `tools` |
| `tool_need` | noul | | request | The request | `tools` |
| `answer_check` | noul | | response | The request and the answer, whole | `escalate` `accept.systemone` |

The domains are `biology`, `business`, `chemistry`, `computer science`,
`economics`, `engineering`, `health`, `history`, `law`, `math`, `other`,
`philosophy`, `physics` and `psychology`.

### Harm categories

A `safety` rule's `labels` name some of these 34 categories, the labels of
the trained vela3 hazard question:

- **AEGIS 2.0 (21):** Controlled/Regulated Substances,
  Copyright/Trademark/Plagiarism, Criminal Planning/Confessions,
  Fraud/Deception, Guns and Illegal Weapons, Harassment, Hate/Identity Hate,
  High Risk Gov Decision Making, Illegal Activity, Immoral/Unethical, Needs
  Caution, Other, PII/Privacy, Political/Misinformation/Conspiracy,
  Profanity, Sexual, Sexual (minor), Suicide and Self Harm, Threat,
  Unauthorized Advice, Violence.
- **MLCommons hazard codes (13):** S1 (violent crimes), S2 (non-violent
  crimes), S3 (sex-related crimes), S4 (child sexual exploitation), S5
  (defamation), S6 (specialised advice), S7 (privacy), S8 (intellectual
  property), S9 (indiscriminate weapons), S10 (hate), S11 (suicide and
  self-harm), S12 (sexual content) and S14 (code interpreter abuse).

vela3-fast asks its trained 34-label set once. At the request stage, the
Vela 3.0 d3 sizes are asked one `hazard:<label>` yes/no question per category,
in the form vela3-lite's hazard adapter was trained on; vela3-mini, -flash and
-turbo ship no hazard adapter and answer those questions zero-shot. Every other
model answers the same labels zero-shot, in the same call.

### Personal data labels

`PERSON`, `EMAIL_ADDRESS`, `PHONE_NUMBER`, `US_SSN`, `CREDIT_CARD`,
`LOCATION`, `IP_ADDRESS`, `DATE_TIME`, `US_DRIVER_LICENSE`, `US_PASSPORT`
and `IBAN_CODE`. A `pii` rule's `ignore_labels` and a masking plugin's
`entity_types` name some of them.

## Support matrix

**TC** trained and calibrated, **T** trained, **Z** zero-shot, **—** cannot
run.

| Preset | vela3-fast | vela3-lite | vela3-mini, -flash, -turbo | d3 (local or hosted) | Decision 2.0 | Vela 2.0 |
| --- | --- | --- | --- | --- | --- | --- |
| domain, fact_check, feedback, jailbreak | TC | T | T | Z | Z | TC |
| harm | TC | T | T | Z | Z | TC |
| harm categories | T | T at the request stage; Z at the response stage | Z | Z | Z | Z |
| pii | TC | T | T | — | — | TC |
| hallucination | T | T | T | — | — | T |
| refusal | T | T | T | Z | Z | Z |
| modality | TC for `text`, `image`, `text_image`; Z for the others | T for the three; Z | T for the three; Z | Z | Z | Z |
| turn_relation, complexity, relevance, same_request, tool_rank, tool_need, answer_check | Z | Z | Z | Z | Z | Z |

- The Vela 3.0 d3 sizes turn trained adapters on by question ID: `domain`,
  `factcheck`, `feedback`, `attack`, `p_harm` and `r_harm`, `r_refusal`,
  `modality`, and, on vela3-lite only, the `hazard:` prefix of the harm
  categories; vela3-fast and Vela 2.0 by wording.
- Image input is read by the d3 family and the Vela 3.0 d3 sizes. No current
  decision model reads audio or video for routing rules.

## Calibrated thresholds

The default `predicate` of a rule that writes none, on the models measured on
the router signal suite. Every other model uses 0.5, with an
`uncalibrated_preset` warning.

| Model | jailbreak | domain | pii | fact_check | feedback | harm | modality |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `vllm-sr/vela3-fast` | 0.88 | 0.41 | 0.18 | 0.9999 | 0.47 | 0.52 | 0.63 |
| `vllm-sr/Vela-2.0-0.3B` | 0.75 | 0.28 | 0.01 | 0.93 | 0.37 | 0.46 | 0.51 |
| `vllm-sr/Vela-2.0-0.8B` | 0.71 | 0.38 | 0.07 | 0.994 | 0.34 | | |
| `vllm-sr/Vela-2.0-4B` | 0.63 | 0.45 | 0.05 | 0.9984 | 0.33 | | |
| `vllm-sr/Vela-2.0-9B` | 0.42 | 0.46 | 0.14 | 0.998 | 0.35 | | |

The `hallucination` preset's spans keep the model's own thresholds. The
records behind these values are in the
[model runtime records](https://github.com/vllm-project/semantic-router/tree/main/src/model-runtime/docs/records).

## Configuration fields

### Where the deployment comes from

| Field | Meaning |
| --- | --- |
| `global.model_catalog.system.decision_model` | `{deployment: <name>}`: answers every question nothing else binds. Default `{deployment: primary}` |
| `global.model_catalog.bindings.<type>` | The deployment of one application signal type's rules |
| `recipes[].model_bindings.<type>` | The same, for one recipe |
| `global.model_catalog.deployments.<name>` | A deployment: `provider: model_runtime` (local or attached) or `provider: systemone` (a remote API) |
| `global.model_catalog.signal_timeout_ms` | The deadline of all of a request's signals |

A `systemone` deployment takes `endpoint`, `model`, `auth` (`api_key_env` or
`api_key_file`, or `credential: delegated`; `header`, `prefix`), `headers`,
`timeout_ms`, `max_retries`, `tls` and `capabilities` (`question_types`,
`presets`, `modalities`).

### Signals

A `systemone` rule, under `recipes[].signals.systemone`:

| Field | Meaning |
| --- | --- |
| `name` | The rule's name, which conditions and projections use |
| `question` | `type`, `instructions`, and `choices`, `levels` or `labels`; `threshold` and `head` for set and span |
| `labels` | The options or labels that match the rule |
| `ignore_labels` | Set and span: labels that never count |
| `predicate` | The bound the answer must meet: `gt`, `gte`, `lt`, `lte` |
| `deployment` | The deployment that answers |
| `stage` | `request` (default) or `response` |
| `input` | `history`, `previous_turns` (0 to 32), `tool_results`, `answer`, `context`, `image`, `audio`, `video` |
| `coverage` | `truncate` (default) or `full` |
| `on_unscanned` | With `coverage: full`: `match`, `no_match` or `unknown` (default) |
| `timeout_ms` | Up to 60,000 |

An application rule takes `name`, `description`, `deployment`, `timeout_ms`
and the fields its type allows:

| Type | Further fields |
| --- | --- |
| `domains` | `labels` (default `[name]`), `predicate` |
| `fact_check`, `user_feedbacks` | `labels`, `predicate` |
| `jailbreak` | `predicate`, `stage`, `input.history`, `on_unknown` |
| `safety` | `labels` (harm categories), `predicate`, `stage` |
| `pii` | `predicate`, `ignore_labels`, `input.history`, `input.tool_results`, `on_unknown` |
| `hallucination`, `refusal` | `predicate` |
| `turn_relation` | `labels`, `input.previous_turns` (default 4) |
| `complexity` | `labels` |
| `modality` | `labels`, `predicate` |

Any other field is refused at load.

### The `match` block

```text
match:
  recall: {method: embedding, top_k: 20, min_similarity: 0.3}    # optional; method: embedding | bm25
  judge:  {preset: relevance, top_k: 6, min_score: 0.5, max_candidates: 64}   # optional
```

Each feature writes the block where it finds similar items:
`response_cache.semantic.match`, `context.sources[].match`,
`context.compress.match` and `tools.select.match`.

| Field | Meaning |
| --- | --- |
| `recall.method` | `embedding` (default; the store's embedding deployment) or `bm25` (the lexical index) |
| `recall.top_k`, `recall.min_similarity` | How many candidates recall returns (required when they come from an index), and the least similarity: a cosine for `embedding`, a BM25 score for `bm25` |
| `judge.preset` | `relevance`, `same_request` or `tool_rank` |
| `judge.question` | A custom question instead of a preset: `choice` (the candidates are its options), `noul` or `score` |
| `judge.deployment` | The deployment that answers; without one, the decision model |
| `judge.top_k`, `judge.min_score`, `judge.min_ratio` | Which judged candidates are kept: the best `top_k`, those at least `min_score`, those at least `min_ratio` of the best |
| `judge.max_candidates` | The most candidates the judge reads: 255 (also the most) for a choice judge, 64 by default for a per-candidate judge; below 1, or above 255 on a choice judge, fails to load |
| `judge.on_error` | `skip` (default; keep the recall result, or the candidates as they came without recall) or `fail` (the feature's match fails) |
| `judge.timeout_ms` | The judge call's deadline; else the deployment's `timeout_ms`, else 1000 ms |

A block needs at least one stage (`match_empty` otherwise).

### Algorithms

| Field | Meaning |
| --- | --- |
| `algorithm.systemone` | The selector: `instructions`, `candidates` (model to description), `deployment`, `timeout_ms` |
| `algorithm.escalate.accept.systemone` | `{gte, deployment, timeout_ms}`: accept an answer whose `answer_check` probability is at least `gte` |
| `algorithm.escalate.accept.refusal` | `false`: accept an answer the `refusal` question does not flag |
| `algorithm.escalate.stages` | System One entrypoints: the decision models to ask in turn, each with native acceptance rules |

## Load-time codes

| Code | Kind | Means |
| --- | --- | --- |
| `span_required` | error | A span-needing rule or plugin resolves to a deployment without spans |
| `input_modality_unsupported` | error | A rule reads a medium its deployment does not read |
| `match_empty` | error | A `match` block has neither `recall` nor `judge` |
| `uncalibrated_preset` | warning | A preset is answered with the neutral threshold 0.5 |
| `renamed_signal_type` | warning | `signals.decision` or a `type: decision` condition, which load as `systemone` |
| `renamed_algorithm_type` | warning | `algorithm.type: decision`, which loads as `systemone` |
| `removed_signal_type`, `removed_field`, ... | error | A removed form; see the [migration guide](../installation/migrate-to-v0-5) |
