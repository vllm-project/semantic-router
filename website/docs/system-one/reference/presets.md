---
title: System One presets
sidebar_label: Presets
sidebar_position: 1
description: Every built-in System One question the Router asks, with its type, options, stage and the input it reads.
---

# System One presets

A preset is a built-in System One question: the application signals, the
judges and the checks each ask one. The wording and options below are the
canonical, zero-shot definition; a family with a trained binding is asked the
trained form (see the [support matrix](./support-matrix.md)).

## Preset catalog {#preset-catalog}

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

## Harm categories

A `safety` rule's `labels` name some of these 34 categories, the labels of the
trained vela3 hazard question:

- **AEGIS 2.0 (21):** Controlled/Regulated Substances,
  Copyright/Trademark/Plagiarism, Criminal Planning/Confessions,
  Fraud/Deception, Guns and Illegal Weapons, Harassment, Hate/Identity Hate,
  High Risk Gov Decision Making, Illegal Activity, Immoral/Unethical, Needs
  Caution, Other, PII/Privacy, Political/Misinformation/Conspiracy, Profanity,
  Sexual, Sexual (minor), Suicide and Self Harm, Threat, Unauthorized Advice,
  Violence.
- **MLCommons hazard codes (13):** S1 (violent crimes), S2 (non-violent
  crimes), S3 (sex-related crimes), S4 (child sexual exploitation), S5
  (defamation), S6 (specialised advice), S7 (privacy), S8 (intellectual
  property), S9 (indiscriminate weapons), S10 (hate), S11 (suicide and
  self-harm), S12 (sexual content) and S14 (code interpreter abuse).

How each family is asked them:

- vela3-fast asks its trained 34-label set once.
- At the request stage, the Vela 3.0 d3 sizes are asked one `hazard:<label>`
  yes/no question per category, in the form vela3-lite's hazard adapter was
  trained on; vela3-mini, -flash and -turbo ship no hazard adapter and answer
  those questions zero-shot.
- Every other model answers the same labels zero-shot, in the same call.

## Personal data labels

`PERSON`, `EMAIL_ADDRESS`, `PHONE_NUMBER`, `US_SSN`, `CREDIT_CARD`,
`LOCATION`, `IP_ADDRESS`, `DATE_TIME`, `US_DRIVER_LICENSE`, `US_PASSPORT` and
`IBAN_CODE`. A `pii` rule's `ignore_labels` and a masking plugin's
`entity_types` name some of them.
