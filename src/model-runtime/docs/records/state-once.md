# State-once Decision packages

A Decision package whose `attention_mode` is `prefix_bidir` reads each request's
state once and branches every question from it (`StateOnceModel`,
`EngineModel.branch`). This record measures that the branched answers are the
answers each question gets when it runs alone as a row under the same mask: at
the published BF16 precision branching moves answers about as much as batching
the per-row forward does, and every decision that differs is a near-tie. It
also times the served path against the Decision 3.0 path, one row per
question, on the same weights.

- **Date:** 2026-10-15; the `vllm-sr/d3-turbo` rows 2026-10-20.
- **Packages:** `vllm-sr/d3-turbo` (`689b3af1`), which declares
  `attention_mode: prefix_bidir`; and the released Decision 3.0 weights served
  under `prefix_bidir` (`tools/state_once_parity.py --attention-mode
  prefix_bidir` writes a copy of the package that declares it). Those weights
  were not built for the layout, so their answers move more than a
  `prefix_bidir` package's do; the paths must still agree on any weights.
- **Device:** one AMD Instinct MI325X (gfx942) per run, in the router's ROCm
  image (PyTorch 2.12.0 for ROCm 7.2, Triton 3.7.0, FLA 0.5.2) with this
  branch's runtime.
- **Requests:** 4,000 requests (29,339 questions, 24,282 with gold labels)
  sampled from a public decision benchmark; 1 to 64 questions per request,
  prompts up to about 8,000 tokens. For `vllm-sr/d3-turbo`, the tool's default
  requests: the many-question ticket at a ~400-token and a ~2,000-token state
  with 1, 4, 16 and 64 questions.

## State split

`tools/state_once_parity.py --golden --boundary-only`: the runtime's state
block and question blocks against the split that a `prefix_bidir` package's
bundled reference implementation gives for the same requests (the golden
fixture).

| Requests | Questions | Token IDs differ | State length differs |
| --- | ---: | ---: | ---: |
| The 4,000-request sample | 29,339 | 0 | 0 |
| 330 edge-case states: trailing spaces, newlines and punctuation, CRLF, a literal `Question:`, special tokens, Unicode, no-break spaces, JSON objects and arrays, a 3,000-character state, 300 random endings | 657 | 0 | 0 |
| The many-question ticket at two state lengths, 1 to 64 questions (Choice and Noul) | 118 | 0 | 0 |

## Answers

Decisions are the arg-max option; probability gaps are per question, the
largest over its options. `--floor` also runs each question alone as a row,
which measures how far batching the per-row path moves its own answers.

`vllm-sr/d3-turbo` at its published BF16 precision, the tool's default
requests (8 requests, 170 questions):

| Comparison | Identical decisions | Largest gap | Mean gap |
| --- | ---: | ---: | ---: |
| Branched vs per-row, as served | 169 (99.41%) | 0.032 | 0.0041 |
| Branched vs each question alone | 169 (99.41%) | 0.017 | 0.0039 |
| Each question alone vs per-row in a batch | 168 (98.82%) | 0.021 | 0.0041 |

Branching moves d3-turbo's answers about as much as batching moves the
per-row path's own (mean gap 0.0041 against 0.0041); each decision that
differs is a near-tie of one of the two paths (margin at most 0.008).

The Decision 3.0 d3-edge weights (`vllm-sr/d3-edge` `18d61df7`) served under
`prefix_bidir`, on the 4,000-request sample: branched vs per-row 99.54%
identical decisions (136 differ), per-row alone vs in a batch 99.70%, gold
accuracy flat (McNemar p = 0.21). Every BF16 decision that differs between two
paths is a near-tie of one of them.

## Latency

`tools/state_once_bench.py`: `Runtime.call("decisions", ...)` in process (no
HTTP), p50 of 10 timed calls after 3 warm-up ones, one request at a time;
throughput with 8 concurrent copies (2 warm-up and 5 timed rounds). The
released path runs each question as its own row; the state-once path reads
the state once. `usage.input_tokens` is what each path counts for the same
request. Every Decision 3.0 size runs as released and with its weights under
`prefix_bidir`.

p50 milliseconds of one request, released → state-once:

| Size | 400-token state: 1 / 8 / 64 questions | 2,000-token state: 1 / 8 / 64 questions | 1.59 MP image, 8 questions |
| --- | --- | --- | --- |
| d3-edge (0.6B) | 11 → 10 / 14 → 16 / 64 → 23 | 11 → 11 / 31 → 18 / 201 → 31 | 92 → 68 |
| d3-lite (0.8B) | 18 → 19 / 22 → 31 / 106 → 37 | 19 → 20 / 49 → 33 / 333 → 46 | 113 → 88 |
| d3-nano (2B) | 18 → 18 / 30 → 30 / 171 → 48 | 19 → 20 / 77 → 33 / 528 → 59 | 174 → 107 |
| d3-mini (4B) | 24 → 24 / 58 → 40 / 384 → 93 | 31 → 31 / 169 → 44 / 1,237 → 115 | 316 → 132 |
| d3-flash (9B) | 23 → 23 / 87 → 41 / 608 → 141 | 44 → 44 / 259 → 61 / 1,971 → 169 | 501 → 191 |
| d3 (27B) | 58 → 58 / 287 → 106 / 2,113 → 432 | 128 → 130 / 897 → 176 / 7,123 → 510 | 1,541 → 343 |

Questions per second with 8 concurrent copies, released → state-once:

| Size | 2,000-token state, 8 questions | 2,000-token state, 64 questions | 1.59 MP image, 8 questions |
| --- | --- | --- | --- |
| d3-edge | 402 → 1,290 | 434 → 3,018 | 36 → 40 |
| d3-lite | 217 → 752 | 231 → 1,936 | 33 → 38 |
| d3-nano | 130 → 581 | 134 → 1,346 | 26 → 35 |
| d3-mini | 52 → 263 | 55 → 610 | 18 → 31 |
| d3-flash | 33 → 168 | 29 → 397 | 13 → 25 |
| d3 | 9 → 47 | 9 → 127 | 5 → 17 |

`usage.input_tokens` for the same requests, released → state-once: 352 → 352 (400-token state, one
question), 2,702 → 770 (eight questions), 21,548 → 4,160 (64 questions); 1,116 → 1,116, 8,814 → 1,534
and 70,444 → 4,924 on the 2,000-token state; 13,438 → 2,112 for the image request.

A lone question runs as its row, so it costs what it costs on the released
path. Every request stays below 1 s. At the smallest sizes a branched forward
is bound by host-side kernel launches rather than the GPU, so eight
questions on the 2,000-token state gain 1.5-2.3x there against 3.9-5.1x from
4B up.
