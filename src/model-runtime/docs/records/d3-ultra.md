# d3-ultra: prompt layout header-v1 and a temperature per request size

`vllm-sr/d3-ultra` is a state-once Decision 3.0 package (`attention_mode:
prefix_bidir`) whose `decision_config.json` declares two more fields:
`prompt_layout: header-v1`, which lists a text-only request's questions in
front of its state, and `temperature_by_questions`, which answers a
one-question request at 1.046 and any other at 1.0. This record checks that
the runtime gives the model exactly the inputs the package's own runtime gives
it, compares the answers of the two with d3-turbo's own comparison as the
yardstick, and times d3-ultra against d3-turbo on the same build.

- **Date:** 2026-10-25.
- **Packages:** `vllm-sr/d3-ultra` (`2f9a5e5e`) and `vllm-sr/d3-turbo`
  (`689b3af1`).
- **Device:** one AMD Instinct MI325X (gfx942) per run, in the router's ROCm
  image (PyTorch 2.12.0 for ROCm 7.2, Triton 3.7.0, FLA 0.5.2) with this
  branch's runtime. The packages' own runtime ran through `AutoModel` with
  PyTorch 2.13.0, Transformers 5.17.0, Triton 3.8.0 and FLA 0.5.2.
- **Requests:** a fixed sample of 200 text requests (1,575 questions, 1 to 64
  per request, 123 of them with one question) and 40 requests with images (one
  question each).

## Inputs

For every text question of the sample, the token IDs the runtime builds (the
question list inside the state block), its state split and the request's
temperature against the package runtime's (`d3_format.render` with the
request's valid questions as the list, its state split and its
`temperature_for`):

| Questions | Token IDs differ | State length differs | Temperature differs |
| ---: | ---: | ---: | ---: |
| 1,575 | 0 | 0 | 0 |

## Answers

The runtime against the package's own runtime on the same requests. A
decision is the arg-max option (Yes / No: the side of 0.5); a question's gap is
the largest over its options. "Alone" serves one request at a time; "16 in
flight" keeps 16 requests in flight, so the scheduler merges them into shared
forwards.

| Package | Pass | Identical decisions | Image requests | Largest gap | Mean gap |
| --- | --- | ---: | ---: | ---: | ---: |
| d3-ultra | alone | 1,610 (99.69%) | 40 / 40 | 0.045 | 0.0008 |
| d3-ultra | 16 in flight | 1,614 (99.94%) | 40 / 40 | 0.051 | 0.0007 |
| d3-turbo | alone | 1,613 (99.88%) | 39 / 40 | 0.094 | 0.0009 |
| d3-turbo | 16 in flight | 1,613 (99.88%) | 39 / 40 | 0.094 | 0.0010 |

Every decision that differs, for both packages, is a near-tie of the package
runtime: its two most likely options are less than 0.05 apart (9 such
questions for d3-ultra, 7 for d3-turbo), and no question with a wider margin
differs. One-question requests fit the 1.046 temperature: their mean gap is
0.0026, against 0.0057 for the package runtime's answers taken at 1.0. The
alone pass is deterministic: it gives the same answers with and without the
recorded kernel choices.

## Latency

`Runtime.call("decisions", ...)` in process (no HTTP), p50 and p90 of 15 timed
calls after 3 warm-up ones, one request at a time, `max_num_batched_tokens`
4,096 and `max_num_seqs` 256 for both packages. Text: the runtime's public
many-question ticket (about 300 tokens) with the first n of 255 Choice and
Yes / No questions. Photo: a 1,280 x 1,600 JPEG, one pixel changed per call so
the tower always runs, with the same questions.

| Questions | d3-ultra text | d3-turbo text | Text input tokens | d3-ultra photo | d3-turbo photo |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 58 / 63 | 59 / 70 | 380 / 337 | 292 / 335 | 307 / 329 |
| 4 | 82 / 85 | 78 / 96 | 590 / 490 | 316 / 335 | 332 / 359 |
| 16 | 163 / 167 | 134 / 139 | 1,327 / 1,059 | 363 / 385 | 375 / 400 |
| 64 | 439 / 445 | 343 / 348 | 4,303 / 3,351 | 603 / 638 | 617 / 645 |
| 128 | 803 / 817 | 606 / 611 | 8,444 / 6,455 | 905 / 963 | 918 / 945 |
| 255 | 1,675 / 1,705 | 1,188 / 1,199 | 16,764 / 12,630 | 1,511 / 1,555 | 1,526 / 1,554 |

Milliseconds as p50 / p90; input tokens as d3-ultra / d3-turbo
(`usage.input_tokens`; the photo requests count 1,655 to 13,948 for both). The
question list adds 13% (one question) to 33% (255) to a text request's
tokens, and d3-ultra's text latency grows with them: its ratio to d3-turbo's,
divided by the token ratio, is 0.87 to 1.06. Requests with images keep the d3
layout and take the same time as d3-turbo's.

Throughput with 16 concurrent copies (2 warm-up and 5 timed rounds), requests
per second, d3-ultra / d3-turbo: text with 16 questions 7.2 / 8.8, text with 64
questions 2.3 / 2.9, photo with 16 questions 2.9 / 2.9; per input token, the
same.

A request's own forwards are bounded by the kernels' limit, not by
`max_num_batched_tokens`, so the longer state needs no larger token budget:
at 32,768 the latencies above stay within 6%.
