# Decision 3.0: lanes, packed forwards, faster images and batch limits by device

Merged Decision 3.0 forwards plan text rows and media rows in lanes of their
own, pack rows of different lengths without padding where the device has
variable-length kernels, read every request's images in one tower call with
one variable-length attention per block, decode request media in worker
processes, and take their batch limits from the device's memory class and a
startup calibration. On d3-edge a mix of text and image requests answers 2.0 times as many requests per second and a long-tailed length mix 2.3 times as many; d3-mini's mix 31 % more; text-only load is unchanged within run-to-run noise, and d3 (27B), GPU-bound, moves within a few percent. Gold accuracy on the public sample does
not change, and every decision that changed was a near-tie.

- **Date:** 2026-10-14.
- **Device:** AMD Instinct MI325X (gfx942), one GPU per side; the router's
  ROCm image (PyTorch 2.12.0 for ROCm 7.2, Triton 3.7.0, FLA 0.5.2), whose
  PyTorch has `torch.nn.attention.varlen.varlen_attn`.
- **Before:** `main` at 36bc632. **After:** this change. Both with default
  engine arguments, side by side on two GPUs of one node, `vllm-srun serve`
  of the built-in revisions (`registry/tables/decision3.py`), each with its
  own warm kernel cache.

## What each part does

- **Lanes.** Text rows are no longer padded to the width of image rows, no
  text row waits for an image pass, and an image forward's time no longer
  shrinks later text forwards. The batches of both lanes that one planning
  round makes share a scheduler turn: without that, every round of arrivals
  left two batches for one forward, and the backlog filled with small forwards
  (d3-edge, mix: 63 requests/s instead of 230 in a first version).
- **Images.** One variable-length attention call per tower block over every
  image of a merged forward instead of one call per image; on its own it gave
  about 14 % of the d3-edge mix's gain (packing off: 195 requests/s, on: 222,
  in the diagnostic runs). Media worker processes return the resized uint8
  pixels (an eighth of the FP32 patch rows); the patch rows are bit for bit
  the same as decoding in the serving process.
- **Packed forwards.** A merged pass packs its rows when left-padding them
  would add more than 10 % to its tokens. Rows of like lengths (the text-only
  mix: a ticket with one or four questions) stay padded, which is slightly
  cheaper than packing them; the long-tailed mix packs.
- **Batch limits.** Unset limits take the class of the device (MI325X: 32,768
  tokens and 256 rows), then the calibration: d3-edge keeps 32,768 tokens (its
  cost per token is lowest there, and that forward takes under 0.3 s); d3-mini
  takes about 19,000, its 0.3 s cap (18,944 or 19,200 on different starts, by
  rounding to whole rows), since its cost per token is flat from 16,384 on; d3
  takes 3,328, as its 4,096-token forward already takes 0.37 s.

## Throughput and latency

| Model | Mix | Before (main) | After (this PR) | Change |
| --- | --- | ---: | ---: | ---: |
| d3-edge | 60/25/15 mix | 122.0 req/s · 502 / 646 ms | 244.5 req/s · 250 / 338 ms | +100 % |
| d3-edge | text only, 3 alternating runs per side (mean, min-max) | 707.3 req/s (704.1-709.1) · 90 / 99 ms | 708.6 req/s (705.2-710.7) · 90 / 99 ms | +0 % |
| d3-edge | images only | 33.2 req/s · 1794 / 1925 ms | 58.9 req/s · 1023 / 1183 ms | +78 % |
| d3-edge | long-tailed | 112.9 req/s · 542 / 672 ms | 259.1 req/s · 232 / 331 ms | +129 % |
| d3-mini | 60/25/15 mix | 32.9 req/s · 1785 / 2069 ms | 43.2 req/s · 1429 / 1616 ms | +31 % |
| d3-mini | text only, 5 alternating runs per side (mean, min-max) | 84.5 req/s (82.0-86.2) · 739 / 813 ms | 85.6 req/s (84.3-87.0) · 724 / 803 ms | +1 % |
| d3-mini | images only | 11.9 req/s · 4579 / 4639 ms | 13.3 req/s · 4101 / 4343 ms | +12 % |
| d3-mini | long-tailed | 25.7 req/s · 2252 / 3137 ms | 25.0 req/s · 2213 / 3298 ms | -3 % |
| d3 | 60/25/15 mix | 6.9 req/s · 6861 / 7382 ms | 7.2 req/s · 6780 / 7167 ms | +5 % |
| d3 | text only, 3 alternating runs per side (mean, min-max) | 11.9 req/s (11.8-12.1) · 4514 / 4949 ms | 11.7 req/s (11.4-12.0) · 4518 / 5058 ms | -2 % |
| d3 | images only | 1.9 req/s · 15695 / 16073 ms | 1.8 req/s · 16195 / 16378 ms | -2 % |
| d3 | long-tailed | 5.9 req/s · 6403 / 8493 ms | 6.2 req/s · 6906 / 7800 ms | +5 % |

d3-mini's long-tailed mix is the one load that came out lower: 3 % in this
session (one run per side) and about 2 % across the day's runs, with p50
slightly lower and p90 higher. That is inside the run-to-run spread below, but
it is not a gain.

The mixes: 60 / 25 / 15 % of a ~330-token support ticket with 1 question, the
same ticket with 4 questions, and a new 1,456 x 1,092 (1.59 MP) photo with 1
question; text only (the first two, 60 : 25); the photo only; and a long tail
of 2,000 requests drawn from the public sample (a state with up to 4 of its
questions; state tokens p50 50, p90 1,290, p99 6,671, at most 7,303). Every
image request carries new pixels, so no cache can hit. Closed loop: 64 clients
in four processes, each sending its next request when the previous answer
arrived, 30 s after an 8 s warm-up, both sides with warm kernel caches. The
text-only rows alternate main and this change on one pair of servers and give
the mean of the runs with their range; the d3-mini rows come from one session
of the final code, the d3-edge and d3 rows of the other mixes from an earlier
session whose calibrated budgets the final rule keeps.

### Run-to-run noise

| Model | Before (main) | After (this PR) |
| --- | --- | --- |
| d3-edge | 676.7, 691.4, 709.1, 708.9, 704.1 (spread 5 % of the mean) | 710.7, 710.0, 705.2 (spread 1 % of the mean) |
| d3-mini | 83.6, 86.4, 85.3, 79.7, 82.4, 84.2, 86.2, 82.0, 85.9, 84.4 (spread 8 % of the mean) | 84.9, 86.1, 85.7, 84.3, 87.0 (spread 3 % of the mean) |
| d3 | 11.7, 11.8, 11.8, 12.1 (spread 3 % of the mean) | 11.7, 12.0, 11.4 (spread 5 % of the mean) |

Main's own text-only throughput moves between runs by up to 11 % of its mean (d3-edge 677 to 709, d3-mini 77.4 to 86.4, d3 11.7 to 12.1 requests per second over the day). Under closed-loop load a planning round batches the requests that arrived during the forward in flight, and requests answered together return together, so the same load settles into a different number of forwards from one run to the next and keeps it for the run: the servers' counters show about 4,500 to 8,100 tokens per forward on average across the ten d3-mini text-only runs, and throughput follows that size on both sides (82.0 requests per second at 4,531 tokens, 84.2 to 84.9 at 5,800 to 6,700, 87.0 at 8,089). The text-only differences of this change (d3-edge +0.2 %, d3-mini +1.2 %, d3 -1.7 %) are within that spread.

### d3-mini, text only

A first measurement had d3-mini's text-only mix 16 % slower with this change
(70.4 against 83.6 requests per second). That was the first start of this
change's server, with an empty kernel cache: forwards of new shapes compiled
and tuned their kernels inside the measured window. A warm rerun was still
6 % slower (72.5 against 77.4) because that start had calibrated to 8,192
tokens. The rule then was the smallest budget within 2 % of the cheapest cost
per token, and on d3-mini's flat curve timing noise decided: four starts chose
16,384, 16,384, 8,192 and 16,384 tokens. The calibration now keeps the largest
budget on the plateau, capped at 0.3 s per forward, which is about 19,000
tokens on every start. With it, 5 alternating runs per side give 85.6 requests per second (84.3 to 87.0) against main's 84.5 (82.0 to 86.2), p50 724 against 739 ms.

## Answers

| Model | Comparison | Questions | Decisions changed | Max / median \|Δp\| |
| --- | --- | ---: | ---: | --- |
| d3-edge | before vs AutoModel (fixed workloads) | 31 | 0 | 0.0169 / 0.00256 |
| d3-edge | after vs AutoModel (fixed workloads) | 31 | 0 | 0.0169 / 0.00256 |
| d3-edge | after vs before (384-prompt set, one request at a time) | 384 | 1 | 0.0445 / 0.00096 |
| d3-edge | after vs before (public 4k sample, 32 clients) | 29339 | 127 | 0.1318 / 0.00112 |
| d3-mini | before vs AutoModel (fixed workloads) | 31 | 0 | 0.0213 / 0.00301 |
| d3-mini | after vs AutoModel (fixed workloads) | 31 | 0 | 0.0143 / 0.00301 |
| d3-mini | after vs before (384-prompt set, one request at a time) | 384 | 1 | 0.0164 / 0.00234 |
| d3-mini | after vs before (public 4k sample, 32 clients) | 29339 | 46 | 0.0915 / 0.00013 |
| d3-mini | before vs before (public 4k sample, two runs of main) | 29339 | 30 | 0.0416 / 0.00003 |
| d3 | before vs AutoModel (fixed workloads) | 31 | 0 | 0.0090 / 0.00140 |
| d3 | after vs AutoModel (fixed workloads) | 31 | 0 | 0.0090 / 0.00000 |
| d3 | after vs before (384-prompt set, one request at a time) | 384 | 0 | 0.0195 / 0.00192 |
| d3 | after vs before (public 4k sample, 32 clients) | 29339 | 9 | 0.0719 / 0.00001 |

- **Public sample:** the 4,000-request stratified sample of the Decision Index
  0.3 public suite of [decision3-batching.md](decision3-batching.md), answered
  by each side with 32 concurrent clients, so rows of different requests share
  forwards on both. Gold accuracy is the share of Choice and Noul questions
  answered with their expected option; the interval is a paired bootstrap over
  requests. d3-mini's public sample ran again on the final code next to a second run of main: two runs of main differ in 30 decisions (margins up to 0.028), main and this change in 46 (margins up to 0.050).
- **AutoModel:** the package's own `AutoModel.system_one` with the runtime's
  kernel choices, on the fixed text and image workloads (1, 4 and 16 questions
  about a ticket, a photo with 1 and 8 questions, a 31,975-token case log),
  sent to each side one request at a time.
- **384-prompt set:** three states with 128 questions each, one request at a
  time.

| Model | Before | After | Δ (points) | 95 % CI | Changed decisions' margin (max / median) |
| --- | ---: | ---: | ---: | --- | --- |
| d3-edge | 75.37 % | 75.38 % | +0.004 | [-0.072, +0.075] | 0.1227 / 0.0095 |
| d3-mini | 87.03 % | 87.05 % | +0.025 | [-0.025, +0.074] | 0.0502 / 0.0054 |
| d3 | 92.10 % | 92.11 % | +0.012 | [-0.004, +0.031] | 0.0694 / 0.0057 |

## Startup and memory

| Model | Side | Ready, cold / warm (s) | First answer, cold (s) | Peak device memory above idle (GiB) |
| --- | --- | --- | ---: | ---: |
| d3-edge | before | 8.43 / 8.34 | 8.46 | 39.0 |
| d3-edge | after | 8.93 / 8.85 | 8.97 | 32.1 |
| d3-mini | before | 13.49 / 11.04 | 14.18 | 33.5 |
| d3-mini | after | 17.6 / 14.18 | 17.67 | 33.6 |
| d3 | before | 20.34 / 17.54 | 21.09 | 61.4 |
| d3 | after | 22.09 / 18.99 | 22.19 | 61.1 |

Ready comes 0.5 to 4 s later: the calibration's timed forwards (about 3 s on
d3-mini, which times forwards up to 32,768 tokens) and the start of the media
worker processes; an image that arrives before they are up waits for them
once. Peak memory was measured under the closed-loop load; d3-mini's with the
previous rule's 16,384-token budget. Its calibration measures 2.06 GiB of
activations for a 16,384-token forward and 4.12 GiB for 32,768, so the final
budget of about 19,000 adds about 0.35 GiB.
