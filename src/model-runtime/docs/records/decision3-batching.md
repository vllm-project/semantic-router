# Decision 3.0: content caches and cross-request batching

The `decision3` family keeps two content-addressed caches whatever its engine
arguments, and batches rows of concurrent requests, with media rows of
several requests in one forward. The caches change no answer. Batching,
opt-in when this was measured and the runtime's default now, changes answers
by rounding, about as much as answering each request alone does when a
client sends the same questions in separate requests.

- **Date:** 2026-10-13.
- **Device:** AMD Instinct MI325X (gfx942), the router's ROCm image
  (PyTorch 2.12.0 for ROCm 7.2, Triton 3.7.0, FLA 0.5.2), Engine mode: the
  Router serving the public System One API in front of managed workers.
- **Models:** the built-in Decision 3.0 revisions of
  `registry/tables/decision3.py`.

## Content caches (batched or not, exact)

- **Processed media.** A request's image and video data URLs are keyed by the
  SHA-256 of their text. A repeated image or video skips base64 decoding,
  PIL or OpenCV decoding and preprocessing, and reuses its FP32 patch rows
  (4 GiB per model, least recently used out).
- **Vision features.** The tower's features of a request's images are keyed
  by the images' digests and by the number of copies one tower call holds
  (one per question of a pass, as the released runtime runs them). A
  request's later passes of the same size, and later requests with the same
  images and pass size, reuse them (2 GiB of device memory per model).

Answers are byte-identical with and without the caches: 20 / 20 requests on
d3-edge and 9 / 9 on d3 (text, PNG, JPEG and video requests with 1 to 32
questions), each request asked several times so later answers come from the
caches, against the runtime before this change.
`tests/test_decision3_reference.py` still checks every image path bit for bit
against Transformers. Repeated media get much faster (d3-edge, Engine mode,
median ms):

| Request | Before | Repeated, cached |
| --- | --- | --- |
| 1.6 MP PNG, 1 question | 95 | 55 |
| 1.6 MP PNG, 32 questions | 682 | 177 |
| 10 s 1080p H.264 video, 1 question | 322 | 126 |
| 10 s 1080p H.264 video, 32 questions | 498 | 281 |

On d3, a repeated 1.6 MP PNG with 8 questions takes 1,278 ms instead of
1,888 ms.

## Tokenizing a request at once (batched or not, exact)

A request's question prompts are tokenized in one batch call, which runs
without the GIL, so the worker thread keeps launching forwards meanwhile.
On the 4,000-request public sample (3,969 requests, 29,200 questions, mostly
long states) on one d3-edge worker of an MI300X, all 3,969 exact answers are
byte-identical, and the sample takes 111.7 s instead of 126.0 s with each
request alone (8 clients) and 92.9 s instead of 115.4 s batched (48 clients). The GPU
was busy 56% of the time before, because the worker thread waited for the
GIL; afterwards, 70% to 84%.

## Batching for Decision 3.0

Rows of concurrent requests share forwards. Text rows run in one left-padded
pass as a request's rows do. Media rows run in one pass whatever their
requests: each request's images go through the tower once and their features
are copied into its rows, where the exact path runs one copy per row. One
coalesced forward holds as many tokens as its model processes in about
0.3 seconds at the measured cost per token, so a full queue is answered over
several short forwards.

**Accuracy.** A 4,000-request stratified sample of the Decision Index 0.3
public suite (29,339 questions), answered by one worker with each request
alone (a request option of the time) and then batched with 32 concurrent
clients, so rows of different requests shared forwards. Gold accuracy is the
share of Choice and Noul questions answered with their expected option, with
a paired bootstrap 95% interval over requests. The last rows are the
unbatched path against itself, with every question sent as its own request
(one row per forward instead of eight): the noise of the unbatched path's
batch composition.

| Model | Comparison | Argmax agreement | Max / mean \|Δp\| | Gold accuracy | Δ, 95% CI (points) |
| --- | --- | --- | --- | --- | --- |
| d3-edge | batched vs alone | 99.69% (91 changed) | 0.115 / 0.0027 | 75.17% → 75.18% | +0.01 [−0.05, +0.08] |
| d3-mini | batched vs alone | 99.87% (38 changed) | 0.059 / 0.0015 | 86.81% → 86.81% | −0.00 [−0.05, +0.04] |
| d3-edge | alone, questions sent one by one | 99.69% (91 changed) | 0.148 / 0.0028 | 75.17% → 75.19% | +0.02 [−0.04, +0.09] |
| d3-mini | alone, questions sent one by one | 99.86% (41 changed) | 0.059 / 0.0015 | 86.81% → 86.79% | −0.02 [−0.06, +0.03] |

**Throughput** (requests/s through Engine mode, two workers on one GPU,
closed-loop clients; text is a 330-token ticket with one question):

| Model | Alone, best replica count | Batched, 2 workers | p50 / p99 at that load |
| --- | --- | --- | --- |
| d3-edge | 313 (4 workers) | 1,304 | 47 / 97 ms |
| d3-lite | 163 (4 workers) | 700 | 91 / 161 ms |
| d3-nano | 137 (4 workers) | 434 | 143 / 208 ms |
| d3-mini | 90 (2 workers) | 171 | 91 / 141 ms |

Each worker admits at most 32 requests in flight; the Router answers 503 to
requests over that.
