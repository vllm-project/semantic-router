# Long inputs: runtime memory and time on CPU

How much one long input costs the runtime process that serves it, before and
after [#4654](https://github.com/vllm-project/semantic-router/issues/4654).
The router's response cache embeds a request's text with `truncate` and no
token budget, so the Response API's 5 MiB input (`response-api-edge-large-input`)
reached Vela Embedding whole. In Kind it grew the runtime by about 1.9 GB,
and a 2 GiB container limit killed it. Vela 2.0 0.3B read every long request
whole in windows, which timed out long probes on CPU runners.

- **Date:** 2026-10-07.
- **Revisions:** `main` at `458447758` and this change, each in its own
  model-runtime image (`src/model-runtime/Dockerfile`, PyTorch 2.10.0 for
  CPU), one process at a time.
- **CPU:** node A, 32 vCPUs of an AMD EPYC 9575F KVM guest (cores 80–111),
  or 4 of them like a CI runner (80–83, `--threads 4`), with memory on NUMA
  node 1. Other workloads ran on the host's other cores, so the times are
  indicative.
- **Models:** at their pinned revisions, native engine, `exact`, FP32:
  `vllm-sr/Vela-1.0-Encoder-307M-Embedding` (a 32,768-token window),
  `vllm-sr/Vela-1.0-Encoder-307M-Guard` (classify in 512-token windows with
  255 of overlap, at most 32,768 tokens) and `vllm-sr/Vela-2.0-0.3B` (the
  jailbreak question over a `request` part; 8,192-token inputs).
- **Method:** `tools/input_memory.py`. Each case starts a fresh
  `vllm-srun serve` and sends one warm-up request. It then resets the
  process's peak RSS (`/proc/<pid>/clear_refs`) and sends the case as a
  `/v1/bundle` task. The growth is `VmHWM` minus the RSS before the request;
  the model's own memory, about 1.8 GiB, is not in it. A digest of each answer
  compares the two revisions. Embeddings go as the response cache sends them
  (`overflow: truncate`, no `max_tokens`). Classify windows and Vela 2.0 read
  text of common words in random sentences, so no two windows are the same.

## Embeddings (Vela Embedding)

Peak RSS growth of the request, MiB, and its time.

| Case | Input tokens | `main` | This change |
| --- | ---: | ---: | ---: |
| English, 100 KB | 22,225 (whole) | 1,308 (5.6 s) | 656 (6.7 s) |
| English, 150 KB | 33,335 | 1,929 (10.9 s) | 939 (11.6 s) |
| English, 1 MB | 222,225 | 2,057 (11.1 s) | 959 (11.6 s) |
| Chinese, 1 MB, no spaces | 250,003 | 2,053 (11.2 s) | 957 (11.4 s) |
| Base64, 1 MB | 681,373 | 2,190 (11.4 s) | 954 (11.3 s) |
| English, 5 MiB | 1,165,314 | 2,321 (12.4 s) | 968 (11.4 s) |
| English, 5 MiB, `reject` | 1,165,314 | 682 (1.4 s) | 55 (0.25 s) |
| Four English 5 MiB requests at once | 1,165,314 each | 2,993 (28.1 s) | 1,042 (22.6 s) |

Every case gives the same vector digest on both revisions, and both reject
the `reject` case with `max_length_exceeded`. This change reads 43,913 tokens
of the 5 MiB input, and 134,029 to 147,455 of the 1 MB inputs without spaces.

## Classify in windows (Vela 1.0 Guard)

| Case | `main` | This change |
| --- | ---: | ---: |
| 100 KB, 21,356 tokens, 83 windows | 23 (3.9 s) | 23 (3.9 s), the same 83 windows |
| 5 MiB, over the 32,768-token budget | `max_length_exceeded`, 703 (1.6 s) | `scan_budget_exceeded`, 57 (0.25 s) |

## Vela 2.0 0.3B, the jailbreak question

`main` reads a long part whole in 8,192-token windows. This change reads it
whole up to the model's scan budget, four inputs (32,768 tokens) on a CPU, and
fails the question past it, as the router's safety questions ask. A question
with `overflow: truncate`, as its routing questions ask, reads only the part's
first 8,192 tokens, in one forward.

| Case | `main` | This change, whole | This change, `overflow: truncate` |
| --- | ---: | ---: | ---: |
| 30 KB, 6,434 tokens (one input) | | 610 (1.8 s) | |
| 100 KB, 22,501 tokens | 1,077 (7.3 s) | 1,079 (7.3 s), the same answer | 1,015 (2.5 s) |
| 1 MB, 228,828 tokens | 1,202 (68.3 s) | `scan_budget_exceeded`, 36 (0.26 s) | |
| 5 MiB, 1,201,696 tokens | 1,574 (360.6 s) | `scan_budget_exceeded`, 62 (0.29 s) | 1,044 (2.8 s) |
| 30 KB on 4 cores | | 632 (10.8 s) | |
| 100 KB on 4 cores | 1,033 (49.1 s) | 1,033 (49.1 s), the same answer | 1,033 (16.0 s) |

The truncated 100 KB and 5 MiB parts give the same answer: their first 8,192
tokens are the same. On 4 cores the 0.3B reads about 2.2 ms per token, so a
safety question over the whole scan budget takes over a minute there; one that
cannot finish by the signals' deadline is content the guard did not read, and
the router's jailbreak and PII rules match it.

## Where the memory went

- **The forward over the window.** On `main`, a forward over 32,768 tokens
  grew the process by 1,814 MiB. Local layers on the CPU attended in query
  blocks, but copied the whole row's keys, values and mask at once: three
  padded copies, the unfolded keys and values at twice their size, the
  boolean mask and the float mask SDPA makes of it (`aten::where` over
  `[1, 3072, 128, 256]`). Calls of 4,096 query tokens bring the forward to
  861 MiB, and from 6.4 s to 2.3 s for the cache's six layers. These are one
  forward through the `task_heads` family on node A cores 80–159, at
  `246dde1fe`, whose engine and heads `458447758` keeps, and at this change.
- **Tokenizing the whole input.** 5 MiB of English is 1.17 million tokens.
  The tokenizer's peak is 160 to 400 bytes per token, and part of it stays
  allocated, under the forward that follows. Reading only as far as the
  budget needs leaves about 44,000 tokens, read in one pass, and a text
  certainly over a `reject` budget is not tokenized at all.
- **Vela 2.0's windows.** Reading a 5 MiB part took at least 147 windows of
  8,192 tokens and six minutes on 32 cores; the time grows with the length,
  so on a CI runner a 30,000-token request took 86 s to over 120 s. The scan
  budget bounds both, and a routing question reads one input.

The server-process regression test (`tests/test_input_memory.py`) sends the
same 5 MiB request to a tiny embedding package with the same window.
