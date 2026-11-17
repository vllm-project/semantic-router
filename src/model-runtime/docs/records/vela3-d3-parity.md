# Vela 3.0 d3-size parity

The `vela3_d3` family against vela3-lite's own runtime (`modeling_vela3.py`,
`Vela3MiniModel.system_one`) on CPU. Every span question and every question
with a gated adapter answers as the package does. Questions that run with
every adapter off differ from the package by up to 0.023 on this host, the
same way the plain `decision3` family differs from the Decision 3.0 runtime
on it. No decision and no span set changed. vela3-mini and vela3-flash, which
write definitions for open labels, are below: each differs from its package
on one label set's definition.

- **Date:** 2026-11-02.
- **Packages:** `vllm-sr/vela3-lite` `c789afd6` over its pinned base
  `vllm-sr/d3-lite` `9c107031` (the same weights as the built-in d3-lite
  `a8d25a91`: identity `fda5cfc7`).
- **Host:** AMD Ryzen 7 9800X3D (8 cores, 16 threads, AVX-512 with native
  BF16), PyTorch 2.10.0+cpu on both sides.
- **Reference:** the package's runtime over the base's published runtime,
  with Transformers 5.17.0 and Accelerate 1.15.0 (the packages'
  requirements), in its own environment and process. The runtime side
  serves the package through the native engine, each request alone (its
  released forwards).
- **Requests:** `tools/vela3_d3_parity.py generate --count 40 --seed 0`: 40
  requests, 310 questions. Each has one to three ordinary questions (Choice,
  Noul, Score; adapters off), one to four trained router questions (the
  decision adapter and the domain / feedback / fact-check task adapters),
  zero to two `hazard:` Nouls (the hazard adapter), sometimes an opt-in
  question (`decision_adapter: true`; vela3-lite ships no opt-in adapter),
  and one to three span questions (PII, unsupported claims, toxic, evidence,
  relation wordings, named and described open labels, request thresholds),
  over text states, two- and three-part states and a 12-copy document read
  in windows.

## Answers

| Questions | Count | Identical values | Within 1e-4 | Largest difference |
| --- | --- | --- | --- | --- |
| All | 310 | 216 | 267 | 0.023 |
| Span (sets, probabilities, thresholds) | 86 | — | 86 | below 1e-4 |

No Choice arg-max, Noul side or span set differs, and every request's
input tokens agree. Of the 20 requests without the long document:

| Questions | Identical values | Largest difference |
| --- | --- | --- |
| Span | 45 / 45 | 0 |
| Gated (router, task, hazard adapters) | 64 / 73 | 2.4e-7 |
| Adapters off | 31 / 45 | 0.012 |

## Where the differences come from

The rows that differ run with every adapter off; the gated rows of the same
forwards agree to 2.4e-7. The base alone shows the cause, with no
`vela3_d3` code involved: the plain `decision3` family against the
Decision 3.0 runtime, on the ordinary questions of the same requests.

| Base | Requests | Identical values | Largest difference |
| --- | --- | --- | --- |
| d3-lite `a8d25a91` (built in), short inputs | 8 | 14 / 14 | 0 |
| d3-lite `a8d25a91` (built in), the 12-copy document (about 3,000 tokens) | 5 | 1 / 9 | 1.2e-3 |
| d3-lite `9c107031` (vela3-lite's base), the same long requests | 5 | 1 / 9 | 1.2e-3 |

Short rows agree exactly; long rows, and short rows left-padded in a forward
that also holds long trained router prompts, do not. The gated delta rule is
not the cause: the runtime's kernel and Transformers' agree bit for bit at
64 to 3,000 tokens. The built-in d3-lite's golden answers on this host also
differ from the recorded `cpu` reference (13 of 16 numbers equal, the
largest difference 0.172), the host class #5273 and #5274 describe.

## vela3-mini and vela3-flash

The runtime writing open-label definitions, against each package's own
runtime, on `generate --count 40 --seed 0` (290 questions).

- **Date:** 2026-11-16.
- **Host:** an Intel Xeon Platinum 8568Y+ virtual machine (AVX-512 with BF16,
  no AMX exposed), both sides with 12 threads pinned to 16 cores.
- **Environments:** PyTorch 2.10.0+cpu on both sides; Transformers 5.17.0 and
  Accelerate 1.15.0 for the packages' runtime.
- **Precision:** the runtime at `--dtype bfloat16`, as the packages run a BF16
  backbone with an FP32 readout. At `--dtype auto` the runtime computes
  Decision 3.0 in FP32 on CPUs ([record](d3-cpu-fp32.md)) against the
  packages' BF16: even spans that write no definition then differ (up to
  0.013 on vela3-mini, 0.038 on vela3-flash), and no ordinary or opt-in
  question is identical.

### vela3-mini

`vllm-sr/vela3-mini` `bbc0206e` over `vllm-sr/d3-mini` `566da0ea`
(`definitions.writer` `base`). No `hazard:` Nouls: vela3-mini ships no hazard
adapter. Opt-in questions run its opt-in adapter, so the second row holds the
ordinary and opt-in questions rather than adapters-off ones. The package's
runtime took 2,809 s, the runtime 1,349 s.

| Questions | Count | Identical values | Changed | Largest difference |
| --- | --- | --- | --- | --- |
| Gated (router and task adapters) | 106 | 76 | 0 | 0.0082 |
| Ordinary and opt-in questions | 105 | 56 | 0 | 0.013 |
| Span, labels with descriptions or a label set | 41 | 41 | 0 | 0 |
| Span, open labels without a description | 38 | 34 | 0 | 0.019 |

No decision or span set changes, and every request's input tokens agree.
Of the 61 definitions the package writes (one per question and label), 54 are
identical. The 7 that differ are one label set, `["birthplace"]`, asked by 7 questions: the
package writes "The text span covers the place where a person was born." and
the runtime "The text span that covers the place where a person was born."
Four of those questions read the same spans with scores up to 0.019 apart;
the other three read no span on either side.

### vela3-flash

`vllm-sr/vela3-flash` `af48828d` over `vllm-sr/d3-flash` `8b35c853`
(`definitions.writer` `grafted_head`: the base with
`labeldef_lm_head.safetensors`). vela3-flash ships no hazard adapter and no
opt-in adapter, so it has no `hazard:` Nouls and its opt-in questions run every
adapter off. The package's runtime took 3,336 s,
the runtime 2,240 s.

| Questions | Count | Identical values | Changed | Largest difference |
| --- | --- | --- | --- | --- |
| Gated (router and task adapters) | 106 | 74 | 0 | 0.006 |
| Ordinary and opt-in questions | 105 | 55 | 0 | 0.0063 |
| Span, labels with descriptions or a label set | 41 | 41 | 0 | 0 |
| Span, open labels without a description | 38 | 37 | 1 | 0.13 |

No decision changes, and every request's input tokens agree. The runtime
Of the 61 definitions the package writes, 53 are identical. The 8 that differ
are one label set, `["answer"]`, asked by 8 questions: the package writes "The
text span that contains the answer to the question." and the runtime "The
answer entity type covers the text span that directly responds to a question
or prompt." One of those questions changes its span set (request 8): the
package also reads the word "answer" itself (score 0.487, threshold 0.39),
and both read the other two spans; "Monday" scores 0.739 against the
runtime's 0.610, the table's 0.13. The other seven read no span on either
side.

### Definitions

Each differing definition forks where the package's greedy step is nearly a
tie in BF16, so the choice depends on last-bit rounding:

- `birthplace` (vela3-mini) forks at its fourth token: `covers` 15.375
  against `that` 15.25, two BF16 steps apart (a step is 0.0625 below 16).
  Two other steps of the same definition are one step apart (`text` against
  `entity`, `place` against `location`), and both sides agree there.
- `answer` (vela3-flash) forks at its second token: `answer` and `text` tie
  at 17.125.

The package's `generate` ends a definition at the checkpoint's
`eos_token_id` (`<|endoftext|>`): after the `birthplace` definition it emits
`<|endoftext|>` with no `<|im_end|>`. The runtime ends a definition at the
same token.

## vela3-turbo (GPU)

`vllm-sr/vela3-turbo` `ce95e6d9` over `vllm-sr/d3-turbo` `da17b5d0` (a 26B
base, 72 GiB) on a GPU, as d3-turbo's own records are measured: one NVIDIA
RTX PRO 6000 Blackwell Workstation Edition (96 GB, sm_120) with PyTorch
2.10.0+cu128 on both sides, the same request generator (290 questions) and
`--device cuda:0` (bar: max |dp| <= 0.02). The package's runtime took 77 s,
the runtime 65 s.

| Questions | Count | Identical values | Changed | Largest difference |
| --- | --- | --- | --- | --- |
| Gated (router and task adapters; noncausal replays) | 106 | 58 | 0 | 0.018 |
| Adapters off (d3-turbo's state-once path) | 105 | 0 | 0 | 0.015 |
| Span | 79 | 79 | 0 | 0 |

Every question is within the bar, no decision or span set changes and every
request's input tokens agree. vela3-turbo writes no label definitions, so its
open-label span questions are identical too. The runtime runs a gated model
eager on GPUs, where the package's runtime takes its own GPU kernels; the
gated rows' largest difference, 0.018, is the closest to the bar.

## Not measured

- vela3-lite, vela3-mini and vela3-flash on GPUs; vela3-turbo on CPU.
- Golden answers: none are recorded, since this host does not reproduce the
  generic `cpu` references of the d3 bases. Record them on the host class
  the d3 references come from.
- vela3-lite on the vela3-mini and vela3-flash host, and the base-alone
  control (the plain `decision3` family against the Decision 3.0 runtime) on
  that host, so the gated and ordinary differences there have no control of
  their own.
