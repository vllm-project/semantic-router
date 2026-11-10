# Vela 3.0 d3-size parity

The `vela3_d3` family against vela3-lite's own runtime (`modeling_vela3.py`,
`Vela3MiniModel.system_one`) on CPU. Every span question and every question
with a gated adapter answers as the package does. Questions that run with
every adapter off differ from the package by up to 0.023 on this host, the
same way the plain `decision3` family differs from the Decision 3.0 runtime
on it. No decision and no span set changed.

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

## vela3-mini

`vllm-sr/vela3-mini` `bbc0206e` over `vllm-sr/d3-mini` `566da0ea`, the same
host and environments, `generate --count 40 --seed 0` (no `hazard:` Nouls:
vela3-mini ships no hazard adapter; opt-in questions run its opt-in adapter).
The package's runtime took 2,239 s, the runtime 794 s.

| Questions | Count | Identical values | Changed | Largest difference |
| --- | --- | --- | --- | --- |
| Gated (router, task, opt-in adapters) | 106 | 66 | 0 | 0.0075 |
| Adapters off | 105 | 61 | 0 | 0.0044 |
| Span, labels with descriptions or a label set | 41 | 41 | 0 | 0 |
| Span, open labels without a description | 38 | 26 | 20 | 0.156 |

Every change is a span question whose open labels have no description:
vela3-mini's package writes a definition for each with its base
(`definitions.writer` `base`), which changes the option lines, the NER
threshold (with definitions, not names only) and so the spans. The runtime
runs these labels by name until the definition writers land. Every other
span question is identical, and no decision changes. Unlike vela3-lite's,
the gated rows here differ more than the adapters-off rows (0.0075 against
0.0044), and the base-alone control ran only on vela3-lite's requests, so
these gated differences are within the bar but not shown to be this host's
alone.

## vela3-flash

`vllm-sr/vela3-flash` `af48828d` over `vllm-sr/d3-flash` `8b35c853`, the same
host, environments and request generator (290 questions). The package's
runtime took 2,279 s, the runtime 1,052 s.

| Questions | Count | Identical values | Changed | Largest difference |
| --- | --- | --- | --- | --- |
| Gated (router and task adapters) | 106 | 67 | 0 | 0.011 |
| Adapters off | 105 | 61 | 0 | 0.0043 |
| Span, labels with descriptions or a label set | 41 | 41 | 0 | 0 |
| Span, open labels without a description | 38 | 27 | 20 | 0.213 |

As on vela3-mini, every change is an open-label span question without a
description, which vela3-flash's package defines with its base and grafted
LM head (`definitions.writer` `grafted_head`; #5703). No decision changes.
As on vela3-mini, the gated rows differ more than the adapters-off rows
(0.011 against 0.0043) and have no control of their own.

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
- Open-label definitions written by the base (vela3-mini's `base` writer and
  vela3-flash's `grafted_head`): vela3-lite and vela3-turbo run names only.
