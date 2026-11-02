# Vela 3.0 d3-size parity (vela3-lite)

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

## Not measured

- vela3-mini, vela3-flash and vela3-turbo.
- Golden answers: none are recorded, since this host does not reproduce the
  generic `cpu` references of the d3 bases. Record them on the host class
  the d3 references come from.
- Open-label definitions written by the base (vela3-mini's `base` writer and
  vela3-flash's `grafted_head`): vela3-lite and vela3-turbo run names only.
