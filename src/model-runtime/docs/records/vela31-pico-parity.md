# vela3_pico family parity with the Vela 3.1 pico packages

The vela3_pico family answers `/v1/decisions` the way a pico package's own
runtime (`vela3_encoder.Vela3Encoder.system_one`) does, without importing it.
This record compares the two on CPU in FP32, and on one GPU.

- **Date:** 2026-11-12.
- **Packages:** `vllm-sr/vela3.1-pico-380m` at `a79cbf66` and
  `vllm-sr/vela3.1-pico-150m` at `dad77ad7` (the `v1.0` tags; every loaded
  file checked against the registry pins).
- **Machine:** AMD EPYC 9575F (x86_64), PyTorch 2.14.1+cpu; the package side
  ran Transformers 5.17.0, the release's pin. GPU: one AMD Instinct MI325X
  (gfx942), PyTorch 2.14.1+rocm7.14.
- **Tool:** `tools/vela3_pico_parity.py`: `--generate 64 --seed 2` (every path
  the family plans: the one pass with the core questions, several span groups
  and typed parts; decisions alone; the question-by-question path forced by a
  long part, a Score question or a Set question that is not a router question;
  names-only and partly described router options, Nouls with descriptions,
  JSON instructions; every span route and request thresholds), `--long
  --generate 12 --seed 1` (one sequence within 64 tokens of the 8,192-token
  window, on either side), and `--requests vela3-parity-profile.jsonl` (the
  requests the Router sends for the `model-runtime-vela3` E2E profile).

## Results (CPU, FP32)

| Model | Run | Requests (questions) | One pass / question by question | Answers identical | max \|dp\| | Span sets identical |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| pico-380m | generated | 64 (320) | 24 / 40 | 64 | 0 | 89 / 89 |
| pico-380m | 8K boundary | 12 (48) | 1 / 11 | 12 | 0 | 24 / 24 |
| pico-380m | Router profile | 5 (20) | 5 / 0 | 5 | 0 | 10 / 10 |
| pico-150m | generated | 64 (320) | 24 / 40 | 64 | 0 | 89 / 89 |
| pico-150m | 8K boundary | 12 (48) | 1 / 11 | 12 | 0 | 24 / 24 |
| pico-150m | Router profile | 5 (20) | 5 / 0 | 5 | 0 | 10 / 10 |

Every answer both sides give is identical: the same Choice, Noul, Set
selections, Span sets of (label, start, end), the same applied thresholds and,
where neither side fails a question, the same input-token counts; every
probability is equal (max |dp| 0). On every request the family read in one
pass exactly when the package did (`passes`). The boundary set reads one
request in one pass at 8,151 tokens and eleven that land just past the window
on the question-by-question path, as the package does. The 150M model has no
test fixture at its shape (hidden 768, intermediate 1,024, six heads), so this
run is the check that the native UltraBERT reproduces it: it does, bit for bit.

The golden request (decisions, router Choice and Set questions, both span
presets and an entity span, read in one pass) is recorded on CPU in
`registry/golden_answers_vela3_pico.json`; a second load of each model matches
all 41 recorded answers.

## Questions only one side answers, by design

- **A row that does not fit.** Nine questions per model (requests `p2-0006`,
  `-0022`, `-0030`, `-0038`) sit beside a router row that cannot fit one
  8,192-token input: a long context part that row neither reads nor may
  shrink first. The package runs all of a request's router questions in one
  `try`, so it fails all of them; the family fails only the row that does not
  fit (`max_length_exceeded`), as the System One contract requires and as the
  vela3 family does. The tool asks the package the same request again without
  the questions the family failed; every answer of the rows that fit is then
  identical. Asking such a question alone is not the same call (its row would
  hold no other router question), which the first version of the tool did.
- **Both sides fail.** Score questions and Set questions that are not router
  questions are invalid on both sides (eight per model); eleven questions on
  rows that do not fit fail on both sides (the package says
  `invalid_question`, the family `max_length_exceeded`).
- **A Set question's own threshold** (two per model): the package applies 0.5
  and ignores it; the family applies it, as System One does. The tool selects
  from the package's probabilities at the family's threshold.

## GPU (information)

`--device rocm:0`, 16 generated requests each (seed 5), GPU bar 0.02: no
decision or span set changed on either model, max |dp| 0.0039 (380m) and
0.0076 (150m). The package computes its decisions under BF16 autocast on a
GPU and the family in FP32, so GPU answers are not bit-identical; golden
answers are recorded for CPU only.
