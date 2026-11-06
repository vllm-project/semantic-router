# vela3 family parity with the vela3-fast package

The vela3 family answers `/v1/decisions` the way vela3-fast's own runtime
(`vela3_encoder.Vela3Encoder.system_one`) does, without importing it. This
record compares the two on CPU.

- **Date:** 2026-10-27.
- **Package:** `vllm-sr/vela3-fast` at `7cbefd4b`.
- **Machine:** x86_64 CPU (AVX2), PyTorch 2.10.0; the package side ran
  Transformers 5.17.0, the release's pin.
- **Tool:** `tools/vela3_parity.py --generate N --seed S`: synthetic requests
  over every route the family takes (decisions on the backbone, the shared
  router adapter and the domain and fact-check adapters, Choice and Set, every
  span route), with CJK text, URLs, thresholds, JSON states and parts long
  enough to be read in windows (up to 80,596 input tokens).

## Results

| Run | Requests | Answers identical | max \|dp\| |
| --- | ---: | ---: | ---: |
| seed 1 | 8 | 8 | 0 |
| seed 2 | 60 (268 questions) | 57 (three documented differences below) | 0 |

Every answer both sides give is identical: the same Choice, Set selections and
Span sets of (label, start, end), the same applied thresholds and input-token
counts, and every probability equal (max |dp| 0, the bar is 1e-4). The
backbone and each adapter network were checked separately on padded batches of
mixed lengths: their hidden states are bit-identical to the package's
`DecisionEncoder` and `build_adapter` networks.

## The E2E profile's requests

`vela3-parity-profile.jsonl` holds the requests the Router sends for the five
prompts of the `model-runtime-vela3` E2E profile, captured from the Router's
own signal evaluation of that profile's config: one call per prompt with the
prompt guard Choice, the `pii` preset, the `account_access` Noul and the
`order_numbers` span question. Run with `--requests`, all 5 requests and 20
questions are identical to the package (max |dp| 0).

The golden request (decisions, router questions on the shared and task
adapters, both span presets and an entity span) is recorded on CPU in
`registry/golden_answers_vela3.json`; a second load matches all eight answers.

## A row that does not fit

Request `g2-0017` asks two router questions about a state whose context part
is about 6,500 tokens: fact check (over the user part) and refusal (over the
answer part). The refusal row cannot fit one 8,192-token input, because the
context part is neither read by it nor the part the layout shrinks first.

- **The package** runs all of a request's router questions in one `try`, so
  that one row failing (`schema alone does not fit max_length`) fails every
  router question of the request, fact check included.
- **The family** fails only the question whose row does not fit
  (`max_length_exceeded`) and answers fact check, as the System One contract
  requires: a question that cannot be answered fails alone.

The family keeps the contract. The answers it gives where the package gives
none are the ones the package gives when the failing question is asked apart.
`g2-0029` and `g2-0035` fail the same way (1,245 and 2,457 tokens over): asked
apart, the package's answers equal the family's, apart from `g2-0035`'s Set
selection below.

## A Set question's own threshold

The package's router path selects a Set question's labels at its own
`router.set_threshold` (0.5) and ignores the question's `threshold`; the family
applies the question's threshold when it has one, as System One does, and the
package's otherwise. The probabilities are the same either way. The tool
therefore checks a Set answer by selecting from the package's probabilities at
the threshold the family applied, and flags the row (`own_threshold`).
`g2-0035` asks a hazard-category Set at 0.288: asked apart, both sides give
0.365 for one category, which the family selects and the package, at 0.5, does
not.
