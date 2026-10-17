# External NLI full-input contract and container CPU parity

On 2026-10-11, the pinned binary checkpoint
`MoritzLaurer/ModernBERT-base-zeroshot-v2.0` at
`d421c4545a438fd006fb43f8b981c5d908faa1e1` reproduced the missing full-input
proof on PR head `6b39a2dd92ff25ae001c5071c6a7923f6b6d28aa`.
The [raw record](nli-full-input-parity.json) contains the before and after
requests' answers, reference scores, execution environments and image identity.

## Reproduction and fix

The real tiny-ModernBERT runtime process regression first failed over TCP:

```text
POST /v1/decisions, require_full_input=true
HTTP 200, valid Choice and Noul scores
input_coverage missing
AssertionError: None == 'complete'
```

The real checkpoint reproduced the same omission in all six parity cases on
the untouched PR head. Numerical parity passed at tolerance `1e-5`; the
strict-input check failed because the successful answers omitted the proof.

The adapter now retains the validated requirement after every whole
premise/hypothesis pair has passed reject-overflow preparation. Successful
strict readouts emit `input_coverage: "complete"`; invalid questions,
oversized inputs, deadlines and invalid logits do not. Composed Noul items
retain their individual requirements. Ordinary answers keep their existing
shape and scores. Decision serving is limited to binary NLI label sets;
ternary models retain classification until their normalization contract is defined.

## Text-only media refusal regression (2026-10-14)

With the NLI PR at `d7d3db0b1a3750333754b23bdae0167821f55243` merged into
main at `a61d0e9ab109abc6cf1ab3c2201f5dbced2fe8fb`, a tiny local ModernBERT
fixture reproduced the remaining coverage bug through the runtime HTTP app.
Both `images` and `videos` attachments were ignored by a strict Noul request:

```text
Before, for either attachment field:
POST /v1/decisions -> HTTP 200
answers.q: {type: noul, noul: 0.38494440430142096, input_coverage: complete}

After:
POST /v1/decisions -> HTTP 400
error: {code: invalid_request,
        message: "model nli reads text only; images are not supported"}
```

Video requests return the corresponding `videos are not supported` message.
The NLI decision adapter now validates media fields before preparing any
hypotheses: non-list fields are invalid, and nonempty lists are unsupported.
This is input validation, not an image/video decoder or a model numerical change.

The new HTTP suite covers both decision aliases, root and named-state inputs,
ordinary and strict questions, and bundle tasks containing either root or
named-state attachments. A failed bundle task carries status 400 and no
decision answers, while an independent valid text task still succeeds with
complete strict coverage. Direct-request tests also prove hypothesis preparation
is never reached for unsupported media.

The suite reproduced **32 failures / 36 passing controls** before the fix.
After the fix, the combined NLI, strict-input, media, shared full-input and API
contract suites passed **245 tests**. Omitted media, empty lists and null fields
retain identical text-only answers, usage and strict coverage. The guide now
states the binary-only label contract and the text-only media restriction.

The direct-worker example now uses `vllm-srun serve`, which accepts the documented
`--device` and `--port` flags. All **54 documentation/CLI contract tests** passed,
as did type checking for **114 source files**, generated-client freshness,
formatting, lint, security, structure and architecture checks.

The broader CPU runtime run passed **1,183 tests**, skipped 31 and deselected 31.
Three numerical parity failures remained: Qwen's bundled reference,
Omni packed-text invariance and Qwen3.5's Transformers bit identity. All three
also failed on the untouched merged baseline. Those implementations were not
changed by this fix.

These new results use tiny fixtures through the HTTP app. They do not represent
a new full-checkpoint container or GPU parity run; the earlier checkpoint
qualification below remains a separate record.

```bash
HF_HUB_OFFLINE=1 python -m pytest -q \
  src/model-runtime/tests/test_nli_media.py \
  src/model-runtime/tests/test_nli.py \
  src/model-runtime/tests/test_nli_full_input.py \
  src/model-runtime/tests/test_full_input_contract.py \
  src/model-runtime/tests/test_api_contract.py
```

## Container result

The local router image was built with `make vllm-sr-build` and launched through
`vllm-sr serve --engine --platform cpu --minimal --image-pull-policy never`.
The configured deployment used an explicit local snapshot of the pinned
checkpoint, mounted under `/app/models/nli`, with a loopback listener.
The managed runtime became ready, and the public System One listener answered
both `/v1/decisions` and `/v1/systemone`.

Environment: Linux arm64 CPU, Python 3.12.15, PyTorch 2.10.0+cpu, FP32
`exact`; the host Transformers 5.19.0 reference used eager attention.

```json
{
  "passed": true,
  "cases": 6,
  "maximum_absolute_difference": 9.751466235607964e-7,
  "maximum_pipeline_difference": 7.963326892346245e-7
}
```

Every strict Choice and Noul answer included complete coverage and matched
its ordinary counterpart exactly. Candidate reordering and the System One
alias preserved answers. This container run checks the public decision
surfaces; classification parity is recorded separately in
[the original record](nli-modernbert-parity.md). The external checkpoint's
built-in golden status remains `unverified`.

## Repeat the checks

Use the [README's local image and engine configuration](../../README.md#external-nli-models).
Supply the explicit local pinned snapshot as `NLI_PACKAGE`, then run:

```bash
python -m pip install -e 'src/model-runtime[test,reference]'
python src/model-runtime/tools/nli_parity.py \
  --package "$NLI_PACKAGE" --url http://127.0.0.1:8899 \
  --systemone --require-full-input \
  --repo MoritzLaurer/ModernBERT-base-zeroshot-v2.0 \
  --revision d421c4545a438fd006fb43f8b981c5d908faa1e1 \
  --output /tmp/nli-full-input-parity.json

HF_HUB_OFFLINE=1 python -m pytest -q \
  src/model-runtime/tests/test_nli.py \
  src/model-runtime/tests/test_nli_full_input.py \
  src/model-runtime/tests/test_full_input_contract.py

go -C src/semantic-router test ./pkg/modelservice/... -run TestFullInput -count=1
make model-runtime-client-check
```

The focused Python suite passed 122 tests; router full-input transport and
composed-task tests, type checking, and generated-client freshness passed.
The broader CPU runtime suite passed 921 tests, skipped 17 and deselected 19
GPU cases, with three existing numerical parity failures. Those same failures
reproduced on the untouched PR head: Qwen bundled numerics, Qwen3.5 reference
bit identity and Omni packed-text invariance. No related implementations were
changed. A macOS Unix-socket path-length failure passed with a shorter pytest
temporary directory.
