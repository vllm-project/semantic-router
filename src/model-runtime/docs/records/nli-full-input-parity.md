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
