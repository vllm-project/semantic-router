# External ModernBERT NLI parity

On 8 October 2026, the native model runtime loaded the independently published
Apache-2.0 `MoritzLaurer/ModernBERT-base-zeroshot-v2.0` checkpoint and served
both `/v1/classify` pairs and `/v1/decisions` Choice/Noul requests using the
native encoder engine on an Apple arm64 CPU. No package code was executed.

- Revision: `d421c4545a438fd006fb43f8b981c5d908faa1e1`.
- Model identity: `a3858ccb4f576b734e2f304b541fbf5a2d445155eab77de69933ecd67df9ded3`.
- 149,606,402 parameters, FP32, `exact` profile.
- Python 3.14.5, PyTorch 2.14.1, Transformers 5.19.0.
- Reference: the local checkpoint loaded with Transformers, FP32, eager
  attention and reference compilation disabled.
- Six prompts, including negation, a mixed-domain request and an unrelated
  topic. All pair distributions, choice distributions and yes/no scores
  matched the reference within `1e-5`; maximum absolute difference:
  `3.039836883544922e-6`. Every selected choice matched.
- The standard Transformers `pipeline("zero-shot-classification")` was also
  run with the same checkpoint, candidate descriptions and hypothesis
  template. All six winners matched; maximum Choice/Noul score difference:
  `3.2032034132933873e-6`. The raw record includes both references.
- Reversing candidate order preserved each option's probability and the
  winner. Repeating the request through `/v1/systemone` preserved answers.
- Raw record: [nli-modernbert-parity.json](nli-modernbert-parity.json).

The three basic prompts selected code, mathematics and travel respectively.
The unrelated photosynthesis prompt selected mathematics because only
three choices were supplied. This model does not add an abstain option or
recognize a catch-all automatically. The mathematics prompt also gave the
independent programming hypothesis a score of about 0.564: binary hypothesis
scores and relative choice scores answer different questions and require
workload evaluation before setting decision thresholds.

This is a small parity and viability panel, not a routing quality benchmark.
It does not certify calibrated confidence, GPU support, or byte-identical
cross-library results. With no built-in golden answer record, the model's
readiness card continues to report `unverified`; readiness repeats NLI pair
and decision requests to check stable execution.

## Before and after

The same checkpoint and request were run against unmodified `main` at
`56107fb841a2c21128a169090f99b58669cb032d` and this change:

```text
Before: POST /v1/classify  -> 200, results[0].error = invalid_input (text pair)
Before: POST /v1/decisions -> 422, error.code = unsupported_surface
After:  POST /v1/classify  -> 200, label = entailment, P(entailment) = 0.9983238578
After:  POST /v1/decisions -> 200, choice = code, P(code) = 0.9990273344
                                noul = 0.9983238342
```

The prompt was `Write a Python function to sort a list.`; the candidates
were programming, mathematics and travel planning, with the template
`This request is about {label}.`.

The verified model is binary. Decision serving accepts only binary NLI label
sets in this slice. Ternary checkpoints remain available for classification;
their decision normalization contract is deferred to a follow-up. This record
establishes pipeline parity for the pinned binary checkpoint.

## Reproduce

Install the runtime with its `reference` extra from this checkout and supply
a local snapshot of the pinned checkpoint. The CPU parity run used the runtime
executable directly:

```bash
vllm-srun serve "$NLI_PACKAGE" --device cpu --host 127.0.0.1 --port 8100
```

Then compare against Transformers in another terminal:

```bash
python src/model-runtime/tools/nli_parity.py \
  --package "$NLI_PACKAGE" --url http://127.0.0.1:8100 \
  --repo MoritzLaurer/ModernBERT-base-zeroshot-v2.0 \
  --revision d421c4545a438fd006fb43f8b981c5d908faa1e1 \
  --output /tmp/nli-modernbert-parity.json
```

The tool requires explicit local weights and never downloads models. Unit
and runtime process integration tests use tiny random-weight NLI fixtures instead.

For the current public `vllm-sr serve` container path, use the
[README's local image and engine configuration](../../README.md#external-nli-models).
That example builds the local image and writes `nli-config.yaml`, including the
pinned checkpoint, CPU placement and a loopback listener on port 8899. Once the
image and configuration are ready, launch it with:

```bash
VLLM_SR_IMAGE=semantic-router-nli:local vllm-sr serve --engine \
  --config nli-config.yaml --platform cpu --minimal --image-pull-policy never
```

Run the parity tool against `http://127.0.0.1:8899` with `--systemone` for
that public listener; it exposes decision surfaces, so classification parity
continues to use the direct runtime above. See the
[full-input container record](nli-full-input-parity.md#repeat-the-checks) for
the complete public-listener parity command and the later CPU container result.

Docker or Podman is required for the public CLI path. It was not exercised on
the original 8 October CPU parity host, which had neither runtime. GPU
execution was also not tested in that run. The historical records retain the
`exact` profile used at the time; current configuration uses engine arguments
and no longer accepts `profile`.
