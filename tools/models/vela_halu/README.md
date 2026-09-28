# Vela Halu preparation and parity

The default detector loads the pinned native Halu release with Candle on CPU.
`source.json` records the exact Hugging Face revision and source-file digests.
The explicit `vela_halu` adapter validates `operating_point.json`, tokenizes the
published request/context and answer pair, and applies `hallucinated > 0.5` only
to answer tokens. The task budget is 8192 tokens; the encoder's 32768-token
architecture does not increase it. Internal span offsets are UTF-8 bytes.

For an explicit ORT deployment, install `requirements.txt` in a separate build
environment and prepare a derived artifact from that immutable local snapshot:

```sh
python tools/models/vela_halu/export.py --model /models/halu-native \
  --output /models/halu-ort --device cpu --dtype float32
```

The exporter verifies source hashes before loading weights. It uses the shared
classifier exporter for dynamic ONNX numerical parity, copies the published task
policy, and writes `halu_reference.json` from PyTorch pair inference. The source
and output directories must differ. This is an offline preparation step.

Test the actual router binding against those reference spans:

```sh
VLLM_SR_REQUIRE_HALU_TESTS=1 \
VLLM_SR_HALU_MODEL=/models/halu-ort \
VLLM_SR_HALU_REFERENCE=/models/halu-ort/halu_reference.json \
VLLM_SR_MODEL_TEST_PROVIDER=ort VLLM_SR_MODEL_TEST_DEVICE=cpu \
go -C src/semantic-router test ./pkg/modelruntime/native \
  -run '^TestPublishedVelaHalu$' -count=1 -v
```

Use the native snapshot and `provider=candle` for Candle parity. An AMD test must
explicitly select its ORT device and use the prepared graph. GPU deployments
must declare `input.max_tokens` (at most the 8192-token task limit), which fixes
the execution shape and participates in the instance identity. Shorter pairs
are masked and padded; any CPU compute fallback fails model preparation. CPU
export parity alone does not qualify that execution provider. `make verify PROFILE=vela-halu`
exercises supported, contradictory, and Unicode answers plus input rejection
through a deployed router's hallucination plugin probe API.

The lower-level ORT instance reference check is also explicit:

```sh
VELA_HALU_ARTIFACT=/models/halu-ort \
go -C onnx-binding test ./instance -run '^TestPublishedGroundedParity$' -count=1 -v
```

Both reference checks require prepared model inputs and are excluded from the
model-free Core inventory. Routine native CI does not provision their Halu
reference artifacts; the `vela-halu` E2E profile owns deployed behavior coverage.
Set `ORT_DYLIB_PATH` to the installed ONNX Runtime library for ORT tests.

The plugin's optional span filters and NLI policy remain separate from the
published token operating point. The defaults retain single-token spans and
leave NLI filtering disabled. Explicit legacy detector configurations keep their
original model and adapter semantics.
