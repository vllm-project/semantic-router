# Vela Halu preparation and parity

The model runtime serves the pinned native Halu release with the `task_heads`
family's grounded head. Its registry pins the Hugging Face revision and the
digest of every loaded file, and its golden answers fix the head's outputs;
`source.json` here records the same revision and source-file digests. The head
reads the published request/context and answer pair and applies
`hallucinated > 0.5` from `operating_point.json` only to answer tokens. The task
budget is 8192 tokens; the encoder's 32768-token architecture does not increase
it. The runtime reports spans in code points; the router converts them to UTF-8
byte offsets once.

To serve Halu on the runtime's `onnxruntime` engine, install `requirements.txt`
in a separate build environment and prepare a derived package from that
immutable local snapshot:

```sh
python tools/models/vela_halu/export.py --model /models/halu-native \
  --output /models/halu-onnx --device cpu --dtype float32
```

The exporter verifies source hashes before loading weights. It uses the shared
classifier exporter in `tools/models/onnx` for dynamic ONNX numerical parity,
copies the published task policy, and writes `halu_reference.json` from PyTorch
pair inference, the reference for comparing the exported graph's spans with the
native head. The source and output directories must differ. This is an offline
preparation step. GPU deployments must declare `input.max_tokens` (at most the
8192-token task limit), which fixes the execution shape; CPU export parity alone
does not qualify a GPU execution provider.

`make verify PROFILE=vela-halu` exercises supported, contradictory, and Unicode
answers plus input rejection through a deployed router's hallucination plugin
probe API. The plugin's optional span filters remain separate from the published
token operating point; the defaults retain single-token spans.
