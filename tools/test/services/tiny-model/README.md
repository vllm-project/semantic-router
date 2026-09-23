# Real tiny-model smoke

Use this optional runner when a test needs actual token generation. The normal
provider protocol, memory and fault tests use `provider-mocker` and do not download
weights. This runner uses one model: `Qwen/Qwen3-0.6B`, with thinking disabled.

```bash
make tiny-model-smoke       # starts the server, checks it, then removes it
make tiny-model-serve       # foreground server on 127.0.0.1:8000
```

The smoke checks health, generated text, complete SSE termination, and actual
stop-sequence truncation. It first generates a deterministic baseline, chooses a
substring from that response, then confirms an otherwise identical request stops
before that substring. It does not grade model intelligence or replace Router
integration tests.

`model.json` pins the Qwen official Q8_0 GGUF repository revision and SHA256, and
the upstream llama.cpp server image by multi-platform digest. No custom inference
image is built or published. The model download (639,446,688 bytes) goes into the
ignored `.cache/tiny-model` directory and is checksum-verified on every run.

The server is bounded to two CPU cores, 2 GiB RAM, one request slot, a 2,048-token
context and 64 generated tokens. Only localhost is published. The model volume
and container filesystem are read-only. Configure ports or a shared cache with:

```bash
python3 tools/test/services/tiny-model/run.py smoke --port 0 --cache-dir /tmp/sr-model-cache
python3 tools/test/services/tiny-model/run.py serve --port 8001
```

Point a Router Chat Completions backend at this server to exercise real inference
through any supported text ingress codec. The native protocol matrix and error
injection remain the responsibility of the deterministic provider mocker.

To update pins, inspect the [Qwen metadata API](https://huggingface.co/api/models/Qwen/Qwen3-0.6B-GGUF?blobs=true)
and the [llama.cpp registry](https://github.com/ggml-org/llama.cpp/pkgs/container/llama.cpp),
update the single manifest, and rerun the real smoke on the required CPU platform.
The checked-in server digest is version `b11058`, revision
`f072b103714dfa1eee531f80b24512faf38e3dd2`; the registry manifest was verified on
2026-09-21. See [upstream Docker usage](https://github.com/ggml-org/llama.cpp/blob/master/docs/docker.md).
