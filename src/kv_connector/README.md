# Cross-model KV connector

This is the out-of-tree vLLM connector for mapper artifacts from
`src/training/kv_mapper/`. It currently has two parts:

- `vllm_connector.py` loads and checks the artifact against the running target
  model. It always returns zero external cache hits, so every request uses a
  normal prefill while the receive path is being built.
- `transform.py` converts a source Qwen3 post-RoPE cache into target post-RoPE
  K/V tensors using the full-head weights. It supports unscaled Qwen3 RoPE and
  a complete prefix beginning at position zero.

The artifact is checked against the target model id, pinned revision, dtype,
TP degree, KV head count, head dimension, and layer count. The source id and
revision are pinned in connector configuration. Both source and target must
serve at TP=1 for this full-head artifact.

The vLLM server must have this repository root on `PYTHONPATH`, with `torch`,
`numpy`, and `safetensors` installed. Its KV transfer configuration names
`KVMapperConnector` with module path `src.kv_connector.vllm_connector` and
supplies `artifact_path`, `source_model`, `source_revision`, `source_tp`, and
`head_order` in `kv_connector_extra_config`.

vLLM passes request-specific transfer hints in JSON `kv_transfer_params`.
The router's `x-vsr-kv-*` headers therefore need a trusted request adapter.
The source vLLM pod also needs a way to export a complete cache before the
target can claim a hit. Those paths and the paged-cache write are the remaining
C2 integration work; the mapping function alone does not skip prefill.

Run the synthetic artifact and conversion checks from the repository root:

```bash
PYTHONPATH=. python3 -m unittest discover -s src/kv_connector/tests -p 'test_*.py'
```

With vLLM installed, run the connector integration check separately:

```bash
PYTHONPATH=. python3 -m unittest src.kv_connector.tests.vllm_connector_integration
```
