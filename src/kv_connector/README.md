# Cross-model KV connector

This is the out-of-tree vLLM connector for mapper artifacts from
`src/training/kv_mapper/`.

- `vllm_connector.py` exports source KV from a running vLLM producer, checks
  the target's exact token prefix in the scheduler, then maps and loads the KV
  into the consumer's paged cache before its forward pass.
- `transform.py` converts a source Qwen3 post-RoPE cache into target post-RoPE
  K/V tensors using the full-head weights. It supports unscaled Qwen3 RoPE and
  a complete prefix beginning at position zero.
- `snapshot.py` and `handoff.py` provide same-host snapshot storage and the
  mapped cache write. Snapshots are immutable, tenant-scoped, and have a TTL.
  Expired files are cleaned during reads and publishes. Reuse uses complete,
  block-aligned source prefixes.

The artifact is checked against the target model id, pinned revision, dtype,
TP degree, KV head count, head dimension, and layer count. The source id and
revision are pinned in connector configuration. Both source and target must
serve at TP=1 for this full-head artifact.

Both vLLM servers must have this repository root on `PYTHONPATH`, with `torch`,
`numpy`, and `safetensors` installed. Each KV transfer configuration names
`KVMapperConnector` with module path `src.kv_connector.vllm_connector`. The
producer uses `kv_role=kv_producer` and a trusted `snapshot_root`. The consumer
uses `kv_role=kv_consumer`, `kv_load_failure_policy=recompute`, the same
`snapshot_root`, and `artifact_path`, `source_model`, `source_revision`,
`source_tp`, `head_order`, and `source_rope_theta` in
`kv_connector_extra_config`. Both models use bf16 and TP=1 for the published
full-head artifact. The producer records its RoPE theta in each snapshot, and
the consumer checks it against `source_rope_theta`. Both models use unscaled
Qwen3 RoPE.

The consumer runs with `VLLM_USE_V2_MODEL_RUNNER=0` and
`--no-enable-prefix-caching`. Mapped KV is approximate and is used only for
the request carrying the approved transfer hint. Ordinary prefix caching is
disabled on the consumer. The connector permits mapped reuse only when the
consumer configuration explicitly sets `enable_prefix_caching=False`.
The producer can keep prefix caching enabled.

Both requests pass the same `namespace`, `cache_id`, and `mapper_id` in JSON
`kv_transfer_params`. The target prompt must extend the source's exact token
prefix. The two vLLM servers share the snapshot directory on one host.

Run the synthetic artifact and conversion checks from the repository root:

```bash
PYTHONPATH=. python3 -m unittest discover -s src/kv_connector/tests -p 'test_*.py'
```

With vLLM installed, run the connector integration checks manually. These
scripts exercise the scheduler and worker interfaces and are not included in
CI's `test_*.py` discovery:

```bash
PYTHONPATH=. python3 -m unittest src.kv_connector.tests.vllm_connector_integration src.kv_connector.tests.vllm_live_connector_integration
```

On a two-GPU CUDA host, exercise the published mapper geometry and cache write
path with an explicit artifact directory:

```bash
PYTHONPATH=. python3 -m src.kv_connector.gpu_probe --artifact /path/to/mapper-artifact
```

The probe publishes synthetic source KV from GPU 0, loads it through the local
snapshot store, maps all target layers onto GPU 1, and compares the first
target layer against a CPU reference.
