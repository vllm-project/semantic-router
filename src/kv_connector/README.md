# Cross-model KV connector

This is the out-of-tree vLLM connector for mapper artifacts from
`src/training/kv_mapper/`.

- `vllm_connector.py` exports source KV from a running vLLM producer, checks
  the target's exact token prefix in the scheduler, then maps and loads the KV
  into the consumer's paged cache before its forward pass.
- `transform.py` converts a source Qwen3 post-RoPE cache into target post-RoPE
  K/V tensors using the full-head weights. It supports unscaled Qwen3 RoPE and
  a complete prefix beginning at position zero.
- `snapshot.py` and `handoff.py` provide the same-host transport and validated
  map/inject operation. Snapshots are immutable, tenant-scoped, and expire.
  Expired files are removed on read or before the next publish.
  Only complete, block-aligned source prefixes are eligible for reuse.

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
full-head artifact.
The producer records its actual RoPE theta in each snapshot. The consumer
checks that value against `source_rope_theta` and rejects scaled RoPE.

Set `VLLM_USE_V2_MODEL_RUNNER=0` on the consumer. Live vLLM 0.30.0 testing found
that its V2 runner could return an incorrect token after a synchronous KV load
failure despite logging a reschedule. The connector disables cache reuse when
V2 is selected. With V1, a failed load returned the same token as a cold
request in the live probe.

Both requests pass the same `namespace`, `cache_id`, and `mapper_id` in JSON
`kv_transfer_params`. The target prompt must extend the source's exact token
prefix. Missing, expired, or mismatched snapshots use normal prefill. The
router's `x-vsr-kv-*` headers still need a trusted request adapter. Remote
source-pod transport, response status reporting, and a full routed end-to-end
test remain. The shared directory is a same-host test transport.

Run the synthetic artifact and conversion checks from the repository root:

```bash
PYTHONPATH=. python3 -m unittest discover -s src/kv_connector/tests -p 'test_*.py'
```

On a two-GPU CUDA host, exercise the published mapper geometry and cache write
path with an explicit artifact directory:

```bash
PYTHONPATH=. python3 -m src.kv_connector.gpu_probe --artifact /path/to/mapper-artifact
```

The probe publishes synthetic source KV from GPU 0, loads it through the local
snapshot store, maps all target layers onto GPU 1, and compares the first
target layer against a CPU reference. It does not start vLLM or measure TTFT.
