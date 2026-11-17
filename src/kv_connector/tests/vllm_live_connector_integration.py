"""Exercise vLLM scheduler and worker hooks with a small CPU cache."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from src.kv_connector.paged_cache import extract_prefix, inject_prefix
from src.kv_connector.transform import qwen3_rope
from src.training.kv_mapper.artifact import CompatibilitySpec, Manifest, write_artifact

if importlib.util.find_spec("vllm"):
    from vllm.config.kv_transfer import KVTransferConfig
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

    from src.kv_connector.vllm_connector import (
        KVMapperConnector,
        _model_name,
        _rope_theta,
    )


@unittest.skipUnless(importlib.util.find_spec("vllm"), "vLLM is not installed")
class LiveConnectorTests(unittest.TestCase):
    def test_offline_snapshot_model_identity(self) -> None:
        revision = "a" * 40
        snapshot = f"/cache/models--Qwen--Qwen3-32B/snapshots/{revision}"
        self.assertEqual(
            _model_name(SimpleNamespace(model=snapshot, revision=revision)),
            "Qwen/Qwen3-32B",
        )
        self.assertEqual(
            _model_name(SimpleNamespace(model=snapshot, revision="b" * 40)),
            snapshot,
        )

    def test_source_export_target_load_and_failed_load_recompute_signal(self) -> None:

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            artifact_path = root / "artifact"
            snapshots = root / "snapshots"
            source_rev, target_rev = "a" * 40, "b" * 40
            compat = CompatibilitySpec(
                source_model="source",
                source_revision=source_rev,
                target_model="target",
                target_revision=target_rev,
                variant="full_head",
                precision="bf16",
                source_tp=1,
                target_tp=1,
                head_order="contiguous",
                num_kv_heads=1,
                head_dim=2,
            )
            manifest = Manifest(
                mapper_id="tiny-mapper",
                compatibility=compat,
                topk=1,
                ridge_alpha=1,
                centered_inputs=True,
                rope_stripped_on_keys=True,
                source_layers_per_target={"k": {"0": [0]}, "v": {"0": [0]}},
            )
            tensors = {
                f"target.0.{channel}.W": np.eye(2, dtype=np.float32)
                for channel in ("k", "v")
            }
            tensors.update(
                {
                    f"target.0.{channel}.b": np.zeros(2, dtype=np.float32)
                    for channel in ("k", "v")
                }
            )
            write_artifact(artifact_path, manifest, tensors)

            def config(model_name, revision, role, extra):
                return SimpleNamespace(
                    kv_transfer_config=KVTransferConfig(
                        kv_connector="KVMapperConnector",
                        kv_role=role,
                        kv_load_failure_policy="recompute",
                        kv_connector_extra_config=extra,
                    ),
                    model_config=SimpleNamespace(
                        model=model_name,
                        revision=revision,
                        dtype=torch.bfloat16,
                        hf_config=SimpleNamespace(
                            num_key_value_heads=1,
                            head_dim=2,
                            num_hidden_layers=1,
                            rope_parameters={"rope_theta": 1_000_000},
                        ),
                    ),
                    parallel_config=SimpleNamespace(tensor_parallel_size=1),
                    cache_config=SimpleNamespace(
                        block_size=2, enable_prefix_caching=False
                    ),
                )

            hint = {
                "namespace": "tenant-a",
                "cache_id": "session-a",
                "mapper_id": manifest.mapper_id,
            }
            source_config = config(
                "source", source_rev, "kv_producer", {"snapshot_root": str(snapshots)}
            )
            source_config.cache_config.enable_prefix_caching = True
            source = KVMapperConnector(source_config, KVConnectorRole.SCHEDULER, None)
            source_worker = KVMapperConnector(
                source_config, KVConnectorRole.WORKER, None
            )
            source_new = SimpleNamespace(
                req_id="source-request",
                prompt_token_ids=[1, 2, 3, 4],
                block_ids=([2, 0],),
                num_computed_tokens=0,
                mm_features=[],
                lora_request=None,
                sampling_params=SimpleNamespace(
                    extra_args={"kv_transfer_params": hint}
                ),
            )
            source_output = SimpleNamespace(
                scheduled_new_reqs=[source_new],
                num_scheduled_tokens={"source-request": 4},
            )
            source_meta = source.build_connector_meta(source_output)
            self.assertEqual(len(source_meta.saves), 1)
            source_worker.bind_connector_metadata(source_meta)
            keys = qwen3_rope(
                torch.arange(8, dtype=torch.bfloat16).reshape(4, 1, 2) / 10,
                torch.arange(4),
                theta=1_000_000,
            )
            values = torch.arange(8, dtype=torch.bfloat16).reshape(4, 1, 2) / 10
            source_cache = torch.zeros((3, 1, 2, 4), dtype=torch.bfloat16)
            inject_prefix(source_cache, [2, 0], keys, values, layout="lbnhc")
            source_worker.save_kv_layer(
                "model.layers.0.self_attn.attn", source_cache, None
            )
            source_worker.wait_for_save()
            source_worker.clear_connector_metadata()

            target_config = config(
                "target",
                target_rev,
                "kv_consumer",
                {
                    "snapshot_root": str(snapshots),
                    "artifact_path": str(artifact_path),
                    "source_model": "source",
                    "source_revision": source_rev,
                    "source_tp": 1,
                    "head_order": "contiguous",
                    "source_rope_theta": 1_000_000,
                },
            )
            target = KVMapperConnector(target_config, KVConnectorRole.SCHEDULER, None)
            request = SimpleNamespace(
                request_id="target-request",
                kv_transfer_params=hint,
                prompt_token_ids=[1, 2, 3, 4, 5],
                mm_features=[],
                num_preemptions=0,
                lora_request=None,
            )
            self.assertEqual(target.get_num_new_matched_tokens(request, 0), (4, False))

            for cache_config in (
                SimpleNamespace(block_size=2, enable_prefix_caching=True),
                SimpleNamespace(block_size=2),
                None,
            ):
                with self.subTest(cache_config=cache_config):
                    unsafe_config = SimpleNamespace(
                        **{**target_config.__dict__, "cache_config": cache_config}
                    )
                    for role in (KVConnectorRole.SCHEDULER, KVConnectorRole.WORKER):
                        unsafe = KVMapperConnector(unsafe_config, role, None)
                        self.assertIsNone(unsafe.artifact)
                        self.assertEqual(
                            unsafe.get_num_new_matched_tokens(request, 0), (0, False)
                        )
                        unsafe.update_state_after_alloc(request, None, 0)
                        self.assertEqual(
                            unsafe.build_connector_meta(
                                SimpleNamespace(scheduled_new_reqs=[])
                            ).loads,
                            [],
                        )
            wrong_theta_config = config(
                "target",
                target_rev,
                "kv_consumer",
                {
                    **target_config.kv_transfer_config.kv_connector_extra_config,
                    "source_rope_theta": 2_000_000,
                },
            )
            wrong_theta = KVMapperConnector(
                wrong_theta_config, KVConnectorRole.SCHEDULER, None
            )
            self.assertEqual(
                wrong_theta.get_num_new_matched_tokens(request, 0), (0, False)
            )
            target.update_state_after_alloc(request, None, 4)
            target_new = SimpleNamespace(req_id="target-request", block_ids=([1, 2],))
            target_output = SimpleNamespace(scheduled_new_reqs=[target_new])
            target_meta = target.build_connector_meta(target_output)
            self.assertEqual(len(target_meta.loads), 1)
            target_worker = KVMapperConnector(
                target_config, KVConnectorRole.WORKER, None
            )
            target_cache = torch.zeros((3, 1, 2, 4), dtype=torch.bfloat16)
            target_worker.register_kv_caches(
                {"model.layers.0.self_attn.attn": target_cache}
            )
            target_worker.bind_connector_metadata(target_meta)
            target_worker.start_load_kv(None)
            self.assertEqual(target_worker.get_block_ids_with_load_errors(), set())
            actual_key, actual_value = extract_prefix(
                target_cache, [1, 2], 4, heads=1, head_dim=2, layout="lbnhc"
            )
            torch.testing.assert_close(actual_key, keys, atol=0.01, rtol=0.01)
            torch.testing.assert_close(actual_value, values)
            target_worker.clear_connector_metadata()

            snapshot_file = next(snapshots.glob("*.safetensors"))
            snapshot_file.unlink()
            target_worker.bind_connector_metadata(target_meta)
            target_worker.start_load_kv(None)
            self.assertEqual(target_worker.get_block_ids_with_load_errors(), {1, 2})
            self.assertEqual(target_worker.get_block_ids_with_load_errors(), set())

            snapshot_file.write_bytes(b"corrupt")
            corrupt_request = SimpleNamespace(
                **{**request.__dict__, "request_id": "corrupt-request"}
            )
            self.assertEqual(
                target.get_num_new_matched_tokens(corrupt_request, 0), (0, False)
            )

    def test_rejects_scaled_rope_before_cache_reuse(self) -> None:
        self.assertEqual(
            _rope_theta(
                SimpleNamespace(
                    rope_scaling={"rope_theta": 1_000_000, "rope_type": "default"},
                    rope_parameters={"rope_theta": 1_000_000, "rope_type": "default"},
                )
            ),
            1_000_000,
        )
        with self.assertRaisesRegex(ValueError, "scaled RoPE"):
            _rope_theta(
                SimpleNamespace(
                    rope_theta=1_000_000,
                    rope_scaling={"rope_type": "yarn", "factor": 4},
                )
            )
        with self.assertRaisesRegex(ValueError, "scaled RoPE"):
            _rope_theta(
                SimpleNamespace(
                    rope_parameters={"rope_type": "dynamic", "rope_theta": 1_000_000}
                )
            )


if __name__ == "__main__":
    unittest.main()
