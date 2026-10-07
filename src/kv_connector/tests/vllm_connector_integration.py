"""Check the out-of-tree plugin against an installed vLLM interface."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from src.training.kv_mapper.artifact import CompatibilitySpec, Manifest, write_artifact

if importlib.util.find_spec("vllm"):
    from vllm.config.kv_transfer import KVTransferConfig
    from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole


@unittest.skipUnless(importlib.util.find_spec("vllm"), "vLLM is not installed")
class VllmConnectorTests(unittest.TestCase):
    def test_factory_loads_plugin_and_fail_closed_prefill(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            compat = CompatibilitySpec(
                source_model="source",
                source_revision="a" * 40,
                target_model="target",
                target_revision="b" * 40,
                variant="full_head",
                precision="bf16",
                source_tp=1,
                target_tp=1,
                head_order="contiguous",
                num_kv_heads=1,
                head_dim=2,
            )
            manifest = Manifest(
                mapper_id="synthetic",
                compatibility=compat,
                topk=1,
                ridge_alpha=1.0,
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
            write_artifact(path, manifest, tensors)
            extra = {
                "artifact_path": str(path),
                "source_model": "source",
                "source_revision": "a" * 40,
                "source_tp": 1,
                "head_order": "contiguous",
            }
            transfer = KVTransferConfig(
                kv_connector="KVMapperConnector",
                kv_connector_module_path="src.kv_connector.vllm_connector",
                kv_role="kv_both",
                kv_connector_extra_config=extra,
            )
            config = SimpleNamespace(
                kv_transfer_config=transfer,
                model_config=SimpleNamespace(
                    model="target",
                    revision="b" * 40,
                    dtype=torch.bfloat16,
                    hf_config=SimpleNamespace(
                        num_key_value_heads=1,
                        head_dim=2,
                        num_hidden_layers=1,
                    ),
                ),
                parallel_config=SimpleNamespace(tensor_parallel_size=1),
            )
            cls = KVConnectorFactory.get_connector_class(transfer)
            connector = cls(config, KVConnectorRole.SCHEDULER, None)
            self.assertIsNotNone(connector.artifact)
            self.assertEqual(connector.get_num_new_matched_tokens(None, 0), (0, False))
            config.model_config.hf_config.num_hidden_layers = 2
            wrong_layers = cls(config, KVConnectorRole.SCHEDULER, None)
            self.assertIsNone(wrong_layers.artifact)
            config.model_config.hf_config.num_hidden_layers = 1
            transfer.kv_connector_extra_config["artifact_path"] = str(path / "missing")
            missing = cls(config, KVConnectorRole.SCHEDULER, None)
            self.assertIsNone(missing.artifact)
            self.assertEqual(missing.get_num_new_matched_tokens(None, 0), (0, False))
            transfer.kv_connector_extra_config["artifact_path"] = str(path)
            (path / "weights.safetensors").write_bytes(b"corrupt")
            corrupt = cls(config, KVConnectorRole.SCHEDULER, None)
            self.assertIsNone(corrupt.artifact)
            self.assertEqual(corrupt.get_num_new_matched_tokens(None, 0), (0, False))


if __name__ == "__main__":
    unittest.main()
