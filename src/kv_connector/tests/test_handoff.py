"""Same-host source snapshot and mapped target injection contracts."""

from __future__ import annotations

import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from safetensors import SafetensorError

from src.kv_connector.handoff import apply_handoff
from src.kv_connector.paged_cache import extract_prefix
from src.kv_connector.runtime import MapperArtifact
from src.kv_connector.snapshot import LocalSnapshotStore, SourceSnapshot
from src.training.kv_mapper.artifact import CompatibilitySpec, Manifest, write_artifact


class HandoffTests(unittest.TestCase):
    def setUp(self) -> None:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        self.store = LocalSnapshotStore(root / "snapshots")
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
            num_kv_heads=2,
            head_dim=2,
        )
        manifest = Manifest(
            mapper_id="mapper-1",
            compatibility=compat,
            topk=1,
            ridge_alpha=1.0,
            centered_inputs=True,
            rope_stripped_on_keys=True,
            source_layers_per_target={"k": {"0": [0]}, "v": {"0": [0]}},
        )
        path = root / "artifact"
        tensors = {
            f"target.0.{channel}.W": np.eye(4, dtype=np.float32)
            for channel in ("k", "v")
        }
        tensors.update(
            {
                f"target.0.{channel}.b": np.zeros(4, dtype=np.float32)
                for channel in ("k", "v")
            }
        )
        write_artifact(path, manifest, tensors)
        self.artifact = MapperArtifact.open(path, compat)
        keys = torch.arange(16, dtype=torch.bfloat16).reshape(4, 2, 2)
        values = keys + 1
        self.snapshot = SourceSnapshot(
            namespace="tenant-a",
            cache_id="session-1",
            source_model="source",
            source_revision="a" * 40,
            token_ids=(1, 2, 3, 4),
            layers={0: (keys, values)},
            rope_theta=1_000_000,
            expires_at=time.time() + 60,
        )

    def _load(self) -> SourceSnapshot:
        return self.store.load(
            "tenant-a",
            "session-1",
            source_model="source",
            source_revision="a" * 40,
        )

    def test_snapshot_transfer_and_noncontiguous_target_blocks(self) -> None:
        self.store.publish(self.snapshot)
        loaded = self._load()
        self.assertEqual(loaded.rope_theta, 1_000_000)
        cache = torch.zeros((3, 2, 2, 4), dtype=torch.bfloat16)
        count = apply_handoff(
            self.artifact,
            loaded,
            namespace="tenant-a",
            cache_id="session-1",
            mapper_id="mapper-1",
            target_prompt_ids=[1, 2, 3, 4, 5],
            target_caches={0: cache},
            block_ids=[2, 0],
            source_rope_theta=1_000_000,
            target_rope_theta=1_000_000,
            cache_layout="legacy",
        )
        self.assertEqual(count, 4)
        keys, values = extract_prefix(
            cache, [2, 0], 4, heads=2, head_dim=2, layout="legacy"
        )
        torch.testing.assert_close(keys, self.snapshot.layers[0][0])
        torch.testing.assert_close(values, self.snapshot.layers[0][1])
        self.assertEqual(cache[1].count_nonzero().item(), 0)

    def test_tenant_and_expiry_fail_closed(self) -> None:
        self.store.publish(self.snapshot)
        with self.assertRaises(FileExistsError):
            self.store.publish(self.snapshot)
        with self.assertRaises(FileNotFoundError):
            self.store.load(
                "tenant-b",
                "session-1",
                source_model="source",
                source_revision="a" * 40,
            )
        with self.assertRaisesRegex(ValueError, "expired"):
            self.store.load(
                "tenant-a",
                "session-1",
                source_model="source",
                source_revision="a" * 40,
                now=self.snapshot.expires_at + 1,
            )
        self.assertFalse(self.store._path("tenant-a", "session-1").exists())

    def test_expired_snapshots_are_pruned_before_next_publish(self) -> None:
        self.store.publish(self.snapshot)
        newer = replace(self.snapshot, cache_id="session-2")
        self.assertEqual(self.store.prune_expired(self.snapshot.expires_at + 1), 1)
        self.store.publish(newer)
        self.assertFalse(self.store._path("tenant-a", "session-1").exists())
        self.assertTrue(self.store._path("tenant-a", "session-2").exists())

    def test_corrupt_snapshot_reports_read_error(self) -> None:
        self.store.publish(self.snapshot)
        self.store._path("tenant-a", "session-1").write_bytes(b"corrupt")
        with self.assertRaises(SafetensorError):
            self._load()

    def test_bad_hint_or_destination_does_not_modify_cache(self) -> None:
        cache = torch.zeros((3, 2, 2, 4), dtype=torch.bfloat16)
        kwargs = {
            "artifact": self.artifact,
            "snapshot": self.snapshot,
            "namespace": "tenant-a",
            "cache_id": "session-1",
            "mapper_id": "mapper-1",
            "target_prompt_ids": [1, 2, 3, 4, 5],
            "target_caches": {0: cache},
            "block_ids": [2, 0],
            "source_rope_theta": 1_000_000,
            "target_rope_theta": 1_000_000,
            "cache_layout": "legacy",
        }
        with self.assertRaisesRegex(ValueError, "mapper ID"):
            apply_handoff(**{**kwargs, "mapper_id": "wrong"})
        with self.assertRaisesRegex(ValueError, "scope"):
            apply_handoff(**{**kwargs, "namespace": "tenant-b"})
        with self.assertRaisesRegex(ValueError, "cache ID"):
            apply_handoff(**{**kwargs, "cache_id": "another-session"})
        with self.assertRaisesRegex(ValueError, "prompt"):
            apply_handoff(**{**kwargs, "target_prompt_ids": [1, 2, 9, 4, 5]})
        with self.assertRaisesRegex(ValueError, "not enough blocks"):
            apply_handoff(**{**kwargs, "block_ids": [2]})
        self.assertEqual(cache.count_nonzero().item(), 0)


if __name__ == "__main__":
    unittest.main()
