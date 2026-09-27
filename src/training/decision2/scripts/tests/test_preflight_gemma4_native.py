"""CPU-only contracts for the official Gemma source feasibility probe."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import torch
from safetensors.torch import save_file
from scripts import preflight_gemma4_native as probe


class TinyTokenizer:
    bos_token_id = 2
    pad_token_id = 0

    def encode(self, value: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return [ord(char) + 3 for char in value]


class GemmaSourceProbeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_three_types_have_distinct_option_endpoints_and_no_truncation(self) -> None:
        items = probe.encode_probes(TinyTokenizer(), 8192)
        self.assertEqual(
            {item["task_type"] for item in items}, {"choice", "noul", "score"}
        )
        for item in items:
            self.assertEqual(item["ids"][0], TinyTokenizer.bos_token_id)
            self.assertEqual(
                len(set(item["candidate_positions"])), item["option_count"]
            )
            self.assertTrue(
                all(p < item["query_position"] for p in item["candidate_positions"])
            )
            self.assertLess(len(item["ids"]), 8192)
        with self.assertRaisesRegex(ValueError, "exceeds max_length"):
            probe.encode_probes(TinyTokenizer(), 100)

    def test_tensor_headers_separate_text_and_vision_without_loading(self) -> None:
        shard_name = "model-00001-of-00001.safetensors"
        tensors = {
            "model.language_model.embed_tokens.weight": torch.zeros((3, 4)),
            "model.language_model.layers.0.self_attn.q_proj.weight": torch.ones((2, 4)),
            "model.vision_tower.patch_embedder.input_proj.weight": torch.ones((2, 3)),
            "model.embed_vision.embedding_projection.weight": torch.ones((4, 2)),
        }
        save_file(tensors, str(self.root / shard_name))
        index = {
            "metadata": {"total_parameters": 34, "total_size": 136},
            "weight_map": dict.fromkeys(tensors, shard_name),
        }
        (self.root / "model.safetensors.index.json").write_text(
            json.dumps(index), encoding="utf-8"
        )
        report = probe.tensor_inventory(self.root)
        self.assertEqual(report["stored_parameter_count"], 34)
        self.assertEqual(report["stored_parameter_groups"]["language_model"], 20)
        self.assertEqual(report["stored_parameter_groups"]["vision_tower"], 6)
        self.assertEqual(report["stored_parameter_groups"]["vision_projection"], 8)
        index["weight_map"]["missing.weight"] = shard_name
        (self.root / "model.safetensors.index.json").write_text(
            json.dumps(index), encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "missing tensors"):
            probe.tensor_inventory(self.root)

    def test_revision_requires_exact_download_metadata(self) -> None:
        metadata = self.root / ".cache/huggingface/download/config.json.metadata"
        metadata.parent.mkdir(parents=True)
        metadata.write_text(f"{probe.SOURCE_REVISION}\nETag\n", encoding="utf-8")
        self.assertEqual(probe.local_revision(self.root), probe.SOURCE_REVISION)
        metadata.write_text("wrong-revision\nETag\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "pinned official revision"):
            probe.local_revision(self.root)

    def test_loaded_state_accounts_for_tied_head_and_persisted_buffer(self) -> None:
        class TinyOfficial(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = torch.nn.Module()
                self.model.language_model = torch.nn.Module()
                self.model.language_model.embed_tokens = torch.nn.Embedding(3, 4)
                self.model.language_model.register_buffer("layer_scalar", torch.ones(1))
                self.model.vision_tower = torch.nn.Linear(3, 2, bias=False)
                self.lm_head = torch.nn.Linear(4, 3, bias=False)
                self.lm_head.weight = self.model.language_model.embed_tokens.weight

        model = TinyOfficial()
        shard_name = "model-00001-of-00001.safetensors"
        state = model.state_dict()
        stored = {
            key: value.clone().contiguous()
            for key, value in state.items()
            if key != "lm_head.weight"
        }
        save_file(stored, str(self.root / shard_name))
        index = {
            "metadata": {},
            "weight_map": dict.fromkeys(stored, shard_name),
        }
        (self.root / "model.safetensors.index.json").write_text(
            json.dumps(index), encoding="utf-8"
        )
        inventory = probe.tensor_inventory(self.root)
        counts = probe.verify_loaded_state(model, self.root, inventory)
        self.assertEqual(counts["loaded_text_parameters"], 12)
        self.assertEqual(counts["loaded_total_parameters"], 18)
        self.assertEqual(counts["stored_buffer_elements"], 1)
        self.assertEqual(inventory["stored_parameter_count"], 19)
        model.lm_head.weight = torch.nn.Parameter(torch.zeros((3, 4)))
        with self.assertRaisesRegex(RuntimeError, "not tied"):
            probe.verify_loaded_state(model, self.root, inventory)

    def test_receipt_must_be_new_in_owner_private_directory(self) -> None:
        private = self.root / "private"
        private.mkdir(mode=0o700)
        os.chmod(private, 0o700)
        target = private / "receipt.json"
        probe.private_output(target, {"model_quality_evaluated": False})
        self.assertEqual(target.stat().st_mode & 0o777, 0o600)
        with self.assertRaisesRegex(ValueError, "new absolute"):
            probe.private_output(target, {})

    def test_repeated_zero_step_must_keep_type_logits_and_category(self) -> None:
        first = {
            "source_id": probe.SOURCE_ID,
            "source_revision": probe.SOURCE_REVISION,
            "config_sha256": probe.CONFIG_SHA256,
            "tokenizer_json_sha256": "tokenizer",
            "weights": {"shard_sha256": {"part": "weight"}},
            "loaded_text_parameters": 10,
            "loaded_total_parameters": 12,
            "synthetic_probe_token_hashes": {"choice": "c", "noul": "n", "score": "s"},
            "probe_version": probe.PROBE_VERSION,
            "random_untrained_head": True,
            "model_quality_evaluated": False,
            "typed_outputs": {
                kind: {
                    "candidate_count": 2,
                    "input_tokens": 100,
                    "finite_logits": True,
                    "logits": [0.1, 0.9],
                    "wrapper_hidden_max_abs_drift": 0.0,
                }
                for kind in ("choice", "noul", "score")
            },
        }
        second = json.loads(json.dumps(first))
        second["typed_outputs"]["score"]["logits"][1] += 0.0005
        passed = probe.compare_zero_step(first, second)
        self.assertEqual(passed["status"], "zero_step_mechanics_passed")
        self.assertAlmostEqual(passed["max_logit_abs_drift"], 0.0005)
        second["typed_outputs"]["score"]["logits"] = [1.0, 0.1]
        with self.assertRaisesRegex(ValueError, "selected category changed"):
            probe.compare_zero_step(first, second)
        second["typed_outputs"]["score"]["logits"] = [0.1, 0.902]
        with self.assertRaisesRegex(ValueError, "drift exceeds"):
            probe.compare_zero_step(first, second)


if __name__ == "__main__":
    unittest.main()
