"""Retention cache identity and logical-batch loss tests using synthetic inputs."""

# ruff: noqa: PLC0415

import copy
import hashlib
import importlib.util
import json
import os
import tempfile
import unittest
from collections import UserDict
from pathlib import Path

from src.training.model_classifier.sequence_repair.retention import (
    RetentionCache,
    masked_teacher_kl,
    row_identity,
    tokenizer_file_receipt,
    write_retention_cache,
)


class CacheFixture:
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.rows = [
            {
                "id": name,
                "group_id": name,
                "text": f"Synthetic {name}",
                "label": label,
                "retention_replay": name != "new",
            }
            for name, label in [
                ("old-a", "a"),
                ("old-b", "b"),
                ("old-c", "b"),
                ("new", "a"),
            ]
        ]
        self.tokens = {
            row["id"]: {"input_ids": [1, index + 2], "attention_mask": [1, 1]}
            for index, row in enumerate(self.rows)
        }
        self.metadata = {
            "initialization": {
                "base_files": {"model.safetensors": "1" * 64, "config.json": "2" * 64}
            },
            "contract_sha256": "3" * 64,
            "tokenizer_files": {
                "tokenizer.json": "4" * 64,
                "tokenizer_config.json": "5" * 64,
            },
            "label_to_id": {"a": 0, "b": 1},
        }
        records = [
            {**row_identity(row, self.tokens[row["id"]]), "logits": logits}
            for row, logits in zip(
                self.rows, [[2.0, 0.0], [1.0, 1.0], [2.0, 1.0]], strict=False
            )
        ]
        self.path = write_retention_cache(
            self.root / "cache", records=records, **self.metadata
        )

    def load(self, **changes):
        arguments = {
            **self.metadata,
            "train_rows": self.rows,
            "tokenized_rows": self.tokens,
            "replay_ids": [row["id"] for row in self.rows if row["retention_replay"]],
        }
        arguments.update(changes)
        return RetentionCache.load(self.path, **arguments)


class CacheIdentityTests(CacheFixture, unittest.TestCase):
    def test_exact_cache_covers_only_explicit_replay_and_derives_compatibility(self):
        cache = self.load()
        self.assertEqual(cache.receipt["replay_rows"], len(self.rows) - 1)
        self.assertEqual(cache.receipt["eligible_rows"], 1)
        self.assertEqual(
            row_identity(self.rows[0], UserDict(self.tokens["old-a"])),
            row_identity(self.rows[0], self.tokens["old-a"]),
        )

    def test_model_contract_tokenizer_and_label_changes_rejected(self):
        for key in (
            "initialization",
            "contract_sha256",
            "tokenizer_files",
            "label_to_id",
        ):
            value = copy.deepcopy(self.metadata[key])
            if key == "initialization":
                value["base_files"]["model.safetensors"] = "6" * 64
            elif key == "contract_sha256":
                value = "6" * 64
            elif key == "tokenizer_files":
                value["tokenizer.json"] = "6" * 64
            else:
                value = {"a": 1, "b": 0}
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.load(**{key: value})

    def test_row_or_token_identity_changes_rejected(self):
        for key in ("text", "group_id", "label"):
            rows = copy.deepcopy(self.rows)
            rows[0][key] = "b" if key == "label" else "changed"
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.load(train_rows=rows)
        tokens = copy.deepcopy(self.tokens)
        tokens["old-a"]["input_ids"][1] += 1
        with self.assertRaises(ValueError):
            self.load(tokenized_rows=tokens)

    def test_missing_or_new_replay_designation_and_dev_rejected(self):
        for mutation in ("missing", "new", "nonbool", "dev"):
            rows = copy.deepcopy(self.rows)
            if mutation == "missing":
                rows[0].pop("retention_replay")
            elif mutation == "new":
                rows[-1]["retention_replay"] = True
            elif mutation == "nonbool":
                rows[0]["retention_replay"] = 1
            else:
                rows[0]["split"] = "validation"
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                self.load(train_rows=rows)

    def test_byte_changes_and_rehashed_duplicate_targets_rejected(self):
        targets = self.path.parent / "targets.jsonl"
        original = targets.read_bytes()
        targets.write_bytes(original + b"\n")
        with self.assertRaises(ValueError):
            self.load()
        lines = original.splitlines()
        targets.write_bytes(b"\n".join([lines[0], lines[0], lines[2]]) + b"\n")
        manifest = json.loads(self.path.read_text())
        manifest["records_sha256"] = hashlib.sha256(targets.read_bytes()).hexdigest()
        self.path.write_text(json.dumps(manifest))
        with self.assertRaises(ValueError):
            self.load()

    def test_unpadded_complete_tokens_and_float32_serialization_required(self):
        with self.assertRaises(ValueError):
            row_identity(self.rows[0], {"input_ids": [1, 2], "attention_mask": [1, 0]})
        record = {
            **row_identity(self.rows[0], self.tokens["old-a"]),
            "logits": [0.1, 0.0],
        }
        with self.assertRaises(ValueError):
            write_retention_cache(
                self.root / "rounded", records=[record], **self.metadata
            )
        record["logits"] = [1.0, 0.0]
        record["eligible"] = True
        with self.assertRaises(ValueError):
            write_retention_cache(
                self.root / "trusted-mask", records=[record], **self.metadata
            )

    def test_tokenizer_receipt_includes_added_tokens(self):
        base = self.root / "tokenizer"
        base.mkdir()
        for name in ("tokenizer.json", "tokenizer_config.json", "added_tokens.json"):
            (base / name).write_text("{}")
        before = tokenizer_file_receipt(base)
        (base / "added_tokens.json").write_text('{"new":1}')
        self.assertNotEqual(before, tokenizer_file_receipt(base))


@unittest.skipUnless(
    importlib.util.find_spec("torch"), "Optional mathematical tests need Torch"
)
class MaskedLossTests(CacheFixture, unittest.TestCase):
    @property
    def device(self):
        return os.environ.get("SEQUENCE_REPAIR_TEST_DEVICE", "cpu")

    def test_targets_accept_repeated_train_draws_reject_unknown_rows(self):
        cache = self.load()
        values, mask = cache.targets([*self.rows, self.rows[0]], self.device)
        self.assertEqual(
            tuple(values.shape), (len(self.rows) + 1, len(self.metadata["label_to_id"]))
        )
        self.assertEqual(mask.tolist(), [True, False, False, False, True])
        row = {**self.rows[0], "id": "development-only"}
        with self.assertRaises(ValueError):
            cache.targets([row], self.device)

    def test_logical_partition_equivalence_and_teacher_detach(self):
        import torch

        torch.set_num_threads(1)
        logits = torch.tensor(
            [[1.0, -2.0], [0.1, 0.5], [-0.3, 0.6], [2.0, -1.0]],
            requires_grad=True,
            device=self.device,
        )
        teacher = torch.tensor(
            [[0.5, -0.2], [float("nan"), float("nan")], [0.9, -0.1], [0.4, 1.2]],
            requires_grad=True,
            device=self.device,
        )
        mask = torch.tensor([True, False, True, True], device=self.device)
        count, temperature = len(logits), 2.0
        whole = masked_teacher_kl(
            logits, teacher, mask, examples_per_step=count, temperature=temperature
        )
        whole.backward()
        expected = torch.zeros_like(logits)
        expected[mask] = (
            temperature
            * (
                torch.softmax(logits.detach()[mask] / temperature, -1)
                - torch.softmax(teacher.detach()[mask] / temperature, -1)
            )
            / count
        )
        torch.testing.assert_close(logits.grad, expected)
        self.assertIsNone(teacher.grad)
        split = logits.detach().clone().requires_grad_()
        pieces = [(0, 1), (1, 2), (2, count)]
        total = sum(
            masked_teacher_kl(
                split[a:b],
                teacher[a:b],
                mask[a:b],
                examples_per_step=count,
                temperature=temperature,
            )
            for a, b in pieces
        )
        total.backward()
        torch.testing.assert_close(total, whole)
        torch.testing.assert_close(split.grad, logits.grad)

    def test_zero_mask_and_invalid_geometry(self):
        import torch

        logits = torch.tensor([[1e30, 1e30]], requires_grad=True, device=self.device)
        teacher = torch.full_like(logits, float("nan"), requires_grad=True)
        mask = torch.tensor([False], device=self.device)
        loss = masked_teacher_kl(logits, teacher, mask, examples_per_step=1)
        self.assertEqual(float(loss), 0.0)
        loss.backward()
        self.assertEqual(logits.grad.tolist(), [[0.0, 0.0]])
        self.assertIsNone(teacher.grad)
        for kwargs in (
            {"examples_per_step": 0},
            {"examples_per_step": True},
            {"examples_per_step": 1, "temperature": float("inf")},
            {"examples_per_step": 1, "temperature": 0},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                masked_teacher_kl(logits, teacher, mask, **kwargs)
        with self.assertRaises(ValueError):
            masked_teacher_kl(
                logits,
                teacher,
                torch.tensor([True], device=self.device),
                examples_per_step=1,
            )


if __name__ == "__main__":
    unittest.main()
