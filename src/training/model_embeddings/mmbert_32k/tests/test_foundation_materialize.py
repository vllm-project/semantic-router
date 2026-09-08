"""Offline materializer integration with bounded fake writer dependencies."""

from __future__ import annotations

import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from src.training.model_embeddings.mmbert_32k import foundation_materialize
from src.training.model_embeddings.mmbert_32k.foundation_integrity import (
    FOUNDATION_WRITER_BATCH_SIZE,
)

SHA256_HEX_LENGTH = 64
PREFIX_BYTES = 12


class FakeTokenizer:
    sep_token_id = 99
    eos_token_id = 98
    model_max_length = 8

    def __call__(self, text, *, add_special_tokens, truncation):
        if add_special_tokens or truncation:
            raise AssertionError("materializer must tokenize complete documents")
        return {"input_ids": [int(piece) for piece in text.split()]}

    @staticmethod
    def num_special_tokens_to_add(*, pair):
        if pair:
            raise AssertionError("foundation examples are single sequences")
        return 2

    @staticmethod
    def build_inputs_with_special_tokens(payload):
        return [101, *payload, 102]


class FakeAutoTokenizer:
    @staticmethod
    def from_pretrained(model_name, *, revision):
        if (model_name, revision) != ("base", "model-revision"):
            raise AssertionError("tokenizer identity was not pinned")
        return FakeTokenizer()


class FakeValue:
    def __init__(self, dtype):
        self.dtype = dtype


class FakeSequence:
    def __init__(self, feature, *, length):
        self.feature = feature
        self.length = length


class FakeFeatures(dict):
    pass


class FakeGeneratedDataset:
    def __init__(self, rows, features):
        self.rows = rows
        self.features = features

    def __len__(self):
        return len(self.rows)

    def save_to_disk(self, destination):
        destination.mkdir(parents=True)
        payload = json.dumps(self.rows, sort_keys=True).encode("utf-8")
        (destination / "data-00000-of-00001.arrow").write_bytes(payload)


class FakeDatasetAPI:
    generated = None
    writer_batch_size = None

    @classmethod
    def from_generator(
        cls,
        generate_examples,
        *,
        features,
        cache_dir,
        keep_in_memory,
        fingerprint,
        writer_batch_size,
    ):
        if keep_in_memory:
            raise AssertionError("writer must remain disk backed")
        if not Path(cache_dir).is_dir() or len(fingerprint) != SHA256_HEX_LENGTH:
            raise AssertionError("bounded writer cache/fingerprint contract changed")
        cls.writer_batch_size = writer_batch_size
        cls.generated = FakeGeneratedDataset(list(generate_examples()), features)
        return cls.generated


def fake_datasets_module():
    return types.SimpleNamespace(
        Dataset=FakeDatasetAPI,
        Features=FakeFeatures,
        Sequence=FakeSequence,
        Value=FakeValue,
        load_from_disk=lambda _destination: FakeDatasetAPI.generated,
    )


def fake_source(language, *, stats, **contract):
    if language != "en" or contract["expected_prefix_bytes"] != PREFIX_BYTES:
        raise AssertionError("source-prefix contract was not delegated")
    stats.url = "https://data.statmt.org/cc-100/en.txt.xz"
    stats.etag = "strong"
    stats.last_modified = "fixed"
    stats.content_length = 123
    stats.compressed_bytes_read = PREFIX_BYTES
    stats.compressed_prefix_sha256 = "a" * 64
    yield from ("1 2", "3 4 5", "6 7 8 9 10")


class FoundationMaterializeTest(unittest.TestCase):
    def test_fixed_writer_and_replay_receipt_are_persisted(self) -> None:
        datasets_module = fake_datasets_module()

        def fake_import(name):
            if name == "datasets":
                return datasets_module
            if name == "transformers":
                return types.SimpleNamespace(AutoTokenizer=FakeAutoTokenizer)
            raise AssertionError(f"unexpected import: {name}")

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "packed"
            with (
                mock.patch.object(
                    foundation_materialize.importlib,
                    "import_module",
                    side_effect=fake_import,
                ),
                mock.patch.object(
                    foundation_materialize,
                    "iter_cc100_documents",
                    side_effect=fake_source,
                ),
            ):
                dataset = foundation_materialize.prepare_streaming_packed_dataset(
                    dataset_name="statmt/cc100",
                    dataset_revision="loader-revision",
                    output_dir=str(output),
                    languages=["en"],
                    target_sequence_count=2,
                    max_length=8,
                    model_name="base",
                    model_revision="model-revision",
                    source_etags=["strong"],
                    source_content_lengths=[123],
                    max_document_bytes=1024,
                    max_document_tokens=128,
                    acknowledge_cc100_license_unknown=True,
                    expected_source_prefixes={
                        "en": {
                            "compressed_bytes_read": PREFIX_BYTES,
                            "compressed_prefix_sha256": "a" * 64,
                        }
                    },
                )

            self.assertEqual(len(dataset), 2)
            self.assertEqual(
                FakeDatasetAPI.writer_batch_size, FOUNDATION_WRITER_BATCH_SIZE
            )
            self.assertEqual(dataset.features["input_ids"].length, 8)
            self.assertTrue(
                all(row["attention_mask"] == [1] * 8 for row in dataset.rows)
            )
            manifest = json.loads(
                (output / "packing_manifest.json").read_text(encoding="utf-8")
            )
            self.assertTrue(
                manifest["source_streams"]["en"]["prefix_contract_verified"]
            )
            self.assertEqual(
                manifest["writer_contract"]["writer_batch_size"],
                FOUNDATION_WRITER_BATCH_SIZE,
            )
            self.assertTrue(
                manifest["data_governance"]["license_acknowledgement"][
                    "cc100_dataset_license_unknown"
                ]
            )
            self.assertEqual(len(manifest["arrow_shards"]), 1)


if __name__ == "__main__":
    unittest.main()
