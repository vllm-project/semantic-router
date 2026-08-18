"""Packing-manifest and Arrow handoff integrity tests."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from src.training.model_embeddings.mmbert_32k.foundation_integrity import (
    ATTENTION_POLICY,
    CC100_UNDERLYING_CONTENT,
    COMMON_CRAWL_TERMS_URL,
    FOUNDATION_WRITER_BATCH_SIZE,
    INTEGRITY_LIMIT,
    LICENSE_ACKNOWLEDGEMENT,
    LOADER_REVISION_SCOPE,
    MANIFEST_SCHEMA_VERSION,
    PACKING_ALGORITHM,
    RELEASE_GATE,
    SOURCE_HASH_SCOPE,
    arrow_shard_records,
    canonical_json_bytes,
    load_source_prefix_contract,
    validate_packing_manifest,
    write_canonical_manifest,
    write_training_receipt,
)


class ValueFeature:
    def __init__(self, dtype):
        self.dtype = dtype


class SequenceFeature:
    def __init__(self, dtype, length):
        self.feature = ValueFeature(dtype)
        self.length = length


class FakeDataset:
    def __init__(self, rows=2, length=4):
        self.rows = rows
        self.features = {
            "input_ids": SequenceFeature("int32", length),
            "attention_mask": SequenceFeature("int8", length),
            "language": ValueFeature("string"),
        }

    def __len__(self):
        return self.rows


def manifest_payload(root: Path, *, length: int = 4, rows: int = 2) -> dict:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "algorithm": PACKING_ALGORITHM,
        "source": {
            "repo_id": "statmt/cc100",
            "revision": "loader-revision",
            "loader_contract_revision": "loader-revision",
            "loader_revision_scope": LOADER_REVISION_SCOPE,
            "split": "train",
            "streaming": True,
            "text_column": "text",
        },
        "tokenizer": {"repo_id": "base", "revision": "model-revision"},
        "target_length": length,
        "target_sequence_count": rows,
        "dataset_fingerprint": "f" * 64,
        "languages": ["en"],
        "language_quotas": {"en": rows},
        "document_separator_token_id": 99,
        "attention_mask_policy": ATTENTION_POLICY,
        "writer_contract": {
            "writer_batch_size": FOUNDATION_WRITER_BATCH_SIZE,
            "features": {
                "input_ids": {"dtype": "int32", "length": length},
                "attention_mask": {"dtype": "int8", "length": length},
                "language": {"dtype": "string"},
            },
        },
        "document_caps": {"max_bytes": 1024, "max_tokens": 128},
        "language_stats": {
            "en": {
                "documents_consumed": 1,
                "characters_consumed": 4,
                "source_tokens_consumed": 8,
                "separators_inserted": 1,
                "sequences_emitted": rows,
            }
        },
        "source_streams": {
            "en": {
                "url": "https://data.statmt.org/cc-100/en.txt.xz",
                "etag": "strong",
                "etag_strength": "strong",
                "last_modified": "fixed",
                "content_length": 123,
                "compressed_bytes_read": 12,
                "compressed_prefix_sha256": "a" * 64,
                "hash_scope": SOURCE_HASH_SCOPE,
                "full_object_sha256": None,
                "prefix_contract_verified": True,
            }
        },
        "packed_input_ids_sha256": "b" * 64,
        "arrow_shards": arrow_shard_records(root),
        "data_governance": {
            "declared_dataset_license": None,
            "underlying_content": CC100_UNDERLYING_CONTENT,
            "common_crawl_terms_url": COMMON_CRAWL_TERMS_URL,
            "license_acknowledgement": LICENSE_ACKNOWLEDGEMENT,
            "release_gate": RELEASE_GATE,
            "integrity_limit": INTEGRITY_LIMIT,
        },
    }


def validate(root: Path, dataset: FakeDataset, **overrides):
    arguments = {
        "expected_source_repo_id": "statmt/cc100",
        "expected_dataset_revision": "loader-revision",
        "expected_tokenizer_repo_id": "base",
        "expected_tokenizer_revision": "model-revision",
        "expected_target_length": 4,
        "expected_sequence_count": 2,
        "expected_languages": ["en"],
        "expected_source_etags": ["strong"],
        "expected_source_content_lengths": [123],
        "expected_max_document_bytes": 1024,
        "expected_max_document_tokens": 128,
        "acknowledge_cc100_license_unknown": True,
        "verify_attention_values": False,
    }
    arguments.update(overrides)
    return validate_packing_manifest(root, dataset, **arguments)


class FoundationIntegrityTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "data-00000-of-00001.arrow").write_bytes(b"arrow payload")
        self.payload = manifest_payload(self.root)
        self.digest = write_canonical_manifest(self.root, self.payload)

    def tearDown(self):
        self.temporary.cleanup()

    def test_valid_handoff_and_receipt_bind_manifest_digest(self) -> None:
        result = validate(self.root, FakeDataset())
        self.assertEqual(result["packing_manifest_sha256"], self.digest)
        output = self.root / "output"
        write_training_receipt(
            output,
            validation=result,
            model_repo_id="base",
            model_revision="model-revision",
            dataset_revision="loader-revision",
            status="complete",
            optimizer_steps=2,
        )
        raw = (output / "training_receipt.json").read_bytes()
        receipt = json.loads(raw)
        self.assertEqual(raw, canonical_json_bytes(receipt))
        self.assertEqual(receipt["packing"]["packing_manifest_sha256"], self.digest)
        self.assertEqual(
            receipt["packing"]["source_content_receipt"]["en"][
                "compressed_prefix_sha256"
            ],
            "a" * 64,
        )
        self.assertTrue(
            receipt["packing"]["data_governance"]["license_acknowledgement"][
                "cc100_dataset_license_unknown"
            ]
        )

    def test_missing_or_tampered_manifest_and_arrow_fail_closed(self) -> None:
        (self.root / "packing_manifest.sha256").unlink()
        with self.assertRaisesRegex(RuntimeError, "missing"):
            validate(self.root, FakeDataset())
        write_canonical_manifest(self.root, self.payload)
        (self.root / "data-00000-of-00001.arrow").write_bytes(b"tampered")
        with self.assertRaisesRegex(RuntimeError, "Arrow shard"):
            validate(self.root, FakeDataset())

    def test_wrong_revision_length_row_count_and_writer_fail_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "loader revision"):
            validate(
                self.root,
                FakeDataset(),
                expected_dataset_revision="wrong-revision",
            )
        with self.assertRaisesRegex(RuntimeError, "target length"):
            validate(self.root, FakeDataset(), expected_target_length=8)
        with self.assertRaisesRegex(RuntimeError, "row count"):
            validate(self.root, FakeDataset(rows=1))

        changed = dict(self.payload)
        changed["writer_contract"] = dict(self.payload["writer_contract"])
        changed["writer_contract"]["writer_batch_size"] = 1024
        write_canonical_manifest(self.root, changed)
        with self.assertRaisesRegex(RuntimeError, "writer contract"):
            validate(self.root, FakeDataset())

    def test_wrong_quota_caps_or_license_acknowledgement_fail_closed(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "document caps"):
            validate(self.root, FakeDataset(), expected_max_document_bytes=2048)
        with self.assertRaisesRegex(RuntimeError, "explicit acknowledgement"):
            validate(
                self.root,
                FakeDataset(),
                acknowledge_cc100_license_unknown=False,
            )

        changed = json.loads(json.dumps(self.payload))
        changed["language_quotas"]["en"] = 1
        write_canonical_manifest(self.root, changed)
        with self.assertRaisesRegex(RuntimeError, "language quotas"):
            validate(self.root, FakeDataset())

    def test_audit_prefix_can_seed_replay_but_cannot_train_directly(self) -> None:
        audit = json.loads(json.dumps(self.payload))
        audit["source_streams"]["en"]["prefix_contract_verified"] = False
        write_canonical_manifest(self.root, audit)
        contract = load_source_prefix_contract(
            self.root,
            languages=["en"],
            source_etags=["strong"],
            source_content_lengths=[123],
        )
        self.assertEqual(contract["en"]["compressed_bytes_read"], 12)
        with self.assertRaisesRegex(RuntimeError, "external replay contract"):
            validate(self.root, FakeDataset())


HAS_DATASETS = importlib.util.find_spec("datasets") is not None
HAS_PYARROW = importlib.util.find_spec("pyarrow") is not None


@unittest.skipUnless(HAS_DATASETS and HAS_PYARROW, "requires pinned datasets/pyarrow")
class FoundationIntegrityIntegrationTest(unittest.TestCase):
    def test_real_fixed_size_arrow_and_attention_validation(self) -> None:
        from datasets import (  # noqa: PLC0415
            Dataset,
            Features,
            Sequence,
            Value,
            load_from_disk,
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "dataset"
            features = Features(
                {
                    "input_ids": Sequence(Value("int32"), length=4),
                    "attention_mask": Sequence(Value("int8"), length=4),
                    "language": Value("string"),
                }
            )
            rows = (
                {
                    "input_ids": [1, 2, 3, 4],
                    "attention_mask": [1, 1, 1, 1],
                    "language": "en",
                },
                {
                    "input_ids": [5, 6, 7, 8],
                    "attention_mask": [1, 1, 1, 1],
                    "language": "en",
                },
            )
            dataset = Dataset.from_generator(
                lambda: iter(rows),
                features=features,
                cache_dir=Path(temporary) / "cache",
                keep_in_memory=False,
                writer_batch_size=FOUNDATION_WRITER_BATCH_SIZE,
            )
            dataset.save_to_disk(root)
            write_canonical_manifest(root, manifest_payload(root))
            loaded = load_from_disk(root)
            result = validate_packing_manifest(
                root,
                loaded,
                expected_source_repo_id="statmt/cc100",
                expected_dataset_revision="loader-revision",
                expected_tokenizer_repo_id="base",
                expected_tokenizer_revision="model-revision",
                expected_target_length=4,
                expected_sequence_count=2,
                expected_languages=["en"],
                expected_source_etags=["strong"],
                expected_source_content_lengths=[123],
                expected_max_document_bytes=1024,
                expected_max_document_tokens=128,
                acknowledge_cc100_license_unknown=True,
            )
            self.assertRegex(result["packing_manifest_sha256"], r"^[0-9a-f]{64}$")


if __name__ == "__main__":
    unittest.main()
