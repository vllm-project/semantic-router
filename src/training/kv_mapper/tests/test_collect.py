from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np

from src.training.kv_mapper.collect import (
    ActivationRunMeta,
    as_bshd_numpy,
    calibration_windows,
    parse_layer_subset,
    read_run_metadata,
    resolve_stride,
    token_fingerprint,
    validate_activation_chunk,
    write_activation_chunk,
    write_run_metadata,
)


class CollectContractTests(unittest.TestCase):
    def test_calibration_windows_use_requested_count_and_stride(self) -> None:
        documents = [[0, 1, 2], [3, 4, 5, 6, 7, 8, 9], [10, 11, 12, 13]]
        self.assertEqual(
            list(
                calibration_windows(
                    documents, seq_len=4, window_stride=2, num_sequences=3
                )
            ),
            [[3, 4, 5, 6], [5, 6, 7, 8], [10, 11, 12, 13]],
        )

    def test_calibration_windows_reject_short_corpus(self) -> None:
        with self.assertRaisesRegex(ValueError, "only 1 complete windows"):
            list(
                calibration_windows(
                    [[0, 1, 2, 3]], seq_len=4, window_stride=4, num_sequences=2
                )
            )

    def test_stride_defaults_to_disjoint_windows(self) -> None:
        self.assertEqual(resolve_stride(0, 1024), 1024)
        self.assertEqual(resolve_stride(4096, 1024), 4096)
        self.assertEqual(resolve_stride(4, 1024), 4)

    def test_as_bshd_layouts(self) -> None:
        rng = np.random.default_rng(0)
        bshd = rng.standard_normal((1, 5, 8, 16))
        bhmd = np.transpose(bshd, (0, 2, 1, 3))
        self.assertEqual(as_bshd_numpy(bshd, 8, 16).shape, (1, 5, 8, 16))
        self.assertEqual(as_bshd_numpy(bhmd, 8, 16).shape, (1, 5, 8, 16))
        flat = rng.standard_normal((1, 5, 8 * 16))
        self.assertEqual(as_bshd_numpy(flat, 8, 16).shape, (1, 5, 8, 16))

    def test_parse_layer_subset(self) -> None:
        self.assertEqual(parse_layer_subset(None, 4), [0, 1, 2, 3])
        self.assertEqual(parse_layer_subset("all", 4), [0, 1, 2, 3])
        self.assertEqual(parse_layer_subset("0:2", 8), [0, 1])
        self.assertEqual(parse_layer_subset("1,3,5", 8), [1, 3, 5])

    def test_run_metadata_roundtrip(self) -> None:
        meta = ActivationRunMeta(
            corpus="HuggingFaceFW/fineweb-edu",
            dataset_config="sample-10BT",
            dataset_revision="dataset-sha",
            source_model="Qwen/Qwen3-14B",
            source_revision="abc123",
            target_model="Qwen/Qwen3-32B",
            target_revision="def456",
            seed=42,
            seq_len=1024,
            window_stride=1024,
            fitting_token_step=4,
            token_sha256="token-sha",
            num_sequences=8,
            source_layers=[0, 1],
            target_layers=list(range(4)),
            num_kv_heads=8,
            head_dim=128,
            precision="fp16",
        )
        with tempfile.TemporaryDirectory() as directory:
            write_run_metadata(Path(directory), meta)
            got = read_run_metadata(Path(directory))
            raw = json.loads((Path(directory) / "run.json").read_text())
        self.assertEqual(got.source_revision, "abc123")
        self.assertEqual(got.source_layers, [0, 1])
        self.assertTrue(got.rope_stripped_on_keys)
        self.assertEqual(raw["seed"], 42)
        self.assertEqual(raw["corpus"], "HuggingFaceFW/fineweb-edu")

    def test_disjoint_windows_and_sampled_chunk_resume(self) -> None:
        windows = np.asarray(
            list(
                calibration_windows(
                    [list(range(12))], seq_len=4, window_stride=4, num_sequences=3
                )
            )
        )
        np.testing.assert_array_equal(windows[:, 0], [0, 4, 8])
        self.assertEqual(token_fingerprint(windows), token_fingerprint(windows.copy()))
        self.assertNotEqual(
            token_fingerprint(windows), token_fingerprint(windows[:, ::-1])
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source" / "000000.npz"
            sample = np.arange(8, dtype=np.float32).reshape(2, 1, 4)
            write_activation_chunk(path, [sample], [sample + 1])
            self.assertTrue(validate_activation_chunk(path, 1, 2, 1, 4))
            with self.assertRaisesRegex(ValueError, "expected shape"):
                validate_activation_chunk(path, 1, 4, 1, 4)
            path.write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "corrupt"):
                validate_activation_chunk(path, 1, 2, 1, 4)


if __name__ == "__main__":
    unittest.main()
