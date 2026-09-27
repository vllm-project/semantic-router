import copy
import math
import unittest
from pathlib import Path

from training.model import train
from training.model.data import digest, file_sha256
from training.model.gradient_projection import PROJECTION_VERSION
from training.model.gradient_projection_preflight import (
    CODE_FILES,
    CONTROL_TRAIN_SHA256,
    MAX_PROBABILITY_DRIFT,
    PLANNED_UPDATES,
    SCHEMA,
    SELECT_PROBES,
    SOURCE_REVISION,
    _vector_diff,
    validate_receipt,
)
from training.model.train import TYPED_HEAD_PARTITIONS


def _vectors():
    return [
        {
            "token_ids_sha256": f"{index:064x}",
            "prediction_index": 0,
            "probabilities": [0.8, 0.2],
        }
        for index in range(SELECT_PROBES)
    ]


def _receipt():
    files = {
        "backbone/model.safetensors": "a" * 64,
        "decision_head.safetensors": "b" * 64,
    }
    return {
        "schema_version": SCHEMA,
        "status": "PASS_PROJECTED_ONE_UPDATE_PREFLIGHT",
        "created_utc": "2026-09-28T00:00:00+00:00",
        "source_revision": SOURCE_REVISION,
        "train_sha256": TYPED_HEAD_PARTITIONS["train"],
        "select_sha256": TYPED_HEAD_PARTITIONS["select"],
        "cal_sha256": TYPED_HEAD_PARTITIONS["cal"],
        "select32_sha256": "c" * 64,
        "plan_sha256": "d" * 64,
        "ordinary_receipt_sha256": "e" * 64,
        "code_sha256": dict.fromkeys(CODE_FILES, "f" * 64),
        "ordinary_train_code_sha256": CONTROL_TRAIN_SHA256,
        "projection_version": PROJECTION_VERSION,
        "planned_updates": PLANNED_UPDATES,
        "optimizer_steps": 1,
        "zero_vs_ordinary": {"categorical_changes": 0, "max_probability_diff": 0.0},
        "post_vs_reload": {"categorical_changes": 0, "max_probability_diff": 0.0},
        "projection": {
            "projection_version": PROJECTION_VERSION,
            "enabled": True,
            "task_counts": {"choice": 11, "noul": 5, "score": 0},
            "task_norms": {"choice": 1.0, "noul": 1.0, "score": 0.0},
            "pairwise_cosines": {
                "choice_noul": -0.1,
                "choice_score": None,
                "noul_score": None,
            },
            "projected_pairs": 1,
            "ordinary_backbone_norm": 1.0,
            "pre_match_backbone_norm": 1.5,
            "norm_scale": 2 / 3,
            "final_backbone_norm": 1.0,
        },
        "global_gradient_norm": 1.2,
        "checkpoint_files_sha256": files,
        "checkpoint_files_digest": digest(files),
        "device_seconds": 30.0,
        "device_gpu_hours": 30 / 3600,
    }


class ProjectedPreflightReceiptTest(unittest.TestCase):
    def test_archived_ordinary_trainer_source_is_unchanged(self):
        self.assertEqual(file_sha256(Path(train.__file__)), CONTROL_TRAIN_SHA256)

    def test_fixed32_zero_and_reload_parity(self):
        ordinary = _vectors()
        projected = copy.deepcopy(ordinary)
        self.assertEqual(
            _vector_diff(ordinary, projected),
            {"categorical_changes": 0, "max_probability_diff": 0.0},
        )
        projected[0]["probabilities"] = [
            0.8 - 2 * MAX_PROBABILITY_DRIFT,
            0.2 + 2 * MAX_PROBABILITY_DRIFT,
        ]
        self.assertGreater(
            _vector_diff(ordinary, projected)["max_probability_diff"],
            MAX_PROBABILITY_DRIFT,
        )
        projected[0]["prediction_index"] = 1
        self.assertEqual(_vector_diff(ordinary, projected)["categorical_changes"], 1)
        projected[0]["token_ids_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "roster"):
            _vector_diff(ordinary, projected)

    def test_receipt_has_gold_free_allowlist_and_fixed_gate(self):
        receipt = _receipt()
        validate_receipt(receipt)
        bad = copy.deepcopy(receipt)
        bad["gold_label"] = 1
        with self.assertRaisesRegex(ValueError, "unexpected"):
            validate_receipt(bad)
        bad = copy.deepcopy(receipt)
        bad["post_vs_reload"]["max_probability_diff"] = 2 * MAX_PROBABILITY_DRIFT
        with self.assertRaisesRegex(ValueError, "gate failed"):
            validate_receipt(bad)
        bad = copy.deepcopy(receipt)
        bad["projection"]["final_backbone_norm"] = math.nan
        with self.assertRaisesRegex(ValueError, "summary"):
            validate_receipt(bad)
        bad = copy.deepcopy(receipt)
        bad["checkpoint_files_sha256"]["/private/path"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "contract"):
            validate_receipt(bad)


if __name__ == "__main__":
    unittest.main()
