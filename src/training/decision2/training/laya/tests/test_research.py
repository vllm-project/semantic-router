"""Check lineage quarantine and source-aware checkpoint selection."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from training.laya.build_research_data import _question, _quarantine
from training.laya.build_research_data import sha
from training.laya.infer_research import verify_checkpoint
from training.laya.train_research import _select_score, _weights


def item(
    item_id: str, source: str, task: str, label: int, option_keys: list[str]
) -> dict:
    return {
        "id": item_id,
        "source": source,
        "task_type": task,
        "label": label,
        "option_count": len(option_keys),
        "option_keys": option_keys,
    }


class LayaResearchTests(unittest.TestCase):
    def test_noul_option_order_can_be_reversed_without_changing_semantics(self) -> None:
        row = {
            "task_type": "noul",
            "instructions": "Decide",
            "options": [
                {"key": "true", "description": "yes"},
                {"key": "false", "description": "no"},
            ],
        }
        self.assertEqual(set(_question(row)["crit"]), {"false", "true"})
        row["options"][1]["key"] = "unknown"
        with self.assertRaises(ValueError):
            _question(row)

    def test_quarantine_removes_entire_lineage_group(self) -> None:
        common = {"instructions": "Decide", "options": [], "task_type": "choice"}
        train = [
            {
                **common,
                "id": "a1",
                "group_id": "a",
                "input_sha256": "a1",
                "state": "A long paragraph about lunar tides and orbit mechanics with detailed examples.",
            },
            {
                **common,
                "id": "a2",
                "group_id": "a",
                "input_sha256": "a2",
                "state": "A distinct long paragraph about mountain roads and seasonal weather.",
            },
            {
                **common,
                "id": "b1",
                "group_id": "b",
                "input_sha256": "b1",
                "state": "A separate paragraph on surgical instruments and sterile procedure.",
            },
        ]
        holdout = [
            {
                **common,
                "id": "protected",
                "group_id": "other",
                "input_sha256": "protected",
                "state": train[0]["state"],
            }
        ]
        retained, audit = _quarantine(train, holdout)
        self.assertEqual([row["id"] for row in retained], ["b1"])
        self.assertEqual(audit["quarantined_rows"], 2)
        self.assertGreaterEqual(audit["trigger_rows_by_reason"]["exact_context"], 1)

    def test_selection_uses_semantic_keys_and_domain_components(self) -> None:
        rows = []
        for task in (
            "css_pilot:discourse",
            "css_pilot:implicit_hate",
            "css_pilot:semeval_stance",
        ):
            rows.extend(
                [
                    item(task + "0", task, "choice", 1, ["A", "B"]),
                    item(task + "1", task, "choice", 1, ["B", "A"]),
                ]
            )
        for task in ("choice", "noul"):
            rows.extend(
                [
                    item(
                        "go" + task + "0",
                        "google_goemotions_official_train",
                        task,
                        0,
                        ["A", "B"],
                    ),
                    item(
                        "go" + task + "1",
                        "google_goemotions_official_train",
                        task,
                        1,
                        ["A", "B"],
                    ),
                ]
            )
        for task in ("choice", "noul", "score"):
            rows.extend(
                [
                    item(
                        "oracle" + task + "0",
                        "decision2_rights_clean_oracle_holdout_v1",
                        task,
                        0,
                        ["A", "B"],
                    ),
                    item(
                        "oracle" + task + "1",
                        "decision2_rights_clean_oracle_holdout_v1",
                        task,
                        1,
                        ["A", "B"],
                    ),
                ]
            )
        score = _select_score(rows, [row["label"] for row in rows])
        self.assertEqual(score["composite_macro_f1"], 1.0)
        self.assertEqual(
            score["component_macro_f1"],
            {"css": 1.0, "goemotions": 1.0, "structured": 1.0},
        )
        self.assertEqual(score["items"], len(rows))

    def test_training_weights_are_finite_positive_and_normalized(self) -> None:
        rows = [
            item("a", "legacy:stage4", "choice", 0, ["A", "B"]),
            item("b", "human", "choice", 1, ["A", "B"]),
            item("c", "human", "noul", 0, ["false", "true"]),
            item("d", "human", "score", 1, ["0", "1", "2"]),
        ]
        weights = _weights(rows)
        self.assertTrue(all(value > 0 for value in weights))
        self.assertAlmostEqual(sum(weights) / len(weights), 1.0)

    def test_checkpoint_receipt_rejects_tampered_weights_and_partial_run(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            run = root / "run"
            checkpoint = run / "checkpoint-000002"
            checkpoint.mkdir(parents=True)
            base = {"model_id": "pinned-test"}
            (run / "RUN.json").write_text(
                json.dumps(
                    {
                        "schema_version": "decision2-laya-joint-research-run/1",
                        "research_only": True,
                        "base": base,
                    }
                )
            )
            (checkpoint / "model.safetensors").write_bytes(b"fixture weights")
            (checkpoint / "rl_agent_config.json").write_text("{}")
            record = {
                "schema_version": "decision2-laya-joint-checkpoint/1",
                "research_only": True,
                "release_qualified": False,
                "run_sha256": sha(run / "RUN.json"),
                "step": 2,
                "selection": {"macro": 0.42},
                "files": {
                    name: sha(checkpoint / name)
                    for name in ("model.safetensors", "rl_agent_config.json")
                },
            }
            (checkpoint / "CHECKPOINT.json").write_text(json.dumps(record))
            point = {
                "step": 2,
                "checkpoint": {
                    "checkpoint_sha256": sha(checkpoint / "CHECKPOINT.json"),
                    "selection": record["selection"],
                },
            }
            complete = {
                "schema_version": "decision2-laya-joint-research-complete/1",
                "status": "complete",
                "run_sha256": sha(run / "RUN.json"),
                "curve": [point],
            }
            (run / "COMPLETE.json").write_text(json.dumps(complete))
            with patch(
                "training.laya.infer_research.verify_release", return_value=base
            ):
                self.assertEqual(
                    verify_checkpoint(run, checkpoint, root, root)["step"], 2
                )
                (checkpoint / "model.safetensors").write_bytes(b"changed weights")
                with self.assertRaises(ValueError):
                    verify_checkpoint(run, checkpoint, root, root)
                (checkpoint / "model.safetensors").write_bytes(b"fixture weights")
                complete["status"] = "smoke_or_partial"
                (run / "COMPLETE.json").write_text(json.dumps(complete))
                with self.assertRaises(ValueError):
                    verify_checkpoint(run, checkpoint, root, root)
                self.assertEqual(
                    verify_checkpoint(run, checkpoint, root, root, allow_partial=True)[
                        "step"
                    ],
                    2,
                )


if __name__ == "__main__":
    unittest.main()
