"""Gold-free full-panel publication parity contracts."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

from publication import bundle_arena, package_native_arena, panel_parity


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, separators=(",", ":")) + "\n", encoding="utf-8")


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PanelParityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        os.chmod(self.root, 0o700)
        self.model = "a" * 64
        self.calibration = "b" * 64
        self.package_sha = "c" * 64
        self.prompt = {
            "id": "one",
            "state": "The case is pending.",
            "questions": {
                "choice": {
                    "type": "choice",
                    "criteria": {"hold": "wait", "send": "dispatch"},
                },
                "noul": {"type": "noul", "criteria": {"false": "No", "true": "Yes"}},
                "score": {"type": "score", "criteria": ["Low", "High"]},
            },
        }
        self.answers = {
            "choice": {
                "type": "choice",
                "choice": "hold",
                "probabilities": {"hold": 0.8, "send": 0.2},
            },
            "noul": {"type": "noul", "noul": 0.4},
            "score": {
                "type": "score",
                "score": 0.7,
                "probabilities": {"0": 0.3, "1": 0.7},
            },
        }

    def _panel(
        self,
        name: str,
        *,
        count=1,
        single_question=False,
        source_answers=None,
        package_answers=None,
    ):
        prompts = self.root / f"{name}.prompts.jsonl"
        prompt_rows = [
            {
                **self.prompt,
                "id": f"{name}-{index}",
                "questions": (
                    {"choice": self.prompt["questions"]["choice"]}
                    if single_question
                    else self.prompt["questions"]
                ),
            }
            for index in range(count)
        ]
        prompts.write_text(
            "".join(json.dumps(row) + "\n" for row in prompt_rows), encoding="utf-8"
        )
        paths = []
        for role, answers in (
            ("source", source_answers or self.answers),
            ("package", package_answers or self.answers),
        ):
            path = self.root / f"{name}.{role}.jsonl"
            path.write_text(
                "".join(
                    json.dumps(
                        {
                            "id": prompt["id"],
                            "source_input_sha256": package_native_arena.input_digest(
                                prompt
                            ),
                            "answers": (
                                {"choice": answers["choice"]}
                                if single_question
                                else answers
                            ),
                        }
                    )
                    + "\n"
                    for prompt in prompt_rows
                ),
                encoding="utf-8",
            )
            _write_json(
                path.with_name(path.name + ".manifest.json"),
                {
                    "predictions_sha256": _digest(path),
                    "input_sha256": _digest(prompts),
                    "input_items": count,
                    "model_sha256": self.model,
                    "calibration_sha256": self.calibration,
                    **(
                        {"package_manifest_sha256": self.package_sha}
                        if role == "package"
                        else {}
                    ),
                },
            )
            paths.append(path)
        return prompts, *paths

    def _compare(self, panel, name="dev"):
        return panel_parity.compare_panel(
            name=name,
            prompt_path=panel[0],
            source_path=panel[1],
            package_path=panel[2],
            model_sha256=self.model,
            calibration_sha256=self.calibration,
            package_sha256=self.package_sha,
            expected_items=1,
        )

    def test_matching_answers_pass_and_bind_input_and_model(self) -> None:
        panel = self._panel("dev")
        report = self._compare(panel)
        self.assertTrue(report["gate_pass"])
        self.assertEqual(report["answers"], 3)
        self.assertEqual(report["numeric_comparisons"], 6)
        manifest = panel[2].with_name(panel[2].name + ".manifest.json")
        data = json.loads(manifest.read_text())
        data["calibration_sha256"] = "f" * 64
        _write_json(manifest, data)
        with self.assertRaisesRegex(ValueError, "does not bind"):
            self._compare(panel)

    def test_boundary_point_flips_fail_despite_small_probability_drift(self) -> None:
        left = json.loads(json.dumps(self.answers))
        right = json.loads(json.dumps(self.answers))
        left["noul"]["noul"] = 0.5
        right["noul"]["noul"] = 0.500001
        left["score"] = {
            "type": "score",
            "score": 0.5,
            "probabilities": {"0": 0.5, "1": 0.5},
        }
        right["score"] = {
            "type": "score",
            "score": 0.499999,
            "probabilities": {"0": 0.500001, "1": 0.499999},
        }
        report = self._compare(
            self._panel("dev", source_answers=left, package_answers=right)
        )
        self.assertEqual(report["categorical_mismatch_n"], 2)
        self.assertLess(report["probability_drift_max"], 0.005)
        self.assertFalse(report["gate_pass"])

    def test_malformed_outputs_fail_even_when_both_sides_match(self) -> None:
        malformed = json.loads(json.dumps(self.answers))
        malformed["choice"]["choice"] = "not_offered"
        report = self._compare(
            self._panel("dev", source_answers=malformed, package_answers=malformed)
        )
        self.assertEqual(report["malformed_pair_n"], 1)
        self.assertFalse(report["gate_pass"])

    def test_duplicate_prediction_keys_cannot_hide_an_answer(self) -> None:
        path = self.root / "ambiguous.jsonl"
        path.write_text(
            '{"id":"one","answers":{},"answers":{"choice":{"choice":"hold"}}}\n',
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "duplicate key"):
            panel_parity._rows(path)

    def test_generated_receipt_matches_release_bundle_contract(self) -> None:
        package_dir = self.root / "bundle"
        package_dir.mkdir()
        package_manifest = package_dir / "MODEL_MANIFEST.json"
        _write_json(
            package_manifest,
            {
                "bundle_version": package_native_arena.PACKAGE_VERSION,
                "model_id": "llm-semantic-router/dev-2.0-27b",
                "model_sha256": self.model,
                "calibration_sha256": self.calibration,
                "max_length": 4096,
                "temperature_by_type": {"choice": 1.0, "noul": 1.0, "score": 1.0},
                "loader_files_sha256": {"infer.py": "d" * 64},
                "model_files_sha256": {"head": "e" * 64},
                "base": {
                    "repo_id": "Qwen/Qwen3.8-27B",
                    "revision": "f" * 40,
                    "files_sha256": {"config.json": "1" * 64},
                },
                "dependencies": {"torch": "2.12.0", "peft": "0.21.0"},
            },
        )
        self.package_sha = _digest(package_manifest)
        panels = {
            "dev": self._panel("dev", count=1600, single_question=True),
            "css_pilot": self._panel("css", count=1430, single_question=True),
        }
        receipt = panel_parity.build_receipt(
            package_manifest_path=package_manifest,
            panels=panels,
            output_dir=self.root,
        )
        self.assertEqual(receipt["status"], "passed")
        bundle_arena._parity(
            receipt,
            receipt["model_id"],
            receipt["model_revision"],
            self.model,
            receipt["model_files_sha256"],
            self.calibration,
        )
        self.assertEqual(
            _digest(self.root / "dev-parity-detail.json"),
            receipt["panels"]["dev"]["report_sha256"],
        )


if __name__ == "__main__":
    unittest.main()
