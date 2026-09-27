import json
import unittest
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from training.eikos.verify_export import (
    PUBLICATION_DOCUMENTS,
    compare_answers,
    verify,
    verify_sums,
)


class PackagedAnswerComparisonTest(unittest.TestCase):
    def test_choice_and_probability_drift_are_separate(self):
        expected = {
            "type": "choice",
            "choice": "a",
            "probabilities": {"a": 0.5001, "b": 0.4999},
        }
        actual = {
            "type": "choice",
            "choice": "b",
            "probabilities": {"a": 0.4998, "b": 0.5002},
        }
        same, drift, pmax_drift = compare_answers(expected, actual)
        self.assertFalse(same)
        self.assertAlmostEqual(drift, 0.0003)
        self.assertAlmostEqual(pmax_drift, 0.0001)

    def test_noul_boolean_decision_is_checked(self):
        expected = {"type": "noul", "value": True, "probability": 0.501}
        actual = {"type": "noul", "value": False, "probability": 0.499}
        same, drift, pmax_drift = compare_answers(expected, actual)
        self.assertFalse(same)
        self.assertAlmostEqual(drift, 0.002)
        self.assertAlmostEqual(pmax_drift, 0)

    def test_score_uses_modal_level_not_continuous_expectation(self):
        expected = {
            "type": "score",
            "native_score": 2,
            "score": 1.51,
            "probabilities": {"1": 0.49, "2": 0.51},
        }
        actual = {
            "type": "score",
            "native_score": 2,
            "score": 1.52,
            "probabilities": {"1": 0.48, "2": 0.52},
        }
        same, drift, pmax_drift = compare_answers(expected, actual)
        self.assertTrue(same)
        self.assertAlmostEqual(drift, 0.01)
        self.assertAlmostEqual(pmax_drift, 0.01)


class FunctionalPackageRosterTest(unittest.TestCase):
    def test_publication_documents_do_not_change_model_identity(self):
        with TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder / "config.json").write_text("{}", encoding="utf-8")
            digest = sha256(b"{}").hexdigest()
            (folder / "SHA256SUMS").write_text(
                f"{digest}  config.json\n", encoding="utf-8"
            )
            model_sha = sha256((folder / "SHA256SUMS").read_bytes()).hexdigest()
            self.assertEqual(verify_sums(folder), 1)
            for name in PUBLICATION_DOCUMENTS:
                path = folder / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("publication", encoding="utf-8")
            self.assertEqual(verify_sums(folder), 1)
            self.assertEqual(
                sha256((folder / "SHA256SUMS").read_bytes()).hexdigest(), model_sha
            )
            (folder / "unlisted.py").write_text("pass", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "file set differs"):
                verify_sums(folder)


class TorchReferenceDirectParityTest(unittest.TestCase):
    def test_switch_precedes_both_loads_and_hashes_both_predictions(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            merged = root / "merged"
            merged.mkdir()
            (merged / "calib.json").write_text("{}", encoding="utf-8")
            (merged / "SHA256SUMS").write_text("package", encoding="utf-8")
            calibration_sha = sha256(b"{}").hexdigest()
            adapter_sha = "a" * 64
            (merged / "decision2_provenance.json").write_text(
                json.dumps(
                    {
                        "source_revision": "582ffb13f19a4da3f455e3db198584190bd7755b",
                        "source_release": {},
                        "selected_checkpoint": "checkpoint-0232",
                        "adapter_weights_sha256": adapter_sha,
                        "calibration_sha256": calibration_sha,
                    }
                ),
                encoding="utf-8",
            )
            selection = {
                "name": "checkpoint-0232",
                "adapter": root / "adapter",
                "adapter_weights_sha256": adapter_sha,
            }
            row = {
                "id": "item/1",
                "state": "state",
                "questions": {"label": {"type": "choice"}},
            }
            answer = {
                "type": "choice",
                "choice": "A",
                "probabilities": {"A": 0.75, "B": 0.25},
            }
            events = []

            class FakeDecider:
                def decide_all(self, **kwargs):
                    return {"label": (answer, 4)}

            def load(*args, **kwargs):
                events.append("load")
                return FakeDecider()

            def switch():
                events.append("switch")
                return {"gated_delta_backend": "torch-reference"}

            selected = root / "selected.jsonl"
            packaged = root / "packaged.jsonl"
            output = root / "parity.json"
            prompts = root / "unused.prompts.jsonl"
            prompts.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with (
                patch("training.eikos.verify_export.verify_sums", return_value=1),
                patch(
                    "training.eikos.verify_export.selected_checkpoint",
                    return_value=selection,
                ),
                patch("training.eikos.verify_export.load_prompts", return_value=[row]),
                patch("training.eikos.verify_export.load_decider", side_effect=load),
                patch(
                    "training.eikos.verify_export.shared_answer",
                    side_effect=lambda q, a: a,
                ),
                patch(
                    "torch.use_deterministic_algorithms",
                    side_effect=lambda enabled: events.append("deterministic"),
                ),
                patch("torch.are_deterministic_algorithms_enabled", return_value=True),
                patch(
                    "training.eikos.published_infer.use_torch_reference_gated_delta",
                    side_effect=switch,
                ),
            ):
                report = verify(
                    model_path=root,
                    run=root,
                    merged=merged,
                    prompts=prompts,
                    reference=None,
                    output=output,
                    direct_selected=True,
                    deterministic_algorithms=True,
                    torch_reference_gated_delta=True,
                    selected_predictions=selected,
                    merged_predictions=packaged,
                )
            self.assertEqual(events, ["deterministic", "switch", "load", "load"])
            self.assertTrue(report["predeclared_gate"]["pass"])
            self.assertTrue(report["runtime"]["torch_deterministic_algorithms"])
            self.assertEqual(
                report["runtime"]["gated_delta_backend"], "torch-reference"
            )
            self.assertEqual(
                report["selected_predictions_sha256"],
                sha256(selected.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                report["merged_predictions_sha256"],
                sha256(packaged.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                json.loads(selected.read_text().splitlines()[0])["answers"]["label"],
                answer,
            )


if __name__ == "__main__":
    unittest.main()
