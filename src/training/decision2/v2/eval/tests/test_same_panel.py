from __future__ import annotations

import argparse
import json
import math
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from benchmark.generate import generate, prompt_record
from transfer.build import EVALUATION_TASKS, PANEL_VERSION
from v2.eval import adapters, dev_readout, panels, same_panel


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def perfect_answer(question: dict, gold: dict) -> dict:
    kind, value = question["type"], gold["value"]
    if kind == "choice":
        return {
            "type": "choice",
            "choice": value,
            "probabilities": {
                label: float(label == value) for label in question["criteria"]
            },
        }
    if kind == "noul":
        return {"type": "noul", "noul": 1.0 if value else 0.0}
    levels = range(len(question["criteria"]))
    return {
        "type": "score",
        "score": float(value),
        "probabilities": {str(level): float(level == value) for level in levels},
    }


class PanelFixture:
    """Synthetic typed FINAL + 15-task CSS panel registered under a temp root."""

    def __init__(self, root: Path, long_state: bool = False) -> None:
        self.root = root
        self.typed = generate("final", b"dev2-eval-test", groups_per_family=1)
        self.css_prompts, self.css_gold = [], []
        for index, task in enumerate(EVALUATION_TASKS):
            for item, gold in enumerate(("yes", "no")):
                state = (
                    ("word " * 1200)
                    if (long_state and index == 0 and item == 0)
                    else f"text {task} {item}"
                )
                questions = {
                    "label": {
                        "type": "choice",
                        "instructions": "pick",
                        "criteria": {"yes": "y", "no": "n"},
                    }
                }
                item_id = f"{task}-{item}"
                self.css_prompts.append(
                    {"id": item_id, "state": state, "questions": questions}
                )
                self.css_gold.append(
                    {
                        "id": item_id,
                        "panel_version": PANEL_VERSION,
                        "task": task,
                        "role": "evaluation",
                        "labels": ["yes", "no"],
                        "gold": gold,
                        "input_sha256": same_panel.input_digest(state, questions),
                    }
                )
        files = {
            "goldfree/typed-final.prompts.jsonl": [
                prompt_record(row) for row in self.typed
            ],
            "gold/typed-final.gold.jsonl": self.typed,
            "goldfree/css15.prompts.jsonl": self.css_prompts,
            "gold/css15.gold.jsonl": self.css_gold,
        }
        self.registry = {name: dict(spec) for name, spec in panels.ALL.items()}
        for relative, rows in files.items():
            write_jsonl(root / relative, rows)
        for name in ("typed-final", "css15"):
            spec = self.registry[name]
            spec["prompts_sha256"] = panels.sha_file(root / spec["prompts"])
            spec["gold_sha256"] = panels.sha_file(root / spec["gold"])
            spec["originals"] = len(
                (self.typed if name == "typed-final" else self.css_prompts)
            )

    def patch(self):
        return mock.patch.multiple(
            panels,
            ALL=self.registry,
            FORMAL={k: self.registry[k] for k in panels.FORMAL},
        )

    def predictions(
        self, run_dir: Path, wrong_css: int = 0, invalid_typed: int = 0
    ) -> None:
        typed_rows = []
        for index, item in enumerate(self.typed):
            answers = {
                key: perfect_answer(q, item["gold"][key])
                for key, q in item["questions"].items()
            }
            if index < invalid_typed:
                answers = {key: None for key in answers}
            typed_rows.append(
                {
                    "id": item["id"],
                    "answers": answers,
                    "latency_ms": 10.0 + index,
                    "source_input_sha256": same_panel.input_digest(
                        item["state"], item["questions"]
                    ),
                    "model_id": "test/model",
                    "model_revision": "r1",
                }
            )
        css_rows = []
        for index, (prompt, gold) in enumerate(zip(self.css_prompts, self.css_gold)):
            choice = (
                gold["gold"]
                if index >= wrong_css
                else ("no" if gold["gold"] == "yes" else "yes")
            )
            css_rows.append(
                {
                    "id": prompt["id"],
                    "latency_ms": 5.0,
                    "answers": {
                        "label": {
                            "choice": choice,
                            "probabilities": {
                                "yes": float(choice == "yes"),
                                "no": float(choice == "no"),
                            },
                        }
                    },
                    "source_input_sha256": gold["input_sha256"],
                    "model_id": "test/model",
                    "model_revision": "r1",
                }
            )
        write_jsonl(same_panel.prediction_path(run_dir, "typed-final"), typed_rows)
        write_jsonl(same_panel.prediction_path(run_dir, "css15"), css_rows)
        (run_dir / "ADOPT.json").write_text(
            json.dumps({"schema": "dev2-same-panel-adopt/1"})
        )


def report_args(run_dir: Path, root: Path) -> argparse.Namespace:
    return argparse.Namespace(
        run_dir=run_dir,
        panel_root=root,
        label="Test",
        tier="0.6B",
        family="test",
        model_id=None,
        revision=None,
        loaded_parameters=123,
        active_parameters=None,
        parameter_source="test",
        count_safetensors=None,
        batch_policy=None,
    )


class SamePanelTests(unittest.TestCase):
    def test_adapter_commands(self) -> None:
        argv = adapters.REGISTRY["decision1-lux"].command(
            {
                "model": "/m",
                "revision": "abc",
                "input": "/i",
                "output": "/o",
                "device": "cuda:0",
            }
        )
        self.assertEqual(argv[:3], ["python3", "-m", "inference.run"])
        self.assertIn("--over-budget-invalid", argv)
        with self.assertRaises(ValueError):
            adapters.REGISTRY["jpt"].command(
                {"model": "/m", "revision": "r", "input": "i", "output": "o"}
            )
        self.assertTrue(adapters.REGISTRY["kai"].python.endswith("kai-lex/bin/python"))

    def test_perfect_run_scores_100_and_seal_is_gold_free(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, run_dir = Path(tmp) / "panels", Path(tmp) / "run"
            fixture = PanelFixture(root)
            with fixture.patch():
                fixture.predictions(run_dir)
                same_panel.seal(argparse.Namespace(run_dir=run_dir, panel_root=root))
                sealed = json.loads((run_dir / "SEAL.json").read_text())
                self.assertFalse(sealed["gold_or_scores_read"])
                self.assertEqual(
                    sealed["panels"]["typed-final"]["missing_originals"], 0
                )
                same_panel.report(report_args(run_dir, root))
            result = json.loads((run_dir / "REPORT.json").read_text())
            self.assertEqual(result["label"], "post-key same-panel")
            self.assertAlmostEqual(result["v3"]["score"], 100.0)
            self.assertEqual(result["invalid"]["typed-final"]["invalid_or_missing"], 0)
            self.assertEqual(result["slices"]["css15"]["long"]["n"], 0)
            self.assertFalse(result["slices"]["multilingual_measurable"])
            self.assertTrue(result["reused"])
            self.assertIn("| Test | 0.6B |", (run_dir / "REPORT.md").read_text())

    def test_errors_invalids_and_long_slice_stay_in_denominators(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, run_dir = Path(tmp) / "panels", Path(tmp) / "run"
            fixture = PanelFixture(root, long_state=True)
            with fixture.patch():
                fixture.predictions(run_dir, wrong_css=1, invalid_typed=1)
                same_panel.seal(argparse.Namespace(run_dir=run_dir, panel_root=root))
                same_panel.report(report_args(run_dir, root))
            result = json.loads((run_dir / "REPORT.json").read_text())
            invalid_slots = len(fixture.typed[0]["questions"])
            self.assertEqual(
                result["invalid"]["typed-final"]["invalid_or_missing"], invalid_slots
            )
            self.assertLess(result["v3"]["T"], 1.0)
            self.assertEqual(
                result["slices"]["css15"]["long"],
                {"n": 1, "correct": 0, "invalid": 0, "accuracy_all": 0.0},
            )
            expected = 100 * math.sqrt(result["v3"]["T"] * result["v3"]["H"])
            self.assertAlmostEqual(result["v3"]["score"], expected)

    def test_seal_rejects_changed_input(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root, run_dir = Path(tmp) / "panels", Path(tmp) / "run"
            fixture = PanelFixture(root)
            with fixture.patch():
                fixture.predictions(run_dir)
                path = same_panel.prediction_path(run_dir, "css15")
                rows = same_panel.read_jsonl(path)
                rows[0]["source_input_sha256"] = "0" * 64
                write_jsonl(path, rows)
                with self.assertRaises(ValueError):
                    same_panel.seal(
                        argparse.Namespace(run_dir=run_dir, panel_root=root)
                    )

    def test_count_safetensors_reads_headers_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            header = json.dumps(
                {
                    "__metadata__": {},
                    "a": {"dtype": "BF16", "shape": [3, 4], "data_offsets": [0, 24]},
                    "b": {"dtype": "F32", "shape": [5], "data_offsets": [24, 44]},
                }
            ).encode()
            (Path(tmp) / "model.safetensors").write_bytes(
                struct.pack("<Q", len(header)) + header + bytes(44)
            )
            self.assertEqual(same_panel.count_safetensors(Path(tmp))["parameters"], 17)

    def test_language_screen(self) -> None:
        english = {
            "state": "The model reads the text and it is not sure that this is the best answer for you",
            "questions": {},
        }
        spanish = {
            "state": "El modelo lee el texto y no es seguro que la respuesta sea la mejor para los que usan una con",
            "questions": {},
        }
        chinese = {"state": "这是一个中文的问题，请选择最好的答案", "questions": {}}
        self.assertEqual(same_panel.language_guess(english), "en")
        self.assertEqual(same_panel.language_guess(spanish), "es")
        self.assertEqual(same_panel.language_guess(chinese), "non-latin")

    def test_select_readout_full_denominator(self) -> None:
        rows = [
            {
                "id": "a",
                "options": ["x", "y"],
                "label": 0,
                "family": "f1",
                "task_type": "choice",
            },
            {
                "id": "b",
                "options": ["x", "y"],
                "label": 1,
                "family": "f1",
                "task_type": "choice",
            },
            {
                "id": "c",
                "options": ["x", "y", "z"],
                "label": 2,
                "family": "f2",
                "task_type": "score",
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "select.jsonl"
            write_jsonl(
                path,
                [
                    {"id": "a", "probabilities": [0.9, 0.1]},
                    {"id": "b", "probabilities": [0.5, 0.5]},
                ],
            )
            summary = dev_readout.select_summary(rows, path)
        self.assertEqual(summary["correct"], 1)
        self.assertEqual(summary["invalid_or_missing"], 1)
        self.assertAlmostEqual(summary["by_family"]["f2"]["brier"], 1.0)
        self.assertAlmostEqual(summary["family_macro_accuracy"], 0.25)


if __name__ == "__main__":
    unittest.main()
