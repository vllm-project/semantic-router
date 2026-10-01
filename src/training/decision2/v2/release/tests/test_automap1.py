"""Decision 1.0 remote code: System One contract, staging and code hygiene (stdlib only)."""

from __future__ import annotations

import ast
import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

AUTOMAP = Path(__file__).resolve().parents[1] / "automap"
CODE = AUTOMAP / "decision1"
ALLOWED_IMPORTS = {
    "__future__",
    "copy",
    "dataclasses",
    "functools",
    "inspect",
    "json",
    "math",
    "os",
    "pathlib",
    "sys",
    "types",
    "typing",
    "contextlib",
    "torch",
    "transformers",
    "safetensors",
    "huggingface_hub",
}


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


contract = load(CODE / "decision1_system_one.py", "decision1_system_one")
stage1 = load(AUTOMAP / "stage1.py", "stage1")


class ContractTest(unittest.TestCase):
    def test_noul_defaults_and_explicit_null(self):
        question = contract.validate_question(
            "q",
            {"type": "noul", "instructions": "Is it late?", "criteria": {"true": None}},
        )
        row = contract.build_row(
            "q",
            "state",
            question,
            noul_default_false="no",
            noul_default_true="yes",
            noul_explicit_null="preserve_json_null",
        )
        self.assertEqual(
            [(c.key, c.description) for c in row.candidates],
            [("false", "no"), ("true", None)],
        )
        row = contract.build_row(
            "q",
            "state",
            question,
            noul_default_false="no",
            noul_default_true="yes",
            noul_explicit_null="use_default",
        )
        self.assertEqual(row.candidates[1].description, "yes")

    def test_invalid_questions(self):
        for question in (
            {"type": "choice", "instructions": "x", "criteria": {"only": None}},
            {"type": "score", "instructions": "x", "criteria": ["one"]},
            {"type": "noul", "instructions": " "},
            {"type": "noul", "instructions": "x", "extra": 1},
            {"type": "rank", "instructions": "x"},
            {"type": "noul", "instructions": 3},
        ):
            with self.assertRaises(contract.DecisionInputError):
                contract.validate_question("q", question)
        with self.assertRaises(contract.DecisionInputError):
            contract.validate_request("", {"q": {}})
        with self.assertRaises(contract.DecisionInputError):
            contract.validate_request("state", {})
        self.assertEqual(
            contract.error_answer({"type": "score"}, "invalid_question"),
            {"type": "score", "error": "invalid_question"},
        )

    def test_answers_follow_the_runtime_statistics(self):
        row = contract.build_row(
            "s",
            {"b": 1, "a": "é"},
            contract.validate_question(
                "s",
                {"type": "score", "instructions": "x", "criteria": ["lo", "mid", "hi"]},
            ),
            noul_default_false="no",
            noul_default_true="yes",
            noul_explicit_null="use_default",
        )
        self.assertEqual(row.state, '{"a":"é","b":1}')
        result = contract.answer(row, [0.2, 0.3, 0.5])
        self.assertAlmostEqual(result["score"], 1.3)
        variance = 0.2 * 1.3**2 + 0.3 * 0.3**2 + 0.5 * 0.7**2
        self.assertAlmostEqual(result["confidence"], 1 - variance / ((9 - 1) / 12))
        self.assertEqual(result["legend"], {"0": "lo", "1": "mid", "2": "hi"})
        choice = contract.build_row(
            "c",
            "s",
            contract.validate_question(
                "c",
                {
                    "type": "choice",
                    "instructions": "x",
                    "criteria": {"a": None, "b": "B"},
                },
            ),
            noul_default_false="no",
            noul_default_true="yes",
            noul_explicit_null="use_default",
        )
        result = contract.answer(choice, [0.25, 0.75])
        self.assertEqual((result["choice"], result["confidence"]), ("b", 0.5))
        self.assertEqual(
            contract.answer(choice, [0.5, math.nan])["error"], "invalid_model_output"
        )


class StageTest(unittest.TestCase):
    def test_config_keeps_every_decision_key(self):
        original = (
            '{\n  "decision_format": "vllm-sr-decision",\n  "weights": ["a"]\n}\n'
        )
        rendered = stage1.render_config(original)
        self.assertTrue(rendered.startswith(original.rstrip()[:-1].rstrip()))
        self.assertEqual(
            json.loads(rendered), {**json.loads(original), **stage1.HF_KEYS}
        )
        with self.assertRaises(ValueError):
            stage1.render_config(rendered)

    def test_card_section_and_sentence_updates(self):
        spec = stage1.REPOS["Decision-1.0-Kai-0.6B"]
        card = (
            "# Kai\n\n## Download for local inference\n\n"
            + " ".join(old for old, _ in spec["edits"])
            + "\n\n## Use\n\ntext\n"
        )
        rendered = stage1.render_card(
            card, "llm-semantic-router/Decision-1.0-Kai-0.6B", spec
        )
        self.assertIn(stage1.HEADING, rendered)
        self.assertLess(rendered.index(stage1.HEADING), rendered.index("## Use\n"))
        for old, new in spec["edits"]:
            self.assertNotIn(old, rendered)
            self.assertIn(new, rendered)
        blocks = rendered.split("```python\n")[1].split("```")[0]
        ast.parse(blocks)
        with self.assertRaises(ValueError):
            stage1.render_card("# changed card\n## Use\n", "x/y", spec)

    def test_manifest_update(self):
        with tempfile.TemporaryDirectory() as scratch:
            original = json.dumps(
                {"model": "m", "files": {"b": {"sha256": "0", "bytes": 1}}}
            )
            rendered = json.loads(stage1.render_manifest(original, {"a": b"x"}))
            self.assertEqual(list(rendered["files"]), ["a", "b"])
            self.assertEqual(rendered["files"]["a"]["bytes"], 1)
            self.assertTrue(Path(scratch).is_dir())


class HygieneTest(unittest.TestCase):
    def test_remote_code_files(self):
        self.assertEqual(
            sorted(p.name for p in CODE.glob("*.py")), sorted(stage1.CODE_FILES)
        )
        for path in CODE.glob("*.py"):
            source = path.read_text(encoding="utf-8")
            self.assertIn(
                "SPDX-License-Identifier: Apache-2.0",
                source.splitlines()[1] + source.splitlines()[2],
            )
            for node in ast.walk(ast.parse(source)):
                names = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0:
                    names = [node.module]
                for name in names:
                    self.assertIn(
                        name.split(".")[0], ALLOWED_IMPORTS, f"{path.name}: {name}"
                    )

    def test_config_fields_match_the_files(self):
        auto_map = stage1.HF_KEYS["auto_map"]
        for reference in [
            *auto_map.values(),
            stage1.HF_KEYS["custom_pipelines"]["decision"]["impl"],
        ]:
            module, name = reference.split(".")
            self.assertIn(
                f"class {name}(", (CODE / f"{module}.py").read_text(encoding="utf-8")
            )


if __name__ == "__main__":
    unittest.main()
