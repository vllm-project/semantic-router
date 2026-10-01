"""Tests for the Transformers remote code shipped in Decision 2.0 packages (stdlib; torch parts skip)."""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from v2.release import automap, card, examples, layout
from v2.release.runtime import api
from v2.release.tests.test_release import ROSTER, TEXT, card_entries, facts

HAS_TORCH = (
    importlib.util.find_spec("torch") is not None
    and importlib.util.find_spec("transformers") is not None
)


class RemoteCodeFilesTest(unittest.TestCase):
    def test_pointer_gains_the_transformers_fields_only_with_remote_code(self):
        args = dict(calibration=None, base=None, max_input_tokens=16384)
        files = ["backbone/config.json", "backbone/model.safetensors"]
        plain = layout.pointer("qwen-full", "DEV2.0-0.8B", files, **args)
        remote = layout.pointer(
            "qwen-full", "DEV2.0-0.8B", files, remote_code=True, **args
        )
        self.assertNotIn("auto_map", plain)
        self.assertEqual({k: remote[k] for k in plain}, plain)
        self.assertEqual(
            {k: remote[k] for k in automap.CONFIG_FIELDS}, automap.CONFIG_FIELDS
        )
        self.assertEqual(remote["model_type"], "decision2")
        self.assertEqual(remote["custom_pipelines"]["decision"]["pt"], ["AutoModel"])

    def test_copy_into_is_byte_identical(self):
        with tempfile.TemporaryDirectory() as scratch:
            records = automap.copy_into(Path(scratch))
            self.assertEqual(set(records), set(automap.FILES))
            for name, record in records.items():
                self.assertEqual(
                    (Path(scratch) / name).read_bytes(),
                    (automap.SOURCE / name).read_bytes(),
                )
                self.assertEqual(record["sha256"], record["source_sha256"])

    def test_auto_map_names_classes_that_exist(self):
        for ref in [
            *automap.CONFIG_FIELDS["auto_map"].values(),
            automap.CONFIG_FIELDS["custom_pipelines"]["decision"]["impl"],
        ]:
            module, name = ref.split(".")
            tree = ast.parse((automap.SOURCE / f"{module}.py").read_text())
            self.assertIn(
                name, {n.name for n in tree.body if isinstance(n, ast.ClassDef)}
            )

    def test_remote_code_imports_only_flat_relative_modules(self):
        # Transformers copies only flat modules named by "from .x import" lines.
        for name in automap.FILES:
            tree = ast.parse((automap.SOURCE / name).read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.level:
                    self.assertEqual(node.level, 1)
                    self.assertIn(f"{node.module}.py", automap.FILES)
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    top = (
                        node.module
                        if isinstance(node, ast.ImportFrom)
                        else node.names[0].name
                    ).split(".")[0]
                    self.assertTrue(
                        top in sys.stdlib_module_names
                        or top in ("torch", "transformers", "huggingface_hub"),
                        f"{name} imports {top}",
                    )


class CardSectionTest(unittest.TestCase):
    def build(self, remote_code, requirements=None):
        scratch = Path(self.enterContext(tempfile.TemporaryDirectory()))
        out = scratch / "pkg"
        details = {**facts(), "profile": "qwen-adapter", "remote_code": remote_code}
        if requirements:
            details["runtime_requirements"] = requirements
        card.build_card(
            entries=card_entries(),
            roster=ROSTER,
            paired=None,
            facts=details,
            text={**TEXT, "staging_notice": "Staging."},
            work=scratch / "work",
            output=out,
        )
        files = {p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()}
        return (out / "README.md").read_text(), files | {
            "LICENSE",
            "NOTICE",
            "ATTRIBUTIONS.md",
        }

    def test_section_with_example_base_note_and_versions(self):
        base = {
            "repo_id": "Qwen/Qwen3.8-27B",
            "revision": "1" * 40,
            "files_sha256": {f"f{i}": "0" * 64 for i in range(28)},
        }
        readme, files = self.build(
            {"tested": ["5.17.0", "5.18.0"], "base": base},
            {"peft": "0.21", "flash-linear-attention": "0.5.2"},
        )
        self.assertIn(card.TRANSFORMERS_HEADING, readme)
        code = examples.card_block(readme, transformers=True)
        compile(code, "card", "exec")
        self.assertIn(
            'AutoModel.from_pretrained("llm-semantic-router/dev2-release-staging", '
            "trust_remote_code=True)",
            code,
        )
        self.assertIn(json.dumps(examples.EXAMPLES[0]["state"]), code)
        with self.assertRaises(ValueError):
            examples.card_block(readme, transformers=False)
        self.assertIn(
            'pip install "transformers>=5.17" torch safetensors peft\n', readme
        )
        self.assertIn(
            "downloads the pinned base [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B)",
            readme,
        )
        self.assertIn("Install `flash-linear-attention` and `causal-conv1d`", readme)
        self.assertIn("CPU inference was not verified.", readme)
        self.assertNotIn('device_map="cpu"', readme)
        self.assertIn("Tested with Transformers 5.17.0 and 5.18.0.", readme)
        self.assertIn('pipeline("decision"', readme)
        self.assertEqual(card.check_rendered(readme, files), [])
        self.assertEqual(
            card.check_rendered(
                readme.replace(card.TRANSFORMERS_HEADING, "### Other"), files
            ),
            [f"missing section: {card.TRANSFORMERS_HEADING}"],
        )

    def test_plain_install_without_peft_or_kernels(self):
        readme, files = self.build({"tested": ["5.17.0"], "base": None})
        self.assertIn('pip install "transformers>=5.17" torch safetensors\n', readme)
        self.assertNotIn("flash-linear-attention", readme)
        self.assertNotIn("pinned base", readme)
        self.assertIn('pass `device_map="cpu"` or `device_map="cuda:1"`', readme)
        self.assertEqual(card.check_rendered(readme, files), [])


class ExamplesTest(unittest.TestCase):
    def test_card_block_selects_native_or_transformers(self):
        readme = (
            "```python\nfrom decision2 import Decision2\n```\n"
            '```python\nAutoModel.from_pretrained("r", trust_remote_code=True)\n```\n'
        )
        self.assertIn("Decision2", examples.card_block(readme, transformers=False))
        self.assertIn("AutoModel", examples.card_block(readme, transformers=True))
        with self.assertRaises(ValueError):
            examples.card_block(readme + "```python\nx = 1\n```\n", transformers=False)

    def test_compare_answer_files(self):
        import argparse

        def write(path, rows):
            path.write_text("".join(json.dumps(r) + "\n" for r in rows))

        base = {
            "q": {
                "type": "choice",
                "choice": "a",
                "probabilities": {"a": 0.7, "b": 0.3},
            },
            "n": {"type": "noul", "noul": 0.9},
        }
        drift = json.loads(json.dumps(base))
        drift["n"]["noul"] = 0.90002
        flipped = json.loads(json.dumps(base))
        flipped["q"] = {
            "type": "choice",
            "choice": "b",
            "probabilities": {"a": 0.3, "b": 0.7},
        }
        with tempfile.TemporaryDirectory() as scratch:
            scratch = Path(scratch)
            left, right = scratch / "l.jsonl", scratch / "r.jsonl"
            write(left, [{"panel": "p", "id": "1", "answers": base}])

            def run(rows, tolerance=1e-4):
                write(right, rows)
                return examples.compare_answer_files(
                    argparse.Namespace(left=left, right=right, tolerance=tolerance)
                )

            same = run([{"panel": "p", "id": "1", "answers": base}])
            self.assertTrue(same["passed"])
            self.assertEqual(same["panels"]["p"]["identical_prompts"], 1)
            self.assertEqual(same["max_abs_drift"], 0.0)
            close = run([{"panel": "p", "id": "1", "answers": drift}])
            self.assertTrue(close["passed"])
            self.assertAlmostEqual(close["max_abs_drift"], 2e-5)
            self.assertEqual(close["panels"]["p"]["identical_prompts"], 0)
            self.assertFalse(
                run([{"panel": "p", "id": "1", "answers": flipped}])["passed"]
            )
            self.assertFalse(
                run([{"panel": "p", "id": "2", "answers": base}])["passed"]
            )


class GateItemTest(unittest.TestCase):
    def test_remote_code_item_needs_every_automap_receipt(self):
        from v2.release import gate

        with tempfile.TemporaryDirectory() as scratch:
            scratch = Path(scratch)
            receipts, package = scratch / "receipts", scratch / "pkg"
            receipts.mkdir()
            package.mkdir()
            (package / "MODEL_MANIFEST.json").write_text(
                json.dumps({"files_sha256": {}})
            )
            self.assertIsNone(gate.remote_code_item(receipts, {}, package))
            (package / "MODEL_MANIFEST.json").write_text(
                json.dumps({"remote_code": {"files": {}}})
            )
            (receipts / "parity-pre.json").write_text("{}")
            names = [
                "automap-pre",
                "automap-card-pre",
                "automap-post",
                "automap-parity-pre",
                "automap-vs-native-pre",
            ]
            steps = {n: {"passed": True} for n in names}
            steps["automap-vs-native-pre"]["max_abs_drift"] = 0.0
            self.assertFalse(gate.remote_code_item(receipts, steps, package)["passed"])
            (receipts / "automap-hub.json").write_text("{}")
            steps["automap-hub"] = {"passed": True}
            item = gate.remote_code_item(receipts, steps, package)
            self.assertTrue(item["passed"])
            self.assertIn("max drift 0", item["evidence"])
            steps["automap-parity-pre"]["passed"] = False
            self.assertFalse(gate.remote_code_item(receipts, steps, package)["passed"])


class RuntimePromptTest(unittest.TestCase):
    def test_remote_code_prompt_refused_at_once_during_load_only(self):
        modules = types.ModuleType("dynamic_module_utils")
        modules.TIME_OUT_REMOTE_CODE = 15
        package = types.ModuleType("transformers")
        package.dynamic_module_utils = modules
        seen = []

        @api._without_remote_code_prompt
        def load(value):
            seen.append(modules.TIME_OUT_REMOTE_CODE)
            if value == "fail":
                raise ValueError(value)
            return value

        with mock.patch.dict(
            sys.modules,
            {"transformers": package, "transformers.dynamic_module_utils": modules},
        ):
            self.assertEqual(load("ok"), "ok")
            with self.assertRaises(ValueError):
                load("fail")
        self.assertEqual(seen, [0, 0])
        self.assertEqual(modules.TIME_OUT_REMOTE_CODE, 15)
        import inspect

        self.assertIn(
            "bf16_resident", inspect.signature(api.Decision2.from_pretrained).parameters
        )


@unittest.skipUnless(HAS_TORCH, "needs torch and transformers")
class ModelingHelpersTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from v2.release.automap import modeling_decision2

        cls.m = modeling_decision2

    def test_devices(self):
        import torch

        self.assertIsNone(self.m._device(None, None))
        self.assertIsNone(self.m._device(None, "auto"))
        self.assertEqual(self.m._device(None, 1), "cuda:1")
        self.assertEqual(self.m._device("cpu", {"": "cpu"}), "cpu")
        self.assertEqual(self.m._device(torch.device("cuda", 0), None), "cuda:0")
        with self.assertRaises(ValueError):
            self.m._device("cpu", "cuda:0")
        with self.assertRaises(ValueError):
            self.m._device(None, {"layer.0": 0, "layer.1": 1})

    def test_link_free_view(self):
        with tempfile.TemporaryDirectory() as scratch:
            scratch = Path(scratch)
            plain = scratch / "plain"
            (plain / "sub").mkdir(parents=True)
            (plain / "sub" / "a.txt").write_text("a")
            self.assertEqual(self.m._link_free(plain), (plain, None))
            blobs, snapshot = (
                scratch / "repo" / "blobs",
                scratch / "repo" / "snapshots" / "c",
            )
            (snapshot / "sub").mkdir(parents=True)
            blobs.mkdir()
            (blobs / "x").write_text("a")
            (snapshot / "sub" / "a.txt").symlink_to(
                os.path.relpath(blobs / "x", snapshot / "sub")
            )
            view, temporary = self.m._link_free(snapshot)
            self.assertEqual(view, temporary)
            self.assertEqual(view.parent, scratch / "repo")
            target = view / "sub" / "a.txt"
            self.assertFalse(target.is_symlink())
            self.assertEqual(target.stat().st_ino, (blobs / "x").stat().st_ino)

    def test_supported_and_accepting(self):
        def f(a, b=1):
            return a, b

        with self.assertRaises(TypeError):
            self.m._supported(f, {"c": 1}, "f")
        self.assertEqual(self.m._accepting(f)(1, b=2, c=3), (1, 2))


if __name__ == "__main__":
    unittest.main()
