"""Validate how the LoRA training scripts express a warmup ratio.

`transformers.TrainingArguments` dropped `warmup_ratio` in 5.15.0, so the ten
scripts that passed it raise `TypeError` on a fresh install. The replacement is
not a different quantity, only a different name: `warmup_steps` became a float
and `get_warmup_steps` reads a value below 1 as a ratio.

The names are not interchangeable, and getting it wrong is silent rather than
loud, so these tests pin which name is emitted for each shape of the dataclass.
`transformers` is stubbed rather than imported: the helper only inspects the
dataclass fields, and these tests must run without the training dependencies.
"""

from __future__ import annotations

import ast
import dataclasses
import sys
import unittest
from pathlib import Path
from typing import ClassVar
from unittest import mock

TRAINING_ROOT = Path(__file__).resolve().parents[1]
HELPER_SOURCE = TRAINING_ROOT / "model_classifier" / "common_lora_utils.py"


def _load_helper():
    """Import the helper without importing torch.

    `common_lora_utils` imports torch at module scope for its GPU utilities,
    which this test neither needs nor should require in CI, so only the module
    prologue up to the first unrelated function is executed.
    """
    source = HELPER_SOURCE.read_text(encoding="utf-8")
    prologue = source.split("def get_target_modules_for_model")[0].replace(
        "import torch", ""
    )
    namespace: dict = {}
    exec(compile(prologue, str(HELPER_SOURCE), "exec"), namespace)
    return namespace["warmup_kwargs"]


def _stub_transformers(*field_names: str):
    """A fake `transformers` module whose TrainingArguments has these fields.

    Built with `make_dataclass` rather than by hand: `dataclasses.fields()`
    filters on each field's private `_field_type`, so a dict of bare
    `dataclasses.field()` objects reads back as having no fields at all — which
    would make every assertion here pass for the wrong reason.
    """
    training_arguments = dataclasses.make_dataclass(
        "TrainingArguments", [(name, float, 0.0) for name in field_names]
    )
    module = type(sys)("transformers")
    module.TrainingArguments = training_arguments
    return mock.patch.dict(sys.modules, {"transformers": module})


class WarmupKwargsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.warmup_kwargs = staticmethod(_load_helper())

    def test_transformers_4x_still_gets_warmup_ratio(self) -> None:
        """4.x keeps warmup_steps as an int, so the ratio must go to warmup_ratio."""
        with _stub_transformers("warmup_ratio", "warmup_steps", "learning_rate"):
            self.assertEqual(self.warmup_kwargs(0.06), {"warmup_ratio": 0.06})

    def test_transformers_515_gets_warmup_steps(self) -> None:
        """5.15 removed warmup_ratio; warmup_steps below 1 is read as the ratio."""
        with _stub_transformers("warmup_steps", "learning_rate"):
            self.assertEqual(self.warmup_kwargs(0.06), {"warmup_steps": 0.06})

    def test_the_ratio_is_passed_through_untouched(self) -> None:
        """No arithmetic happens here; the Trainer resolves it against its own
        step count, which is the only place the world size, a partial gradient
        accumulation group and the accelerator's batch are all known."""
        for ratio in (0.0, 0.06, 0.1, 0.5, 1.0):
            with self.subTest(ratio=ratio), _stub_transformers("warmup_steps"):
                self.assertEqual(list(self.warmup_kwargs(ratio).values()), [ratio])

    def test_exactly_one_keyword_is_emitted(self) -> None:
        """Passing both names would be a TypeError on 4.x and unknown on 5.x."""
        for fields in (("warmup_ratio", "warmup_steps"), ("warmup_steps",)):
            with self.subTest(fields=fields), _stub_transformers(*fields):
                self.assertEqual(len(self.warmup_kwargs(0.06)), 1)


class NoScriptStillPassesRemovedArgumentsTest(unittest.TestCase):
    """Pin every call site, so a future edit cannot quietly reintroduce one.

    Matching source text would miss a rename or a reformat, so the check walks
    the AST and looks for the shape: a `TrainingArguments(...)` call carrying a
    keyword `transformers` no longer accepts.
    """

    REMOVED: ClassVar[set[str]] = {"warmup_ratio", "logging_dir"}

    def test_no_training_arguments_call_uses_a_removed_keyword(self) -> None:
        offenders = []
        for path in sorted(TRAINING_ROOT.rglob("*.py")):
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeDecodeError):
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                name = (
                    func.attr
                    if isinstance(func, ast.Attribute)
                    else getattr(func, "id", "")
                )
                if name != "TrainingArguments":
                    continue
                used = {k.arg for k in node.keywords if k.arg in self.REMOVED}
                if used:
                    offenders.append(
                        f"{path.relative_to(TRAINING_ROOT)}:{node.lineno} "
                        f"passes {sorted(used)}"
                    )
        self.assertEqual(
            offenders,
            [],
            "transformers 5.15+ rejects these keywords:\n" + "\n".join(offenders),
        )


class EveryCallSiteGoesThroughTheHelperTest(unittest.TestCase):
    """A script that hard-codes either spelling is correct on one version only."""

    def test_no_training_arguments_call_hardcodes_a_warmup_spelling(self) -> None:
        offenders = []
        for path in sorted(TRAINING_ROOT.rglob("*.py")):
            if path.name.startswith("test_"):
                continue
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeDecodeError):
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                name = (
                    func.attr
                    if isinstance(func, ast.Attribute)
                    else getattr(func, "id", "")
                )
                if name != "TrainingArguments":
                    continue
                for kw in node.keywords:
                    if kw.arg == "warmup_steps":
                        offenders.append(
                            f"{path.relative_to(TRAINING_ROOT)}:{node.lineno} "
                            f"hard-codes warmup_steps; use **warmup_kwargs(ratio)"
                        )
        self.assertEqual(offenders, [], "\n".join(offenders))


class ScriptsImportTheHelperResolvablyTest(unittest.TestCase):
    """Every script importing common_lora_utils must be launchable directly.

    Python puts only the script's own directory on sys.path for a
    `python path/to/script.py` launch, and common_lora_utils lives one level
    above the training scripts, so a bare import raises ModuleNotFoundError
    before argument parsing.
    """

    def test_every_importer_sets_up_the_parent_path(self) -> None:
        offenders = []
        for path in sorted(TRAINING_ROOT.rglob("*.py")):
            if path.name in ("common_lora_utils.py",) or path.name.startswith("test_"):
                continue
            text = path.read_text(encoding="utf-8")
            if "from common_lora_utils import" not in text:
                continue
            # append() and insert(0, ...) are both in use in this tree; what
            # matters is that the parent directory is on the path at all, not
            # which call put it there.
            if "sys.path.append" not in text and "sys.path.insert" not in text:
                offenders.append(
                    f"{path.relative_to(TRAINING_ROOT)} imports common_lora_utils "
                    f"without adding its parent directory to sys.path"
                )
        self.assertEqual(offenders, [], "\n".join(offenders))


if __name__ == "__main__":
    unittest.main()
