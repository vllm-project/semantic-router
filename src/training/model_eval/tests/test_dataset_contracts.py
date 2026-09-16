"""Exercise dataset admission without downloading models or source rows."""

import argparse
import ast
import logging
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

from src.training.model_eval.constants import LEGACY_MODEL_REGISTRY, MODEL_REGISTRY
from src.training.model_eval.dataset_contracts import (
    classification_label_id,
    require_default_dataset,
)

ROOT = Path(__file__).parents[1]


def load_definitions(filename, names, namespace):
    tree = ast.parse((ROOT / filename).read_text())
    selected = [node for node in tree.body if getattr(node, "name", None) in names]
    selected += [
        node
        for node in tree.body
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id in names
    ]
    exec(compile(ast.Module(selected, []), filename, "exec"), namespace)
    return namespace


class Rows:
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def map(self, function):
        return Rows([{**row, **function(row)} for row in self.rows])


class DatasetContractTest(unittest.TestCase):
    def test_legacy_aliases_do_not_become_attack_gold(self):
        legacy = LEGACY_MODEL_REGISTRY["jailbreak"]
        guard = MODEL_REGISTRY["jailbreak"]
        require_default_dataset(legacy)
        with self.assertRaisesRegex(ValueError, "no compatible default"):
            require_default_dataset(guard)
        for label, expected in (("safe", 0), ("unsafe", 1)):
            self.assertEqual(classification_label_id(label, legacy), expected)
            with self.assertRaisesRegex(ValueError, "Unrecognized"):
                classification_label_id(label, guard)
        for label, expected in (("benign", 0), ("jailbreak", 1), (0, 0), (1, 1)):
            self.assertEqual(classification_label_id(label, guard), expected)
        for label in ("unknown", -1, 2, True, 0.5, None):
            with self.assertRaises(ValueError):
                classification_label_id(label, guard)

    def test_guard_loader_requires_custom_data_before_any_network_access(self):
        loader = Mock(return_value=Rows([{"text": "Example", "label": "benign"}]))
        namespace = load_definitions(
            "mom_collection_eval.py",
            {"load_eval_data"},
            {
                "model_registry": lambda collection: MODEL_REGISTRY,
                "logging": logging,
                "Path": Path,
                "Dataset": Rows,
                "require_default_dataset": require_default_dataset,
                "classification_label_id": classification_label_id,
                "load_dataset": loader,
                "retry_operation": lambda fn, **kwargs: fn(),
            },
        )
        load = namespace["load_eval_data"]
        args = SimpleNamespace(
            collection="served", custom_dataset=None, limit=None, max_retries=1
        )
        with self.assertRaisesRegex(ValueError, "no compatible default"):
            load("jailbreak", args)
        loader.assert_not_called()
        args.custom_dataset = "reviewed-attacks.json"
        rows = load("jailbreak", args)
        self.assertEqual(rows.rows, [{"text": "Example", "label": 0}])
        loader.return_value = Rows([{"text": "Example", "label": "unsafe"}])
        with self.assertRaisesRegex(ValueError, "Unrecognized"):
            load("jailbreak", args)

    def test_baseline_blocks_incompatible_gold_before_dataset_resolution(self):
        namespace = load_definitions(
            "baseline_tasks.py",
            {"TaskSpec", "TASK_SPECS"},
            {
                "dataclass": dataclass,
                "LEGACY_MODEL_REGISTRY": LEGACY_MODEL_REGISTRY,
                "BaselineError": ValueError,
            },
        )
        specs = namespace["TASK_SPECS"]
        legacy = LEGACY_MODEL_REGISTRY["jailbreak"]["id"]
        specs["jailbreak"].validate_artifact(legacy)
        for repo in (MODEL_REGISTRY["jailbreak"]["id"], "example/unknown-task-head"):
            with self.assertRaisesRegex(ValueError, "legacy toxicity/jailbreak"):
                specs["jailbreak"].validate_artifact(repo)
        dataset_revision = Mock()
        namespace = load_definitions(
            "quality_baseline.py",
            {"run"},
            {
                "argparse": argparse,
                "Any": Any,
                "logging": logging,
                "torch": Mock(),
                "np": Mock(),
                "load_config": Mock(),
                "served_artifacts": lambda config: {"jailbreak": Mock()},
                "TASK_SPECS": specs,
                "resolve_measured_artifact": lambda *args: SimpleNamespace(
                    repo=MODEL_REGISTRY["jailbreak"]["id"]
                ),
                "resolve_hf_revision": dataset_revision,
            },
        )
        with self.assertRaisesRegex(ValueError, "legacy toxicity/jailbreak"):
            namespace["run"](
                SimpleNamespace(seed=42, config="config.yaml", task="jailbreak")
            )
        dataset_revision.assert_not_called()
