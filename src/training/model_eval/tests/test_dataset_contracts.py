"""Exercise dataset admission without downloading models or source rows."""

import argparse
import ast
import importlib
import importlib.util
import logging
import sys
import unittest
from collections.abc import Sequence
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
DOMAIN_SUBSET = {"math": 0, "physics": 1, "law": 2}
# Every law row here comes from MMLU, so the filtered split has none.
PUBLISHED_DOMAIN_ROWS = [
    {"question": "a", "category": "math", "src": "ori_mmlu-algebra"},
    {"question": "b", "category": "physics", "src": "scibench"},
    {"question": "c", "category": "law", "src": "ori_mmlu-jurisprudence"},
    {"question": "d", "category": "math", "src": "theoremqa"},
]
GAP_BUDGETS = {
    "ECE_BUDGET": 0.05,
    "THRESHOLD_VALUE_BUDGET": 0.05,
    "MIN_LABEL_RECALL": 0.7,
    "MAX_SLICE_SPREAD": 0.1,
    "MIN_COMPARISON_SIZE": 2,
    "MIN_SLICE_ROWS": 30,
    "MIN_GATE_RECALL": 0.7,
}


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


def domain_baseline():
    class Split(list):
        def filter(self, keep):
            return Split(row for row in self if keep(row))

    return load_definitions(
        "baseline_tasks.py",
        {"TaskSpec", "TASK_SPECS", "load_rows"},
        {
            "dataclass": dataclass,
            "LEGACY_MODEL_REGISTRY": LEGACY_MODEL_REGISTRY,
            "BaselineError": ValueError,
            "load_dataset": lambda repo, split: Split(PUBLISHED_DOMAIN_ROWS),
            "np": SimpleNamespace(
                array=lambda values, dtype: values, int64=int, ndarray=list
            ),
            "logger": logging.getLogger("test"),
            "MAX_REPORTED_UNMAPPED": 10,
        },
    )


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

    def test_domain_baseline_leaves_out_mmlu_rows_and_the_model_trained_on_them(self):
        namespace = domain_baseline()
        spec = namespace["TASK_SPECS"]["domain"]
        self.assertEqual(spec.split_rule, "by_source")
        texts, labels, available = namespace["load_rows"](spec, DOMAIN_SUBSET, None)
        self.assertEqual((texts, labels, available), (["b", "d"], [1, 0], 2))
        spec.validate_artifact(MODEL_REGISTRY["intent"]["id"])
        for key in ("id", "lora_id"):
            with self.assertRaisesRegex(
                ValueError, "was trained on TIGER-Lab/MMLU-Pro"
            ):
                spec.validate_artifact(LEGACY_MODEL_REGISTRY["intent"][key])

    def test_filtered_domain_report_leaves_labels_without_rows_unmeasured(self):
        namespace = domain_baseline()
        _, labels, _ = namespace["load_rows"](
            namespace["TASK_SPECS"]["domain"], DOMAIN_SUBSET, None
        )
        # Perfect predictions on the rows the filter keeps.
        metrics = load_definitions(
            "provenance/metrics.py",
            {"classification_metrics"},
            {"Any": Any, "Sequence": Sequence},
        )["classification_metrics"](labels, labels, DOMAIN_SUBSET)
        self.assertEqual(metrics["per_label"]["law"]["support"], 0)
        self.assertEqual(metrics["macro_f1"], 1.0)
        report = load_definitions(
            "gap_report.py",
            {"_baseline_findings", "_threshold_findings"},
            {"Any": Any, **GAP_BUDGETS},
        )
        findings = report["_baseline_findings"](
            {
                "task": "domain",
                "artifact": {"repo": MODEL_REGISTRY["intent"]["id"]},
                "metrics": metrics,
                "calibration": {"ece": 0.0, "mce": 0.0},
                "abstention": {"curve": []},
                "slices": [],
            }
        )
        self.assertEqual([kind for kind, _ in findings], ["coverage"])
        self.assertIn("`law`", findings[0][1])
        self.assertIn("unmeasured", findings[0][1])

    @unittest.skipUnless(
        importlib.util.find_spec("torch") and importlib.util.find_spec("sklearn"),
        "needs the training test requirements",
    )
    def test_baseline_summary_lists_labels_the_split_has_no_rows_for(self):
        sys.path.insert(0, str(ROOT))
        numpy = importlib.import_module("numpy")
        summary = importlib.import_module("baseline_metrics").summarise(
            numpy.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.8, 0.1]]),
            numpy.array([0, 1, 1]),
            ["short", "a longer question", "another question"],
            DOMAIN_SUBSET,
            [1.0, 1.0, 1.0],
            10,
            (0.0, 0.5),
            0.0,
        )
        self.assertEqual(summary["unsupported_labels"], ["law"])
        self.assertEqual(summary["metrics"]["macro_f1"], 1.0)
        self.assertEqual(summary["metrics"]["per_label"]["law"]["support"], 0)

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
