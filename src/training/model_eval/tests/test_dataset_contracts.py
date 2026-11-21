"""Exercise dataset admission without downloading models or source rows."""

import argparse
import ast
import contextlib
import importlib
import importlib.util
import io
import json
import logging
import math
import random
import sys
import tempfile
import unittest
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

from src.training.model_eval import gap_report
from src.training.model_eval.artifact_inventory import (
    REGISTRY_ALIASES,
    load_config,
    served_artifacts,
)
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


def label_order_check():
    return load_definitions(
        "baseline_tasks.py",
        {"check_registry_label_order"},
        {
            "REGISTRY_ALIASES": REGISTRY_ALIASES,
            "LEGACY_MODEL_REGISTRY": LEGACY_MODEL_REGISTRY,
            "MODEL_REGISTRY": MODEL_REGISTRY,
        },
    )["check_registry_label_order"]


def legacy_mapping(task):
    return {
        label: index
        for index, label in enumerate(LEGACY_MODEL_REGISTRY[task]["labels"])
    }


class Split(list):
    def filter(self, keep):
        return Split(row for row in self if keep(row))

    def select(self, indices):
        return Split(self[index] for index in indices)

    def select_columns(self, names):
        return Split({name: row[name] for name in names} for row in self)


def baseline_loader(load_dataset):
    return load_definitions(
        "baseline_tasks.py",
        {
            "TaskSpec",
            "TASK_SPECS",
            "load_split",
            "load_rows",
            "select_matched_rows",
            "stratum_of",
            "MATCHED_SELECTION_SEED",
        },
        {
            "dataclass": dataclass,
            "Any": Any,
            "LEGACY_MODEL_REGISTRY": LEGACY_MODEL_REGISTRY,
            "BaselineError": ValueError,
            "load_dataset": load_dataset,
            "random": random,
            "concatenate_datasets": lambda parts: Split(
                row for part in parts for row in part
            ),
            "np": SimpleNamespace(
                array=lambda values, dtype: values, int64=int, ndarray=list
            ),
            "logger": logging.getLogger("test"),
            "MAX_REPORTED_UNMAPPED": 10,
        },
    )


def domain_baseline():
    return baseline_loader(lambda repo, split, revision: Split(PUBLISHED_DOMAIN_ROWS))


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

    def test_feedback_reads_the_pinned_fresh_held_out_set_file_by_file(self):
        published = {
            "text/crosswoz/feedback/fresh-crosswoz.jsonl": [
                {
                    "text": "a",
                    "label": "SAT",
                    "source": "crosswoz",
                    "script": "cjk",
                    "length_bin": 6,
                },
                {
                    "text": "b",
                    "label": "NO_FEEDBACK",
                    "source": "crosswoz",
                    "script": "cjk",
                    "length_bin": 6,
                },
            ],
        }
        reads = []

        def load_dataset(repo, data_files, split, revision):
            reads.append((repo, data_files, split, revision))
            return Split(published[data_files])

        namespace = baseline_loader(load_dataset)
        spec = namespace["TASK_SPECS"]["feedback"]
        self.assertEqual(spec.split_rule, "by_source")
        self.assertRegex(spec.revision, r"^[0-9a-f]{40}$")
        self.assertEqual(spec.dataset_repo, "vllm-sr/router-signal-suite")
        self.assertFalse(
            [name for name in spec.data_files if "feedback-detector" in name]
        )
        for repo in (
            LEGACY_MODEL_REGISTRY["feedback"]["id"],
            LEGACY_MODEL_REGISTRY["feedback"]["lora_id"],
            MODEL_REGISTRY["feedback"]["id"],
        ):
            spec.validate_artifact(repo)
        mapping = {"SAT": 1, "NO_FEEDBACK": 0}
        texts, labels, available, matched, _ = namespace["load_rows"](
            spec, mapping, None, spec.revision
        )
        self.assertEqual((texts, labels, available), (["a", "b"], [1, 0], 2))
        self.assertEqual(spec.match_strata, ("source", "script", "length_bin", "qmark"))
        self.assertEqual(matched["rows_by_label"], {"NO_FEEDBACK": 1, "SAT": 1})
        self.assertEqual(matched["selected_by_label"], {"NO_FEEDBACK": 1, "SAT": 1})
        self.assertEqual(
            reads,
            [
                (spec.dataset_repo, name, "train", spec.revision)
                for name in spec.data_files
            ],
        )

    def test_feedback_fresh_set_is_class_matched_inside_the_stated_strata(self):
        # The pinned file's shape: every SAT row is a question-mark-free thank,
        # most other turns are questions, and the shortest bin holds more
        # NO_FEEDBACK rows than the match keeps.
        published = {
            "text/crosswoz/feedback/fresh-crosswoz.jsonl": [
                {
                    "text": "thanks",
                    "label": "SAT",
                    "source": "crosswoz",
                    "script": "cjk",
                    "length_bin": 6,
                },
                {
                    "text": "thanks a lot",
                    "label": "SAT",
                    "source": "crosswoz",
                    "script": "cjk",
                    "length_bin": 7,
                },
                {
                    "text": "where is it?",
                    "label": "NO_FEEDBACK",
                    "source": "crosswoz",
                    "script": "cjk",
                    "length_bin": 9,
                },
                {
                    "text": "book a table",
                    "label": "NO_FEEDBACK",
                    "source": "crosswoz",
                    "script": "cjk",
                    "length_bin": 6,
                },
                {
                    "text": "cheap food",
                    "label": "NO_FEEDBACK",
                    "source": "crosswoz",
                    "script": "cjk",
                    "length_bin": 6,
                },
                {
                    "text": "a table for two",
                    "label": "NO_FEEDBACK",
                    "source": "crosswoz",
                    "script": "cjk",
                    "length_bin": 6,
                },
                {
                    "text": "open now",
                    "label": "NO_FEEDBACK",
                    "source": "crosswoz",
                    "script": "cjk",
                    "length_bin": 7,
                },
            ],
        }

        def load_dataset(repo, data_files, split, revision):
            return Split(published[data_files])

        mapping = {"SAT": 1, "NO_FEEDBACK": 0}

        def scored():
            namespace = baseline_loader(load_dataset)
            spec = namespace["TASK_SPECS"]["feedback"]
            return namespace["load_rows"](spec, mapping, None, spec.revision)

        texts, labels, available, matched, _ = scored()
        # The question-mark row shares no stratum with any SAT row, so its
        # stratum keeps nothing; each kept stratum holds one row per class.
        self.assertEqual(available, 7)
        self.assertEqual(matched["rows_by_label"], {"NO_FEEDBACK": 5, "SAT": 2})
        self.assertEqual(matched["selected_by_label"], {"NO_FEEDBACK": 2, "SAT": 2})
        self.assertEqual(matched["strata"], ["source", "script", "length_bin", "qmark"])
        self.assertEqual(matched["seed"], 20261011)
        self.assertEqual(texts, ["thanks", "thanks a lot", "book a table", "open now"])
        self.assertEqual(labels, [1, 1, 0, 0])
        # The seeded draw is stable, so a rerun scores the same rows.
        again, _, _, _, _ = scored()
        self.assertEqual(again, texts)
        # The draw is the seed's, not the file's order: another seed keeps the
        # counts and takes a different row from the over-supplied bin.
        namespace = baseline_loader(load_dataset)
        select = namespace["select_matched_rows"]
        spec = namespace["TASK_SPECS"]["feedback"]
        published_rows = published["text/crosswoz/feedback/fresh-crosswoz.jsonl"]
        drawn, _ = select(list(published_rows), spec)
        other, _ = select(list(published_rows), spec, seed=7)
        self.assertEqual(
            [row["text"] for row in drawn],
            ["thanks", "thanks a lot", "book a table", "open now"],
        )
        self.assertEqual(other[2]["text"], "a table for two")

    def test_a_question_mark_row_in_a_scored_bin_is_never_a_candidate(self):
        # The question mark is its own stratum value, so a question in a bin the
        # SAT rows occupy shares no stratum with them and cannot be drawn.
        published = {
            "text/crosswoz/feedback/fresh-crosswoz.jsonl": [
                {
                    "text": "thanks",
                    "label": "SAT",
                    "source": "crosswoz",
                    "script": "cjk",
                    "length_bin": 6,
                },
                {
                    "text": "thanks a lot",
                    "label": "SAT",
                    "source": "crosswoz",
                    "script": "cjk",
                    "length_bin": 6,
                },
                {
                    "text": "book a table",
                    "label": "NO_FEEDBACK",
                    "source": "crosswoz",
                    "script": "cjk",
                    "length_bin": 6,
                },
                {
                    "text": "where is it?",
                    "label": "NO_FEEDBACK",
                    "source": "crosswoz",
                    "script": "cjk",
                    "length_bin": 6,
                },
            ],
        }

        def load_dataset(repo, data_files, split, revision):
            return Split(published[data_files])

        namespace = baseline_loader(load_dataset)
        spec = namespace["TASK_SPECS"]["feedback"]
        texts, _, available, matched, _ = namespace["load_rows"](
            spec, {"SAT": 1, "NO_FEEDBACK": 0}, None, spec.revision
        )
        self.assertEqual(available, 4)
        self.assertEqual(matched["rows_by_label"], {"NO_FEEDBACK": 2, "SAT": 2})
        self.assertEqual(matched["selected_by_label"], {"NO_FEEDBACK": 1, "SAT": 1})
        self.assertEqual(len(texts), 2)
        self.assertNotIn("where is it?", texts)
        self.assertIn("book a table", texts)

    def test_fact_check_reads_the_pinned_corpus_matched_test_file_by_file(self):
        # One file carries lang and the other does not, as the published files do.
        published = {
            "text/nf-cats/fact_check/test.jsonl": [
                {"text": "a", "label": "FACT_CHECK_NEEDED", "source": "nf-cats"},
                {"text": "b", "label": "NO_FACT_CHECK_NEEDED", "source": "nf-cats"},
            ],
            "text/open-question-type/fact_check/test.jsonl": [],
            "text/search-arena/fact_check/test.jsonl": [
                {"text": "c", "label": "NO_FACT_CHECK_NEEDED", "lang": "en"},
            ],
            "text/urs/fact_check/test.jsonl": [
                {"text": "d", "label": "FACT_CHECK_NEEDED", "lang": "en"},
            ],
        }
        reads = []

        def load_dataset(repo, data_files, split, revision):
            reads.append((repo, data_files, split, revision))
            return Split(published[data_files])

        namespace = baseline_loader(load_dataset)
        spec = namespace["TASK_SPECS"]["fact-check"]
        self.assertEqual(spec.split_rule, "by_source")
        self.assertRegex(spec.revision, r"^[0-9a-f]{40}$")
        self.assertFalse([name for name in spec.data_files if "dolly" in name])
        for repo in (
            LEGACY_MODEL_REGISTRY["fact-check"]["id"],
            LEGACY_MODEL_REGISTRY["fact-check"]["lora_id"],
            MODEL_REGISTRY["fact-check"]["id"],
        ):
            spec.validate_artifact(repo)
        mapping = {"NO_FACT_CHECK_NEEDED": 0, "FACT_CHECK_NEEDED": 1}
        texts, labels, available, matched, dropped = namespace["load_rows"](
            spec, mapping, None, spec.revision
        )
        self.assertEqual(
            (texts, labels, available, matched, dropped),
            (["a", "b", "c", "d"], [1, 0, 0, 1], 4, None, {}),
        )
        self.assertEqual(
            reads,
            [
                (spec.dataset_repo, name, "train", spec.revision)
                for name in spec.data_files
            ],
        )

    def test_dropped_rows_name_the_labels_and_counts(self):
        # One row carries a label the artifact's map does not define, the way
        # a checkpoint trained before a contract class was added reads.
        published = {
            "text/nf-cats/fact_check/test.jsonl": [
                {"text": "d", "label": "OUT_OF_CONTRACT", "source": "nf-cats"},
            ],
            "text/open-question-type/fact_check/test.jsonl": [],
            "text/search-arena/fact_check/test.jsonl": [
                {"text": "a", "label": "FACT_CHECK_NEEDED", "source": "search-arena"},
            ],
            "text/urs/fact_check/test.jsonl": [
                {"text": "b", "label": "NO_FACT_CHECK_NEEDED", "source": "urs"},
                {"text": "c", "label": "NO_FACT_CHECK_NEEDED", "source": "urs"},
            ],
        }

        def load_dataset(repo, data_files, split, revision):
            return Split(published[data_files])

        namespace = baseline_loader(load_dataset)
        spec = namespace["TASK_SPECS"]["fact-check"]
        mapping = {"NO_FACT_CHECK_NEEDED": 0, "FACT_CHECK_NEEDED": 1}
        texts, labels, available, _, dropped = namespace["load_rows"](
            spec, mapping, None, spec.revision
        )
        self.assertEqual((texts, labels, available), (["a", "b", "c"], [1, 0, 0], 4))
        self.assertEqual(dropped, {"OUT_OF_CONTRACT": 1})

    def test_pinned_split_is_read_and_recorded_at_its_pin(self):
        class LoadReachedError(Exception):
            pass

        specs = baseline_loader(Mock())["TASK_SPECS"]
        load_rows = Mock(side_effect=LoadReachedError)
        resolve_revision = Mock()
        measured = SimpleNamespace(
            referenced=None, model_dir=Path("model"), repo="repo", revision="0" * 40
        )
        namespace = load_definitions(
            "quality_baseline.py",
            {"run"},
            {
                "argparse": argparse,
                "Any": Any,
                "logging": logging,
                "logger": logging.getLogger("test"),
                "torch": Mock(),
                "np": Mock(),
                "load_config": Mock(),
                "served_artifacts": lambda config: {"fact-check": Mock()},
                "TASK_SPECS": specs,
                "resolve_measured_artifact": lambda args, served, validate: measured,
                "resolve_hf_revision": resolve_revision,
                "resolve_label_mapping": lambda model_dir, served: {"a": 0, "b": 1},
                "SHORT_REVISION": 12,
                "registry_drift": lambda inventory, registry: [],
                "MODEL_REGISTRY": MODEL_REGISTRY,
                "uncovered_artifacts": lambda config: [],
                "check_registry_label_order": lambda task, repo, mapping: [],
                "_provenance_findings": lambda args, measured: [],
                "load_rows": load_rows,
            },
        )
        with self.assertRaises(LoadReachedError):
            namespace["run"](
                SimpleNamespace(
                    seed=42, config="config.yaml", task="fact-check", limit=None
                )
            )
        spec = specs["fact-check"]
        load_rows.assert_called_once_with(spec, {"a": 0, "b": 1}, None, spec.revision)
        resolve_revision.assert_not_called()

    def test_legacy_checkpoints_are_checked_against_the_legacy_labels(self):
        check = label_order_check()
        mapping = legacy_mapping("feedback")
        for key in ("id", "lora_id"):
            self.assertEqual(
                check("feedback", LEGACY_MODEL_REGISTRY["feedback"][key], mapping), []
            )
        # The served registry adds NO_FEEDBACK, which the legacy head never had.
        served = check("feedback", MODEL_REGISTRY["feedback"]["id"], mapping)
        self.assertIn("do not match the artifact labels", served[0])

    def test_legacy_only_report_still_lists_the_served_artifacts_as_unmeasured(self):
        check = label_order_check()
        baselines = []
        for task in ("fact-check", "feedback"):
            repo = LEGACY_MODEL_REGISTRY[task]["id"]
            mapping = legacy_mapping(task)
            baselines.append(
                {
                    "task": task,
                    "artifact": {"repo": repo, "is_served_artifact": False},
                    "dataset": {
                        "repo": "source",
                        "split": "test",
                        "split_rule": "predefined",
                    },
                    "metrics": {
                        "rows": 1,
                        "accuracy": 1.0,
                        "macro_f1": 1.0,
                        "per_label": {
                            label: {"precision": 1.0, "recall": 1.0, "support": 1}
                            for label in mapping
                        },
                    },
                    "calibration": {"ece": 0.0, "mce": 0.0},
                    "abstention": {"curve": []},
                    "gaps": check(task, repo, mapping),
                }
            )
        report = gap_report.render(baselines, gap_report.DEFAULT_CONFIG)
        inventory = served_artifacts(load_config(gap_report.DEFAULT_CONFIG))
        for task in ("fact-check", "feedback"):
            served = inventory[task].artifact_name
            self.assertIn(
                f"- {task}: no baseline has been measured for the served `{served}`",
                report,
            )
        self.assertNotIn("the evaluation registry labels", report)

    def test_domain_baseline_leaves_out_mmlu_rows_and_the_model_trained_on_them(self):
        namespace = domain_baseline()
        spec = namespace["TASK_SPECS"]["domain"]
        self.assertEqual(spec.split_rule, "by_source")
        texts, labels, available, matched, dropped = namespace["load_rows"](
            spec, DOMAIN_SUBSET, None, "revision"
        )
        self.assertEqual(
            (texts, labels, available, matched, dropped),
            (["b", "d"], [1, 0], 2, None, {}),
        )
        spec.validate_artifact(MODEL_REGISTRY["intent"]["id"])
        for key in ("id", "lora_id"):
            with self.assertRaisesRegex(
                ValueError, "was trained on TIGER-Lab/MMLU-Pro"
            ):
                spec.validate_artifact(LEGACY_MODEL_REGISTRY["intent"][key])

    def test_filtered_domain_report_leaves_labels_without_rows_unmeasured(self):
        namespace = domain_baseline()
        _, labels, _, _, _ = namespace["load_rows"](
            namespace["TASK_SPECS"]["domain"], DOMAIN_SUBSET, None, "revision"
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
                "resolve_measured_artifact": lambda args, served, validate_repo: (
                    validate_repo(MODEL_REGISTRY["jailbreak"]["id"])
                ),
                "resolve_hf_revision": dataset_revision,
            },
        )
        with self.assertRaisesRegex(ValueError, "legacy toxicity/jailbreak"):
            namespace["run"](
                SimpleNamespace(seed=42, config="config.yaml", task="jailbreak")
            )
        dataset_revision.assert_not_called()

    def _run_model_free_baseline(
        self,
        *,
        task,
        published_rows,
        mapping,
        artifact_repo,
        is_served,
        predict_class_id,
        limit=None,
        eval_dataset_load=None,
    ):
        def default_load_dataset(repo, data_files=None, split=None, revision=None):
            if data_files is not None:
                return Split(published_rows[data_files])
            return Split(published_rows)

        tasks_ns = baseline_loader(eval_dataset_load or default_load_dataset)
        metrics_ns = load_definitions(
            "provenance/metrics.py",
            {
                "SINGLE_MAPPED_GOLD_CLASS_REASON",
                "abstention_curve",
                "calibration_metrics",
                "classification_metrics",
                "gold_class_comparability",
            },
            {
                "Any": Any,
                "Integral": Integral,
                "MIN_BIN_COUNT": 2,
                "MAX_BIN_COUNT": 100,
                "Sequence": Sequence,
            },
        )
        inventory = served_artifacts(load_config(gap_report.DEFAULT_CONFIG))
        served = inventory[task]
        measured = SimpleNamespace(
            referenced=None,
            model_dir=Path("model"),
            repo=artifact_repo,
            revision="0" * 40,
            is_served_by=lambda candidate: candidate.hf_repo == artifact_repo,
        )

        def fake_measure(
            args, measured_artifact, served_artifact, texts, labels, label_map
        ):
            truth = list(labels)
            if callable(predict_class_id):
                predicted = [predict_class_id(index) for index in range(len(truth))]
            else:
                predicted = [predict_class_id for _ in truth]
            confidences = [0.99 for _ in truth]
            classification = metrics_ns["classification_metrics"](
                truth, predicted, label_map
            )
            thresholds = tuple(sorted({0.0, 0.5, *served_artifact.thresholds}))
            return (
                {
                    "metrics": classification,
                    "unsupported_labels": sorted(
                        name
                        for name, entry in classification["per_label"].items()
                        if not entry["support"]
                    ),
                    "discrimination": None,
                    "calibration": metrics_ns["calibration_metrics"](
                        truth, predicted, confidences, bin_count=args.bin_count
                    ),
                    "abstention": metrics_ns["abstention_curve"](
                        truth, predicted, confidences, thresholds=thresholds
                    ),
                    "slices": [],
                    "performance": {
                        "latency_ms": {"mean": 1.0, "p50": 1.0, "p95": 1.0, "p99": 1.0},
                        "peak_memory_mb": 0.0,
                        "throughput_rows_per_s": 1000.0,
                    },
                },
                {"max_position_embeddings": 32768},
            )

        with tempfile.TemporaryDirectory() as tmp_dir:
            captured_logs: list[str] = []

            class ListHandler(logging.Handler):
                def emit(self, record):
                    captured_logs.append(self.format(record))

            test_logger = logging.getLogger(f"test.{task}.{id(mapping)}.{limit}")
            test_logger.setLevel(logging.INFO)
            test_logger.propagate = False
            handler = ListHandler()
            test_logger.addHandler(handler)
            args = SimpleNamespace(
                task=task,
                config="config.yaml",
                output_dir=Path(tmp_dir),
                device="cpu",
                batch_size=32,
                max_length=512,
                limit=limit,
                seed=42,
                bin_count=10,
                artifact_repo=None if is_served else artifact_repo,
                artifact_dir=None,
                artifact_manifest=None,
                manifest_dir=None,
                warmup_batches=0,
                skip_manifests=True,
            )
            runner_ns = load_definitions(
                "quality_baseline.py",
                {"run", "_build_result", "main"},
                {
                    "Any": Any,
                    "BaselineError": ValueError,
                    "MODEL_REGISTRY": MODEL_REGISTRY,
                    "MeasuredArtifact": Any,
                    "SHORT_REVISION": 12,
                    "ServedArtifact": Any,
                    "TASK_SPECS": tasks_ns["TASK_SPECS"],
                    "_measure": fake_measure,
                    "_provenance_findings": lambda _args, _measured: [],
                    "argparse": argparse,
                    "check_registry_label_order": lambda _task, _repo, _map: [],
                    "emit_baseline_manifests": Mock(),
                    "gold_class_comparability": metrics_ns["gold_class_comparability"],
                    "json": json,
                    "load_config": lambda _path: {},
                    "load_rows": tasks_ns["load_rows"],
                    "logging": logging,
                    "logger": test_logger,
                    "np": SimpleNamespace(
                        random=SimpleNamespace(seed=lambda _seed: None)
                    ),
                    "parse_args": lambda _argv=None: args,
                    "registry_drift": lambda _inv, _reg: [],
                    "resolve_hf_revision": lambda *_args, **_kwargs: "0" * 40,
                    "resolve_label_mapping": lambda _model_dir, _served: mapping,
                    "resolve_measured_artifact": (
                        lambda _args, _served, validate_repo: (
                            validate_repo(artifact_repo) or measured
                        )
                    ),
                    "served_artifacts": lambda _config: {task: served},
                    "sys": sys,
                    "tokenizer_class": lambda _model_dir: "PreTrainedTokenizerFast",
                    "torch": SimpleNamespace(manual_seed=lambda _seed: None),
                    "uncovered_artifacts": lambda _config: [],
                },
            )
            stdout_buffer = io.StringIO()
            try:
                with contextlib.redirect_stdout(stdout_buffer):
                    exit_code = runner_ns["main"]([])
            finally:
                test_logger.removeHandler(handler)
            self.assertEqual(exit_code, 0)
            written = json.loads(
                (Path(tmp_dir) / f"{task}_baseline.json").read_text(encoding="utf-8")
            )
            return written, stdout_buffer.getvalue(), captured_logs

    def test_single_class_feedback_constant_predictor_is_marked_non_comparable_and_isolated(
        self,
    ):
        published = {
            "text/crosswoz/feedback/fresh-crosswoz.jsonl": [
                {
                    "text": "thanks",
                    "label": "SAT",
                    "source": "crosswoz",
                    "script": "cjk",
                    "length_bin": 6,
                },
                {
                    "text": "thanks a lot",
                    "label": "SAT",
                    "source": "crosswoz",
                    "script": "cjk",
                    "length_bin": 7,
                },
                {
                    "text": "book a table",
                    "label": "NO_FEEDBACK",
                    "source": "crosswoz",
                    "script": "cjk",
                    "length_bin": 6,
                },
                {
                    "text": "open now",
                    "label": "NO_FEEDBACK",
                    "source": "crosswoz",
                    "script": "cjk",
                    "length_bin": 7,
                },
            ],
        }
        legacy_repo = LEGACY_MODEL_REGISTRY["feedback"]["id"]
        legacy_map = legacy_mapping("feedback")
        single_class_run, stdout, logs = self._run_model_free_baseline(
            task="feedback",
            published_rows=published,
            mapping=legacy_map,
            artifact_repo=legacy_repo,
            is_served=False,
            predict_class_id=legacy_map["SAT"],
        )

        # Metric formulas and unsupported-label handling remain unchanged.
        self.assertEqual(single_class_run["metrics"]["rows"], 2)
        self.assertEqual(single_class_run["metrics"]["accuracy"], 1.0)
        self.assertEqual(single_class_run["metrics"]["macro_f1"], 1.0)
        self.assertEqual(
            single_class_run["unsupported_labels"],
            ["NEED_CLARIFICATION", "WANT_DIFFERENT", "WRONG_ANSWER"],
        )
        self.assertEqual(
            single_class_run["dataset"]["dropped_rows"], {"NO_FEEDBACK": 2}
        )

        # Explicit run-level non-comparable classification is persisted.
        self.assertIs(single_class_run["comparable"], False)
        self.assertEqual(
            single_class_run["comparability"],
            {
                "reason": "single_mapped_gold_class",
                "surviving_gold_class_count": 1,
                "surviving_gold_classes": [
                    {"class_id": 0, "label": "SAT", "support": 2}
                ],
            },
        )
        self.assertIn("comparable=false (single_mapped_gold_class)", stdout)
        self.assertTrue(any("single_mapped_gold_class" in message for message in logs))

        # Served two-class control run without `comparable` (and realistic 0.7000 accuracy).
        served_repo = MODEL_REGISTRY["feedback"]["id"]
        served_two_class_run = {
            "task": "feedback",
            "artifact": {"repo": served_repo, "is_served_artifact": True},
            "dataset": {
                "repo": "vllm-sr/router-signal-suite",
                "split": "test",
                "split_rule": "by_source",
            },
            "metrics": {
                "rows": 4,
                "accuracy": 0.7,
                "macro_f1": 0.769,
                "per_label": {
                    "SAT": {"precision": 0.9, "recall": 0.98, "support": 2},
                    "NO_FEEDBACK": {"precision": 0.9, "recall": 0.42, "support": 2},
                    "NEED_CLARIFICATION": {
                        "precision": 0.0,
                        "recall": 0.0,
                        "support": 0,
                    },
                    "WRONG_ANSWER": {"precision": 0.0, "recall": 0.0, "support": 0},
                    "WANT_DIFFERENT": {"precision": 0.0, "recall": 0.0, "support": 0},
                },
            },
            "calibration": {"ece": 0.02, "mce": 0.04},
            "abstention": {"curve": []},
            "gaps": [],
        }

        report = gap_report.render(
            [served_two_class_run, single_class_run], gap_report.DEFAULT_CONFIG
        )
        measured_section = report.split("## Measured baselines")[1].split("## ")[0]
        diagnostic_section = report.split("## Non-comparable diagnostics")[1].split(
            "## "
        )[0]
        gaps_section = report.split("## Gaps")[1].split("## Reproduction")[0]

        self.assertIn("Vela-1.0-Encoder-307M-Feedback", measured_section)
        self.assertNotIn("mmbert32k-feedback-detector-merged", measured_section)
        self.assertNotIn("1.0000", measured_section)

        self.assertIn("mmbert32k-feedback-detector-merged", diagnostic_section)
        self.assertIn("`single_mapped_gold_class`", diagnostic_section)
        self.assertIn("`SAT` (2)", diagnostic_section)
        self.assertIn("1.0000", diagnostic_section)

        self.assertNotIn("scores 1.0000 on the same split against", gaps_section)
        self.assertIn(
            "is marked non-comparable (`single_mapped_gold_class`)", gaps_section
        )
        self.assertNotIn("mmbert32k-feedback-detector-merged` has ECE", gaps_section)
        self.assertNotIn("gating on the argmax score", gaps_section)
        self.assertNotIn(
            "mmbert32k-feedback-detector-merged` recalls only", gaps_section
        )

        # Even a non-constant low-recall single-class run does not emit
        # threshold, calibration, or quality gaps.
        low_recall_single, _, _ = self._run_model_free_baseline(
            task="feedback",
            published_rows=published,
            mapping=legacy_map,
            artifact_repo=legacy_repo,
            is_served=False,
            predict_class_id=lambda idx: (
                legacy_map["SAT"] if idx == 0 else legacy_map["WRONG_ANSWER"]
            ),
        )
        self.assertEqual(low_recall_single["metrics"]["accuracy"], 0.5)
        low_recall_report = gap_report.render(
            [low_recall_single], gap_report.DEFAULT_CONFIG
        )
        self.assertNotIn("### Threshold", low_recall_report)
        self.assertNotIn("### Calibration", low_recall_report)
        self.assertNotIn("### Quality", low_recall_report)

    def test_row_limit_collapse_to_single_class_marks_non_comparable(self):
        published = {
            "text/nf-cats/fact_check/test.jsonl": [
                {"text": "a", "label": "FACT_CHECK_NEEDED", "source": "nf-cats"},
                {"text": "b", "label": "FACT_CHECK_NEEDED", "source": "nf-cats"},
                {"text": "c", "label": "NO_FACT_CHECK_NEEDED", "source": "nf-cats"},
            ],
            "text/open-question-type/fact_check/test.jsonl": [],
            "text/search-arena/fact_check/test.jsonl": [],
            "text/urs/fact_check/test.jsonl": [],
        }
        mapping = {"NO_FACT_CHECK_NEEDED": 0, "FACT_CHECK_NEEDED": 1}
        limited_run, _, _ = self._run_model_free_baseline(
            task="fact-check",
            published_rows=published,
            mapping=mapping,
            artifact_repo=MODEL_REGISTRY["fact-check"]["id"],
            is_served=True,
            predict_class_id=mapping["FACT_CHECK_NEEDED"],
            limit=2,
        )
        self.assertIs(limited_run["comparable"], False)
        self.assertEqual(
            limited_run["comparability"],
            {
                "reason": "single_mapped_gold_class",
                "surviving_gold_class_count": 1,
                "surviving_gold_classes": [
                    {"class_id": 1, "label": "FACT_CHECK_NEEDED", "support": 2}
                ],
            },
        )

    def test_exclude_prefix_collapse_to_single_class_marks_non_comparable(self):
        rows = [
            {"question": "a", "category": "math", "src": "ori_mmlu-algebra"},
            {"question": "b", "category": "physics", "src": "scibench"},
            {"question": "c", "category": "physics", "src": "theoremqa"},
        ]
        filtered_run, _, _ = self._run_model_free_baseline(
            task="domain",
            published_rows=rows,
            mapping=DOMAIN_SUBSET,
            artifact_repo=MODEL_REGISTRY["intent"]["id"],
            is_served=True,
            predict_class_id=DOMAIN_SUBSET["physics"],
            eval_dataset_load=lambda repo, split, revision: Split(rows),
        )
        self.assertIs(filtered_run["comparable"], False)
        self.assertEqual(
            filtered_run["comparability"],
            {
                "reason": "single_mapped_gold_class",
                "surviving_gold_class_count": 1,
                "surviving_gold_classes": [
                    {"class_id": 1, "label": "physics", "support": 2}
                ],
            },
        )

    def test_non_comparable_only_and_legacy_missing_field_report_behavior(self):
        inventory = served_artifacts(load_config(gap_report.DEFAULT_CONFIG))
        non_comparable_only = {
            "task": "feedback",
            "artifact": {
                "repo": inventory["feedback"].hf_repo,
                "is_served_artifact": True,
            },
            "dataset": {"repo": "source", "split": "test", "split_rule": "by_source"},
            "metrics": {
                "rows": 2,
                "accuracy": 1.0,
                "macro_f1": 1.0,
                "per_label": {"SAT": {"precision": 1.0, "recall": 1.0, "support": 2}},
            },
            "calibration": {"ece": 0.0, "mce": 0.0},
            "abstention": {"curve": []},
            "comparable": False,
            "gaps": [],
        }
        report = gap_report.render([non_comparable_only], gap_report.DEFAULT_CONFIG)
        measured_section = report.split("## Measured baselines")[1].split("## ")[0]
        self.assertIn("No comparable baseline was supplied.", measured_section)
        self.assertIn("## Non-comparable diagnostics", report)
        self.assertIn(
            f"- feedback: no comparable baseline has been measured for the served "
            f"`{inventory['feedback'].artifact_name}` (only non-comparable "
            "diagnostics were supplied)",
            report,
        )
        self.assertNotIn(
            f"- feedback: no baseline has been measured for the served "
            f"`{inventory['feedback'].artifact_name}`",
            report,
        )
        self.assertIn("is marked non-comparable (`non_comparable`)", report)

        # Malformed comparability payloads do not crash or render `class_id=None`.
        malformed = [
            {
                **non_comparable_only,
                "comparability": {
                    "reason": "single_mapped_gold_class",
                    "surviving_gold_classes": 1,
                },
            },
            {
                **non_comparable_only,
                "comparability": {
                    "reason": "single_mapped_gold_class",
                    "surviving_gold_classes": [{"class_id": None, "support": 2}],
                },
            },
        ]
        malformed_report = gap_report.render(malformed, gap_report.DEFAULT_CONFIG)
        self.assertNotIn("class_id=None", malformed_report)
        self.assertIn("class_id=unknown (2)", malformed_report)

        # Gate single-class diagnostic preserves Safe recall in Non-comparable diagnostics.
        benign_only_gate = {
            "task": "jailbreak",
            "artifact": {
                "repo": inventory["jailbreak"].hf_repo,
                "is_served_artifact": True,
                "configured_thresholds": [0.7],
            },
            "dataset": {"repo": "source", "split": "test", "split_rule": "by_source"},
            "metrics": {
                "rows": 2,
                "accuracy": 1.0,
                "macro_f1": 1.0,
                "per_label": {
                    "benign": {"precision": 1.0, "recall": 1.0, "support": 2},
                    "jailbreak": {"precision": 0.0, "recall": 0.0, "support": 0},
                },
            },
            "calibration": {"ece": 0.0, "mce": 0.0},
            "abstention": {
                "curve": [],
                "operating_points": [
                    {
                        "threshold": 0.7,
                        "positive_labels": ["jailbreak"],
                        "flagged_rate": 0.05,
                        "recall": None,
                        "false_positive_rate": 0.05,
                        "precision": None,
                    }
                ],
            },
            "comparable": False,
            "comparability": {
                "reason": "single_mapped_gold_class",
                "surviving_gold_class_count": 1,
                "surviving_gold_classes": [
                    {"class_id": 0, "label": "benign", "support": 2}
                ],
            },
            "gaps": [],
        }
        gate_report = gap_report.render([benign_only_gate], gap_report.DEFAULT_CONFIG)
        gate_diag = gate_report.split("## Non-comparable diagnostics")[1].split("## ")[
            0
        ]
        self.assertIn("Safe recall", gate_diag)
        self.assertIn("0.9500", gate_diag)

        # Older artifact without `comparable` stays in Measured baselines.
        legacy_artifact_without_flag = {
            "task": "feedback",
            "artifact": {
                "repo": inventory["feedback"].hf_repo,
                "is_served_artifact": True,
            },
            "dataset": {"repo": "source", "split": "test", "split_rule": "by_source"},
            "metrics": {
                "rows": 2,
                "accuracy": 0.85,
                "macro_f1": 0.85,
                "per_label": {
                    "SAT": {"precision": 0.85, "recall": 0.85, "support": 1},
                    "NO_FEEDBACK": {"precision": 0.85, "recall": 0.85, "support": 1},
                },
            },
            "calibration": {"ece": 0.01, "mce": 0.02},
            "abstention": {"curve": []},
            "gaps": [],
        }
        legacy_report = gap_report.render(
            [legacy_artifact_without_flag], gap_report.DEFAULT_CONFIG
        )
        self.assertNotIn("## Non-comparable diagnostics", legacy_report)
        self.assertIn(
            "0.8500", legacy_report.split("## Measured baselines")[1].split("## ")[0]
        )

    def test_zero_usable_rows_continues_to_raise_error(self):
        published = {
            "text/crosswoz/feedback/fresh-crosswoz.jsonl": [
                {
                    "text": "book a table",
                    "label": "NO_FEEDBACK",
                    "source": "crosswoz",
                    "script": "cjk",
                    "length_bin": 6,
                },
            ],
        }
        namespace = baseline_loader(
            lambda repo, data_files, split, revision: Split(published[data_files])
        )
        spec = namespace["TASK_SPECS"]["feedback"]
        with self.assertRaisesRegex(
            ValueError, "produced no rows the artifact can score"
        ):
            namespace["load_rows"](
                spec, legacy_mapping("feedback"), None, spec.revision
            )

        comparability = load_definitions(
            "provenance/metrics.py",
            {"SINGLE_MAPPED_GOLD_CLASS_REASON", "gold_class_comparability"},
            {"Any": Any, "Integral": Integral, "Sequence": Sequence},
        )["gold_class_comparability"]
        with self.assertRaisesRegex(ValueError, "at least one row"):
            comparability([], legacy_mapping("feedback"))

    def test_two_class_feedback_control_keeps_existing_comparative_behavior(self):
        published = {
            "text/crosswoz/feedback/fresh-crosswoz.jsonl": [
                {
                    "text": "thanks",
                    "label": "SAT",
                    "source": "crosswoz",
                    "script": "cjk",
                    "length_bin": 6,
                },
                {
                    "text": "thanks a lot",
                    "label": "SAT",
                    "source": "crosswoz",
                    "script": "cjk",
                    "length_bin": 7,
                },
                {
                    "text": "book a table",
                    "label": "NO_FEEDBACK",
                    "source": "crosswoz",
                    "script": "cjk",
                    "length_bin": 6,
                },
                {
                    "text": "open now",
                    "label": "NO_FEEDBACK",
                    "source": "crosswoz",
                    "script": "cjk",
                    "length_bin": 7,
                },
            ],
        }
        vela_map = {
            label: index
            for index, label in enumerate(MODEL_REGISTRY["feedback"]["labels"])
        }
        two_class_run, stdout, _ = self._run_model_free_baseline(
            task="feedback",
            published_rows=published,
            mapping=vela_map,
            artifact_repo=MODEL_REGISTRY["feedback"]["id"],
            is_served=True,
            predict_class_id=vela_map["SAT"],
        )
        self.assertEqual(two_class_run["metrics"]["rows"], 4)
        self.assertEqual(
            two_class_run["unsupported_labels"],
            ["NEED_CLARIFICATION", "WANT_DIFFERENT", "WRONG_ANSWER"],
        )
        self.assertNotIn("comparable", two_class_run)
        self.assertNotIn("comparability", two_class_run)
        self.assertNotIn("comparable=false", stdout)

        report = gap_report.render([two_class_run], gap_report.DEFAULT_CONFIG)
        self.assertNotIn("## Non-comparable diagnostics", report)
        measured_section = report.split("## Measured baselines")[1].split("## ")[0]
        self.assertIn("Vela-1.0-Encoder-307M-Feedback", measured_section)

    def test_baseline_metrics_summarise_marks_single_class_and_leaves_two_class_unchanged(
        self,
    ):
        class _IntList(list):
            def tolist(self):
                return list(self)

        class _ProbMatrix:
            def __init__(self, rows):
                self._rows = rows

            def argmax(self, axis):
                return _IntList(row.index(max(row)) for row in self._rows)

            def max(self, axis):
                return _IntList(max(row) for row in self._rows)

            def tolist(self):
                return [list(row) for row in self._rows]

        class _MatrixStub:
            def __init__(self, n):
                self._n = n

            def astype(self, _dtype):
                return self

            def tolist(self):
                return [[0] * self._n for _ in range(self._n)]

        metrics_ns = load_definitions(
            "provenance/metrics.py",
            {
                "SINGLE_MAPPED_GOLD_CLASS_REASON",
                "abstention_curve",
                "calibration_metrics",
                "classification_metrics",
                "discrimination",
                "gold_class_comparability",
                "latency_percentiles",
                "operating_points",
            },
            {
                "Any": Any,
                "Integral": Integral,
                "MIN_BIN_COUNT": 2,
                "MAX_BIN_COUNT": 100,
                "Sequence": Sequence,
                "math": math,
            },
        )
        summary_ns = load_definitions(
            "baseline_metrics.py",
            {"summarise", "_length_slice_metrics"},
            {
                "Any": Any,
                "LENGTH_BOUNDARIES": (16, 32, 64, 256, 1024),
                "confusion_matrix": lambda _y, _p, labels: _MatrixStub(len(labels)),
                "np": SimpleNamespace(ndarray=object),
                **metrics_ns,
            },
        )
        mapping = {"SAT": 0, "NO_FEEDBACK": 1}
        single_summary = summary_ns["summarise"](
            _ProbMatrix([[0.99, 0.01], [0.99, 0.01]]),
            _IntList([0, 0]),
            ["thanks", "thanks a lot"],
            mapping,
            [1.0, 1.0],
            10,
            (0.0, 0.5),
            0.0,
        )
        self.assertIs(single_summary["comparable"], False)
        self.assertEqual(
            single_summary["comparability"],
            {
                "reason": "single_mapped_gold_class",
                "surviving_gold_class_count": 1,
                "surviving_gold_classes": [
                    {"class_id": 0, "label": "SAT", "support": 2}
                ],
            },
        )
        self.assertEqual(single_summary["unsupported_labels"], ["NO_FEEDBACK"])
        self.assertEqual(single_summary["metrics"]["macro_f1"], 1.0)

        two_class_summary = summary_ns["summarise"](
            _ProbMatrix([[0.9, 0.1], [0.1, 0.9]]),
            _IntList([0, 1]),
            ["thanks", "book a table"],
            mapping,
            [1.0, 1.0],
            10,
            (0.0, 0.5),
            0.0,
        )
        self.assertNotIn("comparable", two_class_summary)
        self.assertNotIn("comparability", two_class_summary)
        self.assertEqual(two_class_summary["unsupported_labels"], [])

    @unittest.skipUnless(
        importlib.util.find_spec("jsonschema")
        and importlib.util.find_spec("referencing"),
        "needs provenance schema dependencies",
    )
    def test_baseline_manifests_forwards_single_class_comparability_into_validated_manifest(
        self,
    ):
        sys.path.insert(0, str(ROOT))
        emit_mod = importlib.import_module("provenance.emit")
        manifest_mod = importlib.import_module("provenance.manifest")

        class _Mask(list):
            def sum(self):
                return sum(1 for item in self if item)

        class _LabelArray(list):
            __hash__ = None

            def tolist(self):
                return list(self)

            def __eq__(self, other):
                return _Mask(item == other for item in self)

        published = {
            "text/crosswoz/feedback/fresh-crosswoz.jsonl": [
                {
                    "text": "thanks",
                    "label": "SAT",
                    "source": "crosswoz",
                    "script": "cjk",
                    "length_bin": 6,
                },
                {
                    "text": "book a table",
                    "label": "NO_FEEDBACK",
                    "source": "crosswoz",
                    "script": "cjk",
                    "length_bin": 6,
                },
            ]
        }

        def load_dataset(repo, data_files, split, revision):
            return Split(published[data_files])

        tasks_ns = baseline_loader(load_dataset)
        tasks_ns["np"] = SimpleNamespace(
            array=lambda values, dtype: _LabelArray(values),
            int64=int,
            ndarray=_LabelArray,
        )
        metrics_ns = load_definitions(
            "provenance/metrics.py",
            {
                "SINGLE_MAPPED_GOLD_CLASS_REASON",
                "abstention_curve",
                "calibration_metrics",
                "classification_metrics",
                "gold_class_comparability",
            },
            {
                "Any": Any,
                "Integral": Integral,
                "MIN_BIN_COUNT": 2,
                "MAX_BIN_COUNT": 100,
                "Sequence": Sequence,
            },
        )
        legacy_repo = LEGACY_MODEL_REGISTRY["feedback"]["id"]
        legacy_map = legacy_mapping("feedback")
        served = served_artifacts(load_config(gap_report.DEFAULT_CONFIG))["feedback"]

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            model_dir = root / "model"
            model_dir.mkdir()
            (model_dir / "config.json").write_text("{}", encoding="utf-8")

            measured = SimpleNamespace(
                referenced=None,
                model_dir=model_dir,
                repo=legacy_repo,
                revision="a" * 40,
                is_served_by=lambda candidate: candidate.hf_repo == legacy_repo,
            )

            orig_code_revision = emit_mod.code_revision
            orig_dep_versions = emit_mod.dependency_versions
            emit_mod.code_revision = lambda entrypoint, repo_root: {
                "repo": "vllm-project/semantic-router",
                "revision": "b" * 40,
                "entrypoint": entrypoint,
                "dirty": False,
            }
            emit_mod.dependency_versions = lambda: {
                "python": "3.12.4",
                "packages": {"pyyaml": "6.0"},
            }
            try:
                manifests_ns = load_definitions(
                    "baseline_manifests.py",
                    {
                        "REPO_ROOT",
                        "ENTRYPOINT",
                        "emit_baseline_manifests",
                        "_build_measured_artifact",
                        "_slug",
                        "_build_dataset",
                        "_build_evaluation",
                    },
                    {
                        "Any": Any,
                        "ENTRYPOINT": "src/training/model_eval/quality_baseline.py",
                        "Path": Path,
                        "REPO_ROOT": root,
                        "ServedArtifact": Any,
                        "TaskSpec": Any,
                        "argparse": argparse,
                        "build_artifact_manifest": emit_mod.build_artifact_manifest,
                        "build_dataset_manifest": emit_mod.build_dataset_manifest,
                        "build_evaluation_manifest": emit_mod.build_evaluation_manifest,
                        "np": tasks_ns["np"],
                        "os": SimpleNamespace(
                            uname=lambda: SimpleNamespace(machine="x86_64")
                        ),
                        "split_digest": emit_mod.split_digest,
                        "tokenizer_class": lambda _dir: "PreTrainedTokenizerFast",
                        "torch": SimpleNamespace(
                            cuda=SimpleNamespace(get_device_name=lambda _i: "cpu")
                        ),
                        "write_manifest": emit_mod.write_manifest,
                    },
                )

                def fake_measure(
                    args, measured_artifact, served_artifact, texts, labels, label_map
                ):
                    truth = labels.tolist()
                    predicted = [label_map["SAT"] for _ in truth]
                    confidences = [0.99 for _ in truth]
                    return (
                        {
                            "metrics": metrics_ns["classification_metrics"](
                                truth, predicted, label_map
                            ),
                            "unsupported_labels": [
                                "NEED_CLARIFICATION",
                                "WANT_DIFFERENT",
                                "WRONG_ANSWER",
                            ],
                            "discrimination": None,
                            "calibration": metrics_ns["calibration_metrics"](
                                truth, predicted, confidences, bin_count=args.bin_count
                            ),
                            "abstention": metrics_ns["abstention_curve"](
                                truth, predicted, confidences, thresholds=(0.0, 0.5)
                            ),
                            "slices": [],
                            "performance": {
                                "latency_ms": {
                                    "mean": 1.0,
                                    "p50": 1.0,
                                    "p95": 1.0,
                                    "p99": 1.0,
                                },
                                "peak_memory_mb": 0.0,
                                "throughput_rows_per_s": 1000.0,
                            },
                        },
                        {"max_position_embeddings": 32768},
                    )

                args = SimpleNamespace(
                    task="feedback",
                    config="config.yaml",
                    output_dir=root / "out",
                    device="cpu",
                    batch_size=32,
                    max_length=512,
                    limit=None,
                    seed=42,
                    bin_count=10,
                    artifact_repo=legacy_repo,
                    artifact_dir=None,
                    artifact_manifest=None,
                    manifest_dir=None,
                    warmup_batches=0,
                    skip_manifests=False,
                )
                runner_ns = load_definitions(
                    "quality_baseline.py",
                    {"run", "_build_result"},
                    {
                        "Any": Any,
                        "BaselineError": ValueError,
                        "MODEL_REGISTRY": MODEL_REGISTRY,
                        "MeasuredArtifact": Any,
                        "SHORT_REVISION": 12,
                        "ServedArtifact": Any,
                        "TASK_SPECS": tasks_ns["TASK_SPECS"],
                        "_measure": fake_measure,
                        "_provenance_findings": lambda _args, _measured: [],
                        "argparse": argparse,
                        "check_registry_label_order": lambda _task, _repo, _map: [],
                        "emit_baseline_manifests": manifests_ns[
                            "emit_baseline_manifests"
                        ],
                        "gold_class_comparability": metrics_ns[
                            "gold_class_comparability"
                        ],
                        "json": json,
                        "load_config": lambda _path: {},
                        "load_rows": tasks_ns["load_rows"],
                        "logging": logging,
                        "logger": logging.getLogger("test.manifests"),
                        "np": SimpleNamespace(
                            random=SimpleNamespace(seed=lambda _seed: None)
                        ),
                        "registry_drift": lambda _inv, _reg: [],
                        "resolve_hf_revision": Mock(),
                        "resolve_label_mapping": lambda _model_dir, _served: legacy_map,
                        "resolve_measured_artifact": (
                            lambda _args, _served, validate_repo: (
                                validate_repo(legacy_repo) or measured
                            )
                        ),
                        "served_artifacts": lambda _config: {"feedback": served},
                        "tokenizer_class": lambda _dir: "PreTrainedTokenizerFast",
                        "torch": SimpleNamespace(manual_seed=lambda _seed: None),
                        "uncovered_artifacts": lambda _config: [],
                    },
                )
                result = runner_ns["run"](args)
            finally:
                emit_mod.code_revision = orig_code_revision
                emit_mod.dependency_versions = orig_dep_versions

            eval_manifest = manifest_mod.load_manifest(
                Path(result["manifests"]["evaluation"]),
                expected_kind="evaluation",
            )
            self.assertIs(eval_manifest["comparable"], False)
            self.assertEqual(
                eval_manifest["comparability"],
                {
                    "reason": "single_mapped_gold_class",
                    "surviving_gold_class_count": 1,
                    "surviving_gold_classes": [
                        {"class_id": 0, "label": "SAT", "support": 1}
                    ],
                },
            )
