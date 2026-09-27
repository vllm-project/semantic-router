"""First-release v3 numeric and evidence gates; v2 remains untouched."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmark.generate import FINAL_FAMILIES
from jev_arena.arena_v3 import REQUIRED_PROTOCOL_SOURCES, _pair_digest
from publication import bundle_arena_v3 as bundle
from publication.generate_arena_v3 import generate
from publication.tests import test_bundle_arena as legacy_fixture
from publication.tests import test_generate_arena_v3 as artifact_fixture
from transfer.build import EVALUATION_TASKS, PANEL_VERSION


class EikosCollectorBindingTests(unittest.TestCase):
    def test_native_adapter_is_exact_collector_and_stable_full_run(self) -> None:
        from scripts.eikos_stable_runtime_v3 import FLA_BACKEND, TORCH_BACKEND

        native = {
            "collector_source_sha256": bundle.common.sha_file(bundle.EIKOS_COLLECTOR),
            "input_items": 231,
            "evaluated_items": 231,
            "counts": {"items": 231},
            "max_items": None,
            "runtime": {
                "torch_deterministic_algorithms": True,
                "gated_delta_backend_before": FLA_BACKEND,
                "gated_delta_backend": TORCH_BACKEND,
            },
        }
        self.assertEqual(
            bundle._native_adapter_sha(native, "public"),
            native["collector_source_sha256"],
        )
        for altered in (
            native | {"collector_source_sha256": "a" * 64},
            native | {"evaluated_items": 230},
            native | {"max_items": 230},
            native | {"adapter_sha256": "b" * 64},
            native
            | {
                "runtime": native["runtime"] | {"torch_deterministic_algorithms": False}
            },
        ):
            with self.assertRaises(ValueError):
                bundle._native_adapter_sha(altered, "public")


def write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


class NumericReleaseGateTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.panels = {"typed_gold_sha256": "1" * 64, "css_gold_sha256": "2" * 64}
        self.new = {
            "overall": {
                "n": 2000,
                "invalid_or_missing_n": 0,
                "probability_n": 2000,
                "brier": 0.20,
            },
            "by_type": {
                kind: {"accuracy_all": 0.60} for kind in ("choice", "noul", "score")
            },
            "predictions_sha256": "3" * 64,
        }
        self.old = {
            "overall": {
                "n": 2000,
                "invalid_or_missing_n": 0,
                "probability_n": 2000,
                "brier": 0.20,
            },
            "by_type": {
                kind: {"accuracy_all": 0.50} for kind in ("choice", "noul", "score")
            },
            "predictions_sha256": "4" * 64,
        }
        self.new_css = {
            "roles": {"evaluation": {"valid_items": 6547}},
            "predictions_sha256": "5" * 64,
        }
        self.old_css = {
            "roles": {"evaluation": {"valid_items": 6547}},
            "predictions_sha256": "6" * 64,
        }
        self.paths = {
            name: self.root / f"{name}.json"
            for name in ("new_typed", "old_typed", "new_css", "old_css", "aggregate")
        }
        for name, value in (
            ("new_typed", self.new),
            ("old_typed", self.old),
            ("new_css", self.new_css),
            ("old_css", self.old_css),
        ):
            write(self.paths[name], value)
        self.new_row = {
            "key": "new",
            "model_id": "org/DEV2.0-4B",
            "size_b": 4.2,
            "axes": {"typed": 0.6, "transfer": 0.6},
            "score": 60.0,
        }
        self.old_row = {
            "key": "old",
            "group": "decision1",
            "model_id": "org/Decision-1.0-Nox-4B",
            "size_b": 4.0,
            "axes": {"typed": 0.5, "transfer": 0.5},
            "score": 50.0,
        }
        self.arena = {
            "panel_sha256": self.panels,
            "models": [self.new_row, self.old_row],
        }
        self.artifacts = {
            "comparison_pairs": [
                {
                    "new": "new",
                    "old": "old",
                    "report_sha256": {
                        "new_typed_report": bundle.common.sha_file(
                            self.paths["new_typed"]
                        ),
                        "old_typed_report": bundle.common.sha_file(
                            self.paths["old_typed"]
                        ),
                        "new_transfer_report": bundle.common.sha_file(
                            self.paths["new_css"]
                        ),
                        "old_transfer_report": bundle.common.sha_file(
                            self.paths["old_css"]
                        ),
                    },
                }
            ]
        }
        self.binding = {
            "typed": {"predictions_sha256": "3" * 64},
            "css": {"predictions_sha256": "5" * 64},
        }
        panel_digest = hashlib.sha256(
            json.dumps(self.panels, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        self.aggregate = {
            "schema_version": "jevarena-v3-paired-aggregate/1",
            "models": {
                "left": self.new_row["model_id"],
                "right": self.old_row["model_id"],
            },
            "panel_sha256": panel_digest,
            "typed_gold_sha256": self.panels["typed_gold_sha256"],
            "css_gold_sha256": self.panels["css_gold_sha256"],
            "predictions_sha256": {
                "left": {"typed": "3" * 64, "css": "5" * 64},
                "right": {"typed": "4" * 64, "css": "6" * 64},
            },
            "replicates": 5000,
            "seed": bundle.DEFAULT_SEED,
            "source_sha256": {
                name: bundle.common.sha_file(path)
                for name, path in bundle.SCORERS.items()
            },
            "coverage": {
                "typed_items": 1600,
                "typed_independent_groups": 400,
                "css_evaluation_items": 6547,
                "css_evaluation_tasks": 15,
            },
            "bootstrap": {"fixed_css_label_universe": True, "confidence_level": 0.95},
            "point": {
                "left": {"T": 0.6, "H": 0.6, "score": 60.0},
                "right": {"T": 0.5, "H": 0.5, "score": 50.0},
                "delta": {"T": 0.1, "H": 0.1, "score": 10.0},
            },
            "ci95": {"low": 1.0, "high": 18.0},
        }
        write(self.paths["aggregate"], self.aggregate)
        self.artifacts["comparison_pairs"][0]["report_sha256"]["joint_comparison"] = (
            bundle.common.sha_file(self.paths["aggregate"])
        )
        self.freeze = {
            "comparison_pairs": [
                {
                    "candidate": "new",
                    "comparator": "old",
                    "size_relation": "same",
                    "rationale": "",
                }
            ],
            "_validated_plan": {
                "model_roster": [
                    {"key": "new", "size_b": 4.2},
                    {"key": "old", "size_b": 4.0},
                ]
            },
        }

    def check(
        self, *, bind_joint: bool = True, postkey_aggregate_priority: bool = False
    ) -> dict[str, str]:
        if bind_joint:
            self.artifacts["comparison_pairs"][0]["report_sha256"][
                "joint_comparison"
            ] = bundle.common.sha_file(self.paths["aggregate"])
        return bundle._numeric_release_gate(
            pair_report=self.paths["aggregate"],
            old_typed_report=self.paths["old_typed"],
            old_css_report=self.paths["old_css"],
            new_typed_report=self.paths["new_typed"],
            new_css_report=self.paths["new_css"],
            artifact_manifest=self.artifacts,
            arena=self.arena,
            freeze=self.freeze,
            score_key="new",
            row=self.new_row,
            score_binding=self.binding,
            postkey_aggregate_priority=postkey_aggregate_priority,
        )

    def test_requires_joint_positive_paired_interval(self) -> None:
        self.assertEqual(len(self.check()), 3)
        self.aggregate["ci95"]["low"] = 0.0
        write(self.paths["aggregate"], self.aggregate)
        with self.assertRaisesRegex(ValueError, "lower bound"):
            self.check()

    def test_rejects_joint_report_changed_after_card_generation(self) -> None:
        self.aggregate["ci95"]["low"] = 2.0
        write(self.paths["aggregate"], self.aggregate)
        with self.assertRaisesRegex(
            ValueError, "score reports differ from paired card"
        ):
            self.check(bind_joint=False)

    def test_rejects_axis_regression_even_with_positive_aggregate_ci(self) -> None:
        self.new_row["axes"]["transfer"] = 0.49
        self.new_row["score"] = 100 * (0.6 * 0.49) ** 0.5
        self.aggregate["point"]["left"].update(H=0.49, score=self.new_row["score"])
        self.aggregate["point"]["delta"].update(
            H=-0.01, score=self.new_row["score"] - 50
        )
        write(self.paths["aggregate"], self.aggregate)
        with self.assertRaisesRegex(ValueError, "T/H or aggregate"):
            self.check()

    def test_rejects_slice_invalidity_and_brier_failures(self) -> None:
        self.new["by_type"]["score"]["accuracy_all"] = 0.47
        write(self.paths["new_typed"], self.new)
        self.artifacts["comparison_pairs"][0]["report_sha256"]["new_typed_report"] = (
            bundle.common.sha_file(self.paths["new_typed"])
        )
        with self.assertRaisesRegex(ValueError, "score slice regression"):
            self.check()
        self.new["by_type"]["score"]["accuracy_all"] = 0.6
        self.new["overall"]["invalid_or_missing_n"] = 100
        write(self.paths["new_typed"], self.new)
        self.artifacts["comparison_pairs"][0]["report_sha256"]["new_typed_report"] = (
            bundle.common.sha_file(self.paths["new_typed"])
        )
        with self.assertRaisesRegex(ValueError, "invalid/missing guardrail"):
            self.check()
        self.new["overall"].update(invalid_or_missing_n=0, brier=0.30)
        write(self.paths["new_typed"], self.new)
        self.artifacts["comparison_pairs"][0]["report_sha256"]["new_typed_report"] = (
            bundle.common.sha_file(self.paths["new_typed"])
        )
        with self.assertRaisesRegex(ValueError, "Brier guardrail"):
            self.check()

    def test_rejects_unbound_comparator_prediction(self) -> None:
        self.aggregate["predictions_sha256"]["right"]["css"] = "9" * 64
        write(self.paths["aggregate"], self.aggregate)
        with self.assertRaisesRegex(ValueError, "unbound"):
            self.check()

    def test_missing_probability_cannot_improve_brier(self) -> None:
        self.new["overall"].update(probability_n=1400, brier=0.15)
        write(self.paths["new_typed"], self.new)
        self.artifacts["comparison_pairs"][0]["report_sha256"]["new_typed_report"] = (
            bundle.common.sha_file(self.paths["new_typed"])
        )
        with self.assertRaisesRegex(ValueError, "coverage-adjusted typed Brier"):
            self.check()

    def test_typed_invalid_guardrail_uses_answer_count(self) -> None:
        self.new["overall"].update(invalid_or_missing_n=40, probability_n=1960)
        write(self.paths["new_typed"], self.new)
        self.artifacts["comparison_pairs"][0]["report_sha256"]["new_typed_report"] = (
            bundle.common.sha_file(self.paths["new_typed"])
        )
        self.check()
        self.new["overall"]["invalid_or_missing_n"] = 41
        write(self.paths["new_typed"], self.new)
        self.artifacts["comparison_pairs"][0]["report_sha256"]["new_typed_report"] = (
            bundle.common.sha_file(self.paths["new_typed"])
        )
        with self.assertRaisesRegex(ValueError, "invalid/missing guardrail"):
            self.check()

    def test_probability_table_uses_answer_count(self) -> None:
        table = bundle._typed_probability_table(self.new, self.old)
        self.assertIn("2,000/2,000", table)
        self.assertIn("1,600 typed items contain 2,000 scored answers", table)

    def test_explicit_postkey_policy_accepts_disclosed_slice_tradeoff(self) -> None:
        self.new["by_type"]["score"]["accuracy_all"] = 0.47
        write(self.paths["new_typed"], self.new)
        self.artifacts["comparison_pairs"][0]["report_sha256"]["new_typed_report"] = (
            bundle.common.sha_file(self.paths["new_typed"])
        )
        with self.assertRaisesRegex(ValueError, "score slice regression"):
            self.check()
        self.check(postkey_aggregate_priority=True)

    def test_postkey_policy_requires_three_point_composite_gain(self) -> None:
        self.new_row["axes"] = {"typed": 0.529, "transfer": 0.529}
        self.new_row["score"] = 52.9
        self.aggregate["point"]["left"] = {"T": 0.529, "H": 0.529, "score": 52.9}
        self.aggregate["point"]["delta"] = {
            "T": 0.029,
            "H": 0.029,
            "score": 2.9,
        }
        write(self.paths["aggregate"], self.aggregate)
        with self.assertRaisesRegex(ValueError, "below \\+3.0"):
            self.check(postkey_aggregate_priority=True)

    def test_rejects_postkey_comparator_and_wrong_measured_size(self) -> None:
        self.freeze["comparison_pairs"][0]["comparator"] = "other"
        with self.assertRaisesRegex(ValueError, "differs from pre-key pair"):
            self.check()
        self.freeze["comparison_pairs"][0]["comparator"] = "old"
        self.old_row["size_b"] = 3.0
        self.freeze["_validated_plan"]["model_roster"][1]["size_b"] = 3.0
        with self.assertRaisesRegex(ValueError, "measured 1.25 ratio"):
            self.check()

    def test_rejects_postkey_bootstrap_seed_or_draw_count(self) -> None:
        self.aggregate["seed"] += 1
        write(self.paths["aggregate"], self.aggregate)
        with self.assertRaisesRegex(ValueError, "bootstrap is missing or unbound"):
            self.check()
        self.aggregate["seed"] = bundle.DEFAULT_SEED
        self.aggregate["replicates"] += 1
        write(self.paths["aggregate"], self.aggregate)
        with self.assertRaisesRegex(ValueError, "bootstrap is missing or unbound"):
            self.check()


class FullV3PackageTests(unittest.TestCase):
    def setUp(self) -> None:
        legacy = legacy_fixture.ArenaBundleTests(
            "test_assembles_exact_same_panel_package_and_detects_tampering"
        )
        legacy.setUp()
        self.addCleanup(legacy.doCleanups)
        self.legacy = legacy
        cards = artifact_fixture.ArenaV3ArtifactTests(
            "test_generates_separate_v3_and_public_figures"
        )
        cards.setUp()
        self.addCleanup(cards.doCleanups)
        self.cards = cards
        self.root = cards.root
        model_id, revision = legacy.model_id, legacy.revision
        cards.arena["models"][0].update(
            model_id=model_id,
            revision=revision,
            native_model_sha256=legacy.native_sha,
            calibration_sha256=legacy.calibration_sha,
            adapter_sha256="6" * 64,
            size_b=0.8,
        )
        for row in cards.arena["models"]:
            row["task_scores"]["transfer"] = dict.fromkeys(
                EVALUATION_TASKS, row["axes"]["transfer"]
            )
        cards.public["models"][0].update(model_id=model_id, revision=revision)
        for family in bundle.SCORE_FAMILIES:
            predictions = self.root / f"{family}.predictions.jsonl"
            predictions.write_text('{"id":"fixture"}\n', encoding="utf-8")
            native = self.root / f"{family}.native.json"
            write(
                native,
                {
                    "model_id": model_id,
                    "model_revision": revision,
                    "model_sha256": legacy.native_sha,
                    "calibration_sha256": legacy.calibration_sha,
                    "adapter_version": legacy.record["native_adapter_version"],
                    "adapter_sha256": "6" * 64,
                    "predictions_sha256": bundle.common.sha_file(predictions),
                },
            )
            score = self.root / f"{family}.score.json"
            report = {"predictions_sha256": bundle.common.sha_file(predictions)}
            if family == "typed":
                report.update(
                    {
                        "schema_version": "typed-decision-report/2",
                        "split": "final",
                        "items": 1600,
                        "gold_sha256": cards.arena["panel_sha256"]["typed_gold_sha256"],
                        "model": {"id": model_id, "revision": revision},
                        "overall": {
                            "n": 2000,
                            "valid_n": 2000,
                            "correct_n": 1200,
                            "accuracy_all": 0.6,
                            "invalid_or_missing_n": 0,
                            "probability_n": 2000,
                            "brier": 0.2,
                        },
                        "by_family": {
                            name: {
                                "n": 800 if name == "evidence_join" else 400,
                                "valid_n": 800 if name == "evidence_join" else 400,
                                "correct_n": 480 if name == "evidence_join" else 240,
                                "accuracy_all": 0.6,
                                "invalid_or_missing_n": 0,
                            }
                            for name in FINAL_FAMILIES
                        },
                        "by_type": {
                            kind: {
                                "n": n,
                                "valid_n": n,
                                "correct_n": round(n * 0.6),
                                "accuracy_all": 0.6,
                                "invalid_or_missing_n": 0,
                            }
                            for kind, n in (
                                ("choice", 800),
                                ("noul", 800),
                                ("score", 400),
                            )
                        },
                        "macro_family_accuracy": 0.6,
                    }
                )
            elif family == "css":
                tasks = {}
                correct = 0
                for index, name in enumerate(EVALUATION_TASKS):
                    n = 947 if index == len(EVALUATION_TASKS) - 1 else 400
                    task_correct = round(n * 0.6)
                    correct += task_correct
                    tasks[name] = {
                        "role": "evaluation",
                        "n": n,
                        "valid_n": n,
                        "correct_n": task_correct,
                        "accuracy_all": task_correct / n,
                        "invalid_or_missing_n": 0,
                        "macro_f1_all": 0.6,
                    }
                report.update(
                    {
                        "score_schema_version": "css-transfer-score/2",
                        "panel_version": PANEL_VERSION,
                        "gold_sha256": cards.arena["panel_sha256"]["css_gold_sha256"],
                        "roles": {
                            "evaluation": {
                                "items": 6547,
                                "tasks": 15,
                                "valid_items": 6547,
                                "micro_accuracy_all": correct / 6547,
                                "median_task_macro_f1_all": 0.6,
                            }
                        },
                        "tasks": tasks,
                    }
                )
            else:
                report.update(
                    {
                        "score_version": bundle.PUBLIC_SCORE_VERSION,
                        "items": 231,
                        "model_id": model_id,
                        "model_revision": revision,
                        "prompts_sha256": cards.public["panel_sha256"][
                            "prompts_sha256"
                        ],
                        "targets_sha256": cards.public["panel_sha256"][
                            "targets_sha256"
                        ],
                        "panel_manifest_sha256": cards.public["panel_sha256"][
                            "panel_manifest_sha256"
                        ],
                        "prediction_manifest_sha256": bundle.common.sha_file(native),
                        "accuracy_all": 0.6,
                        "valid": 231,
                        "tier_macro_accuracy": 0.6,
                        "tiers": {
                            tier: {"accuracy_all": 0.6}
                            for tier in ("easy", "standard", "hard")
                        },
                    }
                )
            write(score, report)
            self.score_inputs = getattr(self, "score_inputs", {})
            self.score_inputs[family] = {
                "score": score,
                "predictions": predictions,
                "native_manifest": native,
            }
            if family == "public":
                cards.public["models"][0]["report_sha256"] = bundle.common.sha_file(
                    score
                )
            else:
                cards.arena["models"][0]["report_sha256"][family] = (
                    bundle.common.sha_file(score)
                )
        write(
            self.root / "old-typed.json",
            {
                "predictions_sha256": "4" * 64,
                "overall": {
                    "n": 2000,
                    "invalid_or_missing_n": 0,
                    "probability_n": 2000,
                    "brier": 0.2,
                },
                "by_type": {
                    kind: {"accuracy_all": 0.5} for kind in ("choice", "noul", "score")
                },
            },
        )
        write(
            self.root / "old-css.json",
            {
                "predictions_sha256": "5" * 64,
                "roles": {"evaluation": {"valid_items": 6547}},
            },
        )
        cards.arena["models"][1]["report_sha256"] = {
            "typed": bundle.common.sha_file(self.root / "old-typed.json"),
            "css": bundle.common.sha_file(self.root / "old-css.json"),
        }
        cards.arena["models"][1]["native_model_sha256"] = None
        cards.arena["models"][1]["adapter_sha256"] = "b" * 64
        cards.arena["models"][1]["calibration_sha256"] = "c" * 64
        typed_compare = json.loads((self.root / "typed-pair.json").read_text())
        typed_compare["models"]["left"] = model_id
        typed_compare["left_sha256"] = bundle.common.sha_file(
            self.score_inputs["typed"]["predictions"]
        )
        typed_compare["right_sha256"] = "4" * 64
        write(self.root / "typed-pair.json", typed_compare)
        css_compare = json.loads((self.root / "transfer-pair.json").read_text())
        css_compare["model_a"] = model_id
        css_compare["predictions_a_sha256"] = bundle.common.sha_file(
            self.score_inputs["css"]["predictions"]
        )
        css_compare["predictions_b_sha256"] = "5" * 64
        write(self.root / "transfer-pair.json", css_compare)
        joint_compare = json.loads((self.root / "joint-pair.json").read_text())
        joint_compare["models"]["left"] = model_id
        joint_compare["predictions_sha256"]["left"] = {
            "typed": bundle.common.sha_file(self.score_inputs["typed"]["predictions"]),
            "css": bundle.common.sha_file(self.score_inputs["css"]["predictions"]),
        }
        joint_compare["predictions_sha256"]["right"] = {
            "typed": "4" * 64,
            "css": "5" * 64,
        }
        joint_compare["coverage"] = {
            "typed_items": 1600,
            "typed_independent_groups": 400,
            "css_evaluation_items": 6547,
            "css_evaluation_tasks": 15,
        }
        joint_compare["bootstrap"] = {
            "fixed_css_label_universe": True,
            "confidence_level": 0.95,
        }
        joint_compare["ci95"] = {"low": 1.0, "high": 18.0}
        write(self.root / "joint-pair.json", joint_compare)
        cards.config["comparison_pairs"][0]["new_typed_report"] = "typed.score.json"
        cards.config["comparison_pairs"][0]["new_transfer_report"] = "css.score.json"
        write(self.root / "arena.json", cards.arena)
        write(self.root / "public.json", cards.public)
        write(self.root / "config.json", cards.config)
        self.artifacts = self.root / "v3-artifacts"
        self.artifact_manifest = generate(self.root / "config.json", self.artifacts)
        panel = cards.arena["panel_sha256"]
        self.freeze_path = self.root / "v3-freeze.json"
        self.freeze = {
            "schema_version": "jevarena-v3-freeze/2",
            "status": "prekey_frozen",
            "candidate_lock_sha256": "a" * 64,
            "protocol_sha256": bundle.common.sha_file(bundle.POLICY),
            "formula": "100*sqrt(T*H)",
            "paired_bootstrap": {"replicates": 5000, "seed": 20260927},
            "prekey_frozen_at_utc": "2026-09-02T01:30:00+00:00",
            "score_sources_sha256": {
                name: bundle.common.sha_file(path)
                for name, path in bundle.SCORERS.items()
            },
            "panels": panel,
            "models": {
                "new": {
                    "model_id": model_id,
                    "revision": revision,
                    "native_model_sha256": legacy.native_sha,
                    "adapter_sha256": "6" * 64,
                    "calibration_sha256": legacy.calibration_sha,
                    "predictions_sha256": {
                        family: bundle.common.sha_file(
                            self.score_inputs[family]["predictions"]
                        )
                        for family in bundle.SCORE_FAMILIES
                    },
                },
                "old": {
                    "model_id": cards.arena["models"][1]["model_id"],
                    "revision": cards.arena["models"][1]["revision"],
                    "native_model_sha256": None,
                    "adapter_sha256": "b" * 64,
                    "calibration_sha256": "c" * 64,
                    "predictions_sha256": {
                        "typed": "4" * 64,
                        "css": "5" * 64,
                        "public": "7" * 64,
                    },
                },
            },
        }
        source_root = Path(__file__).resolve().parents[2]
        pairs = [
            {
                "candidate": "new",
                "comparator": "old",
                "size_relation": "same",
                "rationale": "",
            }
        ]
        pair_sha = _pair_digest(pairs)
        prompts = {
            "typed": "8" * 64,
            "css": "9" * 64,
            "public": cards.public["panel_sha256"]["prompts_sha256"],
        }
        self.plan_path = self.root / "v3-plan.json"
        self.audit_path = self.root / "v3-prekey-audit.json"
        write(
            self.plan_path,
            {
                "plan_version": "decision2-first-release-v3-plan/2",
                "formula": "100*sqrt(T*H)",
                "candidate_freeze_sha256": self.freeze["candidate_lock_sha256"],
                "gate_document_sha256": self.freeze["protocol_sha256"],
                "comparison_pairs": pairs,
                "comparison_pairs_sha256": pair_sha,
                "paired_ci_commands_after_prekey_freeze": [
                    {
                        **pairs[0],
                        "command": "python -m jev_arena.compare_v3 --replicates 5000 --seed 20260927",
                    }
                ],
                "source_root": str(source_root),
                "source_sha256": {
                    name: bundle.common.sha_file(source_root / name)
                    for name in REQUIRED_PROTOCOL_SOURCES
                },
                "model_roster": [
                    {
                        "key": row["key"],
                        "group": row["group"],
                        "size_b": row["size_b"],
                        **{
                            field: self.freeze["models"][row["key"]][field]
                            for field in (
                                "model_id",
                                "revision",
                                "native_model_sha256",
                                "adapter_sha256",
                                "calibration_sha256",
                            )
                        },
                    }
                    for row in cards.arena["models"]
                ],
                "css_prompts": {"sha256": prompts["css"]},
                "public_panel": {"prompts_sha256": prompts["public"]},
            },
        )
        write(
            self.audit_path,
            {
                "status": "gold_free_prekey_predictions_verified",
                "comparison_pairs_sha256": pair_sha,
                "prompt_sha256": prompts,
                "models": {
                    key: value["predictions_sha256"]
                    for key, value in self.freeze["models"].items()
                },
                "raw_hashes_sha256": "b" * 64,
            },
        )
        chronology_path = self.root / "v3-chronology.json"
        write(
            chronology_path,
            {
                "schema_version": "jevarena-v3-prekey-chronology/1",
                "candidate_lock": {
                    "at_utc": "2026-09-02T00:00:00+00:00",
                    "sha256": self.freeze["candidate_lock_sha256"],
                },
                "prediction_seal": {
                    "at_utc": "2026-09-02T01:00:00+00:00",
                    "sha256": "b" * 64,
                },
                "audit_seal": {
                    "at_utc": "2026-09-02T01:20:00+00:00",
                    "sha256": bundle.common.sha_file(self.audit_path),
                },
            },
        )
        self.freeze.update(
            {
                "chronology": {
                    "path": str(chronology_path),
                    "sha256": bundle.common.sha_file(chronology_path),
                },
                "plan": {
                    "path": str(self.plan_path),
                    "sha256": bundle.common.sha_file(self.plan_path),
                },
                "prediction_audit": {
                    "path": str(self.audit_path),
                    "sha256": bundle.common.sha_file(self.audit_path),
                },
                "comparison_pairs": pairs,
                "comparison_pairs_sha256": pair_sha,
                "prompt_sha256": prompts,
                "raw_prediction_hashes_sha256": "b" * 64,
            }
        )
        write(self.freeze_path, self.freeze)
        cards.arena["freeze_sha256"] = bundle.common.sha_file(self.freeze_path)
        write(self.root / "arena.json", cards.arena)
        # Reissue generator artifacts with the final pre-key freeze digest.
        self.artifacts.rename(self.root / "pre-freeze-artifacts")
        self.artifacts = self.root / "v3-artifacts"
        self.artifact_manifest = generate(self.root / "config.json", self.artifacts)
        self.aggregate_path = self.root / "joint-pair.json"
        panel_sha = hashlib.sha256(
            json.dumps(panel, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        write(
            self.aggregate_path,
            {
                "schema_version": "jevarena-v3-paired-aggregate/1",
                "models": {
                    "left": model_id,
                    "right": cards.arena["models"][1]["model_id"],
                },
                "panel_sha256": panel_sha,
                "typed_gold_sha256": panel["typed_gold_sha256"],
                "css_gold_sha256": panel["css_gold_sha256"],
                "predictions_sha256": {
                    "left": {
                        family: bundle.common.sha_file(
                            self.score_inputs[family]["predictions"]
                        )
                        for family in ("typed", "css")
                    },
                    "right": {"typed": "4" * 64, "css": "5" * 64},
                },
                "source_sha256": {
                    name: bundle.common.sha_file(path)
                    for name, path in bundle.SCORERS.items()
                },
                "coverage": {
                    "typed_items": 1600,
                    "typed_independent_groups": 400,
                    "css_evaluation_items": 6547,
                    "css_evaluation_tasks": 15,
                },
                "bootstrap": {
                    "fixed_css_label_universe": True,
                    "confidence_level": 0.95,
                },
                "replicates": 5000,
                "seed": bundle.DEFAULT_SEED,
                "point": {
                    "left": {"T": 0.6, "H": 0.6, "score": 60.0},
                    "right": {"T": 0.5, "H": 0.5, "score": 50.0},
                    "delta": {"T": 0.1, "H": 0.1, "score": 10.0},
                },
                "ci95": {"low": 1.0, "high": 18.0},
            },
        )
        self.assertEqual(
            bundle.common.sha_file(self.aggregate_path),
            self.artifact_manifest["comparison_pairs"][0]["report_sha256"][
                "joint_comparison"
            ],
        )
        self.comparison_binding = {
            "aggregate_pair_report_sha256": bundle.common.sha_file(self.aggregate_path),
            "old_typed_report_sha256": bundle.common.sha_file(
                self.root / "old-typed.json"
            ),
            "old_css_report_sha256": bundle.common.sha_file(self.root / "old-css.json"),
        }
        self.score_binding = {
            family: {
                "score_sha256": bundle.common.sha_file(paths["score"]),
                "predictions_sha256": bundle.common.sha_file(paths["predictions"]),
                "native_manifest_sha256": bundle.common.sha_file(
                    paths["native_manifest"]
                ),
                "adapter_sha256": "6" * 64,
            }
            for family, paths in self.score_inputs.items()
        }
        self.candidate = {
            "model_id": model_id,
            "model_revision": revision,
            "native_model_sha256": legacy.native_sha,
            "model_files_sha256": legacy.files_digest,
            "calibration_sha256": legacy.calibration_sha,
            "adapter_version": legacy.record["native_adapter_version"],
            "adapter_sha256": "6" * 64,
        }
        self.context_sha = bundle._context_digest(
            bundle.common.sha_file(legacy.record_path),
            bundle.common.sha_file(legacy.parity_path),
            bundle.common.sha_file(self.artifacts / "manifest.json"),
            bundle.common.sha_file(self.root / "arena.json"),
            bundle.common.sha_file(self.root / "public.json"),
            bundle.common.sha_file(self.freeze_path),
            self.score_binding,
            self.comparison_binding,
        )
        self.gate_evidence = {
            name: self.root / f"v3-gate-{name}.json" for name in bundle.GATE_CHECKS
        }
        self.prediction_seals = {
            key: {
                family: {
                    "predictions_sha256": model["predictions_sha256"][family],
                    "sealed_at_utc": "2026-09-02T01:00:00+00:00",
                    **(
                        {
                            "native_manifest_sha256": self.score_binding[family][
                                "native_manifest_sha256"
                            ]
                        }
                        if key == "new"
                        else {}
                    ),
                }
                for family in bundle.SCORE_FAMILIES
            }
            for key, model in self.freeze["models"].items()
        }
        self.timestamp_log_path = self.root / "v3-timestamp-log.json"
        write(
            self.timestamp_log_path,
            {
                "schema_version": bundle.TIMESTAMP_LOG_VERSION,
                "candidate_lock_sha256": self.freeze["candidate_lock_sha256"],
                "candidate_locked_at_utc": "2026-09-02T00:00:00+00:00",
                "prediction_seals": self.prediction_seals,
                "prekey_freeze_sha256": bundle.common.sha_file(self.freeze_path),
                "prekey_frozen_at_utc": self.freeze["prekey_frozen_at_utc"],
                "first_label_opened_at_utc": "2026-09-02T02:00:00+00:00",
            },
        )
        for name, path in self.gate_evidence.items():
            evidence = {
                "schema_version": (
                    bundle.FREEZE_AUDIT_VERSION
                    if name == "candidate_freeze"
                    else bundle.CHECK_VERSION
                ),
                "status": "passed",
                "check": name,
                "model_id": model_id,
                "model_revision": revision,
                "release_context_sha256": self.context_sha,
                "reviewer_identity_sha256": "e" * 64,
                "source_evidence_sha256": "f" * 64,
                "reviewed_at_utc": "2026-09-02T04:00:00+00:00",
            }
            if name == "candidate_freeze":
                evidence.update(
                    {
                        "pretest_freeze_sha256": bundle.common.sha_file(
                            self.freeze_path
                        ),
                        "candidate_lock_sha256": self.freeze["candidate_lock_sha256"],
                        "candidate": self.candidate,
                        "protocol_sha256": self.freeze["protocol_sha256"],
                        "timestamp_log_path": str(self.timestamp_log_path),
                        "timestamp_log_sha256": bundle.common.sha_file(
                            self.timestamp_log_path
                        ),
                        "candidate_locked_at_utc": "2026-09-02T00:00:00+00:00",
                        "first_label_opened_at_utc": "2026-09-02T02:00:00+00:00",
                        "prediction_seals": self.prediction_seals,
                    }
                )
            if name == "release_thresholds":
                evidence.update(
                    {
                        "predeclared_policy_sha256": self.freeze["protocol_sha256"],
                        "comparison_inputs_sha256": self.comparison_binding,
                    }
                )
            write(path, evidence)
        self.gate_path = self.root / "v3-gate.json"
        self.gate = {
            "schema_version": bundle.GATE_VERSION,
            "status": "passed",
            "model_id": model_id,
            "model_revision": revision,
            "package_record_sha256": bundle.common.sha_file(legacy.record_path),
            "parity_receipt_sha256": bundle.common.sha_file(legacy.parity_path),
            "artifact_manifest_sha256": bundle.common.sha_file(
                self.artifacts / "manifest.json"
            ),
            "arena_rank_sha256": bundle.common.sha_file(self.root / "arena.json"),
            "jevbench_public_rank_sha256": bundle.common.sha_file(
                self.root / "public.json"
            ),
            "pretest_freeze_sha256": bundle.common.sha_file(self.freeze_path),
            "native_model_sha256": legacy.native_sha,
            "model_files_sha256": legacy.files_digest,
            "checks": {
                name: {
                    "status": "passed",
                    "evidence_sha256": bundle.common.sha_file(path),
                }
                for name, path in self.gate_evidence.items()
            },
        }
        write(self.gate_path, self.gate)

    def assemble(self) -> dict:
        with patch.object(bundle.common, "_parameter_count", return_value=800_000_000):
            return bundle.assemble(
                model_dir=self.legacy.model,
                artifacts=self.artifacts,
                arena_rank=self.root / "arena.json",
                public_rank=self.root / "public.json",
                package_record=self.legacy.record_path,
                parity_receipt=self.legacy.parity_path,
                release_gate=self.gate_path,
                provenance_inputs=self.legacy.provenance_inputs,
                freeze_manifest=self.freeze_path,
                gate_evidence=self.gate_evidence,
                score_inputs=self.score_inputs,
                score_key="new",
                output=self.root / "v3-package",
                aggregate_pair_report=self.aggregate_path,
                old_typed_report=self.root / "old-typed.json",
                old_css_report=self.root / "old-css.json",
            )

    def test_stages_v3_package_without_authored_or_decision_bench(self) -> None:
        manifest = self.assemble()
        self.assertEqual(manifest["bundle_version"], bundle.VERSION)
        self.assertEqual(
            set(manifest["score_inputs_sha256"]), {"typed", "css", "public"}
        )
        card = (self.root / "v3-package/README.md").read_text()
        self.assertIn("8,147", card)
        self.assertIn("Authored questions are reserved for v3.1", card)
        self.assertIn("Coverage-adjusted Brier", card)
        self.assertNotIn("pareto", card.lower())
        self.assertIn("decision-2-sticker-crossroads-fox-v5.png", card)
        self.assertTrue(
            (
                self.root / "v3-package/decision-2-sticker-crossroads-fox-v5.png"
            ).is_file()
        )
        bundle.verify(self.root / "v3-package")

    def test_rejects_unbound_freeze_and_blocked_review(self) -> None:
        self.freeze["candidate_lock_sha256"] = "9" * 64
        write(self.freeze_path, self.freeze)
        with self.assertRaisesRegex(ValueError, "freeze receipt digest changed"):
            self.assemble()

    def test_rejects_threshold_receipt_without_joint_comparison_binding(self) -> None:
        path = self.gate_evidence["release_thresholds"]
        evidence = json.loads(path.read_text())
        evidence["comparison_inputs_sha256"].pop("aggregate_pair_report_sha256")
        write(path, evidence)
        self.gate["checks"]["release_thresholds"]["evidence_sha256"] = (
            bundle.common.sha_file(path)
        )
        write(self.gate_path, self.gate)
        with self.assertRaisesRegex(ValueError, "frozen policy"):
            self.assemble()

    def test_rejects_review_disabled_despite_valid_scores(self) -> None:
        self.gate["checks"]["train_eval_overlap"]["status"] = "blocked"
        write(self.gate_path, self.gate)
        with self.assertRaisesRegex(ValueError, "review is blocked"):
            self.assemble()

    def test_rejects_postkey_audit_tamper_and_false_chronology(self) -> None:
        audit = json.loads(self.audit_path.read_text())
        audit["models"]["new"]["public"] = "0" * 64
        write(self.audit_path, audit)
        with self.assertRaisesRegex(
            ValueError, "gold-free audit: receipt digest changed"
        ):
            self.assemble()

        audit["models"]["new"]["public"] = self.freeze["models"]["new"][
            "predictions_sha256"
        ]["public"]
        write(self.audit_path, audit)
        evidence_path = self.gate_evidence["candidate_freeze"]
        evidence = json.loads(evidence_path.read_text())
        evidence["candidate_locked_at_utc"] = "2026-09-02T01:15:00+00:00"
        log = json.loads(self.timestamp_log_path.read_text())
        log["candidate_locked_at_utc"] = evidence["candidate_locked_at_utc"]
        write(self.timestamp_log_path, log)
        evidence["timestamp_log_sha256"] = bundle.common.sha_file(
            self.timestamp_log_path
        )
        write(evidence_path, evidence)
        self.gate["checks"]["candidate_freeze"]["evidence_sha256"] = (
            bundle.common.sha_file(evidence_path)
        )
        write(self.gate_path, self.gate)
        with self.assertRaisesRegex(ValueError, "chronology is invalid"):
            self.assemble()


if __name__ == "__main__":
    unittest.main()
