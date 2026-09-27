"""A sealed-core score is versioned, same-panel and bound to frozen predictions."""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

from benchmark.generate import FINAL_FAMILIES
from transfer.build import EVALUATION_TASKS, PANEL_VERSION

from jev_arena.arena_v3 import (
    FREEZE_VERSION,
    REQUIRED_PROTOCOL_SOURCES,
    ROSTER_VERSION,
    SCORER_SOURCE_PATHS,
    _pair_digest,
    rank,
)


def write(path: Path, value: dict) -> str:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summary(n: int, correct: int) -> dict:
    return {
        "n": n,
        "valid_n": n,
        "correct_n": correct,
        "invalid_or_missing_n": 0,
        "accuracy_all": correct / n,
    }


def panel(root: Path, key: str, quality: float) -> tuple[dict, dict]:
    model_id, revision = f"example/{key}", f"revision-{key}"
    typed_by_family = {
        family: summary(400, round(400 * quality)) for family in FINAL_FAMILIES
    }
    typed_total = 4 * round(400 * quality)
    choice_correct = round(534 * quality)
    noul_correct = round(533 * quality)
    typed_by_type = {
        "choice": summary(534, choice_correct),
        "noul": summary(533, noul_correct),
        "score": summary(533, typed_total - choice_correct - noul_correct),
    }
    typed = {
        "schema_version": "typed-decision-report/2",
        "split": "final",
        "items": 1600,
        "gold_sha256": "a" * 64,
        "predictions_sha256": ("1" if key == "small" else "2") * 64,
        "model": {"id": model_id, "revision": revision},
        "overall": summary(1600, typed_total),
        "by_family": typed_by_family,
        "by_type": typed_by_type,
        "macro_family_accuracy": typed_total / 1600,
    }
    css_tasks = {}
    css_correct = 0
    for index, name in enumerate(EVALUATION_TASKS):
        n = 437 if index < 7 else 436
        correct = round(n * quality)
        css_correct += correct
        css_tasks[name] = {
            "role": "evaluation",
            **summary(n, correct),
            "macro_f1_all": quality,
        }
    css = {
        "score_schema_version": "css-transfer-score/2",
        "panel_version": PANEL_VERSION,
        "gold_sha256": "b" * 64,
        "predictions_sha256": ("3" if key == "small" else "4") * 64,
        "tasks": css_tasks,
        "roles": {
            "evaluation": {
                "items": 6547,
                "tasks": 15,
                "valid_items": 6547,
                "micro_accuracy_all": css_correct / 6547,
                "median_task_macro_f1_all": quality,
            }
        },
    }
    for name, report in (("typed", typed), ("css", css)):
        write(root / f"{key}-{name}.json", report)
    return typed, css


def fixture(root: Path) -> tuple[Path, Path]:
    rows, frozen_models = [], {}
    for key, quality, size in (("small", 0.6, 0.6), ("large", 0.8, 4.0)):
        typed, css = panel(root, key, quality)
        model_id, revision = f"example/{key}", f"revision-{key}"
        rows.append(
            {
                "key": key,
                "label": key,
                "group": "open",
                "model_id": model_id,
                "revision": revision,
                "size_b": size,
                "typed_report": f"{key}-typed.json",
                "css_report": f"{key}-css.json",
            }
        )
        frozen_models[key] = {
            "model_id": model_id,
            "revision": revision,
            "native_model_sha256": None,
            "adapter_sha256": "8" * 64,
            "calibration_sha256": None,
            "predictions_sha256": {
                "typed": typed["predictions_sha256"],
                "css": css["predictions_sha256"],
                "public": ("5" if key == "small" else "6") * 64,
            },
        }
    source_root = Path(__file__).resolve().parents[2]
    prompt_shas = {"typed": "c" * 64, "css": "d" * 64, "public": "f" * 64}
    pair_sha = _pair_digest([])
    plan = {
        "plan_version": "decision2-first-release-v3-plan/1",
        "candidate_freeze_sha256": "e" * 64,
        "gate_document_sha256": "7" * 64,
        "comparison_pairs": [],
        "comparison_pairs_sha256": pair_sha,
        "source_root": str(source_root),
        "source_sha256": {
            name: hashlib.sha256((source_root / name).read_bytes()).hexdigest()
            for name in REQUIRED_PROTOCOL_SOURCES
        },
        "model_roster": [
            {
                "key": row["key"],
                "group": row["group"],
                "size_b": row["size_b"],
                **{
                    name: frozen_models[row["key"]][name]
                    for name in (
                        "model_id",
                        "revision",
                        "native_model_sha256",
                        "adapter_sha256",
                        "calibration_sha256",
                    )
                },
            }
            for row in rows
        ],
        "css_prompts": {"sha256": prompt_shas["css"]},
        "public_panel": {"prompts_sha256": prompt_shas["public"]},
    }
    audit = {
        "status": "gold_free_prekey_predictions_verified",
        "comparison_pairs_sha256": pair_sha,
        "prompt_sha256": prompt_shas,
        "models": {
            key: model["predictions_sha256"] for key, model in frozen_models.items()
        },
        "raw_hashes_sha256": "9" * 64,
    }
    plan_path, audit_path = root / "plan.json", root / "audit.json"
    plan_sha, audit_sha = write(plan_path, plan), write(audit_path, audit)
    freeze = {
        "schema_version": FREEZE_VERSION,
        "status": "prekey_frozen",
        "plan": {"path": str(plan_path), "sha256": plan_sha},
        "prediction_audit": {"path": str(audit_path), "sha256": audit_sha},
        "comparison_pairs": [],
        "comparison_pairs_sha256": pair_sha,
        "prekey_frozen_at_utc": "2026-09-02T01:30:00+00:00",
        "prompt_sha256": prompt_shas,
        "raw_prediction_hashes_sha256": audit["raw_hashes_sha256"],
        "score_sources_sha256": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in SCORER_SOURCE_PATHS.items()
        },
        "protocol_sha256": "7" * 64,
        "candidate_lock_sha256": "e" * 64,
        "panels": {
            "typed_gold_sha256": "a" * 64,
            "css_gold_sha256": "b" * 64,
        },
        "models": frozen_models,
    }
    freeze_path, manifest_path = root / "freeze.json", root / "roster.json"
    freeze_sha = write(freeze_path, freeze)
    write(
        manifest_path,
        {
            "schema_version": ROSTER_VERSION,
            "phase": "release",
            "freeze_receipt": "freeze.json",
            "freeze_sha256": freeze_sha,
            "models": rows,
        },
    )
    return manifest_path, freeze_path


class JevArenaV3Test(unittest.TestCase):
    def test_two_sealed_axes_rank_without_public_benchmarks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest, _ = fixture(Path(temporary))
            report = rank(manifest)
            self.assertEqual(report["schema_version"], "jevarena-ranking/3")
            self.assertEqual(
                report["status"], "scored_pending_independent_release_audit"
            )
            self.assertEqual(
                [row["key"] for row in report["models"]], ["large", "small"]
            )
            self.assertEqual(report["policy"]["axes"], ["typed", "transfer"])
            top = report["models"][0]
            self.assertEqual(top["coverage"]["sealed_core_items"], 8147)
            self.assertEqual(len(top["task_scores"]["transfer"]), 15)
            self.assertEqual(
                set(top["task_scores"]["typed"]), {"choice", "noul", "score"}
            )
            self.assertTrue(
                math.isclose(
                    top["score"], 100 * math.sqrt(math.prod(top["axes"].values()))
                )
            )
            self.assertTrue(all(row["pareto_frontier"] for row in report["models"]))

    def test_incomplete_final_panel_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, _ = fixture(root)
            typed_path = root / "small-typed.json"
            typed = json.loads(typed_path.read_text())
            typed["split"] = "dev"
            write(typed_path, typed)
            with self.assertRaisesRegex(ValueError, "FINAL panel"):
                rank(manifest)
            typed["split"] = "final"
            typed["gold_sha256"] = "9" * 64
            write(typed_path, typed)
            with self.assertRaisesRegex(ValueError, "panel digest"):
                rank(manifest)

    def test_mismatched_prediction_and_model_identity_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, _ = fixture(root)
            css_path = root / "large-css.json"
            css = json.loads(css_path.read_text())
            css["predictions_sha256"] = "9" * 64
            write(css_path, css)
            with self.assertRaisesRegex(ValueError, "predictions differ"):
                rank(manifest)
            css["predictions_sha256"] = "4" * 64
            write(css_path, css)
            typed_path = root / "large-typed.json"
            typed = json.loads(typed_path.read_text())
            typed["model"]["id"] = "example/wrong"
            write(typed_path, typed)
            with self.assertRaisesRegex(ValueError, "model identity"):
                rank(manifest)

    def test_component_and_freeze_mutations_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, freeze_path = fixture(root)
            css_path = root / "small-css.json"
            css = json.loads(css_path.read_text())
            css["roles"]["evaluation"]["median_task_macro_f1_all"] = 0.1
            write(css_path, css)
            with self.assertRaisesRegex(ValueError, "median_task_macro_f1_all"):
                rank(manifest)
            fixture(root)
            freeze = json.loads(freeze_path.read_text())
            freeze["score_sources_sha256"]["paired_v3"] = "9" * 64
            freeze_sha = write(freeze_path, freeze)
            roster = json.loads(manifest.read_text())
            roster["freeze_sha256"] = freeze_sha
            write(manifest, roster)
            with self.assertRaisesRegex(ValueError, "scoring source changed"):
                rank(manifest)

    def test_prekey_plan_and_gold_free_audit_are_required(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, freeze_path = fixture(root)
            freeze = json.loads(freeze_path.read_text())
            roster = json.loads(manifest.read_text())
            freeze.pop("plan")
            roster["freeze_sha256"] = write(freeze_path, freeze)
            write(manifest, roster)
            with self.assertRaisesRegex(ValueError, "pre-key plan"):
                rank(manifest)

            manifest, freeze_path = fixture(root)
            freeze = json.loads(freeze_path.read_text())
            audit_path = Path(freeze["prediction_audit"]["path"])
            audit = json.loads(audit_path.read_text())
            audit["models"]["small"]["public"] = "0" * 64
            freeze["prediction_audit"]["sha256"] = write(audit_path, audit)
            roster = json.loads(manifest.read_text())
            roster["freeze_sha256"] = write(freeze_path, freeze)
            write(manifest, roster)
            with self.assertRaisesRegex(ValueError, "audited full-panel predictions"):
                rank(manifest)

    def test_prekey_plan_detects_unfrozen_gate_code(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, freeze_path = fixture(root)
            freeze = json.loads(freeze_path.read_text())
            plan_path = Path(freeze["plan"]["path"])
            plan = json.loads(plan_path.read_text())
            plan["source_sha256"]["publication/bundle_arena_v3.py"] = "0" * 64
            freeze["plan"]["sha256"] = write(plan_path, plan)
            roster = json.loads(manifest.read_text())
            roster["freeze_sha256"] = write(freeze_path, freeze)
            write(manifest, roster)
            with self.assertRaisesRegex(ValueError, "protocol source changed"):
                rank(manifest)

    def test_prekey_pair_mapping_covers_each_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, freeze_path = fixture(root)
            freeze = json.loads(freeze_path.read_text())
            plan_path = Path(freeze["plan"]["path"])
            plan = json.loads(plan_path.read_text())
            plan["model_roster"][0]["group"] = "decision2"
            freeze["plan"]["sha256"] = write(plan_path, plan)
            roster = json.loads(manifest.read_text())
            roster["freeze_sha256"] = write(freeze_path, freeze)
            write(manifest, roster)
            with self.assertRaisesRegex(ValueError, "pairs do not cover"):
                rank(manifest)

    def test_prekey_freeze_time_is_utc(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, freeze_path = fixture(root)
            freeze = json.loads(freeze_path.read_text())
            freeze["prekey_frozen_at_utc"] = "2026-09-02T09:30:00+08:00"
            roster = json.loads(manifest.read_text())
            roster["freeze_sha256"] = write(freeze_path, freeze)
            write(manifest, roster)
            with self.assertRaisesRegex(ValueError, "freeze time needs UTC"):
                rank(manifest)

    def test_invalid_answers_stay_in_full_denominator(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, _ = fixture(root)
            path = root / "small-typed.json"
            typed = json.loads(path.read_text())
            for group in (
                typed["overall"],
                typed["by_family"][FINAL_FAMILIES[0]],
                typed["by_type"]["choice"],
            ):
                group["valid_n"] -= 1
                group["invalid_or_missing_n"] += 1
            write(path, typed)
            result = rank(manifest)
            small = next(row for row in result["models"] if row["key"] == "small")
            self.assertEqual(small["axes"]["typed"], 0.6)
            typed["overall"]["invalid_or_missing_n"] = 0
            write(path, typed)
            with self.assertRaisesRegex(ValueError, "denominator"):
                rank(manifest)

    def test_decision2_needs_native_fingerprint_and_complete_css(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, freeze_path = fixture(root)
            roster = json.loads(manifest.read_text())
            roster["models"][0]["group"] = "decision2"
            roster["models"][1]["group"] = "decision1"
            freeze = json.loads(freeze_path.read_text())
            plan_path = Path(freeze["plan"]["path"])
            plan = json.loads(plan_path.read_text())
            plan["model_roster"][0]["group"] = "decision2"
            plan["model_roster"][1]["group"] = "decision1"
            pairs = [
                {
                    "candidate": "small",
                    "comparator": "large",
                    "size_relation": "nearest",
                    "rationale": "Synthetic nearest-size baseline for this test",
                }
            ]
            plan["comparison_pairs"] = freeze["comparison_pairs"] = pairs
            plan["comparison_pairs_sha256"] = freeze["comparison_pairs_sha256"] = (
                _pair_digest(pairs)
            )
            audit_path = Path(freeze["prediction_audit"]["path"])
            audit = json.loads(audit_path.read_text())
            audit["comparison_pairs_sha256"] = _pair_digest(pairs)
            freeze["prediction_audit"]["sha256"] = write(audit_path, audit)
            freeze["plan"]["sha256"] = write(plan_path, plan)
            roster["freeze_sha256"] = write(freeze_path, freeze)
            write(manifest, roster)
            with self.assertRaisesRegex(ValueError, "native model fingerprint"):
                rank(manifest)
            freeze["models"]["small"]["native_model_sha256"] = "9" * 64
            plan["model_roster"][0]["native_model_sha256"] = "9" * 64
            freeze["plan"]["sha256"] = write(plan_path, plan)
            roster["freeze_sha256"] = write(freeze_path, freeze)
            write(manifest, roster)
            css_path = root / "small-css.json"
            css = json.loads(css_path.read_text())
            css["tasks"].pop(next(iter(EVALUATION_TASKS)))
            write(css_path, css)
            with self.assertRaisesRegex(ValueError, "exactly 15 frozen tasks"):
                rank(manifest)

    def test_v2_roster_cannot_be_relabelled_as_v3(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest, _ = fixture(root)
            roster = json.loads(manifest.read_text())
            roster["schema_version"] = "jevarena-ranking/2"
            write(manifest, roster)
            with self.assertRaisesRegex(ValueError, "release roster schema"):
                rank(manifest)


if __name__ == "__main__":
    unittest.main()
