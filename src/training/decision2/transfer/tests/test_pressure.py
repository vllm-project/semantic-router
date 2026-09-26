from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from transfer import pressure_build as builder
from transfer import pressure_score as scorer


def fake_item(uid: str, *, leaked: bool = False) -> dict:
    text = f"a short diagnostic text for {uid}"
    return {
        "uid": uid,
        "domain": "clinc",
        "text": text,
        "text_sha256": builder.digest(text.encode()),
        "gold": "correct label",
        "leaked": leaked,
        "distractors": {
            "near": [f"near label {i}" for i in range(255)],
            "far": [f"far label {i}" for i in range(255)],
            "ext": [f"ext label {i}" for i in range(255)],
        },
    }


def write_panel(
    root: Path, name: str, rows: list[tuple[dict, dict]]
) -> tuple[Path, Path, Path]:
    prompts = root / f"{name}.prompts.jsonl"
    gold = root / f"{name}.gold.jsonl"
    prompts.write_text(
        "".join(builder.compact(row[0]) + "\n" for row in rows), encoding="utf-8"
    )
    gold.write_text(
        "".join(builder.compact(row[1]) + "\n" for row in rows), encoding="utf-8"
    )
    controls = sum(row[1]["repeat"] for row in rows)
    manifest = root / "pressure-manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "panel_version": builder.PANEL_VERSION,
                "role": "public_development_diagnostic_only",
                "source_revision": builder.SOURCE_REVISION,
                "source_sha256": builder.SOURCE_SHA256,
                "source_data_license": "CC BY-SA 4.0",
                "quality_gate": {
                    "source_gates_passed": False,
                    "failed_gate": "G3_textfree_gold_picker",
                },
                "slices": {
                    name: {
                        "main_requests": len(rows) - controls,
                        "control_requests": controls,
                        "files": {
                            "prompts": {
                                "name": prompts.name,
                                "sha256": builder.file_sha256(prompts),
                            },
                            "gold": {
                                "name": gold.name,
                                "sha256": builder.file_sha256(gold),
                            },
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return manifest, prompts, gold


def prediction(
    prompt: dict, gold: dict, *, choose_gold: bool = True, no_probs: bool = False
) -> dict:
    labels = gold["labels"]
    choice = (
        gold["gold"]
        if choose_gold
        else next(label for label in labels if label != gold["gold"])
    )
    answer = {"type": "choice", "choice": choice}
    if not no_probs:
        answer["probabilities"] = {label: float(label == choice) for label in labels}
    return {
        "id": prompt["id"],
        "answers": {"decision": answer},
        "model_id": "test",
        "model_revision": "r1",
        "adapter_version": "a1",
        "source_input_sha256": gold["input_sha256"],
        "latency_ms": 2.5,
    }


class PressureTests(unittest.TestCase):
    def test_deterministic_gold_free_compiler_and_repeated_order(self) -> None:
        item = fake_item("clinc:0")
        prompt, gold = builder.compile_one(item, "rq2_shared_order", "near", 16, 0)
        repeat_prompt, repeat_gold = builder.compile_one(
            item, "rq2_shared_order", "near", 16, 0, repeat=True
        )
        changed_prompt, changed_gold = builder.compile_one(
            item, "rq2_shared_order", "near", 16, 1
        )
        self.assertEqual(set(prompt), {"id", "state", "questions"})
        self.assertEqual(set(prompt["questions"]), {"decision"})
        self.assertNotIn("gold", prompt)
        self.assertNotIn("uid", prompt)
        self.assertEqual(
            {key: value for key, value in prompt.items() if key != "id"},
            {key: value for key, value in repeat_prompt.items() if key != "id"},
        )
        self.assertNotEqual(prompt["id"], repeat_prompt["id"])
        self.assertEqual(gold["input_sha256"], repeat_gold["input_sha256"])
        self.assertEqual(gold["option_set_sha256"], changed_gold["option_set_sha256"])
        self.assertNotEqual(
            gold["option_order_sha256"], changed_gold["option_order_sha256"]
        )
        self.assertEqual(
            (prompt, gold), builder.compile_one(item, "rq2_shared_order", "near", 16, 0)
        )
        self.assertEqual(
            gold["input_sha256"],
            builder.digest(
                {"state": prompt["state"], "questions": prompt["questions"]}
            ),
        )
        self.assertEqual(len(changed_prompt["questions"]["decision"]["criteria"]), 16)

    def test_near_far_accuracy_and_uid_cluster(self) -> None:
        items = {
            f"clinc:{i}": fake_item(f"clinc:{i}", leaked=bool(i)) for i in range(2)
        }
        with patch.dict(
            builder.SLICES["rq3_shared_hardness"], {"uids": 2, "k": (2, 4)}
        ):
            rows = list(
                builder.compile_slice("rq3_shared_hardness", list(items), items)
            )
            self.assertEqual(len(rows), 8)
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                manifest, prompts, gold = write_panel(root, "rq3_shared_hardness", rows)
                predictions = root / "predictions.jsonl"
                predictions.write_text(
                    "".join(
                        builder.compact(
                            prediction(p, g, choose_gold=g["pool"] == "far")
                        )
                        + "\n"
                        for p, g in rows
                    ),
                    encoding="utf-8",
                )
                report = scorer.score(
                    manifest,
                    "rq3_shared_hardness",
                    prompts,
                    gold,
                    predictions,
                    model="test",
                    revision="r1",
                    adapter_version="a1",
                    bootstrap_iterations=100,
                )
                self.assertEqual(
                    report["metrics"]["all_main_requests"]["accuracy_all"], 0.5
                )
                paired = report["metrics"]["paired_near_far"]["overall"]
                self.assertEqual(
                    paired["accuracy_delta_far_minus_near"]["estimate"], 1.0
                )
                self.assertEqual(
                    paired["accuracy_delta_far_minus_near"]["interval95"],
                    {"low": 1.0, "high": 1.0},
                )
                self.assertEqual(
                    paired["accuracy_delta_far_minus_near"]["uid_clusters"], 2
                )
                self.assertIn("clinc/true", report["metrics"]["by_domain_and_leaked"])
                # A stale prompt receipt is an integrity error, not a silent miss.
                bad = json.loads(predictions.read_text().splitlines()[0])
                bad["source_input_sha256"] = "0" * 64
                predictions.write_text(builder.compact(bad) + "\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "source_input_sha256"):
                    scorer.score(
                        manifest,
                        "rq3_shared_hardness",
                        prompts,
                        gold,
                        predictions,
                        model="test",
                        revision="r1",
                        adapter_version="a1",
                        bootstrap_iterations=100,
                    )

    def test_order_flip_vs_identical_repeat_and_missing_denominator(self) -> None:
        items = {f"clinc:{i}": fake_item(f"clinc:{i}") for i in range(2)}
        with patch.dict(builder.SLICES["rq2_shared_order"], {"uids": 2}):
            rows = list(builder.compile_slice("rq2_shared_order", list(items), items))
            self.assertEqual(len(rows), 24)
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                manifest, prompts, gold = write_panel(root, "rq2_shared_order", rows)
                predictions = root / "predictions.jsonl"
                predictions.write_text(
                    "".join(
                        builder.compact(
                            prediction(p, g, choose_gold=g["permutation"] != 1)
                        )
                        + "\n"
                        for p, g in rows
                        if not (
                            g["uid"] == "clinc:0"
                            and g["pool"] == "near"
                            and g["permutation"] == 4
                        )
                    ),
                    encoding="utf-8",
                )
                report = scorer.score(
                    manifest,
                    "rq2_shared_order",
                    prompts,
                    gold,
                    predictions,
                    model="test",
                    revision="r1",
                    adapter_version="a1",
                    bootstrap_iterations=100,
                )
                order = report["metrics"]["order_stability"]
                self.assertEqual(order["order_changes"]["pairs"], 16)
                self.assertEqual(order["order_changes"]["valid_pairs"], 15)
                self.assertAlmostEqual(
                    order["order_changes"]["choice_flip_fraction_on_valid_pairs"][
                        "estimate"
                    ],
                    4 / 15,
                )
                self.assertEqual(
                    order["identical_order_control"][
                        "choice_flip_fraction_on_valid_pairs"
                    ]["estimate"],
                    0,
                )
                self.assertEqual(
                    report["metrics"]["all_main_requests"]["invalid_or_missing_n"], 1
                )

    def test_optional_distribution_and_malformed_probability(self) -> None:
        _, gold = builder.compile_one(
            fake_item("clinc:0"), "rq1_extended_capacity", "ext", 2, 0
        )
        choice = gold["gold"]
        self.assertTrue(
            scorer.evaluate(gold, {"answers": {"decision": {"choice": choice}}})[
                "valid"
            ]
        )
        malformed = scorer.evaluate(
            gold,
            {
                "answers": {
                    "decision": {"choice": choice, "probabilities": {choice: 1.0}}
                }
            },
        )
        self.assertEqual(malformed["reason"], "probabilities")
        self.assertFalse(malformed["valid"])
        overflow = scorer.evaluate(
            gold, {"answers": {"decision": {"error": "context_overflow"}}}
        )
        self.assertEqual(overflow["reason"], "context_overflow")

    def test_rq1_k_curve_resamples_complete_uid_vectors(self) -> None:
        items = {f"clinc:{i}": fake_item(f"clinc:{i}") for i in range(2)}
        with patch.dict(
            builder.SLICES["rq1_extended_capacity"], {"uids": 2, "k": (2, 4, 8)}
        ):
            rows = list(
                builder.compile_slice("rq1_extended_capacity", list(items), items)
            )
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                manifest, prompts, gold = write_panel(
                    root, "rq1_extended_capacity", rows
                )
                predictions = root / "predictions.jsonl"
                predictions.write_text(
                    "".join(
                        builder.compact(
                            prediction(
                                p, g, choose_gold=g["k"] == 2 or g["uid"] == "clinc:0"
                            )
                        )
                        + "\n"
                        for p, g in rows
                        if not (g["k"] == 8 and g["uid"] == "clinc:1")
                    ),
                    encoding="utf-8",
                )
                report = scorer.score(
                    manifest,
                    "rq1_extended_capacity",
                    prompts,
                    gold,
                    predictions,
                    model="test",
                    revision="r1",
                    adapter_version="a1",
                    bootstrap_iterations=100,
                )
                curve = report["metrics"]["rq1_uid_cluster_bootstrap"]
                self.assertEqual(curve["uid_clusters"], 2)
                self.assertEqual(curve["by_k"]["2"]["accuracy_all"]["estimate"], 1.0)
                self.assertEqual(curve["by_k"]["4"]["accuracy_all"]["estimate"], 0.5)
                self.assertEqual(
                    curve["by_k"]["8"]["validity_fraction"]["estimate"], 0.5
                )
                self.assertEqual(
                    curve["by_k"]["4"]["delta_accuracy_vs_k2"]["estimate"], -0.5
                )
                self.assertEqual(
                    curve,
                    scorer.score(
                        manifest,
                        "rq1_extended_capacity",
                        prompts,
                        gold,
                        predictions,
                        model="test",
                        revision="r1",
                        adapter_version="a1",
                        bootstrap_iterations=100,
                    )["metrics"]["rq1_uid_cluster_bootstrap"],
                )
                for field, wrong in (
                    ("model", "other"),
                    ("revision", "wrong"),
                    ("adapter_version", "other-adapter"),
                ):
                    kwargs = {
                        "model": "test",
                        "revision": "r1",
                        "adapter_version": "a1",
                    }
                    kwargs[field] = wrong
                    with self.assertRaisesRegex(
                        ValueError,
                        {
                            "model": "model_id",
                            "revision": "model_revision",
                            "adapter_version": "adapter_version",
                        }[field],
                    ):
                        scorer.score(
                            manifest,
                            "rq1_extended_capacity",
                            prompts,
                            gold,
                            predictions,
                            bootstrap_iterations=100,
                            **kwargs,
                        )
                saved = predictions.read_text(encoding="utf-8").splitlines()
                tampered = json.loads(saved[0])
                tampered.pop("model_revision")
                predictions.write_text(
                    builder.compact(tampered) + "\n" + "\n".join(saved[1:]) + "\n",
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(ValueError, "model_revision"):
                    scorer.score(
                        manifest,
                        "rq1_extended_capacity",
                        prompts,
                        gold,
                        predictions,
                        model="test",
                        revision="r1",
                        adapter_version="a1",
                        bootstrap_iterations=100,
                    )


if __name__ == "__main__":
    unittest.main()
