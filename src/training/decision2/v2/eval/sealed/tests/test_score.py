from __future__ import annotations

import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from v2.eval.sealed import score

CHOICE = {"type": "choice", "instructions": "?", "criteria": {"x": "X", "y": "Y"}}
NOUL = {"type": "noul", "instructions": "?", "criteria": {"true": "t", "false": "f"}}
LEVELS = {"type": "score", "instructions": "?", "criteria": ["lo", "mid", "hi"]}
RETIRED = [
    *(f"a/choice|{i}" for i in range(4)),
    *(f"d/choice|{i}" for i in range(4)),
    "z/gone|0",
]
WRONG = {"x": "y", "y": "x"}


def item(
    i: int, task: str, question: dict, value, source: str = "src", lang: str = "en"
) -> dict:
    record = {"type": question["type"], "value": value, "semantic_value": value}
    if question["type"] == "choice":
        record["label_to_semantic"] = {k: k for k in question["criteria"]}
    return {
        "id": f"c1-{task}-{i}",
        "task": task,
        "source": source,
        "source_item_id": str(i),
        "group_id": f"g{i // 2}",
        "language": lang,
        "input_chars": 100,
        "long": i % 5 == 0,
        "state": f"s{i}",
        "questions": {"decision": question},
        "gold": {"decision": record},
    }


def answer(question: dict, value) -> dict:
    if question["type"] == "choice":
        return {"type": "choice", "choice": value}
    if question["type"] == "noul":
        return {"type": "noul", "noul": 0.9 if value else 0.1}
    return {"type": "score", "score": value}


def prediction(row: dict, value) -> dict:
    question = row["questions"]["decision"]
    return {
        "id": row["id"],
        "answers": {"decision": answer(question, value)},
        "source_input_sha256": score.input_digest(row["state"], row["questions"]),
        "model_id": "m",
    }


class ScoreTest(unittest.TestCase):
    def setUp(self):
        self.gold = (
            [item(i, "a/choice", CHOICE, "x" if i % 2 else "y") for i in range(20)]
            + [
                item(i, "b/noul", NOUL, bool(i % 2), "hallutruthqa", "ar")
                for i in range(20)
            ]
            + [item(i, "c/score", LEVELS, i % 3) for i in range(21)]
        )

    def write(self, root: Path, name: str, rows) -> Path:
        path = root / name
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        return path

    def test_seal_score_and_compare(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompts = self.write(
                root,
                "prompts.jsonl",
                [
                    {"id": g["id"], "state": g["state"], "questions": g["questions"]}
                    for g in self.gold
                ],
            )
            gold = self.write(root, "gold.jsonl", self.gold)
            perfect = self.write(
                root,
                "perfect.jsonl",
                [prediction(g, g["gold"]["decision"]["value"]) for g in self.gold],
            )
            half = [
                (
                    prediction(g, g["gold"]["decision"]["value"])
                    if i % 2 == 0
                    else {**prediction(g, None), "answers": {"decision": None}}
                )
                for i, g in enumerate(self.gold)
            ]
            weak = self.write(root, "weak.jsonl", half)
            score.seal(
                Namespace(
                    prompts=prompts, predictions=perfect, output=root / "seal.json"
                )
            )
            score.score(
                Namespace(
                    gold=gold,
                    predictions=perfect,
                    seal=root / "seal.json",
                    label="p",
                    output=root / "report.json",
                )
            )
            report = json.loads((root / "report.json").read_text())
            self.assertAlmostEqual(report["c1"], 100.0)
            self.assertAlmostEqual(report["tasks"]["c/score"]["qwk"], 1.0)
            self.assertEqual(report["slices"]["non_english"]["items"], 20)
            self.assertAlmostEqual(
                report["licence_split"]["nc_tasks_mean_macro_f1"], 100.0
            )
            score.compare(
                Namespace(
                    gold=gold,
                    left=perfect,
                    right=weak,
                    left_name="p",
                    right_name="w",
                    replicates=200,
                    output=root / "paired.json",
                )
            )
            paired = json.loads((root / "paired.json").read_text())
            self.assertGreater(paired["delta"], 0)
            self.assertGreater(paired["ci95"][0], 0)
            self.assertEqual(sorted(paired["by_type"]), ["choice", "noul", "score"])
            self.assertEqual(sorted(report["languages"]), ["ar", "en"])
            (root / "perfect.jsonl").write_text(perfect.read_text() + "\n")
            with self.assertRaises(ValueError):
                score.score(
                    Namespace(
                        gold=gold,
                        predictions=perfect,
                        seal=root / "seal.json",
                        label="p",
                        output=root / "report2.json",
                    )
                )

    def test_seal_rejects_changed_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prompts = self.write(
                root,
                "prompts.jsonl",
                [
                    {"id": g["id"], "state": g["state"], "questions": g["questions"]}
                    for g in self.gold
                ],
            )
            rows = [prediction(g, g["gold"]["decision"]["value"]) for g in self.gold]
            rows[0]["source_input_sha256"] = "0" * 64
            bad = self.write(root, "bad.jsonl", rows)
            with self.assertRaises(ValueError):
                score.seal(
                    Namespace(
                        prompts=prompts, predictions=bad, output=root / "seal.json"
                    )
                )

    def test_macro_f1_and_invalid_answers(self):
        self.assertAlmostEqual(score.macro_f1([("a", "a"), ("b", "b")]), 1.0)
        self.assertAlmostEqual(score.macro_f1([("a", None), ("b", "b")]), (0 + 1.0) / 2)
        self.assertIsNone(score.quadratic_kappa([(1, None), (2, None)]))

    def cli(self, *argv) -> dict:
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(score.main([str(a) for a in argv]), 0)
        return json.loads(out.getvalue())

    def retired_case(self, root: Path) -> tuple[dict[str, Path], str]:
        """Gold plus a task that retiring empties; `left` is wrong on retired items only."""
        gold = self.gold + [
            item(i, "d/choice", CHOICE, "x" if i % 2 else "y") for i in range(4)
        ]

        def retired(g: dict) -> bool:
            return f"{g['task']}|{g['source_item_id']}" in RETIRED

        def truth(g: dict):
            return g["gold"]["decision"]["value"]

        paths = {
            "gold": self.write(root, "gold.jsonl", gold),
            "kept": self.write(root, "kept.jsonl", [g for g in gold if not retired(g)]),
            "prompts": self.write(
                root,
                "prompts.jsonl",
                [
                    {"id": g["id"], "state": g["state"], "questions": g["questions"]}
                    for g in gold
                ],
            ),
            "left": self.write(
                root,
                "left.jsonl",
                [
                    prediction(g, WRONG[truth(g)] if retired(g) else truth(g))
                    for g in gold
                ],
            ),
            "right": self.write(
                root,
                "right.jsonl",
                [
                    (
                        prediction(g, truth(g))
                        if i % 2 == 0
                        else {**prediction(g, None), "answers": {"decision": None}}
                    )
                    for i, g in enumerate(gold)
                ],
            ),
            "retired": root / "retired.json",
        }
        data = json.dumps(
            {
                "schema": "dev2-c1-retired/1",
                "version": "v1.2",
                "candidates": RETIRED,
                "protected_rows": ["a/raw.parquet|0"],
            }
        ).encode()
        paths["retired"].write_bytes(data)
        self.cli(
            "seal",
            "--prompts",
            paths["prompts"],
            "--predictions",
            paths["left"],
            "--output",
            root / "seal.json",
        )
        return paths, hashlib.sha256(data).hexdigest()

    def test_retired_items_leave_c1_and_task_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths, sha = self.retired_case(root)
            common = ["score", "--gold", paths["gold"], "--predictions", paths["left"]]
            common += ["--seal", root / "seal.json", "--label", "p"]
            retired = ["--retired", paths["retired"], "--retired-sha", sha]
            summary = self.cli(*common, *retired, "--output", root / "v12.json")
            full = self.cli(*common, "--output", root / "all.json")
            report = json.loads((root / "v12.json").read_text())
            self.assertAlmostEqual(report["c1"], 100.0)
            self.assertLess(full["c1"], 100.0)
            self.assertEqual(sorted(report["tasks"]), ["a/choice", "b/noul", "c/score"])
            self.assertEqual(report["tasks"]["a/choice"]["items"], 16)
            self.assertEqual(report["items"], 57)
            expected = {
                "version": "v1.2",
                "retired_sha256": sha,
                "retired_candidates": 9,
                "gold_items": 65,
                "dropped_items": 8,
                "scored_items": 57,
                "dropped_by_task": {"a/choice": 4, "d/choice": 4},
                "tasks_emptied": ["d/choice"],
            }
            self.assertEqual(report["item_set"], expected)
            self.assertEqual(summary["item_set"], expected)
            self.assertNotIn("item_set", full)
            self.assertNotIn("item_set", json.loads((root / "all.json").read_text()))

    def test_retired_compare_equals_removed_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths, sha = self.retired_case(root)
            common = ["compare", "--left", paths["left"], "--right", paths["right"]]
            common += ["--left-name", "l", "--right-name", "r", "--replicates", "200"]
            summary = self.cli(
                *common,
                "--gold",
                paths["gold"],
                "--retired",
                paths["retired"],
                "--retired-sha",
                sha,
                "--output",
                root / "paired-v12.json",
            )
            self.cli(
                *common, "--gold", paths["kept"], "--output", root / "paired-kept.json"
            )
            self.cli(
                *common, "--gold", paths["gold"], "--output", root / "paired-all.json"
            )
            v12, kept, full = (
                json.loads((root / f"paired-{name}.json").read_text())
                for name in ("v12", "kept", "all")
            )
            self.assertEqual(v12.pop("item_set"), summary["item_set"])
            self.assertEqual(summary["item_set"]["scored_items"], 57)
            self.assertEqual(v12, kept)
            self.assertEqual(summary["delta"], kept["delta"])
            self.assertEqual(summary["ci95"], kept["ci95"])
            self.assertNotEqual(full["delta"], kept["delta"])

    def test_retired_refusals(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths, sha = self.retired_case(root)
            other = root / "other.json"
            other.write_text(
                json.dumps(
                    {
                        "schema": "dev2-c1-retired/0",
                        "version": "v1.2",
                        "candidates": RETIRED,
                        "protected_rows": [],
                    }
                )
            )
            commands = {
                "score": ["--predictions", paths["left"], "--seal", root / "seal.json"]
                + ["--label", "p"],
                "compare": ["--left", paths["left"], "--right", paths["right"]]
                + ["--left-name", "l", "--right-name", "r", "--replicates", "10"],
            }
            cases = {
                "hash mismatch": [
                    "--retired",
                    paths["retired"],
                    "--retired-sha",
                    "0" * 64,
                ],
                "wrong schema": [
                    "--retired",
                    other,
                    "--retired-sha",
                    score.sha_file(other),
                ],
                "list without sha": ["--retired", paths["retired"]],
                "sha without list": ["--retired-sha", sha],
            }
            output = root / "out.json"
            for command, flags in commands.items():
                for name, retired in cases.items():
                    argv = [command, "--gold", paths["gold"], *flags, *retired]
                    with self.subTest(f"{command}: {name}"):
                        with self.assertRaises(ValueError):
                            score.main([str(a) for a in [*argv, "--output", output]])
                        self.assertFalse(output.exists())

    def test_outputs_without_retired_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths, _ = self.retired_case(root)
            summary = self.cli(
                "score",
                "--gold",
                paths["gold"],
                "--predictions",
                paths["left"],
                "--seal",
                root / "seal.json",
                "--label",
                "p",
                "--output",
                root / "cli.json",
            )
            with contextlib.redirect_stdout(io.StringIO()):
                score.score(
                    Namespace(
                        gold=paths["gold"],
                        predictions=paths["left"],
                        seal=root / "seal.json",
                        label="p",
                        output=root / "namespace.json",
                    )
                )
            report = (root / "cli.json").read_bytes()
            self.assertEqual(report, (root / "namespace.json").read_bytes())
            self.assertEqual(list(summary), ["c1", "by_type", "report_sha256"])
            self.assertEqual(
                summary["report_sha256"], hashlib.sha256(report).hexdigest()
            )
            self.assertNotIn("item_set", json.loads(report))
            paired = self.cli(
                "compare",
                "--gold",
                paths["gold"],
                "--left",
                paths["left"],
                "--right",
                paths["right"],
                "--left-name",
                "l",
                "--right-name",
                "r",
                "--replicates",
                "10",
                "--output",
                root / "paired.json",
            )
            self.assertEqual(list(paired), ["delta", "ci95"])
            self.assertEqual(
                sorted(json.loads((root / "paired.json").read_text())),
                sorted(
                    ["delta", "ci95", "by_type", "replicates", "seed", "unit"]
                    + ["schema", "label", "left", "right"]
                ),
            )


if __name__ == "__main__":
    unittest.main()
