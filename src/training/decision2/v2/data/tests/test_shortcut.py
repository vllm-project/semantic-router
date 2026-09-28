from __future__ import annotations

import hashlib
import json
import os
import random
import stat
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

from training.model.data import INPUT_FIELDS, canonical, digest
from v2.data import shortcut

SYLLABLES = [c + v for c in "bcdfghjklmnprstvz" for v in "aeiou"]
RNG = random.Random(11)
WORDS = sorted({"".join(RNG.choice(SYLLABLES) for _ in range(3)) for _ in range(3000)})
NOUL = [("false", "The statement does not follow."), ("true", "The statement follows.")]


def words(rng: random.Random, count: int) -> str:
    return " ".join(rng.choice(WORDS) for _ in range(count))


def make_row(
    row_id: str,
    group: str,
    task_type: str,
    options: list[tuple[str, object]],
    label: int,
    *,
    state: object = "context",
    instructions: object = "Pick the best option.",
    family: str = "family",
    source: str = "source",
    split: str = "train",
) -> dict:
    row = {
        "id": row_id,
        "state": state,
        "instructions": instructions,
        "options": [{"key": key, "description": text} for key, text in options],
        "label": label,
        "task_type": task_type,
        "family": family,
        "group_id": group,
        "language": "en",
        "split": split,
        "source": source,
        "evaluation_role": split,
        "render_template": "test",
        "audit_metadata": {},
    }
    row["input_sha256"] = digest({field: row[field] for field in INPUT_FIELDS})
    return row


def option_leak_rows() -> list[dict]:
    rng = random.Random(1)
    rows = []
    for index in range(160):
        gold = rng.randrange(4)
        options = [
            (key, words(rng, 4) + (" verified" if position == gold else ""))
            for position, key in enumerate("ABCD")
        ]
        rows.append(
            make_row(
                f"leak{index}",
                f"leak-group{index}",
                "choice",
                options,
                gold,
                state=words(rng, 30),
                family="leaky" if index % 2 else "leaky_too",
            )
        )
    return rows


def counterfactual_rows() -> list[dict]:
    rng = random.Random(2)
    rows = []
    for group in range(40):
        instructions = "Which option matches " + words(rng, 6) + "?"
        options = [(key, words(rng, 5)) for key in "ABCD"]
        for gold in range(4):
            rows.append(
                make_row(
                    f"cf-choice{group}-{gold}",
                    f"cf-choice{group}",
                    "choice",
                    options,
                    gold,
                    state=words(rng, 40),
                    instructions=instructions,
                    family="counterfactual_choice",
                )
            )
    for group in range(40):
        instructions = "Does the context entail: " + words(rng, 8) + "?"
        for label in (0, 1):
            rows.append(
                make_row(
                    f"cf-noul{group}-{label}",
                    f"cf-noul{group}",
                    "noul",
                    NOUL,
                    label,
                    state=words(rng, 40),
                    instructions=instructions,
                    family="counterfactual_noul",
                )
            )
    return rows


def hypothesis_leak_rows() -> list[dict]:
    rng = random.Random(3)
    rows = []
    for group in range(60):
        for label in (0, 1):
            claim = words(rng, 8) + (" never" if label == 0 else " indeed")
            rows.append(
                make_row(
                    f"hyp{group}-{label}",
                    f"hyp{group}",
                    "noul",
                    NOUL,
                    label,
                    state={"evidence": words(rng, 30), "claim": claim},
                    instructions="Is the claim supported by the evidence?",
                    family="claims",
                )
            )
    return rows


def gate(receipt: dict, view: str, task_type: str) -> dict:
    return next(
        item
        for item in receipt["gates"]
        if item["view"] == view and item["task_type"] == task_type
    )


class ShortcutGateTest(unittest.TestCase):
    def test_option_text_leak_fails_option_only(self) -> None:
        receipt = shortcut.run_baselines(option_leak_rows())
        result = gate(receipt, "option_only", "choice")
        self.assertFalse(result["pass"])
        self.assertGreater(result["accuracy"], 0.9)
        self.assertLess(result["majority"], 0.45)
        self.assertFalse(gate(receipt, "state_removed", "choice")["pass"])
        self.assertEqual(receipt["verdict"], "FAIL")
        families = {item["family"] for item in result["families_over_margin"]}
        self.assertEqual(families, {"leaky", "leaky_too"})
        self.assertEqual(receipt["views"]["hypothesis_only"], {"n_rows": 0})

    def test_counterfactual_balanced_groups_pass_state_removed_at_chance(self) -> None:
        receipt = shortcut.run_baselines(counterfactual_rows())
        choice = gate(receipt, "state_removed", "choice")
        noul = gate(receipt, "state_removed", "noul")
        self.assertEqual(
            (choice["n"], choice["accuracy"], choice["majority"]), (160, 0.25, 0.25)
        )
        self.assertEqual(
            (noul["n"], noul["accuracy"], noul["majority"]), (80, 0.5, 0.5)
        )
        self.assertTrue(choice["pass"] and noul["pass"])
        self.assertEqual(receipt["verdict"], "PASS")
        self.assertEqual(
            {(item["view"], item["task_type"]) for item in receipt["gates"]},
            {
                (view, task)
                for view in ("state_removed", "option_only")
                for task in ("choice", "noul")
            },
        )
        self.assertEqual(receipt["majority"]["choice"], {"n": 160, "accuracy": 0.25})

    def test_hypothesis_leak_is_caught_only_by_hypothesis_view(self) -> None:
        receipt = shortcut.run_baselines(hypothesis_leak_rows())
        self.assertTrue(gate(receipt, "state_removed", "noul")["pass"])
        hypothesis = gate(receipt, "hypothesis_only", "noul")
        self.assertFalse(hypothesis["pass"])
        self.assertGreater(hypothesis["accuracy"], 0.9)
        self.assertEqual(receipt["views"]["hypothesis_only"]["n_rows"], 120)
        self.assertEqual(receipt["verdict"], "FAIL")

    def test_small_task_types_are_not_gated(self) -> None:
        rows = counterfactual_rows()[:28]
        receipt = shortcut.run_baselines(rows)
        self.assertEqual(receipt["gates"], [])
        self.assertEqual(receipt["verdict"], "INCONCLUSIVE")

    def test_workers_do_not_change_receipt(self) -> None:
        rows = option_leak_rows()[:80] + hypothesis_leak_rows()[:60]
        single = shortcut.run_baselines(rows, workers=1)
        double = shortcut.run_baselines(rows, workers=3)
        self.assertEqual(canonical(single), canonical(double))


def item(
    fold: int,
    task_type: str,
    keys: tuple[str, ...],
    label: int,
    options: tuple[str, ...] | None = None,
    state: str = "",
) -> shortcut.Item:
    return shortcut.Item(
        id=f"i{fold}{label}{keys}",
        group_id="g",
        fold=fold,
        task_type=task_type,
        family="f",
        source="s",
        label=label,
        keys=keys,
        options=options or tuple("x" for _ in keys),
        instructions="",
        state=state,
        hypothesis=None,
    )


class HeuristicTest(unittest.TestCase):
    def test_fold_is_sha256_of_group(self) -> None:
        digest_value = int(hashlib.sha256(b"group-7").hexdigest(), 16)
        self.assertEqual(shortcut.fold_of("group-7"), digest_value % 5)

    def test_position_prior_uses_keys_for_score_and_breaks_ties_low(self) -> None:
        levels = ("0", "1", "2")
        reversed_levels = ("2", "1", "0")
        items = [item(fold % 5, "score", levels, 2) for fold in range(20)]
        items += [item(fold % 5, "score", levels, 0) for fold in range(5)]
        items.append(item(0, "score", reversed_levels, 0))
        predictions = shortcut.position_prior(items)
        self.assertEqual(predictions[0], 2)
        self.assertEqual(predictions[-1], 0)
        tied = [item(0, "noul", ("false", "true"), 1)]
        tied += [
            item(1, "noul", ("false", "true"), 0),
            item(1, "noul", ("false", "true"), 1),
        ]
        self.assertEqual(shortcut.position_prior(tied)[0], 0)
        swapped = [
            item(0, "noul", ("true", "false"), 0),
            item(1, "noul", ("false", "true"), 1),
        ]
        self.assertEqual(shortcut.position_prior(swapped)[0], 0)

    def test_longest_option_and_lexical_overlap(self) -> None:
        choice = item(
            0,
            "choice",
            ("A", "B", "C"),
            1,
            ("short", "a much longer option", "tie tie tie tie tie"),
            "tie apple",
        )
        self.assertEqual(shortcut.longest_option(choice), 1)
        self.assertEqual(shortcut.lexical_overlap(choice), 2)
        flat = item(0, "choice", ("A", "B"), 0, ("same", "same"), "none")
        self.assertEqual(
            (shortcut.longest_option(flat), shortcut.lexical_overlap(flat)), (0, 0)
        )

    def test_hypothesis_text(self) -> None:
        self.assertIsNone(shortcut.hypothesis_text("plain"))
        self.assertIsNone(shortcut.hypothesis_text({"evidence": "x"}))
        self.assertEqual(
            shortcut.hypothesis_text({"claim": "c", "statement": ["s"]}), "c s"
        )
        self.assertEqual(shortcut.hypothesis_text(canonical({"hypothesis": "h"})), "h")

    def test_wilson(self) -> None:
        self.assertEqual(shortcut.wilson(0, 10), [0.0, 0.277533])
        self.assertEqual(shortcut.wilson(5, 10), [0.236593, 0.763407])
        self.assertIsNone(shortcut.wilson(0, 0))


class ShortcutCliTest(unittest.TestCase):
    def test_load_rows_validates_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            rows = counterfactual_rows()[:4]
            rows[1]["input_sha256"] = "0" * 64
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "input_sha256"):
                shortcut.load_rows(path)
            rows = counterfactual_rows()[:2] + counterfactual_rows()[:1]
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "duplicate id"):
                shortcut.load_rows(path)

    def test_cli_writes_private_receipt_and_exit_code(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows_path, receipt_path = root / "arm.jsonl", root / "receipt.json"
            rows_path.write_text(
                "".join(json.dumps(row) + "\n" for row in option_leak_rows()),
                encoding="utf-8",
            )
            argv = [
                "--rows",
                str(rows_path),
                "--receipt",
                str(receipt_path),
                "--workers",
                "2",
            ]
            with redirect_stdout(StringIO()):
                self.assertEqual(shortcut.main(argv), 1)
            self.assertEqual(stat.S_IMODE(os.stat(receipt_path).st_mode), 0o600)
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            self.assertEqual(receipt["verdict"], "FAIL")
            self.assertEqual(receipt["input"]["rows"], 160)
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                with self.assertRaises(SystemExit):
                    shortcut.main(argv)


if __name__ == "__main__":
    unittest.main()
