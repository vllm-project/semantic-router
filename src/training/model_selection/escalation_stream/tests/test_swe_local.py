"""Local SWE judge inputs for the escalation stream (#5374).

Run from the repository root:
    python -m unittest discover -s src/training/model_selection/escalation_stream/tests -p 'test_*.py'
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO_ROOT))

from src.training.model_selection.escalation_stream.swe_local import (  # noqa: E402
    COMMAND_CHARS,
    NOT_RECORDED,
    OMITTED,
    STATE_CHARS,
    WITHHELD_DIFF,
    append_checkpoint,
    auroc,
    build_requests,
    build_state,
    load_checkpoint,
    load_trajectories,
    load_view_scores,
    pending_requests,
    policy_rows,
    ranking,
    score_rows,
)


def record(actions, diff=None, final="done"):
    return {
        "id": "inst::cheap",
        "task": "inst",
        "instruction": "Fix the bug",
        "actions": [
            {"command": command, "output": output} for command, output in actions
        ],
        "final_message": final,
        "diff": diff,
        "attempts": {
            "cheap": {"ok": 0.0, "cost": 0.2},
            "strong": {"ok": 1.0, "cost": 1.0},
        },
    }


def state(body, view):
    return json.loads(build_state(body, view))


class JudgeViewTest(unittest.TestCase):
    def setUp(self):
        self.actions = [(f"C{i}", f"out-{i}") for i in range(6)]

    def test_prefixes_keep_the_first_actions_only(self):
        body = record(self.actions, diff="DIFF-TOKEN", final="FINAL-TOKEN")
        task = state(body, "task")
        first = state(body, "actions-1")
        fourth = state(body, "actions-4")
        self.assertEqual(task["actions_so_far"], [])
        self.assertFalse(task["finished"])
        self.assertEqual([item["command"] for item in first["actions_so_far"]], ["C0"])
        self.assertEqual(
            [item["command"] for item in fourth["actions_so_far"]],
            ["C0", "C1", "C2", "C3"],
        )
        self.assertNotIn("FINAL-TOKEN", json.dumps(first))
        self.assertNotIn("DIFF-TOKEN", json.dumps(fourth))

    def test_finished_view_withholds_the_diff(self):
        token = "DIFF-TOKEN-XYZ"
        separate = record(self.actions, diff=token)
        leaked = record([*self.actions[:-1], ("C5", token)], diff=token)
        hidden = state(separate, "finished")
        shown = state(separate, "finished-diff")
        withheld = state(leaked, "finished")
        kept = state(leaked, "finished-diff")
        self.assertEqual(
            [item["command"] for item in hidden["last_actions"]],
            ["C2", "C3", "C4", "C5"],
        )
        self.assertNotIn(token, json.dumps(hidden))
        self.assertEqual(shown["diff"], token)
        self.assertEqual(withheld["last_actions"][-1]["output"], WITHHELD_DIFF)
        self.assertNotIn(token, json.dumps(withheld))
        self.assertEqual(kept["diff"], token)
        self.assertEqual(kept["last_actions"][-1]["output"], token)
        self.assertNotIn(WITHHELD_DIFF, json.dumps(kept))

    def test_short_traces_do_not_invent_actions(self):
        body = record([("only", "out")], diff=None)
        self.assertEqual(len(state(body, "actions-4")["actions_so_far"]), 1)
        self.assertEqual(len(state(body, "finished")["last_actions"]), 1)

    def test_unresolved_references_are_placeholders(self):
        body = record([("$123", "$9")], diff=None)
        action = state(body, "actions-1")["actions_so_far"][0]
        self.assertEqual(action["command"], NOT_RECORDED)
        self.assertEqual(action["output"], NOT_RECORDED)

    def test_long_command_lists_keep_the_ends(self):
        def command(label):
            return (label + ("x" * COMMAND_CHARS))[:COMMAND_CHARS]

        actions = [command("FIRST")]
        actions += [command("pad") for _ in range(39)]
        actions += [command("MIDTOKEN")]
        actions += [command("pad") for _ in range(38)]
        actions += [command("LAST")]
        packed = state(record([(cmd, "ok") for cmd in actions]), "finished")
        commands = packed["commands"]
        self.assertTrue(any(cmd.startswith("FIRST") for cmd in commands))
        self.assertTrue(any(cmd.startswith("LAST") for cmd in commands))
        self.assertIn(OMITTED, commands)
        self.assertFalse(any("MIDTOKEN" in cmd for cmd in commands))
        self.assertLess(len(commands), 80)
        self.assertLessEqual(len(json.dumps(packed, ensure_ascii=False)), STATE_CHARS)


class CheckpointTest(unittest.TestCase):
    def test_resume_skips_completed_keys_and_rejects_conflicts(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "answers.jsonl"
            first = {"id": "a", "view": "finished", "probs": {"resolve": [0.2, 0.8]}}
            self.assertEqual(append_checkpoint(path, [first]), 1)
            self.assertEqual(append_checkpoint(path, [first]), 0)
            self.assertEqual(len(path.read_text().splitlines()), 1)
            requests = [
                {"id": "a", "view": "finished"},
                {"id": "a", "view": "task"},
            ]
            self.assertEqual(
                pending_requests(requests, load_checkpoint(path)),
                [{"id": "a", "view": "task"}],
            )
            conflict = {
                "id": "a",
                "view": "finished",
                "probs": {"resolve": [0.9, 0.1]},
            }
            with self.assertRaises(ValueError) as raised:
                append_checkpoint(path, [conflict])
            self.assertIn(str(path), str(raised.exception))
            self.assertEqual(len(path.read_text().splitlines()), 1)
            path.write_text(json.dumps(first) + "\n" + json.dumps(conflict) + "\n")
            with self.assertRaises(ValueError) as loaded:
                load_checkpoint(path)
            self.assertIn(f"{path}:2:", str(loaded.exception))

    def test_reordered_unique_scores_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            left = Path(tmp) / "left.jsonl"
            right = Path(tmp) / "right.jsonl"
            rows = [
                {"id": "a", "view": "finished", "score": 0.1},
                {"id": "b", "view": "finished", "score": 0.2},
            ]
            left.write_text("".join(json.dumps(row) + "\n" for row in rows))
            right.write_text("".join(json.dumps(row) + "\n" for row in reversed(rows)))
            self.assertEqual(
                load_view_scores(left, "finished"), load_view_scores(right, "finished")
            )
            duplicate = Path(tmp) / "same.jsonl"
            duplicate.write_text(
                json.dumps(rows[0]) + "\n" + json.dumps(rows[0]) + "\n"
            )
            self.assertEqual(load_view_scores(duplicate, "finished"), {"a": 0.1})


class ScoreAndAurocTest(unittest.TestCase):
    def test_scores_read_list_and_dict_probabilities(self):
        listed = {
            "id": "a",
            "view": "finished",
            "probs": {"resolve": [0.25, 0.75], "outcome": [0.1, 0.7, 0.2]},
        }
        named = {
            "id": "a",
            "view": "finished",
            "probs": {
                "resolve": {"false": 0.25, "true": 0.75},
                "outcome": {
                    "agent_resolves": 0.1,
                    "stronger_needed": 0.7,
                    "unsolvable": 0.2,
                },
            },
        }
        for answer in (listed, named):
            stronger, failure = score_rows([answer])
            self.assertAlmostEqual(stronger[0]["score"], 0.7)
            self.assertAlmostEqual(failure[0]["score"], 0.25)

    def test_failure_auroc_ignores_scores_that_only_separate_tasks(self):
        rows, scores = [], {}
        for task, score, outcomes in (
            ("E", 0.2, [1.0, 1.0, 0.0]),
            ("H", 0.8, [0.0, 0.0, 1.0]),
        ):
            for index, ok in enumerate(outcomes):
                row_id = f"{task}{index}"
                rows.append(
                    {
                        "id": row_id,
                        "task": task,
                        "attempts": {
                            "cheap": {"ok": ok, "cost": 0.1},
                            "strong": {"ok": 1.0, "cost": 1.0},
                        },
                    }
                )
                scores[row_id] = score
        report = ranking(rows, scores, "cheap", "strong")
        self.assertAlmostEqual(report["failure_auroc"], 6 / 9)
        self.assertEqual(report["failure_within_task"], 0.5)
        self.assertEqual(report["failure_within_task_pairs"], 4)

    def test_rescue_auroc_ranks_a_separated_pair(self):
        rows = [
            {
                "id": "rescued",
                "task": "rescued",
                "attempts": {
                    "cheap": {"ok": 0.0, "cost": 0.1},
                    "strong": {"ok": 1.0, "cost": 1.0},
                },
            },
            {
                "id": "kept",
                "task": "kept",
                "attempts": {
                    "cheap": {"ok": 1.0, "cost": 0.1},
                    "strong": {"ok": 1.0, "cost": 1.0},
                },
            },
        ]
        report = ranking(rows, {"rescued": 0.9, "kept": 0.1}, "cheap", "strong")
        self.assertEqual(report["rescue_auroc"], 1.0)
        self.assertAlmostEqual(auroc([0.5], [0.5, 0.0]), 0.75)

    def test_loader_reads_a_directory_and_has_no_download(self):
        source = Path(load_trajectories.__code__.co_filename).read_text()
        for token in ("huggingface", "urlopen", "http://", "https://", "requests.get"):
            self.assertNotIn(token, source)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "local.jsonl"
            body = record([("ls", "ok")], diff="patch")
            path.write_text(json.dumps(body) + "\n")
            loaded = load_trajectories(tmp)
            self.assertEqual(policy_rows(loaded)[0]["id"], body["id"])
            self.assertNotIn("instruction", policy_rows(loaded)[0])
            requests = build_requests(loaded)
            self.assertEqual(len(requests), 7)
            with self.assertRaises(ValueError):
                load_trajectories(path)


if __name__ == "__main__":
    unittest.main()
