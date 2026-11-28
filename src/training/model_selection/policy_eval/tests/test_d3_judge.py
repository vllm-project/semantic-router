from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

# Direct unittest discovery also runs this file without installing the package.
# ruff: noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO_ROOT))

from src.training.model_selection.policy_eval.d3_judge import (
    MIN_COMMANDS,
    STATE_CHARS,
    end_state,
    main,
    scores,
    task_state,
    tb2_requests,
)
from src.training.model_selection.policy_eval.policy_eval import read_scores
from src.training.model_selection.policy_eval.prepare_terminal_bench import (
    STRONG,
    build_rows,
)
from src.training.model_selection.policy_eval.twinrouterbench_eval import (
    main as twinrouterbench_main,
)
from src.training.model_selection.policy_eval.twinrouterbench_eval import tier

DATA = Path(__file__).resolve().parents[1] / "data"


def step(src, cmd="", obs="", msg=""):
    return {"src": src, "tools": json.dumps([{"cmd": cmd}]), "obs": obs, "msg": msg}


def record(model, task, trial, reward, steps):
    return {
        "task_name": task,
        "agent": "terminus-2",
        "model": model,
        "reward": reward,
        "cost_cents": 10,
        "trial_name": trial,
        "steps": json.dumps(steps),
    }


class EndStateTest(unittest.TestCase):
    def test_unresolved_references_and_the_last_actions(self):
        steps = [step("user", msg="task")] + [
            step("agent", cmd=f"cmd{i}", obs=("$12" if i == 1 else f"out{i}"))
            for i in range(6)
        ]
        steps.append(step("agent", msg="done"))
        body = json.loads(end_state("fix it", steps))
        self.assertEqual(body["final_message"], "done")
        self.assertEqual(
            [a["command"] for a in body["last_actions"]],
            ["cmd2", "cmd3", "cmd4", "cmd5"],
        )
        self.assertEqual(len(body["commands"]), 6)
        self.assertIsNone(end_state("fix it", [step("user", msg="task")]))
        early = json.loads(end_state("fix it", [*steps[:3], step("agent", msg="x")]))
        self.assertEqual(early["last_actions"][-1]["output"], "[not recorded]")

    def test_long_states_keep_the_first_and_last_commands(self):
        steps = [step("agent", cmd="x" * 290) for _ in range(200)] + [
            step("agent", msg="done")
        ]
        body = json.loads(end_state("task", steps))
        self.assertLessEqual(len(json.dumps(body, ensure_ascii=False)), STATE_CHARS)
        self.assertIn("... commands omitted ...", body["commands"])
        self.assertGreater(len(body["commands"]), MIN_COMMANDS)

    def test_task_state_has_no_actions(self):
        body = json.loads(task_state("y" * 7000))
        self.assertEqual(len(body["task"]), 6000)
        self.assertEqual(body["actions_so_far"], [])
        self.assertFalse(body["finished"])


class Tb2RequestsTest(unittest.TestCase):
    def test_requests_join_the_rows_by_id(self):
        done = [step("agent", cmd="ls"), step("agent", msg="done")]
        records = [
            record("deepseek-v3.2@deepseek", "t1", "a", 1, done),
            record("deepseek-v3.2@deepseek", "t1", "b", 0, [step("user", msg="x")]),
            record("deepseek-v3.2@deepseek", "t2", "c", 1, done),
        ] + [record(m, t, "s", 1, done) for m in STRONG for t in ("t1",)]
        end, tasks = tb2_requests(records, {"t1": "do t1", "t2": "do t2"})
        rows = {r["id"] for r in build_rows(records)}
        self.assertEqual([r["id"] for r in end], ["t1::deepseek-v3.2::a"])
        self.assertTrue({r["id"] for r in end} <= rows)
        self.assertEqual([r["id"] for r in tasks], ["t1"])
        self.assertEqual(end[0]["k"], 99)


class ScoresTest(unittest.TestCase):
    def test_one_probability_or_its_complement(self):
        answers = [
            {"id": "a", "probs": {"outcome": [0.2, 0.7, 0.1], "resolve": [0.9, 0.1]}}
        ]
        self.assertEqual(
            scores(answers, "outcome", 1, False), [{"id": "a", "score": 0.7}]
        )
        self.assertAlmostEqual(scores(answers, "resolve", -1, True)[0]["score"], 0.9)

    def test_scores_cli_writes_policy_eval_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, out = Path(tmp) / "a.jsonl", Path(tmp) / "s.jsonl"
            src.write_text(
                json.dumps({"id": "a", "probs": {"outcome": [0.5, 0.25, 0.25]}}) + "\n"
            )
            main(
                [
                    "scores",
                    "--answers",
                    str(src),
                    "--question",
                    "outcome",
                    "--index",
                    "1",
                    "--out",
                    str(out),
                ]
            )
            self.assertEqual(json.loads(out.read_text()), {"id": "a", "score": 0.25})


class TierTest(unittest.TestCase):
    def test_argmax_and_cut(self):
        p = [0.5, 0.1, 0.15, 0.25]
        self.assertEqual(tier(p, None), 0)
        self.assertEqual(tier(p, 0.2), 3)
        self.assertEqual(tier(p, 0.4), 2)
        self.assertEqual(tier(p, 0.9), 0)

    def test_repeated_answer_id_is_rejected_before_scoring(self):
        rows = [{"id": k, "k": 0, "probs": {"tier": [1, 0, 0, 0]}} for k in "aba"]
        with tempfile.TemporaryDirectory() as tmp:
            answers = Path(tmp) / "answers.jsonl"
            answers.write_text("".join(json.dumps(r) + "\n" for r in rows))
            with self.assertRaisesRegex(ValueError, f"^{answers}:3: duplicate id 'a'$"):
                twinrouterbench_main(["--checkout", tmp, "--answers", str(answers)])


class ShippedAnswersTest(unittest.TestCase):
    def test_answers_cover_every_public_request(self):
        counts = {
            "tb2_end": (809, {"resolve", "outcome", "difficulty"}),
            "tb2_task": (89, {"resolve", "outcome", "difficulty"}),
            "twinrouterbench": (970, {"tier"}),
        }
        for name, (n, questions) in counts.items():
            # read_scores also rejects a repeated id.
            probs = read_scores(
                DATA / f"{name}.d3-turbo.answers.jsonl", lambda r: r["probs"]
            )
            self.assertEqual(len(probs), n, name)
            self.assertTrue(all(set(p) == questions for p in probs.values()), name)
            for p in probs.values():
                for q in p.values():
                    self.assertAlmostEqual(sum(q), 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
