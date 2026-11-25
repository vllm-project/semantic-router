"""Three independent searches over synthetic graded episodes, through the public CLI."""

import contextlib
import io
import shutil
import unittest

from recipe_lab.improver_eval import METHODS
from recipe_lab.loop_evidence import digest
from recipe_lab.schedule import canonical
from tests import test_loop as fixtures
from tests import test_schedule as scheduling
from tests import test_signature as prefixes


def pin(path):
    return {"file": path.name, "sha256": digest(path.read_bytes())}


class ImproverEvalTest(unittest.TestCase):
    def setUp(self):
        trial = self.trial = scheduling.ScheduleTest()
        trial.setUp()
        self.addCleanup(trial.doCleanups)
        run = trial.run
        root = trial.root
        self.cli = run.cli
        source = root / "target"
        shutil.copytree(root / "fixed", source)
        folder = source / "dsh/runs/task/fixed/main"
        (folder / "requests.jsonl").write_bytes(
            prefixes.encode_rows(prefixes.receipts([4, 8]))
        )
        fixtures.put(
            folder / "result.json",
            dict(fixtures.get(folder / "result.json"), model_requests=2),
        )
        target = self.cli(
            "record",
            "--run",
            source,
            "--arm",
            "fixed",
            "--config",
            source / "config.yaml",
            "--public-model",
            "public",
            "--strong",
            "A",
            "--hypothesis",
            "assess two target windows",
        )["record_id"]
        space = root / "three-windows.json"
        fixtures.put(space, dict(fixtures.get(trial.space), allowed_upper=[3, 4, 8]))
        seed = next(
            s
            for s in range(100)
            if digest(canonical([s, 8]).encode()) < digest(canonical([s, 4]).encode())
        )
        splits = {}
        for split, ids in (("dev", ["a" * 64, "f" * 64]), ("holdout", ["9" * 64])):
            path = root / (split + ".json")
            fixtures.put(
                path,
                {
                    "schema": "recipe-lab-search-split-v1",
                    "methods": METHODS,
                    "study": "three",
                    "split": split,
                    "instruction_sha256": ids,
                },
            )
            splits[split] = pin(path)
        protocol = root / "search.json"
        fixtures.put(
            protocol,
            {
                "schema": "recipe-lab-search-preflight-v1",
                "study": "three",
                "based_on": trial.base,
                "prefixes": trial.base,
                "space": space.name,
                "space_sha256": pin(space)["sha256"],
                "seed": seed,
                "budget": {"episodes": 4, "proposal_model_tokens": 16},
                "split_plans": splits,
            },
        )
        common = [
            "--protocol",
            protocol,
            "--candidate-trials-per-method",
            "1",
            "--target-prefixes",
            target,
        ]
        allocation = root / "allocation.json"
        fixtures.put(
            allocation,
            {
                "schema": "recipe-lab-search-allocation-v1",
                "study": "three",
                "protocol_sha256": pin(protocol)["sha256"],
                "methods": {m: [m] for m in METHODS},
            },
        )
        self.enrollment = self.cli(
            "search-enroll", *common, "--allocation", allocation
        )["enrollment"]
        arms = {}
        for method in METHODS:
            trial.card = root / (method + "-trial.json")
            fixtures.put(trial.card, dict(trial.protocol, study=method))
            step = trial.poll()
            while step["status"] == "run":
                step = trial.poll(trial.fresh_run(step["next"]))
            self.assertEqual(step["status"], "advance")
            usage = root / (method + "-tokens.json")
            calls = []
            if method != "random":
                receipt = root / (method + "-call.json")
                fixtures.put(
                    receipt,
                    {
                        "schema": "recipe-lab-proposal-call-v1",
                        "method": method,
                        "trial_study": method,
                        "enrollment": self.enrollment,
                        "response": {
                            "id": "call-" + method,
                            "usage": {
                                "prompt_tokens": 10,
                                "completion_tokens": 2,
                                "total_tokens": 12,
                            },
                        },
                    },
                )
                calls.append(pin(receipt))
            fixtures.put(
                usage,
                {
                    "schema": "recipe-lab-search-usage-v1",
                    "enrollment": self.enrollment,
                    "method": method,
                    "calls": calls,
                },
            )
            policy = (
                trial.report["base_config_sha256"]
                if method == "random"
                else pin(trial.candidate)["sha256"]
            )
            arms[method] = {
                "policy_sha256": policy,
                "trial_cards": [pin(trial.card)],
                "usage": pin(usage),
            }
        self.selections = root / "choices.json"
        fixtures.put(
            self.selections,
            {
                "schema": "recipe-lab-search-selections-v1",
                "protocol_sha256": pin(protocol)["sha256"],
                "methods": arms,
                "direct_strong_sha256": "d" * 64,
                "holdout_repeats": 1,
            },
        )
        self.args = ["search-freeze", *common, "--selections", self.selections]

    def invalid(self, *args):
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(io.StringIO()):
            self.cli(*args)

    def matrix(self, freeze):
        root = self.trial.root
        policies = {m: freeze["methods"][m]["policy_sha256"] for m in METHODS}
        policies.update(
            baseline=self.trial.report["base_config_sha256"], direct_strong="d" * 64
        )
        rows = [
            {
                "study": "three",
                "split": "holdout",
                "assessment": "valid",
                "repeat": 1,
                "task": "H1",
                "arm": arm,
                "instruction_sha256": "9" * 64,
                "policy_sha256": policy,
                "controls_sha256": "c" * 64,
                "seconds": 20 + i,
                "strict": arm != "random",
                "episode_sha256": digest(arm.encode() + b"-holdout"),
                "freeze_sha256": freeze["freeze_sha256"],
            }
            for i, (arm, policy) in enumerate(policies.items())
        ]
        scores, manifest = root / "scores.json", root / "matrix.json"
        aborted = dict(
            rows[0], episode_sha256="7" * 64, seconds=3, reason="judge unavailable"
        )
        assigned = {r["arm"]: r["assignment_id"] for r in freeze["holdout_assignments"]}
        for index, row in enumerate([*rows, aborted]):
            binding = root / f"start-{index}.json"
            outcome = root / f"end-{index}.json"
            fixtures.put(
                binding,
                dict(
                    assignment_id=assigned[row["arm"]],
                    **{
                        k: row[k]
                        for k in (
                            "instruction_sha256",
                            "policy_sha256",
                            "controls_sha256",
                        )
                    },
                ),
            )
            result = {
                "assignment_id": assigned[row["arm"]],
                "episode_sha256": row["episode_sha256"],
                "seconds": row["seconds"],
            }
            result.update(
                {"assessment": "invalid", "reason": row["reason"]}
                if row is aborted
                else {"assessment": "valid", "strict": row["strict"]}
            )
            fixtures.put(outcome, result)
            row.update(binding=pin(binding), outcome=pin(outcome))
        fixtures.put(scores, rows)
        fixtures.put(
            manifest,
            {
                "study": "three",
                "split": "holdout",
                "repeats": [1],
                "arms": list(policies),
                "tasks": ["H1"],
                "scores": scores.name,
                "scores_sha256": pin(scores)["sha256"],
                "freeze_sha256": freeze["freeze_sha256"],
                "invalid_attempts": [aborted],
            },
        )
        return manifest

    def test_seal_then_grade_and_replay_once(self):
        freeze = self.cli(*self.args)
        self.assertEqual(freeze, self.cli(*self.args))
        manifest = self.matrix(freeze)
        args = ["improver-eval", "--study", "three", "--manifest", manifest]
        result = self.cli(*args)
        self.assertEqual(result["methods"]["guided"]["dev_episodes"], 4)
        self.assertEqual(
            result["guided_versus_other_methods"]["random"]["gained_strict_cells"],
            [["H1", 1]],
        )
        self.assertTrue(
            result["guided_versus_other_methods"]["existing_skill"]["same_policy"]
        )
        self.assertEqual(result["invalid_holdout"]["existing_skill"]["wall_seconds"], 3)
        self.assertEqual(self.cli(*args), result)
        fixtures.put(manifest, dict(fixtures.get(manifest), invalid_attempts=[]))
        self.invalid(*args)
        choices = fixtures.get(self.selections)
        fixtures.put(self.selections, dict(choices, holdout_repeats=2))
        self.invalid(*self.args)

    def test_reject_unbound_holdout_and_overbudget_or_reused_search(self):
        choices = fixtures.get(self.selections)
        usage = self.trial.root / "guided-tokens.json"
        call = self.trial.root / "guided-call.json"
        receipt = fixtures.get(call)
        receipt["response"]["usage"].update(prompt_tokens=15, total_tokens=17)
        fixtures.put(call, receipt)
        fixtures.put(usage, dict(fixtures.get(usage), calls=[pin(call)]))
        choices["methods"]["guided"]["usage"] = pin(usage)
        fixtures.put(self.selections, choices)
        self.invalid(*self.args)
        receipt["response"]["usage"].update(prompt_tokens=10, total_tokens=12)
        fixtures.put(call, receipt)
        fixtures.put(usage, dict(fixtures.get(usage), calls=[pin(call)]))
        choices["methods"]["guided"]["usage"] = pin(usage)
        previous = choices["methods"]["guided"]["trial_cards"]
        choices["methods"]["guided"]["trial_cards"] = choices["methods"]["random"][
            "trial_cards"
        ]
        fixtures.put(self.selections, choices)
        self.invalid(*self.args)
        choices["methods"]["guided"]["trial_cards"] = previous
        fixtures.put(self.selections, choices)
        manifest = self.matrix(self.cli(*self.args))
        scores = self.trial.root / "scores.json"
        rows = fixtures.get(scores)
        rows[0]["freeze_sha256"] = "0" * 64
        fixtures.put(scores, rows)
        fixtures.put(
            manifest, dict(fixtures.get(manifest), scores_sha256=pin(scores)["sha256"])
        )
        self.invalid("improver-eval", "--study", "three", "--manifest", manifest)

    def test_refuse_holdout_used_by_another_study(self):
        previous = (
            self.trial.run.workspace / "search-studies" / "previous" / "freeze.json"
        )
        fixtures.put(previous, {"holdout_instructions": ["9" * 64]})
        self.invalid(*self.args)
        self.assertFalse((previous.parent.parent / "three" / "freeze.json").exists())
