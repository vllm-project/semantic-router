from __future__ import annotations

import contextlib
import io
import json
import subprocess
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

from v2.eval import gates, panels
from v2.eval.sealed import postkey
from v2.eval.sealed import score as c1score
from v2.eval.tests.test_gates import c1_answers, c1_gold

SEALED = Path(postkey.__file__).resolve().parent
SPEC = SEALED / "c1-postkey" / "dev2-0p6b-b2131337.json"
REGISTRY = SEALED / "c1-postkey-baselines.json"
SCRIPT = SEALED / "c1-postkey.sh"
A, B, C = "a" * 64, "b" * 64, "c" * 64


def spec(**changes) -> dict:
    value = json.loads(SPEC.read_text())
    value.update(changes)
    return value


def registry(**tiers) -> dict:
    value = json.loads(REGISTRY.read_text())
    value["tiers"].update(tiers)
    return value


def entry(identity: str = A, seal: str = B) -> dict:
    return {
        "model": "DEV2.0-0.6B",
        "repo": "llm-semantic-router/DEV2.0-0.6B",
        "revision": "b21313375ad77ddf4a8e420fa5195e6e09582043",
        "identity": identity,
        "run": "/runs/0.6B/base/cand",
        "seal_sha256": seal,
        "predictions_sha256": C,
        "c1": 33.0,
    }


def successor(identity: str = C) -> dict:
    value = spec(role="successor", name="dev2-0p6b-next", identity=identity)
    value["package"] = {**value["package"], "identity": identity}
    value["compare"] = []
    return value


class PlanTest(unittest.TestCase):
    def test_committed_spec_plans_the_current_06b_revision(self):
        p = postkey.resolve(spec(), registry())
        self.assertEqual((p["tier"], p["role"]), ("0.6B", "current"))
        self.assertEqual(p["label"], c1score.POSTKEY_LABEL)
        self.assertEqual(
            p["comparisons"],
            [
                {
                    "key": "prev-e61b2b44",
                    "name": "DEV2.0-0.6B e61b2b44 (previous revision, C1 event 2)",
                    "run": "/data/dev2/runs/eval/m4/c1-event2/cand",
                    "seal_sha256": "f9e1279c2bc4b0db9c44f885c3631e059ef4a8ee5b80773a62a4f7de6b582739",
                    "kind": "info",
                }
            ],
        )
        self.assertNotIn("compare", p["model"])
        self.assertIsNone(p["model"]["smoke_items"])

    def test_committed_registry(self):
        value = json.loads(REGISTRY.read_text())
        self.assertEqual(
            value["item_set"]["retired_sha256"], c1score.POSTKEY_RETIRED_SHA256
        )
        self.assertEqual(value["label"], c1score.POSTKEY_LABEL)
        for tier, cell in value["tiers"].items():
            with self.subTest(tier=tier):
                self.assertIn(tier, ("0.6B", "0.8B", "2B", "4B", "9B", "27B"))
                for key in ("identity", "seal_sha256", "predictions_sha256"):
                    self.assertRegex(cell[key], postkey.SHA)
                self.assertTrue(cell["run"].startswith("/data/dev2/runs/eval/"))
                self.assertGreater(cell["c1"], 0)

    def test_successor_is_gated_against_the_registered_baseline(self):
        p = postkey.resolve(successor(), registry(**{"0.6B": entry()}))
        self.assertEqual(
            p["comparisons"][0],
            {
                "key": "baseline",
                "kind": "item8",
                "name": "DEV2.0-0.6B b2131337 (current revision)",
                "run": "/runs/0.6B/base/cand",
                "seal_sha256": B,
            },
        )

    def test_refusals(self):
        with_base = registry(**{"0.6B": entry(identity=spec()["identity"])})
        cases = {
            "placeholder": (spec(revision="PARENT-FILLS"), registry()),
            "tier": (spec(tier="1B"), registry()),
            "role": (spec(role="candidate"), registry()),
            "name": (spec(name="Bad Name"), registry()),
            "image": (spec(image="other"), registry()),
            "adapter": (spec(adapter="kai"), registry()),
            "near parity": (spec(parity={"mode": "near", "stored": "x"}), registry()),
            "unfrozen": (spec(package={"dir": "d", "identity": A}), registry()),
            "identity": (spec(identity=A), registry()),
            "model path": (spec(model_path="/elsewhere"), registry()),
            "smoke items": (spec(smoke_items=0), registry()),
            "no baseline": (successor(), registry()),
            "same weights": (successor(A), registry(**{"0.6B": entry(identity=A)})),
            "baseline exists": (spec(), with_base),
            "item set": (spec(), {**registry(), "item_set": {"retired_sha256": A}}),
            "compare fields": (spec(compare=[{"key": "x"}]), registry()),
            "compare key": (
                spec(
                    compare=[
                        {"key": "baseline", "name": "n", "run": "/r", "seal_sha256": A}
                    ]
                ),
                registry(),
            ),
            "duplicate keys": (
                spec(
                    compare=[{"key": "x", "name": "n", "run": "/r", "seal_sha256": A}]
                    * 2
                ),
                registry(),
            ),
        }
        for name, (value, reg) in cases.items():
            with self.subTest(name):
                with self.assertRaises(ValueError):
                    postkey.resolve(value, reg)

    def test_plan_command_writes_the_plan_or_refuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bad = root / "bad.json"
            bad.write_text(json.dumps(successor()))
            with contextlib.redirect_stderr(io.StringIO()):
                refused = postkey.main(
                    ["plan", "--spec", str(bad), "--registry", str(REGISTRY)]
                    + ["--output", str(root / "refused.json")]
                )
            self.assertEqual(refused, 2)
            self.assertFalse((root / "refused.json").exists())
            with contextlib.redirect_stdout(io.StringIO()):
                ok = postkey.main(
                    ["plan", "--spec", str(SPEC), "--registry", str(REGISTRY)]
                    + ["--output", str(root / "plan.json")]
                )
            self.assertEqual(ok, 0)
            written = json.loads((root / "plan.json").read_text())
            self.assertEqual(written["spec_sha256"], postkey.event3.sha_file(SPEC))
            self.assertEqual(written["schema"], postkey.SCHEMA)


class RunnerArgvTest(unittest.TestCase):
    def plan_file(self, root: Path) -> Path:
        path = root / "plan.json"
        path.write_text(json.dumps(postkey.resolve(spec(), registry())))
        return path

    def argv(self, root: Path, phase: str, cache: str | None) -> list[str]:
        args = ["argv", "--plan", str(self.plan_file(root)), "--phase", phase]
        args += ["--run-dir", "/job/x", "--gpu", "0", "--src", "M", "--src-root", "/s"]
        args += ["--lease-name", "owner.eval", "--shared"]
        if cache:
            args += ["--cache-dir", cache]
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(postkey.main(args), 0)
        return out.getvalue().split("\0")[:-1]

    def test_collect_and_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            collect = self.argv(Path(tmp), "collect", "/job/x-triton")
            smoke = self.argv(Path(tmp), "smoke", "/job/x-triton")
        purpose = collect[collect.index("--purpose") + 1]
        self.assertEqual(purpose, "C1 v1.2 post-key collection: DEV2.0-0.6B b2131337")
        self.assertEqual(collect[collect.index("--panels") + 1], "sealed-c1")
        self.assertIn("HIP_FORCE_DEV_KERNARG=1", collect)
        self.assertIn("TRITON_CACHE_DIR=/job/x-triton", collect)
        self.assertEqual(
            collect[collect.index("--adapter-spec") + 1],
            "/s/v2/06b/records/adapters/dev2-06b-causal-8k-sb.json",
        )
        self.assertEqual(smoke[smoke.index("--panels") + 1], "typed-final,public231")
        self.assertNotIn("--max-items", smoke)
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                self.argv(Path(tmp), "collect", None)


class LedgerTest(unittest.TestCase):
    def test_non_selection(self):
        base = registry(**{"0.6B": entry()})
        first = postkey.resolve(successor(), base)
        sibling = postkey.resolve(successor("d" * 64), base)
        line = {
            "tier": "0.6B",
            "role": "successor",
            "identity": C,
            "job_dir": "/jobs/first",
            "baseline_seal_sha256": B,
        }
        self.assertEqual(postkey.ledger_problems(first, []), [])
        again = postkey.ledger_problems(first, [line])
        self.assertEqual(len(again), 1)
        self.assertIn("already scored", again[0])
        siblings = postkey.ledger_problems(sibling, [line])
        self.assertEqual(len(siblings), 1)
        self.assertIn("never a selection criterion among siblings", siblings[0])
        self.assertEqual(postkey.ledger_problems(sibling, [{**line, "tier": "2B"}]), [])
        moved = {**line, "baseline_seal_sha256": A}
        self.assertEqual(postkey.ledger_problems(sibling, [moved]), [])

    def test_check_needs_an_approval(self):
        base = registry(**{"0.6B": entry()})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = root / "plan.json"
            plan.write_text(json.dumps(postkey.resolve(successor("d" * 64), base)))
            ledger = root / "LEDGER.jsonl"
            ledger.write_text(
                json.dumps(
                    {
                        "tier": "0.6B",
                        "role": "successor",
                        "identity": C,
                        "job_dir": "/jobs/first",
                        "baseline_seal_sha256": B,
                    }
                )
                + "\n"
            )
            argv = ["ledger-check", "--plan", str(plan), "--ledger", str(ledger)]
            with contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(postkey.main(argv), 1)
                with contextlib.redirect_stdout(io.StringIO()) as out:
                    self.assertEqual(
                        postkey.main(argv + ["--approval", "coordinator 2026-09-30"]), 0
                    )
            self.assertIn("coordinator 2026-09-30", out.getvalue())
            self.assertEqual(
                postkey.main(
                    [
                        "ledger-check",
                        "--plan",
                        str(plan),
                        "--ledger",
                        str(root / "none"),
                    ]
                ),
                0,
            )


def write_rows(path: Path, rows) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def sealed_run(run: Path, gold, prompts: Path, wrong: set[int]) -> Path:
    predictions = write_rows(
        run / "output" / "sealed-c1.predictions.jsonl",
        c1_answers(gold, wrong).values(),
    )
    with contextlib.redirect_stdout(io.StringIO()):
        c1score.seal(
            Namespace(
                prompts=prompts, predictions=predictions, output=run / "SEAL-C1.json"
            )
        )
    return predictions


def c1_case(root: Path) -> dict:
    gold = c1_gold()
    paths = {
        "gold": write_rows(root / "gold.jsonl", gold),
        "prompts": write_rows(
            root / "prompts.jsonl",
            [
                {"id": g["id"], "state": g["state"], "questions": g["questions"]}
                for g in gold
            ],
        ),
        "retired": root / "retired.json",
        "gold_rows": gold,
    }
    paths["retired"].write_text(
        json.dumps(
            {
                "schema": "dev2-c1-retired/1",
                "version": "v1.2",
                "candidates": ["a/choice|0"],
                "protected_rows": [],
            }
        )
    )
    return paths


@contextlib.contextmanager
def pinned(paths: dict):
    with (
        mock.patch.dict(
            panels.SEALED["sealed-c1"],
            {
                "prompts_sha256": gates.sha_file(paths["prompts"]),
                "gold_sha256": gates.sha_file(paths["gold"]),
            },
        ),
        mock.patch.object(
            c1score, "POSTKEY_RETIRED_SHA256", gates.sha_file(paths["retired"])
        ),
        mock.patch.object(c1score, "POSTKEY_RETIRED", paths["retired"]),
        mock.patch.object(gates, "PAIRED_REPLICATES", 200),
    ):
        yield


class FinishAndReproduceTest(unittest.TestCase):
    def test_reproduce_matches_the_event_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = c1_case(root)
            event = root / "event"
            stored = root / "e2" / "cand"
            sealed_run(
                event / "cand",
                paths["gold_rows"],
                paths["prompts"],
                set(range(0, 122, 5)),
            )
            sealed_run(
                stored, paths["gold_rows"], paths["prompts"], set(range(1, 122, 7))
            )
            (event / "PLAN.json").write_text(
                json.dumps(
                    {
                        "models": {
                            "cand": {"label": "Candidate"},
                            "prev": {
                                "label": "Previous",
                                "stored": {"seal": str(stored / "SEAL-C1.json")},
                            },
                        },
                        "pairs": [
                            {"left": "cand", "right": "prev", "kind": "candidate"}
                        ],
                    }
                )
            )
            with pinned(paths):
                with contextlib.redirect_stdout(io.StringIO()):
                    c1score.compare(
                        Namespace(
                            gold=paths["gold"],
                            left=event
                            / "cand"
                            / "output"
                            / "sealed-c1.predictions.jsonl",
                            right=stored / "output" / "sealed-c1.predictions.jsonl",
                            left_name="Candidate",
                            right_name="Previous",
                            replicates=200,
                            retired=paths["retired"],
                            retired_sha=gates.sha_file(paths["retired"]),
                            output=event / "PAIRED-C1-cand-vs-prev.json",
                        )
                    )
                argv = [
                    "reproduce",
                    "--event-dir",
                    str(event),
                    "--gold",
                    str(paths["gold"]),
                ]
                argv += ["--workers", "1"]
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(
                        postkey.main(argv + ["--output-dir", str(root / "r1")]), 0
                    )
                result = json.loads((root / "r1" / "REPRODUCE.json").read_text())
                self.assertTrue(result["all_match"])
                self.assertEqual(result["pairs"][0]["pair"], "cand vs prev")
                self.assertIn(result["pairs"][0]["verdict"], ("PASS", "REGRESSION"))
                paired = event / "PAIRED-C1-cand-vs-prev.json"
                value = json.loads(paired.read_text())
                value["ci95"] = [value["ci95"][0], value["ci95"][1] + 1e-9]
                paired.write_text(json.dumps(value))
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(
                        postkey.main(argv + ["--output-dir", str(root / "r2")]), 1
                    )

    def test_finish_writes_the_summary_and_the_ledger(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = c1_case(root)
            job = root / "job"
            base = root / "base"
            sealed_run(base, paths["gold_rows"], paths["prompts"], set())
            sealed_run(
                job / "cand",
                paths["gold_rows"],
                paths["prompts"],
                set(range(0, 122, 2)),
            )
            reg = registry(
                **{
                    "0.6B": entry(
                        identity=A, seal=gates.sha_file(base / "SEAL-C1.json")
                    )
                }
            )
            reg["tiers"]["0.6B"]["run"] = str(base)
            plan = job / "PLAN.json"
            plan.write_text(json.dumps(postkey.resolve(successor(), reg)))
            with pinned(paths):
                with contextlib.redirect_stdout(io.StringIO()):
                    c1score.score(
                        Namespace(
                            gold=paths["gold"],
                            predictions=job
                            / "cand"
                            / "output"
                            / "sealed-c1.predictions.jsonl",
                            seal=job / "cand" / "SEAL-C1.json",
                            label="DEV2.0-0.6B",
                            retired=paths["retired"],
                            retired_sha=gates.sha_file(paths["retired"]),
                            post_key=True,
                            output=job / "cand" / "REPORT-C1.json",
                        )
                    )
                    ledger = root / "pk" / "LEDGER.jsonl"
                    finish = ["finish", "--plan", str(plan), "--job-dir", str(job)]
                    finish += ["--ledger", str(ledger)]
                    self.assertEqual(
                        postkey.main(finish + ["--output", str(job / "S0.json")]), 1
                    )
                    fields = io.StringIO()
                    with contextlib.redirect_stdout(fields):
                        postkey.main(
                            ["gates", "--plan", str(plan), "--job-dir", str(job)]
                        )
                    right, right_name, output = fields.getvalue().split("\0")[:3]
                    gates.main(
                        ["c1", "--left", str(job / "cand"), "--right", right]
                        + ["--left-name", "succ", "--right-name", right_name]
                        + ["--gold", str(paths["gold"]), "--output", output]
                    )
                    self.assertEqual(
                        postkey.main(finish + ["--output", str(job / "S1.json")]), 0
                    )
            summary = json.loads((job / "S1.json").read_text())
            self.assertEqual(summary["label"], c1score.POSTKEY_LABEL)
            self.assertEqual(summary["item8"]["verdict"], "REGRESSION")
            self.assertEqual(summary["comparisons"][0]["kind"], "item8")
            self.assertEqual(summary["baseline_entry"]["run"], str(job / "cand"))
            self.assertEqual(summary["baseline_entry"]["identity"], C)
            self.assertNotIn("tasks", summary)
            lines = [json.loads(x) for x in ledger.read_text().splitlines()]
            self.assertEqual([x["item8_verdict"] for x in lines], [None, "REGRESSION"])
            self.assertEqual(
                lines[1]["baseline_seal_sha256"], reg["tiers"]["0.6B"]["seal_sha256"]
            )
            self.assertEqual(lines[1]["identity"], C)


class ScriptTest(unittest.TestCase):
    def run_script(self, *argv: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["bash", str(SCRIPT), *argv],
            capture_output=True,
            text=True,
            stdin=subprocess.DEVNULL,
            timeout=60,
        )

    def test_syntax_and_argument_checks(self):
        subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
        self.assertEqual(self.run_script().returncode, 2)
        self.assertEqual(self.run_script("collect", "--gpu", "0").returncode, 2)
        self.assertEqual(self.run_script("collect", "--src", "M").returncode, 2)
        self.assertEqual(
            self.run_script(
                "collect", "--src", "M", "--spec", "s", "--gpu", "9"
            ).returncode,
            2,
        )
        incomplete = self.run_script(
            "gate", "--src", "M", "--left", "/a", "--right", "/b", "--left-name", "x"
        )
        self.assertEqual(incomplete.returncode, 2)
        self.assertEqual(
            self.run_script("gate", "--src", "M", "--verify-only").returncode, 2
        )
        relative = self.run_script(
            *("gate", "--src", "M", "--left", "rel", "--right", "/b"),
            *("--left-name", "x", "--right-name", "y", "--output", "/o"),
        )
        self.assertEqual(relative.returncode, 2)
        self.assertIn("paths must be absolute", relative.stderr)
        self.assertIn(
            "paths must be absolute",
            self.run_script("reproduce", "--src", "M", "--output-dir", "out").stderr,
        )
        missing = self.run_script(
            "collect", "--src", "no-such-mirror", "--spec", "s", "--verify-only"
        )
        self.assertEqual(missing.returncode, 1)
        self.assertIn("no verified mirror", missing.stderr)


class ScorePostKeyTest(unittest.TestCase):
    def test_label_and_pinned_item_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = c1_case(root)
            run = root / "run"
            sealed_run(run, paths["gold_rows"], paths["prompts"], {3})
            predictions = run / "output" / "sealed-c1.predictions.jsonl"
            with contextlib.redirect_stdout(io.StringIO()):
                c1score.main(
                    ["seal", "--prompts", str(paths["prompts"]), "--predictions"]
                    + [
                        str(predictions),
                        "--post-key",
                        "--output",
                        str(root / "pk.json"),
                    ]
                )
            self.assertEqual(
                json.loads((root / "pk.json").read_text())["label"],
                c1score.POSTKEY_LABEL,
            )
            self.assertEqual(
                json.loads((run / "SEAL-C1.json").read_text())["label"], c1score.LABEL
            )
            sha = gates.sha_file(paths["retired"])
            common = [
                "score",
                "--gold",
                str(paths["gold"]),
                "--predictions",
                str(predictions),
            ]
            common += ["--seal", str(root / "pk.json"), "--label", "m", "--post-key"]
            retired = ["--retired", str(paths["retired"]), "--retired-sha", sha]
            with self.assertRaises(ValueError):
                c1score.main(common + ["--output", str(root / "r0.json")])
            with self.assertRaises(ValueError):
                c1score.main(common + retired + ["--output", str(root / "r1.json")])
            with mock.patch.object(c1score, "POSTKEY_RETIRED_SHA256", sha):
                with contextlib.redirect_stdout(io.StringIO()):
                    c1score.main(common + retired + ["--output", str(root / "r2.json")])
            report = json.loads((root / "r2.json").read_text())
            self.assertEqual(report["label"], c1score.POSTKEY_LABEL)
            self.assertEqual(report["item_set"]["scored_items"], 121)
            self.assertFalse((root / "r0.json").exists())


if __name__ == "__main__":
    unittest.main()
