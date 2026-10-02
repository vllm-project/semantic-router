from __future__ import annotations

import hashlib
import json
import re
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from v2.eval.sealed import recheck
from v2.eval.sealed.score import RETIRED_SCHEMA

SEALED = Path(recheck.__file__).resolve().parent
SPEC = SEALED / "c1-recheck" / "r1-ib1r3-ib2-pn1r2.json"
SCRIPT = SEALED / "c1-recheck.sh"
INSTRUCTIONS = "Read the passage and choose the label that fits it best."
CRITERIA = {"a": "The first label applies.", "b": "The second label applies."}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text(n: int, words: int = 20) -> str:
    return " ".join(f"tok{n}x{j}" for j in range(words))


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    return path


def candidate(task: str, n: int, state: dict, group: str | None = None) -> dict:
    return {
        "task": task,
        "source": task.split("/")[0],
        "source_item_id": f"s{n}",
        "group_id": group or f"g{n}",
        "language": "en",
        "state": state,
        "question": {
            "type": "choice",
            "instructions": INSTRUCTIONS,
            "criteria": CRITERIA,
        },
        "gold": "a",
        "overlap_texts": [],
        "balance_label": "a",
        "date": None,
    }


def train_row(
    n: int, state: dict, family: str = "fam", instructions: str = "Decide."
) -> dict:
    return {
        "id": f"r{n}",
        "group_id": f"tg{n}",
        "family": family,
        "state": state,
        "instructions": instructions,
        "options": [
            {"key": "yes", "description": "Yes."},
            {"key": "no", "description": "No."},
        ],
    }


SHARED = (
    "the shared builder instruction text appears in this item as its whole state here"
)


class Fixture:
    """Pools, prompts, a retired list and a training file, all synthetic."""

    def __init__(self, root: Path) -> None:
        self.root = root
        pools = root / "pools"
        pools.mkdir()
        alpha = [candidate("alpha/t", n, {"text": text(n)}) for n in range(6)]
        alpha[2]["state"] = {"text": SHARED}
        alpha.append(candidate("alpha/t", 99, {"text": text(99)}))
        beta = [candidate("beta/t", 10 + n, {"text": text(10 + n)}) for n in range(3)]
        write_jsonl(pools / "alpha.jsonl", alpha)
        write_jsonl(pools / "beta.jsonl", beta)
        self.manifest = root / "build-5-manifest.json"
        self.manifest.write_text(
            json.dumps(
                {
                    "candidates_sha256": {
                        "alpha": sha(pools / "alpha.jsonl"),
                        "beta": sha(pools / "beta.jsonl"),
                    }
                }
            )
        )
        self.pools = pools
        chosen = alpha[:6] + beta
        self.prompts = write_jsonl(
            root / "prompts.jsonl",
            [
                {
                    "id": f"c1-{k:016x}",
                    "state": c["state"],
                    "questions": {"decision": c["question"]},
                }
                for k, c in enumerate(chosen)
            ],
        )
        self.retired = root / "RETIRED.json"
        self.retired.write_text(
            json.dumps(
                {
                    "schema": RETIRED_SCHEMA,
                    "version": "v1.2",
                    "candidates": ["alpha/t|s4"],
                }
            )
        )
        self.texts = [c["state"]["text"] for c in chosen]
        self.train = write_jsonl(
            root / "train.jsonl",
            [
                train_row(0, {"passage": text(0)}),
                train_row(1, {"passage": recheck.perturb(text(1))}, family="other"),
                train_row(
                    2, {"passage": "unrelated words only"}, instructions=INSTRUCTIONS
                ),
                train_row(3, {"passage": "three"}, instructions=SHARED),
                train_row(4, {"passage": "four"}, instructions=SHARED),
                train_row(5, {"passage": "five"}, instructions=SHARED),
                train_row(6, {"passage": text(4)}),
                train_row(
                    7, {"passage": "nothing to see in this row at all"}, family="other"
                ),
            ],
        )
        self.clean = write_jsonl(
            root / "clean.jsonl",
            [train_row(20 + n, {"passage": f"clean row {n}"}) for n in range(4)],
        )
        self.spec = root / "spec.json"
        self.spec.write_text(
            json.dumps(
                {
                    "schema": recheck.SPEC_SCHEMA,
                    "name": "test-recheck",
                    "item_set": recheck.ITEM_SET,
                    "datasets": [
                        {
                            "key": "train-a",
                            "label": "A",
                            "source": "synthetic",
                            "path": str(self.train),
                            "sha256": sha(self.train),
                            "rows": 8,
                        },
                        {
                            "key": "clean-b",
                            "label": "B",
                            "source": "synthetic",
                            "path": str(self.clean),
                            "sha256": sha(self.clean),
                            "rows": 4,
                        },
                    ],
                    "arms": [
                        {
                            "track": "T",
                            "arm": "all",
                            "uses": [{"dataset": "train-a"}, {"dataset": "clean-b"}],
                        },
                        {
                            "track": "T",
                            "arm": "no-fam",
                            "uses": [
                                {"dataset": "train-a", "exclude_families": ["fam"]}
                            ],
                        },
                        {
                            "track": "T",
                            "arm": "clean",
                            "uses": [{"dataset": "clean-b"}],
                        },
                    ],
                }
            )
        )

    def pins(self):
        return mock.patch.multiple(
            recheck,
            PROMPTS_SHA256=sha(self.prompts),
            BUILD_MANIFEST_SHA256=sha(self.manifest),
            POSTKEY_RETIRED_SHA256=sha(self.retired),
            TEMPLATE_ITEMS=3,
            TEMPLATE_ROWS=3,
            CONTROLS=2,
        )

    def items(self, expect: int = 8) -> int:
        return recheck.main(
            [
                "items",
                "--prompts",
                str(self.prompts),
                "--candidates-dir",
                str(self.pools),
                "--build-manifest",
                str(self.manifest),
                "--retired",
                str(self.retired),
                "--expect-scored",
                str(expect),
                "--output",
                str(self.root / "items.jsonl"),
                "--protected-output",
                str(self.root / "items.protected.jsonl"),
                "--receipt",
                str(self.root / "ITEMS.json"),
            ]
        )

    def scan(self) -> int:
        return recheck.main(
            [
                "scan",
                "--spec",
                str(self.spec),
                "--items",
                str(self.root / "items.jsonl"),
                "--workers",
                "2",
                "--chunk-rows",
                "3",
                "--output",
                str(self.root / "SCAN.json"),
                "--private",
                str(self.root / "SCAN-PRIVATE.json"),
            ]
        )


class ItemsTest(unittest.TestCase):
    def test_maps_prompts_to_candidates_without_the_gold(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = Fixture(Path(tmp))
            with fx.pins():
                self.assertEqual(fx.items(), 0)
            rows = recheck.read_jsonl(fx.root / "items.jsonl")
            self.assertEqual(len(rows), 9)
            self.assertNotIn("gold", rows[0])
            self.assertEqual(
                [r["source_item_id"] for r in rows if not r["scored"]], ["s4"]
            )
            shaped = recheck.read_jsonl(fx.root / "items.protected.jsonl")
            self.assertEqual(
                set(shaped[0]),
                {"task", "source_item_id", "source", "state", "overlap_texts"},
            )
            receipt = json.loads((fx.root / "ITEMS.json").read_text())
            self.assertEqual(
                (receipt["items"], receipt["scored"], receipt["retired"]), (9, 8, 1)
            )
            self.assertEqual(
                receipt["by_task"]["alpha/t"],
                {"items": 6, "scored": 5, "retired": 1, "groups": 6},
            )
            self.assertEqual(receipt["items_sha256"], sha(fx.root / "items.jsonl"))

    def test_refusals(self):
        with tempfile.TemporaryDirectory() as tmp:
            fx = Fixture(Path(tmp))
            with fx.pins(), self.assertRaisesRegex(ValueError, "expected 7"):
                fx.items(expect=7)
            with self.assertRaisesRegex(ValueError, "SHA-256 differs"):
                fx.items()
            extra = json.loads(fx.prompts.read_text().splitlines()[0])
            extra["state"] = {"text": "no such candidate"}
            fx.prompts.write_text(fx.prompts.read_text() + json.dumps(extra) + "\n")
            with fx.pins(), self.assertRaisesRegex(
                ValueError, "match no build-5 candidate"
            ):
                fx.items()


class ScanJudgeTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.fx = Fixture(Path(self.tmp.name))
        self.pins = self.fx.pins()
        self.pins.start()
        self.assertEqual(self.fx.items(), 0)

    def tearDown(self):
        self.pins.stop()
        self.tmp.cleanup()

    def test_rules_boilerplate_and_controls(self):
        self.assertEqual(self.fx.scan(), 0)
        public = json.loads((self.fx.root / "SCAN.json").read_text())
        self.assertTrue(public["controls"]["pass"])
        self.assertEqual(public["controls"]["planted"], 2)
        a = public["datasets"]["train-a"]
        self.assertEqual(a["exposed"]["rows"], 2)
        self.assertEqual(a["exposed"]["scored_items"], 2)
        self.assertEqual(
            a["exposed"]["rows_by_rule"], {"E": 1, "G": 2, "N1": 1, "N2": 2}
        )
        self.assertEqual(a["exposed"]["rows_by_family"], {"fam": 1, "other": 1})
        self.assertEqual(a["boilerplate_only"]["rows_by_kind"], {"b1": 3, "b2": 1})
        self.assertEqual(a["retired_items"], {"items": 1, "rows_only_on_retired": 1})
        self.assertEqual(a["b1_template_strings"], 3)
        self.assertEqual(public["datasets"]["clean-b"]["rows_matched"], 0)
        blob = json.dumps(public)
        for item_text in self.fx.texts:
            self.assertNotIn(item_text, blob)
        self.assertNotRegex(blob, r"c1-[0-9a-f]{16}")
        private = json.loads((self.fx.root / "SCAN-PRIVATE.json").read_text())
        rows = {r["id"]: r for r in private["datasets"]["train-a"]}
        self.assertEqual(rows["r0"]["row"], 0)
        self.assertEqual(rows["r1"]["hits"]["1"]["data"], ["G", "N2"])

    def judge(self, hits: list[dict]) -> dict:
        self.assertEqual(self.fx.scan(), 0)
        root = self.fx.root
        write_jsonl(root / "hits.jsonl", hits)
        (root / "receipt.json").write_text(
            json.dumps(
                {
                    "protected": {"sha256": sha(root / "items.protected.jsonl")},
                    "corpora": {
                        "train-a": {"verdicts": {"OVERLAP": 0, "REVIEW": len(hits)}},
                        "clean-b": {"verdicts": {"OVERLAP": 0, "REVIEW": 0}},
                    },
                }
            )
        )
        code = recheck.main(
            [
                "judge",
                "--spec",
                str(self.fx.spec),
                "--items",
                str(root / "items.jsonl"),
                "--protected-items",
                str(root / "items.protected.jsonl"),
                "--scan",
                str(root / "SCAN.json"),
                "--scan-private",
                str(root / "SCAN-PRIVATE.json"),
                "--registered-hits",
                str(root / "hits.jsonl"),
                "--registered-receipt",
                str(root / "receipt.json"),
                "--output",
                str(root / "VERDICT.json"),
                "--quarantine-dir",
                str(root / "quarantine"),
            ]
        )
        self.assertEqual(code, 0)
        return json.loads((root / "VERDICT.json").read_text())

    def test_judge_unites_g0_and_the_registered_scanner(self):
        flagged = {
            "id": "alpha/t|s3",
            "source": "alpha",
            "task": "alpha/t",
            "verdict": "REVIEW",
            "labels": {
                "train-a": {
                    "verdict": "REVIEW",
                    "containment": 0.25,
                    "matched": 2,
                    "file": str(self.fx.train),
                    "row": 7,
                    "exact": None,
                }
            },
        }
        retired = {**flagged, "id": "alpha/t|s4"}
        verdict = self.judge([flagged, retired])
        a, b = verdict["datasets"]["train-a"], verdict["datasets"]["clean-b"]
        self.assertEqual(
            (a["verdict"], b["verdict"], verdict["verdict"]),
            ("QUARANTINE", "PASS", "QUARANTINE"),
        )
        self.assertEqual(a["exposed_scored_items"], 3)
        self.assertEqual(
            a["exposed_by"], {"g0_only": 2, "registered_only": 1, "both": 0}
        )
        self.assertEqual(
            a["registered_scanner"]["scored_item_reasons"],
            {"containment_0.2_to_0.5": 1},
        )
        self.assertEqual(a["quarantine_groups"], 3)
        groups = (
            (self.fx.root / "quarantine" / "QUARANTINE-train-a.txt").read_text().split()
        )
        self.assertEqual(groups, ["tg0", "tg1", "tg7"])
        self.assertEqual(a["quarantine_sha256"], recheck.sha256_text(groups))
        arms = {arm["arm"]: arm for arm in verdict["arms"]}
        self.assertEqual(arms["all"]["exposed_scored_items"], 3)
        self.assertEqual(arms["no-fam"]["exposed_scored_items"], 2)
        self.assertEqual(arms["clean"]["item8"], "valid")
        self.assertNotIn("tg0", json.dumps(verdict))

    def test_judge_passes_without_exposure(self):
        lines = self.fx.train.read_text().splitlines()
        self.fx.train.write_text(
            "".join(line + "\n" for i, line in enumerate(lines) if i in (2, 3, 4, 5, 7))
        )
        spec = json.loads(self.fx.spec.read_text())
        spec["datasets"][0].update(sha256=sha(self.fx.train), rows=5)
        self.fx.spec.write_text(json.dumps(spec))
        verdict = self.judge([])
        self.assertEqual(verdict["verdict"], "PASS")
        self.assertTrue(all(arm["item8"] == "valid" for arm in verdict["arms"]))
        self.assertFalse(
            (self.fx.root / "quarantine" / "QUARANTINE-train-a.txt").exists()
        )

    def test_judge_refuses_a_failed_control_or_other_items(self):
        self.assertEqual(self.fx.scan(), 0)
        scan = json.loads((self.fx.root / "SCAN.json").read_text())
        scan["controls"]["pass"] = False
        bad = self.fx.root / "SCAN-bad.json"
        bad.write_text(json.dumps(scan))
        args = [
            "judge",
            "--spec",
            str(self.fx.spec),
            "--items",
            str(self.fx.root / "items.jsonl"),
            "--protected-items",
            str(self.fx.root / "items.protected.jsonl"),
            "--scan",
            str(bad),
            "--scan-private",
            str(self.fx.root / "SCAN-PRIVATE.json"),
            "--registered-hits",
            str(self.fx.train),
            "--registered-receipt",
            str(bad),
            "--output",
            str(self.fx.root / "V.json"),
            "--quarantine-dir",
            str(self.fx.root / "q"),
        ]
        with self.assertRaisesRegex(ValueError, "planted controls"):
            recheck.main(args)

    def test_verify(self):
        out = self.fx.root / "VERIFY.json"
        self.assertEqual(
            recheck.main(["verify", "--spec", str(self.fx.spec), "--output", str(out)]),
            0,
        )
        self.assertTrue(json.loads(out.read_text())["ok"])
        with self.fx.clean.open("a") as stream:
            stream.write(json.dumps(train_row(30, {"passage": "late"})) + "\n")
        out2 = self.fx.root / "VERIFY-2.json"
        self.assertEqual(
            recheck.main(
                ["verify", "--spec", str(self.fx.spec), "--output", str(out2)]
            ),
            1,
        )


class CommittedSpecTest(unittest.TestCase):
    def test_spec_loads_and_pins_the_published_files(self):
        spec = recheck.load_spec(SPEC)
        files = {d["key"]: d for d in spec["datasets"]}
        self.assertEqual(set(files), {"ib1-r3", "ib2", "pn1-r2", "a20ib12"})
        self.assertEqual(files["ib1-r3"]["sha256"][:8], "1e1b08f3")
        self.assertEqual(files["ib2"]["sha256"][:8], "ee137efa")
        self.assertEqual(files["pn1-r2"]["sha256"][:8], "c1cec06b")
        self.assertEqual(files["a20ib12"]["sha256"][:8], "16cb5bbb")
        self.assertIn("@31b200a3", files["ib1-r3"]["source"])
        self.assertIn("@c5dbdd0a", files["ib2"]["source"])
        tracks = {arm["track"] for arm in spec["arms"]}
        self.assertTrue({"27B M6", "9B M9", "decoder M13", "decoder M14"} <= tracks)

    def test_r2_spec_pins_ib3_r2(self):
        spec = recheck.load_spec(SEALED / "c1-recheck" / "r2-ib3r2.json")
        (entry,) = spec["datasets"]
        self.assertEqual((entry["key"], entry["rows"]), ("ib3-r2", 8752))
        self.assertEqual(entry["sha256"][:8], "9d92d92a")
        self.assertIn("@1c8452da", entry["source"])
        self.assertEqual(spec["arms"], [])

    def test_r3_spec_pins_ib4_p1(self):
        spec = recheck.load_spec(SEALED / "c1-recheck" / "r3-ib4p1.json")
        (entry,) = spec["datasets"]
        self.assertEqual((entry["key"], entry["rows"]), ("ib4-p1", 9459))
        self.assertEqual(entry["sha256"][:8], "6045b456")
        self.assertIn("m6/ib4/p1/ib4.train.jsonl", entry["source"])
        self.assertEqual(spec["arms"], [])

    def test_bad_specs_are_refused(self):
        spec = json.loads(SPEC.read_text())
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "spec.json"
            for change, message in (
                ({"item_set": "v1.3"}, "item_set"),
                (
                    {
                        "arms": [
                            {"track": "T", "arm": "x", "uses": [{"dataset": "nope"}]}
                        ]
                    },
                    "unknown dataset",
                ),
                ({"datasets": [{**spec["datasets"][0], "sha256": "x"}]}, "64 hex"),
            ):
                path.write_text(json.dumps({**spec, **change}))
                with self.subTest(message=message), self.assertRaisesRegex(
                    ValueError, message
                ):
                    recheck.load_spec(path)


class CommittedRegistryTest(unittest.TestCase):
    def test_registry_matches_the_specs_and_the_verdict_receipts(self):
        registry = json.loads((SEALED / "c1-recheck-registry.json").read_text())
        self.assertEqual(registry["schema"], "dev2-c1-recheck-registry/1")
        self.assertEqual(registry["item_set"]["version"], recheck.ITEM_SET)
        self.assertEqual(registry["item_set"]["scored_items"], recheck.SCORED_ITEMS)
        root = SEALED.parent.parent.parent
        files = registry["files"]
        for name, run in registry["rechecks"].items():
            with self.subTest(recheck=name):
                spec_path = root / run["spec"]
                self.assertEqual(sha(spec_path), run["spec_sha256"])
                spec = recheck.load_spec(spec_path)
                self.assertEqual(spec["name"], name)
                receipt = root / run["receipts"] / "VERDICT.json"
                self.assertEqual(sha(receipt), run["verdict_sha256"])
                verdict = json.loads(receipt.read_text())
                self.assertEqual(verdict["verdict"], run["verdict"])
                self.assertTrue(verdict["controls"]["pass"])
                for entry in spec["datasets"]:
                    cell = files[entry["sha256"]]
                    result = verdict["datasets"][entry["key"]]
                    self.assertEqual(
                        (cell["key"], cell["rows"]), (entry["key"], entry["rows"])
                    )
                    self.assertEqual(cell["recheck"], name)
                    self.assertEqual(cell["verdict"], result["verdict"])
                    self.assertEqual(
                        cell["exposed_scored_items"], result["exposed_scored_items"]
                    )
                    self.assertEqual(result["sha256"], entry["sha256"])
                self.assertTrue((root / run["record"]).is_file())
        keys = {cell["key"] for cell in files.values()}
        for digest, cell in registry["covered"].items():
            self.assertRegex(digest, recheck.SHA)
            self.assertTrue(set(cell["covered_by"]) <= keys)
            self.assertTrue(
                all(
                    files_cell["verdict"] == "PASS"
                    for files_cell in files.values()
                    if files_cell["key"] in cell["covered_by"]
                )
            )
        valid = {
            arm
            for name, arms in registry["arms_exposure_0"].items()
            if name != "note"
            for arm in arms
        }
        for run in registry["rechecks"].values():
            verdict = json.loads((root / run["receipts"] / "VERDICT.json").read_text())
            exposed = {a["arm"] for a in verdict["arms"] if a["exposed_scored_items"]}
            self.assertFalse(valid & exposed)

    def test_the_postkey_registry_points_here(self):
        notes = json.loads((SEALED / "c1-postkey-baselines.json").read_text())["notes"]
        self.assertTrue(any("c1-recheck-registry.json" in note for note in notes))


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
        self.assertEqual(self.run_script("--src", "M").returncode, 2)
        absolute = self.run_script("--src", "M", "--spec", "/etc/spec.json")
        self.assertEqual(absolute.returncode, 2)
        self.assertIn("inside the mirror", absolute.stderr)
        self.assertEqual(
            self.run_script("--src", "M", "--spec", "../x.json").returncode, 2
        )
        missing = self.run_script(
            "--src", "no-such-mirror", "--spec", "s.json", "--verify-only"
        )
        self.assertEqual(missing.returncode, 1)
        self.assertIn("no verified mirror", missing.stderr)

    def test_custody(self):
        body = SCRIPT.read_text()
        self.assertNotIn("gold.jsonl", body)
        self.assertIn("tar -xOf - v1/build-5/prompts.jsonl", body)
        self.assertEqual(body.count("docker run"), 1)
        self.assertIn(
            '--network none --mount "type=tmpfs,destination=/data/dev2/private/sealed"',
            body,
        )
        self.assertNotRegex(body, r"--device|--gpus")
        order = [
            body.index("verify-only: done"),
            body.index('v2.eval.sealed.overlap scan --protected "$PROT"'),
            body.index("IFS= read -r KEY <&3"),
            body.index("v2.eval.sealed.recheck items"),
            body.index("v2.eval.sealed.recheck scan"),
            body.index('--protected "$T/items.protected.jsonl"'),
            body.index("v2.eval.sealed.recheck judge"),
        ]
        self.assertEqual(order, sorted(order))
        self.assertRegex(
            body, re.escape('KEY=""\nunset KEY\n[ "$(sha "$T/prompts.jsonl")"')
        )
        for pin in (
            recheck.PROMPTS_SHA256,
            recheck.BUILD_MANIFEST_SHA256,
            recheck.PROTECTED_SHA256,
        ):
            self.assertIn(pin, body)


if __name__ == "__main__":
    unittest.main()
