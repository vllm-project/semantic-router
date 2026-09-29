from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from v2.eval.sealed import event3

SEALED = Path(__file__).resolve().parents[1]
ROOT = SEALED.parents[2]
TABLE = SEALED / "event3-models.json"
SCRIPT = SEALED / "event3.sh"
STAGE = SEALED / "event3-stage-nodeA.sh"
HEX = re.compile(r"^[0-9a-f]{64}$")


def run_plan(tmp: Path, *extra: str, table: Path = TABLE) -> tuple[int, dict]:
    out = tmp / "plan.json"
    if out.exists():
        out.unlink()
    code = event3.main(["plan", "--table", str(table), *extra, "--output", str(out)])
    return code, (json.loads(out.read_text()) if code == 0 else {})


class TableTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp)

    def test_default_plan_resolves_the_card_eligible_pair(self) -> None:
        code, plan = run_plan(self.tmp, "--c27", "f1")
        self.assertEqual(code, 0)
        self.assertEqual(plan["selection"][-3:], ["cand27", "autojev27", "eikos27b"])
        self.assertNotIn("jebadiah27b", plan["selection"])
        table = json.loads(TABLE.read_text())
        table["card_eligible_27b"] = "not yet decided"
        path = self.tmp / "table.json"
        path.write_text(json.dumps(table))
        self.assertEqual(run_plan(self.tmp, "--c27", "f1", table=path)[0], 2)

    def test_default_plan(self) -> None:
        code, plan = run_plan(
            self.tmp, "--c27", "f1", "--peers27", "autojev27,eikos27b"
        )
        self.assertEqual(code, 0)
        self.assertEqual(plan["event"], 3)
        self.assertEqual(plan["events_total"], 3)
        pairs = {(p["left"], p["right"], p["kind"]) for p in plan["pairs"]}
        self.assertEqual(
            pairs,
            {
                ("cand2b", "sol1-16k", "candidate"),
                ("cand2b", "decider2b", "candidate"),
                ("cand2b", "thisthat12", "candidate"),
                ("cand4b", "nox1", "candidate"),
                ("cand4b", "decider4b", "candidate"),
                ("cand4b", "jet62", "candidate"),
                ("cand06-e2", "kai1-8k", "candidate"),
                ("cand27", "autojev27", "candidate"),
                ("cand27", "eikos27b", "candidate"),
                ("kai1-8k", "kai1-e2", "reference"),
            },
        )
        self.assertEqual([d["id"] for d in plan["deviations"]], ["kai1-second-scoring"])
        self.assertTrue(plan["deviations"][0]["approved"].startswith("coordinator"))
        collected = event3.collected(plan)
        self.assertNotIn("cand06-e2", collected)
        self.assertNotIn("kai1-e2", collected)
        self.assertEqual(collected[-3:], ["cand27", "autojev27", "eikos27b"])
        self.assertNotIn("${", json.dumps(plan["models"]))

    def test_9b_slot(self) -> None:
        code, plan = run_plan(self.tmp, "--c27", "f1")
        self.assertEqual(code, 0)
        self.assertFalse({"cand9b", "lux1", "nimble2"} & set(plan["selection"]))
        nine = ["--models", "cand9b,lux1,nimble2"]
        self.assertEqual(run_plan(self.tmp, *nine)[0], 2)
        package = [
            "--c9b-package",
            "/data/x/DEV2.0-9B",
            "--c9b-manifest",
            "c" * 64,
            "--c9b-repo",
            "llm-semantic-router/DEV2.0-9B",
            "--c9b-revision",
            "d" * 40,
        ]
        code, plan = run_plan(self.tmp, *nine, *package)
        self.assertEqual(code, 0)
        row = plan["models"]["cand9b"]
        self.assertEqual(
            (row["model_path"], row["package"]["dir"]), ("/data/x/DEV2.0-9B",) * 2
        )
        self.assertEqual(row["package"]["identity"], row["identity"])
        self.assertEqual(row["parity"]["mode"], "exact")
        pairs = {(p["left"], p["right"]) for p in plan["pairs"]}
        self.assertEqual(pairs, {("cand9b", "lux1"), ("cand9b", "nimble2")})
        code, plan = run_plan(
            self.tmp,
            *nine,
            *package,
            "--c9b-calibration",
            "none",
            "--c9b-identity",
            "e" * 64,
            "--c9b-parity-stored",
            "/data/x/typed-final.predictions.jsonl",
        )
        self.assertEqual(code, 0)
        row = plan["models"]["cand9b"]
        self.assertTrue(row["adapter_spec"].endswith("dev2-dec-package-t1.json"))
        self.assertNotIn("calibration", row["extra"])
        self.assertEqual(row["package"]["identity"], "e" * 64)
        self.assertFalse(any("predictions" in f["path"] for f in row["files"]))
        self.assertEqual(run_plan(self.tmp, "--models", "lux1,nimble2")[0], 2)

    def test_card_eligible_rule_takes_the_two_strongest(self) -> None:
        table = json.loads(TABLE.read_text())
        table["card_eligible_27b"] = ["jebadiah27b", "eikos27b", "autojev27"]
        path = self.tmp / "table.json"
        path.write_text(json.dumps(table))
        code, plan = run_plan(self.tmp, "--c27", "f1", "--models", "cand27", table=path)
        self.assertEqual(code, 0)
        self.assertEqual(plan["selection"], ["cand27", "autojev27", "eikos27b"])

    def test_limits(self) -> None:
        cases = {
            "one configuration of each 1.0 model": [
                "--models",
                "cand2b,sol1,sol1-16k,decider2b",
            ],
            "two open peers": [
                "--c27",
                "f1",
                "--peers27",
                "autojev27,eikos27b,jebadiah27b",
                "--models",
                "cand27",
            ],
            "second 0.6B candidate": ["--models", "cand06-e2,cand06-next,kai1-8k"],
            "F2 not filled in": [
                "--c27",
                "f2",
                "--peers27",
                "autojev27",
                "--models",
                "cand27",
            ],
            "comparator without its candidate": ["--models", "sol1-16k,decider2b"],
        }
        for name, args in cases.items():
            with self.subTest(name):
                self.assertEqual(run_plan(self.tmp, *args)[0], 2)

    def test_rule_checks(self) -> None:
        base = {"repo": "r", "label": "x"}
        models = {
            "c": {**base, "tier": "27B", "role": "candidate"},
            "o": {**base, "tier": "27B", "role": "own1"},
            "i": {**base, "tier": "27B", "role": "internal"},
        }
        errors, _ = event3.check_rules(models, set())
        self.assertTrue(any("no Decision 1.0" in e for e in errors))
        self.assertTrue(any("internal-only" in e for e in errors))
        models = {
            "c": {**base, "tier": "0.6B", "role": "candidate"},
            "n": {
                **base,
                "tier": "0.6B",
                "role": "candidate",
                "deviation": {"id": "second", "approved": None},
            },
        }
        errors, _ = event3.check_rules(models, set())
        self.assertTrue(errors)
        errors, deviations = event3.check_rules(models, {"second"})
        self.assertEqual(errors, [])
        self.assertEqual(deviations[0]["approved"], "--allow-deviation")

    def test_node_b_site(self) -> None:
        code, plan = run_plan(
            self.tmp,
            "--c27",
            "f1",
            "--peers27",
            "autojev27",
            "--models",
            "cand27",
            "--site",
            "node-b",
        )
        self.assertEqual(code, 0)
        row = plan["models"]["cand27"]
        self.assertIn("/m3-release-check/", row["model_path"])
        self.assertEqual(
            row["cache"]["frozen"], "/data/dev2/runs/27b/M3-A-soup/formal/triton-cache"
        )
        self.assertEqual(
            row["extra"]["calibration"],
            "/data/dev2/runs/27b/M3-A-soup/package/calibration.json",
        )
        self.assertNotIn("node_b", row)

    def test_final_27b_release_is_a_parameter(self) -> None:
        code, plan = run_plan(
            self.tmp,
            "--c27",
            "f1",
            "--peers27",
            "autojev27",
            "--models",
            "cand27",
            "--c27-package",
            "/data/x/DEV2.0-26B",
            "--c27-manifest",
            "a" * 64,
            "--c27-repo",
            "llm-semantic-router/DEV2.0-26B",
            "--c27-revision",
            "b" * 40,
        )
        self.assertEqual(code, 0)
        row = plan["models"]["cand27"]
        self.assertEqual(row["model_path"], "/data/x/DEV2.0-26B")
        self.assertEqual(row["package"]["dir"], "/data/x/DEV2.0-26B")
        self.assertEqual(row["package"]["manifest_sha256"], "a" * 64)
        self.assertEqual(row["package"]["identity"], row["identity"])
        self.assertEqual(row["revision"], "b" * 40)

    def test_rows_are_pinned(self) -> None:
        table = json.loads(TABLE.read_text())
        rows = {**table["models"], "cand27": table["candidates_27b"]["f1"]}
        for key, row in rows.items():
            if event3.placeholders(row):
                continue
            with self.subTest(key):
                if row.get("stored"):
                    self.assertRegex(row["stored"]["seal_sha256"], HEX)
                    continue
                self.assertIn(row["parity"]["mode"], event3.PARITY_MODES)
                self.assertTrue(
                    row.get("package") or row.get("trees"), "no weights check"
                )
                if row.get("cache"):
                    self.assertRegex(row["cache"]["sha256"], HEX)
                for tree in row.get("trees", []):
                    self.assertRegex(tree["sha256"], HEX)
                if row["role"] == "candidate":
                    self.assertEqual(row["parity"]["mode"], "exact")
                    self.assertEqual(row["identity"], row["package"]["identity"])
                    self.assertRegex(row["package"]["manifest_sha256"], HEX)
                self.assertRegex(
                    row["revision"], r"^([0-9a-f]{40}|manifest-sha256:[0-9a-f]{64})$"
                )
                module = event3.adapter_module(ROOT, event3.expand(row, table["roots"]))
                self.assertTrue(module.is_file(), module)


class ArgvTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        tmp = Path(tempfile.mkdtemp())
        code, cls.plan = run_plan(tmp, "--c27", "f1", "--peers27", "autojev27,eikos27b")
        shutil.rmtree(tmp)
        assert code == 0

    def argv(
        self, key: str, phase: str, cache: str | None = None, shared: bool = False
    ) -> list[str]:
        return event3.runner_argv(
            self.plan["models"][key],
            self.plan["images"],
            phase,
            "/r/run",
            "6",
            "abc-src",
            "/m/src",
            "owner.eval",
            shared,
            cache,
        )

    def test_frozen_cache_package_row(self) -> None:
        smoke = self.argv("cand2b", "smoke", "/r/run-triton")
        runner, collect = smoke[: smoke.index("--")], smoke[smoke.index("--") + 1 :]
        self.assertIn(
            "sha256:f83b1d10f14dbe46ea14ee56fd3e5d01849673f3739fed5311c99ba54cbc2d54",
            runner,
        )
        self.assertIn("TRITON_CACHE_DIR=/r/run-triton", runner)
        self.assertEqual(runner[runner.index("--mount-rw") + 1], "/r/run-triton")
        self.assertEqual(runner[runner.index("--lease-name") + 1], "owner.eval")
        self.assertNotIn("--shared", runner)
        self.assertEqual(
            collect[1], "/m/src/v2/eval/sealed/adapters/dev2-dec-package-t1.json"
        )
        self.assertIn("max_length=16384", collect)
        self.assertEqual(
            collect[-4:], ["--panels", "typed-final,public231", "--max-items", "80"]
        )
        full = self.argv("cand2b", "collect", "/r/run-triton", shared=True)
        self.assertIn("--shared", full)
        self.assertEqual(full[-2:], ["--panels", "sealed-c1"])
        self.assertNotIn("--max-items", full)

    def test_runtime_mounts(self) -> None:
        kai = self.argv("kai1-8k", "smoke")
        self.assertIn("/data/dev2/tools/envs/kai-lex", kai)
        self.assertFalse(any(a.startswith("TRITON_") for a in kai))
        big = self.argv("cand27", "collect", "/r/c")
        self.assertIn(
            "sha256:dbe5f32b2263b2671ba0b9aaaf18ee20abda189541fc22107e216a2f37d440b1",
            big,
        )
        self.assertIn("HIP_FORCE_DEV_KERNARG=1", big)
        self.assertIn("max_length=32768", big)

    def test_stored_smoke_panels(self) -> None:
        for key in event3.collected(self.plan):
            row = self.plan["models"][key]
            paths = [
                event3.stored_predictions(row, panel) for panel in event3.SMOKE_PANELS
            ]
            self.assertEqual(str(paths[0]), row["parity"]["stored"])
            self.assertTrue(
                paths[1].name.endswith("public231.predictions.jsonl"), paths[1]
            )
        pinned = {f["path"] for f in self.plan["models"]["cand27"]["files"]}
        self.assertIn(
            str(event3.stored_predictions(self.plan["models"]["cand27"], "public231")),
            pinned,
        )

    def test_misuse(self) -> None:
        with self.assertRaises(ValueError):
            self.argv("cand2b", "smoke")
        with self.assertRaises(ValueError):
            self.argv("kai1-8k", "smoke", "/r/c")
        with self.assertRaises(ValueError):
            self.argv("cand06-e2", "collect")


def choice(label: str, p: float = 0.7) -> dict:
    return {"type": "choice", "choice": label, "probabilities": {label: p, "zz": 1 - p}}


class ParityTest(unittest.TestCase):
    row = {
        "parity": {"mode": "exact", "tolerance": 1e-4, "max_changed_fraction": 0.1},
        "identity": "m",
    }

    def rows(self, override: dict | None = None, n: int = 20) -> list[dict]:
        out = []
        for i in range(n):
            answers = {"decision": (override or {}).get(i, choice("a"))}
            out.append({"id": f"t{i}", "model_sha256": "m", "answers": answers})
        return out

    def check(self, smoke, mode="exact"):
        stored = {r["id"]: r for r in self.rows()}
        row = {**self.row, "parity": {**self.row["parity"], "mode": mode}}
        return event3.compare_smoke(smoke, stored, row)

    def test_identical(self) -> None:
        result = self.check(self.rows())
        self.assertTrue(result["passed"])
        self.assertEqual((result["answers"], result["changed"]), (20, 0))
        self.assertEqual(result["answers_by_type"], {"choice": 20})

    def test_changed_answer(self) -> None:
        smoke = self.rows({3: choice("zz", 0.8)})
        self.assertFalse(self.check(smoke)["passed"])
        self.assertTrue(self.check(smoke, "near")["passed"])
        smoke = self.rows({i: choice("zz", 0.8) for i in range(3)})
        self.assertFalse(self.check(smoke, "near")["passed"])

    def test_points_and_drift(self) -> None:
        self.assertEqual(event3.point({"type": "noul", "noul": 0.6}), True)
        self.assertIsNone(event3.point({"type": "noul", "noul": 0.5}))
        self.assertEqual(event3.point({"type": "score", "score": 2.2}), 2)
        self.assertAlmostEqual(
            event3.drift(
                {"type": "score", "score": 2.2}, {"type": "score", "score": 2.3}
            ),
            0.1,
        )
        smoke = self.rows({0: choice("a", 0.7002)})
        result = self.check(smoke)
        self.assertEqual(result["changed"], 0)
        self.assertFalse(result["passed"])

    def test_identity_and_missing(self) -> None:
        smoke = self.rows()
        smoke[0]["model_sha256"] = "other"
        self.assertFalse(self.check(smoke, "near")["passed"])
        self.assertFalse(self.check([], "none")["passed"])
        smoke = self.rows()
        smoke[0]["id"] = "unknown"
        self.assertFalse(self.check(smoke, "near")["passed"])


class ParityCmdTest(unittest.TestCase):
    def test_every_smoke_panel_is_compared(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            rows = {
                "typed-final": [{"id": "t", "answers": {"q": choice("a")}}],
                "public231": [
                    {"id": "p", "answers": {"q": {"type": "noul", "noul": 0.8}}}
                ],
            }
            for panel, content in rows.items():
                for folder in ("stored/output", "run/smoke"):
                    path = tmp / folder / f"{panel}.predictions.jsonl"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("".join(json.dumps(r) + "\n" for r in content))
            stored = tmp / "stored/output/typed-final.predictions.jsonl"
            plan = {
                "schema": event3.SCHEMA,
                "selection": ["m"],
                "models": {"m": {"parity": {"mode": "exact", "stored": str(stored)}}},
            }
            path = tmp / "plan.json"
            path.write_text(json.dumps(plan))
            args = [
                "parity",
                "--plan",
                str(path),
                "--key",
                "m",
                "--run-dir",
                str(tmp / "run"),
            ]
            self.assertEqual(event3.main(args + ["--output", str(tmp / "a.json")]), 0)
            report = json.loads((tmp / "a.json").read_text())
            self.assertEqual(
                report["panels"]["public231"]["answers_by_type"], {"noul": 1}
            )
            (tmp / "run/smoke/public231.predictions.jsonl").unlink()
            self.assertEqual(event3.main(args + ["--output", str(tmp / "b.json")]), 1)


class DigestTest(unittest.TestCase):
    def test_matches_the_coreutils_recipe(self) -> None:
        if not (
            shutil.which("find") and shutil.which("sha256sum") and shutil.which("xargs")
        ):
            self.skipTest("coreutils not available")
        with tempfile.TemporaryDirectory() as tmp:
            root, outside = Path(tmp) / "model", Path(tmp) / "blob"
            (root / "backbone").mkdir(parents=True)
            (root / ".cache" / "huggingface").mkdir(parents=True)
            (root / "backbone" / "__pycache__").mkdir()
            outside.write_bytes(b"weights")
            (root / "backbone" / "model.safetensors").symlink_to(outside)
            (root / "config.json").write_text("{}")
            (root / "a b.txt").write_text("space")
            (root / ".cache" / "huggingface" / "x.lock").write_text("ignored")
            (root / "backbone" / "__pycache__" / "m.pyc").write_bytes(b"ignored")
            recipe = (
                "find -L . -type f ! -path './.cache/*' ! -path './.git/*' ! -path '*/__pycache__/*'"
                " -print0 | LC_ALL=C sort -z | xargs -0 sha256sum | sha256sum"
            )
            shell = subprocess.run(
                ["bash", "-c", recipe],
                cwd=root,
                capture_output=True,
                text=True,
                check=True,
            )
            digest, count = event3.tree_digest(root)
            self.assertEqual(digest, shell.stdout.split()[0])
            self.assertEqual(count, 3)


class VerifyTest(unittest.TestCase):
    def test_unstaged_package_is_reported_not_raised(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            code, plan = run_plan(
                tmp, "--c27", "f1", "--peers27", "autojev27", "--models", "cand27"
            )
            self.assertEqual(code, 0)
            row = plan["models"]["cand27"]
            row["package"]["dir"] = str(tmp / "missing")
            out = event3.check_package(row["package"], 1)
            self.assertFalse(out["ok"])
            self.assertTrue(out["problems"][0].startswith("no MODEL_MANIFEST.json"))
            plan["images"] = {}
            path = tmp / "p.json"
            path.write_text(json.dumps(plan))
            code = event3.main(
                [
                    "verify",
                    "--plan",
                    str(path),
                    "--src-root",
                    str(ROOT),
                    "--output",
                    str(tmp / "v.json"),
                ]
            )
            self.assertEqual(code, 1)
            report = json.loads((tmp / "v.json").read_text())
            self.assertFalse(report["models"]["cand27"]["ok"])


class StoredTest(unittest.TestCase):
    def test_seal_checks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            preds = tmp / "p.jsonl"
            preds.write_text('{"id": "x"}\n')
            seal = tmp / "SEAL-C1.json"
            seal.write_text(
                json.dumps(
                    {
                        "prompts_sha256": event3.PROMPTS_SHA256,
                        "predictions_sha256": hashlib.sha256(
                            preds.read_bytes()
                        ).hexdigest(),
                        "missing": 0,
                    }
                )
            )
            plan = {
                "schema": event3.SCHEMA,
                "selection": ["s"],
                "models": {
                    "s": {
                        "stored": {
                            "predictions": str(preds),
                            "seal": str(seal),
                            "seal_sha256": hashlib.sha256(
                                seal.read_bytes()
                            ).hexdigest(),
                        }
                    }
                },
            }
            path = tmp / "plan.json"
            path.write_text(json.dumps(plan))
            self.assertEqual(
                event3.main(
                    ["stored", "--plan", str(path), "--output", str(tmp / "a.json")]
                ),
                0,
            )
            preds.write_text('{"id": "y"}\n')
            self.assertEqual(
                event3.main(
                    ["stored", "--plan", str(path), "--output", str(tmp / "b.json")]
                ),
                1,
            )


class ScriptTest(unittest.TestCase):
    text = SCRIPT.read_text()

    def test_syntax(self) -> None:
        for path in (SCRIPT, STAGE):
            subprocess.run(["bash", "-n", str(path)], check=True)
        if shutil.which("shellcheck"):
            subprocess.run(
                ["shellcheck", "-S", "warning", str(SCRIPT), str(STAGE)], check=True
            )

    def test_key_is_read_once_after_the_preflight_only_exit(self) -> None:
        reads = [m.start() for m in re.finditer(r"read -r KEY", self.text)]
        self.assertEqual(len(reads), 1)
        stop = self.text.index(
            'log "preflight-only: stopped before the key; C1 not touched"'
        )
        self.assertLess(stop, reads[0])
        self.assertIn(
            'if [ "$MODE" = event ]; then\n  exec 3<&0 0</dev/null\nelse\n  exec 0</dev/null\nfi',
            self.text,
        )
        self.assertLess(
            self.text.index("exec 3<&0 0</dev/null"), self.text.index("helper plan")
        )

    def test_sealed_directory_only_in_event_mode(self) -> None:
        before = self.text[
            : self.text.index('log "preflight-only: stopped before the key')
        ]
        uses = [
            line.strip()
            for line in before.splitlines()
            if "$C1" in line and not line.startswith("C1=")
        ]
        self.assertEqual(
            uses,
            [
                'echo "$t EVENT3 $*" >>"$C1/ACCESS.log"',
                'openssl enc -d -aes-256-cbc -pbkdf2 -iter 200000 -pass fd:4 -in "$C1/c1-v1-bundle.tar.enc" 4<<<"$KEY" | tar -xOf - "$1"',
            ],
        )
        log_body = before[before.index("log() {") : before.index("abort() {")]
        self.assertLess(
            log_body.index('if [ "$MODE" = event ]'), log_body.index("ACCESS.log")
        )

    def test_event_is_used_only_when_decryption_starts(self) -> None:
        make = self.text.index('mkdir "$E"')
        self.assertLess(self.text.index("read -r KEY"), make)
        self.assertNotIn('mkdir -p "$E"', self.text)
        first_decrypt = self.text.index("decrypt v1/build-5/prompts.jsonl")
        self.assertLess(make, first_decrypt)
        self.assertLess(self.text.index("DECRYPTED=1\n"), first_decrypt)
        self.assertLess(
            self.text.index('rm -f "$PANEL"\nlog "prompts removed'),
            self.text.index("decrypt v1/build-5/gold.jsonl"),
        )
        self.assertLess(
            self.text.index("score seal"),
            self.text.index("decrypt v1/build-5/gold.jsonl"),
        )

    def test_scan_verdict_interlock_precedes_the_key(self) -> None:
        gate = self.text.index("python3 -m v2.eval.sealed.scanverdict check")
        self.assertLess(gate, self.text.index("read -r KEY"))
        self.assertLess(gate, self.text.index('mkdir "$E"'))
        lines = self.text[:gate].splitlines()
        self.assertEqual(lines[-3], 'if [ "$MODE" = event ]; then')
        self.assertIn(
            "e37e73f9c1519362bda350ec7d475ed6acf7e47ed1f83dc084b3a5a93247acfc",
            self.text,
        )
        self.assertLess(self.text.index('log "verify-only: done"'), gate)

    def test_children_never_hold_the_key_descriptor(self) -> None:
        self.assertIn(
            'helper() { python3 -m v2.eval.sealed.event3 "$@" 3<&-; }', self.text
        )
        self.assertIn(
            'bash "$S/v2/eval/run_same_panel.sh" "${argv[@]}" >"$d.log" 2>&1 3<&-',
            self.text,
        )


if __name__ == "__main__":
    unittest.main()
