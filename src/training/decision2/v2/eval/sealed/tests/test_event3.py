from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from v2.eval.sealed import event3

SEALED = Path(__file__).resolve().parents[1]
ROOT = SEALED.parents[2]
TABLE = SEALED / "event3-models.json"
SCRIPT = SEALED / "event3.sh"
STAGE = SEALED / "event3-stage-nodeA.sh"
HEX = re.compile(r"^[0-9a-f]{64}$")
IDENTITY_FP32 = "b9d973b3ef555457da2dfa839c45cef3d4a98a47a43d1fd724e77fdec4aa125d"
IDENTITY_BF16 = "b1ed5a71038b474902d2a6dfebeee0c6f7ce901097553b4621f6f0a23fb6da98"
NINE = ["--models", "cand9b,lux1,nimble2"]


def run_plan(tmp: Path, *extra: str, table: Path = TABLE) -> tuple[int, dict]:
    out = tmp / "plan.json"
    if out.exists():
        out.unlink()
    code = event3.main(["plan", "--table", str(table), *extra, "--output", str(out)])
    return code, (json.loads(out.read_text()) if code == 0 else {})


def plan_error(tmp: Path, *extra: str) -> tuple[int, str]:
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        code, _ = run_plan(tmp, *extra)
    return code, err.getvalue()


def package_9b(tmp: Path, identity: str = IDENTITY_FP32) -> list[str]:
    """A stand-in DEV2.0-8B package (a manifest only) and its --c9b-package / --c9b-manifest."""
    root = tmp / f"DEV2.0-8B-{identity[:8]}"
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "MODEL_MANIFEST.json"
    manifest.write_text(
        json.dumps(
            {
                "identity": {"model_sha256": identity},
                "files_sha256": {"decision_head.safetensors": "0" * 64},
            }
        )
    )
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    return ["--c9b-package", str(root), "--c9b-manifest", digest]


def write(path: Path, data: str | bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = data.encode() if isinstance(data, str) else data
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def synthetic_27b(b27: Path, package: Path) -> dict:
    """An F2-like candidate as the ~27B track lays it out: package, calibration, formal run."""
    files = {
        name: write(package / name, content)
        for name, content in (
            ("adapter/adapter_model.safetensors", b"lora"),
            ("decision_head.safetensors", b"head"),
            ("decision_config.json", "{}"),
        )
    }
    base = {"config.json": "b" * 64}
    fingerprint = {f"checkpoint/{k}": v for k, v in files.items()}
    fingerprint.update({f"source/{k}": v for k, v in base.items()})
    canonical = json.dumps(fingerprint, sort_keys=True, separators=(",", ":"))
    identity = hashlib.sha256(canonical.encode()).hexdigest()
    manifest = {
        "files_sha256": files,
        "base": {"files_sha256": base},
        "identity": {"model_sha256": identity, "fingerprint_files": fingerprint},
    }
    made = {
        "identity": identity,
        "manifest": write(package / "MODEL_MANIFEST.json", json.dumps(manifest)),
    }
    soup = b27 / "M3-S-soup"
    made["calibration"] = write(
        soup / "package" / "calibration.json", json.dumps({"model_sha256": identity})
    )
    formal = soup / "formal"
    for panel, count in (("typed-final", 3), ("public231", 2)):
        rows = [
            {
                "id": f"{panel}-{i}",
                "model_sha256": identity,
                "calibration_sha256": made["calibration"],
                "answers": {},
            }
            for i in range(count)
        ]
        made[panel] = write(
            formal / "output" / f"{panel}.predictions.jsonl",
            "".join(json.dumps(r) + "\n" for r in rows),
        )
    seal = {"panels": {p: {"predictions_sha256": made[p]} for p in event3.SMOKE_PANELS}}
    seal_sha = write(formal / "SEAL.json", json.dumps(seal))
    write(
        formal / "REPORT.json",
        json.dumps({"v3": {"score": 68.12345}, "seal_sha256": seal_sha}),
    )
    made["collect"] = write(
        formal / "COLLECT.json", json.dumps({"adapter_module_sha256": "a" * 64})
    )
    write(formal / "triton-cache" / "g" / "k.autotune.json", "{}")
    write(formal / "triton-cache" / "g" / "k.hsaco", b"kernel")
    made.update(formal=formal, soup=soup, package=package)
    return made


def receipt_args(made: dict, output: Path, *extra: str) -> list[str]:
    return [
        "c27-receipt",
        "--package-dir",
        str(made["package"]),
        "--formal-run",
        str(made["formal"]),
        "--calibration",
        str(made["soup"] / "package" / "calibration.json"),
        "--cache",
        str(made["formal"] / "triton-cache"),
        *extra,
        "--output",
        str(output),
    ]


class TableTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp)

    def test_default_plan_resolves_the_card_eligible_pair(self) -> None:
        code, plan = run_plan(self.tmp, "--c27", "f1", *package_9b(self.tmp))
        self.assertEqual(code, 0)
        self.assertEqual(plan["selection"][-3:], ["cand27", "autojev27", "eikos27b"])
        self.assertTrue({"cand9b", "lux1", "nimble2"} <= set(plan["selection"]))
        self.assertNotIn("jebadiah27b", plan["selection"])
        table = json.loads(TABLE.read_text())
        table["card_eligible_27b"] = "not yet decided"
        path = self.tmp / "table.json"
        path.write_text(json.dumps(table))
        code, _ = run_plan(self.tmp, "--c27", "f1", *package_9b(self.tmp), table=path)
        self.assertEqual(code, 2)

    def test_default_plan(self) -> None:
        code, plan = run_plan(
            self.tmp,
            "--c27",
            "f1",
            "--peers27",
            "autojev27,eikos27b",
            *package_9b(self.tmp),
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
                ("cand9b", "lux1", "candidate"),
                ("cand9b", "nimble2", "candidate"),
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

    def test_default_plan_needs_the_9b_package(self) -> None:
        flags = package_9b(self.tmp)
        for extra in ([], flags[:2], flags[2:]):
            with self.subTest(extra=extra):
                code, err = plan_error(self.tmp, "--c27", "f1", *extra)
                self.assertEqual(code, 2)
                self.assertIn("--c9b-package DIR", err)
                self.assertIn("--c9b-manifest SHA", err)
        nine = {"cand9b", "lux1", "nimble2"}
        rows = [
            k for k in json.loads(TABLE.read_text())["default_models"] if k not in nine
        ]
        code, plan = run_plan(self.tmp, "--c27", "f1", "--models", ",".join(rows))
        self.assertEqual(code, 0)
        self.assertFalse(nine & set(plan["selection"]))
        self.assertEqual(run_plan(self.tmp, "--models", "lux1,nimble2")[0], 2)

    def test_9b_identity_comes_from_the_manifest(self) -> None:
        derived = "/data/dev2/runs/release/inputs/dev2-8b-t1/derived"
        for identity in (IDENTITY_FP32, IDENTITY_BF16):
            with self.subTest(identity=identity[:8]):
                flags = package_9b(self.tmp, identity)
                code, plan = run_plan(self.tmp, *NINE, *flags)
                self.assertEqual(code, 0)
                row = plan["models"]["cand9b"]
                self.assertEqual(
                    (row["identity"], row["package"]["identity"]), (identity,) * 2
                )
                self.assertEqual(row["package"]["manifest_sha256"], flags[3])
                self.assertEqual(
                    (row["model_path"], row["package"]["dir"]), (flags[1],) * 2
                )
                self.assertEqual(row["revision"], f"manifest-sha256:{flags[3]}")
                self.assertEqual(row["repo"], "llm-semantic-router/DEV2.0-8B")
                self.assertEqual(row["extra"]["model_id"], row["repo"])
                self.assertTrue(
                    row["adapter_spec"].endswith("dev2-dec-package-t1.json")
                )
                self.assertNotIn("calibration", row["extra"])
                self.assertEqual(row["env"], ["HIP_FORCE_DEV_KERNARG=1"])
                self.assertEqual(
                    row["cache"]["frozen"], "/data/dev2/runs/9b/formal-m4/triton-cache"
                )
                self.assertTrue(row["cache"]["sha256"].startswith("5604ffdc5f19"))
                self.assertEqual(
                    row["parity"],
                    {
                        "mode": "exact",
                        "stored": f"{derived}/typed-final.predictions.jsonl",
                        "tolerance": 0.0001,
                    },
                )
                self.assertEqual(
                    {f["path"]: f["sha256"][:8] for f in row["files"]},
                    {
                        f"{derived}/typed-final.predictions.jsonl": "22c9c689",
                        f"{derived}/public231.predictions.jsonl": "813b5441",
                    },
                )
                self.assertFalse({"identity_allowed", "calibrated"} & set(row))
                pairs = {(p["left"], p["right"]) for p in plan["pairs"]}
                self.assertEqual(pairs, {("cand9b", "lux1"), ("cand9b", "nimble2")})
        code, plan = run_plan(
            self.tmp,
            *NINE,
            *package_9b(self.tmp),
            "--c9b-revision",
            "d" * 40,
            "--c9b-repo",
            "llm-semantic-router/DEV2.0-8B-rc",
        )
        self.assertEqual(code, 0)
        row = plan["models"]["cand9b"]
        self.assertEqual(row["revision"], "d" * 40)
        self.assertEqual(row["extra"]["model_id"], "llm-semantic-router/DEV2.0-8B-rc")

    def test_9b_manifest_hash_mismatch_is_refused(self) -> None:
        flags = package_9b(self.tmp)
        flags[3] = "c" * 64
        code, err = plan_error(self.tmp, *NINE, *flags)
        self.assertEqual(code, 2)
        self.assertIn("not --c9b-manifest", err)
        missing = ["--c9b-package", str(self.tmp / "none"), "--c9b-manifest", "c" * 64]
        code, err = plan_error(self.tmp, *NINE, *missing)
        self.assertEqual(code, 2)
        self.assertIn("no MODEL_MANIFEST.json", err)

    def test_9b_identity_not_allowed_is_refused(self) -> None:
        other = package_9b(self.tmp, "e" * 64)
        code, err = plan_error(self.tmp, *NINE, *other)
        self.assertEqual(code, 2)
        self.assertIn("not an allowed DEV2.0-8B weights identity", err)
        code, plan = run_plan(self.tmp, *NINE, *other, "--c9b-identity", "e" * 64)
        self.assertEqual(code, 0)
        self.assertEqual(plan["models"]["cand9b"]["identity"], "e" * 64)
        code, err = plan_error(
            self.tmp, *NINE, *package_9b(self.tmp), "--c9b-identity", "e" * 64
        )
        self.assertEqual(code, 2)
        self.assertIn("is not the manifest identity", err)

    def test_9b_calibration_switches_back_to_the_scored_run(self) -> None:
        calibration = self.tmp / "cal" / "calibration.json"
        write(calibration, json.dumps({"model_sha256": IDENTITY_FP32}))
        switch = ["--c9b-calibration", str(calibration)]
        code, plan = run_plan(self.tmp, *NINE, *package_9b(self.tmp), *switch)
        self.assertEqual(code, 0)
        row = plan["models"]["cand9b"]
        self.assertTrue(row["adapter_spec"].endswith("adapter-spec-infer-dec.json"))
        self.assertEqual(row["extra"]["calibration"], str(calibration))
        self.assertIn(str(calibration.parent), row["mounts"])
        self.assertEqual(
            row["parity"]["stored"],
            "/data/dev2/runs/9b/formal-m4/K-a13-16k/output/typed-final.predictions.jsonl",
        )
        self.assertEqual(
            {f["sha256"][:8] for f in row["files"]}, {"ad672049", "82c5fe2c"}
        )
        code, err = plan_error(
            self.tmp, *NINE, *package_9b(self.tmp, IDENTITY_BF16), *switch
        )
        self.assertEqual(code, 2)
        self.assertIn("runs only at T = 1", err)
        code, plan = run_plan(
            self.tmp,
            *NINE,
            *package_9b(self.tmp),
            "--c9b-calibration",
            "none",
            "--c9b-parity-stored",
            "/data/x/typed-final.predictions.jsonl",
            "--c9b-tolerance",
            "0.001",
        )
        self.assertEqual(code, 0)
        row = plan["models"]["cand9b"]
        self.assertTrue(row["adapter_spec"].endswith("dev2-dec-package-t1.json"))
        self.assertEqual(
            row["parity"]["stored"], "/data/x/typed-final.predictions.jsonl"
        )
        self.assertEqual(row["parity"]["tolerance"], 0.001)
        self.assertFalse(any("predictions" in f["path"] for f in row["files"]))

    def test_9b_cache_override(self) -> None:
        nine = [*NINE, *package_9b(self.tmp)]
        cache = ["--c9b-cache", "/data/x/cache", "--c9b-cache-sha", "f" * 64]
        code, plan = run_plan(self.tmp, *nine, *cache)
        self.assertEqual(code, 0)
        self.assertEqual(
            plan["models"]["cand9b"]["cache"],
            {"frozen": "/data/x/cache", "sha256": "f" * 64},
        )
        self.assertEqual(run_plan(self.tmp, *nine, *cache[:2])[0], 2)

    def test_f1_is_the_uploaded_release(self) -> None:
        code, plan = run_plan(
            self.tmp, "--c27", "f1", "--peers27", "autojev27", "--models", "cand27"
        )
        self.assertEqual(code, 0)
        row = plan["models"]["cand27"]
        dest = "/data/dev2/runs/eval/m4/c1-event3-assets/f1/release/DEV2.0-26B@5683c6f0"
        self.assertEqual(row["label"], "DEV2.0-26B (F1, M3-A soup)")
        self.assertEqual(
            (row["repo"], row["revision"]),
            (
                "llm-semantic-router/DEV2.0-26B",
                "5683c6f04fb6ac88484cee82220f1a1a2fa43392",
            ),
        )
        self.assertEqual((row["model_path"], row["package"]["dir"]), (dest, dest))
        self.assertEqual(
            row["package"]["manifest_sha256"],
            "ab07948eeff64d72b3dc179e72f6083a984b3fe8c481da3f9175480091a07cf5",
        )
        self.assertEqual(row["identity"], row["package"]["identity"])
        self.assertTrue(row["identity"].startswith("b7fd44e3"))
        self.assertEqual(
            row["package"]["base"], "/data/decision20-20260926/models/Qwen3.8-27B"
        )
        self.assertEqual(row["extra"]["model_id"], "llm-semantic-router/DEV2.0-26B")
        steps = row["stage_node_a"]
        self.assertEqual(
            steps[0],
            {
                "kind": "hf",
                "repo": row["repo"],
                "revision": row["revision"],
                "to": dest,
            },
        )
        self.assertEqual([s["kind"] for s in steps[1:]], ["copy"] * 5)
        self.assertFalse(any("m3-release-check" in s.get("from", "") for s in steps))

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
        self.assertEqual(
            row["model_path"],
            "/data/dev2/runs/release/dev2-26b-release-20260929T051011Z/download/DEV2.0-26B",
        )
        self.assertEqual(row["package"]["dir"], row["model_path"])
        self.assertTrue(row["package"]["manifest_sha256"].startswith("ab07948e"))
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
        code, cls.plan = run_plan(
            tmp, "--c27", "f1", "--peers27", "autojev27,eikos27b", *package_9b(tmp)
        )
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
        self.assertIn("model_id=llm-semantic-router/DEV2.0-26B", big)

    def test_9b_package_row(self) -> None:
        smoke = self.argv("cand9b", "smoke", "/r/run-triton")
        runner, collect = smoke[: smoke.index("--")], smoke[smoke.index("--") + 1 :]
        self.assertIn("HIP_FORCE_DEV_KERNARG=1", runner)
        self.assertIn("TRITON_CACHE_DIR=/r/run-triton", runner)
        self.assertEqual(
            collect[1], "/m/src/v2/eval/sealed/adapters/dev2-dec-package-t1.json"
        )
        self.assertIn("model_id=llm-semantic-router/DEV2.0-8B", collect)
        self.assertIn("max_length=16384", collect)
        self.assertFalse(any(a.startswith("calibration=") for a in collect))
        revision = collect[collect.index("--revision") + 1]
        self.assertTrue(revision.startswith("manifest-sha256:"), revision)

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
        for key in ("cand27", "cand9b"):
            row = self.plan["models"][key]
            pinned = {f["path"] for f in row["files"]}
            self.assertIn(str(event3.stored_predictions(row, "public231")), pinned)

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

    def test_a_manifest_must_pin_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            digest = write(
                Path(tmp) / "MODEL_MANIFEST.json",
                json.dumps({"identity": {"model_sha256": "m"}}),
            )
            out = event3.check_package(
                {"dir": tmp, "manifest_sha256": digest, "identity": "m"}, 1
            )
            self.assertEqual(out["problems"], ["manifest pins no package files"])


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


class StageTest(unittest.TestCase):
    def test_steps_carry_the_include_pattern(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            code, plan = run_plan(
                tmp,
                "--c27",
                "f1",
                "--peers27",
                "autojev27,eikos27b",
                "--models",
                "cand27",
            )
            self.assertEqual(code, 0)
            plan["models"]["cand27"]["stage_node_a"][0][
                "include"
            ] = "m3/M3-S-soup/package/*"
            path = tmp / "p.json"
            path.write_text(json.dumps(plan))
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(event3.main(["stage", "--plan", str(path)]), 0)
        fields = out.getvalue().split("\0")[:-1]
        self.assertEqual(len(fields) % 7, 0)
        steps = [fields[i : i + 7] for i in range(0, len(fields), 7)]
        self.assertEqual({s[1] for s in steps}, {"cand27", "autojev27", "eikos27b"})
        self.assertEqual(steps[0][0], "hf")
        self.assertEqual(
            steps[0][4:],
            [
                "llm-semantic-router/DEV2.0-26B",
                "5683c6f04fb6ac88484cee82220f1a1a2fa43392",
                "m3/M3-S-soup/package/*",
            ],
        )
        self.assertEqual(steps[1][6], "")


class C27ReceiptTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp)

    def receipt(
        self, made: dict, *extra: str, name: str = "r.json"
    ) -> tuple[int, dict]:
        out = self.tmp / name
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            code = event3.main(receipt_args(made, out, *extra))
        return code, json.loads(out.read_text())

    def test_receipt_records_what_the_table_pins(self) -> None:
        made = synthetic_27b(self.tmp / "b27", self.tmp / "pkg")
        cache = made["formal"] / "triton-cache"
        digest = event3.cache_digest(event3.SRC_ROOT, cache)
        write(
            cache.with_name("triton-cache.post.json"),
            json.dumps({"post_sha256": digest}),
        )
        code, r = self.receipt(made)
        self.assertEqual(code, 0, r["problems"])
        self.assertEqual(r["manifest_sha256"], made["manifest"])
        self.assertEqual(r["identity"], made["identity"])
        self.assertEqual(r["package"]["fingerprint"]["sha256"], made["identity"])
        self.assertEqual(r["package"]["files"], 3)
        self.assertEqual(r["calibration"]["sha256"], made["calibration"])
        for panel in event3.SMOKE_PANELS:
            self.assertEqual(r["predictions"][panel]["sha256"], made[panel])
            self.assertTrue(r["predictions"][panel]["sealed"])
        self.assertEqual(r["collect"]["sha256"], made["collect"])
        self.assertEqual(r["cache"], {"sha256": digest, "post_sha256": digest})
        self.assertEqual(r["v3"], 68.12345)
        self.assertEqual(
            r["inputs"]["public231"],
            str(made["formal"] / "output" / "public231.predictions.jsonl"),
        )
        self.assertIsNone(r["package"]["runtime_identity"])
        runtime = {"model_sha256": made["identity"]}
        with mock.patch(
            "training.model.infer.checkpoint_fingerprint", return_value=runtime
        ) as fingerprint:
            code, r = self.receipt(
                made, "--base", str(self.tmp / "base"), name="b.json"
            )
        self.assertEqual(code, 0, r["problems"])
        self.assertEqual(r["package"]["runtime_identity"], made["identity"])
        fingerprint.assert_called_once_with(made["package"], self.tmp / "base")

    def test_receipt_failures(self) -> None:
        def other_weights_rows(made: dict) -> None:
            path = made["formal"] / "output" / "public231.predictions.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[0]["model_sha256"] = "0" * 64
            made["public231"] = write(path, "".join(json.dumps(r) + "\n" for r in rows))
            seal = {
                "panels": {
                    p: {"predictions_sha256": made[p]} for p in event3.SMOKE_PANELS
                }
            }
            seal_sha = write(made["formal"] / "SEAL.json", json.dumps(seal))
            report = {"v3": {"score": 1.0}, "seal_sha256": seal_sha}
            write(made["formal"] / "REPORT.json", json.dumps(report))

        cases = {
            "rows lack the identity": other_weights_rows,
            "not the ones in SEAL.json": lambda made: write(
                made["formal"] / "SEAL.json", json.dumps({"panels": {}})
            ),
            "bound to other weights": lambda made: write(
                made["soup"] / "package" / "calibration.json",
                json.dumps({"model_sha256": "0" * 64}),
            ),
            "package files differ": lambda made: write(
                made["package"] / "decision_head.safetensors", b"other"
            ),
            "the cache changed": lambda made: write(
                made["formal"] / "triton-cache.post.json",
                json.dumps({"post_sha256": "0"}),
            ),
        }
        for index, (problem, spoil) in enumerate(cases.items()):
            with self.subTest(problem):
                made = synthetic_27b(
                    self.tmp / f"b27-{index}", self.tmp / f"pkg-{index}"
                )
                spoil(made)
                code, r = self.receipt(made, name=f"r{index}.json")
                self.assertEqual(code, 1)
                self.assertFalse(r["passed"])
                self.assertTrue(any(problem in p for p in r["problems"]), r["problems"])


class FillC27Test(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        text = TABLE.read_text()
        for root, value in (
            ("B27", "/data/dev2/runs/27b"),
            ("X", "/data/dev2/runs/eval/m4/c1-event3-assets"),
        ):
            text = text.replace(
                f'"{root}": "{value}"', f'"{root}": "{self.tmp / root}"'
            )
        self.table = self.tmp / "event3-models.json"
        self.table.write_text(text)
        self.made = synthetic_27b(self.tmp / "B27", self.tmp / "pkg")
        self.receipt = self.make_receipt("r.json")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp)

    def make_receipt(self, name: str, *extra: str) -> Path:
        out = self.tmp / name
        with contextlib.redirect_stdout(io.StringIO()):
            args = receipt_args(self.made, out, *extra)
            assert event3.main(args) == 0, out.read_text()
        return out

    def fill(self, *extra: str, receipt: Path | None = None) -> int:
        args = ["fill-c27", "--table", str(self.table), "--key", "f2"]
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            return event3.main(
                args + ["--receipt", str(receipt or self.receipt), *extra]
            )

    def block(self) -> dict:
        return json.loads(self.table.read_text())["candidates_27b"]["f2"]

    def test_fill_then_plan(self) -> None:
        before = self.table.read_text()
        self.assertEqual(self.fill(), 2)
        self.assertEqual(self.table.read_text(), before)
        self.assertEqual(self.fill("--revision", "e" * 40), 0)
        after = self.table.read_text()
        self.assertTrue(after.startswith(before[: before.index('    "f2": {')]))
        self.assertTrue(after.endswith(before[before.index('\n  "models": {') :]))
        block = self.block()
        hashes = [
            self.made[k] for k in ("calibration", "typed-final", "public231", "collect")
        ]
        self.assertEqual(block["v3"], 68.123)
        self.assertEqual(block["identity"], self.made["identity"])
        self.assertEqual(block["package"]["identity"], self.made["identity"])
        self.assertEqual(block["package"]["manifest_sha256"], self.made["manifest"])
        self.assertEqual(block["revision"], "e" * 40)
        self.assertEqual(block["stage_node_a"][0]["revision"], "e" * 40)
        self.assertEqual([f["sha256"] for f in block["files"]], hashes)
        self.assertEqual([f["sha256"] for f in block["node_b"]["files"]], hashes)
        self.assertEqual(block["node_b"]["package.dir"], str(self.tmp / "pkg"))
        self.assertEqual(
            block["node_b"]["mounts"], ["${M}/Qwen3.8-27B", "${B27}/M3-S-soup/package"]
        )
        peers = ["--peers27", "autojev27", "--models", "cand27"]
        code, plan = run_plan(self.tmp, "--c27", "f2", *peers, table=self.table)
        self.assertEqual(code, 0)
        row = plan["models"]["cand27"]
        self.assertEqual(
            row["package"]["dir"], str(self.tmp / "X" / "f2/release/DEV2.0-26B")
        )
        code, plan = run_plan(
            self.tmp, "--c27", "f2", *peers, "--site", "node-b", table=self.table
        )
        self.assertEqual(code, 0)
        row = plan["models"]["cand27"]
        self.assertEqual(row["model_path"], str(self.tmp / "pkg"))
        self.assertEqual(
            row["parity"]["stored"],
            str(self.made["formal"] / "output" / "typed-final.predictions.jsonl"),
        )
        self.assertEqual(
            row["cache"]["frozen"], str(self.made["formal"] / "triton-cache")
        )

    def test_node_b_paths_must_be_the_hashed_files(self) -> None:
        moved = self.tmp / "B27" / "other" / "calibration.json"
        moved.parent.mkdir(parents=True)
        shutil.copy(self.made["soup"] / "package" / "calibration.json", moved)
        receipt = self.make_receipt("moved.json", "--calibration", str(moved))
        self.assertEqual(self.fill("--revision", "e" * 40, receipt=receipt), 2)
        self.assertEqual(
            self.fill("--revision", "e" * 40, "--adopt-paths", receipt=receipt), 0
        )
        block = self.block()
        self.assertEqual(
            block["stage_node_a"][1]["from"], "${B27}/other/calibration.json"
        )
        self.assertEqual(
            block["node_b"]["extra.calibration"], "${B27}/other/calibration.json"
        )
        self.assertEqual(
            block["node_b"]["mounts"], ["${M}/Qwen3.8-27B", "${B27}/other"]
        )

    def test_include_and_refusals(self) -> None:
        self.assertEqual(self.fill("--revision", "e" * 40, "--include", "bad"), 2)
        staging = ["--repo", "llm-semantic-router/dev2-27b-staging"]
        code = self.fill(
            "--revision", "e" * 40, *staging, "--include", "m3/M3-S-soup/package/*"
        )
        self.assertEqual(code, 0)
        block = self.block()
        step = block["stage_node_a"][0]
        self.assertEqual(
            (step["repo"], step["include"]),
            ("llm-semantic-router/dev2-27b-staging", "m3/M3-S-soup/package/*"),
        )
        self.assertEqual(
            block["package"]["dir"], "${X}/f2/release/DEV2.0-26B/m3/M3-S-soup/package"
        )
        self.assertEqual(block["model_path"], block["package"]["dir"])
        failed = json.loads(self.receipt.read_text())
        failed.update(passed=False, problems=["x"])
        path = self.tmp / "failed.json"
        path.write_text(json.dumps(failed))
        self.assertEqual(self.fill("--revision", "e" * 40, receipt=path), 2)
        partial = self.tmp / "partial.json"
        with contextlib.redirect_stdout(io.StringIO()):
            args = receipt_args(self.made, partial)
            args.remove("--cache")
            args.remove(str(self.made["formal"] / "triton-cache"))
            self.assertEqual(event3.main(args), 0)
        self.assertEqual(self.fill("--revision", "e" * 40, receipt=partial), 2)


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
