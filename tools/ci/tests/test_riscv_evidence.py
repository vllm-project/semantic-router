from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools/ci"))
import riscv_evidence as riscv  # noqa: E402
from check_ci_gate import evaluate_gate  # noqa: E402
from ci_plan import make_plan  # noqa: E402
from ci_results import make_receipt  # noqa: E402

SHA = "a" * 40


def go_events(names):
    rows = [{"Action": "start", "Package": "candle-binding"}]
    for name in names:
        rows.extend(
            {"Action": action, "Package": "candle-binding", "Test": name}
            for action in ("run", "pass")
        )
    return [*rows, {"Action": "pass", "Package": "candle-binding"}]


def write_events(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def router_report():
    bodies = {
        "health": "healthy",
        "ready": "ready",
        "classify": {"classification": {"category": "biology"}},
        "preview": {
            "metrics": {
                "domain": {
                    "confidence_available": True,
                    "confidence": 0.92,
                    "execution_time_ms": 10.3,
                }
            }
        },
    }
    return {
        "source_sha": SHA,
        "responses": [
            {"path": name, "http_status": 200, "body": body}
            for name, body in bodies.items()
        ],
    }


class RISCVTests(unittest.TestCase):
    def test_go_requires_each_discovered_test_and_complete_process(self):
        names = {"TestOwnedA", "TestUtilityFunctions"}
        rows = go_events(sorted(names))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "go.jsonl"
            write_events(path, rows)
            self.assertEqual(len(riscv.go_cases(path, names, "target")), 2)
            for candidate in (
                rows[:-1],
                rows[3:],
                rows + rows,
                [],
                [
                    (
                        {**row, "Action": "skip"}
                        if row.get("Test") == "TestOwnedA" and row["Action"] == "pass"
                        else row
                    )
                    for row in rows
                ],
                [
                    (
                        {**row, "Action": "fail"}
                        if "Test" not in row and row["Action"] == "pass"
                        else row
                    )
                    for row in rows
                ],
            ):
                write_events(path, candidate)
                with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                    riscv.go_cases(path, names, "target")

    def test_go_keeps_parallel_subtests_and_rejects_orphans(self):
        rows = [
            {"Test": "TestOwnedA", "Action": "run"},
            {"Test": "TestOwnedA/sub", "Action": "run"},
            {"Test": "TestOwnedA/sub", "Action": "pass"},
            {"Test": "TestOwnedA", "Action": "pass"},
            {"Action": "pass"},
        ]
        rows = [{"Package": "binding", **row} for row in rows]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "go.jsonl"
            write_events(path, rows)
            self.assertEqual(len(riscv.go_cases(path, {"TestOwnedA"}, "target")), 2)
            write_events(path, rows[1:])
            with self.assertRaises(ValueError):
                riscv.go_cases(path, {"TestOwnedA"}, "target")

    def test_capability_results_must_run_and_pass(self):
        names = {
            "TestEmbeddingCapabilitiesConformance",
            "TestEmbeddingDimensionStateValidation",
        }
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "go.jsonl"
            for name in names:
                for action in ("missing", "skip", "fail"):
                    rows = go_events(sorted(names))
                    if action == "missing":
                        rows = [row for row in rows if row.get("Test") != name]
                    else:
                        for row in rows:
                            if row.get("Test") == name and row["Action"] == "pass":
                                row["Action"] = action
                    write_events(path, rows)
                    with (
                        self.subTest(name=name, action=action),
                        self.assertRaises(ValueError),
                    ):
                        riscv.go_cases(path, names, "qemu-binding")

    def test_rust_requires_both_real_cases_and_no_ignored_tests(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            (directory / "rust-required.txt").write_text(
                "\n".join(riscv.RUST_TESTS) + "\n"
            )
            (directory / "rust-list.txt").write_text(
                "\n".join(name + ": test" for name in riscv.RUST_TESTS)
            )
            for index, name in enumerate(riscv.RUST_TESTS):
                (directory / f"rust-{index}.log").write_text(
                    f"running 1 test\ntest {name} ... output\n ok\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 5 filtered out;\n"
                )
            self.assertEqual(len(riscv.rust_cases(directory)), 2)
            path = directory / "rust-0.log"
            path.write_text(
                path.read_text().replace(
                    "1 passed; 0 failed; 0 ignored", "0 passed; 0 failed; 1 ignored"
                )
            )
            with self.assertRaises(ValueError):
                riscv.rust_cases(directory)

    def test_elf_must_be_actual_riscv_target(self):
        with tempfile.TemporaryDirectory(dir=ROOT / ".agent-harness") as folder:
            path = Path(folder) / "binary"
            header = bytearray(64)
            header[:6] = b"\x7fELF\x02\x01"
            header[18:20] = (243).to_bytes(2, "little")
            path.write_bytes(header)
            record = riscv.target_binary(path)
            self.assertEqual(record["platform"], "linux/riscv64")
            self.assertEqual(len(record["sha256"]), 64)
            header[18:20] = (62).to_bytes(2, "little")
            path.write_bytes(header)
            with self.assertRaises(ValueError):
                riscv.target_binary(path)

    def test_router_receipt_requires_actual_nonplaceholder_inference(self):
        report = router_report()
        self.assertEqual(len(riscv.router_cases(report, SHA)), 4)
        for mutation in (
            "source",
            "placeholder",
            "failed",
            "missing",
            "fallback",
            "unscored",
            "nan",
            "zero-latency",
        ):
            candidate = copy.deepcopy(report)
            if mutation == "source":
                candidate["source_sha"] = "b" * 40
            elif mutation == "placeholder":
                candidate["responses"][2]["body"][
                    "routing_decision"
                ] = "placeholder_response"
            elif mutation == "failed":
                candidate["responses"][-1]["http_status"] = 500
            elif mutation == "missing":
                candidate["responses"].pop()
            elif mutation == "fallback":
                candidate["responses"][2]["body"] = {
                    "classification": {
                        "category": "default-route",
                        "confidence_available": False,
                    },
                    "routing_decision": "default-route",
                    "signal_errors": {"domain:general": "model_inference_failed"},
                }
            elif mutation == "unscored":
                candidate["responses"][3]["body"]["metrics"]["domain"][
                    "confidence_available"
                ] = False
            elif mutation == "nan":
                candidate["responses"][3]["body"]["metrics"]["domain"]["confidence"] = (
                    float("nan")
                )
            else:
                candidate["responses"][3]["body"]["metrics"]["domain"][
                    "execution_time_ms"
                ] = 0
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                riscv.router_cases(candidate, SHA)

    def test_complete_adapter_consumes_every_framework_report(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            model = {
                "name": "Domain",
                "repo_id": "llm-semantic-router/Vela-1.0-Encoder-307M-Domain",
                "revision": SHA,
                "path": "/models/candle/" + SHA,
            }
            (directory / "models.json").write_text(
                json.dumps({"provider": "candle", "models": [model]})
            )
            capabilities = {
                "TestEmbeddingCapabilitiesConformance",
                "TestEmbeddingDimensionStateValidation",
            }
            minimal = {
                "TestOwnedA",
                "TestNewRegexProvider",
                "TestUtilityFunctions",
            } | capabilities
            (directory / "minimal-pattern.txt").write_text(
                "^Test(Owned.*|NewRegexProvider|RegexProvider_.*|UtilityFunctions|"
                "EmbeddingCapabilitiesConformance|EmbeddingDimensionStateValidation)$"
            )
            excluded = "TestOwnedNativeMaintainedHallucinationWithoutLabelMetadata"
            (directory / "minimal-skip.txt").write_text("^" + excluded + "$")
            (directory / "binding-list.txt").write_text(
                "\n".join(sorted(minimal | {excluded}))
            )
            for filename, names in (
                ("host-parity.jsonl", {"TestCandleClassifierParity"}),
                ("qemu-ffi.jsonl", {"TestNativeClassifierFFIIsLinked"}),
                ("qemu-parity.jsonl", {"TestCandleClassifierParity"}),
                ("qemu-minimal.jsonl", minimal),
            ):
                write_events(directory / filename, go_events(sorted(names)))
            (directory / "rust-required.txt").write_text("\n".join(riscv.RUST_TESTS))
            (directory / "rust-list.txt").write_text(
                "\n".join(name + ": test" for name in riscv.RUST_TESTS)
            )
            for index, name in enumerate(riscv.RUST_TESTS):
                (directory / f"rust-{index}.log").write_text(
                    f"running 1 test\ntest {name} ... ok\ntest result: ok. 1 passed; 0 failed; 0 ignored;\n"
                )
            (directory / "router.json").write_text(json.dumps(router_report()))
            for missing in capabilities:
                (directory / "binding-list.txt").write_text(
                    "\n".join(sorted((minimal | {excluded}) - {missing}))
                )
                with (
                    self.subTest(missing=missing),
                    self.assertRaisesRegex(
                        ValueError, "binding discovery is incomplete"
                    ),
                ):
                    riscv.evidence(directory)
            (directory / "binding-list.txt").write_text(
                "\n".join(sorted(minimal | {excluded}))
            )
            with (
                patch.object(
                    riscv.subprocess,
                    "check_output",
                    side_effect=[SHA + "\n", "qemu-riscv64 version 8.2.2\n"],
                ),
                patch.object(riscv, "actual_platform", return_value="linux/amd64"),
                patch.object(
                    riscv.shutil, "which", return_value="/usr/bin/qemu-riscv64-static"
                ),
                patch.object(
                    riscv,
                    "target_binary",
                    return_value={"platform": "linux/riscv64", "sha256": "b" * 64},
                ) as binary,
            ):
                result = riscv.evidence(directory)
            self.assertEqual(len(result["cases"]), 14)
            self.assertEqual(len(result["expected_cases"]), 14)
            self.assertTrue(
                {"qemu-binding:" + name for name in capabilities}
                <= set(result["expected_cases"])
            )
            self.assertEqual(binary.call_count, 3)
            self.assertEqual(result["models"], [model])
            self.assertEqual(
                result["execution"],
                {"mode": "qemu-user", "host_platform": "linux/amd64"},
            )

    def test_emulated_receipt_cannot_claim_native_host_or_shared_libraries(self):
        plan = make_plan([], source_sha=SHA, requested=("native.candle-riscv64-qemu",))
        verification = plan["verifications"][0]
        evidence = {
            "runtime": "candle",
            "device": "cpu",
            "platform": "linux/riscv64",
            "execution": verification["execution"],
            "cases": [{"id": "actual", "status": "passed"}],
            "expected_cases": ["actual"],
        }
        receipt = make_receipt(
            verification,
            evidence,
            source_sha=SHA,
            execution_platform="linux/amd64",
            environ={},
        )
        self.assertTrue(evaluate_gate(plan, [receipt]).passed)
        for mutation in ("mode", "host", "target", "artifact"):
            candidate = copy.deepcopy(receipt)
            if mutation == "mode":
                candidate.pop("execution")
            elif mutation == "host":
                candidate["execution"]["host_platform"] = "linux/arm64"
            elif mutation == "target":
                candidate["platform"] = "linux/amd64"
            else:
                candidate["artifacts"] = [{"id": "native:cpu", "sha256": "b" * 64}]
            with self.subTest(mutation=mutation):
                self.assertFalse(evaluate_gate(plan, [candidate]).passed)
        for producer in ("linux/riscv64", "linux/arm64"):
            with self.assertRaises(ValueError):
                make_receipt(
                    verification,
                    evidence,
                    source_sha=SHA,
                    execution_platform=producer,
                    environ={},
                )
        for field, value in (("execution", {}), ("platform", "linux/amd64")):
            with self.assertRaises(ValueError):
                make_receipt(
                    verification,
                    {**evidence, field: value},
                    source_sha=SHA,
                    execution_platform="linux/amd64",
                    environ={},
                )


if __name__ == "__main__":
    unittest.main()
