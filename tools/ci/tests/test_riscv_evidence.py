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


def router_report():
    bodies = {
        "health": "healthy",
        "ready": "ready",
        "classify": {"classification": {"category": "biology"}},
    }
    return {
        "source_sha": SHA,
        "runtime": {"results": [{"index": 0, "label": "biology"}]},
        "responses": [
            {"path": name, "http_status": 200, "body": body}
            for name, body in bodies.items()
        ],
    }


class RISCVTests(unittest.TestCase):
    def test_elf_must_be_actual_riscv_target(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            path = Path(folder) / "binary"
            header = bytearray(64)
            header[:6] = b"\x7fELF\x02\x01"
            header[18:20] = (243).to_bytes(2, "little")
            path.write_bytes(header)
            record = riscv.target_binary(path)
            self.assertEqual(record["path"], str(path.relative_to(ROOT)))
            self.assertEqual(record["platform"], "linux/riscv64")
            self.assertEqual(len(record["sha256"]), 64)
            header[18:20] = (62).to_bytes(2, "little")
            path.write_bytes(header)
            with self.assertRaises(ValueError):
                riscv.target_binary(path)

    def test_router_classification_must_be_the_runtime_answer(self):
        report = router_report()
        self.assertEqual(len(riscv.router_cases(report, SHA)), 3)
        for mutation in (
            "source",
            "placeholder",
            "failed",
            "missing",
            "fallback",
            "disagrees",
            "no-runtime",
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
                    "classification": {"category": "default-route"},
                    "routing_decision": "default-route",
                    "signal_errors": {"domain:biology": "model_inference_failed"},
                }
            elif mutation == "disagrees":
                candidate["runtime"]["results"][0]["label"] = "other"
            else:
                candidate.pop("runtime")
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                riscv.router_cases(candidate, SHA)

    def test_adapter_reports_the_emulated_router_and_its_binary(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            (directory / "router.json").write_text(json.dumps(router_report()))
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
            self.assertEqual(
                result["expected_cases"],
                ["router:health", "router:ready", "router:classify"],
            )
            binary.assert_called_once_with(ROOT / riscv.ROUTER_BINARY)
            self.assertEqual(result["runtime"], "model-runtime")
            self.assertEqual(
                result["execution"],
                {"mode": "qemu-user", "host_platform": "linux/amd64"},
            )

    def test_emulated_receipt_cannot_claim_native_host_or_artifacts(self):
        plan = make_plan(
            [], source_sha=SHA, requested=("platform.router-riscv64-qemu",)
        )
        verification = plan["verifications"][0]
        evidence = {
            "runtime": "model-runtime",
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
                candidate["artifacts"] = [{"id": "image:extproc", "sha256": "b" * 64}]
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
