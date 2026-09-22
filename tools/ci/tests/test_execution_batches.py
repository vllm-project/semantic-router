"""Batched execution preserves source-bound contracts and actual dependencies."""

from __future__ import annotations

import copy
import os
import socket
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tools/ci"))
from ci_plan import digest, make_plan  # noqa: E402
from execution_batches import e2e_batches, validate_execution_batch  # noqa: E402
from run_native_batch import run_batch, run_command  # noqa: E402


class ExecutionBatchTests(unittest.TestCase):
    def test_completed_native_command_cannot_leave_a_listener_for_the_next_contract(
        self,
    ):
        child = (
            "import pathlib,socket,sys,time; "
            "s=socket.socket(); s.bind(('127.0.0.1',0)); s.listen(); "
            "pathlib.Path(sys.argv[1]).write_text(str(s.getsockname()[1])); time.sleep(60)"
        )
        parent = (
            "import pathlib,subprocess,sys,time\n"
            "subprocess.Popen([sys.executable, '-c', sys.argv[1], sys.argv[2]])\n"
            "deadline=time.monotonic()+5\n"
            "while not pathlib.Path(sys.argv[2]).exists() and time.monotonic()<deadline: time.sleep(.01)\n"
            "sys.exit(int(sys.argv[3]))\n"
        )
        for code in (0, 1):
            with tempfile.TemporaryDirectory() as directory:
                port_file = Path(directory) / "port"
                result = run_command(
                    [sys.executable, "-c", parent, child, str(port_file), str(code)],
                    cwd=ROOT,
                    env=os.environ.copy(),
                    check=False,
                    timeout=10,
                )
                self.assertEqual(result.returncode, code)
                port = int(port_file.read_text())
                deadline = time.monotonic() + 2
                while True:
                    try:
                        connection = socket.create_connection(
                            ("127.0.0.1", port), timeout=0.1
                        )
                    except OSError:
                        break
                    connection.close()
                    if time.monotonic() >= deadline:
                        self.fail(
                            "completed native contract left a live descendant listener"
                        )
                    time.sleep(0.01)

    def test_metadata_stages_use_the_same_contract_deadline(self):
        batch = make_plan([], source_sha="a" * 40, requested=("native.ort-cpu",))[
            "native_batches"
        ][0]
        budgets = []

        def process(_command, **kwargs):
            budgets.append(kwargs["timeout"])
            return SimpleNamespace(returncode=0)

        with tempfile.TemporaryDirectory() as directory, patch(
            "run_native_batch.time.monotonic", side_effect=[0, 10, 20, 30]
        ):
            self.assertTrue(run_batch(batch, Path(directory), run=process))
        self.assertEqual(budgets, [7190, 7180, 7170])

    def test_native_reuses_runtime_worker_without_merging_receipts(self):
        plan = make_plan(
            [],
            source_sha="a" * 40,
            requested=(
                "native.ort-cpu",
                "native.image-calibration-cpu",
                "native.candle-riscv64-qemu",
            ),
        )
        self.assertEqual(len(plan["native_batches"]), 2)
        for batch in plan["native_batches"]:
            validate_execution_batch(batch, "native")
        ort = next(row for row in plan["native_batches"] if row["runtime"] == "ort")
        self.assertEqual(
            {row["category"] for row in ort["verifications"]},
            {"runtime", "conformance"},
        )
        self.assertEqual(ort["timeout_minutes"], 240)
        qemu = next(row for row in plan["native_batches"] if row["runtime"] == "candle")
        self.assertEqual(qemu["dispatch_job"], "native-independent")
        self.assertFalse(qemu["native"])

    def test_e2e_shards_preserve_all_profiles_once_with_bounded_cost(self):
        plan = make_plan([], source_sha="a" * 40, full=True)
        records = [row for row in plan["verifications"] if row["executor"] == "e2e"]
        self.assertLess(len(plan["e2e_batches"]), len(records))
        seen = []
        for batch in plan["e2e_batches"]:
            validate_execution_batch(batch, "e2e")
            self.assertLessEqual(batch["timeout_minutes"] + 30, 360)
            self.assertLessEqual(
                len(batch["verifications"]),
                2 if batch["resource_class"] == "model" else 3,
            )
            for record in batch["verifications"]:
                self.assertEqual(sorted(record["images"]), batch["images"])
                self.assertEqual(record["runtime"], batch["runtime"])
                seen.append(record["id"])
        self.assertEqual(sorted(seen), sorted(row["id"] for row in records))
        self.assertTrue(
            {"e2e.vela-omni", "e2e.vela-halu", "e2e.multimodal-routing"} <= set(seen)
        )

    def test_large_resource_class_batches_profiles_instead_of_naming_models(self):
        plan = make_plan([], source_sha="a" * 40, requested=("e2e.vela-omni",))
        record = plan["verifications"][0]
        records = []
        for i in range(3):
            candidate = {**record, "id": f"e2e.model-{i}", "profile": f"model-{i}"}
            candidate["contract_sha256"] = digest(
                {
                    key: value
                    for key, value in candidate.items()
                    if key != "contract_sha256"
                }
            )
            records.append(candidate)
        batches = e2e_batches(records)
        self.assertEqual([len(row["verifications"]) for row in batches], [2, 1])
        self.assertTrue(all("model-" not in row["display_name"] for row in batches))

    def test_tampered_duplicate_or_mixed_worker_contracts_are_rejected(self):
        plan = make_plan(
            [],
            source_sha="a" * 40,
            requested=("native.ort-cpu", "native.image-calibration-cpu"),
        )
        original = plan["native_batches"][0]
        variants = []
        changed = copy.deepcopy(original)
        changed["images"] = ["extproc"]
        variants.append(changed)
        changed = copy.deepcopy(original)
        changed["verifications"][1] = changed["verifications"][0]
        variants.append(changed)
        changed = copy.deepcopy(original)
        changed["verifications"][0]["target"] = "other"
        variants.append(changed)
        changed = copy.deepcopy(original)
        changed["timeout_minutes"] = 1
        variants.append(changed)
        for batch in variants:
            with self.assertRaises(ValueError):
                validate_execution_batch(batch, "native")

    def test_native_failure_does_not_hide_later_contract_and_reports_stay_isolated(
        self,
    ):
        batch = make_plan(
            [],
            source_sha="a" * 40,
            requested=("native.ort-cpu", "native.image-calibration-cpu"),
        )["native_batches"][0]
        calls = []

        def process(command, **kwargs):
            calls.append((command, kwargs["env"]))
            if command[:2] == ["make", "verify-image-routing-calibration"]:
                return SimpleNamespace(returncode=1)
            if "--verification" in command:
                destination = Path(command[command.index("--output") + 1])
                destination.write_text("{}\n")
            return SimpleNamespace(returncode=0)

        with tempfile.TemporaryDirectory() as directory:
            self.assertFalse(run_batch(batch, Path(directory), run=process))
            results = list((Path(directory) / "results").glob("*.json"))
            self.assertEqual([path.stem for path in results], ["native.ort-cpu"])
            make_envs = [env for command, env in calls if command[0] == "make"]
            self.assertEqual(len(make_envs), 2)
            self.assertNotEqual(
                make_envs[0]["MODEL_TEST_REPORT_DIR"],
                make_envs[1]["MODEL_TEST_REPORT_DIR"],
            )
            self.assertEqual(
                make_envs[0]["MODEL_TEST_MODELS_DIR"],
                make_envs[1]["MODEL_TEST_MODELS_DIR"],
            )


if __name__ == "__main__":
    unittest.main()
