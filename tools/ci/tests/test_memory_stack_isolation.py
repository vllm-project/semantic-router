import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
MEMORY_SCRIPT = REPO_ROOT / "e2e/testing/run_memory_integration.sh"
MILVUS_MAKE = REPO_ROOT / "tools/make/milvus.mk"


class MemoryStackIsolationTests(unittest.TestCase):
    def _run_memory(self, stack="ci-memory-a", offset=1000, **overrides):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            binary_dir = root / "bin"
            binary_dir.mkdir()
            calls = root / "calls.jsonl"
            fixture = root / "fixture.mk"
            fixture.write_text("LOG_TARGET = :\n")
            dispatcher = (
                f"#!{sys.executable}\n"
                + r"""
import json
import os
import signal
import sys
import time
from pathlib import Path

name = Path(sys.argv[0]).name
args = sys.argv[1:]
record = {"command": name, "args": args, "cwd": os.getcwd(),
          "stack": os.environ.get("VLLM_SR_STACK_NAME"),
          "offset": os.environ.get("VLLM_SR_PORT_OFFSET"),
          "milvus": os.environ.get("MILVUS_ADDRESS"),
          "router": os.environ.get("ROUTER_ENDPOINT")}
with open(os.environ["MEMORY_CALLS"], "a") as stream:
    stream.write(json.dumps(record) + "\n")
if name == "fake-runtime":
    if args[:1] == ["inspect"]:
        sys.exit(0 if args[1] == os.environ.get("MEMORY_EXISTING_CONTAINER") else 1)
    if args[:2] == ["network", "inspect"]:
        sys.exit(0 if args[2] == os.environ.get("MEMORY_EXISTING_NETWORK") else 1)
    if args[:1] == ["run"] and "--name" in args:
        container = args[args.index("--name") + 1]
        if container.endswith("-milvus") and os.environ.get("MEMORY_FAIL_MILVUS"):
            sys.exit(51)
        print(container)
    if args[:1] == ["logs"]:
        print(json.dumps({"component": "memory", "event": "milvus_store_initialized",
                          "collection_name": "memory_test_model_identity"}))
    sys.exit(0)
if name == "curl":
    if "-w" in args:
        print("200")
    sys.exit(0)
if name == "python3":
    if args[:2] == ["-m", "pip"] or (args[:1] == ["-c"] and "pymilvus" in args[1]):
        sys.exit(0)
    if args[:1] == ["09-memory-features-test.py"]:
        sys.exit(37 if os.environ.get("MEMORY_FAIL_SUITE") else 0)
    os.execv(os.environ["MEMORY_REAL_PYTHON"], [os.environ["MEMORY_REAL_PYTHON"], *args])
if name == "make":
    if args[:1] == ["-C"]:
        args = args[2:]
    os.execv(os.environ["MEMORY_REAL_MAKE"], [os.environ["MEMORY_REAL_MAKE"], "--no-print-directory",
             "-f", os.environ["MEMORY_MILVUS_MAKE"], "-f", os.environ["MEMORY_MAKE_FIXTURE"], *args])
if name == "vllm-sr":
    if args[:1] == ["serve"]:
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
        while True:
            time.sleep(.01)
    sys.exit(0)
raise SystemExit("unhandled fixture command: " + name)
"""
            )
            for name in ("fake-runtime", "curl", "python3", "make", "vllm-sr"):
                executable = binary_dir / name
                executable.write_text(dispatcher)
                executable.chmod(0o755)
            state = root / "state"
            environment = dict(os.environ)
            environment.update(
                PATH=str(binary_dir) + os.pathsep + environment.get("PATH", ""),
                PYTHONPATH=str(REPO_ROOT / "src/vllm-sr"),
                CONTAINER_RUNTIME="fake-runtime",
                VLLM_SR_STACK_NAME=stack,
                VLLM_SR_PORT_OFFSET=str(offset),
                MEMORY_TEST_DIR=str(state),
                MEMORY_TEST_ARTIFACT_DIR=str(root / "artifacts"),
                MEMORY_TEST_MODEL_DIR=str(root / "model-cache"),
                KEEP_MEMORY_TEST_DIR="1",
                MEMORY_CALLS=str(calls),
                MEMORY_REAL_PYTHON=sys.executable,
                MEMORY_REAL_MAKE=shutil.which("make"),
                MEMORY_MILVUS_MAKE=str(MILVUS_MAKE),
                MEMORY_MAKE_FIXTURE=str(fixture),
            )
            environment.update(overrides)
            result = subprocess.run(
                ["bash", str(MEMORY_SCRIPT)],
                cwd=root,
                env=environment,
                text=True,
                capture_output=True,
                timeout=30,
                check=False,
            )
            invocations = [json.loads(line) for line in calls.read_text().splitlines()]
            config = (
                (state / "config.yaml").read_text()
                if (state / "config.yaml").exists()
                else ""
            )
            artifacts = [
                str(path.relative_to(root / "artifacts"))
                for path in (root / "artifacts").rglob("*.log")
            ]
            return result, invocations, config, artifacts, str(state)

    def _assert_owned_cleanup(self, calls, stack, state):
        stopped = []
        for call in calls:
            if call["command"] == "fake-runtime" and call["args"][0] in {
                "stop",
                "rm",
                "logs",
            }:
                resource = call["args"][1]
                self.assertTrue(resource.startswith(stack + "-"), call)
                stopped.append(resource)
            if call["command"] == "vllm-sr":
                self.assertEqual(call["stack"], stack)
                self.assertEqual(call["cwd"], state)
        self.assertIn(stack + "-llm-katan", stopped)
        self.assertIn(stack + "-vllm-sr-milvus", stopped)

    def test_two_stack_names_have_independent_resources_ports_and_artifacts(self):
        for stack, offset in (("ci-memory-a", 1000), ("ci-memory-b", 2000)):
            with self.subTest(stack=stack):
                result, calls, config, artifacts, state = self._run_memory(
                    stack, offset
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                runs = [
                    call["args"]
                    for call in calls
                    if call["command"] == "fake-runtime" and call["args"][0] == "run"
                ]
                self.assertEqual(len(runs), 2)
                milvus, katan = runs
                self.assertIn(stack + "-vllm-sr-milvus", milvus)
                self.assertIn(f"127.0.0.1:{19530 + offset}:19530", milvus)
                self.assertIn(f"127.0.0.1:{9091 + offset}:9091", milvus)
                self.assertIn(state + "/milvus-data:/var/lib/milvus:z", milvus)
                self.assertIn(stack + "-llm-katan", katan)
                self.assertIn(stack + "-vllm-sr-network", katan)
                self.assertIn(f"127.0.0.1:{8000 + offset}:8000", katan)
                self.assertIn(stack + "-llm-katan:8000", config)
                self.assertIn(stack + "-vllm-sr-milvus:19530", config)
                suite = next(
                    call
                    for call in calls
                    if call["args"] == ["09-memory-features-test.py"]
                )
                self.assertEqual(suite["milvus"], f"localhost:{19530 + offset}")
                self.assertEqual(suite["router"], f"http://localhost:{8888 + offset}")
                self.assertTrue(artifacts)
                self.assertTrue(
                    all(path.startswith("memory-" + stack + "/") for path in artifacts)
                )
                self._assert_owned_cleanup(calls, stack, state)

    def test_failed_suite_keeps_exit_status_and_cleans_only_its_stack(self):
        result, calls, _, artifacts, state = self._run_memory(MEMORY_FAIL_SUITE="1")
        self.assertEqual(result.returncode, 37, result.stdout + result.stderr)
        self.assertTrue(artifacts)
        self._assert_owned_cleanup(calls, "ci-memory-a", state)
        self.assertTrue(
            any(
                call["command"] == "vllm-sr" and call["args"] == ["stop"]
                for call in calls
            )
        )

    def test_preexisting_container_is_never_adopted_or_stopped(self):
        result, calls, _, artifacts, _ = self._run_memory(
            MEMORY_EXISTING_CONTAINER="ci-memory-a-vllm-sr-router-container"
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Refusing to reuse", result.stderr)
        self.assertFalse(artifacts)
        self.assertFalse(any(call["command"] in {"make", "vllm-sr"} for call in calls))
        self.assertTrue(
            all(
                call["args"][0] == "inspect"
                for call in calls
                if call["command"] == "fake-runtime"
            )
        )

    def test_preexisting_network_is_never_adopted_or_removed(self):
        result, calls, _, artifacts, _ = self._run_memory(
            MEMORY_EXISTING_NETWORK="ci-memory-a-vllm-sr-data-network"
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Refusing to reuse", result.stderr)
        self.assertFalse(artifacts)
        self.assertFalse(any(call["command"] in {"make", "vllm-sr"} for call in calls))
        self.assertTrue(
            all(
                "inspect" in call["args"]
                for call in calls
                if call["command"] == "fake-runtime"
            )
        )

    def test_failed_milvus_start_does_not_stop_a_router_that_never_started(self):
        result, calls, _, _, _ = self._run_memory(MEMORY_FAIL_MILVUS="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(any(call["command"] == "vllm-sr" for call in calls))
        removed = [
            call["args"][1]
            for call in calls
            if call["command"] == "fake-runtime" and call["args"][0] in {"stop", "rm"}
        ]
        self.assertEqual(removed, ["ci-memory-a-vllm-sr-milvus"] * 2)

    def test_invalid_offset_fails_before_container_mutation(self):
        result, calls, _, _, _ = self._run_memory(offset=50000)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(any(call["command"] == "fake-runtime" for call in calls))

    def test_milvus_make_preserves_manual_defaults(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory) / "fixture.mk"
            fixture.write_text("LOG_TARGET = :\nCONTAINER_RUNTIME = recorded-runtime\n")
            result = subprocess.run(
                [
                    "make",
                    "--no-print-directory",
                    "-n",
                    "-f",
                    str(MILVUS_MAKE),
                    "-f",
                    str(fixture),
                    "start-milvus",
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=10,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            for expected in (
                '--name "milvus-semantic-cache"',
                '"19530:19530"',
                '"9091:9091"',
                '"/tmp/milvus-data:/var/lib/milvus:z"',
            ):
                self.assertIn(expected, result.stdout)


if __name__ == "__main__":
    unittest.main()
