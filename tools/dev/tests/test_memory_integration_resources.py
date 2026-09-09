"""Run the memory harness against fake services to verify its runtime inputs."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "e2e/testing/run_memory_integration.sh"
MODEL_NAME = "mmbert-embed-32k-2d-matryoshka"

# Every external service command is intercepted. Config generation still runs
# the script's actual Python, and model downloads create a small local fixture.
FAKE_COMMAND = r"""
import json
import os
import signal
import sys
import time
from pathlib import Path

name = Path(sys.argv[0]).name
args = sys.argv[1:]
if name == "python3":
    if args[:2] == ["-m", "pip"] or (args[0] == "-c" and "from pymilvus" in args[1]):
        pass
    elif args[0] == "-" and args[1].startswith("llm-semantic-router/"):
        destination = Path(args[2])
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "config.json").write_text('{"fixture": true}')
    elif args == ["09-memory-features-test.py"]:
        state = Path(os.environ["VLLM_SR_STATE_ROOT_DIR"])
        config = Path(os.environ["MEMORY_TEST_DIR"]) / "config.yaml"
        model = state / "models" / "mmbert-embed-32k-2d-matryoshka" / "config.json"
        Path(os.environ["FIXTURE_RESULT"]).write_text(json.dumps({
            "config": config.read_text(),
            "collection": os.environ["MILVUS_COLLECTION"],
            "model_exists_at_mount": model.is_file(),
            "model_path": str(model.resolve()),
            "milvus_address": os.environ["MILVUS_ADDRESS"],
        }))
        raise SystemExit(int(os.environ.get("FIXTURE_TEST_EXIT", "0")))
    else:
        os.execv(os.environ["FIXTURE_PYTHON"], [os.environ["FIXTURE_PYTHON"], *args])
elif name == "runtime":
    if args[:2] == ["network", "inspect"] and "--format" not in args:
        raise SystemExit(1)
    if "--format" in args:
        print("true", os.environ["VLLM_SR_STACK_NAME"], os.environ["VLLM_SR_RUN_ID"])
elif name == "curl":
    if "-w" in args:
        print("200", end="")
elif name == "vllm-sr" and args[0] == "serve":
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    while True:
        time.sleep(0.1)
"""


class MemoryRuntimeInputsTests(unittest.TestCase):
    def test_isolated_models_and_collection_reach_router_and_assertions(self):
        for shared_cache in (False, True):
            with (
                self.subTest(shared_cache=shared_cache),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                bin_dir = root / "bin"
                bin_dir.mkdir()
                for name in ("python3", "runtime", "make", "curl", "vllm-sr"):
                    command = bin_dir / name
                    command.write_text(f"#!{sys.executable}\n{FAKE_COMMAND}")
                    command.chmod(0o755)
                test_dir = root / "test"
                test_dir.mkdir()
                output_dir = root / "output"
                output_dir.mkdir()
                state = output_dir / "state"
                result_path = root / "result.json"
                environment = {
                    **os.environ,
                    "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
                    "CONTAINER_RUNTIME": str(bin_dir / "runtime"),
                    "VLLM_SR_TEST_ISOLATED": "1",
                    "VLLM_SR_STACK_NAME": "memory-fixture",
                    "VLLM_SR_RUN_ID": "isolated-fixture",
                    "VLLM_SR_PORT_OFFSET": "1200",
                    "VLLM_SR_STATE_ROOT_DIR": str(state),
                    "MEMORY_TEST_DIR": str(test_dir),
                    "MEMORY_OUTPUT_DIR": str(output_dir),
                    "KEEP_MEMORY_TEST_DIR": "1",
                    "USE_DETERMINISTIC_MEMORY_EMBEDDINGS": "0",
                    "FIXTURE_PYTHON": sys.executable,
                    "FIXTURE_RESULT": str(result_path),
                    # Also ensure the wrapper propagates an assertion failure.
                    "FIXTURE_TEST_EXIT": "7" if shared_cache else "0",
                }
                environment.pop("MEMORY_TEST_MODEL_DIR", None)
                model_root = test_dir / "models"
                if shared_cache:
                    model_root = root / "shared model cache"
                    environment["MEMORY_TEST_MODEL_DIR"] = str(model_root)
                result = subprocess.run(
                    ["bash", str(SCRIPT)],
                    cwd=REPO_ROOT,
                    env=environment,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )
                self.assertEqual(
                    result.returncode,
                    int(environment["FIXTURE_TEST_EXIT"]),
                    result.stdout + result.stderr,
                )
                inputs = json.loads(result_path.read_text())
                self.assertEqual(inputs["collection"], "memory_test_ci")
                self.assertIn(f"collection: {inputs['collection']}", inputs["config"])
                self.assertIn("memory-fixture-memory-milvus:19530", inputs["config"])
                self.assertEqual(inputs["milvus_address"], "localhost:20730")
                self.assertTrue(inputs["model_exists_at_mount"])
                self.assertEqual(
                    Path(inputs["model_path"]),
                    (model_root / MODEL_NAME / "config.json").resolve(),
                )


if __name__ == "__main__":
    unittest.main()
