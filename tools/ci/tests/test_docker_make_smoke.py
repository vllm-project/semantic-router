import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


class DockerMakeSmokeTests(unittest.TestCase):
    def run_target(self, target, *, mode="healthy", **variables):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calls = root / "calls.jsonl"
            runtime = root / "runtime"
            runtime.write_text(
                f"#!{sys.executable}\n"
                + """
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
with open(os.environ['SMOKE_CALLS'], 'a') as log:
    log.write(json.dumps(args) + '\\n')
mode = os.environ['SMOKE_MODE']
if args[0] == 'run': print('owned-smoke-container')
if args[0] == 'inspect': print('false' if mode == 'exited' else 'true')
if args[0] == 'exec':
    if args[-1].endswith('/health') and mode == 'exited': sys.exit(1)
    if args[-1].endswith('/v1/models') and mode == 'models-failed': sys.exit(22)
"""
            )
            runtime.chmod(0o700)
            result = subprocess.run(
                [
                    "make",
                    "--no-print-directory",
                    "-f",
                    str(ROOT / "tools/make/docker.mk"),
                    target,
                    f"CONTAINER_RUNTIME={runtime}",
                    "LOG_TARGET=:",
                    "PREBUILT_RUNTIME_IMAGES=1",
                    *[f"{key}={value}" for key, value in variables.items()],
                ],
                cwd=ROOT,
                env={**os.environ, "SMOKE_CALLS": str(calls), "SMOKE_MODE": mode},
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            return result, [json.loads(line) for line in calls.read_text().splitlines()]

    def test_build_and_push_use_the_selected_runtime_image(self):
        for target, action in (
            ("docker-build-vllm-sr-router", "build"),
            ("docker-push-vllm-sr-router", "push"),
        ):
            for overrides, image in (
                (
                    {"DOCKER_REGISTRY": "example.test/sr", "DOCKER_TAG": "v0.4-test"},
                    "example.test/sr/vllm-sr:v0.4-test",
                ),
                ({"VLLM_SR_ROUTER_IMAGE": "local/custom:check"}, "local/custom:check"),
            ):
                with self.subTest(target=target, image=image):
                    result, calls = self.run_target(target, **overrides)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(calls[0][0], action)
                    self.assertIn(image, calls[0])

    def test_smoke_owns_start_readiness_models_and_cleanup(self):
        for mode, success in (
            ("healthy", True),
            ("exited", False),
            ("models-failed", False),
        ):
            with self.subTest(mode=mode):
                result, calls = self.run_target(
                    "docker-test-llm-katan",
                    mode=mode,
                    LLM_KATAN_IMAGE="local/katan:check",
                )
                self.assertEqual(
                    result.returncode == 0, success, result.stdout + result.stderr
                )
                self.assertEqual(calls[0], ["image", "inspect", "local/katan:check"])
                launch = next(call for call in calls if call[0] == "run")
                self.assertIn("echo", launch)
                self.assertIn("none", launch)
                self.assertNotIn("-p", launch)
                self.assertTrue(
                    any(
                        call[0] == "exec" and call[-1].endswith("/health")
                        for call in calls
                    )
                )
                self.assertEqual(calls[-1], ["rm", "--force", "owned-smoke-container"])
                self.assertEqual(sum(call[0] == "rm" for call in calls), 1)


if __name__ == "__main__":
    unittest.main()
