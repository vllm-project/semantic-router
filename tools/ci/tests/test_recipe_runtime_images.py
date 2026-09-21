"""Exercise recipe image preparation through the real shared Make targets."""

import ast
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PROCESS_STUB = r"""
import json, os, pathlib, sys
args = sys.argv[1:]
name = pathlib.Path(sys.argv[0]).name
calls = pathlib.Path(os.environ['IMAGE_CONTRACT_CALLS'])
prepared = calls.with_suffix('.prepared')
if name == 'docker':
    with calls.open('a') as stream:
        stream.write(json.dumps({'docker': args}) + '\n')
    if args[:2] == ['image', 'inspect']:
        sys.exit(0 if args[2] == os.environ['VLLM_SR_ROUTER_IMAGE'] else 1)
    assert args[0] == 'pull', args
    if os.environ.get('IMAGE_CONTRACT_PULL_FAILURE'):
        sys.exit(37)
    prepared.write_text(args[1])
elif name == 'python3':
    assert args[-3:] == ['sources', '--format', 'pipe'], args
    print('default|config/recipes|default|balance,knowledge')
elif name == 'bash':
    assert args == ['e2e/testing/run_recipe_conformance.sh'], args
    selected = os.environ['VLLM_SR_ENVOY_IMAGE']
    assert selected == prepared.read_text()
    assert selected != os.environ['ROUTER_IMAGE']
    with calls.open('a') as stream:
        stream.write(json.dumps({'serve': {
            'router': os.environ['ROUTER_IMAGE'],
            'envoy': selected,
            'recipes': os.environ['RECIPES'],
        }}) + '\n')
else:
    raise AssertionError(name)
"""


class RecipeRuntimeImageTests(unittest.TestCase):
    def test_both_entrypoints_prepare_and_forward_default_or_explicit_envoy(self):
        for target in (
            "recipe-conformance-live-cpu",
            "recipe-conformance-live-cpu-all",
        ):
            for override in (None, "example.test/envoy:pinned"):
                with self.subTest(target=target, override=override):
                    self.run_target(target, override=override)

    def test_unavailable_envoy_prevents_recipe_execution(self):
        for target in (
            "recipe-conformance-live-cpu",
            "recipe-conformance-live-cpu-all",
        ):
            with self.subTest(target=target):
                self.run_target(target, pull_failure=True)

    def run_target(self, target, *, override=None, pull_failure=False):
        constants = ast.parse((ROOT / "src/vllm-sr/cli/consts.py").read_text())
        default_envoy = next(
            ast.literal_eval(node.value)
            for node in constants.body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(item, ast.Name)
                and item.id == "VLLM_SR_ENVOY_CONTAINER_IMAGE_DEFAULT"
                for item in node.targets
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            binaries = root / "bin"
            binaries.mkdir()
            for name in ("docker", "python3", "bash"):
                executable = binaries / name
                executable.write_text(f"#!{sys.executable}\n" + PROCESS_STUB)
                executable.chmod(0o755)
            (root / "Makefile").write_text(
                f"include {ROOT}/tools/make/docker.mk\n"
                f"include {ROOT}/tools/make/recipe-conformance.mk\n"
            )
            environment = {
                key: value
                for key, value in os.environ.items()
                if key not in {"MAKEFLAGS", "MFLAGS", "MAKEOVERRIDES", "MAKEFILES"}
                and not key.startswith("VLLM_SR_")
                and not key.startswith("RECIPE_CONFORMANCE_")
            }
            router = "semantic-router-ci/vllm-sr:" + "a" * 40
            environment.update(
                PATH=str(binaries) + os.pathsep + environment.get("PATH", ""),
                VLLM_SR_IMAGE=router,
                VLLM_SR_ROUTER_IMAGE=router,
                PREBUILT_RUNTIME_IMAGES="1",
                IMAGE_CONTRACT_CALLS=str(root / "calls.jsonl"),
            )
            if pull_failure:
                environment["IMAGE_CONTRACT_PULL_FAILURE"] = "1"
            arguments = [
                "make",
                "--no-print-directory",
                target,
                "SHELL=/bin/sh",
                "CONTAINER_RUNTIME=docker",
                "RECIPE_CONFORMANCE_PYTHON=python3",
                "RECIPE_CONFORMANCE_RECIPES=balance,knowledge",
            ]
            if override:
                arguments.append(f"VLLM_SR_ENVOY_IMAGE={override}")
            result = subprocess.run(
                arguments,
                cwd=root,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
                timeout=15,
            )
            calls = [
                json.loads(line)
                for line in (root / "calls.jsonl").read_text().splitlines()
            ]
            envoy = override or default_envoy
            expected = [
                {"docker": ["image", "inspect", router]},
                {"docker": ["image", "inspect", envoy]},
                {"docker": ["pull", envoy]},
            ]
            if pull_failure:
                self.assertNotEqual(result.returncode, 0)
            else:
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                expected.append(
                    {
                        "serve": {
                            "router": router,
                            "envoy": envoy,
                            "recipes": "balance,knowledge",
                        }
                    }
                )
            self.assertEqual(calls, expected)


if __name__ == "__main__":
    unittest.main()
