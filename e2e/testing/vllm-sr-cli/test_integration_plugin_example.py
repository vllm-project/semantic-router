#!/usr/bin/env python3
"""The plugin guide's "Try it", run as written: install, serve, ask.

The test reads its commands, keyword package and request from the guide
(`website/docs/model-runtime/plugins.md`). Its `vllm-srun` cases need the
model runtime in the current Python environment (`make model-runtime-install`,
whose test extra brings the example's build backend). pip installs the example
plugin from the repository into a private directory on PYTHONPATH instead of
the environment, so no other test sees its entry points; discovery still reads
the entry points of the installed distribution. The container case builds the
guide's image, the router image (`VLLM_SR_IMAGE`) with the example installed,
and runs the guide's explicit worker container.
"""

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from urllib.error import URLError

import tomllib
from runtime_http import (
    HTTP_OK,
    READY_TIMEOUT_SECONDS,
    ServeProcess,
    call,
    container_runtime,
    free_port,
    page_requests,
    router_image,
)

REPO = Path(__file__).resolve().parents[3]
GUIDE = REPO / "website/docs/model-runtime/plugins.md"
EXAMPLE = REPO / "src/model-runtime/examples/third_party_plugin"
BASH_BLOCK = re.compile(r"^```bash\n(.*?)^```", re.M | re.S)
HEREDOC = re.compile(r"cat > (\S+) <<'(\w+)'\n(.*?)\n\2$", re.M | re.S)
OUTCOME = re.compile(r"returns `(\w+)` for the first text and `(\w+)` for the second")
PLUGIN_IMAGE = "vllm-sr-plugin-example:integration"
BUILD_TIMEOUT_SECONDS = 900


def _try_it() -> tuple[list[str], dict[str, str], list[str]]:
    """The guide's pip install arguments, written files and serve arguments."""
    for block in BASH_BLOCK.findall(GUIDE.read_text(encoding="utf-8")):
        if "vllm-srun serve" not in block:
            continue
        files = {path: body + "\n" for path, _, body in HEREDOC.findall(block)}
        lines = HEREDOC.sub("", block).splitlines()
        (install,) = [line for line in lines if line.startswith("pip install ")]
        (serve,) = [line for line in lines if line.startswith("vllm-srun serve ")]
        return shlex.split(install)[2:], files, shlex.split(serve)[2:]
    raise AssertionError(f"{GUIDE} has no block that serves the example")


def _serve_commands(command: str = "vllm-srun serve") -> list[list[str]]:
    """The arguments of every line in the guide that runs ``command``."""
    return [
        shlex.split(line)[len(command.split()) :]
        for block in BASH_BLOCK.findall(GUIDE.read_text(encoding="utf-8"))
        for line in HEREDOC.sub("", block).splitlines()
        if line.startswith(command + " ")
    ]


def _option(arguments: list[str], name: str) -> str:
    return arguments[arguments.index(name) + 1]


def _without_port(arguments: list[str]) -> list[str]:
    at = arguments.index("--port")
    return arguments[:at] + arguments[at + 2 :]


def _with_option(arguments: list[str], name: str, value: str) -> list[str]:
    at = arguments.index(name)
    return [*arguments[: at + 1], value, *arguments[at + 2 :]]


class TestPluginExample(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(tempfile.mkdtemp(prefix="vllm-sr-plugin-"))
        cls.addClassCleanup(shutil.rmtree, cls.root, ignore_errors=True)
        install, files, cls.serve_arguments = _try_it()
        cls.package = Path(cls.serve_arguments[0])

        # The guide installs the runtime and the example from the repository
        # root; the runtime is already installed, so only the example goes in.
        if REPO / install[-1] != EXAMPLE:
            raise AssertionError(f"the guide installs {install}, not the example")
        source = cls.root / "example"
        shutil.copytree(
            EXAMPLE,
            source,
            ignore=shutil.ignore_patterns("build", "*.egg-info", "__pycache__"),
        )
        cls.site = cls.root / "site"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-deps",
                "--no-build-isolation",
                "--target",
                str(cls.site),
                str(source),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        cls.local = cls.root / "keywords"
        for path, body in files.items():
            written = cls.local / Path(path).relative_to(cls.package)
            written.parent.mkdir(parents=True, exist_ok=True)
            written.write_text(body, encoding="utf-8")
        cls.env = {"PYTHONPATH": str(cls.site)}

    def _serve(
        self,
        serve_arguments: list[str] | None = None,
        command: str = "vllm-srun serve",
    ) -> ServeProcess:
        serve_arguments = serve_arguments or self.serve_arguments
        arguments = [str(self.local), *_without_port(serve_arguments[1:])]
        runtime = ServeProcess(
            [*command.split(), *arguments],
            self.root / f"serve-{time.monotonic_ns()}.log",
            env={**self.env, "VLLM_SR_ENGINE_CACHE_DIR": str(self.root / "cache")},
        )
        self.addCleanup(runtime.stop)
        runtime.wait_ready()
        return runtime

    def _build_plugin_image(self) -> str:
        """The guide's image: the router image with the example installed."""
        (build,) = _serve_commands("docker build")
        if REPO / build[-1] != EXAMPLE:
            raise AssertionError(f"the guide builds {build}, not the example")
        runtime = container_runtime()
        subprocess.run(
            [
                runtime,
                "build",
                "-t",
                PLUGIN_IMAGE,
                "--build-arg",
                f"IMAGE={router_image()}",
                str(self.root / "example"),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=BUILD_TIMEOUT_SECONDS,
        )
        self.addCleanup(
            subprocess.run,
            [runtime, "rmi", "-f", PLUGIN_IMAGE],
            capture_output=True,
            check=False,
        )
        return PLUGIN_IMAGE

    def test_the_guides_request_gets_the_guides_answer(self):
        runtime = self._serve()
        prose = " ".join(GUIDE.read_text(encoding="utf-8").split())
        expected = list(OUTCOME.search(prose).groups())

        status, response = call(
            runtime.base, "/v1/classify", page_requests(GUIDE)["/v1/classify"]
        )

        self.assertEqual(status, HTTP_OK, response)
        self.assertEqual([result["label"] for result in response["results"]], expected)

    def test_the_example_serves_on_its_own_accelerator_and_profile(self):
        (arguments,) = [
            arguments
            for arguments in _serve_commands("vllm-srun serve")
            if "--profile" in arguments
        ]
        self._assert_serves_on_its_own_accelerator_and_profile(
            "vllm-srun serve", arguments, "--profile"
        )

    def test_the_guides_container_serves_the_installed_plugin(self):
        (arguments,) = _serve_commands("docker run")
        port = free_port()
        container_port = _option(arguments, "--port")
        name = f"vllm-sr-plugin-example-{port}"
        arguments = _with_option(arguments, "-p", f"127.0.0.1:{port}:{container_port}")
        _, target, mode = _option(arguments, "-v").rsplit(":", 2)
        arguments = _with_option(arguments, "-v", f"{self.local}:{target}:{mode}")
        # Preserve the guide's worker command and options, isolating the image,
        # host port, mount source and container name for concurrent suites.
        image_index = arguments.index("--entrypoint") + 2
        arguments[image_index] = self._build_plugin_image()
        runtime = container_runtime()
        self.addCleanup(
            subprocess.run,
            [runtime, "rm", "-f", name],
            capture_output=True,
            check=False,
        )
        subprocess.run(
            [runtime, "run", "-d", "--name", name, *arguments],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )
        base = f"http://127.0.0.1:{port}"
        deadline = time.monotonic() + READY_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            try:
                status, _ = call(base, "/health")
                if status == HTTP_OK:
                    break
            except (URLError, ConnectionError):
                pass
            time.sleep(0.25)
        else:
            logs = subprocess.run(
                [runtime, "logs", name], capture_output=True, text=True, check=False
            )
            self.fail(
                f"plugin container did not become ready: {logs.stdout}{logs.stderr}"
            )
        self._assert_runtime_answers(base, arguments, "--profile")

    def _assert_serves_on_its_own_accelerator_and_profile(
        self, command: str, arguments: list[str], profile_option: str
    ):
        runtime = self._serve(arguments, command)
        self._assert_runtime_answers(runtime.base, arguments, profile_option)

    def _assert_runtime_answers(
        self, base: str, arguments: list[str], profile_option: str
    ):
        prose = " ".join(GUIDE.read_text(encoding="utf-8").split())
        expected = list(OUTCOME.search(prose).groups())

        status, models = call(base, "/v1/models")
        self.assertEqual(status, HTTP_OK, models)
        (card,) = models["data"]
        self.assertEqual(
            (card["accelerator"], card["device"], card["profile"]),
            (
                _option(arguments, "--device"),
                _option(arguments, "--device"),
                _option(arguments, profile_option),
            ),
        )
        status, response = call(
            base, "/v1/classify", page_requests(GUIDE)["/v1/classify"]
        )
        self.assertEqual(status, HTTP_OK, response)
        self.assertEqual([result["label"] for result in response["results"]], expected)

    def test_the_installed_distribution_is_what_the_runtime_lists(self):
        project = tomllib.loads((EXAMPLE / "pyproject.toml").read_text())["project"]
        declared = {
            name for group in project["entry-points"].values() for name in group
        }
        runtime = self._serve()

        listed = subprocess.run(
            ["vllm-srun", "plugins"],
            check=True,
            capture_output=True,
            text=True,
            env={**os.environ, **self.env},
        )
        status, models = call(runtime.base, "/v1/models")

        plugins = json.loads(listed.stdout)
        self.assertIn("example_keywords", plugins["families"])
        self.assertIn("example_counts", plugins["engines"])
        self.assertEqual(status, HTTP_OK)
        (card,) = models["data"]
        served = {
            plugin["name"]: (plugin["distribution"], plugin["version"])
            for plugin in card["plugins"]
            if plugin["name"] in declared
        }
        self.assertEqual(
            served,
            dict.fromkeys(declared, (project["name"], project["version"])),
        )


if __name__ == "__main__":
    unittest.main()
