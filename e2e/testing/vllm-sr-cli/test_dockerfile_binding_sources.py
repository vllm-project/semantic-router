"""Check that every Docker copy layer carries the binding's capability sources."""

import fnmatch
import shlex
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DOCKERFILES = (
    "src/vllm-sr/Dockerfile",
    "src/vllm-sr/Dockerfile.cuda",
    "src/vllm-sr/Dockerfile.rocm",
    "tools/docker/Dockerfile.extproc",
    "tools/docker/Dockerfile.extproc-rocm",
)
BINDINGS = ("candle-binding", "onnx-binding")


def assert_binding_capability_sources(dockerfile: str) -> None:
    content = (REPO_ROOT / dockerfile).read_text(encoding="utf-8")
    checked = set()
    for line in content.replace("\\\n", " ").splitlines():
        if not line.startswith("COPY "):
            continue
        args = [arg for arg in shlex.split(line)[1:] if not arg.startswith("--")]
        sources, destination = args[:-1], args[-1]
        patterns = [Path(source).name for source in sources]
        if not any(
            fnmatch.fnmatchcase("semantic-router.go", pattern) for pattern in patterns
        ):
            continue
        for binding in BINDINGS:
            if Path(destination).name != binding:
                continue
            required = {
                path.name
                for path in (REPO_ROOT / binding).glob("embedding_capabilities*.go")
                if not path.name.endswith("_test.go")
            }
            missing = {
                name
                for name in required
                if not any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)
            }
            assert not missing, f"{dockerfile}: {destination} misses {sorted(missing)}"
            checked.add(binding)
    # ONNX instance sources do not depend on the legacy root Go wrapper.
    expected = {"candle-binding"}
    assert checked == expected, f"{dockerfile}: binding copy stages not checked"


class TestDockerfileBindingSources(unittest.TestCase):
    def test_binding_capability_sources_reach_go_builder(self) -> None:
        for dockerfile in DOCKERFILES:
            with self.subTest(dockerfile=dockerfile):
                assert_binding_capability_sources(dockerfile)
