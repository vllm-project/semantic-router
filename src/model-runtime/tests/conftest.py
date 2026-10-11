from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from vllm_srun.config import ModelConfig, ServeConfig
from vllm_srun.runtime import Runtime
from vllm_srun.testing.fixtures import write_package


@pytest.fixture(scope="session")
def fixture_root(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("packages")


@pytest.fixture(scope="session")
def qwen3_package(fixture_root) -> Path:
    return write_package(fixture_root / "qwen3", backbone="qwen3", seed=1)


@pytest.fixture(scope="session")
def qwen35_package(fixture_root) -> Path:
    return write_package(
        fixture_root / "qwen3_5", backbone="qwen3_5", seed=2, score_bias=False
    )


@pytest.fixture(scope="session")
def adapter_package(fixture_root) -> Path:
    return write_package(
        fixture_root / "lora", backbone="qwen3_5", seed=3, adapter=True
    )


@pytest.fixture()
def package_copy(tmp_path, qwen3_package) -> Path:
    target = tmp_path / "copy"
    shutil.copytree(qwen3_package, target)
    return target


def start_runtime(package: Path, **overrides) -> Runtime:
    runtime = Runtime(
        ServeConfig(
            models=(ModelConfig(model=str(package), device="cpu"),), **overrides
        )
    )
    runtime.start(background=False)
    return runtime


@pytest.fixture(scope="session")
def qwen3_runtime(qwen3_package):
    runtime = start_runtime(qwen3_package)
    yield runtime
    runtime.stop()


@pytest.fixture(scope="session")
def qwen35_runtime(qwen35_package):
    runtime = start_runtime(qwen35_package)
    yield runtime
    runtime.stop()


QUESTIONS = {
    "domain": {
        "type": "choice",
        "instructions": "Which domain is this request about?",
        "criteria": {"code": "Programming", "math": "Mathematics", "other": None},
    },
    "reasoning": {
        "type": "noul",
        "instructions": "Does this need multi-step reasoning?",
    },
    "difficulty": {
        "type": "score",
        "instructions": "How difficult is it?",
        "criteria": ["Trivial", "Moderate", "Hard"],
    },
}
STATE = "Write a Python function that merges two sorted lists."


@pytest.fixture(scope="session")
def d3_package(fixture_root) -> Path:
    from vllm_srun.testing.decision3 import write_fixture

    return write_fixture(fixture_root / "d3-vision", "vision", seed=11)


@pytest.fixture(scope="session")
def d3_text_package(fixture_root) -> Path:
    from vllm_srun.testing.decision3 import write_fixture

    return write_fixture(fixture_root / "d3-text", "text", seed=12)


@pytest.fixture(scope="session")
def d3_runtime(d3_package):
    runtime = start_runtime(d3_package)
    yield runtime
    runtime.stop()


@pytest.fixture(scope="session")
def png_url():
    """``png_url(width, height, seed)``: a data URL of a deterministic RGB noise-and-gradient PNG."""
    import base64
    import io

    import numpy as np
    from PIL import Image

    def make(width: int, height: int, seed: int = 0, fmt: str = "PNG") -> str:
        rng = np.random.default_rng(seed)
        pixels = rng.integers(0, 128, size=(height, width, 3), dtype=np.uint8)
        pixels += np.linspace(0, 127, width, dtype=np.uint8)[None, :, None]
        buffer = io.BytesIO()
        Image.fromarray(pixels, "RGB").save(buffer, format=fmt)
        return (
            f"data:image/{fmt.lower()};base64,"
            + base64.b64encode(buffer.getvalue()).decode()
        )

    return make
