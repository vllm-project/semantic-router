"""The Decision image must build its wheel from staged catalog resources."""

from pathlib import Path


def test_image_stages_built_in_catalog_before_wheel_build() -> None:
    dockerfile = (
        Path(__file__).resolve().parents[1]
        / "decision_runtime"
        / "image"
        / "Dockerfile"
    ).read_text()
    stage = '"${PYTHON_BIN}" /source/tools/release/stage_model_catalog_package.py'
    wheel = '"${PYTHON_BIN}" -m pip wheel'
    assert dockerfile.count(stage) == 1
    assert dockerfile.index(stage) < dockerfile.index(wheel)


def test_image_context_excludes_preexisting_generated_catalog() -> None:
    dockerignore = (
        Path(__file__).resolve().parents[1]
        / "decision_runtime"
        / "image"
        / "Dockerfile.dockerignore"
    ).read_text()
    assert "src/vllm-sr/cli/model_assets/**" in dockerignore
    assert "!src/vllm-sr/cli/model_assets/__init__.py" in dockerignore
