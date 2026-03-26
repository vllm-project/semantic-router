from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_router_runtime_dockerfile_installs_python_requirements() -> None:
    content = (PROJECT_ROOT / "Dockerfile.router").read_text(encoding="utf-8")

    assert "COPY src/vllm-sr/requirements.txt /app/requirements.txt" in content
    assert "pip install $PIP_EXTRA --no-cache-dir -r /app/requirements.txt" in content


def test_router_rocm_runtime_dockerfile_installs_python_requirements() -> None:
    content = (PROJECT_ROOT / "Dockerfile.router.rocm").read_text(encoding="utf-8")

    assert "COPY src/vllm-sr/requirements.txt /app/requirements.txt" in content
    assert (
        "python3 -m pip install $PIP_EXTRA --no-cache-dir -r /app/requirements.txt"
        in content
    )
