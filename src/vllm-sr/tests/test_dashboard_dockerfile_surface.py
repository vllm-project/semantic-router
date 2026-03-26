from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DASHBOARD_DOCKERFILE = REPO_ROOT / "dashboard" / "backend" / "Dockerfile"


def test_dashboard_dockerfile_retries_backend_builder_apk_installs() -> None:
    content = DASHBOARD_DOCKERFILE.read_text(encoding="utf-8")

    assert "apk_add_with_retry build-base" in content


def test_dashboard_dockerfile_retries_runtime_apk_installs() -> None:
    content = DASHBOARD_DOCKERFILE.read_text(encoding="utf-8")

    assert (
        "apk_add_with_retry ca-certificates curl docker-cli py3-pip python3 su-exec wget"
        in content
    )
