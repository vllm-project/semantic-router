from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
INSTALL_SCRIPT_PATH = REPO_ROOT / "install.sh"
INSTALL_DOC_PATH = REPO_ROOT / "website" / "docs" / "installation" / "installation.md"


def test_install_script_only_mentions_docker_runtime() -> None:
    content = INSTALL_SCRIPT_PATH.read_text(encoding="utf-8")

    assert "podman" not in content.lower()
    assert "--runtime auto|docker|skip" in content
    assert "Linux auto -> docker" in content


def test_installation_doc_only_mentions_docker_runtime() -> None:
    content = INSTALL_DOC_PATH.read_text(encoding="utf-8")

    assert "Podman" not in content
    assert "Docker" in content
