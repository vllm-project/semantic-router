from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
INSTALL_SCRIPT_PATH = REPO_ROOT / "install.sh"
INSTALL_DOC_PATH = REPO_ROOT / "website" / "docs" / "installation" / "installation.md"
PYPI_PUBLISH_WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "pypi-publish.yml"
ROOT_MAKEFILE_PATH = REPO_ROOT / "Makefile"
RELEASE_MAKEFILE_PATH = REPO_ROOT / "tools" / "make" / "release.mk"
OPENCLAW_SKILL_PATH = (
    REPO_ROOT
    / "dashboard"
    / "backend"
    / "skillpacks"
    / "openclaw-vsr-bridge"
    / "SKILL.md"
)
OPENCLAW_INSTALL_DOC_PATH = (
    REPO_ROOT / "website" / "static" / "install" / "agent" / "openclaw-vsr-bridge.md"
)


def test_install_script_only_mentions_docker_runtime() -> None:
    content = INSTALL_SCRIPT_PATH.read_text(encoding="utf-8")

    assert "podman" not in content.lower()
    assert "--runtime auto|docker|skip" in content
    assert "Linux auto -> docker" in content


def test_installation_doc_only_mentions_docker_runtime() -> None:
    content = INSTALL_DOC_PATH.read_text(encoding="utf-8")

    assert "Podman" not in content
    assert "Docker" in content


def test_install_script_defaults_to_dev_channel() -> None:
    content = INSTALL_SCRIPT_PATH.read_text(encoding="utf-8")

    assert 'REQUESTED_CHANNEL="${VLLM_SR_INSTALL_CHANNEL:-dev}"' in content
    assert "--channel stable|dev" in content
    assert "resolve_latest_dev_version" in content
    assert '"vllm-sr==$dev_version"' in content
    assert "resolves and pins the newest" in content


def test_installation_doc_recommends_development_package() -> None:
    content = INSTALL_DOC_PATH.read_text(encoding="utf-8")

    assert "bash -s -- --channel dev" in content
    assert (
        'python -m pip install --upgrade "vllm-sr==${VLLM_SR_DEV_VERSION}"' in content
    )
    assert "newest published `.dev` package" in content


def test_pypi_publish_workflow_does_not_push_back_to_main() -> None:
    content = PYPI_PUBLISH_WORKFLOW_PATH.read_text(encoding="utf-8")

    assert "Bump next development base version on main" not in content
    assert "git push origin HEAD:main" not in content


def test_make_release_target_is_available_from_repo_root() -> None:
    root_makefile = ROOT_MAKEFILE_PATH.read_text(encoding="utf-8")
    release_makefile = RELEASE_MAKEFILE_PATH.read_text(encoding="utf-8")

    assert "tools/make/release.mk" in root_makefile
    assert "release:" in release_makefile
    assert (
        'src/vllm-sr/scripts/release.sh "$(RELEASE_VERSION)" "$(NEXT_VERSION)"'
        in release_makefile
    )


def test_openclaw_install_docs_use_the_validate_config_option() -> None:
    for path in (OPENCLAW_SKILL_PATH, OPENCLAW_INSTALL_DOC_PATH):
        content = path.read_text(encoding="utf-8")

        assert "vllm-sr validate --config config.yaml" in content
        assert "vllm-sr validate config.yaml" not in content
