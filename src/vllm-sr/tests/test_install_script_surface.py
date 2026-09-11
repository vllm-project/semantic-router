from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
INSTALL_SCRIPT_PATH = REPO_ROOT / "install.sh"
INSTALL_DOC_PATH = REPO_ROOT / "website" / "docs" / "installation" / "installation.md"
INSTALL_DATA_PATH = REPO_ROOT / "website" / "src" / "data" / "installation.ts"
HOMEPAGE_INSTALL_PATH = (
    REPO_ROOT
    / "website"
    / "src"
    / "components"
    / "InstallQuickStartSection"
    / "index.tsx"
)
VLLM_SR_AGENT_SKILL_PATH = (
    REPO_ROOT / "website" / "static" / "install" / "agent" / "vllm-sr" / "SKILL.md"
)
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


def test_install_script_runtime_contract_supports_podman_fallback() -> None:
    content = INSTALL_SCRIPT_PATH.read_text(encoding="utf-8")

    # User-facing --runtime choices are unchanged: Podman is an internal
    # fallback during auto detection, not a first-class option.
    assert "--runtime auto|docker|skip" in content

    # Auto detection must prefer Docker but fall back to Podman when Docker
    # is not reachable. The fallback has to be gated on --runtime auto so
    # explicit --runtime docker/skip paths are unaffected.
    assert "podman_ready" in content
    assert 'REQUESTED_RUNTIME" = "auto" ] && podman_ready' in content

    # Linux auto still resolves to Docker first.
    assert "Linux auto -> docker" in content


def test_install_script_persists_selected_runtime() -> None:
    content = INSTALL_SCRIPT_PATH.read_text(encoding="utf-8")

    # The selected runtime is written to runtime.env so later CLI sessions
    # reuse it instead of re-probing the host.
    assert "runtime.env" in content
    assert "CONTAINER_RUNTIME=" in content


def test_installation_doc_documents_runtime_options() -> None:
    content = INSTALL_DOC_PATH.read_text(encoding="utf-8")

    assert "Docker" in content
    # Podman is now documented as a fallback when Docker is absent.
    assert "Podman" in content


def test_install_script_defaults_to_dev_channel() -> None:
    content = INSTALL_SCRIPT_PATH.read_text(encoding="utf-8")

    assert 'REQUESTED_CHANNEL="${VLLM_SR_INSTALL_CHANNEL:-dev}"' in content
    assert "--channel stable|dev" in content
    assert "resolve_latest_dev_version" in content
    assert '"vllm-sr==$dev_version"' in content
    assert "resolves and pins the newest" in content


def test_installation_surfaces_offer_minimal_human_and_agent_paths() -> None:
    docs = INSTALL_DOC_PATH.read_text(encoding="utf-8")
    data = INSTALL_DATA_PATH.read_text(encoding="utf-8")
    homepage = HOMEPAGE_INSTALL_PATH.read_text(encoding="utf-8")
    skill = VLLM_SR_AGENT_SKILL_PATH.read_text(encoding="utf-8")

    for method in ("curl", "pip", "uv", "Agent"):
        assert f"label: '{method}'" in docs

    assert "pip index versions" not in docs
    assert "VLLM_SR_DEV_VERSION" not in docs
    assert "awk" not in docs
    assert "python -m pip install --upgrade vllm-sr" in data
    assert "uv tool install vllm-sr" in data
    assert "--channel stable" in data

    assert "For humans" in homepage
    assert "For agents" in homepage
    assert "AGENT_INSTALL_PROMPT" in homepage
    assert "AGENT_SKILL_PATH" in homepage

    assert "name: vllm-sr" in skill
    assert "vllm-sr config schema" in skill
    assert "vllm-sr config init" in skill
    assert "vllm-sr config validate --config config.yaml" in skill
    assert "vllm-sr config plan --config config.yaml" in skill
    assert "vllm-sr route preview" in skill
    assert "vllm-sr route probe" in skill
    assert "Dashboard is optional" in skill


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

        assert "vllm-sr config validate --config config.yaml" in content
        assert "vllm-sr config validate config.yaml" not in content
