from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
INSTALL_SCRIPT_PATH = REPO_ROOT / "install.sh"
INSTALL_DOC_PATH = REPO_ROOT / "website" / "docs" / "installation" / "installation.md"
AGENT_INSTALL_DOC_PATH = REPO_ROOT / "website" / "docs" / "installation" / "agent.md"
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


def test_install_script_runtime_contract_supports_podman_fallback() -> None:
    content = INSTALL_SCRIPT_PATH.read_text(encoding="utf-8")

    # Podman is now a first-class --runtime option alongside docker.
    assert "--runtime auto|docker|podman|skip" in content

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

    # A failed or empty runtime selection leaves an existing runtime.env untouched.
    assert 'rm -f "$INSTALL_ROOT/runtime.env"' not in content


def test_install_script_launcher_preserves_install_root() -> None:
    content = INSTALL_SCRIPT_PATH.read_text(encoding="utf-8")

    # A custom --install-root writes runtime.env under that root. The
    # generated launcher must export VLLM_SR_INSTALL_ROOT so later CLI
    # sessions resolve the persisted runtime.env next to this installation
    # instead of the default location (#3370).
    assert 'export VLLM_SR_INSTALL_ROOT="$INSTALL_ROOT"' in content


def test_installation_doc_documents_runtime_options() -> None:
    content = INSTALL_DOC_PATH.read_text(encoding="utf-8")

    assert "Docker" in content
    # Podman is now documented as a fallback when Docker is absent.
    assert "Podman" in content


def test_install_script_defaults_to_stable_channel() -> None:
    content = INSTALL_SCRIPT_PATH.read_text(encoding="utf-8")

    assert 'REQUESTED_CHANNEL="${VLLM_SR_INSTALL_CHANNEL:-stable}"' in content
    assert "--channel stable|dev" in content
    assert "resolve_latest_dev_version" in content
    assert '"vllm-sr==$dev_version"' in content
    assert "resolves and pins the newest" in content


def test_installation_surfaces_offer_minimal_human_and_agent_paths() -> None:
    docs = INSTALL_DOC_PATH.read_text(encoding="utf-8")
    agent_docs = AGENT_INSTALL_DOC_PATH.read_text(encoding="utf-8")
    normalized_agent_docs = " ".join(agent_docs.split())
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
    assert "AGENT_INSTALL_DOC_PATH" in homepage

    assert "AGENT_INSTALL_PROMPT" in agent_docs
    assert "AGENT_SKILL_PATH" in agent_docs
    assert "Dashboard and Playground checks are optional" in normalized_agent_docs
    assert "vllm-sr config validate" in agent_docs
    assert "vllm-sr config plan" in agent_docs
    assert "vllm-sr route preview" in agent_docs
    assert "vllm-sr route probe" in agent_docs

    assert "name: vllm-sr" in skill
    assert "--channel stable --mode cli --runtime skip --no-launch" in skill
    assert 'export PATH="$HOME/.local/bin:$PATH"' in skill


def test_agent_skill_installs_a_current_cli_and_verifies_a_routed_answer() -> None:
    skill = VLLM_SR_AGENT_SKILL_PATH.read_text(encoding="utf-8")
    references = VLLM_SR_AGENT_SKILL_PATH.parent / "references"
    agent_docs = " ".join(AGENT_INSTALL_DOC_PATH.read_text(encoding="utf-8").split())

    # A stable release that predates standalone mode falls back to the dev
    # channel, so neither channel is pinned.
    assert "grep -q -- '--gateway'" in skill
    assert "--channel dev --mode cli --runtime skip --no-launch" in skill
    assert "python3 -m ensurepip --version" in skill
    assert "--platform amd" in skill
    # A complete config, bound to loopback; bare serve waits for the Dashboard.
    assert "address: 127.0.0.1" in skill
    assert "Never run `vllm-sr serve` without a complete `--config`" in skill
    assert "vllm-sr serve --config config.yaml" in skill
    # Verification names its success criteria, and the probe caps its answer.
    for evidence in (
        "vllm-sr status",
        "x-vsr-selected-decision: code-route",
        "vllm-sr route preview",
        "--max-completion-tokens 256",
        '"device":"rocm:0"',
    ):
        assert evidence in skill
    assert "through Envoy" not in skill
    assert "through Envoy" not in (references / "route-verification.md").read_text(
        encoding="utf-8"
    )
    assert (references / "troubleshooting.md").is_file()
    assert "Release channel" in agent_docs
    assert "vllm-sr serve --config config.yaml" in agent_docs


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
