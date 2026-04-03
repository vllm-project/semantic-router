from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DOCKER_MK_PATH = REPO_ROOT / "tools" / "make" / "docker.mk"
AGENT_MK_PATH = REPO_ROOT / "tools" / "make" / "agent.mk"
ENVIRONMENTS_DOC_PATH = REPO_ROOT / "docs" / "agent" / "environments.md"


def test_split_topology_defaults_to_rebuilding_router_image() -> None:
    content = DOCKER_MK_PATH.read_text(encoding="utf-8")

    assert "SKIP_ROUTER_IMAGE_DEFAULT := 0" in content
    assert "SKIP_ROUTER_IMAGE_SOURCE := $(origin SKIP_ROUTER_IMAGE)" in content
    assert "router Docker image" in content
    assert "router compatibility Docker image" not in content
    assert 'if [ "$(SKIP_ROUTER_IMAGE_EFFECTIVE)" = "1" ]; then \\' in content


def test_agent_help_hides_legacy_topology_override() -> None:
    content = AGENT_MK_PATH.read_text(encoding="utf-8")

    assert "VLLM_SR_TOPOLOGY=legacy" not in content
    assert "compatibility fallback" not in content


def test_environment_docs_explain_default_split_without_user_topology_flags() -> None:
    content = ENVIRONMENTS_DOC_PATH.read_text(encoding="utf-8")

    assert (
        "Local runtime defaults to the split router/envoy/dashboard topology" in content
    )
    assert (
        "Split local runtime uses the local `vllm-sr` router image directly by default"
        in content
    )
    assert "VLLM_SR_TOPOLOGY=legacy" not in content
