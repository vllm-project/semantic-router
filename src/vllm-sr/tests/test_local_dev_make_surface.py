from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DOCKER_MK_PATH = REPO_ROOT / "tools" / "make" / "docker.mk"
AGENT_MK_PATH = REPO_ROOT / "tools" / "make" / "agent.mk"
ENVIRONMENTS_DOC_PATH = REPO_ROOT / "tools" / "agent" / "docs" / "environments.md"
MEMORY_INTEGRATION_PATH = REPO_ROOT / "e2e" / "testing" / "run_memory_integration.sh"


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


def test_environment_docs_use_the_supported_local_image_serve_flow() -> None:
    content = ENVIRONMENTS_DOC_PATH.read_text(encoding="utf-8")

    assert "`make vllm-sr-dev`" in content
    assert (
        "`VLLM_SR_IMAGE=ghcr.io/vllm-project/semantic-router/vllm-sr:latest "
        "vllm-sr serve --image-pull-policy never`" in content
    )
    assert "make vllm-sr-dev VLLM_SR_PLATFORM=amd" in content
    assert (
        "`VLLM_SR_IMAGE=ghcr.io/vllm-project/semantic-router/vllm-sr-rocm:latest "
        "vllm-sr serve --image-pull-policy never --platform amd`" in content
    )
    assert (
        "`VLLM_SR_IMAGE=ghcr.io/vllm-project/semantic-router/vllm-sr-cuda:latest "
        "vllm-sr serve --platform nvidia --config <recipe> "
        "--image-pull-policy ifnotpresent`" in content
    )
    assert "VLLM_SR_TOPOLOGY=legacy" not in content


def test_memory_integration_uses_installed_agent_venv_cli() -> None:
    content = DOCKER_MK_PATH.read_text(encoding="utf-8")
    target = content.split("memory-test-integration:", 1)[1]

    assert "vllm-sr-install-cli" in target
    assert 'PATH="$(AGENT_VENV)/bin:$$PATH" \\' in target
    assert "run_memory_integration.sh" in target


def test_memory_integration_offsets_all_router_host_endpoints() -> None:
    content = MEMORY_INTEGRATION_PATH.read_text(encoding="utf-8")

    assert 'VLLM_SR_PORT_OFFSET="${VLLM_SR_PORT_OFFSET:-0}"' in content
    assert "from cli.runtime_stack import resolve_runtime_stack" in content
    assert "layout = resolve_runtime_stack()" in content
    assert (
        '"ROUTER_API_HEALTH_URL": f"http://localhost:{layout.api_port}/ready"'
        in content
    )
    assert "layout.host_port(8888, name='memory_listener_port')" in content
    assert 'ROUTER_ENDPOINT="${ROUTER_ENDPOINT}"' in content


def test_cli_integration_uses_an_isolated_runtime_stack() -> None:
    content = DOCKER_MK_PATH.read_text(encoding="utf-8")
    target = content.split("vllm-sr-test-integration:", 1)[1].split(
        "memory-test-integration:", 1
    )[0]

    assert (
        'VLLM_SR_STACK_NAME="$${VLLM_SR_STACK_NAME:-vllm-sr-cli-integration}"' in target
    )
    assert 'VLLM_SR_PORT_OFFSET="$${VLLM_SR_PORT_OFFSET:-4200}"' in target
