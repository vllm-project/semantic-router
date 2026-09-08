import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DOCKER_MK_PATH = REPO_ROOT / "tools" / "make" / "docker.mk"
AGENT_MK_PATH = REPO_ROOT / "tools" / "make" / "agent.mk"
ENVIRONMENTS_DOC_PATH = REPO_ROOT / "tools" / "agent" / "docs" / "environments.md"
MEMORY_INTEGRATION_PATH = REPO_ROOT / "e2e" / "testing" / "run_memory_integration.sh"


def _docker_mk_target(name: str) -> str:
    """Return one Make target's declaration lines plus its recipe body."""
    lines = DOCKER_MK_PATH.read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(f"{name}:"))
    end = start + 1
    while end < len(lines) and (
        not lines[end].strip()
        or lines[end].startswith("\t")
        or lines[end].startswith(f"{name}:")
    ):
        end += 1
    return "\n".join(lines[start:end])


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
        "Split Intelligent Routing for Mixture-of-Models uses the local `vllm-sr` router image directly by default"
        in content
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
    assert "8080 + VLLM_SR_PORT_OFFSET" in content
    assert "8888 + VLLM_SR_PORT_OFFSET" in content
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


def test_router_image_targets_use_the_canonical_image_name() -> None:
    content = DOCKER_MK_PATH.read_text(encoding="utf-8")

    # `vllm-sr-router` names the router container/role, not an image; a tag built
    # from DOCKER_REGISTRY here produced an image no consumer referenced.
    assert "$(DOCKER_REGISTRY)/vllm-sr-router:" not in content

    build = _docker_mk_target("docker-build-vllm-sr-router")
    assert "-t $(VLLM_SR_ROUTER_IMAGE)" in build
    assert "$(VLLM_SR_BUILD_ARGS)" in build

    push = _docker_mk_target("docker-push-vllm-sr-router")
    assert "push $(VLLM_SR_ROUTER_IMAGE)" in push


def test_docker_test_llm_katan_owns_its_container_lifecycle() -> None:
    content = DOCKER_MK_PATH.read_text(encoding="utf-8")
    target = _docker_mk_target("docker-test-llm-katan")

    assert "LLM_KATAN_HOST_PORT ?= 8000" in content
    assert "LLM_KATAN_TEST_CONTAINER ?= llm-katan-docker-test" in content
    assert "docker-test-llm-katan: docker-build-llm-katan" in target

    # Detached start under a dedicated name, so the check cannot collide with
    # the `llm-katan` container the memory integration stack owns.
    assert "run -d" in target
    assert "--name $(LLM_KATAN_TEST_CONTAINER)" in target
    # The image CMD downloads a real model; the test must not.
    assert "--model dummy" in target
    assert "--backend echo" in target
    # Readiness on /health must gate the /v1/models assertion.
    assert "/health" in target
    assert "/v1/models" in target
    assert "logs $(LLM_KATAN_TEST_CONTAINER)" in target

    # Bounded retry with a per-request ceiling, not an unbounded wait.
    assert re.search(r"-lt\s+\d+", target), "readiness loop has no numeric bound"
    assert "--max-time" in target

    handlers = re.findall(r"trap '([^']*)'", target)
    assert handlers, "no trap-based cleanup"
    assert any("stop $(LLM_KATAN_TEST_CONTAINER)" in h for h in handlers)
    assert any("rm $(LLM_KATAN_TEST_CONTAINER)" in h for h in handlers)
    # A signal handler must terminate the shell; otherwise an interrupted run
    # falls through and re-enters the readiness loop.
    assert any("exit" in h for h in handlers)

    assert "docker run" not in target


def test_docker_run_llm_katan_stays_foreground() -> None:
    target = _docker_mk_target("docker-run-llm-katan")

    assert "run --rm -p 8000:8000" in target
    assert " run -d " not in target
