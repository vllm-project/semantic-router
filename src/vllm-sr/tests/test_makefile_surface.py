from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MAKEFILE_PATH = Path(__file__).resolve().parents[1] / "Makefile"
EXPECTED_ENVOY_IMAGE = "envoyproxy/envoy:v1.35.3"


def test_makefile_uses_current_cli_config_commands() -> None:
    content = MAKEFILE_PATH.read_text(encoding="utf-8")

    assert "config-router: install" in content
    assert "config-envoy: install" in content
    assert "python -m cli.main config router --config $(CONFIG_FILE)" in content
    assert "python -m cli.main config envoy --config $(CONFIG_FILE)" in content
    assert "python -m cli.main config validate --config $(CONFIG_FILE)" in content
    assert "docker-build-router:" in content
    assert "docker-build-envoy:" in content
    assert "docker-build-dashboard:" in content
    assert "VLLM_SR_ROUTER_IMAGE=$(FULL_ROUTER_IMAGE)" in content
    assert "VLLM_SR_ENVOY_IMAGE=$(FULL_ENVOY_IMAGE)" in content
    assert "VLLM_SR_DASHBOARD_IMAGE=$(FULL_DASHBOARD_IMAGE)" in content
    assert "VLLM_SR_TOPOLOGY" not in content


def test_makefile_does_not_reference_removed_cli_commands() -> None:
    content = MAKEFILE_PATH.read_text(encoding="utf-8")

    forbidden = [
        "show-config: install",
        "python -m cli.main show-config",
        "generate: install",
        "python -m cli.main generate",
        "make show-config",
        "make generate     - Generate configurations",
        "python -m cli.main validate --config $(CONFIG_FILE)",
    ]
    for needle in forbidden:
        assert (
            needle not in content
        ), f"removed CLI surface leaked into Makefile: {needle}"


def test_envoy_image_default_is_aligned_and_smoke_checked() -> None:
    surfaces = [
        ROOT / "src/vllm-sr/cli/consts.py",
        ROOT / "src/vllm-sr/Makefile",
        ROOT / "src/vllm-sr/rebuild-and-test.sh",
        ROOT / "tools/make/docker.mk",
    ]

    for surface in surfaces:
        content = surface.read_text(encoding="utf-8")
        assert EXPECTED_ENVOY_IMAGE in content, f"stale Envoy default in {surface}"
        assert "envoyproxy/envoy:v1.34-latest" not in content

    local_makefile = MAKEFILE_PATH.read_text(encoding="utf-8")
    root_makefile = (ROOT / "tools/make/docker.mk").read_text(encoding="utf-8")
    assert "docker run --rm $(FULL_ENVOY_IMAGE) --version >/dev/null" in local_makefile
    assert (
        "$(CONTAINER_RUNTIME) run --rm $(VLLM_SR_ENVOY_IMAGE) --version >/dev/null"
        in root_makefile
    )
