"""Static contracts of the router image Dockerfile (tools/docker/Dockerfile.extproc)."""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
ROUTER_DOCKERFILE = REPO_ROOT / "tools" / "docker" / "Dockerfile.extproc"
START_ROUTER = REPO_ROOT / "src" / "vllm-sr" / "start-router.sh"
STAGE = re.compile(r"^FROM\s+(?:--platform=\S+\s+)?(\S+)\s+AS\s+(\S+)\s*$", re.M)
STAGES = (
    "router-build",
    "vela-omni",
    "image-routing-assets",
    "python-base",
    "torch-cpu",
    "torch-rocm",
    "torch-cuda",
    "runtime",
    "router",
    "vllm-sr",
    "extproc",
)


def router_dockerfile() -> str:
    return ROUTER_DOCKERFILE.read_text(encoding="utf-8")


def stages() -> dict[str, tuple[str, str]]:
    """Each stage's base and body, in file order."""
    text = router_dockerfile()
    matches = list(STAGE.finditer(text))
    return {
        match.group(2): (
            match.group(1),
            text[
                match.end() : (
                    matches[index + 1].start()
                    if index + 1 < len(matches)
                    else len(text)
                )
            ],
        )
        for index, match in enumerate(matches)
    }


def test_every_router_image_comes_from_one_stage_graph() -> None:
    graph = stages()

    # extproc stays last: it is the default target of local and E2E builds.
    assert tuple(graph) == STAGES
    assert graph["runtime"][0] == "torch-${ACCELERATOR}"
    for accelerator in ("cpu", "rocm", "cuda"):
        assert graph[f"torch-{accelerator}"][0] == "python-base"
    assert graph["router"][0] == "runtime"
    assert graph["vllm-sr"][0] == "router"
    assert graph["extproc"][0] == "router"
    assert "ARG ACCELERATOR=cpu" in router_dockerfile()


def test_router_images_carry_no_native_model_bindings() -> None:
    text = router_dockerfile()

    for legacy in (
        "cargo",
        "rustup",
        "-binding",
        "onnxruntime_rocm",
        "migraphx",
        "openvino",
        "LD_LIBRARY_PATH",
        "ORT_DYLIB_PATH",
        "AI_BINDING",
        "nvidia/cuda",
        "rocm/dev-ubuntu",
    ):
        assert legacy not in text, legacy


def test_router_binary_needs_only_the_target_c_linker() -> None:
    graph = stages()
    build = graph["router-build"][1]

    assert "CGO_ENABLED=1" in build
    assert "gcc-aarch64-linux-gnu" in build
    assert "go build" in build
    for image_stage in ("python-base", "runtime", "router", "vllm-sr", "extproc"):
        assert "apt-get" not in graph[image_stage][1], image_stage
    assert "COPY --link --from=router-build /out/router /usr/local/bin/router" in (
        graph["router"][1]
    )


def test_runtime_takes_pytorch_from_the_accelerator_wheels() -> None:
    graph = stages()

    assert "https://download.pytorch.org/whl/cpu" in graph["torch-cpu"][1]
    assert "https://download.pytorch.org/whl/rocm" in graph["torch-rocm"][1]
    assert "https://download.pytorch.org/whl/cu" in graph["torch-cuda"][1]
    assert "fla-core" not in graph["torch-cpu"][1]
    for accelerator in ("rocm", "cuda"):
        body = graph[f"torch-{accelerator}"][1]
        assert 'test "$TARGETARCH" = amd64' in body, accelerator
        assert "fla-core==" in body, accelerator


def test_runtime_dependencies_come_from_the_runtime_package() -> None:
    runtime = stages()["runtime"][1]

    assert "source=src/model-runtime/pyproject.toml" in runtime
    assert "pip install --no-deps /tmp/model-runtime" in runtime
    assert "ARG MODEL_RUNTIME_EXTRAS=multimodal" in runtime
    assert "==" not in runtime


def test_router_images_share_the_model_assets() -> None:
    router = stages()["router"][1]

    assert (
        "COPY --link --from=vela-omni /opt/router-model-artifacts/ /opt/router-model-artifacts/"
        in router
    )
    assert (
        "COPY --link --from=image-routing-assets /out/ /app/share/image-routing/"
        in (router)
    )
    assert "COPY config/knowledge_bases/ /app/config/knowledge_bases/" in router
    assert "ENV VLLM_SR_RUNTIME_CACHE_DIR=/app/models/model-runtime" in router


def test_vllm_sr_image_ships_the_cli_runtime_sync_and_catalog() -> None:
    vllm_sr = stages()["vllm-sr"][1]

    assert "COPY src/vllm-sr/cli/ /app/cli/" in vllm_sr
    assert "COPY config/recipes/built-in/ /app/cli/model_assets/" in vllm_sr
    assert (
        "COPY src/semantic-router/pkg/configschema/router-config-v0.3.schema.json "
        "/app/cli/config_schema/router-config-v0.3.schema.json"
    ) in vllm_sr
    # The script is not executable in git; the entrypoint needs the bit.
    assert (
        "COPY --chmod=0755 src/vllm-sr/start-router.sh /app/start-router.sh" in vllm_sr
    )
    assert 'ENTRYPOINT ["/app/start-router.sh"]' in vllm_sr
    assert 'VOLUME ["/app/models"]' in vllm_sr


def test_extproc_image_runs_the_router_with_the_reference_config() -> None:
    extproc = stages()["extproc"][1]

    assert "COPY config/config.yaml /app/config/config.yaml" in extproc
    assert (
        'ENTRYPOINT ["/usr/local/bin/router", "--config=/app/config/config.yaml"]'
        in extproc
    )


def test_base_images_are_fully_qualified_for_podman() -> None:
    local = set(STAGES)
    for base, _ in stages().values():
        if base in local or base == "torch-${ACCELERATOR}":
            continue
        assert base.startswith("${IMAGE_REGISTRY}"), base


def test_router_entrypoint_does_not_override_management_listener_config() -> None:
    content = START_ROUTER.read_text(encoding="utf-8")

    assert "exec /usr/local/bin/router" in content
    assert "-enable-api=true" in content
    assert "-api-port=" not in content
    assert "-api-bind=" not in content


def test_build_context_excludes_runtime_state() -> None:
    content = (REPO_ROOT / ".dockerignore").read_text(encoding="utf-8")

    for pattern in ("**/.vllm-sr/", "**/milvus-data/", "**/etcd/", "**/postgres-data/"):
        assert pattern in content
