"""Static contracts of the router image Dockerfile (tools/docker/Dockerfile.extproc)."""

import re
from pathlib import Path

import tomllib

REPO_ROOT = Path(__file__).resolve().parents[3]
ROUTER_DOCKERFILE = REPO_ROOT / "tools" / "docker" / "Dockerfile.extproc"
START_ROUTER = REPO_ROOT / "src" / "vllm-sr" / "start-router.sh"
MODEL_RUNTIME = REPO_ROOT / "src" / "model-runtime"
STAGE = re.compile(r"^FROM\s+(?:--platform=\S+\s+)?(\S+)\s+AS\s+(\S+)\s*$", re.M)
STAGES = (
    "router-build",
    "image-routing-assets",
    "python-base",
    "torch-cpu",
    "torch-rocm-base",
    "causal-conv1d-rocm",
    "rocm-torch-release",
    "rocm-lib-slim",
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
    for stage in ("torch-cpu", "torch-rocm-base", "torch-cuda"):
        assert graph[stage][0] == "python-base", stage
    assert graph["causal-conv1d-rocm"][0] == "torch-rocm-base"
    assert graph["rocm-torch-release"][0] == "${IMAGE_REGISTRY}${ROCM_TORCH_IMAGE}"
    assert graph["rocm-lib-slim"][0] == "rocm-torch-release"
    assert graph["torch-rocm"][0] == "python-base"
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
    for image_stage in ("python-base", "runtime", "router", "extproc"):
        assert "apt-get" not in graph[image_stage][1], image_stage
    # The CLI's readiness and status probes run curl inside the stack image.
    assert "apt-get install -y --no-install-recommends curl &&" in graph["vllm-sr"][1]
    assert "COPY --link --from=router-build /out/router /usr/local/bin/router" in (
        graph["router"][1]
    )


def test_runtime_takes_pytorch_from_the_accelerator_wheels() -> None:
    graph = stages()

    assert "https://download.pytorch.org/whl/cpu" in graph["torch-cpu"][1]
    assert "https://download.pytorch.org/whl/rocm7.2" in graph["torch-rocm-base"][1]
    assert "https://download.pytorch.org/whl/cu" in graph["torch-cuda"][1]
    assert "fla-core" not in graph["torch-cpu"][1]
    for stage in ("torch-rocm-base", "torch-cuda"):
        body = graph[stage][1]
        assert 'test "$TARGETARCH" = amd64' in body, stage
        assert "fla-core==" in body, stage


def test_rocm_image_builds_causal_conv1d_with_the_released_rocm_sdk() -> None:
    graph = stages()
    build = graph["causal-conv1d-rocm"][1]

    # The released decoders' wheel came from ROCm 7.2.3's compiler, which emits
    # the GPU code their answers were recorded with.
    assert "ARG ROCM_SDK=7.2.3" in build
    assert "ARG CAUSAL_CONV1D=causal-conv1d==1.7.0" in build
    # Instinct MI200, MI300 and MI350: a GPU without code fails its first convolution.
    assert "ARG CAUSAL_CONV1D_ARCHS=gfx90a,gfx942,gfx950" in build
    assert "Pin: release o=repo.radeon.com" in build
    assert "CAUSAL_CONV1D_FORCE_BUILD=TRUE" in build
    assert "--no-build-isolation" in build
    rocm = graph["torch-rocm"][1]
    assert "from=causal-conv1d-rocm,source=/wheels" in rocm
    assert "pip install --no-deps /tmp/wheels/causal_conv1d-" in rocm
    mounts = [
        name for name, (_, body) in graph.items() if "from=causal-conv1d-rocm" in body
    ]
    assert mounts == ["torch-rocm"]


def test_runtime_dependencies_come_from_the_runtime_package() -> None:
    runtime = stages()["runtime"][1]

    assert "source=src/model-runtime/pyproject.toml" in runtime
    assert "pip install --no-deps /tmp/model-runtime" in runtime
    assert "ARG MODEL_RUNTIME_EXTRAS=multimodal" in runtime
    assert "==" not in runtime
    assert "source=src/model-runtime/requirements-lock.txt" in runtime
    assert "pip install -c /tmp/requirements-lock.txt -r" in runtime


def test_runtime_lock_pins_every_dependency_but_pytorch() -> None:
    def name(requirement: str) -> str:
        found = re.match(r"[A-Za-z0-9._-]+", requirement)
        assert found, requirement
        return re.sub(r"[-_.]+", "-", found.group()).lower()

    project = tomllib.loads((MODEL_RUNTIME / "pyproject.toml").read_text())["project"]
    wanted = {
        name(dep)
        for dep in project["dependencies"]
        + project["optional-dependencies"]["multimodal"]
    }
    lines = (MODEL_RUNTIME / "requirements-lock.txt").read_text().splitlines()
    pinned = {name(line) for line in lines if "==" in line and not line.startswith("#")}

    assert wanted - pinned == {"torch"}


def test_router_images_carry_no_onnx_runtime_or_prepared_bundles() -> None:
    pyproject = REPO_ROOT / "src" / "model-runtime" / "pyproject.toml"
    extras = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"][
        "optional-dependencies"
    ]
    assert not any(dep.startswith("onnxruntime") for dep in extras["multimodal"])
    assert any(dep.startswith("onnxruntime") for dep in extras["onnx"])
    assert "router-model-artifacts" not in router_dockerfile()


def test_router_images_share_the_model_assets() -> None:
    router = stages()["router"][1]

    assert (
        "COPY --link --from=image-routing-assets /out/ /app/share/image-routing/"
        in (router)
    )
    assert "COPY config/knowledge_bases/ /app/config/knowledge_bases/" in router
    assert "ENV VLLM_SRUN_CACHE_DIR=/app/models/model-runtime" in router
    # The charts' root filesystem is read-only: GPU caches live in the model volume.
    for cache in (
        "TRITON_CACHE_DIR=/app/models/triton",
        "MIOPEN_USER_DB_PATH=/app/models/miopen",
        "MIOPEN_CUSTOM_CACHE_DIR=/app/models/miopen",
    ):
        assert cache in router, cache


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


def test_rocm_image_serves_with_the_pytorch_of_the_pinned_vllm_rocm_image() -> None:
    graph = stages()
    rocm = graph["torch-rocm"][1]

    # The official wheel's AOTriton misses the released decoders' attention
    # answers; the pinned image's PyTorch and ROCm 7.2.3 libraries match them.
    assert re.search(
        r"^ARG ROCM_TORCH_IMAGE=vllm/vllm-openai-rocm@sha256:[0-9a-f]{64}$",
        router_dockerfile(),
        re.M,
    )
    for path in ("/opt/rocm-7.2.3/lib", "/opt/rocm-7.2.3/share/miopen"):
        assert f"--from=rocm-lib-slim {path} " in rocm, path
    assert "--from=rocm-torch-release ${RELEASE_SITE}/torch " in rocm
    assert "pip uninstall -y torch" in rocm
    assert "torch.version.git_version.startswith('6bbd260')" in rocm
    assert "ldconfig" in rocm


def test_rocm_image_keeps_the_gemm_kernels_of_every_gpu_it_serves() -> None:
    dockerfile = router_dockerfile()
    slim = stages()["rocm-lib-slim"][1]

    archs = re.search(r"^ARG ROCM_GPU_ARCHS=(\S+)$", slim, re.M)
    conv = re.search(r"^ARG CAUSAL_CONV1D_ARCHS=(\S+)$", dockerfile, re.M)
    assert archs and conv
    assert set(archs.group(1).split(",")) == set(conv.group(1).split(","))
    for kept in ("lib/rocblas/library", "lib/hipblaslt/library", "share/miopen/db"):
        assert kept in slim
    assert "find lib -xtype l -delete" in slim


def test_rocm_image_holds_the_cxx_runtime_it_replaces() -> None:
    torch_rocm = stages()["torch-rocm"][1]
    assert "apt-mark hold libstdc++6 libgcc-s1 libgomp1" in torch_rocm
    vllm_sr = stages()["vllm-sr"][1]
    assert vllm_sr.index('python -c "import torch"') > vllm_sr.index("apt-get install")
