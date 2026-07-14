from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DASHBOARD_DOCKERFILE = REPO_ROOT / "dashboard" / "backend" / "Dockerfile"
LEGACY_DASHBOARD_DOCKERFILE = REPO_ROOT / "dashboard" / "Dockerfile"
PRECOMMIT_DOCKERFILE = REPO_ROOT / "tools" / "docker" / "Dockerfile.precommit"
VLLM_SR_DOCKERFILE = REPO_ROOT / "src" / "vllm-sr" / "Dockerfile"
VLLM_SR_ROCM_DOCKERFILE = REPO_ROOT / "src" / "vllm-sr" / "Dockerfile.rocm"
VLLM_SR_CUDA_DOCKERFILE = REPO_ROOT / "src" / "vllm-sr" / "Dockerfile.cuda"
VLLM_SR_CUDA_DOCKERIGNORE = (
    REPO_ROOT / "src" / "vllm-sr" / "Dockerfile.cuda.dockerignore"
)
EXTPROC_DOCKERFILE = REPO_ROOT / "tools" / "docker" / "Dockerfile.extproc"
EXTPROC_ROCM_DOCKERFILE = REPO_ROOT / "tools" / "docker" / "Dockerfile.extproc-rocm"
OPERATOR_DOCKERFILE = REPO_ROOT / "deploy" / "operator" / "Dockerfile"
ONNX_FA_TEST_DOCKERFILE = REPO_ROOT / "onnx-binding" / "Dockerfile.fa-test"
OPERATOR_CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "operator-ci.yml"
CI_CHANGES_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci-changes.yml"
DOCKER_PUBLISH_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "docker-publish.yml"
DASHBOARD_TEST_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "dashboard-test.yml"
PRECOMMIT_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "pre-commit.yml"
DOCKER_MAKEFILE = REPO_ROOT / "tools" / "make" / "docker.mk"
CURRENT_NETWORK_TIP_DOCS = {
    REPO_ROOT
    / "website"
    / "docs"
    / "troubleshooting"
    / "network-tips.md": "Do not set `GOSUMDB=off`",
    REPO_ROOT
    / "website"
    / "i18n"
    / "zh-Hans"
    / "docusaurus-plugin-content-docs"
    / "current"
    / "troubleshooting"
    / "network-tips.md": "不要设置 `GOSUMDB=off`",
}


def _dockerfile_stage(content: str, alias: str) -> str:
    lines = content.splitlines()
    starts = [index for index, line in enumerate(lines) if line.startswith("FROM ")]

    for position, start in enumerate(starts):
        if not lines[start].endswith(f" AS {alias}"):
            continue
        end = starts[position + 1] if position + 1 < len(starts) else len(lines)
        return "\n".join(lines[start:end])

    raise AssertionError(f"Dockerfile stage not found: {alias}")


def _indented_section(content: str, header: str) -> str:
    lines = content.splitlines()
    try:
        start = lines.index(header)
    except ValueError as error:
        raise AssertionError(f"Section not found: {header.strip()}") from error

    indentation = len(header) - len(header.lstrip())
    end = len(lines)
    for index in range(start + 1, len(lines)):
        line = lines[index]
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if len(line) - len(line.lstrip()) <= indentation:
            end = index
            break
    return "\n".join(lines[start:end])


def test_router_dockerfiles_copy_the_complete_candle_go_package() -> None:
    for dockerfile in (
        VLLM_SR_DOCKERFILE,
        VLLM_SR_ROCM_DOCKERFILE,
        VLLM_SR_CUDA_DOCKERFILE,
    ):
        content = dockerfile.read_text(encoding="utf-8")

        assert "COPY candle-binding/go.mod candle-binding/*.go ./" in content
        assert (
            "COPY --from=rust-builder /build/go.mod /build/*.go "
            "/build/../candle-binding/"
        ) in content
        assert content.index(
            'RUN find target -name "libcandle_semantic_router.so"'
        ) < content.index("COPY candle-binding/go.mod candle-binding/*.go ./")


def test_gpu_router_dockerfiles_copy_the_complete_onnx_go_package() -> None:
    for dockerfile in (VLLM_SR_ROCM_DOCKERFILE, VLLM_SR_CUDA_DOCKERFILE):
        content = dockerfile.read_text(encoding="utf-8")

        assert "COPY onnx-binding/go.mod onnx-binding/*.go ./" in content
        assert (
            "COPY --from=onnx-builder /build/go.mod /build/*.go /build/../onnx-binding/"
        ) in content
        assert content.index(
            'RUN find target -name "libonnx_semantic_router.so"'
        ) < content.index("COPY onnx-binding/go.mod onnx-binding/*.go ./")


def test_router_dockerfiles_copy_complete_ml_and_nlp_go_packages() -> None:
    packages = (
        ("ml", "ml-builder", "libml_semantic_router.so"),
        ("nlp", "nlp-builder", "libnlp_binding.so"),
    )
    for dockerfile in (
        VLLM_SR_DOCKERFILE,
        VLLM_SR_ROCM_DOCKERFILE,
        VLLM_SR_CUDA_DOCKERFILE,
    ):
        content = dockerfile.read_text(encoding="utf-8")

        for package, builder, library in packages:
            package_copy = f"COPY {package}-binding/go.mod {package}-binding/*.go ./"
            assert package_copy in content
            assert (
                f"COPY --from={builder} /build/go.mod /build/*.go "
                f"/build/../{package}-binding/"
            ) in content
            assert content.index(f'RUN find target -name "{library}"') < content.index(
                package_copy
            )


def test_extproc_dockerfiles_copy_complete_binding_go_packages() -> None:
    for dockerfile in (EXTPROC_DOCKERFILE, EXTPROC_ROCM_DOCKERFILE):
        content = dockerfile.read_text(encoding="utf-8")

        for package in ("candle", "onnx", "ml", "nlp", "openvino"):
            manifest_copy = f"COPY {package}-binding/go.mod {package}-binding/"
            source_copy = f"COPY {package}-binding/*.go {package}-binding/"

            assert manifest_copy in content
            assert source_copy in content
            assert content.index("RUN cd src/semantic-router && go mod download") < (
                content.index(source_copy)
            )

        assert "go mod download -modfile=go.onnx.mod" in content


def test_go_module_mirror_args_reach_canonical_image_builds() -> None:
    dockerfile_stages = {
        VLLM_SR_DOCKERFILE: ("go-builder",),
        VLLM_SR_ROCM_DOCKERFILE: ("go-builder",),
        VLLM_SR_CUDA_DOCKERFILE: ("go-builder",),
        EXTPROC_DOCKERFILE: ("go-builder",),
        EXTPROC_ROCM_DOCKERFILE: ("go-builder",),
        DASHBOARD_DOCKERFILE: ("wasm-builder", "backend-builder"),
    }
    for dockerfile, stage_aliases in dockerfile_stages.items():
        content = dockerfile.read_text(encoding="utf-8")

        for stage_alias in stage_aliases:
            stage = _dockerfile_stage(content, stage_alias)
            assert "ARG GOPROXY=https://proxy.golang.org,direct" in stage
            assert "ARG GOSUMDB=sum.golang.org" in stage
            assert 'GOPROXY="${GOPROXY}"' in stage
            assert 'GOSUMDB="${GOSUMDB}"' in stage

    makefile = DOCKER_MAKEFILE.read_text(encoding="utf-8")
    assert "GOPROXY ?= https://proxy.golang.org,direct" in makefile
    assert "GOSUMDB ?= sum.golang.org" in makefile
    assert "export GOPROXY GOSUMDB" in makefile
    assert "GO_MODULE_BUILD_ARGS := --build-arg GOPROXY --build-arg GOSUMDB" in makefile
    assert "--build-arg GOPROXY=$(GOPROXY)" not in makefile
    assert "--build-arg GOSUMDB=$(GOSUMDB)" not in makefile
    assert "GIT_SSL_NO_VERIFY ?= 0" in makefile
    assert "GIT_SSL_NO_VERIFY ?= 1" not in makefile
    assert "ifeq ($(GIT_SSL_NO_VERIFY),1)" in makefile
    assert "VLLM_SR_BUILD_ARGS += --build-arg GIT_SSL_NO_VERIFY=1" in makefile
    assert "VLLM_SR_BUILD_ARGS := " in makefile
    assert "$(GO_MODULE_BUILD_ARGS)" in makefile
    for dockerfile in (
        "tools/docker/Dockerfile.extproc",
        "tools/docker/Dockerfile.extproc-rocm",
    ):
        assert f"build $(GO_MODULE_BUILD_ARGS) -f {dockerfile}" in makefile


def test_other_builders_copy_required_binding_go_files() -> None:
    operator = OPERATOR_DOCKERFILE.read_text(encoding="utf-8")
    for package in ("candle", "ml", "nlp"):
        manifest_copy = f"COPY {package}-binding/go.mod {package}-binding/"
        source_copy = f"COPY {package}-binding/*.go /workspace/{package}-binding/"
        assert manifest_copy in operator
        assert source_copy in operator
        assert operator.index(manifest_copy) < operator.index("RUN go mod download")
        assert operator.index("RUN go mod download") < operator.index(source_copy)

    fa_test = ONNX_FA_TEST_DOCKERFILE.read_text(encoding="utf-8")
    assert "COPY *.go ./" in fa_test


def test_network_docs_use_the_canonical_local_image_flow() -> None:
    for document, checksum_warning in CURRENT_NETWORK_TIP_DOCS.items():
        content = document.read_text(encoding="utf-8")

        assert "tools/docker/Dockerfile.extproc.cn" not in content
        assert "docker compose" not in content
        assert "docker-compose.override.yml" not in content
        assert "CN Dockerfile" not in content
        assert "mock-vllm" not in content
        assert "snapshot_download" not in content
        assert "~/.cache/huggingface" not in content
        assert "GOPROXY=https://goproxy.cn,direct" in content
        assert "GOSUMDB=sum.golang.google.cn" in content
        assert "make vllm-sr-dev" in content
        assert "`./models`" not in content
        assert "config/models/" in content
        assert (
            "vllm-sr serve --config config/config.yaml --image-pull-policy never"
            in content
        )
        assert (
            "vllm-sr serve --config config/config.yaml --image-pull-policy never "
            "--platform amd"
        ) in content
        assert "make vllm-sr-dev VLLM_SR_PLATFORM=nvidia" in content
        assert (
            "vllm-sr serve --config config/config.yaml --image-pull-policy never "
            "--platform nvidia"
        ) in content

        hf_mirror_flow = content.split("HF_ENDPOINT=https://hf-mirror.com", maxsplit=1)[
            1
        ].split("```", maxsplit=1)[0]
        assert hf_mirror_flow.index("make vllm-sr-dev") < hf_mirror_flow.index(
            "vllm-sr serve"
        )
        assert checksum_warning in content


def test_native_dependencies_trigger_their_image_consumers() -> None:
    docker_publish = DOCKER_PUBLISH_WORKFLOW.read_text(encoding="utf-8")
    docker_publish_triggers = {
        trigger: _indented_section(docker_publish, f"  {trigger}:")
        for trigger in ("push", "pull_request")
    }
    pull_request_trigger = docker_publish_triggers["pull_request"]
    assert "types: [opened, synchronize, reopened, ready_for_review]" in (
        pull_request_trigger
    )

    for dependency in (
        ".dockerignore",
        "**/.dockerignore",
        "**/*.dockerignore",
        "Makefile",
        "config/**",
        "scripts/**",
        "tools/ci/docker-build-args.sh",
        "tools/make/**",
        "candle-binding/**",
        "onnx-binding/**",
        "ml-binding/**",
        "nlp-binding/**",
        "openvino-binding/**",
        "dashboard/**",
    ):
        for trigger in docker_publish_triggers.values():
            assert f'- "{dependency}"' in trigger

    operator_ci = OPERATOR_CI_WORKFLOW.read_text(encoding="utf-8")
    operator_ci_triggers = {
        trigger: _indented_section(operator_ci, f"  {trigger}:")
        for trigger in ("push", "pull_request")
    }
    for dependency in (
        ".dockerignore",
        "Makefile",
        "config/**",
        "scripts/**",
        "src/semantic-router/**",
        "candle-binding/**",
        "onnx-binding/**",
        "ml-binding/**",
        "nlp-binding/**",
        "openvino-binding/**",
        "tools/docker/Dockerfile.extproc",
        "tools/make/**",
    ):
        for trigger in operator_ci_triggers.values():
            assert f"- '{dependency}'" in trigger

    ci_changes = CI_CHANGES_WORKFLOW.read_text(encoding="utf-8")
    docker_filter = ci_changes.split("            docker:\n", maxsplit=1)[1].split(
        "            make:\n", maxsplit=1
    )[0]
    for dependency in (
        ".dockerignore",
        "**/.dockerignore",
        "**/*.dockerignore",
        "deploy/operator/Dockerfile",
        "dashboard/backend/Dockerfile",
        "candle-binding/**",
        "onnx-binding/**",
        "ml-binding/**",
        "nlp-binding/**",
        "openvino-binding/**",
    ):
        assert f"- '{dependency}'" in docker_filter


def test_dashboard_dockerfile_uses_glibc_builder_for_cgo_backend() -> None:
    content = DASHBOARD_DOCKERFILE.read_text(encoding="utf-8")

    assert "ARG IMAGE_REGISTRY=docker.io/" in content
    assert (
        "FROM ${IMAGE_REGISTRY}library/golang:1.25-bookworm AS backend-builder"
        in content
    )
    assert "apt_get_install_with_retry build-essential" in content


def test_node_build_and_ci_surfaces_use_active_lts() -> None:
    dashboard = DASHBOARD_DOCKERFILE.read_text(encoding="utf-8")
    assert dashboard.count("FROM ${IMAGE_REGISTRY}library/node:24-alpine AS ") == 2

    legacy_dashboard = LEGACY_DASHBOARD_DOCKERFILE.read_text(encoding="utf-8")
    assert "FROM node:24-alpine AS frontend-builder" in legacy_dashboard

    precommit = PRECOMMIT_DOCKERFILE.read_text(encoding="utf-8")
    assert "https://deb.nodesource.com/setup_24.x" in precommit

    dashboard_workflow = DASHBOARD_TEST_WORKFLOW.read_text(encoding="utf-8")
    assert 'node-version: "24"' in dashboard_workflow

    precommit_workflow = PRECOMMIT_WORKFLOW.read_text(encoding="utf-8")
    assert "node-version: 24" in precommit_workflow

    for content in (
        dashboard,
        legacy_dashboard,
        precommit,
        dashboard_workflow,
        precommit_workflow,
    ):
        for unsupported in ("node:18", "node:20", "node:23", "setup_20.x"):
            assert unsupported not in content


def test_dashboard_dockerfile_retries_runtime_apk_installs() -> None:
    content = DASHBOARD_DOCKERFILE.read_text(encoding="utf-8")

    assert "FROM ${IMAGE_REGISTRY}library/python:3.11-slim-bookworm" in content
    assert (
        "apt_get_install_with_retry ca-certificates curl docker.io gosu wget" in content
    )


def test_dashboard_dockerfile_copies_router_packages_for_backend_builds() -> None:
    content = DASHBOARD_DOCKERFILE.read_text(encoding="utf-8")

    required_packages = (
        "internal/nlgen",
        "pkg/config",
        "pkg/dsl",
        "pkg/modelinventory",
        "pkg/nlgen",
        "pkg/observability",
        "pkg/startupstatus",
        "pkg/utils/jsonunicode",
    )
    for package in required_packages:
        assert (
            f"COPY src/semantic-router/{package}/ "
            f"/app/src/semantic-router/{package}/" in content
        )


def test_dashboard_dockerfile_copies_model_eval_scripts_and_requirements() -> None:
    content = DASHBOARD_DOCKERFILE.read_text(encoding="utf-8")

    assert "COPY src/training/model_eval/ /app/src/training/model_eval/" in content
    assert "ARG TORCH_CPU_INDEX_URL=https://download.pytorch.org/whl/cpu" in content
    assert "ARG TORCH_CPU_VERSION=" in content
    assert (
        '"${VIRTUAL_ENV}/bin/pip" install --no-cache-dir --index-url "${TORCH_CPU_INDEX_URL}" "torch==${TORCH_CPU_VERSION}"'
        in content
    )
    assert content.index('"torch==${TORCH_CPU_VERSION}"') < content.index(
        "-r /app/src/training/model_eval/requirements.txt"
    )
    assert (
        '"${VIRTUAL_ENV}/bin/pip" install --no-cache-dir -r /app/src/training/model_eval/requirements.txt'
        in content
    )
