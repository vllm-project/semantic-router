"""`vllm-sr config validate` asks the Router itself (#4696, #4697)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
from cli import container_images, router_validation
from cli.main import main
from cli.router_management_client import RouterManagementError, RouterResponse
from cli.router_validation import (
    RouterValidationUnavailableError,
    RouterVerdict,
    RouterWarning,
    validate_with_endpoint,
    validate_with_image,
    with_moving_tag_hint,
)
from click.testing import CliRunner

CONFIG = """version: v0.5
listeners:
  - name: http-8899
    address: 0.0.0.0
    port: 8899
providers:
  defaults:
    model: m
  models:
    - name: m
      backend_refs:
        - endpoint: host.docker.internal:8000
recipes:
  - name: default
"""

MODALITY_WARNING = {
    "code": "modality_detector_disabled",
    "field": "global.model_catalog.modules.modality_detector",
    "message": 'the modality signal never matches: decision "draw" uses it',
}


def _completed(stdout: str, returncode: int = 0, stderr: str = ""):
    return subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout=stdout.encode(), stderr=stderr.encode()
    )


def test_the_router_image_validates_offline_from_stdin(monkeypatch, tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(CONFIG, encoding="utf-8")
    models = tmp_path / "models"
    models.mkdir()
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return _completed(json.dumps({"valid": True, "warnings": [MODALITY_WARNING]}))

    monkeypatch.setattr(router_validation, "get_container_runtime", lambda: "docker")
    monkeypatch.setattr(router_validation.subprocess, "run", run)

    verdict = validate_with_image(
        config, "router:test", gateway="standalone", models_dir=models
    )

    command, kwargs = calls[0]
    assert command[:8] == [
        "docker",
        "run",
        "--rm",
        "-i",
        "--network",
        "none",
        "--entrypoint",
        "/usr/local/bin/router",
    ]
    assert f"{models.resolve()}:/app/models:ro,z" in command
    assert command[-6:] == [
        "router:test",
        "-validate-config",
        "-config",
        "/dev/stdin",
        "-gateway",
        "standalone",
    ]
    assert kwargs["input"] == CONFIG.encode()
    assert verdict.valid and verdict.source == "the Router in router:test"
    assert verdict.warnings == (RouterWarning(**MODALITY_WARNING),)


def test_the_router_image_refusal_is_the_routers_message(monkeypatch, tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(CONFIG, encoding="utf-8")
    refusal = 'decision "image-gen" uses modality condition "BOTH" but modelRefs must include ...'
    monkeypatch.setattr(router_validation, "get_container_runtime", lambda: "docker")
    monkeypatch.setattr(
        router_validation.subprocess,
        "run",
        lambda *_a, **_k: _completed(
            json.dumps({"valid": False, "error": refusal, "warnings": []}), 1
        ),
    )

    verdict = validate_with_image(config, "router:test", gateway="standalone")

    assert not verdict.valid and verdict.error == refusal


def test_an_image_without_the_flag_is_reported_not_trusted(monkeypatch, tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(CONFIG, encoding="utf-8")
    monkeypatch.setattr(router_validation, "get_container_runtime", lambda: "docker")
    monkeypatch.setattr(
        router_validation.subprocess,
        "run",
        lambda *_a, **_k: _completed(
            "", 2, "flag provided but not defined: -validate-config\nUsage of router:"
        ),
    )

    with pytest.raises(RouterValidationUnavailableError, match="predates"):
        validate_with_image(config, "vllm-sr:v0.4.0", gateway="standalone")


def _host_images(monkeypatch, present, running=None, checkout=None, runtime="docker"):
    monkeypatch.setattr(router_validation, "_container_runtime_installed", lambda: True)
    monkeypatch.setattr(router_validation, "get_container_runtime", lambda: runtime)
    monkeypatch.setattr(router_validation, "_running_router", lambda: running)
    monkeypatch.setattr(router_validation, "_source_checkout", lambda: checkout)
    monkeypatch.setattr(
        router_validation, "container_image_exists", set(present).__contains__
    )
    for name in ("VLLM_SR_ROUTER_IMAGE", "VLLM_SR_IMAGE"):
        monkeypatch.delenv(name, raising=False)


REPOSITORY = "ghcr.io/vllm-project/semantic-router"
CHECKOUT = Path("/src/semantic-router")


def _own_images(monkeypatch, tag, served="vllm-sr"):
    """This CLI's own Router images at *tag*; serve would run *served* here."""
    images = {
        name: f"{REPOSITORY}/{name}:{tag}"
        for name in ("vllm-sr", "vllm-sr-rocm", "vllm-sr-cuda")
    }
    for constant, name in (
        ("VLLM_SR_CONTAINER_IMAGE_DEFAULT", "vllm-sr"),
        ("VLLM_SR_CONTAINER_IMAGE_ROCM", "vllm-sr-rocm"),
        ("VLLM_SR_CONTAINER_IMAGE_CUDA", "vllm-sr-cuda"),
    ):
        monkeypatch.setattr(router_validation, constant, images[name])
    monkeypatch.setattr(
        router_validation, "_served_release_image", lambda: images[served]
    )
    return images


def _docker(monkeypatch, verdict, created="2026-11-06T10:12:13.5Z"):
    """The runtime's answers: the image's verdict, and when the image was built."""
    calls = []

    def run(command, **_kwargs):
        calls.append(command[1:3])
        if command[1:3] == ["image", "inspect"]:
            return subprocess.CompletedProcess(command, 0, f"{created}\n", "")
        return _completed(json.dumps(verdict), 0 if verdict["valid"] else 1)

    monkeypatch.setattr(router_validation.subprocess, "run", run)
    return calls


def _config(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(CONFIG, encoding="utf-8")
    return config


def _flat(output):
    return " ".join(output.split())


@pytest.mark.router_image
def test_validation_image_is_this_clis_never_the_running_stacks(monkeypatch):
    cli_image = router_validation.VLLM_SR_CONTAINER_IMAGE_DEFAULT
    _host_images(monkeypatch, {"stack:old", cli_image}, running=("stack:old", "old-id"))

    assert router_validation.validation_image() == cli_image
    assert router_validation.validation_image("stack:old") == "stack:old"
    with pytest.raises(RouterValidationUnavailableError, match="not present locally"):
        router_validation.validation_image("missing:image")


@pytest.mark.router_image
def test_validation_image_is_the_one_the_environment_names(monkeypatch):
    cli_image = router_validation.VLLM_SR_CONTAINER_IMAGE_DEFAULT
    _host_images(monkeypatch, {"pinned:dev", "router:dev", cli_image})

    monkeypatch.setenv("VLLM_SR_IMAGE", "pinned:dev")
    assert router_validation.validation_image() == "pinned:dev"
    monkeypatch.setenv("VLLM_SR_ROUTER_IMAGE", "router:dev")
    assert router_validation.validation_image() == "router:dev"
    monkeypatch.setenv("VLLM_SR_ROUTER_IMAGE", "router:absent")
    with pytest.raises(
        RouterValidationUnavailableError,
        match="router:absent, which VLLM_SR_ROUTER_IMAGE names, is not present",
    ):
        router_validation.validation_image()


@pytest.mark.router_image
@pytest.mark.parametrize(
    ("tag", "checkout", "runtime", "missing"),
    [
        (
            "v0.5.0",
            None,
            "podman",
            "no Router image of this CLI's version is present locally; `podman pull "
            f"{REPOSITORY}/vllm-sr-rocm:v0.5.0` fetches the one `vllm-sr serve` runs "
            "on this host, which `vllm-sr serve` also pulls",
        ),
        (
            "latest",
            CHECKOUT,
            "docker",
            "no Router image built from this checkout is present locally; build the "
            f"one `vllm-sr serve` runs on this host with `make -C {CHECKOUT} "
            "vllm-sr-build VLLM_SR_PLATFORM=rocm` (a registry `latest` is not this "
            "checkout)",
        ),
    ],
)
def test_validation_never_pulls_nor_falls_back_to_the_running_stack(
    monkeypatch, tag, checkout, runtime, missing
):
    _host_images(
        monkeypatch,
        {"stack:old"},
        running=("stack:old", "old-id"),
        checkout=checkout,
        runtime=runtime,
    )
    _own_images(monkeypatch, tag, served="vllm-sr-rocm")

    with pytest.raises(RouterValidationUnavailableError) as refused:
        router_validation.validation_image()
    assert str(refused.value) == (
        f"{missing}; this stack's running Router image stack:old is not used"
    )


@pytest.mark.router_image
def test_no_router_image_and_no_running_stack_says_how_to_fetch_it(monkeypatch):
    _host_images(monkeypatch, set())
    images = _own_images(monkeypatch, "v0.5.0")

    with pytest.raises(RouterValidationUnavailableError) as refused:
        router_validation.validation_image()
    assert str(refused.value) == (
        "no Router image of this CLI's version is present locally; `docker pull "
        f"{images['vllm-sr']}` fetches the one `vllm-sr serve` runs on this host, "
        "which `vllm-sr serve` also pulls"
    )


@pytest.mark.router_image
def test_a_named_missing_image_says_how_to_fetch_it(monkeypatch):
    images = _own_images(monkeypatch, "latest")
    _host_images(monkeypatch, set(), checkout=CHECKOUT)

    with pytest.raises(RouterValidationUnavailableError) as named:
        router_validation.validation_image("router:absent")
    assert str(named.value) == (
        "the Router image router:absent is not present locally; "
        "`docker pull router:absent` fetches it"
    )
    monkeypatch.setenv("VLLM_SR_IMAGE", "pinned:absent")
    with pytest.raises(RouterValidationUnavailableError) as by_environment:
        router_validation.validation_image()
    assert str(by_environment.value) == (
        "the Router image pinned:absent, which VLLM_SR_IMAGE names, is not present "
        "locally; `docker pull pinned:absent` fetches it"
    )
    # This checkout's own moving tag is built from it, not pulled.
    with pytest.raises(RouterValidationUnavailableError) as own:
        router_validation.validation_image(images["vllm-sr"])
    assert str(own.value).endswith(
        f"build it from this checkout with `make -C {CHECKOUT} vllm-sr-build` (a "
        "registry `latest` is not this checkout)"
    )


def test_the_release_image_a_message_names_is_the_variant_serve_runs_here(
    monkeypatch,
):
    for name in ("VLLM_SR_IMAGE", "VLLM_SR_ROUTER_IMAGE", "VLLM_SR_IMAGE_AMD"):
        monkeypatch.delenv(name, raising=False)
    assert (
        router_validation._served_release_image()
        == container_images.VLLM_SR_CONTAINER_IMAGE_DEFAULT
    )
    monkeypatch.setattr(router_validation, "resolve_execution_platform", lambda: "rocm")
    assert (
        router_validation._served_release_image()
        == container_images.VLLM_SR_CONTAINER_IMAGE_ROCM
    )


@pytest.mark.router_image
def test_a_moving_tag_says_when_its_local_copy_was_built(monkeypatch, tmp_path):
    config = _config(tmp_path)
    _host_images(monkeypatch, {"router:latest", "router", "router:v1"})
    _docker(monkeypatch, {"valid": True})

    for image, checked in (
        ("router:latest", "the Router in router:latest (built 2026-11-06)"),
        ("router", "the Router in router (built 2026-11-06)"),
        ("router:v1", "the Router in router:v1"),
    ):
        result = CliRunner().invoke(
            main, ["config", "validate", "--config", str(config), "--image", image]
        )
        assert result.exit_code == 0, result.output
        assert f"Checked by {checked} as well as the CLI." in result.stdout


@pytest.mark.router_image
def test_a_moving_tag_refusal_says_its_copy_may_be_older_and_how_to_rebuild(
    monkeypatch, tmp_path
):
    config = _config(tmp_path)
    images = _own_images(monkeypatch, "latest")
    _host_images(monkeypatch, {images["vllm-sr"]}, checkout=CHECKOUT)
    _docker(
        monkeypatch,
        {"valid": False, "error": 'unknown field "decisions" in recipes[0]'},
    )

    result = CliRunner().invoke(main, ["config", "validate", "--config", str(config)])

    assert result.exit_code == 1
    assert (
        f'the Router in {images["vllm-sr"]} refuses it: unknown field "decisions" in '
        "recipes[0]. `latest` is a moving tag: this local copy was built on "
        "2026-11-06 and may be older than this CLI; rebuild it from this checkout "
        f"(`make -C {CHECKOUT} vllm-sr-build`) or name the Router with `--image` or "
        "`VLLM_SR_IMAGE`."
    ) in _flat(result.output)


@pytest.mark.router_image
@pytest.mark.parametrize("tag", ["v0.5.0", "61c9e248d8b055bd4c07e1e80d8b36ec6048dc7c"])
def test_a_release_or_commit_tag_prints_nothing_new(monkeypatch, tmp_path, tag):
    config = _config(tmp_path)
    images = _own_images(monkeypatch, tag)
    _host_images(monkeypatch, {images["vllm-sr"]}, checkout=CHECKOUT)
    calls = _docker(monkeypatch, {"valid": False, "error": "unknown field"})

    result = CliRunner().invoke(main, ["config", "validate", "--config", str(config)])

    assert result.exit_code == 1
    assert f"the Router in {images['vllm-sr']} refuses it: unknown field" in _flat(
        result.output
    )
    assert "moving tag" not in result.output and "(built" not in result.output
    assert ["image", "inspect"] not in calls


@pytest.mark.parametrize(
    ("rebuild", "pulled", "update"),
    [
        (
            "make -C /src vllm-sr-build",
            False,
            "rebuild it from this checkout (`make -C /src vllm-sr-build`) or ",
        ),
        (
            "make -C /src vllm-sr-build",
            True,
            "rebuild it from this checkout (`make -C /src vllm-sr-build`) and serve "
            "it with `--image-pull-policy ifnotpresent` or ",
        ),
        ("", False, "pull it again (`docker pull router:latest`) or "),
        ("", True, ""),
    ],
)
def test_the_moving_tag_hint_fits_how_the_copy_arrived(rebuild, pulled, update):
    verdict = RouterVerdict(
        source="the Router in router:latest",
        valid=False,
        built="2026-11-06",
        rebuild=rebuild,
        pull="docker pull router:latest",
    )

    assert with_moving_tag_hint("refused.", verdict, "`--image`", pulled=pulled) == (
        "refused. `latest` is a moving tag: this local copy was built on 2026-11-06 "
        f"and may be older than this CLI; {update}name the Router with `--image`."
    )
    fixed = RouterVerdict(source="the Router in router:v1", valid=False)
    assert with_moving_tag_hint("refused.", fixed, "`--image`") == "refused."


@pytest.mark.router_image
def test_migrate_names_a_moving_tag_copy_that_could_not_migrate(monkeypatch, tmp_path):
    config = _config(tmp_path)
    monkeypatch.setattr(
        router_validation,
        "local_router_verdict",
        lambda *_args, **_kwargs: RouterVerdict(
            source="the Router in router:latest",
            valid=True,
            built="2026-11-06",
            rebuild=f"make -C {CHECKOUT} vllm-sr-build",
            pull="docker pull router:latest",
        ),
    )

    result = CliRunner().invoke(
        main,
        ["config", "migrate", "--config", str(config), "--output", str(tmp_path / "o")],
    )

    assert result.exit_code == 1
    assert (
        "The Router could not migrate the configuration: it wrote nothing. `latest` "
        "is a moving tag: this local copy was built on 2026-11-06 and may be older "
        f"than this CLI; rebuild it from this checkout (`make -C {CHECKOUT} "
        "vllm-sr-build`) or name the Router with `VLLM_SR_IMAGE`."
    ) in _flat(result.output)


@pytest.mark.router_image
@pytest.mark.parametrize(
    ("running_id", "reported"), [("old-id", "stack:old"), ("new-id", "")]
)
def test_a_running_router_of_another_image_is_reported(
    monkeypatch, tmp_path, running_id, reported
):
    config = tmp_path / "config.yaml"
    config.write_text(CONFIG, encoding="utf-8")
    _host_images(monkeypatch, {"router:new"}, running=("stack:old", running_id))
    monkeypatch.setattr(router_validation, "_image_id", lambda _image: "new-id")
    monkeypatch.setattr(
        router_validation,
        "validate_with_image",
        lambda _path, image, **_kwargs: RouterVerdict(
            source=f"the Router in {image}", valid=True
        ),
    )

    result = CliRunner().invoke(
        main, ["config", "validate", "--config", str(config), "--image", "router:new"]
    )

    assert result.exit_code == 0, result.output
    assert "Checked by the Router in router:new" in result.stdout
    note = "This stack's running Router is a different image: stack:old."
    assert (note in result.stdout) == bool(reported)


class _Client:
    base_url = "http://localhost:8080"

    def __init__(self, response=None, error=None):
        self._response, self._error = response, error

    def validate_config(self, _yaml_text):
        if self._error:
            raise self._error
        return self._response


def test_a_running_router_refusal_and_warnings(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(CONFIG, encoding="utf-8")
    refused = validate_with_endpoint(
        config,
        _Client(
            error=RouterManagementError(
                "HTTP 422",
                status=422,
                code="CONFIG_VALIDATION_ERROR",
                detail="method is required",
            )
        ),
    )
    assert not refused.valid and refused.error == "method is required"

    valid = validate_with_endpoint(
        config,
        _Client(
            RouterResponse(payload={"valid": True, "warnings": [MODALITY_WARNING]})
        ),
    )
    assert valid.valid and valid.warnings[0].code == "modality_detector_disabled"


def _validate(monkeypatch, tmp_path, verdict):
    config = tmp_path / "config.yaml"
    config.write_text(CONFIG, encoding="utf-8")
    monkeypatch.setattr(
        router_validation, "validation_image", lambda _image=None: "router:test"
    )
    monkeypatch.setattr(
        router_validation, "validate_with_image", lambda *_a, **_k: verdict()
    )
    return CliRunner().invoke(main, ["config", "validate", "--config", str(config)])


@pytest.mark.router_image
def test_config_validate_refuses_what_the_router_refuses(monkeypatch, tmp_path):
    result = _validate(
        monkeypatch,
        tmp_path,
        lambda: RouterVerdict(
            source="the Router in router:test",
            valid=False,
            error='decision "image-gen" uses modality condition "BOTH" but ...',
        ),
    )

    assert result.exit_code == 1
    assert "Configuration is valid" not in result.output
    assert 'the Router in router:test refuses it: decision "image-gen"' in result.output


@pytest.mark.router_image
def test_config_validate_prints_the_routers_warnings(monkeypatch, tmp_path):
    result = _validate(
        monkeypatch,
        tmp_path,
        lambda: RouterVerdict(
            source="the Router in router:test",
            valid=True,
            warnings=(RouterWarning(**MODALITY_WARNING),),
        ),
    )

    assert result.exit_code == 0, result.output
    assert "Checked by the Router in router:test" in result.stdout
    assert "the modality signal never matches" in result.stderr


def test_config_validate_says_when_only_the_structural_checks_ran(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(CONFIG, encoding="utf-8")

    result = CliRunner().invoke(main, ["config", "validate", "--config", str(config)])

    assert result.exit_code == 0, result.output
    assert (
        "Only the CLI's structural checks ran: Router image validation is off"
        in result.stdout
    )
    assert "--router-binary" in result.stdout
    assert result.stderr == ""
    offline = CliRunner().invoke(
        main, ["config", "validate", "--config", str(config), "--offline"]
    )
    assert offline.exit_code == 0
    assert "Only the CLI's structural checks ran (--offline)" in offline.stdout


@pytest.mark.router_image
@pytest.mark.parametrize(
    ("tag", "checkout", "fetch"),
    [
        (
            "v0.5.0",
            None,
            f"`docker pull {REPOSITORY}/vllm-sr:v0.5.0` fetches the one "
            "`vllm-sr serve` runs on this host",
        ),
        (
            "latest",
            CHECKOUT,
            "build the one `vllm-sr serve` runs on this host with "
            f"`make -C {CHECKOUT} vllm-sr-build`",
        ),
    ],
)
def test_config_validate_without_a_router_image_or_a_stack_runs_the_structural_checks(
    monkeypatch, tmp_path, tag, checkout, fetch
):
    _host_images(monkeypatch, set(), checkout=checkout)
    _own_images(monkeypatch, tag)

    result = CliRunner().invoke(
        main, ["config", "validate", "--config", str(_config(tmp_path))]
    )

    assert result.exit_code == 0, result.output
    output = _flat(result.stdout)
    assert "Configuration is valid" in output
    assert "Only the CLI's structural checks ran: no Router image" in output
    assert fetch in output
    assert "running Router image" not in output
    assert "Configuration summary" in output
