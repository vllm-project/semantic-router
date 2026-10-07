"""`vllm-sr config validate` asks the Router itself (#4696, #4697)."""

from __future__ import annotations

import json
import subprocess

import pytest
from cli import router_validation
from cli.main import main
from cli.router_management_client import RouterManagementError, RouterResponse
from cli.router_validation import (
    RouterValidationUnavailableError,
    RouterVerdict,
    RouterWarning,
    validate_with_endpoint,
    validate_with_image,
)
from click.testing import CliRunner

CONFIG = """\
version: v0.3
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
routing:
  modelCards:
    - name: m
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


@pytest.mark.router_image
def test_validation_image_prefers_the_stacks_router(monkeypatch):
    monkeypatch.setattr(router_validation, "_container_runtime_installed", lambda: True)
    monkeypatch.setattr(
        router_validation, "_container_image", lambda _name: "stack:image"
    )
    present = {"stack:image", router_validation.VLLM_SR_CONTAINER_IMAGE_DEFAULT}
    monkeypatch.setattr(
        router_validation, "container_image_exists", present.__contains__
    )

    assert router_validation.validation_image() == "stack:image"
    with pytest.raises(RouterValidationUnavailableError, match="not present locally"):
        router_validation.validation_image("missing:image")


@pytest.mark.router_image
def test_validation_never_pulls(monkeypatch):
    monkeypatch.setattr(router_validation, "_container_runtime_installed", lambda: True)
    monkeypatch.setattr(router_validation, "_container_image", lambda _name: "")
    monkeypatch.setattr(
        router_validation, "container_image_exists", lambda _image: False
    )

    with pytest.raises(
        RouterValidationUnavailableError, match="no Router image is present"
    ):
        router_validation.validation_image()


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


def test_config_validate_says_when_only_the_cli_checked(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(CONFIG, encoding="utf-8")

    result = CliRunner().invoke(main, ["config", "validate", "--config", str(config)])

    assert result.exit_code == 0, result.output
    assert "Only the CLI's own checks ran" in result.stdout
    assert result.stderr == ""
    offline = CliRunner().invoke(
        main, ["config", "validate", "--config", str(config), "--offline"]
    )
    assert (
        offline.exit_code == 0 and "Only the CLI's own checks ran" not in offline.output
    )
