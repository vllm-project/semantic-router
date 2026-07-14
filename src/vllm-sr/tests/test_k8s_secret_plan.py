"""Generation-planning tests for immutable Kubernetes runtime Secrets."""

import base64
import json
import sys
from pathlib import Path

import pytest

TESTS_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = TESTS_ROOT.parent
for path in (PROJECT_ROOT, TESTS_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from _k8s_test_helpers import k8s_backend  # noqa: E402
from cli.k8s_secret_plan import (  # noqa: E402
    SecretSnapshot,
    desired_encoded_secret_data,
    encode_secret_data,
)


def test_first_revision_creates_and_unchanged_revision_reuses_generation(
    monkeypatch,
):
    instance = k8s_backend()
    monkeypatch.setattr(instance, "_current_router_secret_refs", lambda: [])
    first = instance._plan_env_secret({"VLLM_SR_LOOPER_SHARED_SECRET": "a" * 64})
    assert first.recreate_for_looper_rotation is True

    current_name = first.active_name
    manifest = json.loads(first.new_manifest)
    payload = {
        "metadata": {
            "labels": {
                "app.kubernetes.io/managed-by": "vllm-sr-cli",
                "app.kubernetes.io/instance": "alpha",
                "semantic-router.vllm.ai/runtime-env-secret": "true",
            }
        },
        "immutable": True,
        "data": manifest["data"],
    }
    monkeypatch.setattr(instance, "_current_router_secret_refs", lambda: [current_name])
    monkeypatch.setattr(instance, "_get_secret_json", lambda _: payload)

    unchanged = instance._plan_env_secret({"VLLM_SR_LOOPER_SHARED_SECRET": "a" * 64})
    assert unchanged.active_name == current_name
    assert unchanged.creates_secret is False
    assert unchanged.new_manifest is None
    assert unchanged.recreate_for_looper_rotation is True


def test_explicit_looper_rotation_creates_exact_new_generation(monkeypatch):
    instance = k8s_backend()
    current_name = "alpha-vsr-env-current"
    payload = json.loads(instance._secret_manifest(current_name, {}))
    payload["data"] = instance._encode_secret_data(
        {"VLLM_SR_LOOPER_SHARED_SECRET": "a" * 64, "HF_TOKEN": "omitted"}
    )
    monkeypatch.setattr(instance, "_current_router_secret_refs", lambda: [current_name])
    monkeypatch.setattr(instance, "_get_secret_json", lambda _: payload)

    rotated = instance._plan_env_secret({"VLLM_SR_LOOPER_SHARED_SECRET": "b" * 64})
    assert rotated.active_name != current_name
    assert json.loads(rotated.new_manifest)["data"] == instance._encode_secret_data(
        {"VLLM_SR_LOOPER_SHARED_SECRET": "b" * 64}
    )
    assert rotated.recreate_for_looper_rotation is True


@pytest.mark.parametrize(
    ("cli_managed", "previous_value"),
    [
        (False, encode_secret_data({"VLLM_SR_LOOPER_SHARED_SECRET": "a" * 64})),
        (True, {"VLLM_SR_LOOPER_SHARED_SECRET": "not-valid-base64"}),
    ],
)
def test_unmanaged_or_damaged_previous_looper_key_is_not_reused(
    monkeypatch, cli_managed, previous_value
):
    generated = "c" * 64
    previous = SecretSnapshot(
        name="alpha-vsr-env-previous",
        data=previous_value,
        immutable=True,
        cli_managed=cli_managed,
        data_known=True,
    )
    monkeypatch.setattr("cli.k8s_secret_plan.secrets.token_hex", lambda _: generated)

    desired = desired_encoded_secret_data(None, previous)
    encoded = desired["VLLM_SR_LOOPER_SHARED_SECRET"]
    assert base64.b64decode(encoded).decode() == generated
