"""The deployment fixture uses the released renderer fingerprint as opaque data."""

from __future__ import annotations

import sys

import pytest
from starlette.testclient import TestClient
from vllm_srun.api.app import create_app
from vllm_srun.cli import main
from vllm_srun.config import ModelConfig, ServeConfig
from vllm_srun.families.vela3.family import Vela3Family
from vllm_srun.plugins.base import PackageRef
from vllm_srun.registry.artifacts import read_sums, sha256_file
from vllm_srun.runtime import Runtime


@pytest.fixture(scope="module")
def encoder31(tmp_path_factory):
    root = tmp_path_factory.mktemp("encoder31") / "package"
    assert (
        main(["fixture", str(root), "--family", "vela3", "--variant", "encoder31"]) == 0
    )
    return root


def test_encoder31_fixture_uses_the_released_opaque_renderer(encoder31):
    digest = "d0c8ebc0896c7d3b942f5eba98cdd247ef01e08d329237015bef4fc55009c8ad"
    assert sha256_file(encoder31 / "render31.py") == digest
    assert read_sums(encoder31)["render31.py"] == digest
    verified = Vela3Family().verify(PackageRef(encoder31))
    assert verified.details["package"].render31
    assert verified.details["package"].files["render31.py"] == digest
    assert "render31" not in sys.modules


def test_encoder_default_remains_legacy(tmp_path):
    root = tmp_path / "legacy"
    assert main(["fixture", str(root), "--family", "vela3"]) == 0
    assert not (root / "render31.py").exists()
    verified = Vela3Family().verify(PackageRef(root))
    assert not verified.details["package"].render31


def test_encoder31_deployment_question_matches_equivalent_choice(encoder31):
    runtime = Runtime(
        ServeConfig(models=(ModelConfig(model=str(encoder31), device="cpu"),))
    )
    runtime.start(background=False)
    try:
        body = {
            "state": "[vela31-described-noul] Please refund my duplicate charge.",
            "questions": {
                "vela31_described": {
                    "type": "noul",
                    "instructions": "Does this request need urgent action?",
                    "criteria": {"false": "can wait", "true": "needs immediate action"},
                },
                "equivalent": {
                    "type": "choice",
                    "instructions": "Does this request need urgent action?",
                    "choices": [
                        {"key": "no", "description": "can wait"},
                        {"key": "yes", "description": "needs immediate action"},
                    ],
                },
            },
        }
        with TestClient(create_app(runtime)) as client:
            response = client.post("/v1/decisions", json=body)
        assert response.status_code == 200
        answers = response.json()["answers"]
        probability = answers["vela31_described"]["noul"]
        assert probability == answers["equivalent"]["probabilities"]["yes"]
        assert 0.0001 <= probability <= 1
        model = runtime.lookup(None).model
        # The pre-fix Reader rejects this same deployed question.
        model.reader.render31 = False
        plan = model.plan(body["state"], body["questions"])
        assert plan.errors["vela31_described"]["reason"] == "question_malformed"
        assert "render31" not in sys.modules
    finally:
        runtime.stop()
