"""Vela 3.1 yes/no semantics on a tiny package, without executing package code."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest
from starlette.testclient import TestClient
from vllm_srun.api.app import create_app
from vllm_srun.config import ModelConfig, ServeConfig
from vllm_srun.errors import PackageError
from vllm_srun.families.vela3 import package as packages
from vllm_srun.families.vela3.family import Vela3Family
from vllm_srun.plugins.base import PackageRef
from vllm_srun.registry.artifacts import sha256_file
from vllm_srun.runtime import Runtime
from vllm_srun.testing.vela3 import write_encoder_package

STATE = "Please refund my duplicate charge."
INSTRUCTIONS = "Does this request need urgent action?"
RENDERER = b'raise RuntimeError("package code must not execute")\n'


@pytest.fixture(scope="module")
def legacy_package(tmp_path_factory) -> Path:
    return write_encoder_package(tmp_path_factory.mktemp("vela31") / "legacy")


@pytest.fixture
def package(legacy_package, tmp_path, monkeypatch) -> Path:
    root = Path(shutil.copytree(legacy_package, tmp_path / "renamed-package"))
    renderer = root / packages.RENDER31_FILE
    renderer.write_bytes(RENDERER)
    # The tiny fixture has an inert renderer; exercise the same fingerprint gate.
    monkeypatch.setattr(packages, "RENDER31_SHA256", sha256_file(renderer))
    sums = root / "SHA256SUMS"
    sums.write_text(
        sums.read_text() + f"{sha256_file(renderer)}  {packages.RENDER31_FILE}\n"
    )
    return root


@pytest.fixture
def runtime(package):
    runtime = Runtime(
        ServeConfig(models=(ModelConfig(model=str(package), device="cpu"),))
    )
    runtime.start(background=False)
    try:
        yield runtime
    finally:
        runtime.stop()


@pytest.mark.parametrize(
    "criteria,expected",
    [
        (
            {"false": "can wait", "true": "needs immediate action"},
            (("no", "can wait"), ("yes", "needs immediate action")),
        ),
        ({"true": "urgent"}, (("no", ""), ("yes", "urgent"))),
        ({"false": "not urgent"}, (("no", "not urgent"), ("yes", ""))),
        ({"false": None, "true": None}, (("no", ""), ("yes", ""))),
        ({"true": "立即处理"}, (("no", ""), ("yes", "立即处理"))),
        (None, (("no", ""), ("yes", ""))),
    ],
)
def test_noul_descriptions_reach_the_native_forward(runtime, criteria, expected):
    model = runtime.lookup(None).model
    assert model is not None
    question = {"type": "noul", "instructions": INSTRUCTIONS}
    if criteria is not None:
        question["criteria"] = criteria
    plan = model.plan(STATE, {"q": question})
    assert not plan.errors
    assert plan.request.asked[0].question.options == expected
    equivalent = model.plan(
        STATE,
        {
            "q": {
                "type": "choice",
                "instructions": INSTRUCTIONS,
                "criteria": {
                    name: description or None for name, description in expected
                },
            }
        },
    )
    assert plan.items[0].ids == equivalent.items[0].ids
    assert "render31" not in sys.modules


@pytest.mark.parametrize("path", ["/v1/decisions", "/v1/systemone"])
def test_described_noul_through_http_matches_equivalent_choice(runtime, path):
    body = {
        "state": STATE,
        "questions": {
            "urgent": {
                "type": "noul",
                "instructions": INSTRUCTIONS,
                "criteria": {"false": "can wait", "true": "needs immediate action"},
            },
            "choice": {
                "type": "choice",
                "instructions": INSTRUCTIONS,
                "criteria": {"no": "can wait", "yes": "needs immediate action"},
            },
        },
    }
    with TestClient(create_app(runtime)) as client:
        response = client.post(path, json=body)
    assert response.status_code == 200
    answers = response.json()["answers"]
    assert "error" not in answers["urgent"]
    assert answers["urgent"]["noul"] == answers["choice"]["probabilities"]["yes"]


def test_legacy_package_keeps_rejecting_descriptions(legacy_package):
    from .test_vela3_family import load

    model = load(legacy_package)
    plan = model.plan(
        STATE,
        {
            "q": {
                "type": "noul",
                "instructions": INSTRUCTIONS,
                "criteria": {"true": "urgent"},
            }
        },
    )
    assert not model.package.render31
    assert plan.errors["q"]["reason"] == "question_malformed"
    bare = model.plan(STATE, {"q": {"type": "noul", "instructions": INSTRUCTIONS}})
    assert not bare.errors
    assert bare.request.asked[0].question.options == (("no", ""), ("yes", ""))


def test_router_option_expansion_still_matches_the_outer_package(runtime):
    model = runtime.lookup(None).model
    plan = model.plan(
        STATE,
        {
            "q": {
                "type": "choice",
                "instructions": "Which subject area is this request about?",
                "criteria": {"health": None, "math": None},
            }
        },
    )
    assert not plan.errors
    assert plan.request.asked[0].question.options == (
        ("health", "medicine"),
        ("math", "mathematics"),
    )


def test_blank_noul_descriptions_remain_invalid(runtime):
    model = runtime.lookup(None).model
    plan = model.plan(
        STATE,
        {"q": {"type": "noul", "instructions": INSTRUCTIONS, "criteria": {"true": ""}}},
    )
    assert plan.errors["q"]["reason"] == "question_malformed"
    assert not plan.items


def test_unknown_renderer_is_rejected(package):
    (package / packages.RENDER31_FILE).write_bytes(RENDERER + b"# changed\n")
    with pytest.raises(PackageError, match="renderer fingerprint"):
        Vela3Family().verify(PackageRef(package))


def test_manifest_declared_renderer_must_exist(package):
    (package / packages.RENDER31_FILE).unlink()
    with pytest.raises(PackageError, match=r"render31\.py"):
        Vela3Family().verify(PackageRef(package))


def test_renderer_fingerprint_is_part_of_verified_files(package):
    verified = Vela3Family().verify(PackageRef(package))
    details = verified.details["package"]
    assert details.render31
    assert details.files[packages.RENDER31_FILE] == sha256_file(
        package / packages.RENDER31_FILE
    )
