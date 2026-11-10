"""The Vela 3.0 d3 golden request answers through the real readiness path, with and without a recorded reference."""

from __future__ import annotations

import copy

import pytest
from vllm_srun.accel.cpu import CPUAccelerator
from vllm_srun.errors import VerificationError
from vllm_srun.families.vela3_d3.family import GOLDEN_QUESTIONS, Vela3D3Family
from vllm_srun.plugins.decisions import well_formed
from vllm_srun.supervision.readiness import CPU_TOLERANCE
from vllm_srun.testing import vela3_d3 as fixture

from .conftest import start_runtime

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


@pytest.fixture(scope="module")
def written(tmp_path_factory):
    return fixture.write_fixture(
        tmp_path_factory.mktemp("v3golden") / "vela3-fixture", "text", 7
    )


def start(written):
    base = written.parent / f"{written.name}-base"
    return start_runtime(written, base_path=str(base))


@pytest.fixture(scope="module")
def unrecorded(written):
    runtime = start(written)
    served = runtime.lookup(None)
    goldens = copy.deepcopy(served.family.golden(served.package))
    values = [served.golden_surface(g["surface"], g["body"]) for g in goldens]
    yield runtime, goldens, values
    runtime.stop()


def test_without_a_recorded_reference_the_check_is_unverified_not_failed(
    unrecorded,
) -> None:
    runtime, goldens, _ = unrecorded
    assert [g["expected"] for g in goldens] == [{}]
    served = runtime.lookup(None)
    assert served.health.ready
    assert served.health.golden.status == "unverified"


def test_every_question_answers_well_formed_and_the_span_question_too(
    unrecorded,
) -> None:
    _, _, (values,) = unrecorded
    assert set(values) == set(GOLDEN_QUESTIONS)
    assert all(well_formed(answer) for answer in values.values())
    assert values["contact"]["type"] == "noul"


def test_the_answers_are_deterministic(unrecorded) -> None:
    runtime, goldens, (values,) = unrecorded
    again = runtime.lookup(None).golden_surface(
        goldens[0]["surface"], goldens[0]["body"]
    )
    assert again == values


def serve_against(written, unrecorded, monkeypatch, drift):
    runtime, goldens, (values,) = unrecorded
    reference = copy.deepcopy(values)
    reference["contact"]["noul"] += drift
    key = CPUAccelerator().golden_reference(runtime.lookup(None).placement.device)
    expected = {"cpu": reference, **({key: reference} if key else {})}
    recorded = [{**goldens[0], "expected": expected}]
    monkeypatch.setattr(Vela3D3Family, "golden", lambda self, package: recorded)
    return start(written)


def test_a_matching_reference_for_the_device_class_matches(
    written, unrecorded, monkeypatch
) -> None:
    runtime = serve_against(written, unrecorded, monkeypatch, 0.0)
    try:
        served = runtime.lookup(None)
        assert served.health.golden.status == "matched"
        assert served.health.ready
    finally:
        runtime.stop()


def test_a_span_answer_that_drifts_from_the_reference_fails_the_check(
    written, unrecorded, monkeypatch
) -> None:
    with pytest.raises(VerificationError, match="differ from the reference"):
        serve_against(written, unrecorded, monkeypatch, 2 * CPU_TOLERANCE)
