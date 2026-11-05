"""A hardware reference makes the real runtime ready; a bad reference keeps it unavailable."""

import copy

import pytest
from starlette.testclient import TestClient
from vllm_srun.accel.cpu import CPUAccelerator
from vllm_srun.api.app import create_app
from vllm_srun.config import ModelConfig, ServeConfig
from vllm_srun.families.decision3.family import Decision3Family
from vllm_srun.plugins.decisions import compare_answers
from vllm_srun.runtime import Runtime
from vllm_srun.supervision.readiness import CPU_TOLERANCE, golden_check

from .conftest import start_runtime
from .test_api_contract import check


@pytest.mark.parametrize("valid", [True, False])
def test_real_runtime_hardware_golden_readiness(d3_package, monkeypatch, valid):
    baseline = start_runtime(d3_package)
    try:
        served = baseline.lookup(None)
        goldens = copy.deepcopy(served.family.golden(served.package))
        key = CPUAccelerator().golden_reference(served.placement.device)
        for golden in goldens:
            actual = served.golden_surface(golden["surface"], golden["body"])
            bad = copy.deepcopy(actual)
            name = next(name for name, value in bad.items() if value["type"] == "noul")
            bad[name]["noul"] += 2 * CPU_TOLERANCE
            golden["expected"] = {
                "cpu": bad if valid else actual,
                key: actual if valid else bad,
            }
            # Reproduce the old failure before exercising the real startup wiring.
            if valid:
                old = golden_check(
                    lambda surface, body, produced=actual: produced,
                    lambda surface, values, reference, tolerance: compare_answers(
                        values, reference, tolerance
                    ),
                    [golden],
                    "cpu",
                )
                assert old.status == "failed"
    finally:
        baseline.stop()
    monkeypatch.setattr(Decision3Family, "golden", lambda self, package: goldens)
    runtime = Runtime(
        ServeConfig(models=(ModelConfig(model=str(d3_package), device="cpu"),))
    )
    try:
        runtime.start()
        assert runtime.wait(30) == valid
        client = TestClient(create_app(runtime))
        health = client.get("/health")
        assert health.status_code == (200 if valid else 503)
        models = client.get("/v1/models").json()
        check("ModelList", models)
        card = models["data"][0]
        assert card["ready"] == valid
        assert card["golden"]["reference"] == key
        assert card["golden"]["status"] == ("matched" if valid else "failed")
        response = client.post("/v1/systemone", json=goldens[0]["body"])
        assert response.status_code == (200 if valid else 503)
    finally:
        runtime.stop()
