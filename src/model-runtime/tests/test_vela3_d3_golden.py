"""The Vela 3.0 d3 golden request: its recorded references and its answers on the fixture."""

from __future__ import annotations

import asyncio
import json

import pytest
from vllm_srun.families.vela3_d3.family import (
    GOLDEN_QUESTIONS,
    GOLDEN_STATE,
)
from vllm_srun.plugins.decisions import well_formed
from vllm_srun.registry import builtin
from vllm_srun.registry.tables.common import REGISTRY_DIR
from vllm_srun.testing import vela3_d3 as fixture

from .conftest import start_runtime

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

CHOICES = {
    qid: set(q["criteria"])
    for qid, q in GOLDEN_QUESTIONS.items()
    if q["type"] == "choice"
}


def test_recorded_references_are_for_the_pinned_revisions() -> None:
    path = REGISTRY_DIR / "golden_answers_vela3_d3.json"
    recorded = json.loads(path.read_text(encoding="utf-8"))
    for model in builtin.all_models("vela3_d3"):
        assert recorded[model.repo_id]["revision"] == model.revision, model.repo_id


def test_every_size_has_a_rocm_reference_for_the_golden_request() -> None:
    for model in builtin.all_models("vela3_d3"):
        answers = model.golden_answers.get("rocm")
        assert answers, model.repo_id
        assert set(answers) == set(GOLDEN_QUESTIONS), model.repo_id
        assert all(well_formed(a) for a in answers.values()), model.repo_id
        for qid, criteria in CHOICES.items():
            assert set(answers[qid]["probabilities"]) == criteria, (model.repo_id, qid)


def test_the_golden_request_answers_and_its_span_question_takes_the_pii_label_set(
    tmp_path,
) -> None:
    written = fixture.write_fixture(tmp_path / "vela3-fixture", "text", 7)
    base = written.parent / f"{written.name}-base"
    runtime = start_runtime(written, base_path=str(base))
    try:
        body = {"state": GOLDEN_STATE, "questions": GOLDEN_QUESTIONS}
        status, response = asyncio.run(runtime.call("decisions", body))
        assert status == 200
        assert set(response["answers"]) == set(GOLDEN_QUESTIONS)
        assert all(well_formed(a) for a in response["answers"].values())
        assert response["thresholds"]["contact"] == 0.6
    finally:
        runtime.stop()
