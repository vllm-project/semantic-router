"""NLI pairs and decisions served over HTTP with tiny, local ModernBERT weights."""

from __future__ import annotations

import http.client
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from starlette.testclient import TestClient
from vllm_srun.api.app import create_app
from vllm_srun.config import ModelConfig, ServeConfig
from vllm_srun.families.task_heads.nli import (
    LOGITS_HEAD,
    answer,
    entailment_index,
    hypotheses,
)
from vllm_srun.heads.task import HeadOptions
from vllm_srun.plugins.base import DEADLINE, SurfaceRequest
from vllm_srun.runtime import Runtime
from vllm_srun.testing.fixtures import write_fixture

from .test_api_contract import check
from .test_server import free_port, request, wait_ready

STATE = "Write a Python function to sort a list."
QUESTION = {
    "type": "choice",
    "instructions": "This request is about {label}.",
    "criteria": {
        "code": "programming",
        "math": "mathematics",
        "travel": "travel planning",
    },
}
QUESTIONS = {
    "domain": QUESTION,
    "code": {"type": "noul", "instructions": "This request is about programming."},
}


@pytest.fixture(scope="module")
def packages(tmp_path_factory):
    root = tmp_path_factory.mktemp("nli")
    return {
        variant: write_fixture(
            root / variant, family="task_heads", variant=variant, seed=2
        )
        for variant in ("nli", "nli_ternary", "sequence")
    }


@pytest.fixture(scope="module")
def runtime(packages):
    runtime = Runtime(
        ServeConfig(
            models=tuple(
                ModelConfig(model=str(path), name=name, device="cpu")
                for name, path in packages.items()
            ),
            threads=1,
        )
    )
    runtime.start(background=False)
    yield runtime
    runtime.stop()


@pytest.fixture(scope="module")
def client(runtime):
    return TestClient(create_app(runtime))


def decisions(client, model="nli", questions=None, state=STATE, path="/v1/decisions"):
    response = client.post(
        path, json={"model": model, "state": state, "questions": questions or QUESTIONS}
    )
    assert response.status_code == 200, response.text
    check("DecisionResponse", response.json())
    return response.json()


def test_capabilities_only_enable_nli_decisions(client):
    body = client.get("/v1/models").json()
    check("ModelList", body)
    cards = {card["id"]: card for card in body["data"]}
    for name in ("nli", "nli_ternary"):
        assert cards[name]["ready"]
        assert cards[name]["surfaces"] == ["classify", "decisions"]
        assert cards[name]["question_types"] == ["choice", "noul"]
        assert cards[name]["heads"][0]["inputs"] == ["text", "pair"]
        assert [head["name"] for head in cards[name]["heads"]] == ["default"]
    assert cards["sequence"]["surfaces"] == ["classify"]
    assert cards["sequence"]["question_types"] == []


@pytest.mark.parametrize(
    "labels,index",
    [
        (["entailment", "not_entailment"], 0),
        (["not_entailment", "entailment"], 1),
        (["neutral", "contradiction", "entailment"], 2),
        (["ENTAILMENT", "CONTRADICTION", "NEUTRAL"], 0),
        (["LABEL_0", "LABEL_1"], None),
        (["entailment", "entailment"], None),
        (["entailment", "safe", "unsafe"], None),
    ],
)
def test_nli_requires_unambiguous_label_metadata(labels, index):
    assert entailment_index(labels) == index


@pytest.mark.reference
@pytest.mark.parametrize("name", ["nli", "nli_ternary"])
def test_pairs_and_decisions_match_transformers(client, packages, name):
    transformers = pytest.importorskip("transformers")
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_file=str(packages[name] / "tokenizer.json"), pad_token="<pad>"
    )
    config = transformers.ModernBertConfig.from_pretrained(str(packages[name]))
    config.reference_compile = False
    reference = transformers.ModernBertForSequenceClassification.from_pretrained(
        str(packages[name]),
        config=config,
        dtype=torch.float32,
        attn_implementation="eager",
    ).eval()
    _kind, keys, pairs = hypotheses(QUESTION)
    encoded = tokenizer(
        [STATE] * len(pairs),
        pairs,
        padding=True,
        return_tensors="pt",
        return_token_type_ids=False,
    )
    with torch.inference_mode():
        logits = reference(**encoded).logits
    positive = entailment_index(list(reference.config.id2label.values()))
    response = client.post(
        "/v1/classify",
        json={
            "model": name,
            "input": [{"text": STATE, "text_pair": value} for value in pairs],
        },
    )
    assert response.status_code == 200
    check("ClassifyResponse", response.json())
    ours = torch.tensor([row["probabilities"] for row in response.json()["results"]])
    torch.testing.assert_close(ours, logits.softmax(-1), rtol=0, atol=1e-6)
    body = decisions(client, name)
    expected = logits[:, positive].softmax(0)
    actual = body["answers"]["domain"]
    assert list(actual["probabilities"]) == keys
    torch.testing.assert_close(
        torch.tensor(list(actual["probabilities"].values())),
        expected,
        rtol=0,
        atol=1e-6,
    )
    assert actual["choice"] == keys[expected.argmax()]
    assert body["answers"]["code"]["noul"] == pytest.approx(
        logits[0].softmax(0)[positive].item(), abs=1e-6
    )
    assert body["usage"]["input_tokens"] > 0 and body["usage"]["output_tokens"] == 0


def test_choice_order_alias_and_cache_preserve_answers(client):
    original = decisions(client)
    assert decisions(client)["answers"] == original["answers"]
    assert decisions(client, path="/v1/systemone")["answers"] == original["answers"]
    reordered = {
        **QUESTION,
        "criteria": dict(reversed(list(QUESTION["criteria"].items()))),
    }
    second = decisions(client, questions={"domain": reordered})["answers"]["domain"]
    assert second["probabilities"] == original["answers"]["domain"]["probabilities"]
    assert second["choice"] == original["answers"]["domain"]["choice"]


def test_bundle_shares_classification_and_decisions(client):
    response = client.post(
        "/v1/bundle",
        json={
            "tasks": [
                {
                    "id": "decide",
                    "decisions": {
                        "model": "nli",
                        "state": STATE,
                        "questions": QUESTIONS,
                    },
                },
                {
                    "id": "classify",
                    "classify": {
                        "model": "nli",
                        "input": [
                            {
                                "text": STATE,
                                "text_pair": "This request is about programming.",
                            }
                        ],
                    },
                },
            ]
        },
    )
    assert response.status_code == 200, response.text
    check("BundleResponse", response.json())
    results = {result["id"]: result for result in response.json()["results"]}
    assert results["decide"]["decisions"]["answers"] == decisions(client)["answers"]
    assert results["classify"]["classify"]["results"][0]["probabilities"][
        0
    ] == pytest.approx(results["decide"]["decisions"]["answers"]["code"]["noul"])


@pytest.mark.parametrize(
    "broken",
    [
        {**QUESTION, "instructions": "Which domain?"},
        {**QUESTION, "instructions": "{label.__class__}"},
        {**QUESTION, "instructions": "{label!r}"},
        {**QUESTION, "instructions": "{label} {other}"},
        {**QUESTION, "instructions": "{"},
        {**QUESTION, "type": "score", "criteria": ["low", "high"]},
        {"type": "set", "instructions": "x"},
        {"type": "noul", "instructions": "x", "criteria": {"false": "custom false"}},
        {"type": "noul", "instructions": "x", "unexpected": True},
    ],
)
def test_bad_question_does_not_fail_siblings(client, broken):
    body = decisions(client, questions={"bad": broken, **QUESTIONS})
    assert body["answers"]["bad"]["error"] == "invalid_question"
    assert "error" not in body["answers"]["domain"]


def test_true_criterion_and_null_choice_description(client):
    qs = {
        "yes": {
            "type": "noul",
            "instructions": "fallback",
            "criteria": {"true": "This request is about programming."},
        },
        "choice": {**QUESTION, "criteria": {"programming": None, "mathematics": None}},
    }
    body = decisions(client, questions=qs)
    assert (
        body["answers"]["yes"]["noul"] == decisions(client)["answers"]["code"]["noul"]
    )
    assert set(body["answers"]["choice"]["probabilities"]) == {
        "programming",
        "mathematics",
    }


def test_pair_budget_keeps_hypothesis_and_rejects_windowing(runtime):
    model = next(s.model for s in runtime.served if s.config.name == "nli")
    head = model.head
    value = {"text": STATE * 50, "text_pair": "This request is about programming."}
    options = HeadOptions(overflow="truncate", max_tokens=48)
    prepared = head.prepare(value, options, model.info.model_sha256)
    ids = list(prepared.items[0].ids)
    hypothesis = head.tokenizer.encode(value["text_pair"], add_special_tokens=False).ids
    assert len(ids) == 48 and ids[-len(hypothesis) - 1 : -1] == hypothesis
    assert prepared.usage["truncated"]
    with pytest.raises(ValueError, match="windows"):
        head.prepare(value, HeadOptions(overflow="window", max_tokens=48), "x")
    response = model.plan_surface(
        "decisions",
        SurfaceRequest(
            "decisions",
            {"state": STATE * 2000, "questions": QUESTIONS},
            None,
            "exact",
            False,
            0,
        ),
    )
    assert (
        model.finish_surface(response, [])["answers"]["domain"]["error"]
        == "max_length_exceeded"
    )


def test_invalid_pairs_and_hidden_head_are_rejected(client):
    for value in (
        {"text": "x", "text_pair": " "},
        {"text_pair": "x"},
        {"text": "x", "text_pair": 1},
    ):
        body = client.post(
            "/v1/classify", json={"model": "nli", "input": [value]}
        ).json()
        assert body["results"] == [{"index": 0, "error": "invalid_input"}]
    assert (
        client.post(
            "/v1/classify", json={"model": "nli", "head": LOGITS_HEAD, "input": "x"}
        ).status_code
        == 400
    )


def test_deadline_and_bad_logits_are_reported_per_question(runtime):
    model = next(s.model for s in runtime.served if s.config.name == "nli")
    plan = model.plan_surface(
        "decisions",
        SurfaceRequest(
            "decisions",
            {"state": STATE, "questions": QUESTIONS},
            None,
            "exact",
            False,
            0,
        ),
    )
    assert all(
        a["error"] == "deadline_exceeded"
        for a in model.finish_surface(plan, DEADLINE)["answers"].values()
    )
    results = [(0.0, 1.0)] * len(plan.items)
    results[0] = DEADLINE
    body = model.finish_surface(plan, results)["answers"]
    assert body["domain"]["error"] == "deadline_exceeded"
    assert "error" not in body["code"]
    assert (
        answer("choice", ["a", "b"], [(float("nan"), 0), (0, 1)], 0, 2)["error"]
        == "invalid_model_output"
    )


def test_runtime_process_exposes_nli_decisions(packages):
    """The runtime executable used inside the router image answers through TCP."""
    root = Path(__file__).resolve().parents[3]
    paths = [str(root / "src/model-runtime")]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(paths), HF_HUB_OFFLINE="1")
    port = free_port()
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "vllm_srun",
            "serve",
            str(packages["nli"]),
            "--device",
            "cpu",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    def connect():
        return http.client.HTTPConnection("127.0.0.1", port, timeout=10)

    try:
        wait_ready(connect, process)
        status, payload = request(
            connect(), "POST", "/v1/decisions", {"state": STATE, "questions": QUESTIONS}
        )
        body = json.loads(payload)
        assert status == 200, body
        check("DecisionResponse", body)
        assert all("error" not in a for a in body["answers"].values())
    finally:
        process.terminate()
        process.wait(timeout=30)


def test_ordered_choices_and_ties_preserve_caller_keys(client):
    question = {
        "type": "choice",
        "instructions": QUESTION["instructions"],
        "choices": [
            {"key": "first", "description": "programming"},
            {"key": "second", "description": "programming"},
        ],
    }
    result = decisions(client, questions={"tie": question})["answers"]["tie"]
    assert result["choice"] == "first"
    assert result["probabilities"] == {"first": 0.5, "second": 0.5}
    assert result["confidence"] == 0.0


def test_non_string_question_type_returns_a_contract_valid_error(client):
    result = decisions(client, questions={"bad": {"type": 7}, **QUESTIONS})["answers"][
        "bad"
    ]
    assert result["type"] is None and result["error"] == "invalid_question"
    assert "type must be" in result["message"]


def test_hypothesis_limit_is_a_request_error(client, monkeypatch):
    from vllm_srun.families.task_heads import nli

    monkeypatch.setattr(nli, "MAX_HYPOTHESES", 3)
    response = client.post(
        "/v1/decisions", json={"model": "nli", "state": STATE, "questions": QUESTIONS}
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_request"


def test_nli_preserves_multi_state_validation(client):
    invalid = {"bad": {**QUESTION, "instructions": "Which domain?"}}
    body = {
        "model": "nli",
        "state": STATE,
        "questions": QUESTIONS,
        "states": {"other": {"state": "Another request.", "questions": invalid}},
    }
    response = client.post("/v1/decisions", json=body)
    assert response.status_code == 200, response.text
    check("DecisionResponse", response.json())
    assert (
        response.json()["states"]["other"]["answers"]["bad"]["error"]
        == "invalid_question"
    )
    refused = client.post("/v1/decisions", json={**body, "questions": invalid})
    assert refused.status_code == 400
    assert "no question is valid" in refused.json()["error"]["message"]


def test_nli_rejects_a_scan_budget(client):
    response = client.post(
        "/v1/decisions",
        json={
            "model": "nli",
            "state": STATE,
            "questions": QUESTIONS,
            "options": {"max_tokens": 64},
        },
    )
    assert response.status_code == 400
    assert "scan budget" in response.json()["error"]["message"]


def test_pair_prefix_matches_whole_tokenization(runtime, monkeypatch):
    from vllm_srun.text import bounds

    model = next(s.model for s in runtime.served if s.config.name == "nli")
    value = {"text": STATE * 3000, "text_pair": "This request is about programming."}
    options = HeadOptions(overflow="truncate", max_tokens=48)
    bounded = model.head.prepare(value, options, model.info.model_sha256)
    assert bounded.usage["tokens_lower_bound"]
    with monkeypatch.context() as patch:
        patch.setattr(bounds, "facts", lambda tokenizer: bounds.WHOLE)
        whole = model.head.prepare(value, options, model.info.model_sha256)
    assert bounded.items[0].ids == whole.items[0].ids
    assert bounded.usage["tokens"] <= whole.usage["tokens"]
    assert bounded.usage["processed_tokens"] == whole.usage["processed_tokens"]
