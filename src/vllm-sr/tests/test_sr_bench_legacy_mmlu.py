import json
import threading
from pathlib import Path
from typing import Any

import pytest
from cli.sr_bench import adapters
from cli.sr_bench.adapters import get_adapter
from cli.sr_bench.contracts import BENCHMARK_WEIGHTS, catalog, load_document
from cli.sr_bench.grading import basic_grade
from cli.sr_bench.legacy_mmlu import ADAPTER, grade_answer, normalize_records
from cli.sr_bench.sources import prepare_dataset


def native_row() -> dict[str, Any]:
    return {
        "question_id": 17,
        "question": "Which choice is correct?",
        "options": ["First choice", "N/A", "Third choice"],
        "answer": "C",
        "category": "science",
        "cot_content": "The reference explanation must not enter the plain prompt.",
    }


def test_adapter_is_separate_from_published_mmlu() -> None:
    assert get_adapter("legacy-mmlu-pro") == ADAPTER
    assert ADAPTER.version == "bench-mmlu-pro-plain-v1"
    assert "legacy-mmlu-pro" in {item["id"] for item in catalog()["benchmarks"]}
    assert "legacy-mmlu-pro" not in BENCHMARK_WEIGHTS
    assert BENCHMARK_WEIGHTS["mmlu-pro"] == 0.10
    assert BENCHMARK_WEIGHTS["gpqa-diamond"] == 0.15
    assert len(BENCHMARK_WEIGHTS) == 9
    assert get_adapter("mmlu-pro").version != ADAPTER.version


def test_source_only_worker_discovers_the_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(adapters, "_adapters", {})
    monkeypatch.setattr(adapters, "_loaded", threading.Event())
    monkeypatch.setattr(adapters, "entry_points", lambda **kwargs: ())
    assert get_adapter("legacy-mmlu-pro") == ADAPTER


def test_normalizer_preserves_legacy_plain_prompt_and_source_identity() -> None:
    case = normalize_records([native_row()])[0]
    assert case == {
        "id": "legacy-mmlu-pro/17",
        "benchmark": "legacy-mmlu-pro",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Question: Which choice is correct?\n\nOptions:\n"
                    "A) First choice\nC) Third choice\n\n"
                    "Instructions:\n"
                    "- Read the question carefully\n"
                    "- Consider each option thoroughly\n"
                    "- Choose the single best answer\n"
                    "- Respond with ONLY the format: Answer: [letter]\n"
                    "- Do not include any other text after your answer\n\n"
                    "Your response:"
                ),
            }
        ],
        "answer": "C",
        "metadata": {
            "source_question_id": "17",
            "stratum": "science",
            "choice_count": 3,
        },
    }


@pytest.mark.parametrize(
    "updates",
    [
        {"question_id": None},
        {"question_id": True},
        {"question": ""},
        {"category": None},
        {"options": []},
        {"options": ["Choice"] * 27},
        {"options": "ABC"},
        {"options": ["First choice", None, "Third choice"]},
        {"answer": 2},
        {"answer": "D"},
        {"answer": "c"},
    ],
)
def test_invalid_native_records_fail_explicitly(updates: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="Legacy MMLU"):
        normalize_records([{**native_row(), **updates}])


def test_duplicate_source_ids_are_rejected() -> None:
    with pytest.raises(ValueError, match="Duplicate"):
        normalize_records([native_row(), native_row()])


@pytest.mark.parametrize(
    "metadata", [None, [], {"choice_count": True}, {"choice_count": 0}]
)
def test_preflight_rejects_invalid_choice_metadata(metadata: Any) -> None:
    assert ADAPTER.preflight is not None
    with pytest.raises(ValueError, match="Legacy MMLU"):
        ADAPTER.preflight({"answer": "A", "metadata": metadata}, {}, {})


@pytest.mark.parametrize(
    ("final", "answer", "correct"),
    [
        ("B", "B", True),
        ("Answer: B.", "B", True),
        ("Answer: b!", "b", False),
        ("Answer: (B).", "(B)", False),
        ("Answer: B\nAnswer: A", "B", True),
        ("Answer: B because it fits.", "B because it fits", False),
        ("The correct answer is B.", "B", True),
        ("Option B is the best choice", "B", True),
        ("B is plausible", "E", False),
        ("**B**", None, False),
        ("A", "A", False),
        ("", None, False),
    ],
)
def test_legacy_extraction_is_versioned(
    final: str, answer: str | None, correct: bool
) -> None:
    result = grade_answer({"answer": "B"}, final)
    assert result["answer"] == answer
    assert result["correct"] is correct
    assert result["score"] == float(correct)
    assert result["details"]["grader_version"] == ADAPTER.version


def test_published_grader_keeps_its_conservative_contract() -> None:
    final = "Answer: B\nAnswer: A"
    assert grade_answer({"answer": "B"}, final)["correct"] is True
    assert (
        basic_grade({"benchmark": "mmlu-pro", "answer": "B"}, final)["correct"] is False
    )


def test_preparation_preserves_disjoint_profiles_and_source_digests(
    tmp_path: Path,
) -> None:
    source = tmp_path / "mmlu.json"
    source.write_text(
        json.dumps(
            [
                {**native_row(), "question_id": i, "category": f"subject-{i % 14}"}
                for i in range(2500)
            ]
        )
    )
    kwargs = {
        "benchmark": ADAPTER.id,
        "store": tmp_path / "store",
        "source_path": source,
        "revision": "fixture-v1",
        "seed": 42,
    }
    prepared = {
        profile: prepare_dataset(profile=profile, **kwargs)
        for profile in ("smoke", "quick", "standard")
    }
    ids = {
        profile: {case["id"] for case in load_document(dataset["path"])}
        for profile, dataset in prepared.items()
    }
    assert len(ids["smoke"]) == 14
    assert len(ids["quick"]) == 500
    assert len(ids["standard"]) == 2000
    assert ids["smoke"] <= ids["quick"]
    assert ids["quick"].isdisjoint(ids["standard"])
    assert prepare_dataset(profile="smoke", **kwargs) == prepared["smoke"]
    assert prepared["smoke"]["sources"][ADAPTER.id]["normalizer"] == ADAPTER.version
    assert prepared["smoke"]["sources"][ADAPTER.id]["revision"] == "fixture-v1"
    assert prepared["smoke"]["sources"][ADAPTER.id]["files"][0]["sha256"]

    limited = prepare_dataset(profile="smoke", limit=2, **kwargs)
    assert limited["case_count"] == 2
    assert limited["custom_subset"] is True
    with pytest.raises(ValueError, match="limit"):
        prepare_dataset(profile="smoke", limit=15, **kwargs)
    with pytest.raises(ValueError, match="revision"):
        prepare_dataset(profile="smoke", **{**kwargs, "revision": None})


def test_preparation_does_not_publish_an_incomplete_profile(tmp_path: Path) -> None:
    source = tmp_path / "short.json"
    source.write_text(json.dumps([native_row()]))
    store = tmp_path / "store"
    with pytest.raises(ValueError, match="insufficient"):
        prepare_dataset(
            benchmark=ADAPTER.id,
            profile="smoke",
            store=store,
            source_path=source,
            revision="fixture-v1",
        )
    assert not (store / "datasets").exists()
