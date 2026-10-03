"""KMeans held-out evaluation: fit on train, score on test against the baselines and the oracle."""

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

SERVICE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE_DIR))

from data_loader import RoutingRecord  # noqa: E402
from evaluation import ORACLE  # noqa: E402
from kmeans_evaluation import (  # noqa: E402
    evaluate_kmeans,
    held_out_kmeans_report,
    report_to_dict,
)
from objective import SelectorObjective  # noqa: E402
from query_outcome_set import CandidateOutcome, QueryOutcomeSet  # noqa: E402

OBJECTIVE = SelectorObjective(quality_weight=1.0, latency_weight=0.0, cost_weight=0.0)
GOOD, BAD = 0.9, 0.1


def _records(n_per_category=40):
    """Two well-separated categories: model_a wins math, model_b wins physics."""
    rng = np.random.default_rng(0)
    groups = (("math", 5.0, "model_a"), ("physics", -5.0, "model_b"))
    queries = [
        (category, f"{category} question {i}", centre, winner)
        for category, centre, winner in groups
        for i in range(n_per_category)
    ]
    embeddings = {
        query: centre + rng.normal(0, 0.1, size=4) for _, query, centre, _ in queries
    }
    records = [
        RoutingRecord(
            query=query,
            category=category,
            model_name=model,
            quality=GOOD if model == winner else BAD,
            latency_ms=100.0,
        )
        for category, query, _, winner in queries
        for model in ("model_a", "model_b")
    ]
    return records, embeddings


def test_kmeans_beats_the_global_baselines_on_separable_data():
    records, embeddings = _records()

    reports = held_out_kmeans_report(
        records, embeddings, OBJECTIVE, source="bench", n_clusters=2
    )

    assert set(reports) == {
        "kmeans",
        "strongest",
        "cheapest",
        "global_best",
        "random",
        ORACLE,
    }
    kmeans = reports["kmeans"].overall
    assert kmeans.coverage == 1.0
    assert kmeans.mean_regret == pytest.approx(0.0)
    assert kmeans.mean_regret < reports["global_best"].overall.mean_regret
    assert kmeans.mean_regret < reports["random"].overall.mean_regret
    assert reports[ORACLE].overall.mean_regret == 0.0


def test_the_report_is_deterministic_and_serializable():
    records, embeddings = _records()

    first = report_to_dict(
        held_out_kmeans_report(
            records, embeddings, OBJECTIVE, source="bench", n_clusters=2
        )
    )
    second = report_to_dict(
        held_out_kmeans_report(
            records, embeddings, OBJECTIVE, source="bench", n_clusters=2
        )
    )

    assert first == second
    assert json.loads(json.dumps(first)) == first


def _snapshot(query, *refs):
    return QueryOutcomeSet(
        query=query,
        source="bench",
        category="math",
        outcomes=tuple(
            CandidateOutcome(model_ref=r, success=True, quality=GOOD, latency_ms=100.0)
            for r in refs
        ),
    )


def test_a_test_query_with_only_unknown_candidates_is_an_abstention():
    train = [_snapshot("t1", "a", "b"), _snapshot("t2", "a", "b")]
    test = [_snapshot("q1", "a", "b"), _snapshot("q2", "z")]
    features = {s.query_id: np.array([1.0, 0.0]) for s in (*train, *test)}

    reports = evaluate_kmeans(train, test, features, OBJECTIVE, n_clusters=1)

    assert reports["kmeans"].overall.queries == 2
    assert reports["kmeans"].overall.answered == 1
    assert reports["kmeans"].overall.coverage == 0.5


def test_a_missing_feature_vector_is_an_error_not_a_silent_skip():
    train = [_snapshot("t1", "a", "b")]
    test = [_snapshot("q1", "a", "b")]
    features = {train[0].query_id: np.array([1.0, 0.0])}

    with pytest.raises(ValueError, match="no feature vector"):
        evaluate_kmeans(train, test, features, OBJECTIVE, n_clusters=1)


def test_a_query_without_an_embedding_is_reported():
    records, embeddings = _records()
    del embeddings["math question 3"]

    with pytest.raises(ValueError, match="no embedding"):
        held_out_kmeans_report(records, embeddings, OBJECTIVE, source="bench")


def test_duplicate_test_query_text_is_rejected():
    train = [_snapshot("t1", "a", "b")]
    first = _snapshot("same", "a", "b")
    second = QueryOutcomeSet(
        query="same", source="other", category="math", outcomes=first.outcomes
    )
    features = {s.query_id: np.array([1.0, 0.0]) for s in (*train, first, second)}

    with pytest.raises(ValueError, match="unique"):
        evaluate_kmeans(train, [first, second], features, OBJECTIVE, n_clusters=1)


def _write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(
                json.dumps(
                    {
                        "query": r.query,
                        "category": r.category,
                        "model_name": r.model_name,
                        "performance": r.quality,
                        "response_time": r.latency_ms,
                    }
                )
                + "\n"
            )


def _run_pipeline(tmp_path, monkeypatch, **options):
    records, embeddings = _records()
    data_file = tmp_path / "data.jsonl"
    _write_jsonl(data_file, records)

    # The real embedding module needs a transformer stack; the pipeline only calls this one function.
    fake_embeddings = types.ModuleType("embeddings")
    fake_embeddings.generate_embeddings_for_queries = lambda *a, **k: embeddings
    monkeypatch.setitem(sys.modules, "embeddings", fake_embeddings)
    monkeypatch.delitem(sys.modules, "train", raising=False)
    import train

    return train.run_training_pipeline(
        str(data_file),
        str(tmp_path / "out"),
        cache_dir=str(tmp_path / "cache"),
        algorithm="kmeans",
        kmeans_clusters=2,
        **options,
    )


def test_the_pipeline_writes_the_held_out_report_when_asked(tmp_path, monkeypatch):
    files = _run_pipeline(tmp_path, monkeypatch, evaluate_held_out=True)

    names = sorted(Path(f).name for f in files)
    assert names == ["kmeans_evaluation.json", "kmeans_model.json"]
    report = json.loads((tmp_path / "out" / "kmeans_evaluation.json").read_text())
    assert set(report) == {
        "kmeans",
        "strongest",
        "cheapest",
        "global_best",
        "random",
        ORACLE,
    }
    assert (
        report["kmeans"]["overall"]["mean_regret"]
        <= report["random"]["overall"]["mean_regret"]
    )


def test_the_pipeline_skips_the_report_by_default(tmp_path, monkeypatch):
    files = _run_pipeline(tmp_path, monkeypatch)

    assert [Path(f).name for f in files] == ["kmeans_model.json"]
    assert not (tmp_path / "out" / "kmeans_evaluation.json").exists()
