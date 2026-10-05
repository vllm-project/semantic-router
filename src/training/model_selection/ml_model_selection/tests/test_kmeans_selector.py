"""KMeans selector: query-level fit, mean scores, sparse fallback and the v2 contract."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

SERVICE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE_DIR))

from generate_native_fixtures import kmeans_decision, kmeans_fixtures  # noqa: E402
from models import (  # noqa: E402
    KMeansArtifactError,
    KMeansModel,
    NoEligibleCandidateError,
    TrainingSample,
    assign_clusters,
    drop_empty_clusters,
)
from objective import SelectorObjective  # noqa: E402
from query_outcome_set import CandidateOutcome, QueryOutcomeSet  # noqa: E402

FAST_MS = 100.0
REQUESTED_K = 8
DISTINCT_POINTS = 3
MIN_SUPPORT = 2
OTHER_SEED = 7


def _blob_samples(seed=5, n_queries=60, dim=4):
    rng = np.random.default_rng(seed)
    centers = rng.normal(scale=6.0, size=(3, dim))
    samples = []
    for i in range(n_queries):
        x = centers[i % 3] + rng.normal(scale=0.3, size=dim)
        for j, name in enumerate(("m-a", "m-b", "m-c")):
            quality = 0.9 if j == i % 3 else float(rng.uniform(0.2, 0.6))
            samples.append(TrainingSample(x, name, quality, FAST_MS * (j + 1), f"q{i}"))
    return samples


def _trained(samples, **kwargs):
    model = KMeansModel(**{"n_clusters": 3, **kwargs})
    model.train(samples)
    return model


def _artifact(model):
    return json.loads(json.dumps(model.to_artifact(), allow_nan=False))


def test_repeated_query_does_not_change_the_weights():
    samples = _blob_samples()
    base = _artifact(_trained(samples))
    # The repeat carries a different outcome; the first observation must win.
    repeat = [
        TrainingSample(s.feature_vector, s.model_name, 0.0, s.latency_ms, s.query_id)
        for s in samples[:30]
    ]
    assert _artifact(_trained(samples + repeat)) == base


def test_feature_hash_dedups_when_no_query_id_is_given():
    samples = _blob_samples()
    anonymous = [
        TrainingSample(s.feature_vector, s.model_name, s.quality, s.latency_ms)
        for s in samples
    ]
    model = _trained(anonymous + anonymous[:9])
    assert model.n_train_queries == len(samples) // 3
    assert int(model.support.sum()) == len(samples)


def test_mean_beats_a_sum_of_many_weaker_observations():
    """The old trainer summed scores, so the better-covered model won the cluster."""
    x = np.array([1.0, 0.0])
    samples = [
        TrainingSample(x, "many-mediocre", 0.6, FAST_MS, f"q{i}") for i in range(10)
    ]
    samples += [
        TrainingSample(x, "few-strong", 0.9, FAST_MS, f"q{i}") for i in range(2)
    ]
    model = _trained(samples, n_clusters=1)
    assert model.support[0].tolist() == [2, 10]
    assert model.predict(x) == "few-strong"
    assert model.cluster_models == ["few-strong"]


def test_effective_k_is_capped_by_distinct_train_features():
    points = [[0.0, 0.0], [5.0, 0.0], [0.0, 5.0]]
    # Two query ids share each point, so six queries but three distinct features.
    samples = [
        TrainingSample(np.array(points[i % 3]), name, 0.5, FAST_MS, f"q{i}")
        for i in range(6)
        for name in ("a", "b")
    ]
    model = _trained(samples, n_clusters=REQUESTED_K)
    artifact = model.to_artifact()
    assert model.effective_k == DISTINCT_POINTS
    assert artifact["clustering"]["requested_k"] == REQUESTED_K
    assert artifact["clustering"]["effective_k"] == DISTINCT_POINTS
    assert artifact["num_clusters"] == DISTINCT_POINTS
    assert len(artifact["cluster_models"]) == DISTINCT_POINTS
    assert artifact["cluster_sizes"] == [2, 2, 2]


def test_empty_clusters_are_dropped_and_reindexed():
    points = np.array([[0.0, 0.0], [0.1, 0.0], [10.0, 0.0]])
    # Centroid 1 duplicates centroid 0 and loses every tie; centroid 3 is unreachable.
    centroids = np.array([[0.0, 0.0], [0.0, 0.0], [10.0, 0.0], [99.0, 99.0]])
    labels = assign_clusters(points, centroids)
    assert labels.tolist() == [0, 0, 2]
    kept, relabeled, sizes = drop_empty_clusters(centroids, labels)
    assert kept.tolist() == [[0.0, 0.0], [10.0, 0.0]]
    assert relabeled.tolist() == [0, 0, 1]
    assert sizes.tolist() == [2, 1]
    assert assign_clusters(points, kept).tolist() == relabeled.tolist()


def test_sparse_cells_fall_back_to_the_global_train_mean():
    near, far = np.array([0.0, 0.0]), np.array([20.0, 0.0])
    samples = [TrainingSample(near, "dense", 0.5, FAST_MS, f"n{i}") for i in range(4)]
    samples += [TrainingSample(far, "dense", 0.5, FAST_MS, f"f{i}") for i in range(4)]
    # "rare" has one observation near and three far: the near cell is below min_support.
    samples.append(TrainingSample(near, "rare", 1.0, FAST_MS, "n0"))
    samples += [TrainingSample(far, "rare", 0.1, FAST_MS, f"f{i}") for i in range(3)]
    model = _trained(samples, n_clusters=2, min_support=MIN_SUPPORT)
    rare = model.model_names.index("rare")
    global_rare = model.global_scores[rare]
    near_cluster, far_cluster = model.cluster_of(near), model.cluster_of(far)
    assert model.support[near_cluster, rare] == 1
    assert model.scores[near_cluster, rare] == global_rare
    assert model.scores[far_cluster, rare] < global_rare
    assert model.to_artifact()["fallback"]["min_support"] == MIN_SUPPORT


def test_only_requested_candidates_are_eligible():
    model = _trained(_blob_samples())
    query = model.centroids[0]
    full = model.score(query)
    others = [n for n in model.model_names if n != full.model]
    reduced = model.score(query, [*others, "never-trained"])
    assert reduced.model != full.model
    assert reduced.model == max(others, key=lambda n: full.scores[n])
    assert set(reduced.scores) == set(others)
    assert reduced.cluster_id == full.cluster_id
    with pytest.raises(NoEligibleCandidateError):
        model.score(query, ["never-trained"])
    with pytest.raises(NoEligibleCandidateError):
        model.score(query, [])


def test_candidate_ties_go_to_the_lowest_index():
    x = np.array([1.0])
    samples = [TrainingSample(x, name, 0.5, FAST_MS, "q") for name in ("b", "a")]
    model = _trained(samples, n_clusters=1)
    assert model.predict(x) == "a"
    assert model.predict(x, ["b"]) == "b"


def test_same_seed_is_deterministic_and_order_independent():
    samples = _blob_samples()
    first = _artifact(_trained(samples))
    assert _artifact(_trained(samples)) == first
    shuffled = [samples[i] for i in np.random.default_rng(1).permutation(len(samples))]
    assert _artifact(_trained(shuffled)) == first
    reseeded = _artifact(_trained(samples, seed=OTHER_SEED))
    assert reseeded["clustering"]["seed"] == OTHER_SEED


def test_snapshots_fit_under_the_objective_with_failures_at_the_floor():
    rng = np.random.default_rng(2)
    snapshots, features = [], {}
    for i in range(12):
        snap = QueryOutcomeSet(
            query=f"query {i}",
            source="bench",
            category="math",
            outcomes=(
                CandidateOutcome(
                    "strong", success=i % 4 != 0, quality=0.9, latency_ms=FAST_MS
                ),
                CandidateOutcome("weak", success=True, quality=0.5, latency_ms=FAST_MS),
            ),
        )
        snapshots.append(snap)
        features[snap.query_id] = rng.normal(size=3)
    objective = SelectorObjective()
    model = KMeansModel(n_clusters=1, objective=objective)
    model.train_snapshots(snapshots, features, training_snapshot_id="train-v1")
    strong = model.model_names.index("strong")
    success = objective.score(snapshots[1].outcomes[0])
    assert model.global_scores[strong] == pytest.approx(success * 9 / 12)
    artifact = model.to_artifact()
    assert artifact["training_snapshot_id"] == "train-v1"
    assert artifact["objective"]["id"] == objective.objective_id
    assert artifact["candidate_set"]["id"] == snapshots[0].candidate_set_id
    with pytest.raises(ValueError, match="no feature vector"):
        KMeansModel().train_snapshots(snapshots, {})


def test_training_rejects_ambiguous_query_identity():
    x, y = np.array([1.0, 0.0]), np.array([0.0, 1.0])
    with pytest.raises(ValueError, match="more than one feature"):
        KMeansModel().train(
            [
                TrainingSample(x, "a", 0.5, FAST_MS, "q"),
                TrainingSample(y, "b", 0.5, FAST_MS, "q"),
            ]
        )
    with pytest.raises(ValueError, match="every training sample"):
        KMeansModel().train(
            [
                TrainingSample(x, "a", 0.5, FAST_MS, "q"),
                TrainingSample(y, "b", 0.5, FAST_MS),
            ]
        )
    with pytest.raises(ValueError, match="finite"):
        KMeansModel().train(
            [TrainingSample(np.array([np.nan]), "a", 0.5, FAST_MS, "q")]
        )


def test_v2_round_trip_and_load_time_validation(tmp_path):
    model = _trained(_blob_samples())
    path = tmp_path / "kmeans.json"
    model.save(path)
    restored = KMeansModel.load(
        path,
        candidates=reversed(model.model_names),
        feature_dim=model.feature_dim,
        objective=SelectorObjective(),
    )
    queries = np.random.default_rng(3).normal(scale=6.0, size=(50, model.feature_dim))
    assert [restored.score(q) for q in queries] == [model.score(q) for q in queries]
    assert restored.to_artifact() == _artifact(model)

    with pytest.raises(KMeansArtifactError, match="feature dim"):
        KMeansModel.load(path, feature_dim=model.feature_dim + 1)
    with pytest.raises(KMeansArtifactError, match="candidate"):
        KMeansModel.load(path, candidates=model.model_names[:-1])
    with pytest.raises(KMeansArtifactError, match="objective"):
        KMeansModel.load(path, objective=SelectorObjective(latency_weight=0.5))
    with pytest.raises(ValueError, match="dimension"):
        restored.score(queries[0][:-1])
    with pytest.raises(ValueError, match="finite"):
        restored.score(np.full(model.feature_dim, np.nan))


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("format_version", 3, "format_version"),
        ("format_version", "2", "format_version"),
        ("tie_break", "random", "tie_break"),
        ("target_contract", "selector.model-choice/v0", "target_contract"),
        ("num_clusters", 99, "cluster counts"),
    ],
)
def test_v2_loader_rejects_contract_violations(field, value, reason):
    artifact = _artifact(_trained(_blob_samples()))
    artifact[field] = value
    with pytest.raises(KMeansArtifactError, match=reason):
        KMeansModel.from_artifact(artifact)


def test_v2_loader_rejects_nan_and_shape_errors(tmp_path):
    artifact = _artifact(_trained(_blob_samples()))
    nan_centroid = json.loads(json.dumps(artifact))
    nan_centroid["centroids"][0][0] = float("nan")
    path = tmp_path / "nan.json"
    path.write_text(json.dumps(nan_centroid))
    with pytest.raises(KMeansArtifactError, match="centroids"):
        KMeansModel.load(path)
    short = json.loads(json.dumps(artifact))
    short["scores"].pop()
    with pytest.raises(KMeansArtifactError, match="score tables"):
        KMeansModel.from_artifact(short)
    missing = json.loads(json.dumps(artifact))
    del missing["fallback"]
    with pytest.raises(KMeansArtifactError, match="malformed"):
        KMeansModel.from_artifact(missing)


def test_cluster_models_stay_the_per_cluster_argmax_for_the_current_runtime():
    model = _trained(_blob_samples())
    artifact = model.to_artifact()
    assert artifact["cluster_models"] == [
        artifact["candidate_set"]["models"][int(np.argmax(row))]
        for row in artifact["scores"]
    ]
    assert (
        len(artifact["cluster_models"])
        == artifact["num_clusters"]
        == len(artifact["centroids"])
    )
    for centroid, expected in zip(
        model.centroids, artifact["cluster_models"], strict=True
    ):
        assert model.predict(centroid) == expected


def test_unversioned_artifacts_load_as_one_hot_scores(tmp_path):
    legacy = {
        "algorithm": "kmeans",
        "trained": True,
        "num_clusters": 3,
        "centroids": [[1.0, 0.0], [0.0, 1.0]],
        "cluster_models": ["b", "a", "a"],
        "model_names": ["a", "b", "c"],
        "feature_dim": 2,
    }
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(legacy))
    model = KMeansModel.load(path)
    assert model.format_version == 1
    assert model.predict(np.array([0.9, 0.1])) == "b"
    assert model.score(np.array([0.1, 0.9])).scores == {"a": 1.0, "b": 0.0, "c": 0.0}
    model.save(tmp_path / "resaved.json")
    assert json.loads((tmp_path / "resaved.json").read_text()) == legacy


def test_native_fixtures_match_the_python_scorer():
    fixtures = kmeans_fixtures()
    names = {case["name"] for case in fixtures["cases"]}
    assert names == {"blobs", "k_above_unique", "equidistant"}
    for case in fixtures["cases"]:
        model = KMeansModel.from_artifact(case["artifact"])
        for query in case["queries"]:
            assert kmeans_decision(model, query["vector"], query["candidates"]) == query
        errors = {q.get("error") for q in case["queries"]}
        assert {"no_eligible_candidate", "invalid_query"} <= errors
    equidistant = next(c for c in fixtures["cases"] if c["name"] == "equidistant")
    tie = next(q for q in equidistant["queries"] if q["vector"] == [0.0, 0.5])
    assert tie["cluster_id"] == 0
    for reject in fixtures["artifact_rejects"]:
        with pytest.raises(KMeansArtifactError):
            KMeansModel.from_artifact(reject["artifact"])
