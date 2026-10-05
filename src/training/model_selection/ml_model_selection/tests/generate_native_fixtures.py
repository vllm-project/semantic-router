"""Regenerate the small sklearn training fixtures consumed by cargo test."""

import copy
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import sklearn

TRAINING = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING))
from models import (  # noqa: E402
    KMeansArtifactError,
    KMeansModel,
    KNNModel,
    NoEligibleCandidateError,
    SVMModel,
    TrainingSample,
)

FIXTURES = TRAINING.parents[3] / "ml-binding/tests/fixtures"
BLOB_MIN_SUPPORT = 3
BLOB_DROPOUT = 0.9


def main():
    rng = np.random.default_rng(21)
    samples = [
        TrainingSample(x, f"model-{i % 3}", 0.7 + (i % 3) / 10, 100 + i * 27)
        for i, x in enumerate(rng.normal(size=(18, 3)))
    ]
    queries = [*rng.normal(size=(24, 3)).tolist(), [0.0, 0.0, 0.0]]
    cases = []
    with tempfile.TemporaryDirectory() as temp:
        for name, model in [
            ("linear", SVMModel(kernel="linear", gamma=0.7)),
            ("rbf", SVMModel(kernel="rbf", gamma=0.7)),
            ("knn", KNNModel(k=4)),
        ]:
            model.train(samples)
            path = Path(temp) / f"{name}.json"
            model.save(path)
            cases.append(
                {
                    "name": name,
                    "artifact": json.loads(path.read_text()),
                    "queries": queries,
                    "expected": [model.predict(np.array(q)) for q in queries],
                }
            )
    FIXTURES.mkdir(parents=True, exist_ok=True)
    (FIXTURES / "python_selectors.json").write_text(
        json.dumps({"sklearn_version": sklearn.__version__, "cases": cases}, indent=2)
        + "\n"
    )
    (FIXTURES / "python_kmeans.json").write_text(
        json.dumps(kmeans_fixtures(), indent=2, allow_nan=False) + "\n"
    )


def kmeans_decision(model, vector, candidates):
    """Expected native output for one query, or the typed rejection it must raise."""
    entry = {"vector": list(vector), "candidates": candidates}
    try:
        decision = model.score(np.asarray(vector), candidates)
    except NoEligibleCandidateError:
        return {**entry, "error": "no_eligible_candidate"}
    except ValueError:
        return {**entry, "error": "invalid_query"}
    return {
        **entry,
        "cluster_id": decision.cluster_id,
        "scores": decision.scores,
        "model": decision.model,
    }


def _kmeans_training_sets():
    rng = np.random.default_rng(33)
    centers = np.array([[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]])
    blobs = []
    for i in range(45):
        x = centers[i % 3] + rng.normal(scale=0.4, size=3)
        for j in range(4):
            # Candidates drop out at random, leaving sparse cells below min_support.
            if j != i % 3 and rng.random() < BLOB_DROPOUT:
                continue
            quality = 0.9 if j == i % 3 else 0.3 + 0.1 * j
            blobs.append(
                TrainingSample(x, f"model-{j}", quality, 50.0 + 400 * j, f"b{i}")
            )
    blobs += blobs[:6]
    few = [
        TrainingSample(np.array(x), name, q, 100.0, f"f{i}")
        for i, x in enumerate([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        for name, q in (("a", 0.2 + 0.3 * i), ("b", 0.8 - 0.3 * i))
    ]
    mirror = [
        TrainingSample(np.array(x), name, q, 100.0, f"m{i}")
        for i, x in enumerate([[-1.0, 0.0], [1.0, 0.0]] * 3)
        for name, q in (("left", 0.9 - 0.8 * (i % 2)), ("right", 0.1 + 0.8 * (i % 2)))
    ]
    return [
        ("blobs", KMeansModel(n_clusters=3, min_support=BLOB_MIN_SUPPORT), blobs),
        ("k_above_unique", KMeansModel(n_clusters=8), few + few),
        ("equidistant", KMeansModel(n_clusters=2), mirror),
    ]


def kmeans_fixtures():
    """Query-level cases for the native v2 loader: cluster, scores, selection, rejects."""
    rng = np.random.default_rng(34)
    cases = []
    rejects = []
    for name, model, samples in _kmeans_training_sets():
        model.train(samples)
        artifact = json.loads(json.dumps(model.to_artifact(), allow_nan=False))
        dim = model.feature_dim
        vectors = [
            *rng.normal(scale=3.0, size=(8, dim)).tolist(),
            *model.centroids.tolist(),
            [0.0] * dim,
            *[s.feature_vector.tolist() for s in samples[:4]],
        ]
        if name == "equidistant":
            vectors += [[0.0, 0.5], [0.0, -3.0]]
        queries = []
        for vector in vectors:
            top = model.predict(np.asarray(vector))
            others = [n for n in model.model_names if n != top]
            for candidates in (None, others, [*others[:1], "not-trained"]):
                queries.append(kmeans_decision(model, vector, candidates))
        queries.append(kmeans_decision(model, vectors[0], ["not-trained"]))
        queries.append(kmeans_decision(model, [0.0] * (dim + 1), None))
        # JSON cannot carry NaN, so the native suite builds its non-finite query itself.
        cases.append({"name": name, "artifact": artifact, "queries": queries})
        if name == "blobs":
            rejects += _kmeans_rejects(artifact)
    return {
        "sklearn_version": sklearn.__version__,
        "cases": cases,
        "artifact_rejects": rejects,
    }


def _kmeans_rejects(artifact):
    def mutate(reason, change):
        broken = copy.deepcopy(artifact)
        change(broken)
        try:
            KMeansModel.from_artifact(broken)
        except KMeansArtifactError:
            return {"reason": reason, "artifact": broken}
        raise AssertionError(f"{reason} was accepted by the Python loader")

    def fallback_cell(broken):
        sparse = np.argwhere(np.asarray(broken["support"]) < BLOB_MIN_SUPPORT)[0]
        broken["scores"][sparse[0]][sparse[1]] += 0.5

    def wrong_winner(broken):
        models = broken["candidate_set"]["models"]
        first = broken["cluster_models"][0]
        broken["cluster_models"][0] = next(m for m in models if m != first)

    return [
        mutate("format_version", lambda a: a.update(format_version=3)),
        mutate(
            "feature_dim", lambda a: a["feature"].update(dim=a["feature"]["dim"] + 1)
        ),
        mutate("candidate_set_id", lambda a: a["candidate_set"].update(id="0" * 16)),
        mutate("objective_id", lambda a: a["objective"].update(id="0" * 16)),
        mutate("cluster_models", wrong_winner),
        mutate("fallback_cell", fallback_cell),
        mutate("distance", lambda a: a.update(distance="cosine")),
    ]


if __name__ == "__main__":
    main()
