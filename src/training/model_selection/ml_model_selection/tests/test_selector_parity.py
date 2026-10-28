"""Train real sklearn selectors, export them, and replay them through the router.

The router's pkg/modelselection loads the exported artifacts and selects with
the query embedding plus its category one-hot, as it does in production. Its
choices must equal the trained Python model's on the same queries.

Run from the repository root with numpy, scikit-learn, pytest and Go:
python -m pytest src/training/model_selection/ml_model_selection/tests/test_selector_parity.py
No model downloads or GPU are needed.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

TRAINING = Path(__file__).resolve().parents[1]
ROUTER = TRAINING.parents[2] / "semantic-router"
sys.path.insert(0, str(TRAINING))
from data_loader import CATEGORIES, category_to_onehot  # noqa: E402
from models import KNNModel, SVMModel, TrainingSample  # noqa: E402


@pytest.fixture(scope="session")
def router(tmp_path_factory):
    """Replay queries through the router's selectors with the parity helper."""
    binary = tmp_path_factory.mktemp("selectorparity") / "selectorparity"
    subprocess.run(
        [
            "go",
            "build",
            "-o",
            str(binary),
            str(TRAINING / "selectorparity" / "main.go"),
        ],
        cwd=ROUTER,
        check=True,
    )

    def replay(algorithm, artifact, candidates, queries):
        request = {
            "algorithm": algorithm,
            "artifact": artifact,
            "candidates": sorted(candidates),
            "queries": [
                {"embedding": [float(v) for v in embedding], "category": category}
                for embedding, category in queries
            ],
        }
        result = subprocess.run(
            [str(binary)],
            input=json.dumps(request),
            capture_output=True,
            text=True,
            check=True,
        )
        return json.loads(result.stdout)

    return replay


def features(embedding, category):
    return np.concatenate(
        [np.asarray(embedding, dtype=np.float64), category_to_onehot(category)]
    )


def samples_for_classes(n_classes):
    rng = np.random.default_rng(42)
    # Unequal feature norms expose any extra normalization; overlapping
    # classes require real signed support weights.
    embeddings = rng.normal(size=(90, 4))
    embeddings[:, 0] *= 3
    labels = rng.integers(0, n_classes, 90)
    return [
        (
            TrainingSample(
                features(embedding, CATEGORIES[i % len(CATEGORIES)]),
                f"model-{label}",
                0.3 + (i % 7) / 10,
                100 + 17 * i,
            ),
            (embedding, CATEGORIES[i % len(CATEGORIES)]),
        )
        for i, (embedding, label) in enumerate(zip(embeddings, labels, strict=True))
    ]


def random_queries(seed, count, dims):
    rng = np.random.default_rng(seed)
    return [
        (embedding, CATEGORIES[int(rng.integers(len(CATEGORIES)))])
        for embedding in rng.normal(size=(count, dims))
    ]


def python_predictions(model, queries):
    return [
        model.predict(features(embedding, category)) for embedding, category in queries
    ]


@pytest.mark.parametrize("kernel", ["linear", "rbf"])
@pytest.mark.parametrize("n_classes", [2, 3, 4])
def test_svm_train_export_router_parity(router, tmp_path, kernel, n_classes):
    pairs = samples_for_classes(n_classes)
    model = SVMModel(kernel=kernel, gamma=0.37, C=2.3)
    model.train([sample for sample, _ in pairs])
    path = tmp_path / "svm.json"
    model.save(path)
    artifact = json.loads(path.read_text())
    assert artifact["svc"]["dual_coef"] == model.svm.dual_coef_.tolist()
    assert artifact["svc"]["intercept"] == model.svm.intercept_.tolist()
    queries = [
        *random_queries(7, 400, 4),
        (np.zeros(4), "other"),
        *(query for _, query in pairs),
    ]
    candidates = {sample.model_name for sample, _ in pairs}
    expected = python_predictions(model, queries)
    assert router("svm", artifact, candidates, queries)["selections"] == expected

    restored = SVMModel.load(path)
    assert python_predictions(restored, queries) == expected
    restored.save(tmp_path / "resaved.json")
    resaved = json.loads((tmp_path / "resaved.json").read_text())
    assert router("svm", resaved, candidates, queries)["selections"] == expected

    # Unversioned Python exports carry the exact parameters next to the old
    # approximate per-model classifiers; every loader must use the former.
    legacy = {**artifact, **artifact["svc"]}
    del legacy["svc"]
    del legacy["format_version"]
    legacy["rbf_classifiers"] = [
        {
            "model_name": "wrong",
            "alpha": [1.0],
            "support_vectors": [[1.0]],
            "rho": 0.0,
            "gamma": 1.0,
        }
    ]
    assert router("svm", legacy, candidates, queries)["selections"] == expected
    path.write_text(json.dumps(legacy))
    migrated = SVMModel.load(path)
    assert python_predictions(migrated, queries) == expected
    migrated.save(path)
    canonical = json.loads(path.read_text())
    assert canonical["format_version"] == 2  # noqa: PLR2004 - artifact schema version
    assert "rbf_classifiers" not in canonical
    assert router("svm", canonical, candidates, queries)["selections"] == expected


def test_svm_binary_zero_margin(router, tmp_path):
    model = SVMModel(kernel="linear")
    model.train(
        [
            TrainingSample(features([-1.0], "other"), "a", 1.0, 0.0),
            TrainingSample(features([1.0], "other"), "b", 1.0, 0.0),
        ]
    )
    path = tmp_path / "svm.json"
    model.save(path)
    queries = [([value], "other") for value in (-1.0, 0.0, 1.0)]
    expected = python_predictions(model, queries)
    artifact = json.loads(path.read_text())
    assert router("svm", artifact, {"a", "b"}, queries)["selections"] == expected
    assert python_predictions(SVMModel.load(path), queries) == expected


@pytest.mark.parametrize("k", [1, 2, 7, 200])
def test_knn_train_export_router_parity(router, tmp_path, k):
    pairs = samples_for_classes(4)
    # Repeated feature vectors test neighbor ties; zero quality must not be
    # silently clamped.
    repeated = ([1.0, 0.0, 0.0, 0.0], "math")
    pairs += [
        (TrainingSample(features(*repeated), name, 0.0, 10.0), repeated)
        for name in ["z", "a", "a"]
    ]
    model = KNNModel(k=k)
    model.train([sample for sample, _ in pairs])
    path = tmp_path / "knn.json"
    model.save(path)
    artifact = json.loads(path.read_text())
    queries = [*random_queries(9, 100, 4), *(query for _, query in pairs)]
    candidates = {sample.model_name for sample, _ in pairs}
    expected = python_predictions(model, queries)
    assert router("knn", artifact, candidates, queries)["selections"] == expected
    assert python_predictions(KNNModel.load(path), queries) == expected
    del artifact["format_version"]
    assert router("knn", artifact, candidates, queries)["selections"] == expected


def test_knn_latency_regression(router, tmp_path):
    query = ([1.0, 0.0], "math")
    model = KNNModel(k=2)
    model.train(
        [
            TrainingSample(features(*query), "fast-low-quality", 0.85, 100.0),
            TrainingSample(features(*query), "slow-high-quality", 0.90, 200.0),
        ]
    )
    path = tmp_path / "knn.json"
    model.save(path)
    assert python_predictions(model, [query]) == ["slow-high-quality"]
    artifact = json.loads(path.read_text())
    candidates = {"fast-low-quality", "slow-high-quality"}
    assert router("knn", artifact, candidates, [query])["selections"] == [
        "slow-high-quality"
    ]


@pytest.mark.parametrize("algorithm", ["svm", "knn"])
def test_router_rejects_bad_queries_and_artifacts(router, tmp_path, algorithm):
    pairs = samples_for_classes(3)
    model = SVMModel() if algorithm == "svm" else KNNModel()
    model.train([sample for sample, _ in pairs])
    path = tmp_path / "artifact.json"
    model.save(path)
    artifact = json.loads(path.read_text())
    candidates = {sample.model_name for sample, _ in pairs}
    bad_queries = [([1.0], "math"), ([], "math")]
    assert router(algorithm, artifact, candidates, bad_queries)["selections"] == [
        None,
        None,
    ]
    if algorithm == "svm":
        artifact["svc"]["dual_coef"][0].pop()
    else:
        artifact["labels"].pop()
    assert router(algorithm, artifact, candidates, bad_queries)["error"]
