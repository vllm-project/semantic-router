"""Train real sklearn selectors, export them, and exercise the current Rust C ABI.

Run from the repository root with numpy, scikit-learn and pytest installed:
python -m pytest src/training/model_selection/ml_model_selection/tests/test_native_parity.py
Cargo builds the library from this checkout; no model downloads or GPU are needed.
"""

import ctypes
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

TRAINING = Path(__file__).resolve().parents[1]
REPO = TRAINING.parents[3]
sys.path.insert(0, str(TRAINING))
from models import KNNModel, SVMModel, TrainingSample  # noqa: E402


@pytest.fixture(scope="session")
def native():
    target = REPO / "ml-binding" / "target"
    subprocess.run(
        [
            "cargo",
            "build",
            "--release",
            "--locked",
            "--manifest-path",
            str(REPO / "ml-binding/Cargo.toml"),
            "--target-dir",
            str(target),
        ],
        check=True,
    )
    suffix = "dylib" if sys.platform == "darwin" else "so"
    lib = ctypes.CDLL(str(target / "release" / f"libml_semantic_router.{suffix}"))
    for algorithm in ("knn", "svm"):
        load = getattr(lib, f"ml_{algorithm}_from_json")
        load.argtypes = [ctypes.c_char_p]
        load.restype = ctypes.c_void_p
        select = getattr(lib, f"ml_{algorithm}_select")
        select.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_size_t,
        ]
        select.restype = ctypes.c_void_p
        free = getattr(lib, f"ml_{algorithm}_free")
        free.argtypes = [ctypes.c_void_p]
        free.restype = None
    lib.ml_free_string.argtypes = [ctypes.c_void_p]
    lib.ml_free_string.restype = None
    return lib


def native_predictions(native, algorithm, artifact, queries):
    handle = getattr(native, f"ml_{algorithm}_from_json")(json.dumps(artifact).encode())
    assert handle, "The native loader rejected a valid exported artifact"
    try:
        predictions = []
        for query in queries:
            vector = (ctypes.c_double * len(query))(*query)
            result = getattr(native, f"ml_{algorithm}_select")(
                handle, vector, len(query)
            )
            if result:
                predictions.append(ctypes.string_at(result).decode())
                native.ml_free_string(result)
            else:
                predictions.append(None)
        return predictions
    finally:
        getattr(native, f"ml_{algorithm}_free")(handle)


def samples_for_classes(n_classes):
    rng = np.random.default_rng(42)
    # Unequal feature norms and a categorical component expose the old extra
    # normalization. Overlapping classes require real signed support weights.
    features = rng.normal(size=(90, 4))
    features[:, 0] *= 3
    features = np.column_stack([features, np.ones(90), np.zeros((90, 2))])
    labels = rng.integers(0, n_classes, 90)
    return [
        TrainingSample(x, f"model-{label}", 0.3 + (i % 7) / 10, 100 + 17 * i)
        for i, (x, label) in enumerate(zip(features, labels, strict=True))
    ]


@pytest.mark.parametrize("kernel", ["linear", "rbf"])
@pytest.mark.parametrize("n_classes", [2, 3, 4])
def test_svm_train_export_native_parity(native, tmp_path, kernel, n_classes):
    model = SVMModel(kernel=kernel, gamma=0.37, C=2.3)
    model.train(samples_for_classes(n_classes))
    path = tmp_path / "svm.json"
    model.save(path)
    artifact = json.loads(path.read_text())
    assert artifact["svc"]["dual_coef"] == model.svm.dual_coef_.tolist()
    assert artifact["svc"]["intercept"] == model.svm.intercept_.tolist()
    rng = np.random.default_rng(7)
    queries = np.vstack(
        [
            rng.normal(size=(400, 7)),
            np.zeros((1, 7)),
            [s.feature_vector for s in samples_for_classes(n_classes)],
        ]
    )
    expected = [model.predict(query) for query in queries]
    assert native_predictions(native, "svm", artifact, queries) == expected
    restored = SVMModel.load(path)
    assert [restored.predict(query) for query in queries] == expected
    restored.save(tmp_path / "resaved.json")
    assert (
        native_predictions(
            native, "svm", json.loads((tmp_path / "resaved.json").read_text()), queries
        )
        == expected
    )

    # Unversioned Python exports contain exact parameters in addition to the
    # old approximate per-model classifiers. Both loaders must use the former.
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
    path.write_text(json.dumps(legacy))
    assert native_predictions(native, "svm", legacy, queries) == expected
    migrated = SVMModel.load(path)
    assert [migrated.predict(query) for query in queries] == expected
    migrated.save(path)
    canonical = json.loads(path.read_text())
    assert canonical["format_version"] == 2  # noqa: PLR2004 - artifact schema version
    assert "rbf_classifiers" not in canonical
    assert native_predictions(native, "svm", canonical, queries) == expected


def test_svm_binary_zero_margin(native, tmp_path):
    model = SVMModel(kernel="linear")
    model.train(
        [
            TrainingSample(np.array([-1.0]), "a", 1.0, 0.0),
            TrainingSample(np.array([1.0]), "b", 1.0, 0.0),
        ]
    )
    path = tmp_path / "svm.json"
    model.save(path)
    queries = [[-1.0], [0.0], [1.0]]
    expected = [model.predict(q) for q in queries]
    assert (
        native_predictions(native, "svm", json.loads(path.read_text()), queries)
        == expected
    )
    assert [SVMModel.load(path).predict(q) for q in queries] == expected


@pytest.mark.parametrize("k", [1, 2, 7, 200])
def test_knn_train_export_native_parity(native, tmp_path, k):
    samples = samples_for_classes(4)
    # Repeated feature vectors test tree boundary ties; zero quality must not
    # be silently clamped, and zero vectors have a defined normalized distance.
    samples += [
        TrainingSample(np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), name, 0.0, 10.0)
        for name in ["z", "a", "a"]
    ]
    samples.append(TrainingSample(np.zeros(7), "zero", 1.0, 0.0))
    model = KNNModel(k=k)
    model.train(samples)
    path = tmp_path / "knn.json"
    model.save(path)
    artifact = json.loads(path.read_text())
    rng = np.random.default_rng(9)
    queries = np.vstack(
        [rng.normal(size=(100, 7)), [s.feature_vector for s in samples]]
    )
    expected = [model.predict(query) for query in queries]
    assert native_predictions(native, "knn", artifact, queries) == expected
    restored = KNNModel.load(path)
    assert [restored.predict(query) for query in queries] == expected
    del artifact["format_version"]
    assert native_predictions(native, "knn", artifact, queries) == expected


def test_knn_latency_regression(native, tmp_path):
    samples = [
        TrainingSample(np.array([1.0, 0.0]), "fast-low-quality", 0.85, 100.0),
        TrainingSample(np.array([1.0, 0.0]), "slow-high-quality", 0.90, 200.0),
    ]
    model = KNNModel(k=2)
    model.train(samples)
    path = tmp_path / "knn.json"
    model.save(path)
    assert model.predict(np.array([1.0, 0.0])) == "slow-high-quality"
    assert native_predictions(
        native, "knn", json.loads(path.read_text()), [[1.0, 0.0]]
    ) == ["slow-high-quality"]


@pytest.mark.parametrize("algorithm", ["svm", "knn"])
def test_native_rejects_bad_shapes_without_aborting(native, tmp_path, algorithm):
    model = SVMModel() if algorithm == "svm" else KNNModel()
    model.train(samples_for_classes(3))
    path = tmp_path / "artifact.json"
    model.save(path)
    artifact = json.loads(path.read_text())
    assert native_predictions(
        native, algorithm, artifact, [[1.0], [float("nan")] * 7]
    ) == [None, None]
    if algorithm == "svm":
        artifact["svc"]["dual_coef"][0].pop()
    else:
        artifact["labels"].pop()
    assert not getattr(native, f"ml_{algorithm}_from_json")(
        json.dumps(artifact).encode()
    )
