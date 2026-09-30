"""Regenerate the small sklearn training fixtures consumed by cargo test."""

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import sklearn

TRAINING = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING))
from models import KNNModel, SVMModel, TrainingSample  # noqa: E402


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
    destination = (
        TRAINING.parents[3] / "ml-binding/tests/fixtures/python_selectors.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps({"sklearn_version": sklearn.__version__, "cases": cases}, indent=2)
        + "\n"
    )


if __name__ == "__main__":
    main()
