import json
import sys
from pathlib import Path

import numpy as np
import pytest

TRAINING = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING))
from models import HierShrinkModel, TrainingSample  # noqa: E402

COST_WEIGHT = 0.3
LATENCY_MS = 12.5


def test_hiershrink_exports_centroids_and_observations(tmp_path):
    rng = np.random.default_rng(7)
    queries = rng.normal(size=(6, 3))
    samples = [
        TrainingSample(q, model, quality, LATENCY_MS)
        for q in queries
        for model, quality in (("model-a", 1.0), ("model-b", 0.0))
    ]
    model = HierShrinkModel(
        coarse_clusters=2, fine_clusters=10, cost_weight=COST_WEIGHT
    )
    model.train(samples)
    path = tmp_path / "hiershrink_model.json"
    model.save(str(path))

    artifact = json.loads(path.read_text())
    assert artifact["algorithm"] == "hiershrink"
    assert artifact["cost_weight"] == COST_WEIGHT
    assert np.asarray(artifact["coarse_centroids"]).shape == (2, 3)
    assert np.asarray(artifact["fine_centroids"]).shape == (6, 3)
    assert len(artifact["training"]) == len(samples)
    record = artifact["training"][1]
    assert record["selected_model"] == "model-b"
    assert record["response_quality"] == 0.0
    assert record["response_latency_ns"] == int(LATENCY_MS * 1_000_000)
    assert record["query_embedding"] == queries[0].tolist()


def test_hiershrink_rejects_negative_cost_weight():
    with pytest.raises(ValueError):
        HierShrinkModel(cost_weight=-1).train(
            [TrainingSample(np.zeros(2), "model-a", 1.0, 1.0)]
        )
