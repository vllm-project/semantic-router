"""Feature provenance and the embeddings cache that depends on it."""

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

SERVICE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE_DIR))

import embeddings  # noqa: E402
from feature_snapshot import (  # noqa: E402
    UNPINNED_REVISION,
    FeatureSnapshotError,
    build_snapshot,
    source_digest,
)

QUERIES = ["a", "bb", "ccc"]
DIM = 2


def _snapshot(**changes):
    return replace(
        build_snapshot(QUERIES, "model-x", DIM, model_revision="r1"), **changes
    )


class FakeGenerator:
    """Deterministic stand-in for the sentence-transformers wrapper."""

    created = 0

    def __init__(self, model_name, cache_dir=None, device="cpu", revision=None):
        type(self).created += 1
        self.dim = DIM
        self.tag = float(len(model_name) + (len(revision) if revision else 0))

    def encode(self, texts, batch_size=32, show_progress=True):
        return np.array([[self.tag, float(len(t))] for t in texts])


@pytest.fixture
def fake_generator(monkeypatch):
    FakeGenerator.created = 0
    monkeypatch.setattr(embeddings, "EmbeddingGenerator", FakeGenerator)
    return FakeGenerator


def test_snapshot_id_changes_with_every_part():
    base = _snapshot()
    for change in (
        {"embedding_model": "model-y"},
        {"model_revision": "r2"},
        {"dimension": DIM + 1},
        {"source_digest": source_digest(["other"])},
        {"preprocessing": "lowercase"},
        {"pooling": "mean"},
        {"normalized": False},
        {"signal_schema": ("latency",)},
    ):
        assert _snapshot(**change).snapshot_id != base.snapshot_id, change
    assert _snapshot().snapshot_id == base.snapshot_id


def test_inputs_id_ignores_only_the_dimension():
    base = _snapshot()
    assert _snapshot(dimension=DIM + 1).inputs_id == base.inputs_id
    assert _snapshot(model_revision="r2").inputs_id != base.inputs_id


def test_source_digest_ignores_order_and_duplicates():
    assert source_digest(["b", "a", "a"]) == source_digest(["a", "b"])
    assert source_digest(["a", "b"]) != source_digest(["a", "c"])


def test_unpinned_revision_is_recorded():
    assert build_snapshot(QUERIES, "model-x", DIM).model_revision == UNPINNED_REVISION


@pytest.mark.parametrize(
    "vectors",
    [
        np.zeros((3, DIM + 1)),
        np.zeros(3),
        np.zeros((2, DIM)),
        np.array([[0.0, np.nan]] * 3),
    ],
    ids=["wrong_dimension", "not_a_matrix", "wrong_count", "non_finite"],
)
def test_check_vectors_fails_deterministically(vectors):
    with pytest.raises(FeatureSnapshotError):
        _snapshot().check_vectors(vectors, len(QUERIES))


def test_cache_serves_the_same_inputs_without_rebuilding(tmp_path, fake_generator):
    first = embeddings.generate_embeddings_for_queries(
        QUERIES, "model-x", embeddings_dir=str(tmp_path)
    )
    again = embeddings.generate_embeddings_for_queries(
        QUERIES, "model-x", embeddings_dir=str(tmp_path)
    )
    assert fake_generator.created == 1
    assert all(np.allclose(first[q], again[q]) for q in QUERIES)


def test_cache_is_not_reused_for_a_changed_query_set(tmp_path, fake_generator):
    embeddings.generate_embeddings_for_queries(
        QUERIES[:2], "model-x", embeddings_dir=str(tmp_path)
    )
    grown = embeddings.generate_embeddings_for_queries(
        QUERIES, "model-x", embeddings_dir=str(tmp_path)
    )
    assert set(grown) == set(QUERIES)
    assert fake_generator.created == 2


def test_cache_is_not_reused_across_model_or_revision(tmp_path, fake_generator):
    base = embeddings.generate_embeddings_for_queries(
        QUERIES, "model-x", embeddings_dir=str(tmp_path)
    )
    other_model = embeddings.generate_embeddings_for_queries(
        QUERIES, "model-yy", embeddings_dir=str(tmp_path)
    )
    other_revision = embeddings.generate_embeddings_for_queries(
        QUERIES, "model-x", embeddings_dir=str(tmp_path), model_revision="r2"
    )
    assert other_model["a"][0] != base["a"][0]
    assert other_revision["a"][0] != base["a"][0]
    assert fake_generator.created == 3


def test_alias_and_model_id_share_one_cache(tmp_path, fake_generator):
    model_id = embeddings.EMBEDDING_MODELS["qwen3"]
    embeddings.generate_embeddings_for_queries(
        QUERIES, "qwen3", embeddings_dir=str(tmp_path)
    )
    embeddings.generate_embeddings_for_queries(
        QUERIES, model_id, embeddings_dir=str(tmp_path)
    )
    assert fake_generator.created == 1


def test_cache_file_holds_no_pickled_objects(tmp_path, fake_generator):
    embeddings.generate_embeddings_for_queries(
        QUERIES, "model-x", embeddings_dir=str(tmp_path)
    )
    (path,) = tmp_path.glob("embeddings_*.npz")
    with np.load(path, allow_pickle=False) as data:
        assert data["vectors"].dtype == np.float32
        assert data["queries"].tolist() == QUERIES


def test_wrong_dimension_from_the_generator_is_rejected(tmp_path, monkeypatch):
    class WrongDim(FakeGenerator):
        def encode(self, texts, batch_size=32, show_progress=True):
            return np.zeros((len(texts), DIM + 1))

    monkeypatch.setattr(embeddings, "EmbeddingGenerator", WrongDim)
    with pytest.raises(FeatureSnapshotError):
        embeddings.generate_embeddings_for_queries(
            QUERIES, "model-x", embeddings_dir=str(tmp_path)
        )
    assert not list(tmp_path.glob("embeddings_*.npz"))
