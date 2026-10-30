#!/usr/bin/env python3
"""Content-addressed description of how selector query features were produced.

A feature vector is only comparable to another when the embedding model and
revision, preprocessing, pooling, normalization, dimension, extra-signal schema,
and source queries all match. FeatureSnapshot names that tuple and hashes it, so
a change to any part is a different snapshot and a cache can never serve vectors
made under another one.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass

import numpy as np

FEATURE_SNAPSHOT_VERSION = 1

# A model revision that was not pinned is recorded as such: it cannot detect an upstream change.
UNPINNED_REVISION = "unpinned"

_DIGEST_CHARS = 16


class FeatureSnapshotError(ValueError):
    """Features do not match the snapshot they were loaded or built under."""


def _digest(payload: str) -> str:
    return hashlib.blake2b(payload.encode("utf-8"), digest_size=16).hexdigest()[
        :_DIGEST_CHARS
    ]


def source_digest(queries: Iterable[str]) -> str:
    """Order-independent identity of a set of query texts."""
    h = hashlib.blake2b(digest_size=16)
    for query in sorted(set(queries)):
        h.update(query.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:_DIGEST_CHARS]


@dataclass(frozen=True)
class FeatureSnapshot:
    embedding_model: str
    model_revision: str
    dimension: int
    source_digest: str
    preprocessing: str = "none"
    pooling: str = "model_default"
    normalized: bool = True
    signal_schema: tuple[str, ...] = ()
    version: int = FEATURE_SNAPSHOT_VERSION

    def _canonical(self, *, with_dimension: bool) -> str:
        fields = asdict(self)
        if not with_dimension:
            del fields["dimension"]
        return json.dumps(fields, sort_keys=True, separators=(",", ":"))

    @property
    def snapshot_id(self) -> str:
        """Identity of the features, including the dimension they came out with."""
        return _digest(self._canonical(with_dimension=True))

    @property
    def inputs_id(self) -> str:
        """Identity of what the features are computed from; known before any are computed."""
        return _digest(self._canonical(with_dimension=False))

    def check_vectors(self, vectors: np.ndarray, count: int | None = None) -> None:
        """Fail on a wrong dimension, row count, or non-finite value."""
        if vectors.ndim != 2 or vectors.shape[1] != self.dimension:
            raise FeatureSnapshotError(
                f"expected a matrix with dimension {self.dimension}, got shape {vectors.shape}"
            )
        if count is not None and vectors.shape[0] != count:
            raise FeatureSnapshotError(
                f"expected {count} vectors, got {vectors.shape[0]}"
            )
        if not np.isfinite(vectors).all():
            raise FeatureSnapshotError("feature vectors contain a non-finite value")


def build_snapshot(
    queries: Sequence[str],
    embedding_model: str,
    dimension: int,
    *,
    model_revision: str | None = None,
    signal_schema: Iterable[str] = (),
) -> FeatureSnapshot:
    return FeatureSnapshot(
        embedding_model=embedding_model,
        model_revision=model_revision or UNPINNED_REVISION,
        dimension=dimension,
        source_digest=source_digest(queries),
        signal_schema=tuple(sorted(signal_schema)),
    )


def cache_key(
    queries: Sequence[str], embedding_model: str, model_revision: str | None = None
) -> str:
    """Cache identity for embeddings of `queries`; the dimension is not part of it."""
    return build_snapshot(
        queries, embedding_model, 0, model_revision=model_revision
    ).inputs_id
