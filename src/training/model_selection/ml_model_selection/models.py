#!/usr/bin/env python3
"""
ML models for model selection.

Implements KNN, KMeans, SVM, and MLP using scikit-learn and PyTorch.
Models are saved in JSON format compatible with the Rust inference code.

Reference:
- FusionFactory (arXiv:2507.10540) - Query-level fusion via LLM routers
- Avengers-Pro (arXiv:2508.12631) - Performance-efficiency optimized routing
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sklearn
from objective import SelectorObjective
from query_outcome_set import CandidateOutcome, QueryOutcomeSet, _digest
from sklearn.cluster import KMeans as SKLearnKMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

# Optional PyTorch import for MLP
try:
    import torch
    from torch import nn, optim
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@dataclass
class TrainingSample:
    """A training sample for model selection."""

    feature_vector: np.ndarray  # 1038-dim (1024 embedding + 14 category one-hot)
    model_name: str
    quality: float
    latency_ms: float
    # Groups a query's rows so query-level selectors fit one point per query.
    query_id: str | None = None


class KNNModel:
    """Quality/speed voting on L2-normalized features, shared with Rust.

    Neighbors sort by squared distance then training index. Equal model vote
    totals select the lexicographically first model name.
    """

    def __init__(self, k: int = 5):
        self.k = k
        self.nn = None
        self.features = None
        self.samples: list[TrainingSample] = []
        self.model_names: list[str] = []

    @staticmethod
    def _normalize(features: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(features, axis=-1, keepdims=True)
        return features / np.where(norms > 0, norms, 1.0)

    def train(self, samples: list[TrainingSample]) -> None:
        """Index the same float64 normalized features used by native inference."""
        if self.k <= 0 or not samples:
            raise ValueError("KNN requires k > 0 and nonempty training samples")
        features = np.asarray([s.feature_vector for s in samples], dtype=np.float64)
        if (
            features.ndim != 2  # noqa: PLR2004 - a feature matrix is two-dimensional
            or features.shape[1] == 0
            or not np.isfinite(features).all()
            or any(
                not np.isfinite(s.quality)
                or not np.isfinite(s.latency_ms)
                or s.latency_ms < 0
                or not s.model_name
                for s in samples
            )
        ):
            raise ValueError(
                "KNN requires finite features, quality and nonnegative latency"
            )
        self.samples = samples
        self.features = self._normalize(features)
        self.nn = NearestNeighbors(
            n_neighbors=min(self.k, len(samples)),
            metric="euclidean",
            algorithm="ball_tree",
        )
        self.nn.fit(self.features)
        self.model_names = sorted({s.model_name for s in samples})
        print(f"KNN trained with {len(samples)} samples, k={self.k}")

    def predict(self, feature_vector: np.ndarray) -> str:
        if self.nn is None:
            raise ValueError("Model not trained")
        query = np.asarray(feature_vector, dtype=np.float64)
        if query.shape != (self.features.shape[1],) or not np.isfinite(query).all():
            raise ValueError("Query must be finite and match the KNN feature dimension")
        query = self._normalize(query)
        distances, _ = self.nn.kneighbors([query])
        # Include boundary ties before ordering, since tree traversal differs
        # between sklearn and Linfa and is not part of the routing contract.
        radius = float(distances[0][-1]) + 1e-12
        indices = self.nn.radius_neighbors(
            [query], radius=radius, return_distance=False
        )[0]
        indices = sorted(
            indices,
            key=lambda i: (float(np.sum((self.features[i] - query) ** 2)), int(i)),
        )[: self.k]
        votes: dict[str, float] = {}
        for idx in indices:
            sample = self.samples[idx]
            # Artifacts store integer nanoseconds; quantize before scoring too.
            latency_ns = int(sample.latency_ms * 1_000_000)
            speed_factor = 1.0 / (1.0 + latency_ns / 10_000_000_000.0)
            weight = 0.9 * sample.quality + 0.1 * speed_factor
            votes[sample.model_name] = votes.get(sample.model_name, 0.0) + weight
        return min(votes, key=lambda name: (-votes[name], name))

    def save(self, path: str) -> None:
        if self.nn is None:
            raise ValueError("Model not trained")
        data = {
            "algorithm": "knn",
            "format_version": 2,
            "trained": True,
            "k": self.k,
            "embeddings": [s.feature_vector.tolist() for s in self.samples],
            "labels": [s.model_name for s in self.samples],
            "qualities": [s.quality for s in self.samples],
            "latencies": [int(s.latency_ms * 1_000_000) for s in self.samples],
            "model_names": self.model_names,
            "num_samples": len(self.samples),
            "feature_dim": self.features.shape[1],
        }
        with open(path, "w") as f:
            json.dump(data, f, allow_nan=False)
        print(f"Saved KNN model to {path}")

    @classmethod
    def load(cls, path: str) -> KNNModel:
        with open(path) as f:
            data = json.load(f)
        if data.get("format_version", 1) not in (1, 2):
            raise ValueError("Unsupported KNN artifact version")
        model = cls(k=data["k"])
        if "samples" in data:
            samples = [
                TrainingSample(
                    np.asarray(s["feature_vector"], dtype=np.float64),
                    s["model_name"],
                    s["quality"],
                    s["latency_ms"],
                )
                for s in data["samples"]
            ]
        else:
            embeddings = data["embeddings"]
            labels = data["labels"]
            qualities = data.get("qualities") or [0.5] * len(labels)
            latencies = data.get("latencies") or [0] * len(labels)
            if not (len(embeddings) == len(labels) == len(qualities) == len(latencies)):
                raise ValueError("KNN sample arrays must have equal lengths")
            samples = [
                TrainingSample(
                    np.asarray(emb, dtype=np.float64),
                    label,
                    quality,
                    latency / 1_000_000,
                )
                for emb, label, quality, latency in zip(
                    embeddings, labels, qualities, latencies, strict=True
                )
            ]
        model.train(samples)
        return model


KMEANS_FORMAT_VERSION = 2
SELECTOR_TARGET_CONTRACT = "selector.model-choice/v1"
KMEANS_DISTANCE = "squared_l2"
KMEANS_TIE_BREAK = "lowest_index"
KMEANS_FEATURE_NORMALIZATION = "none"
# Bounds the (chunk, k, d) difference tensor to about 32 MiB of float64.
_DISTANCE_CHUNK_ELEMENTS = 1 << 22


class KMeansArtifactError(ValueError):
    """A KMeans artifact, or the runtime it is loaded into, breaks the v2 contract."""


class NoEligibleCandidateError(ValueError):
    """No candidate offered by the request is in the trained candidate set."""


@dataclass(frozen=True)
class KMeansDecision:
    """One scored query: nearest cluster, eligible candidate scores, winner."""

    cluster_id: int
    scores: dict[str, float]
    model: str


def squared_l2(points: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """(n, k) squared L2 distances, each summed over dimensions in index order."""
    # np.cumsum adds strictly left to right, an order native code reproduces bit for bit.
    distances = np.empty((points.shape[0], centroids.shape[0]), dtype=np.float64)
    step = max(1, _DISTANCE_CHUNK_ELEMENTS // max(1, centroids.size))
    for start in range(0, points.shape[0], step):
        diff = points[start : start + step, None, :] - centroids[None, :, :]
        distances[start : start + step] = np.cumsum(diff * diff, axis=2)[..., -1]
    return distances


def assign_clusters(points: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Nearest centroid per row; np.argmin keeps the lowest index on ties."""
    return np.argmin(squared_l2(points, centroids), axis=1)


def drop_empty_clusters(
    centroids: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Remove clusters that own no training query, keeping the survivors' order."""
    sizes = np.bincount(labels, minlength=len(centroids))
    keep = np.flatnonzero(sizes)
    remap = np.full(len(centroids), -1, dtype=np.intp)
    remap[keep] = np.arange(len(keep))
    return centroids[keep], remap[labels], sizes[keep]


def _content_id(*arrays: np.ndarray) -> str:
    h = hashlib.blake2b(digest_size=8)
    for array in arrays:
        h.update(np.ascontiguousarray(array).tobytes())
        h.update(b"\x00")
    return h.hexdigest()


def _objective_block(objective: SelectorObjective) -> dict:
    return {
        "id": objective.objective_id,
        "version": objective.version,
        "weights": {
            "quality": objective.quality_weight,
            "latency": objective.latency_weight,
            "cost": objective.cost_weight,
        },
        "latency_scale_ms": objective.latency_scale_ms,
        "cost_scale": objective.cost_scale,
    }


class KMeansModel:
    """Query-level KMeans selector scored under the versioned selector objective.

    Centroids are fit on one row per unique train query. Each (cluster, candidate)
    stores the mean objective score and its support; a cell below min_support
    uses the candidate's global train mean. Distances are squared L2 and every
    tie, between clusters or candidates, goes to the lowest index.

    Fit is O(n_unique * k * d * iter); scoring one query is O(k * d + m).
    """

    def __init__(
        self,
        n_clusters: int = 8,
        efficiency_weight: float = 0.1,
        *,
        objective: SelectorObjective | None = None,
        seed: int = 42,
        n_init: int = 10,
        max_iter: int = 300,
        tol: float = 1e-4,
        min_support: int = 1,
    ):
        if n_clusters < 1 or n_init < 1 or max_iter < 1 or min_support < 1:
            raise ValueError(
                "n_clusters, n_init, max_iter and min_support must be >= 1"
            )
        if not np.isfinite(tol) or tol < 0:
            raise ValueError("tol must be finite and non-negative")
        self.n_clusters = n_clusters
        self.efficiency_weight = efficiency_weight
        self.objective = objective or SelectorObjective(
            quality_weight=1.0 - efficiency_weight, latency_weight=efficiency_weight
        )
        self.seed = seed
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.min_support = min_support
        self.format_version = KMEANS_FORMAT_VERSION
        self.centroids: np.ndarray | None = None
        self.cluster_sizes: np.ndarray | None = None
        self.scores: np.ndarray | None = None
        self.support: np.ndarray | None = None
        self.global_scores: np.ndarray | None = None
        self.global_support: np.ndarray | None = None
        self.model_names: list[str] = []
        self.feature_dim: int = 0
        self.n_train_queries: int = 0
        self.training_snapshot_id: str | None = None
        self.feature_snapshot_id: str | None = None
        self.library_versions: dict[str, str] = {}
        self._index: dict[str, int] = {}
        self._legacy_artifact: dict | None = None

    @property
    def effective_k(self) -> int:
        return 0 if self.centroids is None else len(self.centroids)

    @property
    def cluster_models(self) -> list[str]:
        """Per-cluster argmax over the whole candidate set, for the v1 runtime."""
        return [self.model_names[i] for i in np.argmax(self.scores, axis=1)]

    def train(self, samples: list[TrainingSample]) -> None:
        """Fit on (query, candidate) rows; rows sharing a query become one point.

        Rows are keyed by query_id, or by the feature vector when no sample has one.
        """
        if not samples:
            raise ValueError("KMeans requires nonempty training samples")
        features = np.asarray([s.feature_vector for s in samples], dtype=np.float64)
        ids = [s.query_id for s in samples]
        if all(i is None for i in ids):
            keys = features
        elif any(i is None for i in ids):
            raise ValueError("either every training sample has a query_id or none does")
        else:
            keys = np.asarray(ids, dtype=str)
        outcomes = [
            CandidateOutcome(
                model_ref=s.model_name,
                success=True,
                quality=s.quality,
                latency_ms=s.latency_ms,
            )
            for s in samples
        ]
        self._fit(keys, features, outcomes)

    def train_snapshots(
        self,
        snapshots: Sequence[QueryOutcomeSet],
        features: Mapping[str, np.ndarray],
        *,
        training_snapshot_id: str | None = None,
        feature_snapshot_id: str | None = None,
    ) -> None:
        """Fit on train-split query snapshots, with features keyed by query_id."""
        keys: list[str] = []
        rows: list[np.ndarray] = []
        outcomes: list[CandidateOutcome] = []
        for snapshot in snapshots:
            if snapshot.query_id not in features:
                raise ValueError(f"no feature vector for query_id {snapshot.query_id}")
            vector = features[snapshot.query_id]
            for outcome in snapshot.outcomes:
                keys.append(snapshot.query_id)
                rows.append(vector)
                outcomes.append(outcome)
        if not outcomes:
            raise ValueError("KMeans requires at least one candidate outcome")
        self._fit(
            np.asarray(keys, dtype=str),
            np.asarray(rows, dtype=np.float64),
            outcomes,
            training_snapshot_id=training_snapshot_id,
            feature_snapshot_id=feature_snapshot_id,
        )

    def _fit(
        self,
        keys: np.ndarray,
        features: np.ndarray,
        outcomes: list[CandidateOutcome],
        *,
        training_snapshot_id: str | None = None,
        feature_snapshot_id: str | None = None,
    ) -> None:
        if (
            features.ndim != 2  # noqa: PLR2004 - a feature matrix is two-dimensional
            or features.shape[1] == 0
            or not np.isfinite(features).all()
        ):
            raise ValueError("KMeans requires a finite, nonempty feature matrix")
        if any(not o.model_ref for o in outcomes):
            raise ValueError("every outcome needs a model_ref")

        # Sorted unique keys make the fit independent of input row order.
        axis = None if keys.ndim == 1 else 0
        unique_keys, first, inverse = np.unique(
            keys, axis=axis, return_index=True, return_inverse=True
        )
        inverse = inverse.reshape(-1)
        points = features[first]
        if not np.array_equal(points[inverse], features):
            raise ValueError("a query_id maps to more than one feature vector")

        names = sorted({o.model_ref for o in outcomes})
        index = {name: i for i, name in enumerate(names)}
        m = len(names)
        candidates = np.fromiter(
            (index[o.model_ref] for o in outcomes), dtype=np.intp, count=len(outcomes)
        )
        # A repeated (query, candidate) keeps its first observation, as in QueryOutcomeSet.
        _, kept = np.unique(inverse * m + candidates, return_index=True)
        query_idx = inverse[kept]
        cand_idx = candidates[kept]
        raw = np.array(
            [
                (
                    float(outcomes[i].success),
                    outcomes[i].quality,
                    outcomes[i].latency_ms,
                    outcomes[i].cost,
                )
                for i in kept
            ],
            dtype=np.float64,
        )
        observed = np.fromiter(
            (self.objective.score(outcomes[i]) for i in kept),
            dtype=np.float64,
            count=len(kept),
        )

        distinct = len(np.unique(points, axis=0))
        effective_k = min(self.n_clusters, distinct)
        fitted = SKLearnKMeans(
            n_clusters=effective_k,
            init="k-means++",
            n_init=self.n_init,
            max_iter=self.max_iter,
            tol=self.tol,
            algorithm="lloyd",
            random_state=self.seed,
        ).fit(points)
        centroids = np.asarray(fitted.cluster_centers_, dtype=np.float64)
        # Relabel under the contract's own distance and tie-break, not sklearn's.
        labels = assign_clusters(points, centroids)
        centroids, labels, sizes = drop_empty_clusters(centroids, labels)

        self._set_scores(labels[query_idx], cand_idx, observed, len(centroids), m)
        self.centroids = centroids
        self.cluster_sizes = sizes
        self.model_names = names
        self._index = index
        self.feature_dim = features.shape[1]
        self.n_train_queries = len(points)
        self.format_version = KMEANS_FORMAT_VERSION
        self._legacy_artifact = None
        self.feature_snapshot_id = feature_snapshot_id or _content_id(points)
        self.training_snapshot_id = training_snapshot_id or _content_id(
            np.asarray(unique_keys),
            np.asarray(names, dtype=str),
            query_idx,
            cand_idx,
            raw,
        )
        self.library_versions = {
            "scikit-learn": sklearn.__version__,
            "numpy": np.__version__,
        }
        print(
            f"KMeans trained on {len(points)} unique queries, "
            f"k={self.effective_k} (requested {self.n_clusters})"
        )

    def _set_scores(
        self,
        cluster_idx: np.ndarray,
        cand_idx: np.ndarray,
        observed: np.ndarray,
        k: int,
        m: int,
    ) -> None:
        cell = cluster_idx * m + cand_idx
        support = np.bincount(cell, minlength=k * m).reshape(k, m)
        sums = np.bincount(cell, weights=observed, minlength=k * m).reshape(k, m)
        global_support = np.bincount(cand_idx, minlength=m)
        global_scores = np.bincount(cand_idx, weights=observed, minlength=m)
        global_scores = global_scores / global_support
        means = np.divide(sums, support, out=np.zeros_like(sums), where=support > 0)
        self.scores = np.where(support >= self.min_support, means, global_scores)
        self.support = support
        self.global_scores = global_scores
        self.global_support = global_support

    def _check_query(self, feature_vector: np.ndarray) -> np.ndarray:
        if self.centroids is None:
            raise ValueError("Model not trained")
        query = np.asarray(feature_vector, dtype=np.float64)
        if query.shape != (self.feature_dim,) or not np.isfinite(query).all():
            raise ValueError(
                "Query must be finite and match the KMeans feature dimension"
            )
        return query

    def cluster_of(self, feature_vector: np.ndarray) -> int:
        query = self._check_query(feature_vector)
        return int(assign_clusters(query[None, :], self.centroids)[0])

    def score(
        self, feature_vector: np.ndarray, candidates: Iterable[str] | None = None
    ) -> KMeansDecision:
        """Score the request's candidates in the nearest cluster; unknown names are ignored."""
        cluster = self.cluster_of(feature_vector)
        if candidates is None:
            eligible = list(range(len(self.model_names)))
        else:
            eligible = sorted({self._index[c] for c in candidates if c in self._index})
        if not eligible:
            raise NoEligibleCandidateError(
                "none of the requested candidates is in the trained candidate set"
            )
        row = self.scores[cluster, eligible]
        # np.argmax keeps the first maximum, the lowest candidate-set index.
        best = eligible[int(np.argmax(row))]
        return KMeansDecision(
            cluster_id=cluster,
            scores={
                self.model_names[i]: float(s)
                for i, s in zip(eligible, row, strict=True)
            },
            model=self.model_names[best],
        )

    def predict(
        self, feature_vector: np.ndarray, candidates: Iterable[str] | None = None
    ) -> str:
        """Predict the best model for a query."""
        return self.score(feature_vector, candidates).model

    def to_artifact(self) -> dict:
        if self._legacy_artifact is not None:
            return self._legacy_artifact
        if self.centroids is None:
            raise ValueError("Model not trained")
        return {
            "algorithm": "kmeans",
            "format_version": KMEANS_FORMAT_VERSION,
            "trained": True,
            "target_contract": SELECTOR_TARGET_CONTRACT,
            "objective": _objective_block(self.objective),
            "candidate_set": {
                "id": _digest(*self.model_names),
                "models": self.model_names,
            },
            "feature": {
                "snapshot_id": self.feature_snapshot_id,
                "dim": self.feature_dim,
                "normalization": KMEANS_FEATURE_NORMALIZATION,
            },
            "training_snapshot_id": self.training_snapshot_id,
            "clustering": {
                "init": "k-means++",
                "algorithm": "lloyd",
                "dtype": "float64",
                "seed": self.seed,
                "n_init": self.n_init,
                "max_iter": self.max_iter,
                "tol": self.tol,
                "library_versions": self.library_versions,
                "requested_k": self.n_clusters,
                "effective_k": self.effective_k,
                "train_queries": self.n_train_queries,
            },
            "distance": KMEANS_DISTANCE,
            "tie_break": KMEANS_TIE_BREAK,
            "num_clusters": self.effective_k,
            "centroids": self.centroids.tolist(),
            "cluster_sizes": self.cluster_sizes.tolist(),
            "scores": self.scores.tolist(),
            "support": self.support.tolist(),
            "fallback": {
                "min_support": self.min_support,
                "global_scores": self.global_scores.tolist(),
                "global_support": self.global_support.tolist(),
            },
            "cluster_models": self.cluster_models,
            "model_names": self.model_names,
            "feature_dim": self.feature_dim,
        }

    def save(self, path: str) -> None:
        """Write the v2 artifact; cluster_models keeps the current runtime loading it."""
        with open(path, "w") as f:
            json.dump(self.to_artifact(), f, allow_nan=False)
        size_mb = Path(path).stat().st_size / (1024 * 1024)
        print(f"Saved KMeans model to {path} ({size_mb:.1f} MB)")

    @classmethod
    def load(
        cls,
        path: str,
        *,
        candidates: Iterable[str] | None = None,
        feature_dim: int | None = None,
        objective: SelectorObjective | None = None,
    ) -> KMeansModel:
        """Load and validate an artifact against the runtime's configuration."""
        with open(path) as f:
            data = json.load(f)
        return cls.from_artifact(
            data, candidates=candidates, feature_dim=feature_dim, objective=objective
        )

    @classmethod
    def from_artifact(
        cls,
        data: dict,
        *,
        candidates: Iterable[str] | None = None,
        feature_dim: int | None = None,
        objective: SelectorObjective | None = None,
    ) -> KMeansModel:
        version = data.get("format_version", 1)
        if version == 1:
            model = cls._from_v1(data)
        elif version == KMEANS_FORMAT_VERSION:
            model = cls._from_v2(data)
        else:
            raise KMeansArtifactError(f"unsupported KMeans format_version {version!r}")
        if candidates is not None and set(candidates) != set(model.model_names):
            raise KMeansArtifactError(
                "configured candidates differ from the artifact candidate set"
            )
        if feature_dim is not None and feature_dim != model.feature_dim:
            raise KMeansArtifactError(
                f"feature dim {feature_dim} != artifact dim {model.feature_dim}"
            )
        if (
            objective is not None
            and version == KMEANS_FORMAT_VERSION
            and objective.objective_id != model.objective.objective_id
        ):
            raise KMeansArtifactError("configured objective differs from the artifact")
        return model

    @classmethod
    def _from_v2(cls, data: dict) -> KMeansModel:
        try:
            block = data["objective"]
            weights = block["weights"]
            rule = SelectorObjective(
                quality_weight=weights["quality"],
                latency_weight=weights["latency"],
                cost_weight=weights["cost"],
                latency_scale_ms=block["latency_scale_ms"],
                cost_scale=block["cost_scale"],
                version=block["version"],
            )
            clustering = data["clustering"]
            fallback = data["fallback"]
            names = list(data["candidate_set"]["models"])
            dim = data["feature"]["dim"]
            model = cls(
                n_clusters=clustering["requested_k"],
                efficiency_weight=rule.latency_weight,
                objective=rule,
                seed=clustering["seed"],
                n_init=clustering["n_init"],
                max_iter=clustering["max_iter"],
                tol=clustering["tol"],
                min_support=fallback["min_support"],
            )
            centroids = np.asarray(data["centroids"], dtype=np.float64)
            sizes = np.asarray(data["cluster_sizes"])
            scores = np.asarray(data["scores"], dtype=np.float64)
            support = np.asarray(data["support"])
            global_scores = np.asarray(fallback["global_scores"], dtype=np.float64)
            global_support = np.asarray(fallback["global_support"])
        except (KeyError, TypeError, ValueError) as exc:
            raise KMeansArtifactError(f"malformed KMeans v2 artifact: {exc}") from exc

        k, m = len(centroids), len(names)
        problems = []
        if data.get("algorithm") != "kmeans":
            problems.append("algorithm")
        if data.get("target_contract") != SELECTOR_TARGET_CONTRACT:
            problems.append("target_contract")
        if block.get("id") != rule.objective_id:
            problems.append("objective id")
        if m == 0 or len(set(names)) != m or names != sorted(names):
            problems.append("candidate_set models")
        elif data["candidate_set"].get("id") != _digest(*names):
            problems.append("candidate_set id")
        if data.get("distance") != KMEANS_DISTANCE:
            problems.append("distance")
        if data.get("tie_break") != KMEANS_TIE_BREAK:
            problems.append("tie_break")
        if data["feature"].get("normalization") != KMEANS_FEATURE_NORMALIZATION:
            problems.append("feature normalization")
        if (
            not isinstance(dim, int)
            or dim < 1
            or centroids.shape != (k, dim)
            or k == 0
            or not np.isfinite(centroids).all()
        ):
            problems.append("centroids or feature dim")
        if not (
            k
            == clustering.get("effective_k")
            == data.get("num_clusters")
            == len(sizes)
            <= model.n_clusters
        ) or not (sizes.dtype.kind == "i" and (sizes > 0).all()):
            problems.append("cluster counts")
        if (
            scores.shape != (k, m)
            or support.shape != (k, m)
            or global_scores.shape != (m,)
            or global_support.shape != (m,)
            or not np.isfinite(scores).all()
            or not np.isfinite(global_scores).all()
            or support.dtype.kind != "i"
            or global_support.dtype.kind != "i"
            or (support < 0).any()
            or (global_support < 1).any()
        ):
            problems.append("score tables")
        elif not np.array_equal(
            scores[support < model.min_support],
            np.broadcast_to(global_scores, (k, m))[support < model.min_support],
        ):
            problems.append("fallback cells")
        if problems:
            raise KMeansArtifactError(
                f"invalid KMeans v2 artifact: {', '.join(problems)}"
            )

        model.centroids = centroids
        model.cluster_sizes = sizes
        model.scores = scores
        model.support = support
        model.global_scores = global_scores
        model.global_support = global_support
        model.model_names = names
        model._index = {name: i for i, name in enumerate(names)}
        model.feature_dim = dim
        model.n_train_queries = clustering.get("train_queries", int(sizes.sum()))
        model.training_snapshot_id = data.get("training_snapshot_id")
        model.feature_snapshot_id = data["feature"].get("snapshot_id")
        model.library_versions = dict(clustering.get("library_versions", {}))
        if data.get("cluster_models") != model.cluster_models:
            raise KMeansArtifactError("cluster_models is not the per-cluster argmax")
        return model

    @classmethod
    def _from_v1(cls, data: dict) -> KMeansModel:
        """Read an unversioned export as one-hot scores; it re-saves unchanged."""
        centroids = np.asarray(data.get("centroids", []), dtype=np.float64)
        raw = data.get("cluster_models", [])
        if isinstance(raw, dict):
            raw = [raw.get(str(i), raw.get(i)) for i in range(len(centroids))]
        if (
            centroids.ndim != 2  # noqa: PLR2004 - centroids form a matrix
            or len(centroids) == 0
            or centroids.shape[1] == 0
            or not np.isfinite(centroids).all()
            or len(raw) < len(centroids)
            or any(not isinstance(name, str) or not name for name in raw)
        ):
            raise KMeansArtifactError("invalid unversioned KMeans artifact")
        assigned = raw[: len(centroids)]
        names = sorted(set(assigned) | set(data.get("model_names", [])))
        model = cls(
            n_clusters=max(len(centroids), data.get("n_clusters", len(centroids))),
            efficiency_weight=data.get("efficiency_weight", 0.1),
        )
        model.format_version = 1
        model.model_names = names
        model._index = {name: i for i, name in enumerate(names)}
        model.centroids = centroids
        model.feature_dim = centroids.shape[1]
        winners = [model._index[name] for name in assigned]
        model.scores = np.zeros((len(centroids), len(names)))
        model.scores[np.arange(len(centroids)), winners] = 1.0
        model.support = np.zeros(model.scores.shape, dtype=np.int64)
        model._legacy_artifact = data
        return model


class SVMModel:
    """
    Support Vector Machine model for model selection.

    Uses linear or RBF kernels with libsvm one-vs-one voting.
    Quality+latency weighted training samples.
    """

    def __init__(
        self,
        kernel: str = "rbf",
        gamma: float = 1.0,
        C: float = 1.0,  # noqa: N803 - retain sklearn-compatible public parameter
    ):
        if kernel not in ("linear", "rbf"):
            raise ValueError("Only linear and RBF SVM kernels are supported")
        if not np.isfinite(gamma) or gamma <= 0 or not np.isfinite(C) or C <= 0:
            raise ValueError("SVM gamma and C must be finite and positive")
        self.kernel = kernel
        self.gamma = gamma
        self.C = C
        self.svm = None
        self.artifact = None
        self.label_encoder = None
        self.model_names: list[str] = []
        self.feature_dim: int = 0

    def train(self, samples: list[TrainingSample]) -> None:
        """Train SVM model with quality+latency weighted samples."""
        # Weight calculation: duplicate samples based on quality+latency score
        weighted_features = []
        weighted_labels = []

        for sample in samples:
            # Calculate weight: 0.9 * quality + 0.1 * speed_factor
            speed_factor = 1.0 / (1.0 + sample.latency_ms / 10000.0)
            weight = 0.9 * sample.quality + 0.1 * speed_factor

            # Duplicate samples based on weight (1-3 copies)
            n_copies = max(1, min(3, int(weight * 3 + 0.5)))
            for _ in range(n_copies):
                weighted_features.append(sample.feature_vector)
                weighted_labels.append(sample.model_name)

        print(
            f"  SVM training: {len(samples)} records -> {len(weighted_features)} weighted samples"
        )

        # Convert to numpy
        features = np.array(weighted_features, dtype=np.float32)
        self.feature_dim = features.shape[1]

        # Encode labels
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(weighted_labels)
        self.model_names = list(self.label_encoder.classes_)

        # Train SVM
        self.svm = SVC(
            kernel=self.kernel,
            gamma=self.gamma if self.kernel == "rbf" else "scale",
            C=self.C,
            decision_function_shape="ovr",
        )
        self.svm.fit(features, y)
        self.artifact = None

        print(f"SVM trained with {len(weighted_features)} weighted samples")

    def predict(self, feature_vector: np.ndarray) -> str:
        """Predict with the fitted SVC or its exact portable parameters."""
        if self.svm is not None:
            prediction = self.svm.predict([feature_vector])[0]
            return self.label_encoder.inverse_transform([prediction])[0]
        if self.artifact is None:
            raise ValueError("Model not trained")

        query = np.asarray(feature_vector, dtype=np.float64)
        if query.shape != (self.feature_dim,) or not np.isfinite(query).all():
            raise ValueError("Query must be finite and match the SVM feature dimension")
        svc = self.artifact["svc"]
        vectors = np.asarray(svc["support_vectors"], dtype=np.float64)
        coefficients = np.asarray(svc["dual_coef"], dtype=np.float64)
        if self.kernel == "linear":
            kernels = vectors @ query
        else:
            kernels = np.exp(-self.gamma * np.sum((vectors - query) ** 2, axis=1))
        starts = np.cumsum([0] + svc["n_support"])
        votes = np.zeros(len(self.model_names), dtype=int)
        pair = 0
        for i in range(len(votes)):
            for j in range(i + 1, len(votes)):
                score = (
                    coefficients[j - 1, starts[i] : starts[i + 1]]
                    @ kernels[starts[i] : starts[i + 1]]
                    + coefficients[i, starts[j] : starts[j + 1]]
                    @ kernels[starts[j] : starts[j + 1]]
                    + svc["intercept"][pair]
                )
                # sklearn exposes reversed signs for binary public parameters.
                if len(votes) == 2:  # noqa: PLR2004 - binary SVC has two classes
                    votes[j if score >= 0 else i] += 1
                else:
                    votes[i if score > 0 else j] += 1
                pair += 1
        return self.model_names[int(np.argmax(votes))]

    def save(self, path: str) -> None:
        """Export exact SVC parameters; inference uses the original input scale."""
        if self.svm is not None:
            data = {
                "algorithm": "svm",
                "format_version": 2,
                "trained": True,
                "model_names": self.model_names,
                "kernel_type": "Rbf" if self.kernel == "rbf" else "Linear",
                "gamma": float(self.svm._gamma),
                "feature_dim": self.feature_dim,
                "input_normalization": "none",
                "svc": {
                    "support_vectors": self.svm.support_vectors_.tolist(),
                    "dual_coef": self.svm.dual_coef_.tolist(),
                    "intercept": self.svm.intercept_.tolist(),
                    "n_support": self.svm.n_support_.tolist(),
                },
                "C": self.C,
            }
        elif self.artifact is not None:
            data = self.artifact
        else:
            raise ValueError("Model not trained")
        with open(path, "w") as f:
            json.dump(data, f, allow_nan=False)
        size_mb = Path(path).stat().st_size / (1024 * 1024)
        print(f"Saved SVM model to {path} ({size_mb:.1f} MB)")

    @classmethod
    def load(cls, path: str) -> SVMModel:
        """Load portable SVC parameters without reconstructing sklearn internals.

        Unversioned Python exports already contain the exact SVC parameters;
        migrate them in memory instead of using their approximate classifiers.
        """
        with open(path) as f:
            data = json.load(f)
        if data.get("format_version", 1) not in (1, 2):
            raise ValueError("Unsupported SVM artifact version")
        if "svc" not in data:
            fields = ("support_vectors", "dual_coef", "intercept", "n_support")
            if not all(name in data for name in fields):
                raise ValueError(
                    "SVM artifact has no exact SVC parameters; re-export the trained model"
                )
            data["svc"] = {name: data[name] for name in fields}
        if data.get("input_normalization", "none") != "none":
            raise ValueError("Unsupported SVM input normalization")
        kernel = data.get("kernel_type", data.get("kernel", "rbf")).lower()
        if kernel not in ("linear", "rbf"):
            raise ValueError("Only linear and RBF SVM kernels are supported")
        model = cls(kernel=kernel, gamma=data["gamma"], C=data.get("C", 1.0))
        model.model_names = data["model_names"]
        model.feature_dim = data["feature_dim"]
        svc = data["svc"]
        vectors = np.asarray(svc["support_vectors"], dtype=np.float64)
        coefficients = np.asarray(svc["dual_coef"], dtype=np.float64)
        n_classes = len(model.model_names)
        n_support = svc["n_support"]
        if (
            n_classes < 2  # noqa: PLR2004 - SVC requires at least two classes
            or len(set(model.model_names)) != n_classes
            or vectors.ndim != 2  # noqa: PLR2004 - support vectors form a matrix
            or vectors.shape[1] != model.feature_dim
            or model.feature_dim == 0
            or len(vectors) == 0
            or coefficients.shape != (n_classes - 1, len(vectors))
            or len(svc["intercept"]) != n_classes * (n_classes - 1) // 2
            or len(n_support) != n_classes
            or any(not isinstance(n, int) or n < 0 for n in n_support)
            or sum(n_support) != len(vectors)
            or not np.isfinite(vectors).all()
            or not np.isfinite(coefficients).all()
            or not np.isfinite(svc["intercept"]).all()
            or not np.isfinite(model.gamma)
            or model.gamma <= 0
        ):
            raise ValueError("Invalid SVM parameter shapes or values")
        model.artifact = {
            "algorithm": "svm",
            "format_version": 2,
            "trained": True,
            "model_names": model.model_names,
            "kernel_type": "Rbf" if kernel == "rbf" else "Linear",
            "gamma": model.gamma,
            "feature_dim": model.feature_dim,
            "input_normalization": "none",
            "svc": svc,
            "C": model.C,
        }
        return model


class MLPModel:
    """
    Multi-Layer Perceptron model for model selection.

    Uses a neural network with configurable hidden layers for query-model routing.
    Reference: FusionFactory (arXiv:2507.10540) - Query-level fusion via tailored LLM routers

    This model supports GPU acceleration via PyTorch/CUDA.
    """

    def __init__(
        self,
        hidden_sizes: list[int] | None = None,
        learning_rate: float = 0.001,
        epochs: int = 100,
        batch_size: int = 32,
        dropout: float = 0.1,
        device: str = "cpu",
    ):
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for MLP. Install with: pip install torch"
            )

        self.hidden_sizes = hidden_sizes or [256, 128]
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.device = device
        self.model = None
        self.label_encoder = None
        self.model_names: list[str] = []
        self.feature_dim: int = 0
        self.n_classes: int = 0

    def _build_network(self, input_dim: int, output_dim: int) -> nn.Module:
        """Build the MLP network architecture."""
        layers = []
        prev_dim = input_dim

        # Hidden layers
        for hidden_size in self.hidden_sizes:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_size),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_size),
                    nn.Dropout(self.dropout),
                ]
            )
            prev_dim = hidden_size

        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))

        return nn.Sequential(*layers)

    def train(self, samples: list[TrainingSample]) -> None:
        """Train MLP model with quality+latency weighted samples."""
        # Prepare data
        features = np.array([s.feature_vector for s in samples], dtype=np.float32)
        self.feature_dim = features.shape[1]

        # Encode labels
        self.label_encoder = LabelEncoder()
        labels = [s.model_name for s in samples]
        y = self.label_encoder.fit_transform(labels)
        self.model_names = list(self.label_encoder.classes_)
        self.n_classes = len(self.model_names)

        # Calculate sample weights based on quality+latency
        weights = []
        for sample in samples:
            speed_factor = 1.0 / (1.0 + sample.latency_ms / 10000.0)
            weight = 0.9 * sample.quality + 0.1 * speed_factor
            weights.append(max(0.1, weight))  # Minimum weight of 0.1
        weights = np.array(weights, dtype=np.float32)

        # Convert to PyTorch tensors
        x_tensor = torch.tensor(features, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        weights_tensor = torch.tensor(weights, dtype=torch.float32)

        # Create weighted dataset
        dataset = TensorDataset(x_tensor, y_tensor, weights_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Build and train model
        self.model = self._build_network(self.feature_dim, self.n_classes)
        self.model = self.model.to(self.device)

        criterion = nn.CrossEntropyLoss(reduction="none")
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10
        )

        print(f"  MLP training: {len(samples)} samples, {self.n_classes} classes")
        print(
            f"  Architecture: {self.feature_dim} -> {self.hidden_sizes} -> {self.n_classes}"
        )
        print(f"  Device: {self.device}")

        best_loss = float("inf")
        patience_counter = 0
        max_patience = 20

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0.0
            num_batches = 0

            for batch_features, batch_labels, batch_weights in dataloader:
                x_batch = batch_features.to(self.device)
                y_batch = batch_labels.to(self.device)
                w_batch = batch_weights.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(x_batch)
                loss = criterion(outputs, y_batch)
                # Apply sample weights
                weighted_loss = (loss * w_batch).mean()
                weighted_loss.backward()
                optimizer.step()

                total_loss += weighted_loss.item()
                num_batches += 1

            avg_loss = total_loss / num_batches
            scheduler.step(avg_loss)

            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    print(f"  Early stopping at epoch {epoch + 1}")
                    break

            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch + 1}/{self.epochs}, Loss: {avg_loss:.4f}")

        print(f"MLP trained with {len(samples)} samples, final loss: {best_loss:.4f}")

    def predict(self, feature_vector: np.ndarray) -> str:
        """Predict best model."""
        if self.model is None:
            raise ValueError("Model not trained")

        self.model.eval()
        with torch.no_grad():
            features = torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(0)
            features = features.to(self.device)
            outputs = self.model(features)
            _, predicted = torch.max(outputs, 1)
            return self.label_encoder.inverse_transform(predicted.cpu().numpy())[0]

    def predict_proba(self, feature_vector: np.ndarray) -> dict[str, float]:
        """Predict probabilities for each model."""
        if self.model is None:
            raise ValueError("Model not trained")

        self.model.eval()
        with torch.no_grad():
            features = torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(0)
            features = features.to(self.device)
            outputs = self.model(features)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            return {
                name: float(prob)
                for name, prob in zip(self.model_names, probs, strict=True)
            }

    def save(self, path: str) -> None:
        """Save model to JSON format compatible with Rust/Candle inference."""
        if self.model is None:
            raise ValueError("Model not trained")

        # Extract model weights
        state_dict = self.model.state_dict()
        layers = []

        # Parse the sequential model layers
        layer_idx = 0
        for name, param in state_dict.items():
            param_np = param.cpu().numpy()

            if "weight" in name:
                layers.append(
                    {
                        "type": (
                            "linear"
                            if "Linear" in str(type(self.model[layer_idx]))
                            else "other"
                        ),
                        "weight": param_np.tolist(),
                    }
                )
            elif "bias" in name:
                # Add bias to the last added layer
                if layers:
                    layers[-1]["bias"] = param_np.tolist()
                layer_idx += 1

        # Reconstruct layer structure with activations
        mlp_layers = []

        for i, module in enumerate(self.model):
            if isinstance(module, nn.Linear):
                weight_key = f"{i}.weight"
                bias_key = f"{i}.bias"
                mlp_layers.append(
                    {
                        "type": "linear",
                        "in_features": module.in_features,
                        "out_features": module.out_features,
                        "weight": state_dict[weight_key].cpu().numpy().tolist(),
                        "bias": (
                            state_dict[bias_key].cpu().numpy().tolist()
                            if bias_key in state_dict
                            else None
                        ),
                    }
                )
            elif isinstance(module, nn.ReLU):
                mlp_layers.append({"type": "relu"})
            elif isinstance(module, nn.BatchNorm1d):
                weight_key = f"{i}.weight"
                bias_key = f"{i}.bias"
                mean_key = f"{i}.running_mean"
                var_key = f"{i}.running_var"
                mlp_layers.append(
                    {
                        "type": "batch_norm",
                        "num_features": module.num_features,
                        "weight": (
                            state_dict[weight_key].cpu().numpy().tolist()
                            if weight_key in state_dict
                            else None
                        ),
                        "bias": (
                            state_dict[bias_key].cpu().numpy().tolist()
                            if bias_key in state_dict
                            else None
                        ),
                        "running_mean": (
                            state_dict[mean_key].cpu().numpy().tolist()
                            if mean_key in state_dict
                            else None
                        ),
                        "running_var": (
                            state_dict[var_key].cpu().numpy().tolist()
                            if var_key in state_dict
                            else None
                        ),
                        "eps": module.eps,
                    }
                )
            elif isinstance(module, nn.Dropout):
                mlp_layers.append({"type": "dropout", "p": module.p})

        data = {
            "algorithm": "mlp",
            "trained": True,
            "model_names": self.model_names,
            "feature_dim": self.feature_dim,
            "n_classes": self.n_classes,
            "hidden_sizes": self.hidden_sizes,
            "dropout": self.dropout,
            "layers": mlp_layers,
            # Legacy fields for Python reloading
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
        }

        with open(path, "w") as f:
            json.dump(data, f)

        size_mb = Path(path).stat().st_size / (1024 * 1024)
        print(f"Saved MLP model to {path} ({size_mb:.2f} MB)")

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> MLPModel:
        """Load model from JSON."""
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for MLP. Install with: pip install torch"
            )

        with open(path) as f:
            data = json.load(f)

        model = cls(
            hidden_sizes=data.get("hidden_sizes", [256, 128]),
            learning_rate=data.get("learning_rate", 0.001),
            epochs=data.get("epochs", 100),
            batch_size=data.get("batch_size", 32),
            dropout=data.get("dropout", 0.1),
            device=device,
        )
        model.model_names = data.get("model_names", [])
        model.feature_dim = data.get("feature_dim", 0)
        model.n_classes = data.get("n_classes", len(model.model_names))

        # Rebuild label encoder
        model.label_encoder = LabelEncoder()
        model.label_encoder.classes_ = np.array(model.model_names)

        # Rebuild network from saved layers
        if "layers" in data and model.feature_dim > 0:
            model.model = model._build_network(model.feature_dim, model.n_classes)

            # Load weights from saved layers
            state_dict = {}
            layer_idx = 0
            for saved_layer in data["layers"]:
                if saved_layer["type"] == "linear":
                    state_dict[f"{layer_idx}.weight"] = torch.tensor(
                        saved_layer["weight"], dtype=torch.float32
                    )
                    if saved_layer.get("bias") is not None:
                        state_dict[f"{layer_idx}.bias"] = torch.tensor(
                            saved_layer["bias"], dtype=torch.float32
                        )
                    layer_idx += 1
                elif saved_layer["type"] == "relu":
                    layer_idx += 1
                elif saved_layer["type"] == "batch_norm":
                    if saved_layer.get("weight") is not None:
                        state_dict[f"{layer_idx}.weight"] = torch.tensor(
                            saved_layer["weight"], dtype=torch.float32
                        )
                    if saved_layer.get("bias") is not None:
                        state_dict[f"{layer_idx}.bias"] = torch.tensor(
                            saved_layer["bias"], dtype=torch.float32
                        )
                    if saved_layer.get("running_mean") is not None:
                        state_dict[f"{layer_idx}.running_mean"] = torch.tensor(
                            saved_layer["running_mean"], dtype=torch.float32
                        )
                    if saved_layer.get("running_var") is not None:
                        state_dict[f"{layer_idx}.running_var"] = torch.tensor(
                            saved_layer["running_var"], dtype=torch.float32
                        )
                    state_dict[f"{layer_idx}.num_batches_tracked"] = torch.tensor(0)
                    layer_idx += 1
                elif saved_layer["type"] == "dropout":
                    layer_idx += 1

            # Load state dict with strict=False to handle any missing keys
            try:
                model.model.load_state_dict(state_dict, strict=False)
            except Exception as e:
                print(f"Warning: Partial weight loading: {e}")

            model.model = model.model.to(device)
            model.model.eval()

        return model
