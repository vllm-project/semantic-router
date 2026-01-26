#!/usr/bin/env python3
"""
ML models for model selection.

Implements KNN, KMeans, and SVM using scikit-learn.
Models are saved in JSON format compatible with the Rust inference code.

Reference:
- FusionFactory (arXiv:2507.10540) - Query-level fusion via LLM routers
- Avengers-Pro (arXiv:2508.12631) - Performance-efficiency optimized routing
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import KMeans as SKLearnKMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC


@dataclass
class TrainingSample:
    """A training sample for model selection."""

    feature_vector: np.ndarray  # 1038-dim (1024 embedding + 14 category one-hot)
    model_name: str
    quality: float
    latency_ms: float


class KNNModel:
    """
    K-Nearest Neighbors model for model selection.

    Uses quality-weighted voting among k nearest neighbors.
    """

    def __init__(self, k: int = 5):
        self.k = k
        self.nn = None
        self.samples: List[TrainingSample] = []
        self.model_names: List[str] = []

    def train(self, samples: List[TrainingSample]) -> None:
        """Train KNN model."""
        self.samples = samples

        # Extract feature vectors
        features = np.array([s.feature_vector for s in samples], dtype=np.float32)

        # Fit nearest neighbors index
        self.nn = NearestNeighbors(
            n_neighbors=min(self.k, len(samples)), metric="cosine"
        )
        self.nn.fit(features)

        # Get unique model names
        self.model_names = sorted(set(s.model_name for s in samples))

        print(f"✓ KNN trained with {len(samples)} samples, k={self.k}")

    def predict(self, feature_vector: np.ndarray) -> str:
        """Predict best model using quality-weighted voting."""
        if self.nn is None:
            raise ValueError("Model not trained")

        # Find k nearest neighbors
        distances, indices = self.nn.kneighbors([feature_vector])

        # Quality-weighted voting
        votes: Dict[str, float] = {}
        for idx in indices[0]:
            sample = self.samples[idx]
            # Calculate weight: 0.9 * quality + 0.1 * speed_factor
            speed_factor = 1.0 / (1.0 + sample.latency_ms / 10000.0)
            weight = 0.9 * sample.quality + 0.1 * speed_factor
            votes[sample.model_name] = votes.get(sample.model_name, 0.0) + weight

        # Return model with highest vote
        return max(votes, key=votes.get)

    def save(self, path: str) -> None:
        """Save model to JSON format compatible with Rust/Linfa."""
        # Convert to Rust-expected format
        embeddings = [s.feature_vector.tolist() for s in self.samples]
        labels = [s.model_name for s in self.samples]
        qualities = [s.quality for s in self.samples]
        latencies = [
            int(s.latency_ms * 1_000_000) for s in self.samples
        ]  # Convert ms to ns

        data = {
            "algorithm": "knn",
            "trained": True,
            "k": self.k,
            "embeddings": embeddings,
            "labels": labels,
            "qualities": qualities,
            "latencies": latencies,
            # Keep legacy fields for Python reloading
            "model_names": self.model_names,
            "num_samples": len(self.samples),
            "feature_dim": len(self.samples[0].feature_vector) if self.samples else 0,
        }

        with open(path, "w") as f:
            json.dump(data, f)

        size_mb = Path(path).stat().st_size / (1024 * 1024)
        print(f"✓ Saved KNN model to {path} ({size_mb:.1f} MB)")

    @classmethod
    def load(cls, path: str) -> "KNNModel":
        """Load model from JSON."""
        with open(path, "r") as f:
            data = json.load(f)

        model = cls(k=data["k"])
        model.model_names = data["model_names"]
        model.samples = [
            TrainingSample(
                feature_vector=np.array(s["feature_vector"], dtype=np.float32),
                model_name=s["model_name"],
                quality=s["quality"],
                latency_ms=s["latency_ms"],
            )
            for s in data["samples"]
        ]

        # Rebuild index
        features = np.array([s.feature_vector for s in model.samples], dtype=np.float32)
        model.nn = NearestNeighbors(
            n_neighbors=min(model.k, len(model.samples)), metric="cosine"
        )
        model.nn.fit(features)

        return model


class KMeansModel:
    """
    KMeans clustering model for model selection.

    Assigns models to clusters based on quality + efficiency weighting.
    Reference: Avengers-Pro (arXiv:2508.12631)
    """

    def __init__(self, n_clusters: int = 8, efficiency_weight: float = 0.1):
        self.n_clusters = n_clusters
        self.efficiency_weight = efficiency_weight
        self.quality_weight = 1.0 - efficiency_weight
        self.kmeans = None
        self.cluster_models: Dict[int, str] = {}
        self.model_names: List[str] = []
        self.feature_dim: int = 0

    def train(self, samples: List[TrainingSample]) -> None:
        """Train KMeans model."""
        # Extract features
        features = np.array([s.feature_vector for s in samples], dtype=np.float32)
        self.feature_dim = features.shape[1]

        # Fit KMeans
        self.kmeans = SKLearnKMeans(
            n_clusters=min(self.n_clusters, len(samples)),
            random_state=42,
            n_init=10,
        )
        cluster_labels = self.kmeans.fit_predict(features)

        # Get unique model names
        self.model_names = sorted(set(s.model_name for s in samples))

        # Assign best model to each cluster using quality+efficiency weighting
        cluster_scores: Dict[int, Dict[str, float]] = {}
        for sample, cluster_id in zip(samples, cluster_labels):
            if cluster_id not in cluster_scores:
                cluster_scores[cluster_id] = {}

            # Calculate combined score
            speed_factor = 1.0 / (1.0 + sample.latency_ms / 10000.0)
            score = (
                self.quality_weight * sample.quality
                + self.efficiency_weight * speed_factor
            )

            model = sample.model_name
            cluster_scores[cluster_id][model] = (
                cluster_scores[cluster_id].get(model, 0.0) + score
            )

        # Pick best model for each cluster
        for cluster_id, scores in cluster_scores.items():
            self.cluster_models[cluster_id] = max(scores, key=scores.get)

        print(
            f"✓ KMeans trained with {len(samples)} samples, {self.n_clusters} clusters"
        )

    def predict(self, feature_vector: np.ndarray) -> str:
        """Predict best model for a query."""
        if self.kmeans is None:
            raise ValueError("Model not trained")

        cluster_id = self.kmeans.predict([feature_vector])[0]
        return self.cluster_models.get(cluster_id, self.model_names[0])

    def save(self, path: str) -> None:
        """Save model to JSON format compatible with Rust/Linfa."""
        # Convert cluster_models dict to ordered list (Rust expects Vec<String>)
        # Rust expects cluster_models[i] = model for cluster i
        cluster_models_list = []
        for i in range(self.n_clusters):
            cluster_models_list.append(self.cluster_models.get(i, self.model_names[0]))

        data = {
            "algorithm": "kmeans",
            "trained": True,
            "num_clusters": self.n_clusters,
            "centroids": self.kmeans.cluster_centers_.tolist(),
            "cluster_models": cluster_models_list,
            # Keep legacy fields for Python reloading
            "n_clusters": self.n_clusters,
            "efficiency_weight": self.efficiency_weight,
            "model_names": self.model_names,
            "feature_dim": self.feature_dim,
        }

        with open(path, "w") as f:
            json.dump(data, f)

        size_mb = Path(path).stat().st_size / (1024 * 1024)
        print(f"✓ Saved KMeans model to {path} ({size_mb:.1f} MB)")

    @classmethod
    def load(cls, path: str) -> "KMeansModel":
        """Load model from JSON."""
        with open(path, "r") as f:
            data = json.load(f)

        model = cls(
            n_clusters=data["n_clusters"],
            efficiency_weight=data["efficiency_weight"],
        )
        model.model_names = data["model_names"]
        model.feature_dim = data["feature_dim"]
        model.cluster_models = {int(k): v for k, v in data["cluster_models"].items()}

        # Rebuild KMeans from centroids
        centroids = np.array(data["centroids"], dtype=np.float32)
        model.kmeans = SKLearnKMeans(n_clusters=len(centroids), n_init=1)
        model.kmeans.cluster_centers_ = centroids
        model.kmeans._n_features_out = centroids.shape[1]

        return model


class SVMModel:
    """
    Support Vector Machine model for model selection.

    Uses RBF kernel with one-vs-all classification.
    Quality+latency weighted training samples.
    """

    def __init__(self, kernel: str = "rbf", gamma: float = 1.0, C: float = 1.0):
        self.kernel = kernel
        self.gamma = gamma
        self.C = C
        self.svm = None
        self.label_encoder = None
        self.model_names: List[str] = []
        self.feature_dim: int = 0

    def train(self, samples: List[TrainingSample]) -> None:
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
        X = np.array(weighted_features, dtype=np.float32)
        self.feature_dim = X.shape[1]

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
        self.svm.fit(X, y)

        print(f"✓ SVM trained with {len(weighted_features)} weighted samples")

    def predict(self, feature_vector: np.ndarray) -> str:
        """Predict best model."""
        if self.svm is None:
            raise ValueError("Model not trained")

        prediction = self.svm.predict([feature_vector])[0]
        return self.label_encoder.inverse_transform([prediction])[0]

    def save(self, path: str) -> None:
        """Save model to JSON format compatible with Rust/Linfa."""
        # Convert sklearn SVC to Rust-compatible format
        # Rust expects rbf_classifiers with {model_name, alpha, support_vectors, rho, gamma}

        # For multi-class, sklearn uses one-vs-one. We need to convert to one-vs-rest style
        # For now, create a simple RBF classifier per model using the support vectors

        rbf_classifiers = []
        n_classes = len(self.model_names)

        # sklearn's dual_coef_ has shape (n_classes-1, n_SV) for OvO
        # We'll create simplified per-model classifiers
        sv_per_class = self.svm.n_support_
        sv_start = 0

        for i, model_name in enumerate(self.model_names):
            n_sv = sv_per_class[i] if i < len(sv_per_class) else 0
            if n_sv > 0:
                model_svs = self.svm.support_vectors_[
                    sv_start : sv_start + n_sv
                ].tolist()
                # Use simplified alpha (all 1s for now - this is approximate)
                alpha = [1.0] * n_sv
                rho = (
                    float(self.svm.intercept_[0])
                    if len(self.svm.intercept_) > 0
                    else 0.0
                )

                rbf_classifiers.append(
                    {
                        "model_name": model_name,
                        "alpha": alpha,
                        "support_vectors": model_svs,
                        "rho": rho,
                        "gamma": self.gamma,
                    }
                )
                sv_start += n_sv

        data = {
            "algorithm": "svm",
            "trained": True,
            "model_names": self.model_names,
            "kernel_type": "Rbf" if self.kernel == "rbf" else "Linear",
            "gamma": self.gamma,
            "linear_classifiers": [],  # Empty for RBF kernel
            "rbf_classifiers": rbf_classifiers,
            # Keep legacy fields for Python reloading
            "kernel": self.kernel,
            "C": self.C,
            "feature_dim": self.feature_dim,
            "n_classes": n_classes,
            "support_vectors": self.svm.support_vectors_.tolist(),
            "dual_coef": self.svm.dual_coef_.tolist(),
            "intercept": self.svm.intercept_.tolist(),
            "n_support": self.svm.n_support_.tolist(),
            "classes": self.svm.classes_.tolist(),
        }

        with open(path, "w") as f:
            json.dump(data, f)

        size_mb = Path(path).stat().st_size / (1024 * 1024)
        print(f"✓ Saved SVM model to {path} ({size_mb:.1f} MB)")

    @classmethod
    def load(cls, path: str) -> "SVMModel":
        """Load model from JSON."""
        with open(path, "r") as f:
            data = json.load(f)

        model = cls(
            kernel=data["kernel"],
            gamma=data["gamma"],
            C=data["C"],
        )
        model.model_names = data["model_names"]
        model.feature_dim = data["feature_dim"]

        # Rebuild label encoder
        model.label_encoder = LabelEncoder()
        model.label_encoder.classes_ = np.array(data["model_names"])

        # Note: Full SVM reconstruction requires sklearn internals
        # For inference, use Rust implementation which reads these parameters

        return model
