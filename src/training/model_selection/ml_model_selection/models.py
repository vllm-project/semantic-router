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

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
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
        self.cluster_models: dict[int, str] = {}
        self.model_names: list[str] = []
        self.feature_dim: int = 0

    def train(self, samples: list[TrainingSample]) -> None:
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
        self.model_names = sorted({s.model_name for s in samples})

        # Assign best model to each cluster using quality+efficiency weighting
        cluster_scores: dict[int, dict[str, float]] = {}
        for sample, cluster_id in zip(samples, cluster_labels, strict=True):
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

        print(f"KMeans trained with {len(samples)} samples, {self.n_clusters} clusters")

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
        print(f"Saved KMeans model to {path} ({size_mb:.1f} MB)")

    @classmethod
    def load(cls, path: str) -> KMeansModel:
        """Load model from JSON."""
        with open(path) as f:
            data = json.load(f)

        model = cls(
            n_clusters=data.get("n_clusters", data.get("num_clusters", 8)),
            efficiency_weight=data.get("efficiency_weight", 0.1),
        )
        model.model_names = data.get("model_names", [])
        model.feature_dim = data.get("feature_dim", 0)

        # Handle both dict format (old) and list format (new Rust-compatible)
        cluster_models_raw = data.get("cluster_models", {})
        if isinstance(cluster_models_raw, list):
            # New format: list where index is cluster_id
            model.cluster_models = dict(enumerate(cluster_models_raw))
        elif isinstance(cluster_models_raw, dict):
            # Old format: dict with string keys
            model.cluster_models = {int(k): v for k, v in cluster_models_raw.items()}
        else:
            model.cluster_models = {}

        # Rebuild KMeans from centroids
        centroids = np.array(data["centroids"], dtype=np.float32)
        model.kmeans = SKLearnKMeans(n_clusters=len(centroids), n_init=1)
        model.kmeans.cluster_centers_ = centroids
        model.kmeans._n_features_out = centroids.shape[1]

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
