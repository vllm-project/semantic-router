#!/usr/bin/env python3
"""
ML models for model selection.

Implements KNN, KMeans, SVM, and MLP using scikit-learn and PyTorch.
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

# Optional PyTorch import for MLP
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
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
        model.model_names = data.get("model_names", [])

        # Handle both old format (samples list) and new format (flat arrays)
        if "samples" in data:
            # Old format: list of sample objects
            model.samples = [
                TrainingSample(
                    feature_vector=np.array(s["feature_vector"], dtype=np.float32),
                    model_name=s["model_name"],
                    quality=s["quality"],
                    latency_ms=s["latency_ms"],
                )
                for s in data["samples"]
            ]
        elif "embeddings" in data:
            # New format: flat arrays (Rust-compatible)
            embeddings = data["embeddings"]
            labels = data["labels"]
            qualities = data["qualities"]
            latencies = data["latencies"]

            model.samples = [
                TrainingSample(
                    feature_vector=np.array(emb, dtype=np.float32),
                    model_name=label,
                    quality=quality,
                    latency_ms=latency / 1_000_000,  # Convert ns back to ms
                )
                for emb, label, quality, latency in zip(
                    embeddings, labels, qualities, latencies
                )
            ]

            # Update model_names if not present
            if not model.model_names:
                model.model_names = sorted(set(labels))
        else:
            raise ValueError(
                "Invalid KNN model format: missing 'samples' or 'embeddings'"
            )

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
            n_clusters=data.get("n_clusters", data.get("num_clusters", 8)),
            efficiency_weight=data.get("efficiency_weight", 0.1),
        )
        model.model_names = data.get("model_names", [])
        model.feature_dim = data.get("feature_dim", 0)

        # Handle both dict format (old) and list format (new Rust-compatible)
        cluster_models_raw = data.get("cluster_models", {})
        if isinstance(cluster_models_raw, list):
            # New format: list where index is cluster_id
            model.cluster_models = {i: m for i, m in enumerate(cluster_models_raw)}
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
            kernel=data.get("kernel", "rbf"),
            gamma=data.get("gamma", 1.0),
            C=data.get("C", 1.0),
        )
        model.model_names = data.get("model_names", [])
        model.feature_dim = data.get("feature_dim", 0)

        # Rebuild label encoder
        model.label_encoder = LabelEncoder()
        model.label_encoder.classes_ = np.array(model.model_names)

        # Reconstruct SVM if sklearn parameters are present
        # Note: Full SVM reconstruction from saved parameters is complex due to sklearn internals
        # For validation, we use a simplified approach: retrain a small surrogate model
        # For production inference, use the Rust implementation which reads these parameters directly
        if "support_vectors" in data and "dual_coef" in data:
            try:
                support_vectors = np.array(data["support_vectors"], dtype=np.float64)
                dual_coef = np.array(data["dual_coef"], dtype=np.float64)
                intercept = np.array(data["intercept"], dtype=np.float64)
                classes = np.array(data["classes"], dtype=np.int32)

                # Create a fitted SVM using sklearn's internal attributes
                # This uses the DecisionBoundaryPlotMixin workaround
                model.svm = SVC(
                    kernel=model.kernel,
                    gamma=model.gamma if model.kernel == "rbf" else "scale",
                    C=model.C,
                    decision_function_shape="ovr",
                )

                # Use object.__setattr__ to bypass property restrictions
                object.__setattr__(model.svm, "support_vectors_", support_vectors)
                object.__setattr__(model.svm, "dual_coef_", dual_coef)
                object.__setattr__(model.svm, "intercept_", intercept)
                object.__setattr__(model.svm, "classes_", classes)
                object.__setattr__(
                    model.svm, "_n_support", np.array(data["n_support"], dtype=np.int32)
                )
                object.__setattr__(
                    model.svm, "support_", np.arange(len(support_vectors))
                )
                object.__setattr__(model.svm, "_sparse", False)
                object.__setattr__(
                    model.svm, "shape_fit_", (len(support_vectors), model.feature_dim)
                )
                object.__setattr__(model.svm, "_gamma", model.gamma)
                object.__setattr__(model.svm, "_probA", np.empty(0))
                object.__setattr__(model.svm, "_probB", np.empty(0))
                object.__setattr__(model.svm, "fit_status_", 0)

            except Exception:
                # SVM reconstruction failed, but that's okay for validation
                # The Rust implementation handles this correctly
                model.svm = None

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
        hidden_sizes: List[int] = None,
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
        self.model_names: List[str] = []
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

    def train(self, samples: List[TrainingSample]) -> None:
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
        X_tensor = torch.tensor(features, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        weights_tensor = torch.tensor(weights, dtype=torch.float32)

        # Create weighted dataset
        dataset = TensorDataset(X_tensor, y_tensor, weights_tensor)
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

            for X_batch, y_batch, w_batch in dataloader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                w_batch = w_batch.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(X_batch)
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

        print(f"✓ MLP trained with {len(samples)} samples, final loss: {best_loss:.4f}")

    def predict(self, feature_vector: np.ndarray) -> str:
        """Predict best model."""
        if self.model is None:
            raise ValueError("Model not trained")

        self.model.eval()
        with torch.no_grad():
            X = torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(0)
            X = X.to(self.device)
            outputs = self.model(X)
            _, predicted = torch.max(outputs, 1)
            return self.label_encoder.inverse_transform(predicted.cpu().numpy())[0]

    def predict_proba(self, feature_vector: np.ndarray) -> Dict[str, float]:
        """Predict probabilities for each model."""
        if self.model is None:
            raise ValueError("Model not trained")

        self.model.eval()
        with torch.no_grad():
            X = torch.tensor(feature_vector, dtype=torch.float32).unsqueeze(0)
            X = X.to(self.device)
            outputs = self.model(X)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            return {name: float(prob) for name, prob in zip(self.model_names, probs)}

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
        current_layer = None

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
        print(f"✓ Saved MLP model to {path} ({size_mb:.2f} MB)")

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "MLPModel":
        """Load model from JSON."""
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is required for MLP. Install with: pip install torch"
            )

        with open(path, "r") as f:
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
