"""Typed model_selection compatibility blocks for the TD001 CLI migration."""

from pydantic import BaseModel, ConfigDict

from cli.models import (
    AutoMixSelectionConfig,
    EloSelectionConfig,
    HybridSelectionConfig,
    RouterDCSelectionConfig,
)


class ModelSelectionEloCompatConfig(EloSelectionConfig):
    """Typed schema for model_selection.elo."""

    model_config = ConfigDict(extra="forbid")


class ModelSelectionRouterDCCompatConfig(RouterDCSelectionConfig):
    """Typed schema for model_selection.router_dc."""

    model_config = ConfigDict(extra="forbid")


class ModelSelectionAutoMixCompatConfig(AutoMixSelectionConfig):
    """Typed schema for model_selection.automix."""

    model_config = ConfigDict(extra="forbid")


class ModelSelectionHybridCompatConfig(HybridSelectionConfig):
    """Typed schema for model_selection.hybrid."""

    model_config = ConfigDict(extra="forbid")


class ModelSelectionMLKNNCompatConfig(BaseModel):
    """Typed schema for model_selection.ml.knn."""

    model_config = ConfigDict(extra="forbid")

    k: int | None = None
    pretrained_path: str | None = None


class ModelSelectionMLKMeansCompatConfig(BaseModel):
    """Typed schema for model_selection.ml.kmeans."""

    model_config = ConfigDict(extra="forbid")

    num_clusters: int | None = None
    efficiency_weight: float | None = None
    pretrained_path: str | None = None


class ModelSelectionMLSVMCompatConfig(BaseModel):
    """Typed schema for model_selection.ml.svm."""

    model_config = ConfigDict(extra="forbid")

    kernel: str | None = None
    gamma: float | None = None
    pretrained_path: str | None = None


class ModelSelectionMLMLPCompatConfig(BaseModel):
    """Typed schema for model_selection.ml.mlp."""

    model_config = ConfigDict(extra="forbid")

    device: str | None = None
    pretrained_path: str | None = None


class ModelSelectionMLCompatConfig(BaseModel):
    """Typed schema for model_selection.ml."""

    model_config = ConfigDict(extra="forbid")

    models_path: str | None = None
    embedding_dim: int | None = None
    knn: ModelSelectionMLKNNCompatConfig | None = None
    kmeans: ModelSelectionMLKMeansCompatConfig | None = None
    svm: ModelSelectionMLSVMCompatConfig | None = None
    mlp: ModelSelectionMLMLPCompatConfig | None = None


class ModelSelectionCompatConfig(BaseModel):
    """Typed schema for the top-level model_selection block."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool | None = None
    method: str | None = None
    elo: ModelSelectionEloCompatConfig | None = None
    router_dc: ModelSelectionRouterDCCompatConfig | None = None
    automix: ModelSelectionAutoMixCompatConfig | None = None
    hybrid: ModelSelectionHybridCompatConfig | None = None
    ml: ModelSelectionMLCompatConfig | None = None
