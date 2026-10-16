"""Content safety signal contracts, independent of prompt attack detection."""

from pydantic import BaseModel, ConfigDict, Field, model_validator

MIN_SAFETY_LABELS = 2


def _labels_valid(labels: list[str]) -> bool:
    return (
        bool(labels)
        and len(set(labels)) == len(labels)
        and all(
            label and label == label.strip() and ":" not in label for label in labels
        )
    )


class SafetyHazardRule(BaseModel):
    """Optional multi-label hazard filter evaluated only after an unsafe verdict."""

    model_config = ConfigDict(extra="forbid")
    model: str = ""
    labels: list[str] = Field(min_length=2)
    categories: list[str] = Field(min_length=1)
    threshold: float = Field(gt=0, le=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_hazard(self):
        if self.model != self.model.strip():
            raise ValueError("hazard model must be trimmed")
        if not _labels_valid(self.labels) or not _labels_valid(self.categories):
            raise ValueError(
                "hazard labels and categories must be unique trimmed labels without ':'"
            )
        if not set(self.categories) <= set(self.labels):
            raise ValueError("hazard categories must select declared labels")
        return self


class SafetyRule(BaseModel):
    """Sum unsafe label probabilities, optionally gated by hazard categories."""

    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1)
    description: str | None = None
    model: str = ""
    labels: list[str] = Field(default_factory=lambda: ["safe", "unsafe"])
    unsafe_labels: list[str] = Field(default_factory=lambda: ["unsafe"])
    threshold: float = Field(gt=0, le=1, allow_inf_nan=False)
    hazard: SafetyHazardRule | None = None

    @model_validator(mode="after")
    def validate_safety(self):
        if self.name != self.name.strip() or ":" in self.name:
            raise ValueError("safety name must be trimmed and cannot contain ':'")
        if self.model != self.model.strip():
            raise ValueError("safety model must be trimmed")
        # Match canonical Go defaults for omitted or empty optional lists.
        self.labels = self.labels or ["safe", "unsafe"]
        self.unsafe_labels = self.unsafe_labels or ["unsafe"]
        if (
            len(self.labels) < MIN_SAFETY_LABELS
            or not _labels_valid(self.labels)
            or not _labels_valid(self.unsafe_labels)
        ):
            raise ValueError("safety labels must be unique trimmed labels without ':'")
        if not set(self.unsafe_labels) < set(self.labels):
            raise ValueError(
                "unsafe_labels must select a nonempty proper subset of labels"
            )
        return self
