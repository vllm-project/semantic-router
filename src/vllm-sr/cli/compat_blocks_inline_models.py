"""Typed inline model compatibility blocks for the TD001 CLI migration."""

from pydantic import BaseModel, ConfigDict


class BertModelCompatConfig(BaseModel):
    """Typed schema for bert_model."""

    model_config = ConfigDict(extra="forbid")

    model_id: str | None = None
    threshold: float | None = None
    use_cpu: bool | None = None


class ClassifierCategoryModelCompatConfig(BaseModel):
    """Typed schema for classifier.category_model."""

    model_config = ConfigDict(extra="forbid")

    model_id: str | None = None
    threshold: float | None = None
    use_cpu: bool | None = None
    use_modernbert: bool | None = None
    use_mmbert_32k: bool | None = None
    category_mapping_path: str | None = None
    fallback_category: str | None = None


class ClassifierMCPCategoryModelCompatConfig(BaseModel):
    """Typed schema for classifier.mcp_category_model."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool | None = None
    transport_type: str | None = None
    command: str | None = None
    args: list[str] | None = None
    env: dict[str, str] | None = None
    url: str | None = None
    tool_name: str | None = None
    threshold: float | None = None
    timeout_seconds: int | None = None


class ClassifierPIIModelCompatConfig(BaseModel):
    """Typed schema for classifier.pii_model."""

    model_config = ConfigDict(extra="forbid")

    model_id: str | None = None
    threshold: float | None = None
    use_cpu: bool | None = None
    use_mmbert_32k: bool | None = None
    pii_mapping_path: str | None = None


class ClassifierPreferenceModelCompatConfig(BaseModel):
    """Typed schema for classifier.preference_model."""

    model_config = ConfigDict(extra="forbid")

    use_contrastive: bool | None = None
    embedding_model: str | None = None


class ClassifierCompatConfig(BaseModel):
    """Typed schema for classifier."""

    model_config = ConfigDict(extra="forbid")

    category_model: ClassifierCategoryModelCompatConfig | None = None
    mcp_category_model: ClassifierMCPCategoryModelCompatConfig | None = None
    pii_model: ClassifierPIIModelCompatConfig | None = None
    preference_model: ClassifierPreferenceModelCompatConfig | None = None


class FeedbackDetectorCompatConfig(BaseModel):
    """Typed schema for feedback_detector."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool | None = None
    model_id: str | None = None
    threshold: float | None = None
    use_cpu: bool | None = None
    use_modernbert: bool | None = None
    use_mmbert_32k: bool | None = None
    feedback_mapping_path: str | None = None


class HallucinationFactCheckModelCompatConfig(BaseModel):
    """Typed schema for hallucination_mitigation.fact_check_model."""

    model_config = ConfigDict(extra="forbid")

    model_id: str | None = None
    threshold: float | None = None
    use_cpu: bool | None = None
    use_mmbert_32k: bool | None = None


class HallucinationModelCompatConfig(BaseModel):
    """Typed schema for hallucination_mitigation.hallucination_model."""

    model_config = ConfigDict(extra="forbid")

    model_id: str | None = None
    threshold: float | None = None
    use_cpu: bool | None = None
    min_span_length: int | None = None
    min_span_confidence: float | None = None
    context_window_size: int | None = None
    enable_nli_filtering: bool | None = None
    nli_entailment_threshold: float | None = None


class HallucinationNLIModelCompatConfig(BaseModel):
    """Typed schema for hallucination_mitigation.nli_model."""

    model_config = ConfigDict(extra="forbid")

    model_id: str | None = None
    threshold: float | None = None
    use_cpu: bool | None = None


class HallucinationMitigationCompatConfig(BaseModel):
    """Typed schema for hallucination_mitigation."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool | None = None
    fact_check_model: HallucinationFactCheckModelCompatConfig | None = None
    hallucination_model: HallucinationModelCompatConfig | None = None
    nli_model: HallucinationNLIModelCompatConfig | None = None
    on_hallucination_detected: str | None = None


class ModalityDetectorClassifierCompatConfig(BaseModel):
    """Typed schema for modality_detector.classifier."""

    model_config = ConfigDict(extra="forbid")

    model_path: str | None = None
    use_cpu: bool | None = None


class ModalityDetectorCompatConfig(BaseModel):
    """Typed schema for modality_detector."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool | None = None
    prompt_prefixes: list[str] | None = None
    method: str | None = None
    classifier: ModalityDetectorClassifierCompatConfig | None = None
    keywords: list[str] | None = None
    both_keywords: list[str] | None = None
    confidence_threshold: float | None = None
    lower_threshold_ratio: float | None = None
