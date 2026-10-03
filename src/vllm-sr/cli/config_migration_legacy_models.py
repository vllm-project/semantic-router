"""Legacy router model aliases and their built-in runtime replacements.

The native bindings served several generations of task models. The model
runtime serves the Vela 1.0 releases, which keep the label names of the
models they replace, so a rewrite never changes what a rule matches. Models
without a runtime equivalent are retired (the NLI explainer) or need an
operator decision; both are reported instead of guessed.
"""

from __future__ import annotations

from dataclasses import dataclass

VELA = "models/Vela-1.0-Encoder-307M-"
OMNI_NANO = "models/vela-1.0-omni-nano"
OMNI_MINI = "models/vela-1.0-omni-mini"
QWEN3_EMBEDDING = "models/mom-embedding-pro"


@dataclass(frozen=True)
class Replacement:
    """A legacy model's runtime replacement.

    ``mapping_file`` names the label map shipped in the replacement package, so
    a module mapping path that pointed into the legacy directory follows the
    model. ``reembed`` marks embedding replacements whose vectors live in a
    different space.
    """

    target: str
    note: str
    mapping_file: str = ""
    reembed: bool = False


_DOMAIN = Replacement(
    VELA + "Domain",
    "Vela Domain keeps the 14 MMLU-Pro domain labels",
    mapping_file="category_mapping.json",
)
_PII = Replacement(
    VELA + "PII",
    "Vela PII keeps the 35 BIO labels of the 17 PII types",
    mapping_file="pii_mapping.json",
)
_GUARD = Replacement(
    VELA + "Guard",
    "Vela Guard keeps the benign / jailbreak labels",
    mapping_file="jailbreak_type_mapping.json",
)
_FACT_CHECK = Replacement(
    VELA + "FactCheck",
    "Vela FactCheck keeps the FACT_CHECK_NEEDED / NO_FACT_CHECK_NEEDED labels",
)
_HALU = Replacement(
    VELA + "Halu",
    "Vela Halu reads the same context, question and answer and returns answer spans",
)
_FEEDBACK = Replacement(
    VELA + "Feedback",
    "Vela Feedback keeps the four feedback labels and adds NO_FEEDBACK for messages without feedback",
)
_MODALITY = Replacement(
    VELA + "Modality",
    "Vela Modality keeps the AR / DIFFUSION / BOTH labels",
)
_EMBEDDING = Replacement(
    VELA + "Embedding",
    "Vela Embedding replaces this embedding model; stored vectors must be re-embedded "
    "and similarity thresholds re-checked",
    reembed=True,
)
_OMNI = Replacement(
    OMNI_NANO,
    "Vela Omni Nano replaces this multimodal embedding model; stored vectors must be "
    "re-embedded and similarity thresholds re-checked",
    reembed=True,
)

# Every legacy local path and alias the router registry accepted, by replacement.
_LEGACY_ALIASES: dict[Replacement, tuple[str, ...]] = {
    _DOMAIN: (
        "models/mom-domain-classifier",
        "domain-classifier",
        "intent-classifier",
        "category-classifier",
        "category_classifier_modernbert-base_model",
        "lora_intent_classifier_bert-base-uncased_model",
        "models/mmbert32k-intent-classifier-lora",
        "mmbert32k-intent",
        "mmbert-32k-intent",
        "intent-classifier-32k",
        "models/mmbert32k-intent-classifier-merged",
        "mmbert32k-intent-merged",
        "intent-classifier-32k-merged",
    ),
    _PII: (
        "models/mom-pii-classifier",
        "pii-detector",
        "pii-classifier",
        "privacy-guard",
        "lora_pii_detector_bert-base-uncased_model",
        "models/mom-mmbert-pii-detector",
        "mmbert-pii-detector",
        "mmbert-pii-detector-merged",
        "pii_classifier_modernbert-base_presidio_token_model",
        "pii_classifier_modernbert-base_model",
        "pii_classifier_modernbert_model",
        "pii_classifier_modernbert_ai4privacy_token_model",
        "models/mmbert32k-pii-detector-merged",
        "mmbert32k-pii-merged",
        "pii-detector-32k-merged",
        "models/mmbert32k-pii-detector-lora",
        "mmbert32k-pii",
        "mmbert-32k-pii",
        "pii-detector-32k",
    ),
    _GUARD: (
        "models/mom-jailbreak-classifier",
        "jailbreak-detector",
        "prompt-guard",
        "safety-classifier",
        "jailbreak_classifier_modernbert-base_model",
        "lora_jailbreak_classifier_bert-base-uncased_model",
        "jailbreak_classifier_modernbert_model",
        "models/mmbert32k-jailbreak-detector-lora",
        "mmbert32k-jailbreak",
        "mmbert-32k-jailbreak",
        "jailbreak-detector-32k",
        "prompt-guard-32k",
        "models/mmbert32k-jailbreak-detector-merged",
        "mmbert32k-jailbreak-merged",
        "jailbreak-detector-32k-merged",
    ),
    _FACT_CHECK: (
        "models/mom-halugate-sentinel",
        "hallucination-sentinel",
        "halugate-sentinel",
        "models/mmbert32k-factcheck-classifier-lora",
        "mmbert32k-factcheck",
        "mmbert-32k-factcheck",
        "factcheck-classifier-32k",
        "fact-check-32k",
        "models/mmbert32k-factcheck-classifier-merged",
        "mmbert32k-factcheck-merged",
        "factcheck-classifier-32k-merged",
    ),
    _HALU: (
        "models/mom-halugate-detector",
        "hallucination-detector",
        "halugate-detector",
        "lettucedect",
        "KRLabsOrg/lettucedect-base-modernbert-en-v1",
        "models/lettucedect-v2-mmbert-base",
        "hallucination-detector-multilingual",
        "lettucedect-v2-mmbert",
        "KRLabsOrg/lettucedect-v2-mmbert-base",
    ),
    _FEEDBACK: (
        "models/mom-feedback-detector",
        "feedback-detector",
        "user-feedback-classifier",
        "models/mmbert32k-feedback-detector-lora",
        "mmbert32k-feedback",
        "mmbert-32k-feedback",
        "feedback-detector-32k",
        "models/mmbert32k-feedback-detector-merged",
        "mmbert32k-feedback-merged",
        "feedback-detector-32k-merged",
    ),
    _MODALITY: (
        "models/mmbert32k-modality-router-merged",
        "modality-classifier",
        "modality-router",
        "mmbert32k-modality-router",
    ),
    _EMBEDDING: (
        "models/mmbert-embed-32k-2d-matryoshka",
        "mom-embedding-ultra",
        "mmbert-embed-32k-2d-matryoshka",
        "mmbert-embedding",
        "embedding-mmbert",
        "embedding-ultra",
        "models/mom-embedding-flash",
        "embeddinggemma-300m",
        "embedding-flash",
        "google/embeddinggemma-300m",
        "models/mom-embedding-light",
        "all-MiniLM-L12-v2",
        "embedding-light",
        "bert-light",
        "sentence-transformers/all-MiniLM-L12-v2",
    ),
    _OMNI: (
        "models/mom-embedding-multimodal",
        "multi-modal-embed-small",
        "multimodal-embedding",
        "embedding-multimodal",
        "mom-embedding-multimodal",
        "vllm-sr/multi-modal-embed-small",
        "vllm-sr/multi-modal-embed-large",
        "multi-modal-embed-large",
    ),
}

# The NLI explainer has no runtime replacement (design: retired).
RETIRED_NLI_ALIASES = frozenset(
    {
        "models/mom-halugate-explainer",
        "hallucination-explainer",
        "halugate-explainer",
        "nli-explainer",
        "tasksource/ModernBERT-base-nli",
    }
)

# Hub repositories of the registry paths a model_runtime deployment may name.
# A model_runtime artifact is a Hub repository or an absolute package path.
HUB_REPOSITORIES = {
    **{
        VELA + name: "vllm-sr/Vela-1.0-Encoder-307M-" + name
        for name in (
            "Domain",
            "Guard",
            "Safety",
            "Shield",
            "FactCheck",
            "Feedback",
            "Modality",
            "Hazard",
            "PII",
            "Halu",
            "Embedding",
            "Reranker",
        )
    },
    "models/Vela-1.0-Encoder-307M": "vllm-sr/Vela-1.0-Encoder-307M",
    QWEN3_EMBEDDING: "Qwen/Qwen3-Embedding-0.6B",
}

# Omni repositories hold PyTorch source; the runtime serves the verified ONNX
# bundle that tools/models/vela_omni prepares, which router images ship here.
PREPARED_BUNDLE_ROOT = "/opt/router-model-artifacts"
PREPARED_BUNDLES = {
    OMNI_NANO: f"{PREPARED_BUNDLE_ROOT}/vela-1.0-omni-nano",
    OMNI_MINI: f"{PREPARED_BUNDLE_ROOT}/vela-1.0-omni-mini",
}


def _alias_keys(alias: str) -> tuple[str, ...]:
    if "/" in alias:
        return (alias,)
    return (alias, "models/" + alias)


_REPLACEMENTS: dict[str, Replacement] = {
    key: replacement
    for replacement, aliases in _LEGACY_ALIASES.items()
    for alias in aliases
    for key in _alias_keys(alias)
}

_RETIRED_KEYS = frozenset(
    key for alias in RETIRED_NLI_ALIASES for key in _alias_keys(alias)
)


def _normalized(value: str) -> str:
    return value.strip().rstrip("/")


def replacement_for(value: object) -> Replacement | None:
    """The runtime replacement of a legacy model path or alias, if it has one."""
    if not isinstance(value, str):
        return None
    return _REPLACEMENTS.get(_normalized(value))


def is_retired_nli(value: object) -> bool:
    return isinstance(value, str) and _normalized(value) in _RETIRED_KEYS


def runtime_artifact(value: object) -> str | None:
    """The model_runtime artifact of a registry path, after any legacy replacement:
    its Hub repository, or the prepared bundle router images ship."""
    if not isinstance(value, str):
        return None
    path = _normalized(value)
    replacement = _REPLACEMENTS.get(path)
    if replacement is not None:
        path = replacement.target
    return HUB_REPOSITORIES.get(path) or PREPARED_BUNDLES.get(path)


def follow_mapping_path(
    mapping_path: object, legacy_model: str, replacement: Replacement
) -> str | None:
    """Move a label map that lived inside the legacy model's directory."""
    if not isinstance(mapping_path, str) or not mapping_path.strip():
        return None
    prefix = _normalized(legacy_model) + "/"
    if not mapping_path.startswith(prefix):
        return None
    if replacement.mapping_file:
        return f"{replacement.target}/{replacement.mapping_file}"
    return ""
