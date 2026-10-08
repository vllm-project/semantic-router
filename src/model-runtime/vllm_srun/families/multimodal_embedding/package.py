"""Vela 1.0 Omni as published: one ``model.safetensors`` and the towers' configs, run on the native engine.

The Hub repositories hold the model's Python source (never downloaded,
imported or executed) next to the weights of every tower, the readout
projections and the CLAP residual in one ``model.safetensors``, and the
towers' Transformers configs and preprocessor configs under ``components/``.
The family reads only the files in ``FILES``, whose SHA-256 the built-in
table pins per revision. ``LAYOUTS`` says where each variant keeps its towers
and readouts in the checkpoint; the towers run on the native engine and the
readouts in the family (``readout.py``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...errors import PackageError
from ...registry.artifacts import safetensors_header
from . import audio
from .bundle import VARIANTS, Variant

CONFIG = "config.json"
WEIGHTS = "model.safetensors"
TEXT = "components/text"
IMAGE = "components/image"
SPEECH = "components/audio"
CLAP = "components/audio_clap"
# Every file the family reads; the published Python source is never fetched.
FILES = (
    CONFIG,
    WEIGHTS,
    f"{TEXT}/config.json",
    f"{TEXT}/tokenizer.json",
    f"{IMAGE}/config.json",
    f"{IMAGE}/preprocessor_config.json",
    f"{SPEECH}/config.json",
    f"{SPEECH}/preprocessor_config.json",
    f"{CLAP}/config.json",
    f"{CLAP}/preprocessor_config.json",
)
CLAP_WEIGHTS = "audio_residual.clap."


@dataclass(frozen=True)
class Layout:
    """Where one variant keeps each tower and readout in ``model.safetensors``, and its ``config.json``."""

    config: dict[str, Any]
    text_type: str
    text: str
    image: str
    speech: str
    image_projection: str
    speech_projection: str
    normalize_pooled_image: bool
    strip_whitespace: bool
    clap: str = CLAP_WEIGHTS + "audio_model.audio_encoder."
    clap_projection: str = CLAP_WEIGHTS + "audio_projection."
    residual: str = "audio_residual."


LAYOUTS = {
    "nano": Layout(
        config={
            "architectures": ["VelaOmni"],
            "variant": "nano",
            "format_version": 9,
            "embedding_dim": 384,
            "max_text_length": 512,
            "inference_profile": "single_modality",
            "text_pooling": "cls",
            "text_projection": "identity",
            "audio_backend": "tiny_clap_residual_v1",
            "audio_pooling": "mean",
            "audio_projection": "linear",
            "residual_audio_readout": "tiny_full_mean_clap_endpoint_zero_linear",
        },
        text_type="bert",
        text="text_encoder.encoder.",
        image="image_encoder.vision_encoder.",
        speech="audio_encoder.encoder.",
        image_projection="image_encoder.projection.",
        speech_projection="audio_encoder.projection.",
        normalize_pooled_image=False,
        strip_whitespace=False,
    ),
    "mini": Layout(
        config={
            "architectures": ["VelaOmni"],
            "variant": "mini",
            "format_version": 7,
            "embedding_dim": 768,
            "max_text_length": 32768,
            "text_backend": "qwen3_mrl768",
            "text_pooling": "last_token_mrl768",
            "text_projection": "identity",
            "text_instruction_api": "qwen_optional_instruction_v1",
            "text_prompt": "",
            "image_pooling": "siglip_attention_l2",
            "audio_backend": "medium_clap_residual_v1",
            "audio_pooling": "mean",
            "audio_projection": "identity",
            "residual_audio_readout": "medium_full_mean_clap_endpoint_zero_linear",
        },
        text_type="qwen3",
        text="text_model.0.model.",
        image="image_model.vision_model.",
        speech="audio_model.",
        image_projection="image_proj.",
        speech_projection="audio_proj.",
        normalize_pooled_image=True,
        strip_whitespace=True,
    ),
}
# Text model types per variant's text backbone, as their configs declare them.
TEXT_MODEL_TYPES = {"bert": "bert", "qwen3": "qwen3"}


@dataclass(frozen=True)
class OmniPackage:
    """A verified published snapshot: its variant, the towers' configs and the preprocessor settings."""

    root: Path
    variant: str
    contract: Variant
    layout: Layout
    text_config: dict[str, Any]
    vision_config: dict[str, Any]
    speech_config: dict[str, Any]
    clap_config: dict[str, Any]
    image: dict[str, Any]
    speech_features: dict[str, Any]
    clap_features: dict[str, Any]
    parameters: int

    @property
    def weights(self) -> Path:
        return self.root / WEIGHTS

    @property
    def tokenizer(self) -> Path:
        return self.root / TEXT / "tokenizer.json"


def _json(root: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads((root / name).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PackageError(f"cannot read {name}: {exc}") from exc
    if not isinstance(value, dict):
        raise PackageError(f"{name} must hold a JSON object")
    return value


def detect(root: Path) -> bool:
    """Whether ``root`` holds a published Omni snapshot (its ``config.json`` names the architecture)."""
    try:
        config = json.loads((Path(root) / CONFIG).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return (
        isinstance(config, dict)
        and config.get("architectures") == ["VelaOmni"]
        and config.get("variant") in LAYOUTS
    )


def read(root: Path) -> OmniPackage:
    """The snapshot's variant and configs, checked against the variant's published contract."""
    root = Path(root)
    config = _json(root, CONFIG)
    variant = config.get("variant")
    if variant not in LAYOUTS:
        raise PackageError(f"{CONFIG} names no known Vela Omni variant")
    layout = LAYOUTS[variant]
    changed = sorted(
        key for key, value in layout.config.items() if config.get(key) != value
    )
    if changed:
        raise PackageError(
            f"{CONFIG} differs from the published {variant} model: {changed}"
        )
    parameters = config.get("parameter_count")
    if type(parameters) is not int or parameters <= 0:
        raise PackageError(f"{CONFIG} declares no parameter count")
    contract = VARIANTS[variant]
    text = _json(root, f"{TEXT}/config.json")
    if text.get("model_type") != TEXT_MODEL_TYPES[layout.text_type]:
        raise PackageError(f"the {variant} text tower must be {layout.text_type}")
    width = text.get("hidden_size")
    if (
        not isinstance(width, int)
        or width < contract.dimension
        or (layout.text_type == "bert" and width != contract.dimension)
    ):
        raise PackageError(
            f"the {variant} text tower is {width} wide; its readout needs {contract.dimension}"
        )
    image_config = _json(root, f"{IMAGE}/config.json")
    vision = image_config.get("vision_config")
    if image_config.get("model_type") != "siglip" or not isinstance(vision, dict):
        raise PackageError("the image tower must be a SigLIP vision model")
    image = _json(root, f"{IMAGE}/preprocessor_config.json")
    size = image.get("size") or {}
    if (
        vision.get("image_size") != contract.image_size
        or size.get("height") != contract.image_size
    ):
        raise PackageError(
            f"the {variant} image tower takes {contract.image_size}-pixel images"
        )
    speech = _json(root, f"{SPEECH}/config.json")
    if speech.get("model_type") != "whisper":
        raise PackageError("the speech tower must be a Whisper encoder")
    clap = _json(root, f"{CLAP}/config.json").get("audio_config")
    if not isinstance(clap, dict) or clap.get("model_type") != "clap_audio_model":
        raise PackageError("the CLAP tower must be a CLAP audio model")
    speech_features = _json(root, f"{SPEECH}/preprocessor_config.json")
    clap_features = _json(root, f"{CLAP}/preprocessor_config.json")
    try:
        audio.Spectrum.whisper(speech_features)
        audio.Spectrum.clap_window(clap_features)
    except ValueError as exc:
        raise PackageError(str(exc)) from exc
    return OmniPackage(
        root=root,
        variant=variant,
        contract=contract,
        layout=layout,
        text_config=text,
        vision_config={"model_type": "siglip_vision_model", **vision},
        speech_config=speech,
        clap_config=clap,
        image=image,
        speech_features=speech_features,
        clap_features=clap_features,
        parameters=parameters,
    )


def weight_names(package: OmniPackage) -> dict[str, list[str]]:
    """Every tensor of ``model.safetensors`` by the part that loads it; nothing may be left over.

    Read from the header only. The towers' prefixes and the readouts' tensors
    must cover the checkpoint exactly, so no tensor goes unread.
    """
    layout = package.layout
    owners = {
        "text": layout.text,
        "image": layout.image,
        "speech": layout.speech,
        "clap": layout.clap,
        "image_projection": layout.image_projection,
        "speech_projection": layout.speech_projection,
        "clap_projection": layout.clap_projection,
    }
    residual = {f"{layout.residual}{name}" for name in ("weight", "mean", "scale")}
    parts: dict[str, list[str]] = {name: [] for name in (*owners, "residual")}
    for name in safetensors_header(package.weights):
        if name == "__metadata__":
            continue
        if name in residual:
            parts["residual"].append(name)
            continue
        owner = next(
            (part for part, prefix in owners.items() if name.startswith(prefix)), None
        )
        if owner is None:
            raise PackageError(
                f"{WEIGHTS} holds a tensor no part of the model reads: {name}"
            )
        parts[owner].append(name)
    empty = sorted(part for part, names in parts.items() if not names)
    if empty or len(parts["residual"]) != len(residual):
        raise PackageError(
            f"{WEIGHTS} misses parts of the {package.variant} model: {empty}"
        )
    return parts
