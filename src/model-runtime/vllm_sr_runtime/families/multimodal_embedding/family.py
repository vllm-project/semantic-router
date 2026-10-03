"""Multimodal embeddings (Phase 3): the Vela 1.0 Omni prepared bundles.

A bundle (``tools/models/vela_omni``) holds four verified ONNX graphs (text,
image, CLAP, audio) and the exact processor constants, described by
``vela_omni_manifest.json``. The family serves text, image and audio
embeddings in one normalized space on ``/v1/embeddings`` through the
``onnxruntime`` engine. The published Hub repositories hold native source, so
a Hub ID resolves to the bundle prepared from its pinned revision
(``VLLM_SR_RUNTIME_PREPARED_DIR``, default ``/opt/router-model-artifacts``).
"""

from __future__ import annotations

import base64
import io
import json
import math
import os
import wave
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ...errors import PackageError
from ...heads import embedding
from ...plugins.base import (
    BackboneSpec,
    DtypePolicy,
    EmbeddingInfo,
    EncoderBatch,
    EngineModel,
    LoadedModel,
    ModelFamily,
    ModelInfo,
    ModelSpec,
    PackageRef,
    SurfacePlan,
    SurfaceRequest,
    UnsupportedSurfaceError,
    VerifiedPackage,
)
from ...registry.tables import omni as pins
from . import bundle as bundles
from .processors import AudioFeatures, AudioProcessor, ImageProcessor, TextProcessor

PREPARED_DIR_ENV = "VLLM_SR_RUNTIME_PREPARED_DIR"
DEFAULT_PREPARED_DIR = "/opt/router-model-artifacts"
UNIT_NORM_TOLERANCE = 0.005
GOLDEN_TEXT = "Route this request to the model that answers it best."
LATE_TONE_SECONDS = 0.3


def prepared_dir() -> Path:
    return Path(os.environ.get(PREPARED_DIR_ENV, DEFAULT_PREPARED_DIR))


def golden_image() -> bytes:
    """A deterministic 23 x 19 PNG with edges and gradients."""
    from PIL import Image

    yy, xx = np.indices((19, 23))
    rgb = np.stack(
        ((xx * 13 + yy * 7) % 256, ((xx // 3 + yy // 5) % 2) * 255, (xx * yy) % 256), -1
    )
    buffer = io.BytesIO()
    Image.fromarray(rgb.astype(np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


def golden_audio(rate: int = 16000, seconds: float = 0.5) -> bytes:
    """A deterministic 16-bit mono WAV: two tones and a late third one."""
    t = np.arange(round(rate * seconds)) / rate
    tone = 0.2 * np.sin(2 * math.pi * 317 * t) + 0.1 * np.sin(
        2 * math.pi * 1793 * t * (t > LATE_TONE_SECONDS)
    )
    samples = np.round(np.clip(tone, -1, 1) * 32767).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(rate)
        writer.writeframes(samples.tobytes())
    return buffer.getvalue()


class MultimodalEmbeddingFamily(ModelFamily):
    name = "multimodal_embedding"
    surfaces = frozenset({"embeddings"})

    @classmethod
    def descriptor(cls) -> dict[str, Any]:
        return {
            "surfaces": sorted(cls.surfaces),
            "formats": ["vela-omni-bundle/1"],
            "modalities": ["text", "image", "audio"],
            "engines": ["onnxruntime"],
        }

    def detect(self, package: PackageRef) -> bool:
        if bundles.is_bundle(package.root):
            return True
        return package.repo_id is not None and pins.lookup(package.repo_id) is not None

    def fetch(self, package: PackageRef) -> PackageRef:
        """A Hub ID resolves to the bundle prepared from its pinned source revision."""
        if bundles.is_bundle(package.root):
            return package
        pinned = pins.lookup(package.repo_id or "")
        if pinned is None:
            raise PackageError(f"{package.repo_id} has no prepared Omni bundle")
        root = prepared_dir() / pins.bundle_name(pinned)
        if not bundles.is_bundle(root):
            raise PackageError(
                f"no prepared bundle at {root}: build it with tools/models/vela_omni "
                f"(prepare.py --variants {pins.variant(pinned)}) or set {PREPARED_DIR_ENV}"
            )
        return PackageRef(root=root, repo_id=package.repo_id, revision=package.revision)

    def verify(self, package: PackageRef) -> VerifiedPackage:
        verified = bundles.load(package.root)
        repo_id, revision = verified.source
        pinned = pins.lookup(repo_id)
        if package.repo_id is not None and (
            package.repo_id.lower() != repo_id.lower()
            or (package.revision and package.revision != revision)
        ):
            raise PackageError(
                f"the bundle was prepared from {repo_id}@{revision}, "
                f"not {package.repo_id}@{package.revision}"
            )
        if pinned is not None and pinned.revision != revision:
            raise PackageError(
                f"the bundle was prepared from {repo_id}@{revision}; "
                f"the built-in pin is {pinned.revision}"
            )
        return VerifiedPackage(
            ref=PackageRef(verified.root, repo_id, revision),
            family=self.name,
            model_name=repo_id.split("/")[-1],
            manifest=verified.manifest,
            manifest_sha256=verified.manifest_sha256,
            model_sha256=verified.model_sha256,
            max_input_tokens=verified.contract.max_tokens,
            licence=pins.LICENCE if pinned else None,
            details={"bundle": verified},
        )

    def describe(self, package: VerifiedPackage) -> ModelSpec:
        verified: bundles.OmniBundle = package.details["bundle"]
        return ModelSpec(
            name=package.model_name,
            backbone=BackboneSpec(model_type="vela_omni", config={}, weight_files=()),
            dtype=DtypePolicy(autocast=None, bf16_resident=False),
            max_input_tokens=package.max_input_tokens,
            graphs=verified.graphs,
            encoder=True,
        )

    def load(
        self, package: VerifiedPackage, spec: ModelSpec, engine_model: EngineModel
    ) -> OmniModel:
        verified: bundles.OmniBundle = package.details["bundle"]
        config = json.loads(
            verified.file(verified.processors["audio"]["file"]).read_text(
                encoding="utf-8"
            )
        )
        text = TextProcessor(verified)
        info = ModelInfo(
            id=package.model_name,
            family=self.name,
            repo=package.ref.repo_id,
            revision=package.ref.revision,
            model_sha256=package.model_sha256,
            manifest_sha256=package.manifest_sha256,
            surfaces=tuple(sorted(self.surfaces)),
            question_types=(),
            limits={"max_input_tokens": package.max_input_tokens},
            licence=package.licence,
            parameters=engine_model.parameter_count(),
            dtype="fp32",
            embedding=EmbeddingInfo(
                dimensions=(verified.contract.dimension,),
                layers=(),
                modalities=("text", "image", "audio"),
                pooling=verified.contract.text_pooling,
                input_types=text.input_types,
            ),
        )
        return OmniModel(
            info,
            engine_model,
            text,
            ImageProcessor(verified),
            AudioProcessor(verified, config),
        )

    def golden(self, package: VerifiedPackage) -> list[dict[str, Any]]:
        """One request through all four graphs; references per device class when recorded."""
        repo_id, revision = package.details["bundle"].source
        pinned = pins.lookup(repo_id)
        expected = {}
        if pinned is not None and pinned.revision == revision:
            expected = dict(pinned.golden_answers)
        image = base64.b64encode(golden_image()).decode()
        sound = base64.b64encode(golden_audio()).decode()
        body = {
            "input": [
                GOLDEN_TEXT,
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image}"},
                },
                {
                    "type": "input_audio",
                    "input_audio": {"data": sound, "format": "wav"},
                },
            ]
        }
        return [{"surface": "embeddings", "body": body, "expected": expected}]


class OmniModel(LoadedModel):
    """One bundle on the onnxruntime engine; each input runs its modality's graphs."""

    def __init__(
        self,
        info: ModelInfo,
        engine_model: EngineModel,
        text: TextProcessor,
        image: ImageProcessor,
        audio: AudioProcessor,
    ):
        self.info = info
        self.engine_model = engine_model
        self.text = text
        self.image = image
        self.audio = audio
        assert info.embedding is not None
        self.dimension = info.embedding.dimensions[0]

    def plan_surface(self, surface: str, request: SurfaceRequest) -> SurfacePlan:
        if surface != "embeddings":
            raise UnsupportedSurfaceError(surface, self.info.id)
        assert self.info.embedding is not None
        parsed = embedding.parse_request(
            request, self.info.embedding, self.info.limits["max_input_tokens"]
        )
        if parsed.overflow != "reject":
            raise ValueError(
                "this model rejects over-long input (options.overflow: reject)"
            )
        items = [self._item(entry, parsed) for entry in parsed.inputs]
        rep = embedding.representation(self.info.model_sha256, 0, self.dimension, True)
        return embedding.plan(request, parsed, items, rep)

    def _item(
        self, entry: embedding.EmbeddingInput, parsed: embedding.EmbeddingRequest
    ) -> tuple[embedding.EmbedItem, dict[str, Any] | None] | str:
        if entry.error is not None:
            return entry.error
        if entry.modality == "text":
            assert entry.text is not None
            encoded = self.text.encode(entry.text, parsed.max_tokens, parsed.input_type)
            if isinstance(encoded, str):
                return encoded
            ids, usage = encoded
            key = embedding.content_key(self.info.model_sha256, "text", ids)
            return embedding.EmbedItem(entry.index, "text", ids, key), usage
        assert entry.data is not None
        key = embedding.content_key(self.info.model_sha256, entry.modality, entry.data)
        if entry.modality == "image":
            pixels = self.image.pixels(entry.data)
            if isinstance(pixels, str):
                return pixels
            return (
                embedding.EmbedItem(
                    entry.index, "image", [], key, {"pixel_values": pixels}
                ),
                None,
            )
        features = self.audio.features(entry.data, entry.media_type)
        if isinstance(features, str):
            return features
        features_by_graph = {"clap": features.clap, "whisper": features.whisper}
        return (
            embedding.EmbedItem(entry.index, "audio", [], key, features_by_graph),
            None,
        )

    def run(self, items: list[Any], shared_prefix: int = 0) -> list[Any]:
        return [self._embed(item) for item in items]

    def finish_surface(self, plan: SurfacePlan, results: Any) -> dict[str, Any]:
        return embedding.finish(plan, results)

    def _graph(
        self, name: str, size: int, ids: list[int] | None = None, **inputs: np.ndarray
    ) -> np.ndarray | None:
        """One graph's embedding, or None unless it is a finite unit vector of ``size``."""
        tokens = torch.tensor([ids or [0]], dtype=torch.long)
        batch = EncoderBatch(
            input_ids=tokens,
            attention_mask=torch.ones_like(tokens),
            graph=name,
            graph_inputs={
                key: torch.from_numpy(value) for key, value in inputs.items()
            },
            outputs=("embedding",),
        )
        vector = (
            self.engine_model.encode(batch).outputs["embedding"].numpy().reshape(-1)
        )
        if vector.shape != (size,) or not np.isfinite(vector).all():
            return None
        if abs(float(np.linalg.norm(vector)) - 1) > UNIT_NORM_TOLERANCE:
            return None
        return vector

    def _embed(self, item: embedding.EmbedItem) -> list[float] | None:
        if item.modality == "text":
            vector = self._graph("text", self.dimension, item.ids)
        elif item.modality == "image":
            vector = self._graph("image", self.dimension, **item.features)
        else:
            vector = self._audio(
                AudioFeatures(item.features["clap"], item.features["whisper"])
            )
        return None if vector is None else vector.astype(np.float32).tolist()

    def _audio(self, features: AudioFeatures) -> np.ndarray | None:
        windows = []
        for window in features.clap:
            vector = self._graph("clap", bundles.CLAP_DIMENSION, input_features=window)
            if vector is None:
                return None
            windows.append(vector)
        clap = np.sum(windows, axis=0, dtype=np.float32)
        if len(windows) > 1:
            clap = clap / np.float32(len(windows))
            norm = np.sqrt(np.sum(clap * clap, dtype=np.float32))
            if not norm > embedding.NORM_EPSILON:
                return None
            clap = clap / norm
        return self._graph(
            "audio",
            self.dimension,
            input_features=features.whisper,
            clap_embedding=clap[None].astype(np.float32),
        )


__all__ = ["MultimodalEmbeddingFamily", "OmniModel", "golden_audio", "golden_image"]
