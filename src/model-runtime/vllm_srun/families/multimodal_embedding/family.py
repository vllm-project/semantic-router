"""Multimodal embeddings (Phase 3): Vela 1.0 Omni Nano and Mini.

Omni embeds text, images and audio into one normalized space on
``/v1/embeddings``. Two package formats serve it:

- **The published repository** (``package.py``): the pinned revision's
  ``model.safetensors`` and configs, verified by the SHA-256 of every file
  the family reads. Its text, image, Whisper and CLAP towers run on the
  native engine (``ModelSpec.towers``) and its readouts here; this is what a
  Hub ID resolves to.
- **A prepared bundle** (``bundle.py``): four ONNX graphs exported by
  ``tools/models/vela_omni`` and their processor constants, served on the
  ``onnxruntime`` engine (install the ``onnx`` extra) when the model is a
  bundle directory.
"""

from __future__ import annotations

import base64
import importlib.util
import io
import json
import math
import wave
from typing import Any

import numpy as np

from ...accel import onednn
from ...accel.kernels import CONTIGUOUS
from ...errors import PackageError
from ...plugins.base import (
    BackboneSpec,
    DtypePolicy,
    EmbeddingInfo,
    EngineModel,
    ModelFamily,
    ModelInfo,
    ModelSpec,
    PackageRef,
    VerifiedPackage,
)
from ...registry.artifacts import named_files, sha256_json
from ...registry.resolve import fetch
from ...registry.tables import omni as pins
from . import audio
from . import bundle as bundles
from . import package as snapshots
from .model import GraphOmniModel, NativeOmniModel, OmniModel, batch_invariant
from .processors import AudioProcessor, ImageProcessor, TextProcessor
from .readout import OmniReadout

# A media input's scheduler cost in text tokens of the same model: its CPU
# forward takes about as long as a text that long (measured on 16 cores).
MEDIA_COST = {
    "nano": {"image": 1200, "audio": 1600},
    "mini": {"image": 300, "audio": 550},
}
# CPU threads per graph of a prepared bundle (ModelSpec.graph_threads). Nano's
# text graph runs up to CONCURRENT_INPUTS texts at once, and on 16 cores 12
# threads leave a core to each concurrent caller: 16 oversubscribe the cores
# (p50 +0.5 ms against legacy, 3 % fewer texts per second with four callers),
# and 8 slow the texts of 64-104 tokens that set its p95. Every other graph
# needs all 16 for one request (Nano image 109 ms, 163 on 8; Mini text 33 ms,
# 41-45 on 8).
GRAPH_THREADS: dict[str, dict[str, int]] = {"nano": {"text": 12}, "mini": {}}
# Idle-thread spin per graph (ModelSpec.graph_spin_us): with the engine's 2 ms
# some of Nano's 12 text threads sleep inside a 2-8 ms run (p50 0.3-0.4 ms
# slower than with 10 ms).
GRAPH_SPIN_US: dict[str, dict[str, int]] = {"nano": {"text": 10_000}, "mini": {}}
# Import name -> distribution, per format.
EXTRAS = {
    "snapshot": {"PIL": "Pillow"},
    "bundle": {"onnxruntime": "onnxruntime", "PIL": "Pillow"},
}
INSTALL = {"snapshot": "multimodal", "bundle": "multimodal,onnx"}
# The published model computes in FP32 on every device.
EXACT_DTYPE = DtypePolicy(
    weights="float32", autocast=None, head="float32", bf16_resident=False
)
# oneDNN's packed FP32 linear (x86) and GeGLU on contiguous rows: each row's
# result is then the same alone or in a batch (the load-time probe checks).
KERNELS = {"linear": onednn.PACKED, "geglu": CONTIGUOUS}
IDENTITY_FORMAT = "vela-omni/1"
GOLDEN_TEXT = "Route this request to the model that answers it best."
LATE_TONE_SECONDS = 0.3


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


def identity(files: dict[str, str]) -> str:
    """A published snapshot's identity: a digest of its loaded files' digests."""
    return sha256_json({"format": IDENTITY_FORMAT, "files": files})


class MultimodalEmbeddingFamily(ModelFamily):
    name = "multimodal_embedding"
    surfaces = frozenset({"embeddings"})
    builtin_table = "vllm_srun.registry.tables.omni"
    fixture_writer = "vllm_srun.testing.multimodal_embedding"

    @classmethod
    def descriptor(cls) -> dict[str, Any]:
        return {
            "surfaces": sorted(cls.surfaces),
            "formats": [IDENTITY_FORMAT, "vela-omni-bundle/1"],
            "modalities": ["text", "image", "audio"],
            "engines": ["native", "onnxruntime"],
        }

    def detect(self, package: PackageRef) -> bool:
        if bundles.is_bundle(package.root) or snapshots.detect(package.root):
            return True
        return package.repo_id is not None and pins.lookup(package.repo_id) is not None

    def fetch(self, package: PackageRef) -> PackageRef:
        """A Hub package that is not a built-in pin downloads the files this family reads."""
        if bundles.is_bundle(package.root) or pins.lookup(package.repo_id or ""):
            return package
        return fetch(
            package,
            list(snapshots.FILES),
            cache_dir=self.options.cache_dir,
            offline=self.options.offline,
        )

    def verify(self, package: PackageRef) -> VerifiedPackage:
        if bundles.is_bundle(package.root):
            return self._verify_bundle(package)
        files = named_files(package.root, snapshots.FILES)
        model_sha256 = identity(files)
        known = pins.lookup(package.repo_id or "")
        if known is not None and known.revision != package.revision:
            known = None
        if known is not None:
            if dict(known.files) != files:
                changed = sorted(set(known.files.items()) ^ set(files.items()))
                raise PackageError(
                    f"{package.repo_id}@{package.revision} files differ from the pinned digests: "
                    f"{sorted({name for name, _ in changed})}"
                )
            if known.model_sha256 != model_sha256:
                raise PackageError(
                    f"{package.repo_id} identity differs from the pinned identity"
                )
        omni = snapshots.read(package.root)
        snapshots.weight_names(omni)
        if known is not None and known.loaded_parameters != omni.parameters:
            raise PackageError(
                f"{package.repo_id} declares {omni.parameters:,} parameters, not {known.loaded_parameters:,}"
            )
        name = (package.repo_id or package.root.name).rsplit("/", 1)[-1]
        return VerifiedPackage(
            ref=package,
            family=self.name,
            model_name=name,
            manifest={},
            manifest_sha256="",
            model_sha256=model_sha256,
            max_input_tokens=omni.contract.max_tokens,
            licence=pins.LICENCE if known else None,
            loaded_parameters=omni.parameters,
            details={
                "snapshot": omni,
                "files": files,
                "verification": "builtin" if known else "local",
            },
        )

    def _verify_bundle(self, package: PackageRef) -> VerifiedPackage:
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

    @staticmethod
    def _require(kind: str) -> None:
        missing = [
            name
            for module, name in EXTRAS[kind].items()
            if importlib.util.find_spec(module) is None
        ]
        if missing:
            raise RuntimeError(
                f"Omni {'bundles' if kind == 'bundle' else 'models'} need {' and '.join(missing)}: "
                f"pip install './src/model-runtime[{INSTALL[kind]}]' from a repository checkout"
            )

    def describe(self, package: VerifiedPackage) -> ModelSpec:
        if "bundle" in package.details:
            self._require("bundle")
            verified: bundles.OmniBundle = package.details["bundle"]
            return ModelSpec(
                name=package.model_name,
                backbone=BackboneSpec(
                    model_type="vela_omni", config={}, weight_files=()
                ),
                dtype=DtypePolicy(autocast=None, bf16_resident=False),
                max_input_tokens=package.max_input_tokens,
                graphs=verified.graphs,
                encoder=True,
                graph_threads=GRAPH_THREADS[verified.variant],
                graph_spin_us=GRAPH_SPIN_US[verified.variant],
            )
        self._require("snapshot")
        omni: snapshots.OmniPackage = package.details["snapshot"]
        layout, weights = omni.layout, (omni.weights,)
        return ModelSpec(
            name=package.model_name,
            backbone=BackboneSpec(
                layout.text_type, omni.text_config, weights, layout.text
            ),
            dtype=EXACT_DTYPE,
            max_input_tokens=package.max_input_tokens,
            encoder=True,
            kernel_variants=KERNELS,
            towers={
                "image": BackboneSpec(
                    "siglip_vision_model", omni.vision_config, weights, layout.image
                ),
                "speech": BackboneSpec(
                    "whisper_encoder", omni.speech_config, weights, layout.speech
                ),
                "clap": BackboneSpec(
                    "clap_audio_model", omni.clap_config, weights, layout.clap
                ),
            },
        )

    def load(
        self, package: VerifiedPackage, spec: ModelSpec, engine_model: EngineModel
    ) -> OmniModel:
        if "bundle" in package.details:
            return self._load_bundle(package, engine_model)
        omni: snapshots.OmniPackage = package.details["snapshot"]
        contract = omni.contract
        readout = OmniReadout(omni)
        parameters = engine_model.parameter_count() + sum(
            parameter.numel() for parameter in readout.parameters()
        )
        if parameters != omni.parameters:
            raise PackageError(
                f"loaded {parameters:,} parameters; the model declares {omni.parameters:,}"
            )
        readout = engine_model.place(readout).eval()
        text = TextProcessor(
            omni.tokenizer,
            strip_whitespace=omni.layout.strip_whitespace,
            instruction_api=contract.instruction_api,
        )
        processors = (
            text,
            ImageProcessor.from_preprocessor(omni.image),
            AudioProcessor(
                contract.sample_rates,
                audio.MAX_RATE,
                audio.Spectrum.whisper(omni.speech_features),
                audio.Spectrum.clap_window(omni.clap_features),
            ),
        )
        info = self._info(package, text, contract, parameters)
        model = NativeOmniModel(
            info,
            engine_model,
            *processors,
            MEDIA_COST[omni.variant],
            readout,
            packed_text=False,
        )
        model.packed_text = engine_model.batch_invariant and batch_invariant(
            model, int(omni.text_config["vocab_size"])
        )
        model.batch_invariant = True
        return model

    def _load_bundle(
        self, package: VerifiedPackage, engine_model: EngineModel
    ) -> OmniModel:
        verified: bundles.OmniBundle = package.details["bundle"]
        config = json.loads(
            verified.file(verified.processors["audio"]["file"]).read_text(
                encoding="utf-8"
            )
        )
        text = TextProcessor.from_bundle(verified)
        info = self._info(
            package, text, verified.contract, engine_model.parameter_count()
        )
        return GraphOmniModel(
            info,
            engine_model,
            text,
            ImageProcessor.from_bundle(verified),
            AudioProcessor.from_bundle(verified, config),
            MEDIA_COST[verified.variant],
        )

    def _info(
        self,
        package: VerifiedPackage,
        text: TextProcessor,
        contract: bundles.Variant,
        parameters: int,
    ) -> ModelInfo:
        return ModelInfo(
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
            parameters=parameters,
            dtype="fp32",
            embedding=EmbeddingInfo(
                dimensions=(contract.dimension,),
                layers=(),
                modalities=("text", "image", "audio"),
                pooling=contract.text_pooling,
                input_types=text.input_types,
            ),
        )

    def golden(self, package: VerifiedPackage) -> list[dict[str, Any]]:
        """One request through every tower; references per device class when recorded."""
        repo_id, revision = package.ref.repo_id, package.ref.revision
        pinned = pins.lookup(repo_id or "")
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


__all__ = ["MultimodalEmbeddingFamily", "OmniModel", "golden_audio", "golden_image"]
