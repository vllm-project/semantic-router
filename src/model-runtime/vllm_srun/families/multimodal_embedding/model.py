"""Loaded Vela Omni models: one planning path, two ways to run the inputs.

``OmniModel`` validates and renders ``/v1/embeddings`` requests (text tokens,
image pixels, audio features) and assembles the answers. ``NativeOmniModel``
runs the published model's towers on the native engine and its readouts here;
``GraphOmniModel`` runs a prepared bundle's four graphs on ONNX Runtime.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, wait
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch

from ...heads import embedding
from ...plugins.base import (
    EncoderBatch,
    EngineModel,
    LoadedModel,
    ModelInfo,
    SurfacePlan,
    SurfaceRequest,
    UnsupportedSurfaceError,
)
from .bundle import CLAP_DIMENSION
from .processors import AudioFeatures, AudioProcessor, ImageProcessor, TextProcessor
from .readout import OmniReadout

if TYPE_CHECKING:
    from numpy.typing import NDArray

UNIT_NORM_TOLERANCE = 0.005
CONCURRENT_INPUTS = 4
# Token counts of the texts that probe a loaded model's batch invariance.
INVARIANCE_PROBE = (3, 9, 9, 17, 40, 130)


class OmniModel(LoadedModel[embedding.EmbedItem, list[float] | None]):
    """``/v1/embeddings`` over text, image and audio inputs; subclasses run the items."""

    def __init__(
        self,
        info: ModelInfo,
        engine_model: EngineModel,
        text: TextProcessor,
        image: ImageProcessor,
        audio: AudioProcessor,
        media_cost: dict[str, int],
    ):
        self.info = info
        self.engine_model = engine_model
        self.text = text
        self.image = image
        self.audio = audio
        self.media_cost = media_cost
        assert info.embedding is not None
        self.dimension = info.embedding.dimensions[0]

    def plan_surface(
        self, surface: str, request: SurfaceRequest
    ) -> SurfacePlan[embedding.EmbedItem]:
        if surface != "embeddings":
            raise UnsupportedSurfaceError(surface, self.info.id)
        assert self.info.embedding is not None
        parsed = embedding.parse_request(
            request, self.info.embedding, self.info.limits["max_input_tokens"]
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
            encoded = self.text.encode(
                entry.text, parsed.max_tokens, parsed.input_type, parsed.overflow
            )
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
            inputs = {"pixel_values": pixels}
            cost = self.media_cost["image"]
            return (
                embedding.EmbedItem(entry.index, "image", [], key, inputs, cost),
                None,
            )
        features = self.audio.features(entry.data, entry.media_type)
        if isinstance(features, str):
            return features
        by_graph = {"clap": features.clap, "whisper": features.whisper}
        cost = self.media_cost["audio"]
        return embedding.EmbedItem(entry.index, "audio", [], key, by_graph, cost), None

    def finish_surface(
        self, plan: SurfacePlan[embedding.EmbedItem], results: Any
    ) -> dict[str, Any]:
        return embedding.finish(plan, results)

    def close(self) -> None:
        self.image.close()
        super().close()


class NativeOmniModel(OmniModel):
    """The published model's towers on the native engine and its readouts in FP32.

    Texts run through the text backbone, packed back to back in one forward
    when the loaded model computes each the same alone or packed
    (``batch_invariant``, probed at load), else one at a time. Each image runs
    the vision tower alone; each audio input runs CLAP over its windows
    together and Whisper over its features. An input's answer never depends
    on the other inputs of its batch, so the ``exact`` profile may run
    queued requests together.
    """

    packs_rows = True

    def __init__(
        self,
        info: ModelInfo,
        engine_model: EngineModel,
        text: TextProcessor,
        image: ImageProcessor,
        audio: AudioProcessor,
        media_cost: dict[str, int],
        readout: OmniReadout,
        packed_text: bool,
    ):
        super().__init__(info, engine_model, text, image, audio, media_cost)
        self.readout = readout
        self.packed_text = packed_text

    def run(self, items: list[Any]) -> list[Any]:
        results: list[Any] = [None] * len(items)
        texts = [index for index, item in enumerate(items) if item.modality == "text"]
        if texts:
            groups = [texts] if self.packed_text else [[index] for index in texts]
            for group in groups:
                vectors = self._texts([items[index].ids for index in group])
                for index, vector in zip(group, vectors, strict=True):
                    results[index] = vector
        for index, item in enumerate(items):
            if item.modality == "image":
                results[index] = self._image(item)
            elif item.modality == "audio":
                results[index] = self._audio(item)
        return results

    def _texts(self, sequences: list[list[int]]) -> list[list[float] | None]:
        """Embeddings of distinct texts from one packed forward."""
        unique = list(dict.fromkeys(tuple(ids) for ids in sequences))
        lengths = [len(ids) for ids in unique]
        output = self.engine_model.encode(
            EncoderBatch(
                torch.tensor(
                    [token for ids in unique for token in ids], dtype=torch.long
                ),
                None,
                lengths=lengths,
            )
        )
        hidden = next(iter(output.hidden.values()))
        vectors: dict[tuple[int, ...], list[float] | None] = {}
        start = 0
        with torch.inference_mode():
            for ids, length in zip(unique, lengths, strict=True):
                vectors[ids] = _listed(
                    self.readout.text(hidden[start : start + length])
                )
                start += length
        return [vectors[tuple(ids)] for ids in sequences]

    def _tower(self, name: str, **inputs: NDArray[np.float32]) -> torch.Tensor:
        output = self.engine_model.encode(
            EncoderBatch(
                torch.zeros(0, dtype=torch.long),
                None,
                tower=name,
                graph_inputs={
                    key: torch.from_numpy(value) for key, value in inputs.items()
                },
            )
        )
        return next(iter(output.outputs.values()))

    def _image(self, item: embedding.EmbedItem) -> list[float] | None:
        pooled = self._tower("image", pixel_values=item.features["pixel_values"])
        with torch.inference_mode():
            return _listed(self.readout.image(pooled))

    def _audio(self, item: embedding.EmbedItem) -> list[float] | None:
        windows = item.features["clap"].reshape(
            -1, 1, *item.features["clap"].shape[-2:]
        )
        pooled = self._tower("clap", input_features=windows)
        speech = self._tower("speech", input_features=item.features["whisper"])
        with torch.inference_mode():
            clap = self.readout.clap(pooled)
            return None if clap is None else _listed(self.readout.audio(speech, clap))


def _listed(vector: torch.Tensor | None) -> list[float] | None:
    return None if vector is None else vector.reshape(-1).cpu().tolist()


class GraphOmniModel(OmniModel):
    """A prepared bundle on the onnxruntime engine; each input runs its modality's graphs.

    Every input runs its graphs alone, so its vector is the same in any batch:
    the ``exact`` profile hands every queued request to one ``run``, which runs
    up to ``CONCURRENT_INPUTS`` text and audio inputs at once on their graphs'
    ONNX Runtime pools (one short input's run leaves most of a pool idle).
    Images run one after another beside them: one image's run keeps every
    core busy, and two at once only contend for the cores. An input that
    raises fails the whole batch, as any other model's failed forward does:
    ``run`` cancels the batch's inputs that haven't started and waits for the
    running ones before it raises, so none of them runs beside the next batch.
    """

    device_thread = False
    batch_invariant = True

    def __init__(
        self,
        info: ModelInfo,
        engine_model: EngineModel,
        text: TextProcessor,
        image: ImageProcessor,
        audio: AudioProcessor,
        media_cost: dict[str, int],
    ):
        super().__init__(info, engine_model, text, image, audio, media_cost)
        self.inputs = ThreadPoolExecutor(
            max_workers=CONCURRENT_INPUTS, thread_name_prefix="vllm-sr-omni"
        )

    def run(self, items: list[Any]) -> list[Any]:
        if len(items) == 1:
            return [self._embed(items[0])]
        shared: dict[int, Future[list[float] | None]] = {}
        try:
            for index, item in enumerate(items):
                if item.modality != "image":
                    shared[index] = self.inputs.submit(self._embed, item)
            images = {
                index: self._embed(item)
                for index, item in enumerate(items)
                if item.modality == "image"
            }
            return [
                images[index] if index in images else shared[index].result()
                for index in range(len(items))
            ]
        except BaseException:
            for future in shared.values():
                future.cancel()
            wait(shared.values())
            raise

    def close(self) -> None:
        self.inputs.shutdown(wait=True)
        super().close()

    def _graph(
        self,
        name: str,
        size: int,
        ids: list[int] | None = None,
        /,
        **inputs: NDArray[np.float32],
    ) -> NDArray[np.float32] | None:
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

    def _audio(self, features: AudioFeatures) -> NDArray[np.float32] | None:
        windows = []
        for window in features.clap:
            vector = self._graph("clap", CLAP_DIMENSION, input_features=window)
            if vector is None:
                return None
            windows.append(vector)
        clap = cast("NDArray[np.float32]", np.sum(windows, axis=0, dtype=np.float32))
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


def batch_invariant(model: NativeOmniModel, vocab: int) -> bool:
    """Whether texts embed the same alone and packed with others (probed on this host)."""
    generator = torch.Generator().manual_seed(0)
    sequences = [
        torch.randint(5, vocab, (length,), generator=generator).tolist()
        for length in INVARIANCE_PROBE
    ]
    alone = [model._texts([ids])[0] for ids in sequences]
    together = model._texts(sequences[::-1])[::-1]
    return alone == together
