"""The Decision 3.0 model family plugin: d3 code-readout packages, with image and video inputs.

A question is one prompt (``prompt.py``) whose options are listed under
single-token answer codes. The backbone reads it left-padded with noncausal
full attention, and the FP32 readout scores the question's codes at the last
position: softmax over the codes at the package temperature, then the answer.
A request's questions run in request order, eight per forward pass, as the
released runtime runs them. Images (``images.py``) and videos (``videos.py``)
are shared by every question of a request. Without videos, each question of a
pass carries its own copy of the images through the vision tower, as the
released runtime's processor hands them on; with videos, the tower reads the
request's images and videos once and every question reuses their features, as
the d3 runtime does.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, ClassVar

import torch

from ...errors import (
    INVALID_MODEL_OUTPUT,
    INVALID_QUESTION,
    MAX_LENGTH_EXCEEDED,
    PackageError,
    QuestionError,
    question_error,
)
from ...plugins.base import (
    BackboneSpec,
    DtypePolicy,
    EngineModel,
    ForwardBatch,
    ImageInputs,
    ModelFamily,
    ModelInfo,
    ModelSpec,
    PackageRef,
    VerifiedPackage,
    VideoInputs,
)
from ...plugins.decisions import DecisionModel, RequestPlan
from ...registry import builtin, policy
from ...systemone import (
    GOLDEN_STATE,
    MAX_LEVELS,
    MAX_OPTIONS,
    MIN_LEVELS,
    MIN_OPTIONS,
    canonical,
    golden_questions,
    read_question,
    valid_state,
)
from . import images as img
from . import package as pkg
from . import prompt
from . import videos as vid

__all__ = ["Decision3Family", "Decision3Model"]

BATCH_SIZE = 8
VISION_TOWER = "vision"
# MIOpen's naive depthwise convolution (FP64 accumulation), which the released runtime ran on ROCm,
# alone and inside the fused gated-delta preparation.
CONV_VARIANT = {"causal_conv1d": "fp64_naive", "gdn_prep": "fp64_naive"}
GOLDEN_QUESTIONS = golden_questions("Anything else")
# A 64 x 48 PNG whose rows fade from pink to green: the image golden request reads it.
GOLDEN_IMAGE = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAEAAAAAwCAIAAAAuKetIAAAA6ElEQVR42tXCAUYAAAAEwX1OkiRJkiRJkiRJkiRJkiRJkiRJ"
    "kiRJkiRJkiRJkiTJSpIkSZKkd/SOG4MFf9Gx6Cc6lnxFx7KP6FjxFh2rXqJjzVN0rHuIjg330bHpNjq2XEfHtsvo2HEeHbtO"
    "o2PPcXTsO4yOA/vRcWg3Oo5sR8exzeg4sR4dp1aj48xydJxbjI4L89FxaTY6rkxHx7XJ6LgxHh23RqPjznB03BuMjgf90fGo"
    "NzqedEfHs87oeNEeHa9ao+NNc3S8a4yO1kfHx9ro+FwdHV8ro+N7eXT8LI2O38XR8bcw+j9BJQlL/YCbYAAAAABJRU5ErkJg"
    "gg=="
)
GOLDEN_IMAGE_QUESTIONS = {
    "image_colour": {
        "type": "choice",
        "instructions": "Which colour dominates the image?",
        "criteria": {
            "red": "Red",
            "green": "Green",
            "blue": "Blue",
            "other": "Anything else",
        },
    },
    "image_photo": {
        "type": "noul",
        "instructions": "Is the image a photograph?",
    },
}


@dataclass(frozen=True)
class RequestMedia:
    """A request's processed images and videos, shared by its questions.

    The images' patch rows are copied to a device once (``rows_on``). With
    videos, the tower's features of the images and of the videos are computed
    once per device (``features``). ``video_ids`` holds each video's
    placeholder expansion as token IDs.
    """

    images: tuple[img.ProcessedImage, ...]
    videos: tuple[vid.ProcessedVideo, ...] = ()
    video_ids: tuple[tuple[int, ...], ...] = ()
    placed: dict[str, torch.Tensor] = field(
        default_factory=dict, compare=False, repr=False
    )
    features: dict[str, tuple[torch.Tensor, torch.Tensor]] = field(
        default_factory=dict, compare=False, repr=False
    )

    def pixel_rows(self) -> torch.Tensor:
        return torch.cat([image.pixel_values for image in self.images])

    def rows_on(self, device: torch.device) -> torch.Tensor:
        rows = self.placed.get(str(device))
        if rows is None:
            rows = torch.cat([image.pixel_values.to(device) for image in self.images])
            self.placed[str(device)] = rows
        return rows


@dataclass(frozen=True)
class Item:
    """One rendered question: its full token IDs (media placeholders expanded) and how its answer reads."""

    question_id: str
    task_type: str
    ids: list[int]
    keys: list[str]
    descriptions: list[Any]
    media: RequestMedia | None = None


class Decision3Family(ModelFamily):
    name = "decision3"
    surfaces = frozenset({"decisions"})
    builtin_table = "vllm_srun.registry.tables.decision3"
    fixture_writer = "vllm_srun.testing.decision3"

    @classmethod
    def descriptor(cls) -> dict[str, Any]:
        return {
            "surfaces": sorted(cls.surfaces),
            "formats": [pkg.MANIFEST_SCHEMA],
            "question_types": ["choice", "noul", "score"],
            "modalities": ["text", "image", "video"],
        }

    def detect(self, package: PackageRef) -> bool:
        return pkg.is_package(package.root)

    def verify(self, package: PackageRef) -> VerifiedPackage:
        details, manifest_sha256 = pkg.verify(package.root)
        licence = policy.check(details.manifest, self.options.accept_licences)
        known = builtin.lookup(package.repo_id) if package.repo_id else None
        if (
            known is not None
            and known.revision == package.revision
            and (
                known.model_sha256 != details.model_sha256
                or known.manifest_sha256 != manifest_sha256
            )
        ):
            raise PackageError(
                f"{package.repo_id}@{package.revision} differs from the built-in pinned identity"
            )
        return VerifiedPackage(
            ref=package,
            family=self.name,
            model_name=str(details.manifest.get("model_name") or package.root.name),
            manifest=details.manifest,
            manifest_sha256=manifest_sha256,
            model_sha256=details.model_sha256,
            max_input_tokens=details.max_input_tokens,
            licence=licence,
            loaded_parameters=details.manifest["parameters"]["loaded"],
            details={"package": details},
        )

    def describe(self, package: VerifiedPackage) -> ModelSpec:
        details: pkg.Decision3Package = package.details["package"]
        weights = tuple(
            details.root / name
            for name in details.weights
            if name.endswith(".safetensors")
        )
        text = {
            **details.config["text_config"],
            "attention_mode": details.decision_config.get("attention_mode", "causal"),
        }
        backbone = BackboneSpec(
            model_type="qwen3_5_text",
            config=text,
            weight_files=weights,
            weight_prefix=details.text_prefix,
        )
        towers = {}
        if details.vision_prefix is not None and details.processor_config is not None:
            towers[VISION_TOWER] = BackboneSpec(
                model_type="qwen3_5_vision",
                config=details.config["vision_config"],
                weight_files=weights,
                weight_prefix=details.vision_prefix,
            )
        return ModelSpec(
            name=package.model_name,
            backbone=backbone,
            dtype=DtypePolicy(
                autocast=None,
                bf16_resident=False,
                gpu_weights="bfloat16",
                cpu_weights="bfloat16",
            ),
            max_input_tokens=package.max_input_tokens,
            kernel_variants=CONV_VARIANT,
            requires=backbone.requires,
            towers=towers,
        )

    def load(
        self, package: VerifiedPackage, spec: ModelSpec, engine_model: EngineModel
    ) -> Decision3Model:
        from safetensors.torch import load_file

        details: pkg.Decision3Package = package.details["package"]
        root = details.root
        prompt.check_chat_template(root)
        backend, pad_id = prompt.load_tokenizer(root)
        decision = details.decision_config
        prompt.check_codes(backend, decision["codes"], decision["token_ids"])
        readout_dtype = decision.get("readout_dtype", "float32")
        readout = load_file(str(root / "readout.safetensors"))["weight"].to(
            engine_model.device, getattr(torch, readout_dtype)
        )
        parameters = engine_model.parameter_count() + readout.numel()
        expected = package.manifest["parameters"]["loaded"]
        reads_images = VISION_TOWER in spec.towers
        if not reads_images:
            expected -= package.manifest["parameters"].get("vision", 0)
        if parameters != expected:
            raise PackageError(
                f"loaded {parameters:,} parameters; the manifest declares {expected:,}"
            )
        settings = (
            img.ProcessorSettings.from_config(details.processor_config)
            if reads_images and details.processor_config is not None
            else None
        )
        image_token = backend.token_to_id(prompt.IMAGE_TOKEN)
        if reads_images and image_token != details.config.get("image_token_id"):
            raise PackageError("the tokenizer's image token differs from config.json")
        video_settings = (
            vid.VideoSettings.from_config(details.video_config)
            if settings is not None
            and details.video_config is not None
            and vid.available() is None
            else None
        )
        video_token = backend.token_to_id(prompt.VIDEO_TOKEN)
        if video_settings is not None and video_token != details.config.get(
            "video_token_id"
        ):
            raise PackageError("the tokenizer's video token differs from config.json")
        limits: dict[str, Any] = {
            "max_input_tokens": package.max_input_tokens,
            "min_options": MIN_OPTIONS,
            "max_options": MAX_OPTIONS,
            "min_levels": MIN_LEVELS,
            "max_levels": MAX_LEVELS,
        }
        if settings is not None:
            limits |= {
                "image_max_pixels": settings.max_pixels,
                "image_source_max_pixels": img.MAX_SOURCE_PIXELS,
                "image_max_bytes": img.MAX_IMAGE_BYTES,
            }
        modalities: tuple[str, ...] = ("text", "image") if settings else ("text",)
        if video_settings is not None:
            limits |= {
                "video_fps": vid.FPS,
                "video_max_frames": vid.MAX_FRAMES,
                "video_max_pixels": vid.MAX_PIXELS,
                "video_max_tokens": vid.MAX_TOKENS,
                "video_max_bytes": vid.MAX_VIDEO_BYTES,
                "video_max_seconds": vid.MAX_SECONDS,
                "video_source_max_pixels": vid.MAX_SOURCE_PIXELS,
            }
            modalities += ("video",)
        info = ModelInfo(
            id=package.model_name,
            family=self.name,
            repo=package.ref.repo_id,
            revision=package.ref.revision,
            model_sha256=package.model_sha256,
            manifest_sha256=package.manifest_sha256,
            surfaces=tuple(sorted(self.surfaces)),
            question_types=("choice", "noul", "score"),
            limits=limits,
            licence=package.licence,
            parameters=parameters,
            dtype=f"bf16/{'fp32' if readout_dtype == 'float32' else 'bf16'}-readout",
            modalities=modalities,
        )
        return Decision3Model(
            info,
            engine_model,
            backend,
            pad_id,
            readout,
            decision,
            settings,
            image_token,
            video_settings,
            video_token,
        )

    def golden(self, package: VerifiedPackage) -> list[dict[str, Any]]:
        requests = builtin.golden(
            package.model_sha256,
            "decisions",
            {"state": GOLDEN_STATE, "questions": GOLDEN_QUESTIONS},
        )
        details: pkg.Decision3Package = package.details["package"]
        if details.vision_prefix is not None and details.processor_config is not None:
            requests += builtin.golden(
                package.model_sha256,
                "decisions",
                {
                    "state": "Describe the attached image.",
                    "questions": GOLDEN_IMAGE_QUESTIONS,
                    "images": [GOLDEN_IMAGE],
                },
            )
        return requests


def _confidence(values: list[float]) -> float:
    """One minus the normalized entropy, clipped to [0, 1]."""
    if len(values) <= 1:
        return 1.0
    entropy = -sum(p * math.log(p) for p in values if p > 0)
    return max(0.0, min(1.0, 1.0 - entropy / math.log(len(values))))


def product_answer(item: Item, probabilities: list[float]) -> dict[str, Any]:
    """The released runtime's answer: probabilities renormalized on the host, argmax with first-option ties."""
    values = [float(v) for v in probabilities]
    if (
        len(values) != len(item.keys)
        or any(not math.isfinite(v) or v < 0 for v in values)
        or sum(values) <= 0
    ):
        raise ValueError("need one finite non-negative probability per option")
    total = sum(values)
    values = [v / total for v in values]
    if item.task_type == "noul":
        return {"type": "noul", "noul": values[1]}
    if item.task_type == "choice":
        best = max(range(len(values)), key=values.__getitem__)
        answer: dict[str, Any] = {
            "type": "choice",
            "choice": item.keys[best],
            "probabilities": dict(zip(item.keys, values, strict=True)),
        }
        answer["confidence"] = _confidence(list(answer["probabilities"].values()))
        return answer
    return {
        "type": "score",
        "score": sum(i * p for i, p in enumerate(values)),
        "probabilities": dict(zip(item.keys, values, strict=True)),
        "confidence": _confidence(values),
        "legend": {
            key: level if isinstance(level, str) else canonical(level)
            for key, level in zip(item.keys, item.descriptions, strict=True)
        },
    }


class Decision3Model(DecisionModel[Item, list[float] | None]):
    image_inputs: ClassVar[bool] = True

    def __init__(
        self,
        info: ModelInfo,
        engine_model: EngineModel,
        tokenizer: Any,
        pad_id: int,
        readout: torch.Tensor,
        decision: dict[str, Any],
        settings: img.ProcessorSettings | None,
        image_token: int | None,
        video_settings: vid.VideoSettings | None = None,
        video_token: int | None = None,
    ):
        self.info = info
        self.engine_model = engine_model
        self.tokenizer = tokenizer
        self.pad_id = pad_id
        self.readout = readout
        self.codes: list[str] = decision["codes"]
        self.temperature = float(decision.get("temperature", 1.0))
        self.settings = settings
        self.image_token = image_token
        self.video_settings = video_settings
        self.video_token = video_token
        self.video_inputs = video_settings is not None
        self.limit = info.limits["max_input_tokens"]

    def forward_token_budget(self) -> int | None:
        return self.engine_model.max_forward_tokens()

    def shared_context(self, items: list[Item], token_budget: int | None) -> int | None:
        return 0

    def exact_batches(self, items: list[Item]) -> list[list[int]]:
        """Request order, eight questions per pass."""
        return [
            list(range(start, min(start + BATCH_SIZE, len(items))))
            for start in range(0, len(items), BATCH_SIZE)
        ]

    # ------------------------------------------------------------------ planning

    def _question(
        self, question: Any
    ) -> tuple[str, Any, list[str], list[str], list[Any]]:
        """(type, instructions, keys, rendered option texts, descriptions) of one valid question."""
        parsed = read_question(question)
        if parsed.kind == "choice":
            keys = list(parsed.criteria)
            return (
                parsed.kind,
                parsed.instructions,
                keys,
                prompt.option_texts("choice", parsed.criteria),
                [parsed.criteria[key] for key in keys],
            )
        if parsed.kind == "noul":
            return (
                parsed.kind,
                parsed.instructions,
                ["false", "true"],
                prompt.option_texts("noul", parsed.criteria),
                [None, None],
            )
        levels = [prompt.describe(level) for level in parsed.criteria]
        if any(not text for text in levels) or len(set(levels)) != len(levels):
            raise QuestionError(
                INVALID_QUESTION, "score levels must be distinct and nonempty"
            )
        return (
            parsed.kind,
            parsed.instructions,
            [str(index) for index in range(len(levels))],
            levels,
            list(parsed.criteria),
        )

    def _plan(
        self,
        state: Any,
        questions: dict[str, Any],
        media: RequestMedia | None,
    ) -> RequestPlan[Item]:
        if not valid_state(state):
            raise ValueError("state must be text, an object, or an array")
        images = len(media.images) if media else 0
        videos = len(media.videos) if media else 0
        visual = 0
        if media:
            visual += sum(image.tokens for image in media.images)
            visual += sum(len(ids) for ids in media.video_ids)
        items: list[Item] = []
        errors: dict[str, dict[str, Any]] = {}
        tokens = 0
        for question_id, question in questions.items():
            kind = question.get("type") if isinstance(question, dict) else None
            try:
                kind, instructions, keys, texts, descriptions = self._question(question)
                text = prompt.render(
                    prompt.user_prompt(state, instructions, texts, self.codes),
                    images,
                    videos,
                )
                if media and (
                    text.count(prompt.IMAGE_TOKEN) != images
                    or text.count(prompt.VIDEO_TOKEN) != videos
                ):
                    raise QuestionError(
                        INVALID_QUESTION,
                        "the state or question contains a literal image or video placeholder token",
                    )
                ids = self.tokenizer.encode(text, add_special_tokens=False).ids
                length = len(ids) - images - videos + visual
                if length > self.limit:
                    raise QuestionError(
                        MAX_LENGTH_EXCEEDED,
                        f"the question prompt has {length} tokens, over the maximum context length of "
                        f"{self.limit} tokens; nothing was truncated",
                    )
                if media:
                    ids = self._expand(ids, media)
            except QuestionError as exc:
                errors[question_id] = question_error(kind, exc)
                continue
            tokens += len(ids)
            items.append(Item(question_id, kind, ids, keys, descriptions, media))
        return RequestPlan(
            question_ids=list(questions),
            items=items,
            errors=errors,
            input_tokens=tokens,
            complete_inputs=frozenset(
                key
                for key, question in questions.items()
                if isinstance(question, dict)
                and question.get("require_full_input") is True
                and key not in errors
            ),
        )

    def _expand(self, ids: list[int], media: RequestMedia) -> list[int]:
        """Each image placeholder repeated once per input token of its image, and each video placeholder as its
        expansion (timestamped frame pairs), images and videos in order."""
        out: list[int] = []
        images = iter(media.images)
        videos = iter(media.video_ids)
        for token in ids:
            if token == self.image_token:
                out.extend([token] * next(images).tokens)
            elif media.videos and token == self.video_token:
                out.extend(next(videos))
            else:
                out.append(token)
        return out

    def plan(
        self, state: Any, questions: dict[str, Any], scan: int | None = None
    ) -> RequestPlan[Item]:
        return self._plan(state, questions, None)

    def plan_images(
        self,
        state: Any,
        questions: dict[str, Any],
        images: list[Any],
        videos: list[Any] | None = None,
    ) -> RequestPlan[Item]:
        if self.settings is None:
            raise ValueError(
                f"model {self.info.id} has no vision tower; images are not supported"
            )
        processed = []
        for number, value in enumerate(images):
            try:
                picture, digest = img.decode(value)
                processed.append(img.preprocess(picture, digest, self.settings))
            except ValueError as exc:
                raise ValueError(f"images[{number}]: {exc}") from exc
        clips: list[vid.ProcessedVideo] = []
        if videos:
            settings = self.video_settings
            if settings is None:
                raise ValueError(f"model {self.info.id} reads no videos")
            decoded = []
            for number, value in enumerate(videos):
                try:
                    decoded.append(vid.decode(value))
                except ValueError as exc:
                    raise ValueError(f"videos[{number}]: {exc}") from exc
            try:
                planned = sum(
                    vid.input_tokens(settings, *clip.frames.shape[:3])
                    for clip in decoded
                )
            except ValueError as exc:
                raise ValueError(f"video rejected by the processor: {exc}") from exc
            if planned > vid.MAX_TOKENS:
                raise ValueError(
                    f"the videos take {planned:,} input tokens, over the {vid.MAX_TOKENS:,} tokens a "
                    "request may spend on videos; send fewer or shorter videos"
                )
            clips = [vid.preprocess(clip, settings) for clip in decoded]
        expansions = tuple(
            tuple(
                self.tokenizer.encode(
                    clip.placeholder(
                        prompt.VISION_START, prompt.VIDEO_TOKEN, prompt.VISION_END
                    ),
                    add_special_tokens=False,
                ).ids
            )
            for clip in clips
        )
        return self._plan(
            state,
            questions,
            RequestMedia(tuple(processed), tuple(clips), expansions),
        )

    # ------------------------------------------------------------------ forwards

    def run(self, items: list[Item]) -> list[list[float] | None]:
        """Each run of items that share their images as one left-padded pass (released numerics)."""
        out: list[list[float] | None] = []
        start = 0
        while start < len(items):
            end = start + 1
            while end < len(items) and items[end].media is items[start].media:
                end += 1
            out.extend(self._pass(items[start:end]))
            start = end
        return out

    def _pass(self, items: list[Item]) -> list[list[float] | None]:
        width = max(len(item.ids) for item in items)
        rows = len(items)
        ids = torch.full((rows, width), self.pad_id, dtype=torch.long)
        mask = torch.zeros((rows, width), dtype=torch.long)
        for row, item in enumerate(items):
            ids[row, width - len(item.ids) :] = torch.as_tensor(
                item.ids, dtype=torch.long
            )
            mask[row, width - len(item.ids) :] = 1
        last = torch.full((rows,), width - 1, dtype=torch.long)
        shared = items[0].media
        images = None
        if shared is not None and shared.videos:
            assert self.image_token is not None and self.video_token is not None
            image_features, video_features = self._features(shared)
            images = ImageInputs(
                pixel_values=None,
                grids=[image.grid for _ in items for image in shared.images],
                token_id=self.image_token,
                tower=VISION_TOWER,
                features=torch.cat([image_features] * rows),
                videos=VideoInputs(
                    features=torch.cat([video_features] * rows),
                    grids=[clip.grid for _ in items for clip in shared.videos],
                    token_id=self.video_token,
                ),
            )
        elif shared is not None:
            assert self.image_token is not None
            placed = shared.rows_on(self.engine_model.device)
            images = ImageInputs(
                pixel_values=placed if rows == 1 else torch.cat([placed] * rows),
                grids=[image.grid for _ in items for image in shared.images],
                token_id=self.image_token,
                tower=VISION_TOWER,
            )
        output = self.engine_model.forward(
            ForwardBatch(
                input_ids=ids,
                attention_mask=mask,
                gather=last[:, None],
                query=last,
                lengths=[len(item.ids) for item in items],
                images=images,
            )
        )
        counts = torch.as_tensor(
            [len(item.keys) for item in items], device=self.readout.device
        )
        with torch.inference_mode():
            hidden = output.query
            if self.readout.dtype == torch.float32:
                logits = hidden.float() @ self.readout.T
            else:
                logits = torch.nn.functional.linear(hidden, self.readout).float()
            invalid = (
                torch.arange(MAX_OPTIONS, device=logits.device)[None] >= counts[:, None]
            )
            probs = (
                logits.masked_fill(invalid, float("-inf")) / self.temperature
            ).softmax(-1)
            values = probs.cpu().tolist()
        return [row[: len(item.keys)] for row, item in zip(values, items, strict=True)]

    def _features(self, media: RequestMedia) -> tuple[torch.Tensor, torch.Tensor]:
        """The tower's features of a request's images and of its videos (one call each), once per device."""
        device = self.engine_model.device
        cached = media.features.get(str(device))
        if cached is None:
            engine = self.engine_model
            if media.images:
                image_features = engine.tower_features(
                    VISION_TOWER,
                    media.rows_on(device),
                    [image.grid for image in media.images],
                )
            else:
                image_features = torch.empty(
                    (0, self.readout.shape[1]), device=device, dtype=torch.bfloat16
                )
            video_features = engine.tower_features(
                VISION_TOWER,
                torch.cat([clip.pixel_values for clip in media.videos]).to(device),
                [clip.grid for clip in media.videos],
            )
            cached = media.features[str(device)] = (image_features, video_features)
        return cached

    def answer(self, item: Any, values: list[float] | None) -> dict[str, Any]:
        if values is None:
            return {"type": item.task_type, "error": INVALID_MODEL_OUTPUT}
        try:
            return product_answer(item, values)
        except ValueError as exc:
            return {
                "type": item.task_type,
                "error": INVALID_MODEL_OUTPUT,
                "message": str(exc),
            }
