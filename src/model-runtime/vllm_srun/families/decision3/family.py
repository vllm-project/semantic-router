"""The Decision 3.0 model family plugin: d3 code-readout packages, with image and video inputs.

A question is one prompt (``prompt.py``) whose options are listed under
single-token answer codes. The backbone reads it left-padded with noncausal
full attention, and the FP32 readout scores the question's codes at the last
position: softmax over the codes at the package temperature, then the answer.
A package with ``temperature_by_questions`` sets that temperature per request
from its number of Choice, Yes / No and Score questions (Box and track keep
the package temperature), and one with ``prompt_layout`` ``header-v1`` lists a
text-only request's questions in front of its state.
A request's questions run in request order, eight per forward pass, as the
released runtime runs them. Images (``images.py``) and videos (``videos.py``)
are shared by every question of a request. Without videos, each question of a
pass carries its own copy of the images through the vision tower, as the
released runtime's processor hands them on; with videos, the tower reads the
request's images and videos once and every question reuses their features, as
the d3 runtime does. A package whose ``attention_mode`` gives no state token a
view of a question loads as a ``StateOnceModel`` (``state_once.py``): its
requests read their state once and branch every question from it.

A forward that merges requests (``run_approximate``) holds text rows or rows
that read media (``lane``), each lane with its own cost per token. Where the
engine packs rows (``EngineModel.supports_packed``) the tower reads the images
of every request in the forward in one packed call, and the rows run back to
back without padding when left-padding them would cost more than
``PACK_MIN_PADDING`` of their tokens; elsewhere they run left-padded. Request images are
decoded and resized in the process's media worker processes when it has any
(``vllm_srun.media``).
"""

from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass, field, replace
from typing import Any, ClassVar, cast

import torch

from ... import media as workers
from ... import quantization
from ...errors import (
    INVALID_MODEL_OUTPUT,
    INVALID_QUESTION,
    MAX_LENGTH_EXCEEDED,
    MediaError,
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
from ...plugins.decisions import LONG_CONTEXT, DecisionModel, RequestPlan
from ...registry import builtin, policy
from ...systemone import (
    GOLDEN_STATE,
    MAX_LEVELS,
    MAX_OPTIONS,
    MIN_LEVELS,
    MIN_OPTIONS,
    canonical,
    choice_confidence,
    golden_questions,
    read_question,
    score_confidence,
    valid_state,
)
from . import cache as mc
from . import images as img
from . import package as pkg
from . import prompt
from . import videos as vid

__all__ = ["Decision3Family", "Decision3Model"]

BATCH_SIZE = 8
VISION_TOWER = "vision"
# Device time one forward that merges requests should take, so a full queue is
# answered over several short forwards instead of one long one; the running cost per token sizes it.
COALESCED_FORWARD_SECONDS = 0.3
# Dense BF16 throughput assumed for the first coalesced forward, before one is measured.
ASSUMED_FLOPS = 3e14
MIN_COALESCED_TOKENS = 2048
COST_SMOOTHING = 0.2
# A merged pass packs its rows only when left-padding them would add more than this share of
# their tokens: a packed forward costs slightly more than a padded one without padding.
PACK_MIN_PADDING = 0.1
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


@dataclass(frozen=True, eq=False)
class RequestState:
    """A state-once request's state block, shared by its questions: its token IDs (media placeholders
    expanded) and its media."""

    ids: tuple[int, ...]
    media: RequestMedia | None = None
    # The tokens after the state that open each of its question blocks (a track window's own state).
    opening: tuple[int, ...] = ()


@dataclass(frozen=True)
class Item:
    """One rendered question: its full token IDs (media placeholders expanded) and how its answer reads.

    A state-once question also names its request's ``state``: its IDs start
    with the state block's, and ``cost`` holds its question block plus its
    share of the state, the tokens it adds to a branched forward.
    """

    question_id: str
    task_type: str
    ids: list[int]
    keys: list[str]
    descriptions: list[Any]
    media: RequestMedia | None = None
    state: RequestState | None = None
    cost: int | None = None
    # Composed questions (``composed``): the readouts as (offset from the last token, codes read), None for the
    # last token over ``keys``; the question's own images in block order; a Box's ``at`` time.
    readouts: tuple[tuple[int, int], ...] | None = None
    block_images: tuple[img.ProcessedImage, ...] = ()
    at: float | None = None
    # The request's temperature (``temperature_by_questions``); None: the package's.
    temperature: float | None = None


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
        quantized = quantization.parse(details.config)
        backbone = BackboneSpec(
            model_type="qwen3_5_text",
            config=text,
            weight_files=weights,
            weight_prefix=details.text_prefix,
            quantized=quantized,
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
                # docs/records/decision3-quantized.md: the fused int4 kernel and,
                # in the FP8 d3-mini table, the fused element-wise kernels change
                # few of a quantized package's answers; BF16 packages keep exact
                # kernels.
                approximate_kernels=quantized is not None,
            ),
            max_input_tokens=package.max_input_tokens,
            kernel_variants=CONV_VARIANT,
            requires=backbone.requires,
            towers=towers,
        )

    def load(
        self, package: VerifiedPackage, spec: ModelSpec, engine_model: EngineModel
    ) -> Decision3Model:
        """A ``Decision3Model``, or a ``StateOnceModel`` for a package whose layout is state-once."""
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
            **(
                {"validated_input_tokens": decision["validated_length"]}
                if decision.get("validated_length")
                and decision["validated_length"] < package.max_input_tokens
                else {}
            ),
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
            # The longest clip a box + at or track question reads, declared by a size that
            # answers track: decision_config systemone.box_video_max_seconds.
            systemone = details.decision_config.get("systemone") or {}
            if "track" in systemone.get("question_types", ()) and systemone.get(
                "box_video_max_seconds"
            ):
                limits["box_video_max_seconds"] = int(
                    systemone["box_video_max_seconds"]
                )
            modalities += ("video",)
        info = ModelInfo(
            id=package.model_name,
            family=self.name,
            repo=package.ref.repo_id,
            revision=package.ref.revision,
            model_sha256=package.model_sha256,
            manifest_sha256=package.manifest_sha256,
            surfaces=tuple(sorted(self.surfaces)),
            question_types=(
                "choice",
                "noul",
                "score",
                *(
                    kind
                    for kind in ("box", "track")
                    if kind
                    in (details.decision_config.get("systemone") or {}).get(
                        "question_types", ()
                    )
                ),
            ),
            image_placements=tuple(
                (details.decision_config.get("systemone") or {}).get(
                    "image_placements", ()
                )
            ),
            limits=limits,
            licence=package.licence,
            parameters=parameters,
            dtype=f"bf16/{'fp32' if readout_dtype == 'float32' else 'bf16'}-readout",
            modalities=modalities,
        )
        model_class = Decision3Model
        if decision.get("attention_mode", "causal") in pkg.STATE_ONCE_MODES:
            from .state_once import StateOnceModel

            model_class = StateOnceModel
            # A state-once model counts each state once in usage.input_tokens, and that count of one request
            # is capped at the context length (max_request_tokens).
            info = replace(
                info,
                usage="state_once",
                limits={**info.limits, "max_request_tokens": package.max_input_tokens},
            )
        return model_class(
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
        answer["confidence"] = choice_confidence(values)
        return answer
    return {
        "type": "score",
        "score": sum(i * p for i, p in enumerate(values)),
        "probabilities": dict(zip(item.keys, values, strict=True)),
        "confidence": score_confidence(values),
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
        self.temperatures = pkg.temperature_table(decision)
        self.layout = decision.get("prompt_layout")
        self.settings = settings
        self.image_token = image_token
        self.video_settings = video_settings
        self.video_token = video_token
        self.video_inputs = video_settings is not None
        self.limit = info.limits["max_input_tokens"]
        # The length the model was validated at caps each question unless the request asks for long context.
        self.validated: int | None = info.limits.get("validated_input_tokens")
        self.media_cache: mc.LRUCache[Any] = mc.LRUCache(mc.MEDIA_CACHE_BYTES)
        self.feature_cache: mc.LRUCache[torch.Tensor] = mc.LRUCache(
            mc.FEATURE_CACHE_BYTES
        )
        # The package's isotonic Box-confidence map (decision_config ``systemone.box_calibration``:
        # [[raw, calibrated], ...]), fitted on BoxSet-dev for a size that passed its Box bar.
        self.box_calibration = (decision.get("systemone") or {}).get("box_calibration")
        self.packs_merged_rows = engine_model.supports_packed
        assumed = 2 * info.parameters / ASSUMED_FLOPS
        self._token_seconds = {"text": assumed, "media": assumed}
        if settings is not None:
            workers.start()

    def forward_token_budget(self) -> int | None:
        return self.engine_model.max_forward_tokens()

    def lane(self, item: Item) -> str:
        """Rows that read images or videos run apart from text rows when requests merge."""
        return "text" if item.media is None else "media"

    def coalesced_token_budget(self, lane: str = "text") -> int | None:
        """Tokens one coalesced forward of ``lane`` may hold: ``COALESCED_FORWARD_SECONDS`` at its measured cost per token."""
        seconds = self._token_seconds.get(lane, self._token_seconds["text"])
        cap = max(MIN_COALESCED_TOKENS, int(COALESCED_FORWARD_SECONDS / seconds))
        limit = self.forward_token_budget()
        return cap if limit is None else min(cap, limit)

    def calibration_items(self, rows: int, row_tokens: int) -> list[Item] | None:
        """Text rows of ``row_tokens`` tokens of a two-option question, for the startup calibration."""
        filler = self.tokenizer.encode(
            "Decide whether this request needs a careful answer. ",
            add_special_tokens=False,
        ).ids
        ids = list(itertools.islice(itertools.cycle(filler), row_tokens))
        return [
            Item(
                question_id=f"calibration{row}",
                task_type="noul",
                ids=ids,
                keys=["false", "true"],
                descriptions=[None, None],
            )
            for row in range(rows)
        ]

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

    def _read(self, questions: dict[str, Any]) -> tuple[
        list[tuple[str, str, Any, list[str], list[str], list[Any]]],
        dict[str, dict[str, Any]],
    ]:
        """Each valid question as (ID, ``_question``'s fields), in request order, and the others' errors."""
        read = []
        errors: dict[str, dict[str, Any]] = {}
        for question_id, question in questions.items():
            kind = question.get("type") if isinstance(question, dict) else None
            try:
                read.append((question_id, *self._question(question)))
            except QuestionError as exc:
                errors[question_id] = question_error(kind, exc)
        return read, errors

    def _header(
        self,
        read: list[tuple[str, str, Any, list[str], list[str], list[Any]]],
        media: RequestMedia | None,
    ) -> str:
        """The question list in front of the state: a text-only request's, on a ``header-v1`` package."""
        if self.layout != pkg.HEADER_V1 or media or not read:
            return ""
        return prompt.header_block(
            [
                (instructions, None if kind == "noul" else texts)
                for _, kind, instructions, _, texts, _ in read
            ]
        )

    def request_temperature(self, questions: int) -> float | None:
        """The temperature of a request with ``questions`` Choice, Yes / No and Score questions under the package's
        ``temperature_by_questions``; None without one (each question keeps the package temperature).
        """
        if self.temperatures is None:
            return None
        return pkg.temperature_for(self.temperatures, questions)

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
        tokens = 0
        read, errors = self._read(questions)
        header = self._header(read, media)
        rendered: list[tuple[str, str, list[str], list[Any], str]] = []
        for question_id, kind, instructions, keys, texts, descriptions in read:
            text = prompt.render(
                prompt.user_prompt(state, instructions, texts, self.codes, header),
                images,
                videos,
            )
            if media and (
                text.count(prompt.IMAGE_TOKEN) != images
                or text.count(prompt.VIDEO_TOKEN) != videos
            ):
                errors[question_id] = question_error(
                    kind,
                    QuestionError(
                        INVALID_QUESTION,
                        "the state or question contains a literal image or video placeholder token",
                    ),
                )
                continue
            rendered.append((question_id, kind, keys, descriptions, text))
        temperature = self.request_temperature(len(rendered))
        # One call for all questions: the tokenizer releases the GIL for a batch, not for a single text.
        encodings = self.tokenizer.encode_batch(
            [entry[-1] for entry in rendered], add_special_tokens=False
        )
        for (question_id, kind, keys, descriptions, _), encoding in zip(
            rendered, encodings, strict=True
        ):
            ids = encoding.ids
            length = len(ids) - images - videos + visual
            refused = self.length_error(length)
            if refused is not None:
                errors[question_id] = question_error(kind, refused)
                continue
            if media:
                ids = self._expand(ids, media)
            tokens += len(ids)
            items.append(
                Item(
                    question_id,
                    kind,
                    ids,
                    keys,
                    descriptions,
                    media,
                    temperature=temperature,
                )
            )
        return RequestPlan(
            question_ids=list(questions),
            items=items,
            errors={key: errors[key] for key in questions if key in errors},
            input_tokens=tokens,
            complete_inputs=frozenset(
                key
                for key, question in questions.items()
                if isinstance(question, dict)
                and question.get("require_full_input") is True
                and key not in errors
            ),
        )

    def question_cap(self) -> tuple[int, bool]:
        """The per-question token cap of the request being planned, and whether it is the validated length: a
        package that declares one caps each question at it unless the request asks for long context.
        """
        if self.validated is not None and not LONG_CONTEXT.get():
            return self.validated, True
        return self.limit, False

    def length_error(self, length: int) -> QuestionError | None:
        """Why a question prompt of ``length`` tokens is refused (over ``question_cap``), or None."""
        limit, capped = self.question_cap()
        if length <= limit:
            return None
        hint = (
            f"the length this model was validated at; long context "
            f"(options.long_context) reads up to "
            f"{self.limit} tokens, slower and less accurate"
            if capped
            else "the maximum context length"
        )
        return QuestionError(
            MAX_LENGTH_EXCEEDED,
            f"the question prompt has {length} tokens, over {limit} tokens, {hint}; "
            "nothing was truncated",
            "question_tokens",
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
        return self._plan(state, questions, self.request_media(images, videos))

    def image(self, value: Any, where: str) -> img.ProcessedImage:
        """One image decoded and preprocessed (cached by its data URL's digest)."""
        if self.settings is None:
            raise MediaError(
                "image_unsupported",
                f"model {self.info.id} has no vision tower; images are not supported",
            )
        key = mc.url_digest(value)
        cached = self.media_cache.get(("image", key)) if key else None
        if cached is not None:
            return cast(img.ProcessedImage, cached)
        try:
            picture, digest = img.decode(value)
            image = img.preprocess(picture, digest, self.settings)
        except MediaError as exc:
            raise exc.at(where) from exc
        except ValueError as exc:
            raise ValueError(f"{where}: {exc}") from exc
        if key:
            self.media_cache.put(("image", key), image, _bytes(image.pixel_values))
        return image

    def request_media(
        self, images: list[Any], videos: list[Any] | None = None
    ) -> RequestMedia:
        """A request's processed images and videos (the state's media)."""
        if self.settings is None:
            raise MediaError(
                "image_unsupported",
                f"model {self.info.id} has no vision tower; images are not supported",
            )
        keys = [mc.url_digest(value) for value in images]
        found: list[img.ProcessedImage | None] = [
            self.media_cache.get(("image", key)) if key else None for key in keys
        ]
        pending = (
            {
                number: workers.submit(img.load, value, self.settings)
                for number, value in enumerate(images)
                if found[number] is None
            }
            if workers.workers() > 0
            else {}
        )
        processed = []
        try:
            for number, value in enumerate(images):
                image = found[number]
                if image is None:
                    try:
                        if number in pending:
                            pixels, digest = workers.outcome(
                                pending[number], img.load, value, self.settings
                            )
                            image = img.loaded(pixels, digest, self.settings)
                        else:
                            picture, digest = img.decode(value)
                            image = img.preprocess(picture, digest, self.settings)
                    except MediaError as exc:
                        raise exc.at(f"images[{number}]") from exc
                    except ValueError as exc:
                        raise ValueError(f"images[{number}]: {exc}") from exc
                    if keys[number]:
                        self.media_cache.put(
                            ("image", keys[number]), image, _bytes(image.pixel_values)
                        )
                processed.append(image)
        finally:
            for future in pending.values():
                future.cancel()
        clips: list[vid.ProcessedVideo] = []
        if videos:
            settings = self.video_settings
            if settings is None:
                raise MediaError(
                    "video_unsupported", f"model {self.info.id} reads no videos"
                )
            keys = [mc.url_digest(value) for value in videos]
            known: list[tuple[vid.ProcessedVideo, int] | None] = [
                self.media_cache.get(("video", key)) if key else None for key in keys
            ]
            decoded: dict[int, vid.Video] = {}
            loading = (
                {
                    number: workers.submit(vid.load, value)
                    for number, value in enumerate(videos)
                    if known[number] is None
                }
                if workers.workers() > 0
                else {}
            )
            try:
                for number, value in enumerate(videos):
                    if known[number] is not None:
                        continue
                    try:
                        decoded[number] = (
                            vid.loaded(
                                workers.outcome(loading[number], vid.load, value)
                            )
                            if number in loading
                            else vid.decode(value)
                        )
                    except MediaError as exc:
                        raise exc.at(f"videos[{number}]") from exc
                    except ValueError as exc:
                        raise ValueError(f"videos[{number}]: {exc}") from exc
            finally:
                for future in loading.values():
                    future.cancel()
            try:
                tokens = {
                    number: vid.input_tokens(settings, *clip.frames.shape[:3])
                    for number, clip in decoded.items()
                }
            except MediaError as exc:
                raise exc.at("video rejected by the processor") from exc
            except ValueError as exc:
                raise MediaError(
                    "video_malformed", f"video rejected by the processor: {exc}"
                ) from exc
            planned = sum(tokens.values()) + sum(
                entry[1] for entry in known if entry is not None
            )
            if planned > vid.MAX_TOKENS:
                raise MediaError(
                    "video_tokens",
                    f"the videos take {planned:,} input tokens, over the {vid.MAX_TOKENS:,} tokens a "
                    "request may spend on videos; send fewer or shorter videos",
                    vid.MAX_TOKENS,
                )
            for number, entry in enumerate(known):
                if entry is not None:
                    clips.append(entry[0])
                    continue
                clip = vid.preprocess(decoded[number], settings)
                clips.append(clip)
                if keys[number]:
                    self.media_cache.put(
                        ("video", keys[number]),
                        (clip, tokens[number]),
                        _bytes(clip.pixel_values),
                    )
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
        return RequestMedia(tuple(processed), tuple(clips), expansions)

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

    def run_approximate(self, items: list[Item]) -> list[list[float] | None]:
        """A batch that merges requests: rows of several requests, text rows and media rows in a pass each.

        The planner gives each lane its own batches, so a batch normally holds one. Text rows run packed where
        the engine packs rows, else as one left-padded pass, as a request's text rows do. Media rows run in one
        pass whatever their requests, the images of every request through the tower in one call and each
        request's features copied into its rows (the exact path runs one copy per row, which rounds
        differently). Each pass's time per token sizes its lane's later batches.
        """
        out: list[list[float] | None] = [None] * len(items)
        for lane, run in (("text", self._text_pass), ("media", self._mixed_pass)):
            indices = [i for i, item in enumerate(items) if self.lane(item) == lane]
            if not indices:
                continue
            rows = [items[i] for i in indices]
            started = time.perf_counter()
            for index, value in zip(indices, run(rows), strict=True):
                out[index] = value
            sample = (time.perf_counter() - started) / sum(len(i.ids) for i in rows)
            self._token_seconds[lane] += COST_SMOOTHING * (
                sample - self._token_seconds[lane]
            )
        return out

    def _text_pass(self, items: list[Item]) -> list[list[float] | None]:
        if not self._packs(items):
            return self._pass(items)
        ids, last = self._packed(items)
        return self._readout(items, ids, torch.ones_like(ids), last, None, packed=True)

    def _packs(self, items: list[Item]) -> bool:
        """Whether a merged pass lays its rows back to back: where the engine packs and padding would cost enough."""
        if not self.packs_merged_rows:
            return False
        lengths = [len(item.ids) for item in items]
        return len(lengths) * max(lengths) > (1 + PACK_MIN_PADDING) * sum(lengths)

    def _packed(self, items: list[Item]) -> tuple[torch.Tensor, torch.Tensor]:
        """Rows back to back as one ``[1, T]`` row, and each row's last position in it."""
        ids = torch.tensor([[token for item in items for token in item.ids]])
        ends = torch.tensor(list(itertools.accumulate(len(item.ids) for item in items)))
        return ids, ends - 1

    def _padded(
        self, items: list[Item]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        width = max(len(item.ids) for item in items)
        rows = len(items)
        ids = torch.full((rows, width), self.pad_id, dtype=torch.long)
        mask = torch.zeros((rows, width), dtype=torch.long)
        for row, item in enumerate(items):
            ids[row, width - len(item.ids) :] = torch.as_tensor(
                item.ids, dtype=torch.long
            )
            mask[row, width - len(item.ids) :] = 1
        return ids, mask, torch.full((rows,), width - 1, dtype=torch.long)

    def _pass(self, items: list[Item]) -> list[list[float] | None]:
        ids, mask, last = self._padded(items)
        rows = len(items)
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
            images = ImageInputs(
                pixel_values=None,
                grids=[image.grid for _ in items for image in shared.images],
                token_id=self.image_token,
                tower=VISION_TOWER,
                features=self._image_features(shared, rows),
            )
        return self._readout(items, ids, mask, last, images)

    def _mixed_pass(self, items: list[Item]) -> list[list[float] | None]:
        """Media rows of any requests in one pass: their images read by the tower in one call, videos once each."""
        assert self.image_token is not None
        image_features, image_grids = [], []
        video_features, video_grids = [], []
        merged = self._merged_image_features(
            [item.media for item in items if item.media is not None]
        )
        for item in items:
            media = item.media
            assert media is not None
            if media.videos:
                assert self.video_token is not None
                features, clips = self._features(media)
                video_features.append(clips)
                video_grids += [clip.grid for clip in media.videos]
            else:
                features = merged[id(media)]
            if media.images:
                image_features.append(features)
                image_grids += [image.grid for image in media.images]
        device = self.engine_model.device
        images = ImageInputs(
            pixel_values=None,
            grids=image_grids,
            token_id=self.image_token,
            tower=VISION_TOWER,
            features=(
                torch.cat(image_features)
                if image_features
                else torch.empty(
                    (0, self.readout.shape[1]), device=device, dtype=torch.bfloat16
                )
            ),
            videos=(
                VideoInputs(
                    features=torch.cat(video_features),
                    grids=video_grids,
                    token_id=self.video_token,
                )
                if video_features and self.video_token is not None
                else None
            ),
        )
        if self._packs(items):
            ids, last = self._packed(items)
            return self._readout(
                items, ids, torch.ones_like(ids), last, images, packed=True
            )
        ids, mask, last = self._padded(items)
        return self._readout(items, ids, mask, last, images)

    def _merged_image_features(
        self, shared: list[RequestMedia]
    ) -> dict[int, torch.Tensor]:
        """The tower's features of each request's images (by ``id`` of its media), for a pass that merges requests.

        A request whose features one call over its own images gave (``_image_features``) reuses them; the
        others' images go through the tower in one call, packed where the engine packs rows. Those features
        round as that call does, so they are cached apart from the exact path's.
        """
        out: dict[int, torch.Tensor] = {}
        pending: dict[int, RequestMedia] = {}
        for media in shared:
            if id(media) in out or id(media) in pending:
                continue
            if not media.images or media.videos:
                continue
            digests = tuple(image.digest for image in media.images)
            cached = self.feature_cache.get((digests, 1))
            if cached is None:
                cached = self.feature_cache.get(("merged", digests))
            if cached is not None:
                out[id(media)] = cached
            else:
                pending[id(media)] = media
        missing = list(pending.values())
        if len(missing) == 1:
            out[id(missing[0])] = self._image_features(missing[0], 1)
        elif missing:
            device = self.engine_model.device
            features = self.engine_model.tower_features(
                VISION_TOWER,
                torch.cat([media.rows_on(device) for media in missing]),
                [image.grid for media in missing for image in media.images],
                packed=self.packs_merged_rows,
            )
            ends = itertools.accumulate(
                sum(image.tokens for image in media.images) for media in missing
            )
            start = 0
            for media, end in zip(missing, ends, strict=True):
                part = features[start:end].clone()
                start = end
                digests = tuple(image.digest for image in media.images)
                self.feature_cache.put(("merged", digests), part, _bytes(part))
                out[id(media)] = part
        return out

    def _readout(
        self,
        items: list[Item],
        ids: torch.Tensor,
        mask: torch.Tensor,
        last: torch.Tensor,
        images: ImageInputs | None,
        states: list[int] | None = None,
        packed: bool = False,
    ) -> list[list[float] | None]:
        output = self.engine_model.forward(
            ForwardBatch(
                input_ids=ids,
                attention_mask=mask,
                gather=last[:, None],
                query=last,
                lengths=[len(item.ids) for item in items],
                images=images,
                states=states,
                packed=packed,
            )
        )
        return self._probabilities(items, output.query)

    def _probabilities(
        self, items: list[Item], hidden: torch.Tensor
    ) -> list[list[float] | None]:
        """Each question's option probabilities from its last token's final hidden state, at its temperature."""
        counts = torch.as_tensor(
            [len(item.keys) for item in items], device=self.readout.device
        )
        temperatures = [
            self.temperature if item.temperature is None else item.temperature
            for item in items
        ]
        with torch.inference_mode():
            if self.readout.dtype == torch.float32:
                logits = hidden.float() @ self.readout.T
            else:
                logits = torch.nn.functional.linear(hidden, self.readout).float()
            invalid = (
                torch.arange(MAX_OPTIONS, device=logits.device)[None] >= counts[:, None]
            )
            probs = _scaled(
                logits.masked_fill(invalid, float("-inf")), temperatures
            ).softmax(-1)
            values = probs.cpu().tolist()
        return [row[: len(item.keys)] for row, item in zip(values, items, strict=True)]

    def _image_features(self, media: RequestMedia, copies: int) -> torch.Tensor:
        """The tower's features of ``copies`` copies of a request's images in one call, cached by content.

        One tower call over k copies is what a k-row pass of the released runtime runs, so the cached value is
        that pass's features exactly, for every later pass of k rows over the same images.
        """
        key = (tuple(image.digest for image in media.images), copies)
        cached = self.feature_cache.get(key)
        if cached is not None:
            return cached
        placed = media.rows_on(self.engine_model.device)
        features = self.engine_model.tower_features(
            VISION_TOWER,
            placed if copies == 1 else torch.cat([placed] * copies),
            [image.grid for _ in range(copies) for image in media.images],
        )
        self.feature_cache.put(key, features, _bytes(features))
        return features

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
            if item.task_type in ("box", "track"):
                from . import composed

                # A Box or track item's values are one distribution per readout, not one.
                rows = cast(list[list[float]], values)
                edges = rows[-len(composed.EDGES) :]
                answer = composed.box_answer(edges, self.box_calibration)
                if item.task_type == "track":
                    return {
                        "time": item.at,
                        "box": answer["box"],
                        "confidence": answer["confidence"],
                        "visible": float(rows[0][1] / sum(rows[0])),
                    }
                if item.at is not None:
                    answer["time"] = item.at
                return answer
            return product_answer(item, values)
        except ValueError as exc:
            return {
                "type": item.task_type,
                "error": INVALID_MODEL_OUTPUT,
                "message": str(exc),
            }


def _scaled(logits: torch.Tensor, temperatures: list[float]) -> torch.Tensor:
    """Each row of ``logits`` divided by its temperature: one division by a scalar per temperature, as a request's
    own pass divides (a GPU divides by a scalar as a product with its reciprocal, not as an element-wise quotient).
    """
    distinct = sorted(set(temperatures))
    if len(distinct) == 1:
        return logits / distinct[0]
    out = torch.empty_like(logits)
    for value in distinct:
        rows = torch.tensor(
            [row for row, t in enumerate(temperatures) if t == value],
            device=logits.device,
        )
        out[rows] = logits[rows] / value
    return out


def _bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()
