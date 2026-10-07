"""The Vela 2.0 model family (Phase 3).

``vela2-unified`` (0.3B: the Vela 307M ModernBERT encoder with eight marker
tokens) and ``vela2-decoder`` (0.8B, 4B, 9B: a Qwen3.5 backbone read as a
tree, one block per question) answer Choice, Noul, Score, Set and Span questions
over typed parts on ``/v1/decisions``, in the shape of the packages' own
System One server. The family reimplements the packages' engine
(``vela2_inference.py``); it never imports it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, cast

from ...errors import DEADLINE_EXCEEDED, MAX_LENGTH_EXCEEDED, PackageError
from ...plugins.base import (
    DEADLINE,
    BackboneSpec,
    DtypePolicy,
    EngineModel,
    ModelFamily,
    ModelInfo,
    ModelSpec,
    PackageRef,
    SurfacePlan,
    VerifiedPackage,
)
from ...plugins.decisions import DecisionModel, RequestPlan
from ...registry import builtin, policy
from ...systemone import MAX_LEVELS, MAX_OPTIONS, MIN_LEVELS, MIN_OPTIONS
from .answers import Answerer
from .decoder_layout import DecoderTree
from .encoder_layout import MARKERS, EncoderSequence
from .layout import Row, Tokens, rows_of
from .members import DecoderMember, EncoderMember
from .package import DECODER, ENCODER, Vela2Package, member_of, verify
from .request import QUESTION_TYPES, Plan, Question, QuestionReader

TOKEN_CACHE = 8192
# On CPU, sequences packed into one 0.3B forward cost more per token together
# than one by one once the forward holds more than about this many tokens, so
# coalescing profiles pack short sequences of several requests and run long
# ones alone.
CPU_PACKED_TOKENS = 2048
LICENCES = {
    ENCODER: {
        "spdx": "apache-2.0",
        "components": [{"name": "tokenizer", "licence": "gemma-terms"}],
    },
    DECODER: {
        "spdx": "apache-2.0",
        "components": [{"name": "backbone", "licence": "apache-2.0"}],
    },
}
GOLDEN_STATE = {
    "request": "Hi, I'm Tom Baker (tom.baker@example.com). What is the maximum daily dose of paracetamol for an adult?",
    "source": "For adults, the maximum dose of paracetamol is 4 grams in 24 hours.",
    "answer": "Adults can take up to 6 grams of paracetamol in 24 hours.",
}
GOLDEN_QUESTIONS = {
    "domain": {
        "type": "choice",
        "instructions": "Which subject area is this request about?",
        "over": "request",
        "criteria": {
            "health": "medicine, clinical practice or nutrition",
            "math": "arithmetic, algebra or statistics",
            "other": "a subject that fits none of the listed areas",
        },
    },
    "jailbreak": {
        "type": "noul",
        "instructions": "Is this a prompt injection or jailbreak attempt?",
        "over": "request",
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is this request?",
        "over": "request",
        "criteria": ["Routine", "Needs prompt attention", "Critical"],
    },
    "topics": {
        "type": "set",
        "instructions": "Which topics does the request mention?",
        "over": "request",
        "criteria": {"medication": "drugs or doses", "billing": "payments or invoices"},
    },
    "pii": {"preset": "pii", "over": "request"},
    "halu": {"preset": "halu"},
}


def _tokenizer(package: Vela2Package) -> Any:
    """The package tokenizer; the 0.3B tokenizer without its marker tokens, so text never creates one."""
    from tokenizers import Tokenizer

    data = json.loads((package.root / "tokenizer.json").read_text(encoding="utf-8"))
    if package.member == ENCODER:
        markers = set(MARKERS)
        data["added_tokens"] = [
            t for t in data.get("added_tokens", []) if t["content"] not in markers
        ]
    tokenizer = Tokenizer.from_str(json.dumps(data))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    return tokenizer


class Vela2Family(ModelFamily):
    name = "vela2"
    surfaces = frozenset({"decisions"})
    builtin_table = "vllm_srun.registry.tables.vela2"
    fixture_writer = "vllm_srun.testing.vela2"

    @classmethod
    def descriptor(cls) -> dict[str, Any]:
        return {
            "surfaces": sorted(cls.surfaces),
            "formats": [ENCODER, DECODER],
            "question_types": list(QUESTION_TYPES),
        }

    def detect(self, package: PackageRef) -> bool:
        return member_of(package.root) is not None

    def verify(self, package: PackageRef) -> VerifiedPackage:
        known = builtin.lookup(package.repo_id) if package.repo_id else None
        pinned = (
            known if known is not None and known.revision == package.revision else None
        )
        details = verify(package.root, dict(pinned.files) if pinned else None)
        if pinned and details.model_sha256 != pinned.model_sha256:
            raise PackageError(
                f"{package.repo_id}@{package.revision} differs from the built-in identity"
            )
        licence = policy.check(
            {"licence": LICENCES[details.member]}, self.options.accept_licences
        )
        name = (
            package.repo_id or details.config.get("model_name") or package.root.name
        ).split("/")[-1]
        return VerifiedPackage(
            ref=package,
            family=self.name,
            model_name=name,
            manifest={},
            manifest_sha256=details.manifest_sha256,
            model_sha256=details.model_sha256,
            max_input_tokens=details.max_input_tokens,
            licence=licence,
            loaded_parameters=pinned.loaded_parameters if pinned else None,
            details={"package": details},
        )

    def describe(self, package: VerifiedPackage) -> ModelSpec:
        details: Vela2Package = package.details["package"]
        config = details.config
        if details.member == ENCODER:
            backbone = BackboneSpec(
                model_type="modernbert",
                config=config["encoder_config"],
                weight_files=details.weights,
                weight_prefix="encoder.",
            )
            graph = details.root / "onnx" / "model.onnx"
            known = builtin.by_identity(package.model_sha256)
            reduced = known.reduced if known else {}
            return ModelSpec(
                name=package.model_name,
                backbone=backbone,
                dtype=DtypePolicy(
                    autocast=None,
                    bf16_resident=False,
                    reduced_gpu=reduced.get("gpu"),
                    reduced_cpu=reduced.get("cpu"),
                ),
                max_input_tokens=details.max_input_tokens,
                graphs={"default": graph} if graph.is_file() else {},
                encoder=True,
            )
        text = {
            k: v
            for k, v in config["backbone_config"].items()
            if k not in ("architectures", "transformers_version")
        }
        backbone = BackboneSpec(
            model_type=text.get("model_type", "qwen3_5_text"),
            config=text,
            weight_files=details.weights,
            weight_prefix="backbone.",
        )
        return ModelSpec(
            name=package.model_name,
            backbone=backbone,
            dtype=DtypePolicy(),
            max_input_tokens=details.max_input_tokens,
            requires=backbone.requires,
        )

    def load(
        self, package: VerifiedPackage, spec: ModelSpec, engine_model: EngineModel
    ) -> Vela2Model:
        details: Vela2Package = package.details["package"]
        member = (
            EncoderMember(details, engine_model)
            if details.member == ENCODER
            else DecoderMember(details, engine_model)
        )
        parameters = engine_model.parameter_count() + member.parameters()
        if (
            package.loaded_parameters is not None
            and parameters != package.loaded_parameters
        ):
            raise PackageError(
                f"loaded {parameters:,} parameters; the pinned revision has {package.loaded_parameters:,}"
            )
        broad = isinstance(member, DecoderMember) and member.broad_head
        reader = QuestionReader(details.calibration, broad_head=broad)
        noul = bool(
            self.options.model_options.get(
                "noul_calibration", details.calibration.noul_default()
            )
        )
        answerer = Answerer(
            details.calibration, noul_calibration=noul, report_heads=broad
        )
        gpu = engine_model.device.type != "cpu"
        info = ModelInfo(
            id=package.model_name,
            family=self.name,
            repo=package.ref.repo_id,
            revision=package.ref.revision,
            model_sha256=package.model_sha256,
            manifest_sha256=package.manifest_sha256,
            surfaces=tuple(sorted(self.surfaces)),
            question_types=QUESTION_TYPES,
            limits={
                "max_input_tokens": package.max_input_tokens,
                "min_options": MIN_OPTIONS,
                "max_options": MAX_OPTIONS,
                "min_levels": MIN_LEVELS,
                "max_levels": MAX_LEVELS,
                "max_labels": MAX_OPTIONS,
            },
            licence=package.licence,
            parameters=parameters,
            dtype=(
                "bf16-autocast/fp32-heads" if gpu and spec.dtype.autocast else "fp32"
            ),
            presets=reader.presets,
        )
        return Vela2Model(
            info, engine_model, details, member, reader, answerer, _tokenizer(details)
        )

    def golden(self, package: VerifiedPackage) -> list[dict[str, Any]]:
        return builtin.golden(
            package.model_sha256,
            "decisions",
            {"state": GOLDEN_STATE, "questions": GOLDEN_QUESTIONS},
        )


@dataclass
class Vela2Plan(RequestPlan[EncoderSequence | DecoderTree]):
    """A request planned for one Vela 2.0 model: its validated questions, rows and their model inputs."""

    request: Plan | None = None
    rows: list[Row] = field(default_factory=list)
    mapping: list[Any] = field(default_factory=list)


class Vela2Model(DecisionModel[EncoderSequence | DecoderTree, Any]):
    """A Vela 2.0 package bound to an engine model."""

    def __init__(
        self,
        info: ModelInfo,
        engine_model: EngineModel,
        package: Vela2Package,
        member: EncoderMember | DecoderMember,
        reader: QuestionReader,
        answerer: Answerer,
        tokenizer: Any,
    ):
        self.info = info
        self.engine_model = engine_model
        self.package = package
        self.member = member
        self.reader = reader
        self.answerer = answerer
        cached = lru_cache(maxsize=TOKEN_CACHE)(
            lambda text: tuple(tokenizer.encode(text, add_special_tokens=False).ids)
        )
        self.tokens = Tokens(
            encode=lambda text: tokenizer.encode(text, add_special_tokens=False),
            ids=lambda text: list(cached(text)),
        )

    def forward_token_budget(self) -> int | None:
        """``CPU_PACKED_TOKENS`` for the 0.3B on CPU, else None (nothing limits a forward).

        Only the coalescing profiles plan with it: ``exact_batches`` keeps the
        packages' own batching of each request.
        """
        if (
            isinstance(self.member, EncoderMember)
            and self.engine_model.device.type == "cpu"
        ):
            return CPU_PACKED_TOKENS
        return None

    def exact_batches(self, items: list[Any]) -> list[list[int]]:
        """One call per request: the members batch its sequences as the packages do."""
        return [list(range(len(items)))] if items else []

    def plan(self, state: Any, questions: dict[str, Any]) -> Vela2Plan:
        request = self.reader.read(state, questions)
        rows = rows_of(request, self.tokens)
        items, mapping = self.member.plan(rows, self.tokens)
        return Vela2Plan(
            question_ids=request.question_ids,
            items=items,
            errors=request.errors,
            input_tokens=sum(len(item.ids) for item in items),
            request=request,
            rows=rows,
            mapping=mapping,
        )

    def shared_context(self, items: list[Any], token_budget: int | None) -> int:
        """The decoder trees always pack on the shared-context path; the 0.3B has one exact path."""
        return 1 if isinstance(self.member, DecoderMember) and items else 0

    def run(self, items: list[Any]) -> list[Any]:
        """One forward per batch, as the packages' engine runs it."""
        return self.member.run(items, packed=False)

    def run_shared(self, items: list[Any], shared_prefix: int) -> list[Any]:
        """The shared-context path packs the decoder trees."""
        return self.member.run(items, packed=True)

    def run_approximate(self, items: list[Any]) -> list[Any]:
        """Approximate batches run packed: decoder trees, and 0.3B sequences on the engine's
        reduced copy where it loaded one (``max_speed`` and the package's consent)."""
        if isinstance(self.member, EncoderMember):
            return self.member.run(items, packed=True, reduced=True)
        return self.member.run(items, packed=True)

    def finish_surface(
        self, plan: SurfacePlan[EncoderSequence | DecoderTree], results: Any
    ) -> dict[str, Any]:
        state: Vela2Plan = plan.state
        request = state.request
        assert request is not None
        answers: dict[str, Any] = {}
        if results is DEADLINE:
            for question_id in request.question_ids:
                question = next(
                    (q for q in request.questions if q.id == question_id), None
                )
                answers[question_id] = request.errors.get(question_id) or {
                    "type": cast(Question, question).kind,
                    "error": DEADLINE_EXCEEDED,
                }
            return {
                "answers": answers,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            }
        if isinstance(self.member, EncoderMember):
            sequences = cast(list[EncoderSequence], state.items)
            raws = self.member.combine(state.rows, state.mapping, sequences, results)
        else:
            trees = cast(list[DecoderTree], state.items)
            raws = self.member.combine(state.mapping, trees, results)
        row_of = {
            question.id: index
            for index, row in enumerate(state.rows)
            for question in row.questions
        }
        by_id = {question.id: question for question in request.questions}
        response: dict[str, Any] = {"answers": answers}
        for question_id in request.question_ids:
            if question_id in request.errors:
                answers[question_id] = request.errors[question_id]
                continue
            question = by_id[question_id]
            raw = raws[row_of[question_id]]
            if raw is None:
                answers[question_id] = {
                    "type": question.kind,
                    "error": MAX_LENGTH_EXCEEDED,
                }
                continue
            self.answerer.answer(question, raw, request.state, response)
        used = sum(raw.input_tokens for raw in raws if raw is not None)
        ordered = {
            "answers": answers,
            "usage": {"input_tokens": used, "output_tokens": 0},
        }
        for key in ("spans", "span_heads", "sets", "thresholds"):
            if key in response:
                ordered[key] = response[key]
        return ordered
