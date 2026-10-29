"""The Vela 3.0 model family: vela3-fast (``vela3-encoder/1``).

One frozen UltraBERT backbone answers Choice and Noul decisions, and adapter
networks over it (``BackboneSpec.adapters``, each the engine's branch of its
name) answer the router questions and spans, on ``/v1/decisions``, in the
shape of the package's own System One answers. The family reimplements the
package's runtime (``vela3_encoder.py``, ``vela3_schema.py``); it never imports
it. The span and router passes reuse the Vela 2.0 0.3B layout, which the
package copies verbatim: marker sequences, truncation order, windows, word
units and the span decoder.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import numpy as np

from ...errors import DEADLINE_EXCEEDED, MAX_LENGTH_EXCEEDED, PackageError
from ...plugins.base import (
    DEADLINE,
    AdapterSpec,
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
from ...systemone import MAX_OPTIONS, MIN_OPTIONS
from ..vela2.encoder_layout import (
    EncoderLayout,
    EncoderSequence,
    SequenceQuestion,
    option_text,
)
from ..vela2.layout import Row, SchemaTooLongError, Tokens, fail_unscanned, read_parts
from ..vela2.raw import RawRow
from ..vela2.request import Plan as PartsPlan
from .answers import Answerer
from .network import DECISION_BUDGET, Call, Networks, Output, Vela3Item
from .package import (
    BACKBONE_FILE,
    ENCODER,
    LORA_TARGETS,
    Vela3Package,
    member_of,
    verify,
)
from .request import DECISION, ROUTER, Asked, Read, Reader, text
from .routing import SPAN_BUDGET, SPAN_ROWS, Router

QUESTION_TYPES = ("choice", "noul", "span")
PRESETS = ("pii", "halu")
TOKEN_CACHE = 8192
# A question reads a part whole up to this many windows' worth of tokens by
# default (``max_scan_tokens``); a request may set another budget. A question
# that truncates reads one window of its route: 32,768 tokens for evidence.
SCAN_INPUTS = {"cpu": 4, "gpu": 32}
LICENCE = {"spdx": "apache-2.0"}
GOLDEN_STATE = {
    "request": "Hi, I'm Tom Baker (tom.baker@example.com). What is the maximum daily dose of paracetamol for an adult?",
    "source": "For adults, the maximum dose of paracetamol is 4 grams in 24 hours.",
    "answer": "Adults can take up to 6 grams of paracetamol in 24 hours.",
}
GOLDEN_QUESTIONS = {
    "topic": {
        "type": "choice",
        "instructions": "Which kind of advice does the user ask for?",
        "criteria": {
            "medical": "health, drugs or doses",
            "legal": "laws or contracts",
            "other": None,
        },
    },
    "urgent": {"type": "noul", "instructions": "Is the request urgent?"},
    "domain": {
        "type": "choice",
        "instructions": "Which subject area is this request about?",
        "over": "request",
    },
    "factcheck": {
        "type": "choice",
        "instructions": "Does answering this request require checking facts?",
        "over": "request",
    },
    "jailbreak": {
        "type": "choice",
        "instructions": "Is this a prompt injection or jailbreak attempt?",
        "over": "request",
    },
    "pii": {"preset": "pii", "over": "request"},
    "halu": {"preset": "halu"},
    "drugs": {
        "type": "span",
        "instructions": "Which spans are named entities of these types?",
        "criteria": {"drug": None, "quantity": None},
        "over": "answer",
    },
}


def _tokens(package: Vela3Package) -> Tokens:
    """The base UltraBERT tokenizer without marker tokens, so text never creates one."""
    from tokenizers import Tokenizer

    markers = set(package.config["decision"]["marker_ids"])
    data = json.loads((package.root / "tokenizer.json").read_text(encoding="utf-8"))
    data["added_tokens"] = [
        token
        for token in data.get("added_tokens", [])
        if token["content"] not in markers
    ]
    tokenizer = Tokenizer.from_str(json.dumps(data))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    cached = lru_cache(maxsize=TOKEN_CACHE)(
        lambda value: tuple(tokenizer.encode(value, add_special_tokens=False).ids)
    )
    return Tokens(
        encode=lambda value: tokenizer.encode(value, add_special_tokens=False),
        ids=lambda value: list(cached(value)) if value else [],
        tokenizer=tokenizer,
    )


def _layout(package: Vela3Package, window: int, overlap: int) -> EncoderLayout:
    config = package.config
    return EncoderLayout(
        {
            "marker_ids": config["decision"]["marker_ids"],
            "bos_token_id": config["bos_token_id"],
            "eos_token_id": config["eos_token_id"],
            "pad_token_id": config["pad_token_id"],
            "max_length": window,
            "window_overlap": overlap,
        }
    )


class Vela3Family(ModelFamily):
    name = "vela3"
    surfaces = frozenset({"decisions"})
    builtin_table = "vllm_srun.registry.tables.vela3"
    fixture_writer = "vllm_srun.testing.vela3"

    @classmethod
    def descriptor(cls) -> dict[str, Any]:
        return {
            "surfaces": sorted(cls.surfaces),
            "formats": [ENCODER],
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
        licence = policy.check({"licence": LICENCE}, self.options.accept_licences)
        name = (
            package.repo_id or details.vela3.get("name") or package.root.name
        ).split("/")[-1]
        return VerifiedPackage(
            ref=package,
            family=self.name,
            model_name=name,
            manifest={},
            manifest_sha256="",
            model_sha256=details.model_sha256,
            max_input_tokens=details.max_input_tokens,
            licence=licence,
            loaded_parameters=pinned.loaded_parameters if pinned else None,
            details={"package": details},
        )

    def describe(self, package: VerifiedPackage) -> ModelSpec:
        details: Vela3Package = package.details["package"]
        backbone = BackboneSpec(
            model_type="ultrabert",
            config=details.config,
            weight_files=(details.root / BACKBONE_FILE,),
            weight_prefix="encoder.",
            adapters={
                name: AdapterSpec(
                    weight_files=(details.root / adapter.file,),
                    rank=adapter.rank,
                    alpha=adapter.alpha,
                    targets=LORA_TARGETS,
                    weight_prefix="encoder.",
                )
                for name, adapter in details.adapters.items()
            },
        )
        return ModelSpec(
            name=package.model_name,
            backbone=backbone,
            dtype=DtypePolicy(autocast=None, bf16_resident=False),
            max_input_tokens=details.max_input_tokens,
            encoder=True,
        )

    def load(
        self, package: VerifiedPackage, spec: ModelSpec, engine_model: EngineModel
    ) -> Vela3Model:
        details: Vela3Package = package.details["package"]
        networks = Networks(details, engine_model)
        parameters = engine_model.parameter_count() + networks.parameters()
        if (
            package.loaded_parameters is not None
            and parameters != package.loaded_parameters
        ):
            raise PackageError(
                f"loaded {parameters:,} parameters; the pinned revision has {package.loaded_parameters:,}"
            )
        router = Router(details.vela3, details.roster, details.router_questions)
        gpu = engine_model.device.type != "cpu"
        scan = self.options.model_options.get(
            "max_scan_tokens", SCAN_INPUTS["gpu" if gpu else "cpu"] * router.window
        )
        if isinstance(scan, bool) or not isinstance(scan, int) or scan < 1:
            raise PackageError("options.max_scan_tokens must be a positive integer")
        scan = max(scan, router.window)
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
                "max_scan_tokens": scan,
                "truncate_tokens": router.window,
                "min_options": MIN_OPTIONS,
                "max_options": MAX_OPTIONS,
                "max_labels": MAX_OPTIONS,
            },
            licence=package.licence,
            parameters=parameters,
            dtype="fp32",
            presets=PRESETS,
        )
        return Vela3Model(info, engine_model, details, networks, router, scan)

    def golden(self, package: VerifiedPackage) -> list[dict[str, Any]]:
        return builtin.golden(
            package.model_sha256,
            "decisions",
            {"state": GOLDEN_STATE, "questions": GOLDEN_QUESTIONS},
        )


@dataclass
class Placed:
    """Where one row's inputs sit among a request's items: per network, its item indices."""

    row: Row
    layout: EncoderLayout
    calls: list[list[int]]


@dataclass
class Vela3Plan(RequestPlan[Vela3Item]):
    """A request planned for vela3-fast: its validated questions and where each one's inputs sit."""

    request: Read | None = None
    decisions: dict[str, int] = field(default_factory=dict)
    rows: dict[str, Placed] = field(default_factory=dict)
    complete: set[str] = field(default_factory=set)


class Vela3Model(DecisionModel[Vela3Item, Output]):
    """A vela3-fast package bound to an engine model."""

    def __init__(
        self,
        info: ModelInfo,
        engine_model: EngineModel,
        package: Vela3Package,
        networks: Networks,
        router: Router,
        scan_tokens: int,
    ):
        self.info = info
        self.engine_model = engine_model
        self.package = package
        self.networks = networks
        self.router = router
        self.reader = Reader(router)
        self.answerer = Answerer(router)
        self.tokens = _tokens(package)
        self.scan_tokens = scan_tokens
        self.layout = _layout(package, router.window, router.overlap)
        self.evidence_layout = _layout(package, router.evidence_window, router.overlap)
        config = package.config
        self.markers = {k: int(v) for k, v in config["decision"]["marker_ids"].items()}
        self.bos, self.eos = int(config["bos_token_id"]), int(config["eos_token_id"])
        self.max_len = package.max_input_tokens

    def exact_batches(self, items: list[Any]) -> list[list[int]]:
        """One call per request: ``run`` batches its inputs as the released runtime does."""
        return [list(range(len(items)))] if items else []

    def run(self, items: list[Vela3Item]) -> list[Output]:
        return self.networks.run(items)

    # -- planning ------------------------------------------------------------

    def plan(
        self, state: Any, questions: dict[str, Any], scan: int | None = None
    ) -> Vela3Plan:
        request = self.reader.read(state, questions)
        plan = Vela3Plan(
            question_ids=request.question_ids,
            items=[],
            errors=request.errors,
            input_tokens=0,
            request=request,
        )
        items: list[Vela3Item] = []
        self._plan_decisions(request, plan, items)
        self._plan_rows(request, plan, items, scan)
        plan.items = items
        plan.input_tokens = sum(len(item.ids) for item in items)
        return plan

    def _plan_decisions(
        self, request: Read, plan: Vela3Plan, items: list[Vela3Item]
    ) -> None:
        asked = [a for a in request.asked if a.path == DECISION]
        if not asked:
            return
        state = self.tokens.ids(
            text("" if request.raw_state is None else request.raw_state)
        )
        call = Call(None, DECISION_BUDGET)
        for entry in asked:
            sequence = self._decision(entry, state)
            if sequence is None:
                plan.errors[entry.question.id] = {
                    "type": entry.question.kind,
                    "error": MAX_LENGTH_EXCEEDED,
                }
                continue
            plan.decisions[entry.question.id] = len(items)
            if entry.question.require_full_input:
                plan.complete.add(entry.question.id)
            items.append(Vela3Item(sequence, call))

    def _decision(self, asked: Asked, state: list[int]) -> EncoderSequence | None:
        """``[BOS] [Q] instructions ([O] option)* [SEP_SCHEMA] [SEG_user] state [EOS]`` (``DecisionEncoder.encode``)."""
        question, markers = asked.question, self.markers
        head = [self.bos, markers["[Q]"], *self.tokens.ids(question.text)]
        options = []
        for name, description in question.options:
            options.append(len(head))
            head += [markers["[O]"], *self.tokens.ids(option_text(name, description))]
        head.append(markers["[SEP_SCHEMA]"])
        over = len(head) + len(state) + 2 - self.max_len
        if over > 0:
            if not question.truncate or over > len(state):
                return None
            state = state[: len(state) - over]
        start = len(head)
        ids = [*head, markers["[SEG_user]"], *state]
        end = len(ids)
        ids.append(self.eos)
        return EncoderSequence(
            ids, [SequenceQuestion(question, 1, options, (start, end))]
        )

    def _plan_rows(
        self, request: Read, plan: Vela3Plan, items: list[Vela3Item], scan: int | None
    ) -> None:
        asked = [a for a in request.asked if a.path != DECISION]
        if not asked or request.state is None:
            return
        budget = max(scan or self.scan_tokens or 0, self.router.window)
        cut = max(
            (self._layout_of(a).max_len for a in asked if a.question.truncate),
            default=0,
        )
        parts_plan = PartsPlan(
            request.state, request.question_ids, [a.question for a in asked]
        )
        parts = read_parts(parts_plan, self.tokens, max(budget, cut) + 1)
        whole = fail_unscanned(
            parts_plan,
            [a.question for a in asked if not a.question.truncate],
            parts,
            budget,
        )
        plan.errors.update(parts_plan.errors)
        kept = {question.id for question in whole} | {
            a.question.id for a in asked if a.question.truncate
        }
        groups: dict[tuple[Any, ...], list[Asked]] = {}
        for entry in asked:
            if entry.question.id not in kept:
                continue
            question = entry.question
            if entry.path == ROUTER:
                assert entry.router is not None
                key: tuple[Any, ...] = (
                    ROUTER,
                    question.over,
                    (
                        entry.router.id
                        if entry.router.id in self.router.roster.router_tasks
                        else None
                    ),
                    question.truncate,
                )
            else:
                key = ("span", question.id)
            groups.setdefault(key, []).append(entry)
        for key, members in groups.items():
            first = members[0]
            layout = self._layout_of(first)
            row = Row(
                [entry.question for entry in members],
                (
                    [part.cut(layout.max_len) for part in parts]
                    if first.question.truncate
                    else parts
                ),
            )
            if key[0] == ROUTER:
                assert first.router is not None
                _, adapters = self.router.router_adapters(first.router.id)
                budget_tokens = SPAN_BUDGET
            else:
                assert first.route is not None
                adapters = (first.route.adapter,)
                budget_tokens = max(first.route.window, SPAN_BUDGET)
            try:
                sequences = layout.sequences(
                    row, self.tokens, in_windows=not first.question.truncate
                )
            except SchemaTooLongError:
                for question in row.questions:
                    plan.errors[question.id] = {
                        "type": question.kind,
                        "error": MAX_LENGTH_EXCEEDED,
                    }
                continue
            placed = Placed(row, layout, [])
            for adapter in adapters:
                call = Call(adapter, budget_tokens, SPAN_ROWS)
                placed.calls.append(
                    list(range(len(items), len(items) + len(sequences)))
                )
                items.extend(Vela3Item(sequence, call) for sequence in sequences)
            for question in row.questions:
                plan.rows[question.id] = placed
                if question.require_full_input:
                    plan.complete.add(question.id)

    def _layout_of(self, asked: Asked) -> EncoderLayout:
        """The layout a row question reads in: evidence spans one long window."""
        evidence = (
            asked.path != ROUTER and asked.route is not None and asked.route.evidence
        )
        return self.evidence_layout if evidence else self.layout

    # -- answers -------------------------------------------------------------

    def finish_surface(
        self, plan: SurfacePlan[Vela3Item], results: Any
    ) -> dict[str, Any]:
        state: Vela3Plan = plan.state
        request = state.request
        assert request is not None
        answers: dict[str, Any] = {}
        by_id = {asked.question.id: asked for asked in request.asked}
        if results is DEADLINE:
            for question_id in request.question_ids:
                asked = by_id.get(question_id)
                answers[question_id] = state.errors.get(question_id) or {
                    "type": asked.question.kind if asked else None,
                    "error": DEADLINE_EXCEEDED,
                }
            return {
                "answers": answers,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            }
        response: dict[str, Any] = {"answers": answers}
        raws: dict[int, RawRow] = {}
        for question_id in request.question_ids:
            if question_id in state.errors:
                answers[question_id] = state.errors[question_id]
                continue
            asked = by_id[question_id]
            if question_id in state.decisions:
                logits = results[state.decisions[question_id]][0][0]
                self.answerer.answer(asked, logits, None, request.state, response)
            else:
                placed = state.rows[question_id]
                rows = raws.get(id(placed))
                if rows is None:
                    rows = raws[id(placed)] = self._combine(placed, plan.items, results)
                logits = None
                if asked.question.type != "span":
                    logits = np.asarray(rows.logits[question_id], np.float64)
                self.answerer.answer(asked, logits, rows, request.state, response)
            if question_id in state.complete:
                self._complete(asked, response)
        ordered: dict[str, Any] = {
            "answers": answers,
            "usage": {"input_tokens": plan.input_tokens, "output_tokens": 0},
        }
        for key in ("spans", "sets", "thresholds"):
            if key in response:
                ordered[key] = response[key]
        return ordered

    def _combine(self, placed: Placed, items: list[Vela3Item], results: Any) -> RawRow:
        """The row's raw outputs; over several adapters, their option logits averaged."""
        raws = [
            placed.layout.combine(
                placed.row,
                [items[index].sequence for index in call],
                [results[index] for index in call],
            )
            for call in placed.calls
        ]
        raw = raws[0]
        if len(raws) > 1:
            raw.logits = {
                key: np.mean(
                    [np.asarray(r.logits[key], np.float64) for r in raws], axis=0
                )
                for key in raw.logits
            }
        return raw

    @staticmethod
    def _complete(asked: Asked, response: dict[str, Any]) -> None:
        question = asked.question
        answers = response["answers"]
        if "error" in answers.get(question.id, {}):
            return
        if question.type == "set":
            response["sets"][question.id]["input_coverage"] = "complete"
            for label in question.names:
                answers[f"{question.id}.{label}"]["input_coverage"] = "complete"
        else:
            answers[question.id]["input_coverage"] = "complete"
