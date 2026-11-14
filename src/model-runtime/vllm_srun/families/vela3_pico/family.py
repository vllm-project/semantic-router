"""The Vela 3.1 pico family: vela3.1-pico-380m and -150m (``vela3-pico/1``).

One UltraBERT encoder, fine-tuned whole, answers every question type a pico
reads on ``/v1/decisions``, in the shape of the package's own System One
answers. The family reimplements the package's runtime (``render31.py``,
``vela3_encoder.py``, ``vela3_schema.py``); it never imports it.

A request whose questions the package's ``system_one`` would read in one pass
(``request.Reader.one_pass``) is one sequence and one forward when it fits the
window uncut (``sequence.build``); otherwise it is answered question by
question (``fallback``): decisions on the decision layout, router questions
and span questions in the Vela 2.0 layout and windows the package copies.
Questions are rendered as given (``families.render31``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ...errors import DEADLINE_EXCEEDED, PackageError
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
from ...systemone import MAX_OPTIONS, MIN_OPTIONS
from ..vela2.layout import read_parts
from ..vela2.raw import RawRow, RawSpan
from ..vela2.request import Plan as PartsPlan
from ..vela3_answers import Answerer
from ..vela3_calls import Call, Output, Vela3Item
from ..vela3_shared import PRESETS, layout, tokens
from . import fallback, sequence
from .network import Network
from .package import FORMAT, WEIGHTS_FILE, PicoPackage, member_of, verify
from .request import Asked, Read, Reader
from .routing import Routes

QUESTION_TYPES = ("choice", "noul", "span")
# A question reads a part whole up to this many windows' worth of tokens by
# default (``max_scan_tokens``) on the question-by-question path; a request
# may set another budget. The one pass reads one window.
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
    "jailbreak": {
        "type": "choice",
        "instructions": "Is this a prompt injection or jailbreak attempt?",
        "over": "request",
    },
    "hazards": {
        "type": "set",
        "instructions": "Which hazard categories apply to this request?",
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


def refuse_blank(state: Any) -> None:
    """Refuse a blank text state, as the other Vela families do (``vela2.request.read_state``)."""
    if isinstance(state, str) and not state.strip():
        raise ValueError("state must not be empty or whitespace")


class Vela3PicoFamily(ModelFamily):
    name = "vela3_pico"
    surfaces = frozenset({"decisions"})
    builtin_table = "vllm_srun.registry.tables.vela3_pico"
    fixture_writer = "vllm_srun.testing.vela3_pico"

    @classmethod
    def descriptor(cls) -> dict[str, Any]:
        return {
            "surfaces": sorted(cls.surfaces),
            "formats": [FORMAT],
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
        name = (package.repo_id or details.pico.get("name") or package.root.name).split(
            "/"
        )[-1]
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
        details: PicoPackage = package.details["package"]
        return ModelSpec(
            name=package.model_name,
            backbone=BackboneSpec(
                model_type="ultrabert",
                config=details.config,
                weight_files=(details.root / WEIGHTS_FILE,),
                weight_prefix="encoder.",
            ),
            dtype=DtypePolicy(autocast=None, bf16_resident=False),
            max_input_tokens=details.max_input_tokens,
            encoder=True,
        )

    def load(
        self, package: VerifiedPackage, spec: ModelSpec, engine_model: EngineModel
    ) -> Vela3PicoModel:
        details: PicoPackage = package.details["package"]
        network = Network(details, engine_model)
        parameters = engine_model.parameter_count() + network.parameters()
        if (
            package.loaded_parameters is not None
            and parameters != package.loaded_parameters
        ):
            raise PackageError(
                f"loaded {parameters:,} parameters; the pinned revision has {package.loaded_parameters:,}"
            )
        routes = Routes(details.pico, details.router_questions)
        gpu = engine_model.device.type != "cpu"
        scan = self.options.model_options.get(
            "max_scan_tokens", SCAN_INPUTS["gpu" if gpu else "cpu"] * routes.window
        )
        if isinstance(scan, bool) or not isinstance(scan, int) or scan < 1:
            raise PackageError("options.max_scan_tokens must be a positive integer")
        scan = max(scan, routes.window)
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
                "truncate_tokens": routes.window,
                "min_options": MIN_OPTIONS,
                "max_options": MAX_OPTIONS,
                "max_labels": MAX_OPTIONS,
            },
            licence=package.licence,
            parameters=parameters,
            dtype="fp32",
            presets=PRESETS,
        )
        return Vela3PicoModel(info, engine_model, details, network, routes, scan)

    def golden(self, package: VerifiedPackage) -> list[dict[str, Any]]:
        return builtin.golden(
            package.model_sha256,
            "decisions",
            {"state": GOLDEN_STATE, "questions": GOLDEN_QUESTIONS},
        )


@dataclass
class PicoPlan(RequestPlan[Vela3Item]):
    """A request planned for a pico: its read questions, and its one pass or its question-by-question plan."""

    request: Read | None = None
    one_pass: sequence.OnePass | None = None
    fallback: fallback.Fallback | None = None


class Vela3PicoModel(DecisionModel[Vela3Item, Output]):
    """A pico package bound to an engine model."""

    def __init__(
        self,
        info: ModelInfo,
        engine_model: EngineModel,
        package: PicoPackage,
        network: Network,
        routes: Routes,
        scan_tokens: int,
    ):
        self.info = info
        self.engine_model = engine_model
        self.package = package
        self.network = network
        self.routes = routes
        self.reader = Reader(routes)
        self.answerer = Answerer(routes)
        config = package.config
        markers = {k: int(v) for k, v in config["decision"]["marker_ids"].items()}
        self.tokens = tokens(package.root, markers)
        self.scan_tokens = scan_tokens
        bos, eos = int(config["bos_token_id"]), int(config["eos_token_id"])
        self.pass_layout = sequence.Layout(markers, bos, eos, routes.window)
        self.layouts = fallback.Layouts(
            markers=markers,
            bos=bos,
            eos=eos,
            max_len=package.max_input_tokens,
            window=layout(config, routes.window, routes.overlap),
            evidence=layout(config, routes.evidence_window, routes.overlap),
        )
        self.pass_call = Call(None, routes.window, 1)

    def exact_batches(self, items: list[Any]) -> list[list[int]]:
        """One call per request: ``run`` batches its inputs as the released runtime does."""
        return [list(range(len(items)))] if items else []

    def run(self, items: list[Vela3Item]) -> list[Output]:
        return self.network.run(items)

    def run_approximate(self, items: list[Vela3Item]) -> list[Output]:
        """On a GPU, batches that merge requests share forwards between their equal calls (vela3's rule)."""
        return self.network.run(items, merged=self.engine_model.device.type != "cpu")

    # -- planning ------------------------------------------------------------

    def plan(
        self, state: Any, questions: dict[str, Any], scan: int | None = None
    ) -> PicoPlan:
        refuse_blank(state)
        request = self.reader.read(state, questions)
        plan = PicoPlan(
            question_ids=request.question_ids,
            items=[],
            errors=request.errors,
            input_tokens=0,
            request=request,
        )
        items: list[Vela3Item] = []
        if request.one_pass and request.state is not None and request.asked:
            plan.one_pass = self._one_pass(request)
        if plan.one_pass is not None:
            items.append(Vela3Item(plan.one_pass.sequence, self.pass_call))
        else:
            budget = max(scan or self.scan_tokens or 0, self.routes.window)
            plan.fallback = fallback.plan(
                request, self.tokens, self.layouts, budget, items, plan.errors
            )
        plan.items = items
        plan.input_tokens = sum(len(item.ids) for item in items)
        return plan

    def _one_pass(self, request: Read) -> sequence.OnePass | None:
        """The request in one sequence, or None when a part would have to be cut."""
        assert request.state is not None
        window = self.pass_layout.max_len
        parts = read_parts(
            PartsPlan(
                request.state,
                request.question_ids,
                [a.question for a in request.asked],
            ),
            self.tokens,
            window + 1,
        )
        if any(not part.complete for part in parts):
            return None
        return sequence.build(request.asked, parts, self.tokens, self.pass_layout)

    # -- answers -------------------------------------------------------------

    def finish_surface(
        self, plan: SurfacePlan[Vela3Item], results: Any
    ) -> dict[str, Any]:
        state: PicoPlan = plan.state
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
        readout = (
            self._one_pass_readout(state.one_pass, results[0])
            if state.one_pass is not None
            else self._fallback_readout(state, plan.items, results)
        )
        complete = (
            {a.question.id for a in request.asked if a.question.require_full_input}
            if state.one_pass is not None
            else (state.fallback.complete if state.fallback else set())
        )
        for question_id in request.question_ids:
            if question_id in state.errors:
                answers[question_id] = state.errors[question_id]
                continue
            asked = by_id[question_id]
            logits, raw = readout[question_id]
            self.answerer.answer(asked, logits, raw, request.state, response)
            if question_id in complete:
                _complete(asked, response)
        ordered: dict[str, Any] = {
            "answers": answers,
            "usage": {"input_tokens": plan.input_tokens, "output_tokens": 0},
        }
        for key in ("spans", "sets", "thresholds"):
            if key in response:
                ordered[key] = response[key]
        return ordered

    @staticmethod
    def _one_pass_readout(
        one: sequence.OnePass, output: Output
    ) -> dict[str, tuple[Any, RawRow | None]]:
        """Each question's option logits, and each span group's words x labels, from the one forward."""
        option_logits, block = output
        readout: dict[str, tuple[Any, RawRow | None]] = {}
        for placed, logits in zip(one.sequence.questions, option_logits, strict=True):
            readout[placed.question.id] = (np.asarray(logits, np.float64), None)
        for group in one.groups:
            assert block is not None
            raw = RawRow(tokens={group.role: group.target_tokens}, windows=0)
            raw.span = RawSpan(
                group.asked.question.names,
                np.asarray(group.offsets),
                block[
                    group.words[0] : group.words[1], group.labels[0] : group.labels[1]
                ].astype(np.float64),
            )
            readout[group.asked.question.id] = (None, raw)
        return readout

    @staticmethod
    def _fallback_readout(
        state: PicoPlan, items: Any, results: Any
    ) -> dict[str, tuple[Any, RawRow | None]]:
        """Each question's option logits or raw row, from the question-by-question forwards."""
        planned = state.fallback
        assert planned is not None
        readout: dict[str, tuple[Any, RawRow | None]] = {}
        for question_id, index in planned.decisions.items():
            readout[question_id] = (
                np.asarray(results[index][0][0], np.float64),
                None,
            )
        rows: dict[int, RawRow] = {}
        for question_id, placed in planned.rows.items():
            raw = rows.get(id(placed))
            if raw is None:
                raw = rows[id(placed)] = placed.layout.combine(
                    placed.row,
                    [items[index].sequence for index in placed.items],
                    [results[index] for index in placed.items],
                )
            logits = raw.logits.get(question_id)
            readout[question_id] = (
                None if logits is None else np.asarray(logits, np.float64),
                raw,
            )
        return readout


def _complete(asked: Asked, response: dict[str, Any]) -> None:
    """Mark an answered question that read its whole input (``require_full_input``)."""
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
