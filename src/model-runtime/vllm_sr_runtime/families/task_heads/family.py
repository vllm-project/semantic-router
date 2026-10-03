"""Encoder task heads (Phase 3): Vela 1.0 and compatible HF ModernBERT task models.

A package carries one head over the native ModernBERT backbone: ``sequence``
(softmax distribution), ``scores`` (independent sigmoid per label, packaged
operating point), ``token`` (BIO spans) or ``grounded`` (answer spans against
a context). The family serves ``/v1/classify`` with reject, truncate and
window overflow as the legacy router bindings applied them.

Every forward is shared: a micro-batch's items are deduplicated by token IDs
(across inputs, windows, heads and bundled tasks), the distinct sequences run
as one packed forward (no padding) with the union of the exits the heads
read, and each head reads its rows in one batched call. Items carry content
keys, so repeated inputs are answered from the runtime's result cache.
"""

from __future__ import annotations

from itertools import chain
from typing import Any

import torch

from ...errors import INVALID_INPUT, MAX_LENGTH_EXCEEDED, PackageError
from ...heads.grounded import GroundedHead, GroundingPolicy, PairEnvelope
from ...heads.scores import OperatingPoint, ScoresHead
from ...heads.sequence import SequenceHead
from ...heads.task import ClassifierHead, HeadOptions, Item, Rows, TaskHead
from ...heads.token import TokenHead
from ...plugins.base import (
    DEADLINE,
    BackboneSpec,
    DtypePolicy,
    EncoderBatch,
    EngineModel,
    HeadInfo,
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
from ...registry import builtin
from ...registry.artifacts import named_files, safetensors_elements
from ...registry.resolve import fetch
from ...text.windows import Envelope, InputTooLongError
from . import package as pkg

OVERFLOW = ("reject", "truncate", "window")
HEAD_NAME = "default"
MAX_INPUTS = 2048
FORWARD_TOKEN_BUDGET = 65536
EXACT_DTYPE = DtypePolicy(
    weights="float32", autocast=None, head="float32", bf16_resident=False
)
GOLDEN_TEXTS = (
    "Write a Python function that merges two sorted lists.",
    "Meine Telefonnummer ist 030 1234567 und ich wohne in Berlin.",
)
GOLDEN_GROUNDED = {
    "context": "The Eiffel Tower is in Paris and was completed in 1889.",
    "question": "When was the Eiffel Tower completed?",
    "answer": "It was completed in 1889 in Rome.",
}


def classify_inputs(value: Any) -> list[Any]:
    inputs = [value] if isinstance(value, str | dict) else value
    if not isinstance(inputs, list) or not inputs:
        raise ValueError("input is a string, an object or a non-empty list of them")
    if len(inputs) > MAX_INPUTS:
        raise ValueError(f"at most {MAX_INPUTS} inputs per request")
    return inputs


class TaskHeadsModel(LoadedModel):
    """A task checkpoint's heads over one native encoder."""

    fuse_bundled_jobs = True

    def __init__(
        self,
        info: ModelInfo,
        engine_model: EngineModel,
        heads: dict[str, TaskHead],
        limit: int,
        defaults: dict[str, Any],
    ):
        self.info = info
        self.engine_model = engine_model
        self.heads = heads
        self.primary = next(iter(heads))
        self.limit = limit
        self.defaults = defaults

    def forward_token_budget(self) -> int | None:
        return max(FORWARD_TOKEN_BUDGET, self.limit)

    # -- planning ---------------------------------------------------------

    def options(self, head: TaskHead, raw: dict[str, Any]) -> HeadOptions:
        """The request's options for ``head``, validated against the model's limits."""
        card = head.describe()
        overflow = raw.get("overflow") or card["overflow"]
        if overflow not in OVERFLOW:
            raise ValueError(f"overflow must be one of {', '.join(OVERFLOW)}")
        if overflow == "window" and card.get("window") is None and "window" not in raw:
            raise ValueError("overflow window needs options.window for this head")
        if overflow == "window" and isinstance(head, GroundedHead):
            raise ValueError("grounded inputs are not read in windows")
        default_tokens = self.defaults.get("max_tokens", self.limit)
        max_tokens = raw.get("max_tokens", default_tokens)
        if (
            not isinstance(max_tokens, int)
            or isinstance(max_tokens, bool)
            or max_tokens < 1
        ):
            raise ValueError("max_tokens must be a positive integer")
        if max_tokens > self.limit:
            raise ValueError(f"max_tokens exceeds the model's limit of {self.limit}")
        window = card.get("window")
        if "window" in raw:
            spec = raw["window"]
            if not isinstance(spec, dict) or not isinstance(spec.get("tokens"), int):
                raise ValueError("window is {tokens, overlap}")
            window = (spec["tokens"], int(spec.get("overlap", 0)))
        if overflow == "window" and window is not None and window[0] > self.limit:
            raise ValueError(f"window tokens exceed the model's limit of {self.limit}")
        threshold = raw.get("threshold")
        if threshold is not None and not (
            isinstance(threshold, int | float) and 0.0 <= threshold <= 1.0
        ):
            raise ValueError("threshold must lie in [0, 1]")
        return HeadOptions(
            overflow=overflow,
            max_tokens=max_tokens,
            window=window if overflow == "window" else None,
            threshold=None if threshold is None else float(threshold),
            return_tokens=bool(raw.get("return_tokens", False)),
        )

    def plan_surface(self, surface: str, request: SurfaceRequest) -> SurfacePlan:
        if surface != "classify":
            raise UnsupportedSurfaceError(surface, self.info.id)
        body = request.body
        name = body.get("head") or self.primary
        head = self.heads.get(name)
        if head is None:
            raise ValueError(
                f"unknown head {name!r}; this model has {sorted(self.heads)}"
            )
        options = self.options(head, request.options)
        if options.overflow == "window":
            # A window that cannot hold the envelope and one content token is a request error.
            size, overlap = options.window
            if size > options.max_tokens:
                raise ValueError("window tokens exceed max_tokens")
            if overlap < 0:
                raise ValueError("window overlap must not be negative")
        entries: list[tuple[int, Any]] = []
        items: list[Item] = []
        input_tokens = 0
        for index, value in enumerate(classify_inputs(body.get("input"))):
            try:
                prepared = head.prepare(value, options, self.info.model_sha256)
            except InputTooLongError as exc:
                entries.append((index, MAX_LENGTH_EXCEEDED))
                input_tokens += exc.tokens
                continue
            except ValueError:
                entries.append((index, INVALID_INPUT))
                continue
            span = (len(items), len(items) + len(prepared.items))
            entries.append((index, (prepared, span)))
            items.extend(prepared.items)
            input_tokens += prepared.usage["tokens"]
        return SurfacePlan("classify", items, input_tokens, (head, options, entries))

    # -- execution --------------------------------------------------------

    def run(self, items: list[Item], shared_prefix: int = 0) -> list[Any]:
        """One packed forward over the distinct sequences of ``items``; each head reads its rows."""
        index: dict[tuple[int, ...], int] = {}
        for item in items:
            index.setdefault(item.ids, len(index))
        sequences = list(index)
        lengths = [len(ids) for ids in sequences]
        starts, offset = [], 0
        for length in lengths:
            starts.append(offset)
            offset += length
        layers = tuple(sorted({item.layer for item in items}))
        output = self.engine_model.encode(
            EncoderBatch(
                torch.tensor(list(chain.from_iterable(sequences)), dtype=torch.long),
                None,
                layers=layers,
                lengths=lengths,
            )
        )
        rows = Rows(output.hidden, starts, lengths)
        results: list[Any] = [None] * len(items)
        by_head: dict[str, list[int]] = {}
        for position, item in enumerate(items):
            by_head.setdefault(item.head, []).append(position)
        with torch.inference_mode():
            for name, positions in by_head.items():
                wanted = list(dict.fromkeys(index[items[p].ids] for p in positions))
                values = dict(
                    zip(wanted, self.heads[name].readout(rows, wanted), strict=True)
                )
                for position in positions:
                    results[position] = values[index[items[position].ids]]
        return results

    # -- answers ----------------------------------------------------------

    def finish_surface(self, plan: SurfacePlan, results: Any) -> dict[str, Any]:
        head, options, entries = plan.state
        out: list[dict[str, Any]] = []
        for index, entry in entries:
            if isinstance(entry, str):
                out.append({"index": index, "error": entry})
                continue
            prepared, (start, end) = entry
            values = DEADLINE if results is DEADLINE else results[start:end]
            if values is DEADLINE or any(value is DEADLINE for value in values):
                out.append({"index": index, "error": "deadline_exceeded"})
                continue
            result = head.result(prepared, values)
            if options.threshold is not None and isinstance(head, TokenHead):
                result["spans"] = [
                    span
                    for span in result["spans"]
                    if span["probability"] >= options.threshold
                ]
            out.append({"index": index, **result})
        return {
            "head": head.name,
            "kind": head.kind,
            "labels": list(head.labels),
            "results": out,
            "usage": {"input_tokens": plan.input_tokens, "output_tokens": 0},
        }


class TaskHeadsFamily(ModelFamily):
    name = "task_heads"
    surfaces = frozenset({"classify"})

    @classmethod
    def descriptor(cls) -> dict[str, Any]:
        return {
            "surfaces": sorted(cls.surfaces),
            "formats": ["hf-modernbert"],
            "heads": ["sequence", "scores", "token", "grounded"],
            "overflow": list(OVERFLOW),
        }

    def detect(self, package: PackageRef) -> bool:
        return pkg.detect(package.root)

    def fetch(self, package: PackageRef) -> PackageRef:
        if builtin.lookup(package.repo_id or "") is not None:
            return package
        return fetch(
            package,
            [*pkg.REQUIRED, pkg.OPERATING_POINT],
            cache_dir=self.options.cache_dir,
            offline=self.options.offline,
        )

    def verify(self, package: PackageRef) -> VerifiedPackage:
        task = pkg.read(package.root)
        files = named_files(package.root, task.files)
        known = None
        if package.repo_id is not None:
            known = builtin.lookup(package.repo_id)
            if known is not None and known.revision != package.revision:
                known = None
        identity = pkg.identity(files)
        if known is not None:
            if dict(known.files) != files:
                changed = sorted(set(known.files.items()) ^ set(files.items()))
                raise PackageError(
                    f"{package.repo_id}@{package.revision} files differ from the pinned digests: "
                    f"{sorted({name for name, _ in changed})}"
                )
            if known.model_sha256 != identity:
                raise PackageError(
                    f"{package.repo_id} identity differs from the pinned identity"
                )
        parameters = safetensors_elements(package.root / pkg.WEIGHTS)
        if known is not None and known.loaded_parameters != parameters:
            raise PackageError(
                f"{package.repo_id} carries {parameters:,} parameters, not {known.loaded_parameters:,}"
            )
        name = (package.repo_id or package.root.name).rsplit("/", 1)[-1]
        return VerifiedPackage(
            ref=package,
            family=self.name,
            model_name=name,
            manifest={},
            manifest_sha256="",
            model_sha256=identity,
            max_input_tokens=self._limit(task),
            licence="apache-2.0" if known is not None else None,
            loaded_parameters=parameters,
            details={
                "package": task,
                "files": files,
                "verification": "builtin" if known is not None else "local",
            },
        )

    @staticmethod
    def _limit(task: pkg.TaskPackage) -> int:
        limit = task.max_positions
        if task.kind == "grounded":
            limit = min(limit, int(task.operating_point["max_input_tokens"]))
        return limit

    def describe(self, package: VerifiedPackage) -> ModelSpec:
        task: pkg.TaskPackage = package.details["package"]
        return ModelSpec(
            name=package.model_name,
            backbone=BackboneSpec(
                model_type=pkg.MODEL_TYPE,
                config=task.config,
                weight_files=(task.root / pkg.WEIGHTS,),
                weight_prefix=pkg.BACKBONE_PREFIX,
            ),
            dtype=EXACT_DTYPE,
            max_input_tokens=package.max_input_tokens,
            encoder=True,
        )

    def load(
        self, package: VerifiedPackage, spec: ModelSpec, engine_model: EngineModel
    ) -> LoadedModel:
        from tokenizers import Tokenizer

        task: pkg.TaskPackage = package.details["package"]
        tokenizer = Tokenizer.from_file(str(task.root / "tokenizer.json"))
        tokenizer.no_truncation()
        tokenizer.no_padding()
        classifier = ClassifierHead.load(
            [task.root / pkg.WEIGHTS], task.config, len(task.labels)
        ).to(engine_model.device)
        parameters = engine_model.parameter_count() + sum(
            parameter.numel() for parameter in classifier.parameters()
        )
        if parameters != package.loaded_parameters:
            raise PackageError(
                f"loaded {parameters:,} parameters; the checkpoint holds {package.loaded_parameters:,}"
            )
        layer = int(task.config["num_hidden_layers"])
        head, defaults = self._head(task, tokenizer, classifier, layer)
        heads = {head.name: head}
        info = ModelInfo(
            id=package.model_name,
            family=self.name,
            repo=package.ref.repo_id,
            revision=package.ref.revision,
            model_sha256=package.model_sha256,
            manifest_sha256=package.manifest_sha256,
            surfaces=tuple(sorted(self.surfaces)),
            question_types=(),
            limits={
                "max_input_tokens": package.max_input_tokens,
                "max_inputs": MAX_INPUTS,
            },
            licence=package.licence,
            parameters=parameters,
            dtype="fp32",
            heads=tuple(HeadInfo(**h.describe()) for h in heads.values()),
        )
        return TaskHeadsModel(
            info, engine_model, heads, package.max_input_tokens, defaults
        )

    @staticmethod
    def _head(
        task: pkg.TaskPackage, tokenizer: Any, classifier: ClassifierHead, layer: int
    ) -> tuple[TaskHead, dict[str, Any]]:
        args = (HEAD_NAME, task.labels, layer, tokenizer)
        if task.kind == "grounded":
            policy = GroundingPolicy.parse(task.operating_point, task.labels)
            return (
                GroundedHead(*args, PairEnvelope.of(tokenizer), classifier, policy),
                {},
            )
        envelope = Envelope.of(tokenizer)
        if task.kind == "token":
            return TokenHead(*args, envelope, classifier), {}
        pooling = task.config.get("classifier_pooling", "cls")
        if task.kind == "scores":
            point = None
            if task.operating_point is not None:
                point = OperatingPoint.parse(task.operating_point, task.labels)
            head = ScoresHead(
                *args, envelope, classifier, pooling, operating_point=point
            )
            defaults = (
                {}
                if point is None or point.max_tokens is None
                else {"max_tokens": point.max_tokens}
            )
            return head, defaults
        return SequenceHead(*args, envelope, classifier, pooling), {}

    def golden(self, package: VerifiedPackage) -> list[dict[str, Any]]:
        task: pkg.TaskPackage = package.details["package"]
        known = builtin.by_identity(package.model_sha256)
        expected = dict(known.golden_answers) if known else {}
        if task.kind == "grounded":
            body = {"input": [GOLDEN_GROUNDED]}
        else:
            body = {"input": list(GOLDEN_TEXTS)}
        return [{"surface": "classify", "body": body, "expected": expected}]
