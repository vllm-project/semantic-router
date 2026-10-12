"""Encoder task heads (Phase 3): Vela 1.0 and compatible HF task models.

A classifier carries one head over the native ModernBERT backbone:
``sequence`` (softmax distribution), ``scores`` (independent sigmoid per
label, packaged operating point), ``token`` (BIO spans) or ``grounded``
(answer spans against a context), served on ``/v1/classify`` with reject,
truncate and window overflow as the legacy router bindings applied them. An
embedder (Vela Embedding, Qwen3-Embedding) serves ``/v1/embeddings`` from
pooled layer exits and a reranker (Vela Reranker) ``/v1/rerank`` from its
pair-scorer exits; on the onnxruntime engine both run the package's exit
graphs instead of the native backbone.

Every forward is shared: a micro-batch's items are deduplicated by token IDs
(across inputs, windows, heads and bundled tasks), the distinct sequences run
as one packed forward (no padding) with the union of the exits the heads
read, and each head reads its rows in one batched call. Items carry content
keys, so repeated inputs are answered from the runtime's result cache.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from itertools import chain
from typing import Any, cast

import torch

from ...accel import onednn
from ...accel.kernels import CONTIGUOUS
from ...errors import INVALID_INPUT, PackageError
from ...heads.grounded import GroundedHead, GroundingPolicy, PairEnvelope
from ...heads.pooled import EmbeddingSurface, PooledLayout
from ...heads.relevance import LOGITS, RelevanceHead, RelevanceLayout, RerankSurface
from ...heads.scores import OperatingPoint, ScoresHead
from ...heads.sequence import SequenceHead
from ...heads.task import (
    ClassifierHead,
    Head,
    HeadOptions,
    Item,
    Rows,
    TaskHead,
    identical,
)
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
# A graph batch (no packing) grows while its padding stays within this share of its real tokens.
PADDING_ALLOWANCE = 0.25
EXACT_DTYPE = DtypePolicy(
    weights="float32", autocast=None, head="float32", bf16_resident=False
)
# Task heads' exact reference is the legacy path's recorded agreement (classify,
# embeddings and rerank alike), so on CPUs they run the batch-invariant variants:
# oneDNN's packed FP32 linear (x86) and GeGLU on contiguous rows.
TASK_KERNELS = {"linear": onednn.PACKED, "geglu": CONTIGUOUS}
# Token counts of the rows that probe a loaded model's batch invariance.
INVARIANCE_PROBE = (3, 9, 9, 17, 40, 130)
GOLDEN_TEXTS = (
    "Write a Python function that merges two sorted lists.",
    "Meine Telefonnummer ist 030 1234567 und ich wohne in Berlin.",
)
GOLDEN_GROUNDED = {
    "context": "The Eiffel Tower is in Paris and was completed in 1889.",
    "question": "When was the Eiffel Tower completed?",
    "answer": "It was completed in 1889 in Rome.",
}
GOLDEN_RERANK = {
    "query": "How do I reset my password?",
    "documents": [
        "Open Settings, then Security, and choose Reset password.",
        "Our offices are closed on public holidays.",
    ],
}


def length_buckets(lengths: Sequence[int], budget: int) -> list[list[int]]:
    """Indices grouped by length for padded batches: padding stays within the allowance, tokens within ``budget``."""
    order = sorted(range(len(lengths)), key=lambda index: (lengths[index], index))
    buckets: list[list[int]] = []
    real = 0
    for index in order:
        length = lengths[index]
        if buckets:
            count = len(buckets[-1]) + 1
            padded = length * count
            if padded <= budget and padded <= (real + length) * (1 + PADDING_ALLOWANCE):
                buckets[-1].append(index)
                real += length
                continue
        buckets.append([index])
        real = length
    return buckets


@contextmanager
def single_threaded() -> Iterator[None]:
    """Torch on one thread around graph runs, restored after.

    A graph engine's own thread pool spins on the cores between runs; a
    parallel torch op there (a pooled head over 64 or more tokens) makes the
    next run share them: 28 ms became 39 ms at 64 tokens on 16 cores.
    """
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(threads)


def batch_invariant(model: TaskHeadsModel, vocab: int) -> bool:
    """Whether every head answers probe rows the same alone and inside one mixed batch.

    Kernels are invariant by construction only on some PyTorch builds (a
    narrow classifier through oneDNN 3.11 is not at small batch sizes), so
    the loaded model is probed on its own host before ``exact`` may batch.
    """
    generator = torch.Generator().manual_seed(0)
    items = [
        Item(
            tuple(torch.randint(5, vocab, (length,), generator=generator).tolist()),
            name,
            head.layer,
        )
        for name, head in model.heads.items()
        for length in INVARIANCE_PROBE
    ]
    alone = [model.run([item])[0] for item in items]
    together = model.run(items[::-1])[::-1]
    return all(map(identical, alone, together))


def graph_name(head: Head) -> str:
    """The ``ModelSpec.graphs`` entry of an exit head: ``layer:<L>`` or ``layer:<L>/dim:<D>``."""
    if isinstance(head, RelevanceHead):
        return f"layer:{head.exit[0]}/dim:{head.exit[1]}"
    return f"layer:{head.layer}"


def classify_inputs(value: Any) -> list[Any]:
    inputs = [value] if isinstance(value, str | dict) else value
    if not isinstance(inputs, list) or not inputs:
        raise ValueError("input is a string, an object or a non-empty list of them")
    if len(inputs) > MAX_INPUTS:
        raise ValueError(f"at most {MAX_INPUTS} inputs per request")
    return inputs


class TaskHeadsModel(LoadedModel[Item, Any]):
    """A task checkpoint's heads over one shared forward.

    ``heads`` are every readout of the forward (classify heads, or one
    pooled / relevance head per served exit); ``planners`` serve the
    embeddings or rerank surface over them; ``normalize_exits`` is the
    package's representation contract for intermediate exits.
    """

    fuse_bundled_jobs = True

    def __init__(
        self,
        info: ModelInfo,
        engine_model: EngineModel,
        heads: Mapping[str, Head],
        limit: int,
        defaults: dict[str, Any],
        planners: dict[str, EmbeddingSurface | RerankSurface] | None = None,
        normalize_exits: bool = False,
    ):
        self.info = info
        self.engine_model = engine_model
        self.heads = heads
        self.primary = next(iter(heads))
        self.limit = limit
        self.defaults = defaults
        self.planners = planners or {}
        self.normalize_exits = normalize_exits
        self.packs_rows = engine_model.hidden_states

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

    def plan_surface(self, surface: str, request: SurfaceRequest) -> SurfacePlan[Item]:
        if surface in self.planners:
            return self.planners[surface].plan(
                request, self.info.model_sha256, self.limit
            )
        if surface != "classify" or "classify" not in self.info.surfaces:
            raise UnsupportedSurfaceError(surface, self.info.id)
        body = request.body
        name = body.get("head") or self.primary
        head = cast(TaskHead | None, self.heads.get(name))
        if head is None:
            raise ValueError(
                f"unknown head {name!r}; this model has {sorted(self.heads)}"
            )
        options = self.options(head, request.options)
        if options.overflow == "window":
            # A window that cannot hold the envelope and one content token is a request error.
            assert options.window is not None
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
                entries.append((index, exc.code))
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

    def shared_context(self, items: list[Item], token_budget: int | None) -> int:
        """Encoder heads read whole sequences, so no job shares a prefix (0: run exactly)."""
        return 0

    def run_approximate(self, items: list[Item]) -> list[Any]:
        """``run`` on the engine's reduced copy of the backbone, where it loaded one (``max_speed``)."""
        return self.run(items, reduced=True)

    def run(self, items: list[Item], reduced: bool = False) -> list[Any]:
        """One packed forward over the distinct sequences of ``items``; each head reads its rows."""
        if not self.engine_model.hidden_states:
            return self._run_graphs(items)
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
                normalize_exits=self.normalize_exits,
                lengths=lengths,
                reduced=reduced,
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

    def _run_graphs(self, items: list[Item]) -> list[Any]:
        """Each head's exit graph over its distinct sequences, in length buckets of padded rows."""
        with single_threaded():
            return self._graph_results(items)

    def _graph_results(self, items: list[Item]) -> list[Any]:
        results: list[Any] = [None] * len(items)
        by_head: dict[str, list[int]] = {}
        for position, item in enumerate(items):
            by_head.setdefault(item.head, []).append(position)
        for name, positions in by_head.items():
            head = self.heads[name]
            sequences = list(dict.fromkeys(items[p].ids for p in positions))
            lengths = [len(ids) for ids in sequences]
            values: dict[tuple[int, ...], Any] = {}
            for bucket in length_buckets(lengths, FORWARD_TOKEN_BUDGET):
                rows = self._graph_rows(head, [sequences[i] for i in bucket])
                for local, value in enumerate(head.readout(rows, range(len(bucket)))):
                    values[sequences[bucket[local]]] = value
            for position in positions:
                results[position] = values[items[position].ids]
        return results

    def _graph_rows(self, head: Head, sequences: list[tuple[int, ...]]) -> Rows:
        width = max(len(ids) for ids in sequences)
        input_ids = torch.zeros(len(sequences), width, dtype=torch.long)
        mask = torch.zeros(len(sequences), width, dtype=torch.long)
        for row, ids in enumerate(sequences):
            input_ids[row, : len(ids)] = torch.tensor(ids)
            mask[row, : len(ids)] = 1
        output = self.engine_model.encode(
            EncoderBatch(input_ids, mask, graph=graph_name(head))
        )
        lengths = [len(ids) for ids in sequences]
        if LOGITS in output.outputs:
            return Rows({}, [], lengths, {LOGITS: output.outputs[LOGITS]})
        padded = output.outputs["last_hidden_state"]
        hidden = torch.cat([padded[row, :length] for row, length in enumerate(lengths)])
        starts = [sum(lengths[:row]) for row in range(len(lengths))]
        return Rows({head.layer: hidden}, starts, lengths)

    # -- answers ----------------------------------------------------------

    def finish_surface(self, plan: SurfacePlan[Item], results: Any) -> dict[str, Any]:
        if plan.surface in self.planners:
            return self.planners[plan.surface].finish(plan, results)
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
    surfaces = frozenset({"classify", "embeddings", "rerank"})
    builtin_table = "vllm_srun.registry.tables.vela1"
    fixture_writer = "vllm_srun.testing.task_heads"

    @classmethod
    def descriptor(cls) -> dict[str, Any]:
        return {
            "surfaces": sorted(cls.surfaces),
            "formats": [
                "hf-modernbert",
                "sentence-transformers",
                "matryoshka-reranker",
            ],
            "backbones": [pkg.MODEL_TYPE, pkg.DECODER_TYPE],
            "heads": ["sequence", "scores", "token", "grounded", "pooled", "relevance"],
            "overflow": list(OVERFLOW),
        }

    def detect(self, package: PackageRef) -> bool:
        return pkg.detect(package.root)

    def fetch(self, package: PackageRef) -> PackageRef:
        if builtin.lookup(package.repo_id or "") is not None:
            return package
        return fetch(
            package,
            list(pkg.FETCH_PATTERNS),
            cache_dir=self.options.cache_dir,
            offline=self.options.offline,
        )

    def _option_exits(self, name: str) -> list[Any]:
        value = self.options.model_options.get(name)
        if value is None:
            return []
        values = value if isinstance(value, list) else [value]
        if name == "layers":
            if not all(isinstance(v, int) and not isinstance(v, bool) for v in values):
                raise PackageError("model option layers is a list of layer exits")
            return values
        exits = []
        for entry in values:
            if not isinstance(entry, dict) or set(entry) != {"layer", "dimension"}:
                raise PackageError(f"model option {name} is {{layer, dimension}}")
            exits.append((int(entry["layer"]), int(entry["dimension"])))
        return exits

    def verify(self, package: PackageRef) -> VerifiedPackage:
        """Check the package and return its identity.

        A missing file raises ``PackageError`` before the backbone is built.
        mmBERT-shaped ModernBERT classifiers, including Vela 1.0, load through
        this family. Verification returns before any backbone is constructed.
        """
        selection = self._option_exits("pair_scorer")
        task = pkg.read(package.root, selection[0] if selection else None)
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
            assert task.operating_point is not None
            limit = min(limit, int(task.operating_point["max_input_tokens"]))
        return limit

    def _graph_exits(self, task: pkg.TaskPackage, strict: bool = False) -> list[Any]:
        """The exits a graph engine serves: the package default plus the model options' extra exits.

        Exits the options name must have graphs; ``strict`` (loading on a graph
        engine) refuses a default exit without one too.
        """
        layout = task.layout
        if isinstance(layout, PooledLayout):
            requested = self._option_exits("layers")
            wanted = requested or [layout.layers[-1]]
        elif isinstance(layout, RelevanceLayout):
            requested = self._option_exits("pair_scorers")
            wanted = list(dict.fromkeys([layout.default, *requested]))
        else:
            return []
        missing = [exit for exit in wanted if exit not in layout.graphs]
        if missing and (strict or requested):
            raise PackageError(f"the package ships no exit graph for {missing}")
        return [exit for exit in wanted if exit in layout.graphs]

    def describe(self, package: VerifiedPackage) -> ModelSpec:
        task: pkg.TaskPackage = package.details["package"]
        graphs = {}
        for exit in self._graph_exits(task):
            if isinstance(task.layout, PooledLayout):
                graphs[f"layer:{exit}"] = task.layout.graphs[exit]
            else:
                relevance = cast(RelevanceLayout, task.layout)
                graphs[f"layer:{exit[0]}/dim:{exit[1]}"] = relevance.graphs[exit]
        return ModelSpec(
            name=package.model_name,
            backbone=BackboneSpec(
                model_type=task.model_type,
                config=task.config,
                weight_files=(task.root / pkg.WEIGHTS,),
                weight_prefix=task.weight_prefix,
            ),
            dtype=EXACT_DTYPE,
            max_input_tokens=package.max_input_tokens,
            graphs=graphs,
            encoder=True,
            kernel_variants=TASK_KERNELS,
        )

    def load(
        self, package: VerifiedPackage, spec: ModelSpec, engine_model: EngineModel
    ) -> TaskHeadsModel:
        from tokenizers import Tokenizer

        task: pkg.TaskPackage = package.details["package"]
        tokenizer = Tokenizer.from_file(str(task.root / "tokenizer.json"))
        tokenizer.no_truncation()
        tokenizer.no_padding()
        if task.layout is not None:
            return self._load_exits(package, task, engine_model, tokenizer)
        classifier = ClassifierHead.load(
            [task.root / pkg.WEIGHTS], task.config, len(task.labels)
        )
        parameters = engine_model.parameter_count() + sum(
            parameter.numel() for parameter in classifier.parameters()
        )
        classifier = engine_model.place(classifier)
        if parameters != package.loaded_parameters:
            raise PackageError(
                f"loaded {parameters:,} parameters; the checkpoint holds {package.loaded_parameters:,}"
            )
        layer = int(task.config["num_hidden_layers"])
        head, defaults = self._head(task, tokenizer, classifier, layer)
        heads: dict[str, Head] = {head.name: head}
        described = head.describe()
        if task.operating_point is not None:
            described["operating_point_sha256"] = package.details["files"][
                pkg.OPERATING_POINT
            ]
        info = self._info(
            package, ("classify",), parameters, heads=(HeadInfo(**described),)
        )
        model = TaskHeadsModel(
            info, engine_model, heads, package.max_input_tokens, defaults
        )
        model.batch_invariant = engine_model.batch_invariant and batch_invariant(
            model, int(task.config["vocab_size"])
        )
        return model

    def _info(
        self,
        package: VerifiedPackage,
        surfaces: tuple[str, ...],
        parameters: int,
        **descriptors: Any,
    ) -> ModelInfo:
        return ModelInfo(
            id=package.model_name,
            family=self.name,
            repo=package.ref.repo_id,
            revision=package.ref.revision,
            model_sha256=package.model_sha256,
            manifest_sha256=package.manifest_sha256,
            surfaces=surfaces,
            question_types=(),
            limits={
                "max_input_tokens": package.max_input_tokens,
                "max_inputs": MAX_INPUTS,
            },
            licence=package.licence,
            parameters=parameters,
            dtype="fp32",
            **descriptors,
        )

    def _load_exits(
        self,
        package: VerifiedPackage,
        task: pkg.TaskPackage,
        engine_model: EngineModel,
        tokenizer: Any,
    ) -> TaskHeadsModel:
        """An embedder or reranker: one head per exit the engine serves (every exit natively)."""
        parameters = engine_model.parameter_count()
        if engine_model.hidden_states and parameters != package.loaded_parameters:
            raise PackageError(
                f"loaded {parameters:,} parameters; the checkpoint holds {package.loaded_parameters:,}"
            )
        layout = task.layout
        if isinstance(layout, PooledLayout):
            layers = (
                layout.layers
                if engine_model.hidden_states
                else self._graph_exits(task, strict=True)
            )
            planner: EmbeddingSurface | RerankSurface = EmbeddingSurface(
                layout, tokenizer, layers
            )
            heads = {head.name: head for head in planner.heads.values()}
            info = self._info(
                package, ("embeddings",), parameters, embedding=planner.info
            )
            normalize_exits = layout.normalize_exits
        else:
            assert isinstance(layout, RelevanceLayout)
            native = engine_model.hidden_states
            exits = layout.exits if native else self._graph_exits(task, strict=True)
            scorers = layout.scorers(exits) if native else dict.fromkeys(exits)
            relevance = {
                exit: RelevanceHead(
                    exit,
                    (
                        None
                        if scorers[exit] is None
                        else engine_model.place(cast(torch.nn.Module, scorers[exit]))
                    ),
                )
                for exit in exits
            }
            planner = RerankSurface(layout, tokenizer, relevance)
            heads = {head.name: head for head in relevance.values()}
            info = self._info(package, ("rerank",), parameters, rerank=planner.info)
            normalize_exits = True
        surface = info.surfaces[0]
        model = TaskHeadsModel(
            info,
            engine_model,
            heads,
            package.max_input_tokens,
            {},
            planners={surface: planner},
            normalize_exits=normalize_exits,
        )
        model.batch_invariant = engine_model.batch_invariant and batch_invariant(
            model, int(task.config["vocab_size"])
        )
        return model

    @staticmethod
    def _head(
        task: pkg.TaskPackage, tokenizer: Any, classifier: ClassifierHead, layer: int
    ) -> tuple[TaskHead, dict[str, Any]]:
        args = (HEAD_NAME, task.labels, layer, tokenizer)
        if task.kind == "grounded":
            assert task.operating_point is not None
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
        surface = "classify"
        if task.kind == "grounded":
            body: dict[str, Any] = {"input": [GOLDEN_GROUNDED]}
        elif task.kind == "relevance":
            surface, body = "rerank", dict(GOLDEN_RERANK)
        else:
            body = {"input": list(GOLDEN_TEXTS)}
            surface = "embeddings" if task.kind == "pooled" else surface
        return [{"surface": surface, "body": body, "expected": expected}]
