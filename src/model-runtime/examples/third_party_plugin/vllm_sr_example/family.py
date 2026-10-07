"""A keyword model family: the smallest complete family plugin.

A package is a directory with ``example_model.json``::

    {"format": "vllm-sr-example/1", "labels": ["billing", "shipping"],
     "keywords": {"billing": ["refund", "invoice"], "shipping": ["parcel"]}}

The family renders text into keyword token IDs, the ``example_counts``
engine turns them into per-token label vectors, and the family's readouts
pool those vectors into a label distribution (``/v1/classify``), a
normalised embedding (``/v1/embeddings``) and a dot-product relevance
(``/v1/rerank``). Package parsing, rendering and readout stay in the family;
the engine never reads the package.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any

import torch
from vllm_srun.errors import PackageError
from vllm_srun.plugins.base import (
    DEADLINE,
    BackboneSpec,
    DtypePolicy,
    EmbeddingInfo,
    EncoderBatch,
    EngineModel,
    HeadInfo,
    LoadedModel,
    ModelFamily,
    ModelInfo,
    ModelSpec,
    PackageRef,
    RerankInfo,
    SurfacePlan,
    SurfaceRequest,
    VerifiedPackage,
)

PACKAGE_FILE = "example_model.json"
FORMAT = "vllm-sr-example/1"
MAX_INPUT_TOKENS = 512
MIN_LABELS = 2
WORD = re.compile(r"\w+")


@dataclass(frozen=True)
class Item:
    """One forward row: keyword token IDs and what its readout needs."""

    ids: list[int]
    kind: str
    cache_key: str


def vocabulary(config: dict[str, Any]) -> tuple[dict[str, int], list[int]]:
    """Keyword token IDs (0 is any other word) and the label index of each ID."""
    words: dict[str, int] = {}
    label_of: list[int] = [-1]
    for label_index, label in enumerate(config["labels"]):
        for word in config["keywords"].get(label, []):
            if word.lower() not in words:
                words[word.lower()] = len(label_of)
                label_of.append(label_index)
    return words, label_of


def _key(*parts: Any) -> str:
    return hashlib.sha256(json.dumps(parts, separators=(",", ":")).encode()).hexdigest()


class KeywordFamily(ModelFamily):
    name = "example_keywords"
    surfaces = frozenset({"classify", "embeddings", "rerank"})

    @classmethod
    def descriptor(cls) -> dict[str, Any]:
        return {"surfaces": sorted(cls.surfaces), "formats": [FORMAT]}

    def detect(self, package: PackageRef) -> bool:
        return (package.root / PACKAGE_FILE).is_file()

    def verify(self, package: PackageRef) -> VerifiedPackage:
        raw = (package.root / PACKAGE_FILE).read_bytes()
        config = json.loads(raw)
        labels = config.get("labels")
        keywords = config.get("keywords")
        if (
            config.get("format") != FORMAT
            or not isinstance(labels, list)
            or len(labels) < MIN_LABELS
        ):
            raise PackageError(f"{PACKAGE_FILE} is not a {FORMAT} package")
        if not isinstance(keywords, dict) or set(keywords) - set(labels):
            raise PackageError("keywords must map known labels to word lists")
        digest = hashlib.sha256(raw).hexdigest()
        return VerifiedPackage(
            ref=package,
            family=self.name,
            model_name=package.root.name,
            manifest=config,
            manifest_sha256=digest,
            model_sha256=digest,
            max_input_tokens=MAX_INPUT_TOKENS,
            licence="apache-2.0",
            loaded_parameters=0,
        )

    def describe(self, package: VerifiedPackage) -> ModelSpec:
        _, label_of = vocabulary(package.manifest)
        backbone = BackboneSpec(
            model_type="example_counts",
            config={"labels": len(package.manifest["labels"]), "label_of": label_of},
            weight_files=(),
        )
        return ModelSpec(
            name=package.model_name,
            backbone=backbone,
            dtype=DtypePolicy(weights="float32", autocast=None, bf16_resident=False),
            max_input_tokens=package.max_input_tokens,
            encoder=True,
        )

    def load(
        self, package: VerifiedPackage, spec: ModelSpec, engine_model: EngineModel
    ) -> LoadedModel:
        words, _ = vocabulary(package.manifest)
        labels = tuple(package.manifest["labels"])
        info = ModelInfo(
            id=package.model_name,
            family=self.name,
            repo=package.ref.repo_id,
            revision=package.ref.revision,
            model_sha256=package.model_sha256,
            manifest_sha256=package.manifest_sha256,
            surfaces=tuple(sorted(self.surfaces)),
            question_types=(),
            limits={"max_input_tokens": package.max_input_tokens, "max_inputs": 64},
            licence=package.licence,
            parameters=0,
            dtype="fp32",
            heads=(HeadInfo(name="keywords", kind="sequence", labels=labels),),
            embedding=EmbeddingInfo(
                dimensions=(len(labels),), layers=(1,), pooling="sum"
            ),
            rerank=RerankInfo(exits=((1, len(labels)),), default=(1, len(labels))),
        )
        return KeywordModel(info, engine_model, words, labels)


class KeywordModel(LoadedModel):
    fuse_bundled_jobs = True

    def __init__(
        self,
        info: ModelInfo,
        engine_model: EngineModel,
        vocabulary: dict[str, int],
        labels: tuple[str, ...],
    ):
        self.info = info
        self.engine_model = engine_model
        self.vocabulary = vocabulary
        self.labels = labels
        self.forwards = 0

    def render(self, text: Any) -> list[int]:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("every input must be nonempty text")
        ids = [self.vocabulary.get(word.lower(), 0) for word in WORD.findall(text)]
        if len(ids) > MAX_INPUT_TOKENS:
            raise ValueError(f"an input exceeds {MAX_INPUT_TOKENS} tokens")
        return ids or [0]

    def plan_surface(self, surface: str, request: SurfaceRequest) -> SurfacePlan:
        body = request.body
        if surface == "rerank":
            texts, kind = [body["query"], *body["documents"]], "embed"
        elif surface in ("classify", "embeddings"):
            texts = (
                body["input"] if isinstance(body["input"], list) else [body["input"]]
            )
            kind = "classify" if surface == "classify" else "embed"
        else:
            return super().plan_surface(surface, request)
        items = []
        for text in texts:
            ids = self.render(text)
            items.append(Item(ids, kind, _key(kind, ids)))
        return SurfacePlan(
            surface, items, sum(len(item.ids) for item in items), state=body
        )

    def run(self, items: list[Item]) -> list[list[float]]:
        self.forwards += 1
        width = max(len(item.ids) for item in items)
        input_ids = torch.zeros((len(items), width), dtype=torch.long)
        mask = torch.zeros_like(input_ids)
        for row, item in enumerate(items):
            input_ids[row, : len(item.ids)] = torch.tensor(item.ids)
            mask[row, : len(item.ids)] = 1
        hidden = self.engine_model.encode(
            EncoderBatch(input_ids, mask, layers=(1,))
        ).hidden[1]
        counts = (hidden * mask.unsqueeze(-1)).sum(dim=1)
        return counts.tolist()

    def finish_surface(self, plan: SurfacePlan, results: Any) -> dict[str, Any]:
        if results is DEADLINE:
            results = [DEADLINE] * len(plan.items)
        if plan.surface == "classify":
            return {
                "head": "keywords",
                "kind": "sequence",
                "labels": list(self.labels),
                "results": [
                    _classify(index, counts, self.labels)
                    for index, counts in enumerate(results)
                ],
            }
        if plan.surface == "embeddings":
            data = []
            for index, counts in enumerate(results):
                if counts is DEADLINE:
                    data.append(
                        {
                            "object": "embedding",
                            "index": index,
                            "error": "deadline_exceeded",
                        }
                    )
                else:
                    data.append(
                        {
                            "object": "embedding",
                            "index": index,
                            "embedding": _normalised(counts),
                        }
                    )
            tokens = plan.input_tokens
            return {
                "object": "list",
                "data": data,
                "usage": {"prompt_tokens": tokens, "total_tokens": tokens},
                "meta": {
                    "representation": {
                        "model_sha256": self.info.model_sha256,
                        "layer": 1,
                        "dimension": len(self.labels),
                        "normalized": True,
                    }
                },
            }
        query, documents = results[0], results[1:]
        ranked = []
        for index, document in enumerate(documents):
            if query is DEADLINE or document is DEADLINE:
                ranked.append({"index": index, "error": "deadline_exceeded"})
                continue
            logit = float(sum(a * b for a, b in zip(query, document, strict=True)))
            ranked.append(
                {
                    "index": index,
                    "logit": logit,
                    "relevance_score": 1 / (1 + math.exp(-logit)),
                }
            )
        ranked.sort(key=lambda result: -result.get("logit", -math.inf))
        top_n = plan.state.get("top_n")
        return {"results": ranked[:top_n] if top_n else ranked}


def _classify(index: int, counts: Any, labels: tuple[str, ...]) -> dict[str, Any]:
    if counts is DEADLINE:
        return {"index": index, "error": "deadline_exceeded"}
    peak = max(counts)
    weights = [math.exp(value - peak) for value in counts]
    total = sum(weights)
    probabilities = [weight / total for weight in weights]
    best = max(range(len(labels)), key=probabilities.__getitem__)
    return {"index": index, "label": labels[best], "probabilities": probabilities}


def _normalised(counts: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in counts)) or 1.0
    return [value / norm for value in counts]
