"""Native adapter registry for the same-panel runner.

Each adapter names the collector module that wraps a model's own published or
packaged inference path, the argument template, the interpreter and the
recommended image. Placeholders: {model} {revision} {input} {output} {device}
{source} {base} {checkpoint} {calibration} {model_id}. Extra values come from
``--extra KEY=VALUE`` on the runner command line.

A new Decision 2.0 package registers a JSON spec with the same fields
(``--adapter-spec``); it must emit one JSONL row per prompt with ``id``,
``answers`` (question id -> native answer), ``latency_ms``,
``source_input_sha256`` and inline ``model_id``/``model_revision``, and must
count native over-budget inputs as invalid instead of truncating.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_IMAGE = "decision20-train-fast:host2"
KAI_LEX_ENV = "/data/dev2/tools/envs/kai-lex"
KAI_LEX_PYTHON = f"{KAI_LEX_ENV}/bin/python"
IO_ARGS = ("--input", "{input}", "--output", "{output}")


@dataclass(frozen=True)
class Adapter:
    name: str
    module: str
    args: tuple[str, ...]
    model_id: str | None
    python: str = "python3"
    image: str = DEFAULT_IMAGE
    batch_policy: str = "one prompt (all its questions) per native call"
    requires: tuple[str, ...] = ()
    extra_mounts: tuple[str, ...] = field(default_factory=tuple)

    def command(self, values: dict[str, str]) -> list[str]:
        missing = [key for key in self.requires if not values.get(key)]
        if missing:
            raise ValueError(f"adapter {self.name} needs --extra {', '.join(missing)}")
        argv = [self.python, "-m", self.module]
        for token in self.args:
            try:
                argv.append(token.format(**values))
            except KeyError as exc:
                raise ValueError(f"adapter {self.name}: no value for {exc}") from exc
        return argv

    def describe(self) -> dict[str, Any]:
        return asdict(self)


def _decision1(backend: str, model_id: str, policy: str) -> Adapter:
    return Adapter(
        name=f"decision1-{backend}",
        module="inference.run",
        args=(
            "--backend",
            backend,
            "--model-path",
            "{model}",
            "--model-revision",
            "{revision}",
            *IO_ARGS,
            "--device",
            "{device}",
            "--over-budget-invalid",
        ),
        model_id=model_id,
        batch_policy=policy,
    )


REGISTRY: dict[str, Adapter] = {
    adapter.name: adapter
    for adapter in (
        _decision1(
            "lux",
            "llm-semantic-router/Decision-1.0-Lux-9B",
            "published DecisionModel.decide, one prompt per call; 16,384-token limit, over-budget invalid",
        ),
        _decision1(
            "nox",
            "llm-semantic-router/Decision-1.0-Nox-4B",
            "published DecisionModel.decide, one prompt per call; over-budget invalid",
        ),
        _decision1(
            "sol",
            "llm-semantic-router/Decision-1.0-Sol-2B",
            "published DecisionModel.decide, one prompt per call; 16,384-token limit, over-budget invalid",
        ),
        _decision1(
            "eos",
            "llm-semantic-router/Decision-1.0-Eos-0.8B",
            "published Eos DecisionModel.decide, eight-question physical batch limit",
        ),
        Adapter(
            name="decider",
            module="inference.run",
            args=(
                "--backend",
                "decider",
                "--model-path",
                "{model}",
                "--model-revision",
                "{revision}",
                *IO_ARGS,
                "--device",
                "{device}",
            ),
            model_id=None,
            batch_policy="published decider system_one, eager (no CUDA graphs), one prompt per call",
        ),
        Adapter(
            name="kai",
            module="inference.kai_lex",
            args=(
                "--backend",
                "kai",
                "--model-path",
                "{model}",
                "--model-revision",
                "{revision}",
                *IO_ARGS,
                "--device",
                "{device}",
            ),
            model_id="llm-semantic-router/Decision-1.0-Kai-0.6B",
            python=KAI_LEX_PYTHON,
            batch_policy="published SystemOne default physical B8 scheduling; 1,024-token complete-input limit, overflow invalid",
            extra_mounts=(KAI_LEX_ENV,),
        ),
        Adapter(
            name="lex",
            module="inference.kai_lex",
            args=(
                "--backend",
                "lex",
                "--model-path",
                "{model}",
                "--model-revision",
                "{revision}",
                *IO_ARGS,
                "--device",
                "{device}",
            ),
            model_id="llm-semantic-router/Decision-1.0-Lex-0.6B",
            python=KAI_LEX_PYTHON,
            batch_policy="published SystemOne default physical B8 scheduling; 1,024-token complete-input limit, overflow invalid",
            extra_mounts=(KAI_LEX_ENV,),
        ),
        Adapter(
            name="bosun06",
            module="inference.bosun06",
            args=("--model-path", "{model}", "--base-path", "{base}", *IO_ARGS),
            model_id="Hanno-Labs/bosun-v3.1-0.6b",
            requires=("base",),
        ),
        Adapter(
            name="jpt",
            module="inference.jpt",
            args=(
                "--model-path",
                "{model}",
                "--source-path",
                "{source}",
                "--model-revision",
                "{revision}",
                *IO_ARGS,
            ),
            model_id="kirp/jpt-9b",
            requires=("source",),
        ),
        *(
            Adapter(
                name=f"jpt-{size}",
                module="inference.jpt",
                args=(
                    "--size",
                    size,
                    "--model-path",
                    "{model}",
                    "--source-path",
                    "{source}",
                    "--model-revision",
                    "{revision}",
                    *IO_ARGS,
                ),
                model_id=f"kirp/jpt-{size}",
                batch_policy="llm2jev HF backend label log-probs, one prompt per call",
                requires=("source",),
            )
            for size in ("0.8b", "4b")
        ),
        Adapter(
            name="kev-0.8b",
            module="inference.kev",
            args=(
                "--size",
                "0.8b",
                "--model-path",
                "{model}",
                "--source-path",
                "{source}",
                "--model-revision",
                "{revision}",
                *IO_ARGS,
                "--device",
                "{device}",
            ),
            model_id="jaredpalmer/kev-0.8b",
            batch_policy="Kev FP32 Checkpoint.load/DecisionModel.probs, one prompt per call; strict 8,192-token limits, overflow invalid",
            requires=("source",),
        ),
        Adapter(
            name="this-that-1.2",
            module="inference.this_that",
            args=(
                "--version",
                "1.2",
                "--model-path",
                "{model}",
                "--source-path",
                "{source}",
                "--model-revision",
                "{revision}",
                *IO_ARGS,
                "--device",
                "{device}",
            ),
            model_id="flock-io/this-that-model-1.2",
            batch_policy="TypedDecider.decide; Noul/Score are option projections; 1,536-token state limit, overflow invalid",
            requires=("source",),
        ),
        Adapter(
            name="jet-v6.2",
            module="v2.eval.native_jet",
            args=(
                "--model-path",
                "{model}",
                "--model-revision",
                "{revision}",
                *IO_ARGS,
                "--device",
                "{device}",
            ),
            model_id="michaljach/jet",
            batch_policy="bundled Jet().decide, questions scored separately; 16,384-token limit, rejection invalid",
        ),
        Adapter(
            name="nimble-v2",
            module="v2.eval.native_nimble",
            args=(
                "--model-path",
                "{model}",
                "--model-revision",
                "{revision}",
                *IO_ARGS,
                "--device",
                "{device}",
            ),
            model_id="bespokelabs/Bespoke-Nimble-9B-v2",
            batch_policy=(
                "bundled ParallelScorer.score, one native schema per item, T=2.179;"
                " 8,192-token limit, rejected item invalid"
            ),
        ),
        Adapter(
            name="eikos-27b",
            module="inference.eikos",
            args=(
                "--size",
                "27b",
                "--model-path",
                "{model}",
                "--model-revision",
                "{revision}",
                *IO_ARGS,
                "--device",
                "{device}",
            ),
            model_id="caiovicentino1/Eikos-27B",
            batch_policy="bundled serve.Decider letter-logit readout with released calibration; one pass up to 160 options",
        ),
        Adapter(
            name="intern-0.8b",
            module="inference.intern_decision",
            args=(
                "--model-path",
                "{model}",
                "--model-revision",
                "{revision}",
                *IO_ARGS,
                "--device",
                "{device}",
            ),
            model_id="internlm/Intern-Decision-0.8B",
            batch_policy="bundled DecisionEngine.predict, one forward per prompt; native rejections (8,192 tokens, 62 options) invalid",
        ),
        Adapter(
            name="bosun17",
            module="inference.bosun06",
            args=(
                "--size",
                "1.7b",
                "--model-path",
                "{model}",
                "--base-path",
                "{base}",
                *IO_ARGS,
            ),
            model_id="Hanno-Labs/bosun-v3.1-1.7b",
            batch_policy="BosunForDecision.predict, one question per call",
            requires=("base",),
        ),
        Adapter(
            name="autojev27",
            module="inference.autojev27",
            args=(
                "--model-path",
                "{model}",
                "--source-path",
                "{source}",
                "--model-revision",
                "{revision}",
                *IO_ARGS,
                "--device",
                "{device}",
            ),
            model_id="denis-pplx/autojev-27b",
            requires=("source",),
        ),
        Adapter(
            name="gliner25",
            module="inference.gliner25",
            args=(
                "--model-path",
                "{model}",
                "--variant",
                "{variant}",
                "--model-revision",
                "{revision}",
                *IO_ARGS,
                "--device",
                "{device}",
            ),
            model_id="fastino/GLiNER2.5-Decide",
            image="decision20-gliner25:host2",
            requires=("variant",),
        ),
        Adapter(
            name="this-that",
            module="inference.this_that",
            args=(
                "--model-path",
                "{model}",
                "--source-path",
                "{source}",
                "--model-revision",
                "{revision}",
                *IO_ARGS,
                "--device",
                "{device}",
            ),
            model_id="flock-io/this-that-model-1.0",
            requires=("source",),
        ),
        Adapter(
            name="decision2-typed",
            module="training.model.infer",
            args=(
                "--checkpoint",
                "{model}",
                "--model-id",
                "{model_id}",
                "--model-revision",
                "{revision}",
                "--max-length",
                "{max_length}",
                "--calibration",
                "{calibration}",
                *IO_ARGS,
            ),
            model_id=None,
            batch_policy="one prompt per call, all its questions batched; no truncation",
            requires=("model_id", "calibration", "max_length"),
        ),
    )
}


def load(name: str | None, spec_path: Path | None) -> Adapter:
    if (name is None) == (spec_path is None):
        raise ValueError("give exactly one of --adapter or --adapter-spec")
    if name is not None:
        if name not in REGISTRY:
            raise ValueError(
                f"unknown adapter {name}; known: {', '.join(sorted(REGISTRY))}"
            )
        return REGISTRY[name]
    raw = json.loads(spec_path.read_text(encoding="utf-8"))
    allowed = set(Adapter.__dataclass_fields__)
    unknown = set(raw) - allowed
    if unknown or not {"name", "module", "args"} <= set(raw):
        raise ValueError(
            f"adapter spec needs name/module/args; unknown keys {sorted(unknown)}"
        )
    for key in ("args", "requires", "extra_mounts"):
        if key in raw:
            raw[key] = tuple(raw[key])
    raw.setdefault("model_id", None)
    return Adapter(**raw)


def module_path(source_root: Path, adapter: Adapter) -> Path:
    return source_root.joinpath(*adapter.module.split(".")).with_suffix(".py")


def module_sha256(source_root: Path, adapter: Adapter) -> str | None:
    path = module_path(source_root, adapter)
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()
