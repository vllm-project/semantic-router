"""The Decision 2.0 model family plugin."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

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
    LoRASpec,
    ModelFamily,
    ModelInfo,
    ModelSpec,
    PackageRef,
    VerifiedPackage,
)
from ...plugins.decisions import DecisionModel, RenderedItem, RequestPlan

__all__ = ["Decision2Family", "Decision2Model"]
from ...heads.candidate import CandidateHead, forward_logits, load_head
from ...registry import builtin, policy
from ...registry.artifacts import read_json
from ...registry.resolve import download_base
from ...systemone import (
    GOLDEN_STATE,
    MAX_LEVELS,
    MAX_OPTIONS,
    MIN_LEVELS,
    golden_questions,
    question_options,
    valid_state,
)
from ...text.segments import collate, encode
from ...text.tokenizer import Tokenizer
from . import package as pkg
from .answers import apply_score_bias, product_answer

GOLDEN_QUESTIONS = golden_questions("Anything else")


class Decision2Family(ModelFamily):
    name = "decision2"
    surfaces = frozenset({"decisions"})
    builtin_table = "vllm_srun.registry.tables.decision2"
    fixture_writer = "vllm_srun.testing.decision2"

    @classmethod
    def descriptor(cls) -> dict[str, Any]:
        return {
            "surfaces": sorted(cls.surfaces),
            "formats": ["vllm-sr-decision/2"],
            "question_types": ["choice", "noul", "score"],
        }

    def detect(self, package: PackageRef) -> bool:
        return pkg.is_package(package.root)

    def verify(self, package: PackageRef) -> VerifiedPackage:
        root = package.root
        pointer, manifest, manifest_sha256 = pkg.verify_manifest(root)
        licence = policy.check(manifest, self.options.accept_licences)
        decision_config = read_json(root / "decision_config.json")
        backbone_type = pkg.check_decision_config(decision_config)
        base_root = None
        if manifest["profile"] == "qwen-adapter":
            if decision_config.get("checkpoint_format") != pkg.LORA_FORMAT:
                raise PackageError("an adapter package needs a peft-lora/1 checkpoint")
            pkg.verify_adapter(root, decision_config)
            base = manifest.get("base")
            if not isinstance(base, dict) or not pkg.REVISION.fullmatch(
                str(base.get("revision"))
            ):
                raise PackageError("adapter package has no pinned base revision")
            if self.options.base_path is not None:
                base_root = Path(self.options.base_path).resolve()
            else:
                base_root = download_base(
                    base["repo_id"],
                    base["revision"],
                    sorted(base["files_sha256"]),
                    cache_dir=self.options.cache_dir,
                    offline=self.options.offline,
                )
            pkg.verify_base(base_root, base)
        elif decision_config.get("checkpoint_format") == pkg.LORA_FORMAT:
            raise PackageError("a qwen-full package cannot hold a LoRA checkpoint")
        model_sha256 = pkg.model_identity(root, decision_config, base_root)
        identity = (
            cast(dict[str, Any], manifest.get("identity"))
            if isinstance(manifest.get("identity"), dict)
            else {}
        )
        if model_sha256 != identity.get("model_sha256"):
            raise PackageError(
                "model identity differs from the scored checkpoint in MODEL_MANIFEST.json"
            )
        known = builtin.lookup(package.repo_id) if package.repo_id else None
        if (
            known is not None
            and known.revision == package.revision
            and (
                known.model_sha256 != model_sha256
                or known.manifest_sha256 != manifest_sha256
            )
        ):
            raise PackageError(
                f"{package.repo_id}@{package.revision} differs from the built-in pinned identity"
            )
        details = pkg.Decision2Package(
            root=root,
            manifest=manifest,
            pointer=pointer,
            decision_config=decision_config,
            profile=manifest["profile"],
            backbone_type=backbone_type,
            base_root=base_root,
            model_sha256=model_sha256,
            temperatures=pkg.load_temperatures(root, manifest, model_sha256),
            score_bias=pkg.load_score_bias(root, manifest, model_sha256),
        )
        return VerifiedPackage(
            ref=package,
            family=self.name,
            model_name=manifest["model_name"],
            manifest=manifest,
            manifest_sha256=manifest_sha256,
            model_sha256=model_sha256,
            max_input_tokens=manifest["max_input_tokens"],
            licence=licence,
            loaded_parameters=manifest["parameters"].get("loaded"),
            details={"package": details},
        )

    def describe(self, package: VerifiedPackage) -> ModelSpec:
        details: pkg.Decision2Package = package.details["package"]
        root = details.root
        if details.profile == "qwen-full":
            config = read_json(root / "backbone" / "config.json")
            files = tuple(sorted((root / "backbone").glob("*.safetensors")))
            backbone = BackboneSpec(
                model_type=_model_type(config, details.backbone_type),
                config=config,
                weight_files=files,
            )
        else:
            base_root = details.base_root
            assert base_root is not None
            full = read_json(base_root / "config.json")
            config = full.get("text_config", full)
            files = tuple(sorted(base_root.glob("*.safetensors")))
            contract = details.decision_config["lora"]
            lora = LoRASpec(
                adapter_config=read_json(root / "adapter" / "adapter_config.json"),
                weight_files=(root / "adapter" / "adapter_model.safetensors",),
                rank=int(contract["rank"]),
                alpha=float(contract["alpha"]),
                target_modules=tuple(contract["target_modules"]),
            )
            prefix = "model.language_model." if "text_config" in full else ""
            backbone = BackboneSpec(
                model_type=_model_type(config, details.backbone_type),
                config=config,
                weight_files=files,
                weight_prefix=prefix,
                lora=lora,
            )
        return ModelSpec(
            name=package.model_name,
            backbone=backbone,
            # cuda-fused-approximate.md: no decision changed with CUDA's
            # approximate fused kernels.
            dtype=DtypePolicy(approximate_kernels=True),
            max_input_tokens=package.max_input_tokens,
            requires=backbone.requires,
        )

    def load(
        self, package: VerifiedPackage, spec: ModelSpec, engine_model: EngineModel
    ) -> Decision2Model:
        details: pkg.Decision2Package = package.details["package"]
        hidden = spec.backbone.config["hidden_size"]
        head = load_head(
            details.root / "decision_head.safetensors",
            hidden,
            details.decision_config["head_dim"],
        )
        head = head.to(engine_model.device)
        tokenizer = Tokenizer.from_package(details.root, package.max_input_tokens)
        parameters = engine_model.parameter_count() + sum(
            p.numel() for p in head.parameters()
        )
        expected = package.manifest["parameters"]["loaded"]
        if parameters != expected:
            raise PackageError(
                f"loaded {parameters:,} parameters; the manifest declares {expected:,}"
            )
        dtype = (
            "fp32" if engine_model.device.type == "cpu" else "bf16-autocast/fp32-head"
        )
        info = ModelInfo(
            id=package.model_name,
            family=self.name,
            repo=package.ref.repo_id,
            revision=package.ref.revision,
            model_sha256=package.model_sha256,
            manifest_sha256=package.manifest_sha256,
            surfaces=tuple(sorted(self.surfaces)),
            question_types=("choice", "noul", "score"),
            limits={
                "max_input_tokens": package.max_input_tokens,
                "min_options": 2,
                "max_options": MAX_OPTIONS,
                "min_levels": MIN_LEVELS,
                "max_levels": MAX_LEVELS,
            },
            licence=package.licence,
            parameters=parameters,
            dtype=dtype,
        )
        return Decision2Model(info, engine_model, head, tokenizer, details)

    def golden(self, package: VerifiedPackage) -> list[dict[str, Any]]:
        return builtin.golden(
            package.model_sha256,
            "decisions",
            {"state": GOLDEN_STATE, "questions": GOLDEN_QUESTIONS},
        )


def _model_type(config: dict[str, Any], declared: str) -> str:
    model_type = config.get("model_type")
    if model_type == "qwen3_5":
        model_type = "qwen3_5_text"
    if model_type != declared:
        raise PackageError(
            f"backbone config model_type {model_type!r} differs from decision_config ({declared})"
        )
    checked: str = model_type
    return checked


class Decision2Model(DecisionModel[RenderedItem, list[float] | None]):
    def __init__(
        self,
        info: ModelInfo,
        engine_model: EngineModel,
        head: CandidateHead,
        tokenizer: Tokenizer,
        details: pkg.Decision2Package,
    ):
        self.info = info
        self.engine_model = engine_model
        self.head = head
        self.tokenizer = tokenizer
        self.details = details

    def forward_token_budget(self) -> int | None:
        return self.engine_model.max_forward_tokens()

    def plan(
        self, state: Any, questions: dict[str, Any], scan: int | None = None
    ) -> RequestPlan[RenderedItem]:
        if not valid_state(state):
            raise ValueError("state must be text, an object, or an array")
        items: list[RenderedItem] = []
        errors: dict[str, dict[str, Any]] = {}
        tokens = 0
        for question_id, question in questions.items():
            try:
                kind, instructions, options = question_options(question)
                encoded = encode(
                    question_id,
                    state,
                    kind,
                    instructions,
                    options,
                    self.tokenizer.encode,
                    self.info.limits["max_input_tokens"],
                )
            except QuestionError as exc:
                errors[question_id] = question_error(
                    question.get("type") if isinstance(question, dict) else None,
                    exc,
                    (
                        exc.code
                        if exc.code in (INVALID_QUESTION, MAX_LENGTH_EXCEEDED)
                        else INVALID_QUESTION
                    ),
                )
                continue
            tokens += len(encoded["ids"])
            items.append(
                RenderedItem(
                    question_id=question_id,
                    task_type=kind,
                    ids=encoded["ids"],
                    gather=encoded["endpoints"],
                    query=encoded["query"],
                    keys=[option["key"] for option in options],
                    descriptions=[option["description"] for option in options],
                )
            )
        return RequestPlan(
            complete_inputs=frozenset(
                key
                for key, question in questions.items()
                if isinstance(question, dict)
                and question.get("require_full_input") is True
                and key not in errors
            ),
            question_ids=list(questions),
            items=items,
            errors=errors,
            input_tokens=tokens,
        )

    def run(self, items: list[RenderedItem]) -> list[list[float] | None]:
        return self.run_shared(items, 0)

    def run_shared(
        self, items: list[RenderedItem], shared_prefix: int
    ) -> list[list[float] | None]:
        batch = collate(items, self.tokenizer.pad_id)
        scores = forward_logits(
            self.engine_model,
            self.head,
            batch,
            [len(item.ids) for item in items],
            shared_prefix,
        )
        if scores.shape[0] != len(items):
            raise RuntimeError("model returned the wrong number of question answers")
        return [
            scores[index, : len(item.keys)].float().cpu().tolist()
            for index, item in enumerate(items)
        ]

    def answer(self, item: RenderedItem, values: list[float] | None) -> dict[str, Any]:
        if values is None:
            return {"type": item.task_type, "error": INVALID_MODEL_OUTPUT}
        try:
            if self.details.score_bias is not None and item.task_type == "score":
                values = apply_score_bias(
                    self.details.score_bias, values, len(item.keys)
                )
            return product_answer(
                item.task_type,
                item.keys,
                values,
                self.details.temperatures[item.task_type],
                item.descriptions,
            )
        except ValueError:
            return {"type": item.task_type, "error": INVALID_MODEL_OUTPUT}
