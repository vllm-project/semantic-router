"""The Decision 1.0 model family (Phase 2).

Decision 1.0 packages carry the root pointer ``{"decision_format":
"vllm-sr-decision", "format_version": 1, "runtime_family": ...}`` and run one
of two runtimes: ``vela-encoder`` (Kai, Lex, Route; ``vela.py``) or
``qwen3.5-decision`` (Eos, Sol, Nox, Lux; ``qwen.py``). The family
reimplements the packages' bundled runtime on the native engine's ModernBERT
and Qwen3.5 backbones, the typed marker heads and the shared candidate head;
it never imports the bundled Python.
"""

from __future__ import annotations

from typing import Any

from ...errors import PackageError
from ...heads.candidate import load_head
from ...heads.typed import load_readout
from ...plugins.base import (
    BackboneSpec,
    BranchSpec,
    DtypePolicy,
    EngineModel,
    ModelFamily,
    ModelInfo,
    ModelSpec,
    PackageRef,
    VerifiedPackage,
)
from ...registry import builtin, policy
from ...registry.resolve import fetch
from ...registry.tables.common import BuiltinModel
from ...systemone import (
    GOLDEN_STATE,
    MAX_LEVELS,
    MAX_OPTIONS,
    MIN_LEVELS,
    MIN_OPTIONS,
    golden_questions,
)
from ...text.tokenizer import Tokenizer
from . import package as pkg
from . import qwen, vela
from .model import Decision1Model, QwenDecisionModel, VelaDecisionModel
from .questions import KINDS, MAX_QUESTIONS

__all__ = ["Decision1Family"]

# The licences of the released packages: Apache-2.0, with the Gemma terms the
# encoders' tokenizer inherits through Vela and mmBERT.
LICENCES = {
    pkg.VELA: {
        "spdx": "apache-2.0",
        "components": [{"name": "tokenizer", "licence": "gemma-terms-of-use"}],
    },
    pkg.QWEN: {"spdx": "apache-2.0", "components": []},
}
# The released golden answers were recorded with a null catch-all description.
GOLDEN_QUESTIONS = golden_questions(None)


def expected_parameters(details: pkg.Decision1Package) -> int | None:
    """Parameters the package declares: the encoder's total, or the decoder's backbone plus its head."""
    config = details.model_config
    if details.files.runtime == pkg.VELA:
        return config.get("parameters")
    text = config.get("text_parameter_count")
    if text is None:
        return None
    hidden, head_dim = details.backbone_config["hidden_size"], config["head_dim"]
    parameters: int = text + 4 * hidden + 4 * hidden * head_dim + 2 * head_dim
    return parameters


class Decision1Family(ModelFamily):
    name = "decision1"
    surfaces = frozenset({"decisions"})
    builtin_table = "vllm_srun.registry.tables.decision1"
    fixture_writer = "vllm_srun.testing.decision1"

    @classmethod
    def descriptor(cls) -> dict[str, Any]:
        return {
            "surfaces": sorted(cls.surfaces),
            "formats": ["vllm-sr-decision/1"],
            "runtimes": list(pkg.RUNTIMES),
            "question_types": list(KINDS),
            "presets": True,
        }

    def detect(self, package: PackageRef) -> bool:
        return pkg.read_pointer(package.root) is not None

    def _pinned(self, package: PackageRef) -> BuiltinModel | None:
        known = builtin.lookup(package.repo_id) if package.repo_id else None
        if (
            known is None
            or known.family != self.name
            or known.revision != package.revision
        ):
            return None
        return known

    def fetch(self, package: PackageRef) -> PackageRef:
        """A Hub package without a built-in entry arrives with its pointer only: download its file map."""
        pointer = pkg.read_pointer(package.root)
        if (
            package.repo_id is None
            or self._pinned(package) is not None
            or pointer is None
        ):
            return package
        names = [
            *pkg.file_map(pointer).model_files(),
            pkg.MANIFEST_FILE,
            pkg.PRESETS_FILE,
        ]
        return fetch(
            package,
            names,
            cache_dir=self.options.cache_dir,
            offline=self.options.offline,
        )

    def verify(self, package: PackageRef) -> VerifiedPackage:
        known = self._pinned(package)
        details = pkg.verify(package.root, known.files if known else None)
        if known is not None and details.model_sha256 != known.model_sha256:
            raise PackageError(
                f"{package.repo_id}@{package.revision} differs from the built-in pinned identity"
            )
        runtime = details.files.runtime
        if runtime == pkg.VELA:
            vela.check_config(details.model_config)
        else:
            qwen.check_config(details.model_config)
        manifest = {"licence": LICENCES[runtime]} if known else {}
        return VerifiedPackage(
            ref=package,
            family=self.name,
            model_name=details.files.model_name,
            manifest=manifest,
            manifest_sha256=details.manifest_sha256,
            model_sha256=details.model_sha256,
            max_input_tokens=(
                vela.MAX_INPUT_TOKENS if runtime == pkg.VELA else qwen.MAX_INPUT_TOKENS
            ),
            licence=policy.check(manifest, self.options.accept_licences),
            loaded_parameters=expected_parameters(details),
            details={"package": details},
        )

    def describe(self, package: VerifiedPackage) -> ModelSpec:
        details: pkg.Decision1Package = package.details["package"]
        files, root = details.files, details.root
        weights = tuple(root / name for name in files.backbone_weights)
        if files.runtime == pkg.VELA:
            branches = {
                branch: BranchSpec(
                    weight_files=(root / files.decision_weights[f"{branch}_encoder"],),
                    layers=f"{branch}_blocks",
                    final_norm=f"{branch}_final_norm",
                )
                for branch in vela.BRANCH.values()
                if branch is not None
            }
            known = builtin.by_identity(package.model_sha256)
            reduced = known.reduced if known else {}
            return ModelSpec(
                name=package.model_name,
                backbone=BackboneSpec(
                    model_type="modernbert",
                    config=details.backbone_config,
                    weight_files=weights,
                    branches=branches,
                ),
                dtype=DtypePolicy(
                    autocast=None,
                    bf16_resident=False,
                    reduced_gpu=reduced.get("gpu"),
                    reduced_cpu=reduced.get("cpu"),
                ),
                max_input_tokens=package.max_input_tokens,
                encoder=True,
                kernel_variants={"sdpa": "additive_masks"},
            )
        config = details.backbone_config.get("text_config", details.backbone_config)
        if config.get("model_type") != "qwen3_5_text":
            raise PackageError(
                "a qwen3.5-decision backbone must be a qwen3_5_text model"
            )
        variants = (
            {"causal_conv1d": "fp64_accumulate"}
            if files.model_name in qwen.FP64_CONV_ON_GFX942
            else {}
        )
        backbone = BackboneSpec(
            model_type="qwen3_5_text", config=config, weight_files=weights
        )
        return ModelSpec(
            name=package.model_name,
            backbone=backbone,
            dtype=DtypePolicy(bf16_resident=False, gpu_weights="bfloat16"),
            max_input_tokens=package.max_input_tokens,
            kernel_variants=variants,
            requires=backbone.requires,
        )

    def load(
        self, package: VerifiedPackage, spec: ModelSpec, engine_model: EngineModel
    ) -> Decision1Model:
        details: pkg.Decision1Package = package.details["package"]
        files, root = details.files, details.root
        tokenizer_dir = (root / files.tokenizer["json"]).parent
        if (root / files.tokenizer["json"]).name != "tokenizer.json":
            raise PackageError("the tokenizer file must be named tokenizer.json")
        tokenizer = Tokenizer.from_package(tokenizer_dir, package.max_input_tokens)
        hidden = spec.backbone.config["hidden_size"]
        if files.runtime == pkg.VELA:
            head = vela.check_config(details.model_config)
            readout = load_readout(
                root / files.decision_weights["decision_heads"], hidden, head
            )
            extra = sum(parameter.numel() for parameter in readout.parameters())
            info = self._info(package, engine_model, extra, "fp32")
            return VelaDecisionModel(
                info,
                engine_model,
                tokenizer,
                details.presets,
                readout=readout,
                special=vela.special_ids(
                    tokenizer.backend, root, dict(files.tokenizer)
                ),
                exit_layer=spec.backbone.config["num_hidden_layers"],
            )
        candidate = load_head(
            root / files.decision_weights["decision_head"],
            hidden,
            qwen.check_config(details.model_config),
        )
        extra = sum(parameter.numel() for parameter in candidate.parameters())
        on_cpu = engine_model.device.type == "cpu"
        info = self._info(
            package, engine_model, extra, "fp32" if on_cpu else "bf16/fp32-head"
        )
        assert details.temperatures is not None
        return QwenDecisionModel(
            info,
            engine_model,
            tokenizer,
            details.presets,
            head=candidate,
            temperatures=details.temperatures,
            null_choice_as_key=files.model_name in qwen.NULL_CHOICE_AS_KEY,
        )

    def _info(
        self,
        package: VerifiedPackage,
        engine_model: EngineModel,
        extra: int,
        dtype: str,
    ) -> ModelInfo:
        details: pkg.Decision1Package = package.details["package"]
        parameters = engine_model.parameter_count() + extra
        expected = package.loaded_parameters
        if expected is not None and parameters != expected:
            raise PackageError(
                f"loaded {parameters:,} parameters; the package declares {expected:,}"
            )
        return ModelInfo(
            id=package.model_name,
            family=self.name,
            repo=package.ref.repo_id,
            revision=package.ref.revision,
            model_sha256=package.model_sha256,
            manifest_sha256=package.manifest_sha256,
            surfaces=tuple(sorted(self.surfaces)),
            question_types=KINDS,
            limits={
                "max_input_tokens": package.max_input_tokens,
                "max_questions": MAX_QUESTIONS,
                "min_options": MIN_OPTIONS,
                "max_options": MAX_OPTIONS,
                "min_levels": MIN_LEVELS,
                "max_levels": MAX_LEVELS,
            },
            licence=package.licence,
            parameters=parameters,
            dtype=dtype,
            presets=tuple(sorted(details.presets)),
        )

    def golden(self, package: VerifiedPackage) -> list[dict[str, Any]]:
        details: pkg.Decision1Package = package.details["package"]
        questions = dict(GOLDEN_QUESTIONS)
        if details.presets:
            first = sorted(details.presets)[0]
            questions[f"preset:{first}"] = {"preset": first}
        return builtin.golden(
            package.model_sha256,
            "decisions",
            {"state": GOLDEN_STATE, "questions": questions},
        )
