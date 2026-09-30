"""Open a Decision 2.0 package for serving with the package's own runtime code.

Every package ships ``decision2/`` (its System One runtime) and
``decision2/_vendor/dev2model/`` (the exact training/model sources that produced
its scored predictions). The endpoint renders prompts, tokenizes, applies
calibration and Score offsets and formats answers with those files, imported
from the package under a private module name, so the served contract is the
package's, not this plugin's. Loading verifies the package the way
``Decision2.from_pretrained`` does: every file against MODEL_MANIFEST.json, then
the model identity that calibration and Score offsets are bound to.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable

from .contract import read_decision_config

SERVED_PROFILES = ("qwen-full",)


@dataclass(frozen=True)
class Runtime:
    """The package's vendored functions used on the request path."""

    question_to_row: Callable[..., dict[str, Any]]
    encode: Callable[..., dict[str, Any]]
    product_answer: Callable[..., dict[str, Any]]
    api_json_payload: Callable[..., bool]
    canonical: Callable[[Any], str]
    apply_score_bias: Callable[..., list[float]] | None


@dataclass(frozen=True)
class Decision2Package:
    root: Path
    manifest: dict[str, Any]
    manifest_sha256: str
    model_name: str
    max_input_tokens: int
    model_sha256: str
    temperatures: dict[str, float]
    score_bias: dict[int, list[float]] | None
    tokenizer: Any
    runtime: Runtime


def import_runtime(root: Path) -> str:
    """Import ``<root>/decision2`` under a name unique to this package; return that name."""
    init = root / "decision2" / "__init__.py"
    if not init.is_file():
        raise ValueError(f"{root} has no decision2/ runtime")
    name = "_vllm_sr_decision2_" + hashlib.sha256(str(root).encode()).hexdigest()[:16]
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, init, submodule_search_locations=[str(init.parent)]
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            del sys.modules[name]
            raise
    return name


def _module(runtime: str, relative: str) -> ModuleType:
    return importlib.import_module(f"{runtime}.{relative}")


def load_package(path: str | Path) -> Decision2Package:
    if Path(path).is_symlink():
        raise ValueError("Package root cannot be a link")
    root = Path(path).resolve(strict=True)
    runtime = import_runtime(root)
    api = _module(runtime, "api")
    manifest = api.verify_bundle(root)
    if manifest["profile"] not in SERVED_PROFILES:
        raise NotImplementedError(
            f"profile {manifest['profile']!r} is not served by this plugin yet "
            f"(served: {', '.join(SERVED_PROFILES)})"
        )
    if manifest.get("storage"):
        raise NotImplementedError(
            "packages with a weight storage codec are not served yet"
        )

    read_decision_config(root)
    infer = _module(runtime, "_vendor.dev2model.infer")
    identity = infer.checkpoint_fingerprint(root, None)
    model_sha256 = identity["model_sha256"]
    if model_sha256 != manifest["identity"]["model_sha256"]:
        raise ValueError("Model identity differs from the scored checkpoint")

    qwen = _module(runtime, "qwen")
    score_bias = qwen.load_score_bias_entry(root, manifest, model_sha256)
    calibration = manifest.get("calibration")
    if calibration:
        load_calibration = _module(
            runtime, "_vendor.dev2model.calibration"
        ).load_calibration
        temperatures, report = load_calibration(
            root / calibration["file"], model_sha256
        )
        if report.get("inference", {}).get("max_length") not in (
            None,
            manifest["max_input_tokens"],
        ):
            raise ValueError("Calibration context differs from the package limit")
    else:
        temperatures = {"choice": 1.0, "noul": 1.0, "score": 1.0}

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer needs a pad or EOS token")

    return Decision2Package(
        root=root,
        manifest=manifest,
        manifest_sha256=hashlib.sha256(
            (root / "MODEL_MANIFEST.json").read_bytes()
        ).hexdigest(),
        model_name=manifest["model_name"],
        max_input_tokens=manifest["max_input_tokens"],
        model_sha256=model_sha256,
        temperatures=dict(temperatures),
        score_bias=score_bias,
        tokenizer=tokenizer,
        runtime=Runtime(
            question_to_row=infer.question_to_row,
            encode=_module(runtime, "_vendor.dev2model.decision_model").encode,
            product_answer=infer.product_answer,
            api_json_payload=infer._api_json_payload,
            canonical=_module(runtime, "_vendor.dev2model.data").canonical,
            apply_score_bias=(
                _module(runtime, "_vendor.dev2model.score_bias").apply
                if score_bias is not None
                else None
            ),
        ),
    )


def describe(package: Decision2Package) -> dict[str, Any]:
    """Identity fields for logs and run manifests."""
    return {
        "model_name": package.model_name,
        "profile": package.manifest["profile"],
        "manifest_sha256": package.manifest_sha256,
        "model_sha256": package.model_sha256,
        "max_input_tokens": package.max_input_tokens,
        "temperatures": package.temperatures,
        "score_bias": package.score_bias is not None,
        "runtime_source": package.manifest.get("runtime", {}).get("runtime_source"),
        "files": len(package.manifest["files_sha256"]),
    }
