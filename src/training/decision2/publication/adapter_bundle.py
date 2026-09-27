"""Prepare an unmerged Decision 2.0 PEFT artifact without uploading it.

This is a prospective functional package, not a release qualification. The
existing full-materialization builders and their gates remain unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

from training.model.calibration import load_calibration
from training.model.data import canonical
from training.model.infer import checkpoint_fingerprint
from training.model.lora import LORA_FORMAT, verify_adapter_config
from training.model.source import verify_source

from .adapter_runtime import (
    HF_ID,
    REQUIRED_PACKAGES,
    REVISION,
    SHA,
    VERSION,
    _hash,
    _inventory,
)
from .bundle import _screen_file
from .bundle_arena import _public_text, _tensor_counts
from .download_config import build_adapter_download_config

LOADER_SOURCES = (
    "calibration.py",
    "data.py",
    "decision_model.py",
    "infer.py",
    "lora.py",
    "source.py",
)
MODEL_ROOT_FILES = {
    "decision_config.json",
    "decision_head.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "tokenizer.model",
    "chat_template.jinja",
}
MODEL_ADAPTER_FILES = {"adapter_config.json", "adapter_model.safetensors"}
NONFUNCTIONAL_CHECKPOINT_FILES = {"checkpoint.json", "trainer_state.pt"}
MODEL_ID = re.compile(r"llm-semantic-router/DEV2\.0-(?:0\.6B|0\.8B|2B|4B|9B|27B)\Z")
SAFE_VERSION = re.compile(r"[A-Za-z0-9][A-Za-z0-9.+!_-]*\Z")
QWEN35_ARCHITECTURE = "qwen3.5-text-endpoints-global-query-shared-bilinear-mlp"
QWEN3_ARCHITECTURE = "qwen3-text-endpoints-global-query-shared-bilinear-mlp"


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must be a JSON object")
    return value


def _screen_public_file(path: Path) -> None:
    if path.name == "tokenizer.json":
        # Qwen's public BPE merge tokens include "/" and "//" as values.
        # They are tokenizer data, not filesystem paths. Keep the general
        # absolute-value rejection for every other tokenizer JSON position.
        content = path.read_text(encoding="utf-8")
        _public_text(content, path.name)

        def reject_paths(value: Any, location: tuple[str, ...] = ()) -> None:
            if isinstance(value, str) and value.startswith("/"):
                merge_token = (
                    len(location) == 4
                    and location[:2] == ("model", "merges")
                    and location[2].isdigit()
                    and location[3] in {"0", "1"}
                )
                if not merge_token:
                    raise ValueError("Absolute path value found in tokenizer.json")
            elif isinstance(value, dict):
                for key, item in value.items():
                    reject_paths(item, (*location, str(key)))
            elif isinstance(value, list):
                for index, item in enumerate(value):
                    reject_paths(item, (*location, str(index)))

        reject_paths(json.loads(content))
        return
    _screen_file(path)
    if path.suffix == ".safetensors":
        with path.open("rb") as source:
            length = int.from_bytes(source.read(8), "little")
            metadata = json.loads(source.read(length))
        _public_text(json.dumps(metadata, ensure_ascii=False), path.name)
    else:
        _public_text(path.read_text(encoding="utf-8"), path.name)


def _model_files(checkpoint: Path) -> dict[str, str]:
    if not checkpoint.is_dir() or checkpoint.is_symlink():
        raise ValueError("PEFT checkpoint directory is missing")
    paths = list(checkpoint.iterdir())
    if any(
        path.is_symlink() or (path.is_dir() and path.name != "adapter")
        for path in paths
    ):
        raise ValueError("PEFT checkpoint has a symlink or unexpected directory")
    root_names = {path.name for path in paths if path.is_file()}
    if (
        not {"decision_config.json", "decision_head.safetensors", "tokenizer.json"}
        <= root_names
    ):
        raise ValueError("PEFT checkpoint lacks head, configuration or tokenizer")
    if not root_names <= MODEL_ROOT_FILES | NONFUNCTIONAL_CHECKPOINT_FILES:
        raise ValueError("PEFT checkpoint has an unsupported root model file")
    adapter = checkpoint / "adapter"
    if not adapter.is_dir() or adapter.is_symlink():
        raise ValueError("PEFT adapter directory is missing")
    adapter_paths = list(adapter.iterdir())
    if any(path.is_symlink() or not path.is_file() for path in adapter_paths):
        raise ValueError("PEFT adapter has a nonregular file")
    adapter_names = {path.name for path in adapter_paths}
    if (
        not adapter_names >= MODEL_ADAPTER_FILES
        or not adapter_names <= MODEL_ADAPTER_FILES | {"README.md"}
    ):
        raise ValueError("PEFT adapter has missing or unexpected functional files")
    selected = [
        *sorted(
            path for path in paths if path.is_file() and path.name in MODEL_ROOT_FILES
        ),
        *sorted(path for path in adapter_paths if path.name in MODEL_ADAPTER_FILES),
    ]
    for path in selected:
        _screen_public_file(path)
    return {path.relative_to(checkpoint).as_posix(): _hash(path) for path in selected}


def _full_parameter_count(
    source: Path, checkpoint: Path, metadata: dict[str, Any], source_kind: str
) -> dict[str, int]:
    # Match the exact loaded Qwen3.5 text model or Qwen3 decoder, excluding
    # vision and language-model output heads.
    weight_root = source / "backbone" if source_kind == "decision1" else source
    weights = sorted(weight_root.glob("*.safetensors"))
    if not weights or any(path.suffix == ".bin" for path in weight_root.iterdir()):
        raise ValueError("Pinned source requires safetensors text weights")
    architecture = metadata.get("architecture")
    if architecture not in {QWEN35_ARCHITECTURE, QWEN3_ARCHITECTURE}:
        raise ValueError("Unsupported Decision 2.0 architecture")
    if source_kind == "decision1" and architecture != QWEN35_ARCHITECTURE:
        raise ValueError("Own Decision 1.0 source has incompatible architecture")
    text_prefix = (
        "model." if architecture == QWEN3_ARCHITECTURE else "model.language_model."
    )
    all_names: set[str] = set()
    base_count = 0
    for path in weights:
        for name, count in _tensor_counts(path).items():
            if name in all_names:
                raise ValueError("Duplicate source tensor across shards")
            all_names.add(name)
            if source_kind == "decision1" or name.startswith(text_prefix):
                base_count += count
    expected = metadata.get("text_parameter_count")
    if type(expected) is not int or expected < 1 or base_count != expected:
        raise ValueError(
            "Full text backbone parameter count differs from scored checkpoint"
        )
    adapter_count = sum(
        _tensor_counts(checkpoint / "adapter/adapter_model.safetensors").values()
    )
    head_count = sum(_tensor_counts(checkpoint / "decision_head.safetensors").values())
    if adapter_count < 1 or head_count < 1:
        raise ValueError("Adapter or decision head has no parameters")
    return {
        "base_text": base_count,
        "adapter": adapter_count,
        "head": head_count,
        "total": base_count + adapter_count + head_count,
    }


def _lock(path: Path, scored: dict[str, Any]) -> dict[str, str]:
    value = _object(path)
    if set(value) != {"python", *REQUIRED_PACKAGES} or any(
        not isinstance(item, str) or SAFE_VERSION.fullmatch(item) is None
        for item in value.values()
    ):
        raise ValueError("Runtime lock must pin Python and every loader dependency")
    if value["peft"] != scored.get("peft_version") or value["torch"] != scored.get(
        "torch_version"
    ):
        raise ValueError("Runtime lock differs from native scored PEFT/Torch versions")
    return value


def _verify_staged_runtime(package: Path, source: Path) -> None:
    init = package / "decision2/__init__.py"
    spec = importlib.util.spec_from_file_location(
        "decision2", init, submodule_search_locations=[str(init.parent)]
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import staged Decision 2.0 adapter runtime")
    previous = {
        name: sys.modules.pop(name)
        for name in list(sys.modules)
        if name == "decision2" or name.startswith("decision2.")
    }
    bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        module = importlib.util.module_from_spec(spec)
        sys.modules["decision2"] = module
        spec.loader.exec_module(module)
        module.verify_bundle(package, source)
    finally:
        sys.dont_write_bytecode = bytecode
        for name in list(sys.modules):
            if name == "decision2" or name.startswith("decision2."):
                del sys.modules[name]
        sys.modules.update(previous)


def assemble(
    *,
    checkpoint: Path,
    source: Path,
    calibration: Path,
    scored_manifest: Path,
    dependency_lock: Path,
    base_repo_id: str,
    base_revision: str,
    model_id: str,
    output: Path,
) -> dict[str, Any]:
    """Stage exact PEFT weights, native loader and external-base pin."""
    if MODEL_ID.fullmatch(model_id) is None or HF_ID.fullmatch(base_repo_id) is None:
        raise ValueError("Model or upstream Hugging Face ID is invalid")
    if REVISION.fullmatch(base_revision) is None:
        raise ValueError("Upstream base revision must be a 40-character commit SHA")
    for path in (
        checkpoint,
        source,
        calibration,
        scored_manifest,
        dependency_lock,
        output,
    ):
        if path.is_symlink():
            raise ValueError("Package input or output cannot be a symlink")
    checkpoint = checkpoint.resolve(strict=True)
    source = source.resolve(strict=True)
    calibration = calibration.resolve(strict=True)
    scored_manifest = scored_manifest.resolve(strict=True)
    dependency_lock = dependency_lock.resolve(strict=True)
    output = output.resolve()
    if output.exists() or any(
        output.is_relative_to(path) for path in (checkpoint, source)
    ):
        raise FileExistsError("Output exists or is nested inside an input")
    model_files = _model_files(checkpoint)
    metadata = _object(checkpoint / "decision_config.json")
    contract = metadata.get("lora")
    if (
        metadata.get("checkpoint_format") != LORA_FORMAT
        or metadata.get("architecture") not in {QWEN35_ARCHITECTURE, QWEN3_ARCHITECTURE}
        or metadata.get("prompt_version")
        != "decision2-segmented-options-global-query-v1"
        or not isinstance(contract, dict)
    ):
        raise ValueError("Input is not a Decision 2.0 PEFT LoRA checkpoint")
    source_kind = contract.get("source_kind")
    if source_kind not in {"base", "posttrained", "decision1"}:
        raise ValueError("Checkpoint has an unsupported publication source kind")
    if source_kind in {"base", "posttrained"} and not base_repo_id.startswith("Qwen/"):
        raise ValueError("Official Qwen source is required for this model family")
    if (
        source_kind in {"base", "posttrained"}
        and contract.get("base_revision") != base_revision
    ):
        raise ValueError("Checkpoint source kind or immutable revision differs")
    if source_kind == "decision1" and not base_repo_id.startswith(
        "llm-semantic-router/Decision-1.0-"
    ):
        raise ValueError("Decision 1.0 source must be an own-family repository")
    if source_kind == "decision1":
        old = _object(source / "decision_config.json")
        if (
            old.get("architecture")
            != "contextual-candidate-endpoint-plus-global-query-shared-bilinear-mlp"
            or old.get("prompt_version")
            != "structured-segmented-candidate-endpoints-global-query-v2"
            or old.get("head_dim") != metadata.get("head_dim")
            or not (source / "backbone/config.json").is_file()
        ):
            raise ValueError(
                "Decision 1.0 source is incompatible with the native loader"
            )
    verify_source(source, contract.get("source_fingerprint"))
    verify_adapter_config(checkpoint / "adapter", contract)
    source_files = _inventory(source, ignore_cache=True)
    if not source_files or not any(
        name.endswith(".safetensors") for name in source_files
    ):
        raise ValueError("Upstream base source has no weight files")
    for name in source_files:
        _public_text(name, "upstream source filename")
    _public_text(base_repo_id, "upstream repository ID")
    model_identity = checkpoint_fingerprint(checkpoint, source)
    scored = _object(scored_manifest)
    if (
        scored.get("checkpoint_format") != LORA_FORMAT
        or scored.get("model_sha256") != model_identity["model_sha256"]
        or scored.get("model_files_sha256") != model_identity["files_sha256"]
        or not isinstance(scored.get("predictions_sha256"), str)
        or SHA.fullmatch(scored["predictions_sha256"]) is None
    ):
        raise ValueError(
            "Native scored checkpoint identity differs from package source"
        )
    runtime_root = Path(__file__).resolve().parents[1] / "training/model"
    loader_sources = {name: runtime_root / name for name in LOADER_SOURCES}
    loader_sources["api.py"] = Path(__file__).with_name("adapter_runtime.py")
    scored_sources = scored.get("adapter_files_sha256")
    expected_scored_sources = {
        name: _hash(loader_sources[name]) for name in LOADER_SOURCES
    }
    if (
        scored_sources != expected_scored_sources
        or scored.get("adapter_sha256")
        != hashlib.sha256(
            canonical(expected_scored_sources).encode("utf-8")
        ).hexdigest()
    ):
        raise ValueError("Native inference loader changed since the scored run")
    _screen_public_file(calibration)
    cal_sha = _hash(calibration)
    if scored.get("calibration", {}).get("file_sha256") != cal_sha:
        raise ValueError("Scored inference used another CAL file")
    temperature, report = load_calibration(calibration, model_identity["model_sha256"])
    max_length = report.get("inference", {}).get("max_length")
    if (
        type(max_length) is not int
        or max_length < 1
        or scored.get("max_length") != max_length
    ):
        raise ValueError("Scored inference context differs from calibrated contract")
    lock = _lock(dependency_lock, scored)
    parameters = _full_parameter_count(source, checkpoint, metadata, source_kind)
    for path in loader_sources.values():
        _screen_public_file(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        for name in model_files:
            destination = temporary / "model" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(checkpoint / name, destination)
        shutil.copyfile(calibration, temporary / "calibration.json")
        (temporary / "decision2").mkdir()
        for name, path in loader_sources.items():
            shutil.copyfile(path, temporary / "decision2" / name)
        (temporary / "decision2/__init__.py").write_text(
            "from .api import Decision2, verify_bundle\n", encoding="utf-8"
        )
        (temporary / "requirements.txt").write_text(
            "\n".join(f"{name}=={lock[name]}" for name in REQUIRED_PACKAGES) + "\n",
            encoding="utf-8",
        )
        (temporary / "config.json").write_text(
            json.dumps(
                build_adapter_download_config(
                    temporary, model_id, base_repo_id, base_revision
                ),
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        source_label = (
            "Decision 1.0 source" if source_kind == "decision1" else "Qwen base model"
        )
        (temporary / "README.md").write_text(
            f"# {model_id.rsplit('/', 1)[1]}: adapter artifact candidate\n\n"
            f"This unmerged PEFT package depends on an immutable {source_label}. "
            "The artifact is a preparation candidate only; it has no release or score "
            "claim until independent native parity and the full JevArena release gate pass.\n\n"
            "Use `decision2.Decision2.from_pretrained(package_path, source_path=base_snapshot)` "
            "for the native Choice/Noul/Score API. The loader verifies every base and "
            "package file before inference. Without `source_path`, it downloads the "
            "pinned base commit and verifies the same bytes.\n",
            encoding="utf-8",
        )
        for path in temporary.rglob("*"):
            if path.is_file():
                _screen_public_file(path)
        files = _inventory(temporary)
        loader_files = {
            name.removeprefix("decision2/"): digest
            for name, digest in files.items()
            if name.startswith("decision2/")
        }
        manifest = {
            "bundle_version": VERSION,
            "model_id": model_id,
            "base": {
                "repo_id": base_repo_id,
                "revision": base_revision,
                "source_kind": source_kind,
                "files_sha256": source_files,
            },
            "model_sha256": model_identity["model_sha256"],
            "model_files_sha256": model_files,
            "loader_files_sha256": loader_files,
            "calibration_sha256": cal_sha,
            "temperature_by_type": temperature,
            "max_length": max_length,
            "scored_prediction_manifest_sha256": _hash(scored_manifest),
            "scored_predictions_sha256": scored["predictions_sha256"],
            "dependencies": lock,
            "parameter_breakdown": parameters,
            "parameter_count": parameters["total"],
            "files_sha256": files,
            "publication_status": "candidate-parity-pending",
        }
        manifest_text = (
            json.dumps(
                manifest, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
            )
            + "\n"
        )
        _public_text(manifest_text, "adapter package manifest")
        (temporary / "MODEL_MANIFEST.json").write_text(manifest_text, encoding="utf-8")
        _verify_staged_runtime(temporary, source)
        if output.exists():
            raise FileExistsError(output)
        os.replace(temporary, output)
        return manifest
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "checkpoint",
        "source",
        "calibration",
        "scored_manifest",
        "dependency_lock",
        "output",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    for name in ("base_repo_id", "base_revision", "model_id"):
        parser.add_argument(f"--{name.replace('_', '-')}", required=True)
    args = parser.parse_args()
    manifest = assemble(**vars(args))
    print(
        json.dumps(
            {
                "model_sha256": manifest["model_sha256"],
                "parameter_count": manifest["parameter_count"],
                "publication_status": manifest["publication_status"],
            }
        )
    )


if __name__ == "__main__":
    main()
