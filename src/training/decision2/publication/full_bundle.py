"""Stage a self-contained package from the scored official-Qwen full checkpoint.

The source checkpoint is never merged, quantized, or rewritten. This builder
rejects LoRA checkpoints and excludes optimizer/trainer state. It cannot make
a release claim: package-native GPU parity and exact Hub readback are separate.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from training.model.calibration import load_calibration
from training.model.infer import checkpoint_fingerprint

from .adapter_bundle import _screen_public_file
from .bundle import _screen_file, sha_file
from .full_runtime_api import MANIFEST_VERSION, _tensor_count

MODEL_ID = "llm-semantic-router/DEV2.0-0.6B"
SOURCE_ID = "Qwen/Qwen3-0.6B-Base"
SOURCE_REVISION = "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"
QWEN_LICENSE_SHA256 = "832dd9e00a68dd83b3c3fb9f5588dad7dcf337a0db50f7d9483f310cd292e92e"
MODEL_SOURCES = (
    "calibration.py",
    "data.py",
    "decision_model.py",
    "infer.py",
    "lora.py",
    "source.py",
)
CONTENT_FILES = {
    "README.md",
    "LICENSE",
    "LICENSE-Qwen",
    "NOTICE",
    "ATTRIBUTIONS.md",
    "assets/DEV2.0-0.6B-owl-banner.png",
    "assets/jevarena-rank.svg",
    "assets/jevarena-task-matrix.svg",
    "assets/jevbench-public-rank.svg",
    "evaluation/EVALUATION.md",
    "evaluation/manifest.json",
}
PANELS = {"typed": (1600, 2000), "css": (6547, 6547), "public": (231, 231)}


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path.name}")
    return value


def _content_files(content: Path) -> dict[str, Path]:
    if not content.is_dir() or content.is_symlink():
        raise ValueError("Content directory is absent or linked")
    paths = {
        path.relative_to(content).as_posix(): path
        for path in content.rglob("*")
        if path.is_file()
    }
    if set(paths) != CONTENT_FILES or any(
        path.is_symlink() for path in content.rglob("*")
    ):
        raise ValueError("Product content differs from the public whitelist")
    for name, path in paths.items():
        if path.suffix == ".png":
            if not path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"):
                raise ValueError("Owl banner must be a PNG")
            data = path.read_bytes()
            if any(tag in data for tag in (b"tEXt", b"iTXt", b"zTXt", b"eXIf")):
                raise ValueError("Owl banner contains metadata chunks")
        else:
            _screen_file(path)
    if sha_file(paths["LICENSE-Qwen"]) != QWEN_LICENSE_SHA256:
        raise ValueError("Official Qwen license bytes differ from the pinned source")
    return paths


def _checkpoint(checkpoint: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if not checkpoint.is_dir() or checkpoint.is_symlink():
        raise ValueError("Selected full checkpoint is absent or linked")
    identity = checkpoint_fingerprint(checkpoint)
    metadata = _object(checkpoint / "decision_config.json")
    if (
        metadata.get("architecture")
        != "qwen3-text-endpoints-global-query-shared-bilinear-mlp"
        or metadata.get("training_mode") != "full"
        or metadata.get("source_stage") != "base"
        or metadata.get("base_revision") != SOURCE_REVISION
        or metadata.get("checkpoint_format") not in (None, "full")
    ):
        raise ValueError("Selected checkpoint is not the official-Qwen full arm")
    expected = set(identity["files_sha256"]) | {"checkpoint.json", "trainer_state.pt"}
    actual = {
        path.relative_to(checkpoint).as_posix()
        for path in checkpoint.rglob("*")
        if path.is_file()
    }
    if actual != expected or any(path.is_symlink() for path in checkpoint.rglob("*")):
        raise ValueError("Selected checkpoint has missing or untracked files")
    for relative in identity["files_sha256"]:
        _screen_public_file(checkpoint / relative)
    if _tensor_count(checkpoint / "backbone/model.safetensors") != metadata.get(
        "text_parameter_count"
    ):
        raise ValueError("Backbone parameter count differs from checkpoint")
    return identity, metadata


def _scored(
    manifests: dict[str, Path],
    seal_path: Path,
    roster_path: Path,
    identity: dict[str, Any],
    calibration: Path,
    metadata: dict[str, Any],
    sources: Path,
) -> dict[str, Any]:
    roster, seal = _object(roster_path), _object(seal_path)
    if (
        sha_file(roster_path) != seal.get("roster_sha256")
        or seal.get("schema")
        != "decision2-official-qwen06b-v3-postkey-prediction-seal/1"
        or not seal.get("post_key_same_panel")
    ):
        raise ValueError("Selected formal prediction seal or roster differs")
    frozen = roster["candidate"]
    if (
        frozen.get("model_sha256") != identity["model_sha256"]
        or frozen.get("base_model") != SOURCE_ID
        or frozen.get("base_revision") != metadata.get("base_revision")
        or frozen.get("calibration_sha256") != sha_file(calibration)
    ):
        raise ValueError("Package checkpoint differs from sealed candidate")
    temperatures, report = load_calibration(calibration, identity["model_sha256"])
    if report.get("inference", {}).get("max_length") != frozen["max_length"]:
        raise ValueError("CAL context differs from sealed inference")
    observed = {}
    for panel, (items, slots) in PANELS.items():
        path = manifests[panel]
        scored = _object(path)
        pinned = seal["candidate"][panel]
        if (
            sha_file(path) != pinned["manifest_sha256"]
            or scored.get("model_sha256") != identity["model_sha256"]
            or scored.get("model_files_sha256") != identity["files_sha256"]
            or scored.get("adapter_sha256") != frozen["adapter_sha256"]
            or scored.get("adapter_version") != frozen["adapter_version"]
            or scored.get("calibration", {}).get("temperature_by_type") != temperatures
            or scored.get("calibration", {}).get("file_sha256") != sha_file(calibration)
            or scored.get("input_sha256") != roster["panel"][panel]["prompts_sha256"]
            or scored.get("input_items") != items
            or scored.get("counts", {}).get("questions") != slots
            or scored.get("predictions_sha256") != pinned["sha256"]
            or scored.get("max_length") != frozen["max_length"]
            or scored.get("checkpoint_format") != "full"
        ):
            raise ValueError(f"{panel} scored native manifest differs")
        for name in MODEL_SOURCES:
            if scored["adapter_files_sha256"].get(name) != sha_file(sources / name):
                raise ValueError(f"{panel} scored source changed: {name}")
        observed[panel] = {
            "manifest_sha256": sha_file(path),
            "predictions_sha256": pinned["sha256"],
        }
    return {
        "temperatures": temperatures,
        "max_length": frozen["max_length"],
        "scored_adapter_sha256": frozen["adapter_sha256"],
        "scored_adapter_version": frozen["adapter_version"],
        "scored": observed,
        "prediction_seal_sha256": sha_file(seal_path),
        "roster_sha256": sha_file(roster_path),
    }


def build(
    *,
    checkpoint: Path,
    calibration: Path,
    scored_manifests: dict[str, Path],
    prediction_seal: Path,
    roster: Path,
    content: Path,
    sources: Path,
    output: Path,
) -> dict[str, Any]:
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    identity, metadata = _checkpoint(checkpoint)
    frozen = _scored(
        scored_manifests,
        prediction_seal,
        roster,
        identity,
        calibration,
        metadata,
        sources,
    )
    product = _content_files(content)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".decision2-full-", dir=output.parent
    ) as temporary:
        stage = Path(temporary) / "package"
        stage.mkdir(mode=0o700)
        for relative in identity["files_sha256"]:
            target = stage / "model" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(checkpoint / relative, target)
        for name in MODEL_SOURCES:
            target = stage / "decision2" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(sources / name, target)
        shutil.copy2(
            Path(__file__).with_name("full_runtime_api.py"), stage / "decision2/api.py"
        )
        (stage / "decision2/__init__.py").write_text(
            "from .api import Decision2, verify_bundle\n", encoding="utf-8"
        )
        shutil.copy2(calibration, stage / "calibration.json")
        for relative, path in product.items():
            target = stage / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        files = {
            path.relative_to(stage).as_posix(): sha_file(path)
            for path in sorted(stage.rglob("*"))
            if path.is_file()
        }
        manifest = {
            "bundle_version": MANIFEST_VERSION,
            "model_id": MODEL_ID,
            "model_sha256": identity["model_sha256"],
            "model_files_sha256": identity["files_sha256"],
            "loader_files_sha256": {
                name.removeprefix("decision2/"): digest
                for name, digest in files.items()
                if name.startswith("decision2/")
            },
            "files_sha256": files,
            "base_model": SOURCE_ID,
            "base_revision": metadata["base_revision"],
            "method": "full",
            "parameter_count": metadata["text_parameter_count"]
            + _tensor_count(checkpoint / "decision_head.safetensors"),
            "max_length": frozen["max_length"],
            "temperature_by_type": frozen["temperatures"],
            "calibration_sha256": sha_file(calibration),
            "scored_adapter_sha256": frozen["scored_adapter_sha256"],
            "scored_adapter_version": frozen["scored_adapter_version"],
            "scored_predictions": frozen["scored"],
            "prediction_seal_sha256": frozen["prediction_seal_sha256"],
            "roster_sha256": frozen["roster_sha256"],
        }
        (stage / "MODEL_MANIFEST.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        # The package is imported in a clean process during CPU preflight.
        # Here, verify the staged exact inventory without loading model weights.
        if {
            **files,
            "MODEL_MANIFEST.json": sha_file(stage / "MODEL_MANIFEST.json"),
        } != {
            path.relative_to(stage).as_posix(): sha_file(path)
            for path in stage.rglob("*")
            if path.is_file()
        }:
            raise ValueError("Staged package inventory changed")
        stage.rename(output)
    return {
        "model_sha256": identity["model_sha256"],
        "manifest_sha256": sha_file(output / "MODEL_MANIFEST.json"),
        "parameter_count": manifest["parameter_count"],
        "files": len(files),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "checkpoint",
        "calibration",
        "prediction-seal",
        "roster",
        "content",
        "sources",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    for panel in PANELS:
        parser.add_argument("--" + panel + "-manifest", type=Path, required=True)
    args = parser.parse_args()
    result = build(
        checkpoint=args.checkpoint,
        calibration=args.calibration,
        scored_manifests={
            panel: getattr(args, panel + "_manifest") for panel in PANELS
        },
        prediction_seal=args.prediction_seal,
        roster=args.roster,
        content=args.content,
        sources=args.sources,
        output=args.output,
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
