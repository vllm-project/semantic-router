"""Gold-free prospective JevArena v3 lock and prediction seal for own-Sol 2B.

This is a post-key same-panel comparison: earlier unrelated work exposed the
v3 labels. Neither command reads answer files or scores a prediction. The
selected checkpoint, package and comparator versions are immutable constants;
the private config supplies only local paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from inference.run import ADAPTER_VERSION as BASELINE_ADAPTER_VERSION
from inference.run import digest as baseline_digest
from inference.run import local_revision
from publication.adapter_bundle import _verify_staged_runtime
from publication.package_native_arena import (
    ADAPTER_VERSION as PACKAGE_ADAPTER_VERSION,
)
from publication.package_native_arena import input_digest, load_gold_free
from scripts.baseline_repeat_smoke_v3 import PROMPTS_SHA as SMOKE_SHA
from scripts.baseline_repeat_smoke_v3 import compare as compare_smoke
from scripts.plan_final_eval import sha_file

LOCK_VERSION = "decision2-sol2b-v3-postkey-candidate-lock/1"
SEAL_VERSION = "decision2-sol2b-v3-postkey-prediction-seal/1"
CANDIDATE_ID = "llm-semantic-router/DEV2.0-2B"
CANDIDATE_SHA = "49ce326b5b116ed81397cda874f6e287e8c37126ec2f1f86e7140a4ca39ff629"
PACKAGE_SHA = "a917ffe0cc64aa6e551825f0d4458d16deffe56ab0bcb451bd8f91687c953d8a"
CAL_SHA = "0ed1805105febba3c1639a8b230781085913d0c6e80e456a18f036f354c61227"
OWN_ID = "llm-semantic-router/Decision-1.0-Sol-2B"
OWN_REVISION = "0665a41108e8f0b33a9515c98311c45947b99399"
PEER_ID = "Mapika/decider-2b"
PEER_REVISION = "533964dae8be954c5b5e19fa4948e48408094c1e"
PARAMETERS = 1_900_750_144
PANELS = {
    "typed": (
        1600,
        2000,
        "e2a4a86bc978fc7497823e106533d8aa896a0307453d712f7bf99ee3174e87bd",
    ),
    "css": (
        6547,
        6547,
        "7a527357e8ac3ca8da8f8663da66684d04c568a8c728261125c194294dd34af6",
    ),
    "public": (
        231,
        231,
        "642d3fac1b6521fe33df72f9228e4e4e364b7be7ea277893207f97da5bc75ddd",
    ),
}
SOURCE_FILES = (
    "publication/package_native_arena.py",
    "publication/adapter_bundle.py",
    "publication/adapter_runtime.py",
    "inference/run.py",
    "benchmark/score.py",
    "transfer/score.py",
    "jev_arena/arena_v3.py",
    "jev_arena/compare_v3.py",
    "jev_arena/jevbench_public.py",
    "scripts/baseline_repeat_smoke_v3.py",
    "scripts/preflight_sol2b_v3.py",
    "research/sol2b-own-source-adapter-package-preflight-2026-09-27.md",
)


def _object(path: Path) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError("Required file is absent or a symlink")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Required JSON is not an object")
    return value


def _write_private(path: Path, value: dict[str, Any]) -> str:
    if not path.is_absolute() or path.exists() or path.is_symlink():
        raise ValueError("Receipt path must be a new absolute path")
    parent = path.parent.resolve(strict=True)
    if (
        stat.S_IMODE(parent.stat().st_mode) != 0o700
        or parent.stat().st_uid != os.getuid()
    ):
        raise ValueError("Receipt directory must be owned and mode 0700")
    content = (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode()
    with os.fdopen(
        os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600), "wb"
    ) as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    return hashlib.sha256(content).hexdigest()


def _path(cfg: dict[str, Any], key: str) -> Path:
    path = Path(cfg[key])
    if not path.is_absolute() or path.is_symlink():
        raise ValueError(f"{key} must be an absolute real path")
    return path.resolve(strict=True)


def _files(root: Path) -> dict[str, str]:
    result = {}
    for file in sorted(root.rglob("*")):
        if any(
            part in {".cache", "__pycache__"} for part in file.relative_to(root).parts
        ):
            continue
        if file.is_symlink():
            raise ValueError("Model snapshot contains a symlink")
        if file.is_file():
            result[file.relative_to(root).as_posix()] = sha_file(file)
    return result


def _panel(path: Path, key: str) -> dict[str, Any]:
    items, answers, digest = PANELS[key]
    if sha_file(path) != digest:
        raise ValueError(f"{key}: prompt hash changed")
    rows = load_gold_free(path)
    if len(rows) != items or sum(len(row["questions"]) for row in rows) != answers:
        raise ValueError(f"{key}: item or answer count changed")
    return {"path": str(path), "sha256": digest, "items": items, "answers": answers}


def _passed_parity(path: Path, items: int) -> str:
    receipt = _object(path)
    if (
        receipt.get("passed") is not True
        or receipt.get("items") != items
        or receipt.get("invalid_or_missing_n") != 0
        or receipt.get("categorical_mismatch_n") != 0
        or receipt.get("max_probability_or_score_drift", 1) > 1e-4
        or receipt.get("package_manifest_sha256") not in (None, PACKAGE_SHA)
    ):
        raise ValueError("Unmerged source/package parity gate failed")
    return sha_file(path)


def _repeat(prompt: Path, first: Path, second: Path, *, own: bool) -> dict[str, Any]:
    result = compare_smoke(
        prompt,
        first,
        second,
        model_id=OWN_ID if own else None,
        revision=OWN_REVISION if own else PEER_REVISION,
        backend="sol" if own else "decider",
        adapter_version=BASELINE_ADAPTER_VERSION,
    )
    if result["gate_pass"] is not True or result.get("stable_invalid_n", 0):
        raise ValueError("Comparator two-process repeatability failed")
    return {
        "first_sha256": sha_file(first),
        "second_sha256": sha_file(second),
        "max_probability_drift": result["max_option_probability_drift"],
        "category_changes": result["categorical_mismatch_n"],
    }


def _code(root: Path) -> dict[str, str]:
    return {name: sha_file(root / name) for name in SOURCE_FILES}


def lock(config: Path, output: Path) -> str:
    cfg = _object(config)
    source_root = _path(cfg, "source_root")
    package = _path(cfg, "package")
    source = _path(cfg, "own1_source")
    peer = _path(cfg, "peer")
    if sha_file(package / "MODEL_MANIFEST.json") != PACKAGE_SHA:
        raise ValueError("Candidate package differs from qualified package")
    _verify_staged_runtime(package, source)
    manifest = _object(package / "MODEL_MANIFEST.json")
    if (
        manifest.get("model_id") != CANDIDATE_ID
        or manifest.get("model_sha256") != CANDIDATE_SHA
        or manifest.get("parameter_count") != PARAMETERS
        or manifest.get("calibration_sha256") != CAL_SHA
        or manifest.get("base", {}).get("repo_id") != OWN_ID
        or manifest["base"].get("revision") != OWN_REVISION
    ):
        raise ValueError("Candidate lineage, weights or CAL changed")
    if not local_revision(source, OWN_REVISION) or not local_revision(
        peer, PEER_REVISION
    ):
        raise ValueError("Comparator HF snapshot revision is unattested")
    # The package verifier checks the complete own-source file inventory and
    # allows only import-generated bytecode. Reuse its attested inventory here.
    own_files, peer_files = manifest["base"]["files_sha256"], _files(peer)
    if "model.safetensors" not in peer_files or "decider/infer.py" not in peer_files:
        raise ValueError("Native Decider peer files are incomplete")
    smoke = _path(cfg, "smoke_prompts")
    if sha_file(smoke) != SMOKE_SHA:
        raise ValueError("Gold-free smoke roster changed")
    parity = {
        "package32": _passed_parity(_path(cfg, "package_parity"), 32),
        "dev1600": _passed_parity(_path(cfg, "dev_parity"), 1600),
        "css1430": _passed_parity(_path(cfg, "css_parity"), 1430),
    }
    own_repeat = _repeat(
        smoke, _path(cfg, "own_smoke_first"), _path(cfg, "own_smoke_second"), own=True
    )
    peer_repeat = _repeat(
        smoke,
        _path(cfg, "peer_smoke_first"),
        _path(cfg, "peer_smoke_second"),
        own=False,
    )
    panels = {key: _panel(_path(cfg, f"{key}_prompts"), key) for key in PANELS}
    output_dir = Path(cfg["prediction_dir"])
    if not output_dir.is_absolute() or output_dir.exists() or output_dir.is_symlink():
        raise ValueError("Prediction directory must be an absent absolute path")
    if stat.S_IMODE(output_dir.parent.stat().st_mode) != 0o700:
        raise ValueError("Prediction parent must be mode 0700")
    digest_peer = hashlib.sha256(
        json.dumps(peer_files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    value = {
        "schema_version": LOCK_VERSION,
        "status": "locked_before_formal_prediction; not_a_release_pass",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
        "label_exposure": "prospective_post_key_same_panel; v3 labels previously accessed in unrelated work; not virgin blind",
        "model": {
            "id": CANDIDATE_ID,
            "revision": f"package-sha256:{PACKAGE_SHA}",
            "package": str(package),
            "package_manifest_sha256": PACKAGE_SHA,
            "direct_source_id": OWN_ID,
            "direct_source_revision": OWN_REVISION,
            "checkpoint_sha256": CANDIDATE_SHA,
            "calibration_sha256": CAL_SHA,
            "loaded_parameters": PARAMETERS,
            "parity_sha256": parity,
        },
        "comparators": {
            "own1": {
                "id": OWN_ID,
                "revision": OWN_REVISION,
                "model_dir": str(source),
                "files_sha256": hashlib.sha256(
                    json.dumps(
                        own_files, sort_keys=True, separators=(",", ":")
                    ).encode()
                ).hexdigest(),
                "repeat": own_repeat,
            },
            "decider2b": {
                "id": PEER_ID,
                "revision": PEER_REVISION,
                "model_dir": str(peer),
                "files_sha256": digest_peer,
                "repeat": peer_repeat,
                "native_adapter": "decider.infer.Decider.system_one; eager BF16; per-type release temperatures",
            },
        },
        "panels": panels,
        "prediction_dir": str(output_dir),
        "runtime": {
            "container_digest": cfg["container_digest"],
            "source_sha256": _code(source_root),
        },
        "scoring": {
            "version": "JevArena v3",
            "formula": "100*sqrt(T*H)",
            "T": "typed FINAL four-family macro accuracy with invalid as wrong",
            "H": "15-task median macro-F1 with invalid as wrong",
            "minimum_own1_aggregate_gain_points": 2.0,
            "paired_ci95_lower_bound_must_exceed_zero": True,
            "bootstrap_replicates": 5000,
            "bootstrap_seed": 20260927,
            "public231_role": "separate non-blind public diagnostic; no contribution to v3",
        },
    }
    return _write_private(output, value)


def seal(lock_path: Path, expected_lock_sha: str, output: Path) -> str:
    if sha_file(lock_path) != expected_lock_sha:
        raise ValueError("Prospective lock changed")
    frozen = _object(lock_path)
    if frozen.get("schema_version") != LOCK_VERSION:
        raise ValueError("Unknown candidate lock")
    package = Path(frozen["model"]["package"])
    source = Path(frozen["comparators"]["own1"]["model_dir"])
    if sha_file(package / "MODEL_MANIFEST.json") != PACKAGE_SHA:
        raise ValueError("Candidate package mutated")
    _verify_staged_runtime(package, source)
    source_root = Path(__file__).resolve().parents[1]
    if _code(source_root) != frozen["runtime"]["source_sha256"]:
        raise ValueError("Formal inference/scoring source mutated")
    panel_rows = {}
    for key, panel in frozen["panels"].items():
        path = Path(panel["path"])
        if _panel(path, key) != panel:
            raise ValueError("Formal panel changed")
        panel_rows[key] = load_gold_free(path)
    predictions = {}
    for model_key in ("candidate", "own1", "decider2b"):
        predictions[model_key] = {}
        for panel, prompts in panel_rows.items():
            path = (
                Path(frozen["prediction_dir"])
                / f"{model_key}.{panel}.predictions.jsonl"
            )
            rows = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]
            if len(rows) != len(prompts):
                raise ValueError("Missing formal prediction rows")
            for question, answer in zip(prompts, rows):
                expected_input = (
                    input_digest(question)
                    if model_key == "candidate"
                    else baseline_digest(
                        {"state": question["state"], "questions": question["questions"]}
                    )
                )
                if (
                    answer.get("id") != question["id"]
                    or answer.get("source_input_sha256") != expected_input
                    or set(answer.get("answers", {})) != set(question["questions"])
                ):
                    raise ValueError("Formal prediction does not match frozen input")
                if model_key == "candidate":
                    if (
                        answer.get("model_id") != CANDIDATE_ID
                        or answer.get("model_revision")
                        != f"package-sha256:{PACKAGE_SHA}"
                        or answer.get("package_manifest_sha256") != PACKAGE_SHA
                        or answer.get("adapter_version") != PACKAGE_ADAPTER_VERSION
                    ):
                        raise ValueError("Candidate prediction identity changed")
                else:
                    comparator = frozen["comparators"][model_key]
                    if (
                        answer.get("model_id")
                        != (OWN_ID if model_key == "own1" else None)
                        or answer.get("model_revision") != comparator["revision"]
                        or answer.get("backend")
                        != ("sol" if model_key == "own1" else "decider")
                        or answer.get("adapter_version") != BASELINE_ADAPTER_VERSION
                        or answer.get("revision_attested") is not True
                    ):
                        raise ValueError("Comparator prediction identity changed")
            manifest_path = Path(str(path) + ".manifest.json")
            native_manifest_sha = None
            if model_key == "candidate":
                manifest = _object(manifest_path)
                if (
                    manifest.get("package_manifest_sha256") != PACKAGE_SHA
                    or manifest.get("predictions_sha256") != sha_file(path)
                    or manifest.get("input_sha256") != frozen["panels"][panel]["sha256"]
                ):
                    raise ValueError("Package-native scored manifest changed")
                native_manifest_sha = sha_file(manifest_path)
            predictions[model_key][panel] = {
                "items": len(rows),
                "answers": sum(len(row["answers"]) for row in rows),
                "predictions_sha256": sha_file(path),
                "native_manifest_sha256": native_manifest_sha,
            }
    value = {
        "schema_version": SEAL_VERSION,
        "status": "all_gold_free_predictions_sealed; not_scored_or_release_qualified",
        "sealed_at_utc": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
        "candidate_lock_sha256": expected_lock_sha,
        "label_exposure": frozen["label_exposure"],
        "predictions": predictions,
    }
    return _write_private(output, value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    first = commands.add_parser("lock")
    first.add_argument("--config", type=Path, required=True)
    first.add_argument("--output", type=Path, required=True)
    second = commands.add_parser("seal")
    second.add_argument("--lock", type=Path, required=True)
    second.add_argument("--expected-lock-sha256", required=True)
    second.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    digest = (
        lock(args.config, args.output)
        if args.command == "lock"
        else seal(args.lock, args.expected_lock_sha256, args.output)
    )
    print(
        json.dumps(
            {
                "schema_version": (
                    LOCK_VERSION if args.command == "lock" else SEAL_VERSION
                ),
                "sha256": digest,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
