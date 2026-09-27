"""Lock and seal the official-Qwen 0.6B post-key, gold-free v3 comparison.

The previous-generation and open-peer predictions were already sealed for the
identical panels. This tool verifies their original seal rather than rerunning
them, and locks the selected full checkpoint before new formal inference.
It accepts no gold path and computes no score.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from inference.run import digest, file_digest, load_prompts
from training.model.data import canonical

PANELS = ("typed", "css", "public")
CONTROLS = {"previous_generation": "kai1", "index_selected_peer": "bosun"}
SCHEMA = "decision2-official-qwen06b-v3-postkey-lock/1"
SEAL_SCHEMA = "decision2-official-qwen06b-v3-postkey-prediction-seal/1"


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path.name}")
    return value


def _exclusive_json(path: Path, value: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o600)


def _prompts(roster: dict[str, Any], paths: dict[str, Path]) -> dict[str, list]:
    panels = {}
    for panel in PANELS:
        config = roster["panel"][panel]
        if file_digest(paths[panel]) != config["prompts_sha256"]:
            raise ValueError(f"{panel} prompt hash differs")
        rows = load_prompts(paths[panel])
        if (
            len(rows) != config["items"]
            or sum(len(row["questions"]) for row in rows) != config["answer_slots"]
        ):
            raise ValueError(f"{panel} item or answer count differs")
        panels[panel] = rows
    return panels


def _source(roster: dict[str, Any], source_root: Path) -> None:
    for relative, expected in roster["scoring_sources_sha256"].items():
        if file_digest(source_root / relative) != expected:
            raise ValueError(f"Scorer changed: {relative}")


def _control_rows(
    path: Path, prompts: list[dict], model: dict[str, Any]
) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ValueError("Control prediction is missing or linked")
    expected = {
        row["id"]: (
            digest({"state": row["state"], "questions": row["questions"]}),
            set(row["questions"]),
        )
        for row in prompts
    }
    seen = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        item_id = row.get("id")
        if item_id not in expected or item_id in seen:
            raise ValueError("Control prediction has an unknown or duplicate ID")
        source_hash, questions = expected[item_id]
        if (
            row.get("model_id") != model["model_id"]
            or row.get("model_revision") != model["weight_revision"]
            or row.get("adapter_version") != model["adapter_version"]
            or row.get("source_input_sha256") != source_hash
            or not isinstance(row.get("answers"), dict)
            or set(row["answers"]) != questions
        ):
            raise ValueError("Control prediction provenance differs")
        seen.add(item_id)
    if seen != set(expected):
        raise ValueError("Control prediction is incomplete")
    return {"sha256": file_digest(path), "items": len(seen)}


def _controls(
    roster: dict[str, Any],
    prior_roster: Path,
    prior_seal: Path,
    prior_predictions: Path,
    panels: dict[str, list],
) -> dict[str, dict[str, Any]]:
    if (
        file_digest(prior_roster) != roster["reused_comparator_roster_sha256"]
        or file_digest(prior_seal) != roster["reused_comparator_prediction_seal_sha256"]
    ):
        raise ValueError("Archived comparator roster or seal changed")
    sealed = _object(prior_seal)
    if sealed.get("roster_sha256") != file_digest(prior_roster):
        raise ValueError("Archived seal does not bind its roster")
    result = {}
    for role, stem in CONTROLS.items():
        model = roster["comparators"][role]
        result[stem] = {}
        for panel in PANELS:
            expected = model["prediction_sha256"][panel]
            prior = sealed["predictions"][stem][panel]
            if prior["sha256"] != expected:
                raise ValueError("Archived prediction does not match prior seal")
            path = prior_predictions / f"{stem}.{panel}.jsonl"
            observed = _control_rows(path, panels[panel], model)
            if observed["sha256"] != expected:
                raise ValueError("Archived prediction bytes changed")
            result[stem][panel] = observed
    return result


def _checkpoint(
    roster: dict[str, Any],
    checkpoint: Path,
    development_manifest: Path,
    calibration: Path,
    best: Path,
    complete: Path,
) -> dict[str, Any]:
    model = roster["candidate"]
    if (
        file_digest(best) != model["best_sha256"]
        or file_digest(complete) != model["complete_sha256"]
        or _object(best).get("checkpoint") != f"checkpoint-{model['best_step']:07d}"
        or _object(complete).get("best") != f"checkpoint-{model['best_step']:07d}"
    ):
        raise ValueError("Selected BEST checkpoint changed")
    if file_digest(calibration) != model["calibration_sha256"]:
        raise ValueError("Selected CAL fit changed")
    fit = _object(calibration)
    if fit.get("model_sha256") != model["model_sha256"]:
        raise ValueError("CAL fit belongs to another model")
    manifest = _object(development_manifest)
    for field in ("model_id", "model_revision", "model_sha256", "adapter_sha256"):
        if manifest.get(field) != model[field]:
            raise ValueError(f"Development adapter {field} changed")
    if (
        manifest.get("adapter_version") != model["adapter_version"]
        or manifest.get("calibration", {}).get("file_sha256")
        != model["calibration_sha256"]
        or manifest.get("max_length") != model["max_length"]
    ):
        raise ValueError("Development adapter or calibration changed")
    hashes = manifest.get("model_files_sha256")
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("Development manifest has no model file identity")
    for relative, expected in hashes.items():
        name = Path(relative)
        if name.is_absolute() or ".." in name.parts:
            raise ValueError("Unsafe model file path")
        if file_digest(checkpoint / name) != expected:
            raise ValueError("Selected model file changed")
    calculated = hashlib.sha256(canonical(hashes).encode()).hexdigest()
    if calculated != model["model_sha256"]:
        raise ValueError("Selected model fingerprint changed")
    return {"model_sha256": calculated, "files_sha256": hashes}


def lock(args: argparse.Namespace) -> dict[str, Any]:
    roster = _object(args.roster)
    if (
        roster.get("schema") != "decision2-official-qwen06b-v3-postkey-roster/1"
        or roster.get("not_a_never_unsealed_blind_test") is not True
    ):
        raise ValueError("Wrong roster or label-access status")
    _source(roster, args.source_root)
    paths = {name: getattr(args, f"{name}_prompts") for name in PANELS}
    panels = _prompts(roster, paths)
    if (
        file_digest(args.public_manifest)
        != roster["panel"]["public"]["manifest_sha256"]
    ):
        raise ValueError("Public panel manifest changed")
    if file_digest(args.overlap_audit) != roster["overlap_audit_sha256"]:
        raise ValueError("TRAIN/panel overlap audit changed")
    audit = _object(args.overlap_audit)
    for panel in ("typed_final", "css15", "public231"):
        if any(audit["panels"][panel]["counts"].values()):
            raise ValueError("TRAIN overlaps a formal panel")
    model = _checkpoint(
        roster,
        args.checkpoint,
        args.development_manifest,
        args.calibration,
        args.best,
        args.complete,
    )
    controls = _controls(
        roster,
        args.prior_roster,
        args.prior_seal,
        args.prior_predictions,
        panels,
    )
    result = {
        "schema": SCHEMA,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "post_key_same_panel": True,
        "roster_sha256": file_digest(args.roster),
        "code_sha256": file_digest(Path(__file__)),
        "model": model,
        "calibration_sha256": file_digest(args.calibration),
        "overlap_audit_sha256": file_digest(args.overlap_audit),
        "prompts_sha256": {name: file_digest(path) for name, path in paths.items()},
        "controls": controls,
        "scoring_sources_sha256": roster["scoring_sources_sha256"],
        "claim": "candidate locked before new formal predictions; labels previously accessed elsewhere in project",
    }
    _exclusive_json(args.output, result)
    return result


def _candidate_rows(
    path: Path,
    prompts: list[dict],
    model: dict[str, Any],
    prompt_sha256: str,
) -> dict[str, Any]:
    manifest_path = path.with_name(path.name + ".manifest.json")
    manifest = _object(manifest_path)
    expected_manifest = {
        "model_id": model["model_id"],
        "model_revision": model["model_revision"],
        "model_sha256": model["model_sha256"],
        "adapter_sha256": model["adapter_sha256"],
        "adapter_version": model["adapter_version"],
        "input_sha256": prompt_sha256,
        "max_length": model["max_length"],
    }
    for field, expected in expected_manifest.items():
        if manifest.get(field) != expected:
            raise ValueError(f"Candidate prediction manifest changed: {field}")
    if (
        manifest.get("calibration", {}).get("file_sha256")
        != model["calibration_sha256"]
        or file_digest(path) != manifest.get("predictions_sha256")
        or manifest.get("input_items") != len(prompts)
    ):
        raise ValueError("Candidate prediction/calibration/panel bytes changed")
    expected = {
        row["id"]: (
            digest({"state": row["state"], "questions": row["questions"]}),
            set(row["questions"]),
        )
        for row in prompts
    }
    seen = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        item_id = row.get("id")
        if item_id not in expected or item_id in seen:
            raise ValueError("Candidate prediction has unknown or duplicate ID")
        source_hash, questions = expected[item_id]
        if (
            row.get("model_sha256") != model["model_sha256"]
            or row.get("adapter_sha256") != model["adapter_sha256"]
            or row.get("calibration_sha256") != model["calibration_sha256"]
            or row.get("source_input_sha256") != source_hash
            or not isinstance(row.get("answers"), dict)
            or set(row["answers"]) != questions
        ):
            raise ValueError("Candidate prediction provenance or answers changed")
        seen.add(item_id)
    if seen != set(expected):
        raise ValueError("Candidate prediction is incomplete")
    return {
        "sha256": file_digest(path),
        "manifest_sha256": file_digest(manifest_path),
        "items": len(seen),
        "answer_slots": sum(len(row["questions"]) for row in prompts),
        "counts": manifest["counts"],
    }


def seal(args: argparse.Namespace) -> dict[str, Any]:
    if file_digest(args.lock) != args.lock_sha256:
        raise ValueError("Candidate lock bytes changed")
    frozen = _object(args.lock)
    roster = _object(args.roster)
    if (
        frozen.get("schema") != SCHEMA
        or frozen.get("roster_sha256") != file_digest(args.roster)
        or frozen.get("code_sha256") != file_digest(Path(__file__))
    ):
        raise ValueError("Candidate lock or sealing code changed")
    _source(roster, args.source_root)
    paths = {name: getattr(args, f"{name}_prompts") for name in PANELS}
    panels = _prompts(roster, paths)
    if frozen["prompts_sha256"] != {
        name: file_digest(path) for name, path in paths.items()
    }:
        raise ValueError("Formal prompts changed after lock")
    controls = _controls(
        roster,
        args.prior_roster,
        args.prior_seal,
        args.prior_predictions,
        panels,
    )
    if controls != frozen["controls"]:
        raise ValueError("Comparator predictions changed after lock")
    candidate = {
        panel: _candidate_rows(
            args.predictions / f"qwen.{panel}.jsonl",
            panels[panel],
            roster["candidate"],
            frozen["prompts_sha256"][panel],
        )
        for panel in PANELS
    }
    result = {
        "schema": SEAL_SCHEMA,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "post_key_same_panel": True,
        "candidate_lock_sha256": file_digest(args.lock),
        "roster_sha256": file_digest(args.roster),
        "code_sha256": file_digest(Path(__file__)),
        "prompts_sha256": frozen["prompts_sha256"],
        "candidate": candidate,
        "controls": controls,
        "claim": "all three candidate predictions sealed before this run's formal scoring; controls reused from prior sealed same-panel run",
    }
    _exclusive_json(args.output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for name in ("lock", "seal"):
        command = sub.add_parser(name)
        for field in (
            "roster",
            "source_root",
            "typed_prompts",
            "css_prompts",
            "public_prompts",
            "prior_roster",
            "prior_seal",
            "prior_predictions",
            "output",
        ):
            command.add_argument(
                "--" + field.replace("_", "-"), required=True, type=Path
            )
        if name == "lock":
            for field in (
                "public_manifest",
                "overlap_audit",
                "checkpoint",
                "development_manifest",
                "calibration",
                "best",
                "complete",
            ):
                command.add_argument(
                    "--" + field.replace("_", "-"), required=True, type=Path
                )
        else:
            command.add_argument("--lock", required=True, type=Path)
            command.add_argument("--lock-sha256", required=True)
            command.add_argument("--predictions", required=True, type=Path)
    args = parser.parse_args()
    result = lock(args) if args.action == "lock" else seal(args)
    print(
        json.dumps(
            {"schema": result["schema"], "sha256": file_digest(args.output)},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
