#!/usr/bin/env python3
"""Run and attest the production image scorer and its authored threshold gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shlex
import subprocess
from pathlib import Path

import yaml
from ci_results import actual_platform, collection_errors
from workflow_evidence import go_cases

ROOT = Path(__file__).resolve().parents[2]
CASES = Path("tools/calibration/image-routing/testdata/calibration-set.json")
RULES = Path("config/fragments/signal/embedding/image-routing.yaml")
OMNI_MANIFEST = "vela_omni_manifest.json"


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def file_sha(path: Path) -> str:
    with path.open("rb") as stream:
        return "sha256:" + hashlib.file_digest(stream, "sha256").hexdigest()


def model_identity(manifest: dict) -> tuple[dict, dict[str, str]]:
    models = manifest.get("models", [])
    if manifest.get("provider") != "ort" or len(models) != 1:
        raise ValueError("image calibration requires one ORT Omni model")
    model = models[0]
    if (
        model.get("name") != "Multimodal"
        or model.get("env") != "MULTIMODAL_MODEL_PATH"
        or not re.fullmatch(r"[0-9a-f]{40}", model.get("revision", ""))
    ):
        raise ValueError("image calibration model identity is invalid")
    directory = Path(model["path"]).resolve()
    artifact = read(directory / OMNI_MANIFEST)
    if (
        artifact.get("format_version") != 1
        or artifact.get("adapter") != "vela_omni"
        or artifact.get("source")
        != {"repo_id": model["repo_id"], "revision": model["revision"]}
        or not artifact.get("files")
    ):
        raise ValueError("Omni manifest source identity mismatch")
    hashes = {OMNI_MANIFEST: file_sha(directory / OMNI_MANIFEST)}
    for name, expected in artifact["files"].items():
        path = (directory / name).resolve()
        if (
            Path(name).is_absolute()
            or not path.is_relative_to(directory)
            or ".." in Path(name).parts
        ):
            raise ValueError(f"Omni manifest path escape: {name}")
        actual = file_sha(path)
        if actual != "sha256:" + expected:
            raise ValueError(f"Omni manifest checksum mismatch: {name}")
        hashes[name] = actual
    return model, hashes


def prepare_manifest(artifact: Path, destination: Path) -> None:
    source = read(artifact / OMNI_MANIFEST)["source"]
    manifest = {
        "provider": "ort",
        "models": [
            {
                "name": "Multimodal",
                "env": "MULTIMODAL_MODEL_PATH",
                "path": str(artifact.resolve()),
                **source,
            }
        ],
    }
    model_identity(manifest)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2) + "\n")


def indexed(rows: list[dict], key: str, expected: set[str]) -> dict[str, dict]:
    ids = [row[key] for row in rows]
    if len(ids) != len(set(ids)) or set(ids) != expected:
        raise ValueError(f"incomplete or duplicate image calibration {key} inventory")
    return dict(zip(ids, rows, strict=True))


def evidence(directory: Path, *, root: Path = ROOT) -> dict:
    execution = read(directory / "execution.json")
    source = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    if execution.get("source_sha") != source or execution.get("commands") != {
        "profile-discovery": 0,
        "profile-tests": 0,
        "calibration": 0,
    }:
        raise ValueError("image calibration did not complete all source-bound commands")
    report = read(directory / "report.json")
    if (
        report["source"].get("repo_commit") != source
        or report["source"].get("repo_dirty") is not False
    ):
        raise ValueError("image calibration report source differs or is dirty")
    model, hashes = model_identity(read(directory / "models.json"))
    if (
        report["model"]["repository"] != model["repo_id"]
        or report["model"]["artifact_revision"] != model["revision"]
        or report["model"]["artifact_files"] != hashes
    ):
        raise ValueError(
            "image calibration report model bytes differ from the pinned inputs"
        )
    manifest = read(root / CASES)
    rules = yaml.safe_load((root / RULES).read_text())["routing"]["signals"][
        "embeddings"
    ]
    rule_names = {rule["name"] for rule in rules}
    if not rule_names or len(rule_names) != len(rules):
        raise ValueError(
            "image calibration source rule inventory is empty or duplicated"
        )
    positives: dict[str, set[str]] = {}
    for row in manifest["positives"]:
        positives.setdefault(row["image_file"], set()).add(row["signal_name"])
    fixture_names = set(manifest["negatives"]) | set(positives)
    if not fixture_names:
        raise ValueError("image calibration source fixture inventory is empty")
    fixtures = indexed(report["fixtures"], "path", fixture_names)
    cases = []
    for name, row in fixtures.items():
        if row["sha256"] != file_sha(root / name) or set(
            row.get("positive_for", [])
        ) != positives.get(name, set()):
            raise ValueError(
                f"image calibration fixture identity or labels differ: {name}"
            )
        scores = row["scores"]
        if set(scores) != rule_names or any(
            not math.isfinite(value) for value in scores.values()
        ):
            raise ValueError(
                f"image calibration fixture has missing or nonfinite scores: {name}"
            )
        # A scored observation is not a claim that every gold label is separable.
        cases.append({"id": "score/" + name, "status": "passed", "scores": scores})
    indexed(report["rules"], "name", rule_names)
    threshold_ids = {"threshold/" + name for name in rule_names}
    prototype_names = {
        rule["name"]
        for rule in rules
        if rule.get("image_candidates") or rule.get("negative_image_candidates")
    }
    if prototype_names:
        for field, relative in (
            ("prototype_manifest_sha256", "config/assets/image-routing/manifest.json"),
            (
                "prototype_protocol_sha256",
                "tools/calibration/image-routing/testdata/prototype-protocol.json",
            ),
        ):
            if report["source"].get(field) != file_sha(root / relative):
                raise ValueError(
                    "image prototype protocol or assets differ from reported inputs"
                )
        reported_rules = {row["name"]: row for row in report["rules"]}
        for name in prototype_names:
            validation = reported_rules[name].get("validation")
            if (
                not isinstance(validation, dict)
                or validation.get("true_positive", 0) <= 0
                or validation.get("false_positive") != 0
                or validation.get("false_negative") != 0
            ):
                raise ValueError(f"image prototype held-out quality failed: {name}")
        threshold_ids |= {"validation/" + name for name in prototype_names}
    assertions = indexed(report.get("checks", []), "id", threshold_ids)
    cases.extend(
        {"id": name, "status": "passed" if row.get("passed") is True else "failed"}
        for name, row in assertions.items()
    )
    excluded = {
        row["image_file"]: row["reason"] for row in manifest.get("excluded", [])
    }
    provenance = indexed(
        report["source"].get("excluded_fixtures", []), "path", set(excluded)
    )
    for name, row in provenance.items():
        if row["reason"] != excluded[name] or row["sha256"] != file_sha(root / name):
            raise ValueError(f"excluded fixture provenance differs: {name}")
    profile_cases, profile_expected = go_cases(
        directory / "profile-tests.jsonl", directory / "profile-discovery.jsonl", set()
    )
    profile_evidence = {"cases": profile_cases, "expected_cases": profile_expected}
    errors = collection_errors(profile_evidence, "test")
    if errors:
        raise ValueError("; ".join(errors))
    cases.extend({**row, "id": "profile/" + row["id"]} for row in profile_cases)
    expected = [
        *("score/" + name for name in sorted(fixture_names)),
        *sorted(threshold_ids),
        *("profile/" + name for name in profile_expected),
    ]
    return {
        "runtime": "ort",
        "device": "cpu",
        "platform": actual_platform(),
        "cases": cases,
        "expected_cases": expected,
        "models": [{**model, "files": hashes}],
        "excluded_fixtures": list(provenance.values()),
        "contract": "Complete production scoring and authored threshold consistency; not perfect image classification.",
    }


def run(manifest: Path, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    model, _ = model_identity(read(manifest))
    if manifest.resolve() != (output / "models.json").resolve():
        (output / "models.json").write_text(manifest.read_text())
    execution = {
        "source_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "commands": {},
    }
    commands = (
        (
            "profile-discovery",
            ROOT / "e2e",
            [
                "go",
                "test",
                "-json",
                "-list",
                "^(Test|Fuzz|Example)",
                "./profiles/multimodal-routing",
            ],
        ),
        (
            "profile-tests",
            ROOT / "e2e",
            [
                "go",
                "test",
                "-json",
                "-count=1",
                "-timeout=10m",
                "./profiles/multimodal-routing",
            ],
        ),
        (
            "calibration",
            ROOT,
            [
                "make",
                "--no-print-directory",
                "run-image-routing-calibration",
                "GO_TOOL_ARGS="
                + shlex.join(
                    [
                        "-model",
                        model["path"],
                        "-artifact-revision",
                        model["revision"],
                        "-expect-artifact-revision",
                        model["revision"],
                        "-rules",
                        str(ROOT / RULES),
                        "-cases",
                        str(ROOT / CASES),
                        "-fixture-root",
                        str(ROOT),
                        "-output",
                        str(output / "report.json"),
                        "-markdown",
                        str(output / "report.md"),
                        "-check",
                        "-require-clean",
                    ]
                ),
            ],
        ),
    )
    for name, cwd, command in commands:
        print(f"Running image calibration {name}", flush=True)
        filename = name + (".log" if name == "calibration" else ".jsonl")
        with (output / filename).open("w") as log:
            result = subprocess.run(
                command,
                cwd=cwd,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        execution["commands"][name] = result.returncode
        (output / "execution.json").write_text(json.dumps(execution, indent=2) + "\n")
        if result.returncode:
            raise RuntimeError(
                f"{name} failed ({result.returncode}); see {output / filename}"
            )
    normalized = evidence(output)
    errors = collection_errors(normalized, "test")
    if errors:
        raise ValueError("; ".join(errors))
    print(
        f"Image calibration completed {len(normalized['cases'])} required observations/assertions",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--prepare-manifest", action="store_true")
    args = parser.parse_args()
    if args.prepare_manifest:
        if args.artifact is None:
            parser.error("--prepare-manifest requires --artifact")
        prepare_manifest(args.artifact.resolve(), args.manifest.resolve())
    elif args.output is None:
        parser.error("--output is required to run calibration")
    else:
        run(args.manifest.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
