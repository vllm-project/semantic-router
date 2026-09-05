"""Cross-reference validation across a Router Model provenance bundle.

Schema validation proves each manifest is well formed. It does not prove the
four manifests describe the same model. These checks close that gap: every
reference must resolve, the artifact digest must be reproducible from the file
list, and the label mapping must be identical everywhere it appears.

The label-mapping check is the one that matters most in practice. A checkpoint
whose class order differs from the order the evaluation harness assumed still
produces plausible-looking accuracy numbers, so the mismatch is invisible in
metrics and only a cross-reference can catch it.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .manifest import ManifestError, load_manifests

__all__ = ["artifact_identity_digest", "validate_bundle"]


def artifact_identity_digest(files: list[dict[str, Any]]) -> str:
    """Digest the sorted (path, digest) file list into one immutable identity.

    Runtime qualification and model cards can compare this single value instead
    of walking the file list.
    """
    hasher = hashlib.sha256()
    for entry in sorted(files, key=lambda item: str(item.get("path", ""))):
        hasher.update(str(entry.get("path", "")).encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(str(entry.get("digest", "")).encode("utf-8"))
        hasher.update(b"\0")
    return f"sha256:{hasher.hexdigest()}"


def validate_bundle(directory: Path) -> dict[str, Any]:
    """Validate every manifest under ``directory`` and their mutual references.

    Returns a summary of what was checked. Raises :class:`ManifestError` listing
    every problem found, so one run reports the full set rather than the first.
    """
    grouped = load_manifests(directory)
    problems: list[str] = []

    datasets = _index(grouped["dataset"], problems)
    runs = _index(grouped["run"], problems)
    artifacts = _index(grouped["artifact"], problems)
    evaluations = _index(grouped["evaluation"], problems)

    for path, artifact in artifacts.values():
        _check_artifact_digest(path, artifact, problems)
        _check_run_ref(path, artifact, runs, problems)

    for path, run in runs.values():
        _check_dataset_refs(path, run, datasets, problems)

    for path, evaluation in evaluations.values():
        _check_evaluation_refs(path, evaluation, datasets, artifacts, problems)

    if not evaluations:
        problems.append(
            f"{directory} contains no evaluation manifest; an artifact without a "
            "measured evaluation is not qualified"
        )

    if problems:
        raise ManifestError(
            f"{directory} failed cross-reference validation:\n  - "
            + "\n  - ".join(problems)
        )

    return {
        "directory": str(directory),
        "datasets": sorted(datasets),
        "runs": sorted(runs),
        "artifacts": sorted(artifacts),
        "evaluations": sorted(evaluations),
    }


def _index(
    entries: list[tuple[Path, dict[str, Any]]], problems: list[str]
) -> dict[str, tuple[Path, dict[str, Any]]]:
    indexed: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path, manifest in entries:
        identifier = str(manifest["id"])
        if identifier in indexed:
            problems.append(
                f"{path} reuses id {identifier!r} already declared by "
                f"{indexed[identifier][0]}"
            )
            continue
        indexed[identifier] = (path, manifest)
    return indexed


def _check_artifact_digest(
    path: Path, artifact: dict[str, Any], problems: list[str]
) -> None:
    declared = artifact["identity"]["digest"]
    recomputed = artifact_identity_digest(artifact["files"])
    if declared != recomputed:
        problems.append(
            f"{path} identity.digest is {declared}, but the file list hashes to "
            f"{recomputed}"
        )
    num_labels = artifact["runtime"]["num_labels"]
    if num_labels != len(artifact["label_mapping"]):
        problems.append(
            f"{path} runtime.num_labels is {num_labels} but label_mapping declares "
            f"{len(artifact['label_mapping'])} labels"
        )


def _check_run_ref(
    path: Path,
    artifact: dict[str, Any],
    runs: dict[str, tuple[Path, dict[str, Any]]],
    problems: list[str],
) -> None:
    run_ref = artifact.get("run_ref")
    if not run_ref:
        problems.append(
            f"{path} has no run_ref; an artifact with no training run cannot be "
            "traced back to its data"
        )
        return
    resolved = runs.get(run_ref["id"])
    if resolved is None:
        problems.append(f"{path} run_ref {run_ref['id']!r} resolves to no run manifest")
        return
    run_path, run = resolved
    _compare_task(path, artifact, run_path, run, problems)
    _compare_label_mapping(path, artifact, run_path, run, problems)


def _check_dataset_refs(
    path: Path,
    run: dict[str, Any],
    datasets: dict[str, tuple[Path, dict[str, Any]]],
    problems: list[str],
) -> None:
    for ref in run["dataset_refs"]:
        resolved = datasets.get(ref["id"])
        if resolved is None:
            problems.append(
                f"{path} dataset_refs entry {ref['id']!r} resolves to no dataset manifest"
            )
            continue
        dataset_path, dataset = resolved
        _compare_revision(path, "dataset_refs", ref, dataset_path, dataset, problems)
        _compare_task(path, run, dataset_path, dataset, problems)
        _compare_label_mapping(path, run, dataset_path, dataset, problems)
        _check_splits_exist(path, ref, dataset_path, dataset, problems)


def _check_evaluation_refs(
    path: Path,
    evaluation: dict[str, Any],
    datasets: dict[str, tuple[Path, dict[str, Any]]],
    artifacts: dict[str, tuple[Path, dict[str, Any]]],
    problems: list[str],
) -> None:
    dataset_ref = evaluation["dataset_ref"]
    resolved_dataset = datasets.get(dataset_ref["id"])
    if resolved_dataset is None:
        problems.append(
            f"{path} dataset_ref {dataset_ref['id']!r} resolves to no dataset manifest"
        )
    else:
        dataset_path, dataset = resolved_dataset
        _compare_revision(
            path, "dataset_ref", dataset_ref, dataset_path, dataset, problems
        )
        _compare_task(path, evaluation, dataset_path, dataset, problems)
        _compare_label_mapping(path, evaluation, dataset_path, dataset, problems)
        _check_splits_exist(path, dataset_ref, dataset_path, dataset, problems)
        _check_row_budget(path, evaluation, dataset_ref, dataset, problems)

    artifact_ref = evaluation["artifact_ref"]
    resolved_artifact = artifacts.get(artifact_ref["id"])
    if resolved_artifact is None:
        problems.append(
            f"{path} artifact_ref {artifact_ref['id']!r} resolves to no artifact manifest"
        )
        return
    artifact_path, artifact = resolved_artifact
    identity = artifact["identity"]
    if artifact_ref["revision"] != identity["revision"]:
        problems.append(
            f"{path} artifact_ref.revision {artifact_ref['revision']} does not match "
            f"{artifact_path} identity.revision {identity['revision']}"
        )
    if artifact_ref["digest"] != identity["digest"]:
        problems.append(
            f"{path} artifact_ref.digest {artifact_ref['digest']} does not match "
            f"{artifact_path} identity.digest {identity['digest']}"
        )
    _compare_task(path, evaluation, artifact_path, artifact, problems)
    _compare_label_mapping(path, evaluation, artifact_path, artifact, problems)
    _check_metric_labels(path, evaluation, problems)


def _compare_task(
    left_path: Path,
    left: dict[str, Any],
    right_path: Path,
    right: dict[str, Any],
    problems: list[str],
) -> None:
    if left["task"] != right["task"]:
        problems.append(
            f"{left_path} task {left['task']!r} does not match {right_path} task "
            f"{right['task']!r}"
        )


def _compare_label_mapping(
    left_path: Path,
    left: dict[str, Any],
    right_path: Path,
    right: dict[str, Any],
    problems: list[str],
) -> None:
    left_mapping = left.get("label_mapping")
    right_mapping = right.get("label_mapping")
    if not left_mapping or not right_mapping or left_mapping == right_mapping:
        return
    differences = sorted(
        set(left_mapping.items()) ^ set(right_mapping.items()),
        key=lambda item: (item[1], item[0]),
    )
    rendered = ", ".join(f"{name}={index}" for name, index in differences[:8])
    problems.append(
        f"{left_path} label_mapping differs from {right_path}; class order is part "
        f"of the contract and a permuted order silently scores the wrong class "
        f"(differing entries: {rendered})"
    )


def _compare_revision(
    path: Path,
    field: str,
    ref: dict[str, Any],
    dataset_path: Path,
    dataset: dict[str, Any],
    problems: list[str],
) -> None:
    declared = dataset["source"]["revision"]
    if ref["revision"] != declared:
        problems.append(
            f"{path} {field}.revision {ref['revision']} does not match {dataset_path} "
            f"source.revision {declared}"
        )


def _check_splits_exist(
    path: Path,
    ref: dict[str, Any],
    dataset_path: Path,
    dataset: dict[str, Any],
    problems: list[str],
) -> None:
    known = {split["name"] for split in dataset["splits"]}
    for name in ref.get("splits", []):
        if name not in known:
            problems.append(
                f"{path} references split {name!r} which {dataset_path} does not "
                f"declare (known splits: {', '.join(sorted(known))})"
            )


def _check_row_budget(
    path: Path,
    evaluation: dict[str, Any],
    ref: dict[str, Any],
    dataset: dict[str, Any],
    problems: list[str],
) -> None:
    """Catch a measured row count that the referenced split cannot supply."""
    names = ref.get("splits") or []
    if len(names) != 1:
        return
    available = next(
        (split["rows"] for split in dataset["splits"] if split["name"] == names[0]),
        None,
    )
    if available is None:
        return
    measured = evaluation["metrics"]["rows"]
    if measured > available:
        problems.append(
            f"{path} reports {measured} evaluated rows but split {names[0]!r} holds "
            f"only {available}"
        )
    limit = evaluation["harness"]["sample_limit"]
    if limit is None and measured < available:
        problems.append(
            f"{path} declares no sample_limit but measured {measured} of "
            f"{available} rows; an unexplained subset is not a baseline"
        )


def _check_metric_labels(
    path: Path, evaluation: dict[str, Any], problems: list[str]
) -> None:
    declared = set(evaluation["label_mapping"])
    reported = set(evaluation["metrics"]["per_label"])
    missing = sorted(declared - reported)
    extra = sorted(reported - declared)
    if missing:
        problems.append(
            f"{path} metrics.per_label omits declared labels: {', '.join(missing)}"
        )
    if extra:
        problems.append(
            f"{path} metrics.per_label reports undeclared labels: {', '.join(extra)}"
        )
