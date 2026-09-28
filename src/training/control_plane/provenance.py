"""Bridge classifier profiles to the existing strict v1 provenance validator."""

from collections.abc import Callable
from pathlib import Path

from src.training.model_eval.provenance.crossref import (
    file_digest,
    load_validated_bundle,
)
from src.training.model_eval.provenance.manifest import ManifestError


def validate_classifier_provenance(
    directory: Path,
    profile: dict,
    base_model: dict,
    variant: dict,
    resolve_file: Callable[[str], Path],
) -> dict:
    """Validate a worker-owned bundle and its link to the frozen classifier profile and run model.

    ``directory`` and ``resolve_file`` are provided privately by the worker adapter
    from owned handles, never management API request paths. The resolver returns
    immutable owned files for the schema-validated variant being qualified.
    """
    summary, manifests = load_validated_bundle(directory)
    classifier = profile["classifier"]
    for entries in manifests.values():
        for _, manifest in entries:
            if (
                "label_mapping" in manifest
                and manifest["label_mapping"] != classifier["label_mapping"]
            ):
                raise ManifestError("profile and manifest label_mapping differ")
            if manifest["kind"] == "run":
                expected = base_model
                if manifest["base_model"] != {
                    "repo": expected["repository"],
                    "revision": expected["revision"],
                }:
                    raise ManifestError("run and manifest base_model differ")
    _validate_variant_files(manifests, variant, resolve_file)
    return summary


def _validate_variant_files(
    manifests: dict, variant: dict, resolve_file: Callable[[str], Path]
) -> None:
    files = variant["files"]
    expected = {
        name: (entry["digest"], entry["size_bytes"]) for name, entry in files.items()
    }
    evaluated_ids = {
        manifest["artifact_ref"]["id"] for _, manifest in manifests["evaluation"]
    }
    for _, manifest in manifests["artifact"]:
        declared = {
            entry["path"]: (entry["digest"], entry["size_bytes"])
            for entry in manifest["files"]
        }
        if manifest["id"] in evaluated_ids and declared == expected:
            break
    else:
        raise ManifestError("variant files do not match an evaluated artifact manifest")

    for name, entry in files.items():
        path = resolve_file(entry["handle"])
        if not path.is_file():
            raise ManifestError(f"owned variant file {name} is missing")
        if (
            path.stat().st_size != entry["size_bytes"]
            or file_digest(path) != entry["digest"]
        ):
            raise ManifestError(f"owned variant file {name} differs from its manifest")
