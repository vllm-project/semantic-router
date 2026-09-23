"""Bridge classifier profiles to the existing strict v1 provenance validator."""

from pathlib import Path

from src.training.model_eval.provenance.crossref import validate_bundle
from src.training.model_eval.provenance.manifest import ManifestError, load_manifests


def validate_classifier_provenance(
    directory: Path, profile: dict, base_model: dict
) -> dict:
    """Validate a worker-owned bundle and its link to the frozen classifier profile and run model.

    ``directory`` is resolved privately by the worker adapter from owned handles;
    it is never a management API request field.
    """
    summary = validate_bundle(directory)
    classifier = profile["classifier"]
    manifests = load_manifests(directory)
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
    return summary
