"""Decision Index Engine over a released Decision 2.0 package (``dev2-package/1``).

The kit runner loads it as
``publication.decision_index_release_engine:ReleasedDecisionIndexEngine``.
Every request goes once through the package's own runtime,
``Decision2.from_pretrained(package).system_one(state=..., questions=...)``,
imported from the package directory and nowhere else. The state, question IDs,
instructions, option keys, option order and descriptions reach the package
unchanged; the package encodes all questions of one request in one batch.

Nothing is truncated. A question the package refuses (``max_length_exceeded``
over its ``max_input_tokens``, or ``invalid_question`` for a shape its
``question_to_row`` rejects) makes the whole request ``Unsupported`` with that
reason. Any other defect is an error: a missing or extra answer, another model
identity, a probability outside [0, 1] or not finite, a distribution that does
not sum to 1 within 1e-5, or a chosen key other than the package's argmax
(exact ties within 1e-8 resolve to the first key in the caller's order).

Set ``DECISION2_PACKAGE_DIR`` (and, for a base-bound adapter package,
optionally ``DECISION2_BASE_DIR``; otherwise the pinned base is resolved from
the Hugging Face cache) in the private runtime environment. The kit writes the
engine options to environment.json, so filesystem paths stay out of the options
and the provenance.
"""

from __future__ import annotations

import hashlib
import importlib.util
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

SHA256 = re.compile(r"[0-9a-f]{64}\Z")
REVISION = re.compile(r"[0-9a-f]{40}\Z")
DEVICE = re.compile(r"cuda:[0-9]+\Z")
REFUSALS = ("invalid_question", "max_length_exceeded")
TIE = 1e-8
SUM_TOLERANCE = 1e-5


class PackageRefusal(ValueError):
    """The package declined a question of the request; the request is unsupported."""


def import_package_runtime(package: Path) -> Any:
    """Import ``decision2`` from ``package/decision2`` only."""
    if any(
        name == "decision2" or name.startswith("decision2.") for name in sys.modules
    ):
        raise RuntimeError("Another decision2 runtime is already imported")
    init = package / "decision2" / "__init__.py"
    if init.is_symlink() or not init.is_file():
        raise ValueError("The package has no local decision2 runtime")
    spec = importlib.util.spec_from_file_location(
        "decision2", init, submodule_search_locations=[str(init.parent)]
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import the package runtime")
    module = importlib.util.module_from_spec(spec)
    sys.modules["decision2"] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        for name in list(sys.modules):
            if name == "decision2" or name.startswith("decision2."):
                del sys.modules[name]
        raise
    if Path(module.__file__).resolve() != init.resolve():
        raise RuntimeError("decision2 was imported from outside the package")
    return module


def _probability(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1


def check_response(
    response: Any, questions: dict[str, dict[str, Any]], model_name: str
) -> dict[str, Any]:
    """Return the package response unchanged if it is a complete valid answer set."""
    if not isinstance(response, dict) or response.get("model") != model_name:
        raise ValueError("The package answered under another model identity")
    answers = response.get("answers")
    if not isinstance(answers, dict) or set(answers) != set(questions):
        raise ValueError("The package returned missing or extra answers")
    usage = response.get("usage")
    if (
        not isinstance(usage, dict)
        or set(usage) != {"input_tokens", "output_tokens"}
        or any(type(value) is not int or value < 0 for value in usage.values())
    ):
        raise ValueError("The package returned invalid token usage")
    refused = set()
    for key, question in questions.items():
        answer = answers[key]
        if not isinstance(answer, dict):
            raise ValueError("The package returned a malformed answer")
        if "error" in answer:
            if answer["error"] not in REFUSALS:
                raise ValueError(f"The package returned {answer['error']!r}")
            refused.add(answer["error"])
    if refused:
        raise PackageRefusal(",".join(sorted(refused)))
    for key, question in questions.items():
        answer = answers[key]
        if answer.get("type") != question.get("type"):
            raise ValueError("The package returned the wrong answer type")
        if question["type"] == "noul":
            if not _probability(answer.get("noul")):
                raise ValueError("The package returned an invalid Noul probability")
            continue
        keys = list(question.get("criteria") or {})
        probabilities = answer.get("probabilities")
        if not isinstance(probabilities, dict) or set(probabilities) != set(keys):
            raise ValueError("The package omitted or added option probabilities")
        values = [probabilities[option] for option in keys]
        if not all(_probability(value) for value in values) or not math.isclose(
            sum(values), 1.0, rel_tol=0, abs_tol=SUM_TOLERANCE
        ):
            raise ValueError("The package returned an invalid option distribution")
        maximum = max(values)
        first = next(o for o, v in zip(keys, values) if abs(v - maximum) <= TIE)
        if answer.get("choice") != first:
            raise ValueError("The chosen option is not the package's argmax")
    return response


class ReleasedDecisionIndexEngine:
    """Duck-typed kit Engine: original state and questions in, package answers out."""

    name = "decision2-released-package-native"
    latency = (
        "In-process request wall time: the package's own prompt encoding, one "
        "batched forward pass over the request's questions, answer normalization "
        "and validation; excludes model loading."
    )

    def __init__(
        self,
        *,
        model_id: str,
        revision: str,
        package_manifest_sha256: str,
        device: str,
    ) -> None:
        if (
            not isinstance(model_id, str)
            or not model_id
            or not isinstance(revision, str)
            or REVISION.fullmatch(revision) is None
            or not isinstance(package_manifest_sha256, str)
            or SHA256.fullmatch(package_manifest_sha256) is None
            or not isinstance(device, str)
            or DEVICE.fullmatch(device) is None
        ):
            raise ValueError(
                "Model ID, pinned revision, package manifest digest and GPU required"
            )
        self.model = self._load(package_manifest_sha256, device)
        manifest = self.model.manifest
        if manifest.get("repo_id") != model_id:
            raise ValueError("The package belongs to another repository")
        self.model_name = manifest["model_name"]
        self.device = device
        base = manifest.get("base")
        self.provenance = {
            "kind": "decision2-released-package-native-choice-noul",
            "model_id": model_id,
            "revision": revision,
            "package_manifest_sha256": package_manifest_sha256,
            "model_sha256": manifest["identity"]["model_sha256"],
            "profile": manifest["profile"],
            "base": (
                {"repo_id": base["repo_id"], "revision": base["revision"]}
                if isinstance(base, dict)
                else None
            ),
            "calibration": manifest.get("calibration"),
            "policy": (
                "Original state and questions to the package's own system_one, one "
                "request per call; no truncation or option change; a refused "
                "question makes the request unsupported; malformed outputs are errors."
            ),
        }

    @staticmethod
    def _load(package_manifest_sha256: str, device: str) -> Any:
        package_name = os.environ.get("DECISION2_PACKAGE_DIR")
        if not package_name:
            raise RuntimeError("Set the private DECISION2_PACKAGE_DIR")
        package = Path(package_name)
        if package.is_symlink():
            raise ValueError("The package directory cannot be a link")
        package = package.resolve(strict=True)
        manifest = package / "MODEL_MANIFEST.json"
        digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
        if digest != package_manifest_sha256:
            raise ValueError("MODEL_MANIFEST.json differs from the pinned digest")
        base_name = os.environ.get("DECISION2_BASE_DIR")
        runtime = import_package_runtime(package)
        return runtime.Decision2.from_pretrained(
            package,
            device=device,
            base_path=Path(base_name).resolve(strict=True) if base_name else None,
        )

    def __call__(
        self, state: Any, questions: dict[str, dict[str, Any]]
    ) -> tuple[dict[str, Any], None]:
        from decision_index.engines import Unsupported

        response = self.model.system_one(state=state, questions=questions)
        try:
            return check_response(response, questions, self.model_name), None
        except PackageRefusal as exc:
            raise Unsupported(str(exc)) from exc

    def warmup(self) -> None:
        question = {
            "warmup": {
                "type": "choice",
                "instructions": "Which color is named?",
                "criteria": {"red": "red", "blue": "blue"},
            }
        }
        for _ in range(2):
            self("The color is red.", question)

    def synchronize(self) -> None:
        backend = self.model.backend
        if backend.device.type == "cuda":
            backend.torch.cuda.synchronize(backend.device)

    def runtime(self) -> dict[str, Any]:
        backend = self.model.backend
        torch = backend.torch
        manifest = self.model.manifest
        return {
            "native_max_input_tokens": manifest["max_input_tokens"],
            "native_parameters_loaded": manifest["parameters"]["loaded"],
            "native_temperatures": dict(backend.temperatures),
            "torch": torch.__version__,
            "hip": getattr(torch.version, "hip", None),
            "device_name": (
                torch.cuda.get_device_name(backend.device)
                if backend.device.type == "cuda"
                else "cpu"
            ),
        }

    def close(self) -> None:
        self.model = None
