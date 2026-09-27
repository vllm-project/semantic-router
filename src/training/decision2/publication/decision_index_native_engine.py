"""Decision Index Engine contract over a sealed Decision 2.0 native package.

This module intentionally does not implement an Index edition or scorer. The
public reproduction kit still targets 0.2; a 0.2.1 score requires a separately
verified 0.2.1 corpus and scorer. The runner may load this class as
``publication.decision_index_native_engine:NativeDecisionIndexEngine``.

Set ``DECISION2_PACKAGE_DIR`` and ``DECISION2_BASE_DIR`` in the *private*
runtime environment. The kit records engine options in environment.json, so
filesystem paths are deliberately absent from those options and provenance.
"""

from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import Any

from . import package_native_arena as arena

SHA256 = re.compile(r"[0-9a-f]{64}\Z")
DEVICE = re.compile(r"cuda:[0-9]+\Z")


class NativeUnsupported(ValueError):
    """A complete original request cannot be represented by this native head."""


def _load_native(
    *, model_id: str, package_manifest_sha256: str, device: str
) -> tuple[Any, dict[str, Any], Any]:
    package_name = os.environ.get("DECISION2_PACKAGE_DIR")
    source_name = os.environ.get("DECISION2_BASE_DIR")
    if not package_name or not source_name:
        raise RuntimeError("Set private Decision 2.0 package and base directories")
    package = Path(package_name).resolve(strict=True)
    source = Path(source_name).resolve(strict=True)
    revision = f"package-sha256:{package_manifest_sha256}"
    manifest, _ = arena._package_manifest(
        package, package_manifest_sha256, model_id, revision
    )
    api, infer = arena._import_sealed_package(package)
    model = api.Decision2.from_pretrained(package, source_path=source, device=device)
    return model, manifest, infer.question_to_row


def _preflight(
    state: Any, questions: Any, question_to_row: Any
) -> dict[str, dict[str, Any]]:
    if not isinstance(state, (str, dict)):
        raise NativeUnsupported("unsupported_state_type")
    if (
        not isinstance(questions, dict)
        or not questions
        or any(type(key) is not str or not key for key in questions)
    ):
        raise NativeUnsupported("unsupported_question_mapping")
    for key, question in questions.items():
        if not isinstance(question, dict) or question.get("type") not in {
            "choice",
            "noul",
        }:
            raise NativeUnsupported("unsupported_question_type")
        if not isinstance(question.get("instructions"), str):
            raise NativeUnsupported("native_structured_instructions_unsupported")
        criteria = question.get("criteria")
        if question["type"] == "choice":
            if isinstance(criteria, dict) and len(criteria) > 255:
                raise NativeUnsupported("native_option_count_exceeded")
            if isinstance(criteria, dict) and any(
                not isinstance(value, str) for value in criteria.values()
            ):
                raise NativeUnsupported("native_structured_option_unsupported")
        elif (
            criteria is not None
            and isinstance(criteria, dict)
            and set(criteria)
            != {
                "false",
                "true",
            }
        ):
            raise NativeUnsupported("native_noul_criteria_unsupported")
        # The sealed package uses this same default for a Noul question with
        # no explicit criteria. Do not modify the original runner payload.
        normalized = dict(question)
        if normalized["type"] == "noul" and "criteria" not in normalized:
            normalized["criteria"] = {"false": "No", "true": "Yes"}
        try:
            question_to_row({"id": "index", "state": state}, key, normalized)
        except ValueError as exc:
            # Structured instructions/descriptions and excessive option counts
            # are currently outside this immutable package's native contract.
            # The complete request is refused rather than changing any prompt,
            # dropping options, or returning a partial answer.
            raise NativeUnsupported("native_question_contract") from exc
    return questions


def _validate_native(
    response: Any, questions: dict[str, dict[str, Any]], model_id: str
) -> dict[str, Any]:
    if not isinstance(response, dict) or response.get("model") != model_id:
        raise ValueError("Native package returned a different model identity")
    answers = response.get("answers")
    if not isinstance(answers, dict) or set(answers) != set(questions):
        raise ValueError("Native package returned missing or extra answers")
    usage = response.get("usage")
    if (
        not isinstance(usage, dict)
        or set(usage) != {"input_tokens", "output_tokens"}
        or any(type(value) is not int or value < 0 for value in usage.values())
    ):
        raise ValueError("Native package returned invalid token usage")
    for key, question in questions.items():
        answer = answers[key]
        if not isinstance(answer, dict) or answer.get("type") != question["type"]:
            raise ValueError("Native package returned the wrong answer type")
        if answer.get("error") == "max_length_exceeded":
            raise NativeUnsupported("native_max_length_exceeded")
        if "error" in answer:
            raise ValueError("Native package returned invalid model output")
        if question["type"] == "noul":
            probability = answer.get("noul")
            if (
                type(probability) not in (int, float)
                or not math.isfinite(probability)
                or not 0 <= probability <= 1
            ):
                raise ValueError("Native package returned invalid Noul probability")
            continue
        criteria = question["criteria"]
        probabilities = answer.get("probabilities")
        if not isinstance(probabilities, dict) or set(probabilities) != set(criteria):
            raise ValueError("Native package omitted or added option probabilities")
        values = list(probabilities.values())
        if any(
            type(value) not in (int, float)
            or not math.isfinite(value)
            or not 0 <= value <= 1
            for value in values
        ) or not math.isclose(sum(values), 1.0, rel_tol=0, abs_tol=1e-5):
            raise ValueError("Native package returned invalid option probabilities")
        selected = answer.get("choice")
        if selected not in criteria:
            raise ValueError("Native package returned an invalid chosen option")
        maximum = max(values)
        winners = [
            option
            for option, value in probabilities.items()
            if abs(value - maximum) <= 1e-8
        ]
        if len(winners) != 1 or selected != winners[0]:
            raise ValueError("Native package choice differs from unique argmax")
    return response


class NativeDecisionIndexEngine:
    """Duck-typed upstream Engine: state/questions in; native answers out."""

    name = "decision2-package-native-choice-noul"
    latency = (
        "In-process request wall time including native prompt construction, "
        "model inference and validation; excludes model loading."
    )

    def __init__(
        self,
        *,
        model_id: str,
        package_manifest_sha256: str,
        device: str,
    ) -> None:
        if (
            not isinstance(model_id, str)
            or not model_id
            or not isinstance(package_manifest_sha256, str)
            or SHA256.fullmatch(package_manifest_sha256) is None
            or not isinstance(device, str)
            or DEVICE.fullmatch(device) is None
        ):
            raise ValueError("Model ID, pinned package digest and GPU device required")
        self.model, self.manifest, self.question_to_row = _load_native(
            model_id=model_id,
            package_manifest_sha256=package_manifest_sha256,
            device=device,
        )
        self.model_id = model_id
        self.device = device
        self.provenance = {
            "kind": "sealed-decision2-native-choice-noul",
            "model_id": model_id,
            "package_manifest_sha256": package_manifest_sha256,
            "model_sha256": self.manifest["model_sha256"],
            "calibration_sha256": self.manifest["calibration_sha256"],
            "base_repo_id": self.manifest["base"]["repo_id"],
            "base_revision": self.manifest["base"]["revision"],
            "policy": (
                "Pass original state and all questions to packaged native heads; "
                "do not truncate or change options; incompatible shape and "
                "context are unsupported; malformed outputs are errors."
            ),
        }

    def __call__(
        self, state: Any, questions: dict[str, dict[str, Any]]
    ) -> tuple[dict[str, Any], None]:
        from decision_index.engines import Unsupported

        try:
            _preflight(state, questions, self.question_to_row)
            response = self.model.system_one(state=state, questions=questions)
            return _validate_native(response, questions, self.model_id), None
        except NativeUnsupported as exc:
            raise Unsupported(str(exc)) from exc

    def warmup(self) -> None:
        question = {
            "warmup": {
                "type": "choice",
                "instructions": "Which color is named?",
                "criteria": {"red": "red", "blue": "blue"},
            }
        }
        self("The color is red.", question)

    def synchronize(self) -> None:
        if self.model.device.type == "cuda":
            self.model.torch.cuda.synchronize(self.model.device)

    def runtime(self) -> dict[str, Any]:
        return {
            "native_max_length": self.manifest["max_length"],
            "native_parameter_count": self.manifest["parameter_count"],
            "native_temperature_by_type": {
                key: self.manifest["temperature_by_type"][key]
                for key in ("choice", "noul")
            },
        }

    def close(self) -> None:
        self.model = None
