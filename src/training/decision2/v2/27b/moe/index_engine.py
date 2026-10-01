"""Decision Index Engine over a frozen 27B MoE package, through the package's native path.

The kit runner loads it as ``v2.27b.moe.index_engine:MoEPackageIndexEngine``. A frozen MoE
package (``decision2-27b-moe-package/1``: ``PACKAGE.json``, ``calibration.json`` and the LoRA
``checkpoint/`` on the pinned official MoE base) has no release runtime, so its native path is the
one its formal run used (``v2.27b.moe.collect`` -> ``training.model.infer``):

- the checkpoint fingerprint bound to the base files (``infer.checkpoint_fingerprint``) must equal
  the package's ``model_sha256``, and the package calibration (an adopted CAL698 fit or its T = 1
  binding) must be the frozen file and load for that model (``load_calibration``);
- ``DecisionModel.from_checkpoint`` with the checkpoint's experts pin and prompt encoder, FP32
  resident with BF16 autocast and an FP32 head, as ``infer.main``;
- one request per call, all its questions in one batch, through ``infer.run_prompts`` at the
  package's ``max_input_tokens`` (never truncated; over-budget or malformed questions are refused).

An exact Choice tie (within 1e-8) resolves to the first key in the caller's order, the System One
rule. Every response then passes the released-package adapter's ``check_response``: a refused
question makes the whole request ``Unsupported``; any other defect is an error, as in IX1.

Set ``DECISION2_MOE_PACKAGE_DIR`` (the staged package) and ``DECISION2_BASE_DIR`` (the pinned base
snapshot) in the private runtime environment; the engine options carry only the model ID, the
``PACKAGE.json`` SHA-256 and the device, so paths stay out of the kit's provenance.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable

from publication.decision_index_release_engine import PackageRefusal, check_response
from training.model import infer
from training.model.calibration import load_calibration
from training.model.data import canonical, file_sha256

SCHEMA = "decision2-27b-moe-package/1"
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
DEVICE = re.compile(r"cuda:[0-9]+\Z")
TIE = 1e-8
SUPPORTED_ARCHITECTURE_PREFIX = "gemma4-moe"
SOURCES = (
    "infer.py",
    "decision_model.py",
    "data.py",
    "lora.py",
    "source.py",
    "calibration.py",
)


def package_files(package_dir: Path) -> dict[str, Path]:
    return {
        "package": package_dir / "PACKAGE.json",
        "calibration": package_dir / "calibration.json",
        "checkpoint": package_dir / "checkpoint",
    }


def load_package(package_dir: Path, package_sha256: str) -> dict[str, Any]:
    """The staged package's manifest, pinned by digest, with its frozen calibration file."""
    if package_dir.is_symlink():
        raise ValueError("The package directory cannot be a link")
    files = package_files(package_dir)
    if file_sha256(files["package"]) != package_sha256:
        raise ValueError("PACKAGE.json differs from the pinned digest")
    package = json.loads(files["package"].read_text(encoding="utf-8"))
    if package.get("schema") != SCHEMA:
        raise ValueError(f"Not a {SCHEMA} package")
    if not str(package.get("architecture", "")).startswith(
        SUPPORTED_ARCHITECTURE_PREFIX
    ):
        raise ValueError(
            f"Unsupported MoE architecture {package.get('architecture')!r}"
        )
    if file_sha256(files["calibration"]) != package["calibration"]["sha256"]:
        raise ValueError("calibration.json differs from the frozen package calibration")
    metadata = json.loads(
        (files["checkpoint"] / "decision_config.json").read_text(encoding="utf-8")
    )
    for field in ("architecture", "experts_implementation", "prompt_version"):
        if metadata.get(field) != package.get(field):
            raise ValueError(f"The checkpoint's {field} differs from the package")
    if metadata["lora"]["base_revision"] != package["base"]["revision"]:
        raise ValueError("The checkpoint pins another base revision")
    if (
        type(package.get("max_input_tokens")) is not int
        or package["max_input_tokens"] < 2
    ):
        raise ValueError("The package has no valid max_input_tokens")
    return package


def bind(package_dir: Path, base: Path, package: dict[str, Any]) -> dict[str, Any]:
    """Model identity and temperatures exactly as ``infer.main`` derives them for the formal run."""
    files = package_files(package_dir)
    identity = infer.checkpoint_fingerprint(files["checkpoint"], base)
    if identity["model_sha256"] != package["model_sha256"]:
        raise ValueError(
            "The checkpoint and base do not give the package's model_sha256"
        )
    temperatures, report = load_calibration(
        files["calibration"], identity["model_sha256"]
    )
    if temperatures != package["calibration"]["temperature_by_type"]:
        raise ValueError("The calibration temperatures differ from the package")
    here = Path(infer.__file__).parent
    adapter_sources = {name: file_sha256(here / name) for name in SOURCES}
    adapter_sources["index_engine.py"] = file_sha256(Path(__file__))
    return {
        "model_sha256": identity["model_sha256"],
        "temperatures": temperatures,
        "calibration_sha256": package["calibration"]["sha256"],
        "cal_sha256": report.get("cal_sha256"),
        "adapter_sha256": hashlib.sha256(
            canonical(adapter_sources).encode("utf-8")
        ).hexdigest(),
        "adapter_files_sha256": adapter_sources,
    }


def system_one_answers(
    answers: dict[str, dict[str, Any]], questions: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """``normalized_answer`` leaves an exact Choice tie unchosen; System One picks the caller's first key."""
    out = {}
    for key, answer in answers.items():
        if (
            answer.get("type") == "choice"
            and "error" not in answer
            and answer.get("choice") is None
        ):
            probabilities = answer["probabilities"]
            top = max(probabilities.values())
            choice = next(
                option
                for option in questions[key]["criteria"]
                if abs(probabilities[option] - top) <= TIE
            )
            answer = {**answer, "choice": choice}
        out[key] = answer
    return out


def answer_request(
    state: Any,
    questions: dict[str, dict[str, Any]],
    *,
    tokenizer: Any,
    max_length: int,
    temperatures: dict[str, float],
    encode_fn: Callable[[dict[str, Any], Any, int], dict[str, Any]],
    predict_fn: Callable[[list[dict[str, Any]]], list[list[float]]],
    binding: dict[str, Any],
    model_name: str,
) -> dict[str, Any]:
    records, _ = infer.run_prompts(
        [{"id": "request", "state": state, "questions": questions}],
        tokenizer=tokenizer,
        max_length=max_length,
        temperature=temperatures,
        encode_fn=encode_fn,
        predict_fn=predict_fn,
        model_sha256=binding["model_sha256"],
        adapter_sha256=binding["adapter_sha256"],
        calibration_sha256=binding["calibration_sha256"],
    )
    record = records[0]
    return {
        "model": model_name,
        "answers": system_one_answers(record["answers"], questions),
        "usage": record["usage"],
    }


def _version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


class MoEPackageIndexEngine:
    """Duck-typed kit Engine: original state and questions in, the package's native answers out."""

    name = "decision2-27b-moe-package-native"
    latency = (
        "In-process request wall time: the package's native prompt encoding, one batched "
        "forward pass over the request's questions, answer normalization and validation; "
        "excludes model loading."
    )

    def __init__(self, *, model_id: str, package_sha256: str, device: str) -> None:
        if (
            not isinstance(model_id, str)
            or not model_id
            or not isinstance(package_sha256, str)
            or SHA256.fullmatch(package_sha256) is None
            or not isinstance(device, str)
            or DEVICE.fullmatch(device) is None
        ):
            raise ValueError("Model ID, PACKAGE.json digest and GPU required")
        package_name = os.environ.get("DECISION2_MOE_PACKAGE_DIR")
        base_name = os.environ.get("DECISION2_BASE_DIR")
        if not package_name or not base_name:
            raise RuntimeError(
                "Set the private DECISION2_MOE_PACKAGE_DIR and DECISION2_BASE_DIR"
            )
        package_dir = Path(package_name)
        package = load_package(package_dir, package_sha256)
        package_dir = package_dir.resolve(strict=True)
        base = Path(base_name).resolve(strict=True)
        self.binding = bind(package_dir, base, package)
        self.package = package
        self.model_name = model_id.rsplit("/", 1)[-1]
        self.max_length = package["max_input_tokens"]
        self._load(package_files(package_dir)["checkpoint"], base, device)
        self.provenance = {
            "kind": "decision2-27b-moe-package-native-choice-noul",
            "model_id": model_id,
            "package_sha256": package_sha256,
            "model_sha256": self.binding["model_sha256"],
            "checkpoint_sha256": package["checkpoint_sha256"],
            "base": {
                k: package["base"][k] for k in ("repo", "revision", "tree_sha256")
            },
            "architecture": package["architecture"],
            "experts_implementation": package["experts_implementation"],
            "prompt_version": package["prompt_version"],
            "lora": package["lora"],
            "calibration": {
                "decision": package["decision"],
                "sha256": self.binding["calibration_sha256"],
                "temperature_by_type": self.binding["temperatures"],
            },
            "adapter_sha256": self.binding["adapter_sha256"],
            "adapter_files_sha256": self.binding["adapter_files_sha256"],
            "policy": (
                "Original state and questions through the formal run's native path "
                "(training.model.infer.run_prompts), one request per call; no truncation or "
                "option change; a refused question makes the request unsupported; malformed "
                "outputs are errors."
            ),
        }

    def _load(self, checkpoint: Path, base: Path, device: str) -> None:
        import torch

        from training.model.decision_model import DecisionModel, collate, encoder_for

        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("A CUDA/ROCm BF16 GPU is required")
        self.torch = torch
        self.device = torch.device(device)
        model, tokenizer = DecisionModel.from_checkpoint(checkpoint, source_path=base)
        model = model.float().to(self.device).eval()
        pad_id = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        )
        if pad_id is None:
            raise ValueError("Tokenizer needs a pad or EOS token")
        target = self.device

        def predict(encoded: list[dict[str, Any]]) -> list[list[float]]:
            batch = {
                key: (
                    value.to(target, non_blocking=True)
                    if torch.is_tensor(value)
                    else value
                )
                for key, value in collate(encoded, pad_id).items()
            }
            torch.cuda.synchronize(target)
            with (
                torch.inference_mode(),
                torch.autocast(device_type="cuda", dtype=torch.bfloat16),
            ):
                logits = model(**batch)
            torch.cuda.synchronize(target)
            return [
                values[: len(item["keys"])].float().cpu().tolist()
                for values, item in zip(logits, encoded)
            ]

        self.model = model
        self.tokenizer = tokenizer
        self.encode_fn = encoder_for(model.metadata)
        self.predict_fn = predict

    def __call__(
        self, state: Any, questions: dict[str, dict[str, Any]]
    ) -> tuple[dict[str, Any], None]:
        from decision_index.engines import Unsupported

        response = answer_request(
            state,
            questions,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            temperatures=self.binding["temperatures"],
            encode_fn=self.encode_fn,
            predict_fn=self.predict_fn,
            binding=self.binding,
            model_name=self.model_name,
        )
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
        self.torch.cuda.synchronize(self.device)

    def runtime(self) -> dict[str, Any]:
        torch = self.torch
        config = getattr(self.model.backbone, "config", None)
        return {
            "native_max_input_tokens": self.max_length,
            "native_parameters_loaded": self.package["loaded_parameters"],
            "native_parameters_active": self.package["active_parameters"],
            "native_temperatures": dict(self.binding["temperatures"]),
            "experts_implementation": getattr(config, "_experts_implementation", None),
            "attn_implementation": getattr(config, "_attn_implementation", None),
            "torch": torch.__version__,
            "hip": getattr(torch.version, "hip", None),
            "transformers": _version("transformers"),
            "peft": _version("peft"),
            "device_name": torch.cuda.get_device_name(self.device),
        }

    def close(self) -> None:
        self.model = None
