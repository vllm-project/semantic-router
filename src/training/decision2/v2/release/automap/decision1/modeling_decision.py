# Copyright 2026 The vLLM Semantic Router Authors.
# SPDX-License-Identifier: Apache-2.0
"""Decision 1.0 for Transformers: System One Choice, Noul and Score decisions.

``AutoModel.from_pretrained(repo, trust_remote_code=True)`` downloads the files
named by the repository's ``config.json`` and builds the native inference path.
``model.system_one(state=..., questions={...})`` takes and returns the System One
request and response bodies served by the vLLM Semantic Router Decision runtime.
There is no text generation and no chat API.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import PreTrainedModel

from .configuration_decision import DecisionConfig
from .decision_system_one import (
    DecisionInputError,
    DecisionInputTooLongError,
    answer,
    build_rows,
    validate_questions,
    validate_state,
    validate_states,
)

# Input limits and prompt policies of the published models, as the Decision runtime serves them.
FAMILY_PROFILES = {
    "vela-encoder": {"max_input_tokens": 1024, "choice_null_description": "render_key"},
    "qwen3.5-decision": {
        "max_input_tokens": 16384,
        "choice_null_description": "preserve_json_null",
    },
}
MODEL_PROFILES = {"Decision-1.0-Nox-4B": {"choice_null_description": "render_key"}}
_IGNORED_LOAD_ARGUMENTS = {
    "trust_remote_code",
    "code_revision",
    "_from_auto",
    "_from_pipeline",
    "_commit_hash",
    "proxies",
    "resume_download",
    "use_safetensors",
    "low_cpu_mem_usage",
    "subfolder",
    "attn_implementation",
    "user_agent",
    "adapter_kwargs",
    "use_auth_token",
}

__all__ = ["DecisionModel", "DecisionInputError", "DecisionInputTooLongError"]


def _device(device: Any, device_map: Any) -> torch.device:
    if device is not None and device_map is not None:
        raise ValueError("Pass device or device_map, not both")
    chosen = device if device is not None else device_map
    if chosen is None or chosen == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if isinstance(chosen, dict):
        raise ValueError(
            "Decision 1.0 models load on one device; pass device='cuda:0' or 'cpu'"
        )
    if isinstance(chosen, int):
        return torch.device("cuda", chosen) if chosen >= 0 else torch.device("cpu")
    return torch.device(chosen)


class DecisionModel(PreTrainedModel):
    config_class = DecisionConfig
    base_model_prefix = "decision"
    _no_split_modules: list[str] = []

    def __init__(self, config: DecisionConfig):
        super().__init__(config)
        self._runtime = None
        self.post_init()

    def _init_weights(self, module):
        """Weights always come from the repository files."""

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *model_args,
        config: DecisionConfig | None = None,
        revision: str | None = None,
        cache_dir: str | None = None,
        token: str | bool | None = None,
        force_download: bool = False,
        local_files_only: bool = False,
        device: Any = None,
        device_map: Any = None,
        dtype: Any = None,
        torch_dtype: Any = None,
        **kwargs: Any,
    ) -> DecisionModel:
        """Load a Decision 1.0 repository (Hub ID or local directory) on one device.

        ``device`` (or ``device_map``) defaults to ``cuda:0`` when a GPU is visible,
        else ``cpu``. Precision is fixed by the model and cannot be overridden.
        """
        if model_args:
            raise TypeError("Decision models take no positional model arguments")
        if kwargs.get("attn_implementation") not in (None, "sdpa"):
            raise ValueError("Decision 1.0 models run with SDPA attention only")
        if kwargs.get("subfolder"):
            raise ValueError("Decision 1.0 repositories load from their root")
        unknown = set(kwargs) - _IGNORED_LOAD_ARGUMENTS
        if unknown:
            raise TypeError(
                f"Unsupported Decision loading arguments: {sorted(unknown)}"
            )
        for value in (dtype, torch_dtype):
            if value not in (None, "auto"):
                raise ValueError(
                    "Decision 1.0 models fix their own precision; omit dtype"
                )
        hub = {
            "cache_dir": cache_dir,
            "token": token if token is not None else kwargs.get("use_auth_token"),
            "force_download": force_download,
            "local_files_only": local_files_only,
        }
        source = str(pretrained_model_name_or_path)
        if config is None:
            config = DecisionConfig.from_pretrained(source, revision=revision, **hub)
        root = Path(source)
        if not root.is_dir():
            from huggingface_hub import snapshot_download

            pinned = (
                getattr(config, "_commit_hash", None)
                or kwargs.get("_commit_hash")
                or revision
            )
            root = Path(
                snapshot_download(
                    repo_id=source,
                    revision=pinned,
                    allow_patterns=config.files(),
                    **hub,
                )
            )
        model = cls(config)
        model._attach(root.resolve(), _device(device, device_map))
        return model

    def _attach(self, root: Path, device: torch.device) -> None:
        descriptor = self.config.descriptor()
        profile = dict(FAMILY_PROFILES[descriptor["runtime_family"]])
        profile.update(MODEL_PROFILES.get(descriptor["model_name"], {}))
        if descriptor["runtime_family"] == "vela-encoder":
            from .decision_vela import VelaRuntime

            runtime = VelaRuntime.load(
                root,
                descriptor,
                max_input_tokens=profile["max_input_tokens"],
                device=device,
            )
        else:
            from .decision_qwen import QwenRuntime

            runtime = QwenRuntime.load(
                root,
                descriptor,
                max_input_tokens=profile["max_input_tokens"],
                choice_null_description=profile["choice_null_description"],
                device=device,
            )
        self.decision = runtime.model
        self._runtime = runtime
        self.eval()

    @property
    def tokenizer(self):
        return self._runtime.tokenizer

    @property
    def max_input_tokens(self) -> int:
        return self._runtime.max_input_tokens

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "Use system_one(state=..., questions=...); there is no tensor API"
        )

    def system_one(self, *, state: Any, questions: dict[str, Any]) -> dict[str, Any]:
        """Answer named Noul, Choice and Score questions about one state.

        Returns ``{"model", "answers", "usage"}``. A question whose complete
        input exceeds the model's token limit raises ``DecisionInputTooLongError``;
        nothing is truncated.
        """
        if self._runtime is None:
            raise RuntimeError("Load the model with from_pretrained")
        checked = validate_questions(questions)
        rows = build_rows(
            validate_state(state),
            checked,
            noul_default_false=self._runtime.noul_default_false,
            noul_default_true=self._runtime.noul_default_true,
            noul_explicit_null=self._runtime.noul_explicit_null,
        )
        probabilities, tokens = self._runtime.predict(rows)
        return {
            "model": self.config.model_name,
            "answers": {
                row.question_id: answer(row, p) for row, p in zip(rows, probabilities)
            },
            "usage": {"input_tokens": sum(tokens), "output_tokens": 0},
        }

    def system_one_batch(
        self, *, states: list[dict[str, Any]], questions: dict[str, Any]
    ) -> dict[str, Any]:
        """The same questions about many identified states; each state is answered as one request."""
        items = validate_states(states)
        validate_questions(questions)
        results = []
        for state_id, state in items:
            response = self.system_one(state=state, questions=questions)
            results.append(
                {
                    "id": state_id,
                    "answers": response["answers"],
                    "usage": response["usage"],
                }
            )
        return {
            "model": self.config.model_name,
            "results": results,
            "usage": {
                "input_tokens": sum(item["usage"]["input_tokens"] for item in results),
                "output_tokens": 0,
            },
        }

    def choice(
        self, state: Any, instructions: Any, criteria: dict[str, Any]
    ) -> dict[str, Any]:
        return self._one(
            state,
            {"type": "choice", "instructions": instructions, "criteria": criteria},
        )

    def noul(
        self, state: Any, instructions: Any, criteria: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        question = {"type": "noul", "instructions": instructions}
        if criteria is not None:
            question["criteria"] = criteria
        return self._one(state, question)

    def score(
        self, state: Any, instructions: Any, criteria: list[Any]
    ) -> dict[str, Any]:
        return self._one(
            state, {"type": "score", "instructions": instructions, "criteria": criteria}
        )

    def _one(self, state: Any, question: dict[str, Any]) -> dict[str, Any]:
        return self.system_one(state=state, questions={"answer": question})["answers"][
            "answer"
        ]
