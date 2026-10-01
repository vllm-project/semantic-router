# Copyright 2026 The vLLM Semantic Router Authors.
# SPDX-License-Identifier: Apache-2.0
"""``pipeline("decision", model=repo, trust_remote_code=True)`` for Decision 1.0 models.

Call it with a System One request body, ``{"state": ..., "questions": {...}}``
(or a list of them); each call returns the System One response body.
"""

from __future__ import annotations

from typing import Any

from transformers import Pipeline


class DecisionPipeline(Pipeline):
    _load_tokenizer = False
    _load_processor = False
    _load_image_processor = False
    _load_feature_extractor = False
    _load_video_processor = False

    def _sanitize_parameters(self, **kwargs):
        if kwargs:
            raise TypeError(
                f"Unsupported decision pipeline arguments: {sorted(kwargs)}"
            )
        return {}, {}, {}

    def preprocess(self, inputs: Any) -> dict[str, Any]:
        if (
            not isinstance(inputs, dict)
            or set(inputs) - {"state", "questions", "model"}
            or not {
                "state",
                "questions",
            }
            <= set(inputs)
        ):
            raise ValueError("A decision request is an object with state and questions")
        return {"state": inputs["state"], "questions": inputs["questions"]}

    def _forward(self, model_inputs: dict[str, Any]) -> dict[str, Any]:
        return self.model.system_one(
            state=model_inputs["state"], questions=model_inputs["questions"]
        )

    def postprocess(self, model_outputs: dict[str, Any]) -> dict[str, Any]:
        return model_outputs
