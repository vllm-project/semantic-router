# Copyright 2026 The vLLM Semantic Router Authors.
# SPDX-License-Identifier: Apache-2.0
"""The ``decision`` pipeline for Decision 1.0 models (``trust_remote_code=True``).

``pipeline("decision", model=repo, trust_remote_code=True)`` loads the model with
``AutoModel`` and answers ``{"state": ..., "questions": {...}}`` requests (or
``state=..., questions=...`` keywords, or a list of requests) with the model's
``system_one`` response. The model batches the questions of one request itself.
"""

from transformers import Pipeline

_UNSET = object()


class Decision1Pipeline(Pipeline):
    _load_tokenizer = False
    _load_processor = False
    _load_image_processor = False
    _load_feature_extractor = False
    _load_video_processor = False

    def _sanitize_parameters(self, **kwargs):
        if kwargs:
            raise TypeError(
                f"The decision pipeline takes no parameters: {sorted(kwargs)}"
            )
        return {}, {}, {}

    def __call__(self, inputs=None, *, state=_UNSET, questions=_UNSET, **kwargs):
        if state is not _UNSET or questions is not _UNSET:
            if inputs is not None:
                raise TypeError("Pass one request, or state= and questions=")
            inputs = {
                "state": None if state is _UNSET else state,
                "questions": None if questions is _UNSET else questions,
            }
        if kwargs.get("batch_size") not in (None, 1):
            raise ValueError(
                "The decision pipeline runs one request at a time (batch_size=1)"
            )
        return super().__call__(inputs, **kwargs)

    def preprocess(self, inputs):
        if not isinstance(inputs, dict) or set(inputs) != {"state", "questions"}:
            raise ValueError(
                'A decision request is {"state": ..., "questions": {<id>: <question>, ...}}'
            )
        return {"state": inputs["state"], "questions": inputs["questions"]}

    def _forward(self, model_inputs):
        return self.model.system_one(
            state=model_inputs["state"], questions=model_inputs["questions"]
        )

    def postprocess(self, model_outputs):
        return model_outputs
