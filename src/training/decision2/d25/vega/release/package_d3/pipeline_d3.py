"""The ``decision`` pipeline for d3 models (``trust_remote_code=True``).

``pipeline("decision", model=repo, trust_remote_code=True)`` loads the model with ``AutoModel`` and answers
``{"state": ..., "questions": {...}}`` requests, optionally with ``"images": [...]`` and ``"videos": [...]`` (any
number per request), or ``state=..., questions=..., images=..., videos=...`` keywords, or a list of requests, with
the model's
``system_one`` response. The model batches the questions of one request itself.
"""

from transformers import Pipeline

_UNSET = object()
REQUEST_KEYS = {"state", "questions"}
OPTIONAL_KEYS = {"images", "videos"}


class D3Pipeline(Pipeline):
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

    def __call__(
        self,
        inputs=None,
        *,
        state=_UNSET,
        questions=_UNSET,
        images=_UNSET,
        videos=_UNSET,
        **kwargs,
    ):
        if any(v is not _UNSET for v in (state, questions, images, videos)):
            if inputs is not None:
                raise TypeError(
                    "Pass one request, or state=, questions=, images= and videos="
                )
            inputs = {
                "state": None if state is _UNSET else state,
                "questions": None if questions is _UNSET else questions,
            }
            if images is not _UNSET:
                inputs["images"] = images
            if videos is not _UNSET:
                inputs["videos"] = videos
        if kwargs.get("batch_size") not in (None, 1):
            raise ValueError(
                "The decision pipeline runs one request at a time (batch_size=1)"
            )
        return super().__call__(inputs, **kwargs)

    def preprocess(self, inputs):
        if not isinstance(inputs, dict) or not (
            REQUEST_KEYS <= set(inputs) <= REQUEST_KEYS | OPTIONAL_KEYS
        ):
            raise ValueError(
                'A decision request is {"state": ..., "questions": {<id>: <question>, ...}} '
                'with optional "images": [...] and "videos": [...]'
            )
        return {
            "state": inputs["state"],
            "questions": inputs["questions"],
            "images": inputs.get("images"),
            "videos": inputs.get("videos"),
        }

    def _forward(self, model_inputs):
        return self.model.system_one(
            state=model_inputs["state"],
            questions=model_inputs["questions"],
            images=model_inputs["images"],
            videos=model_inputs["videos"],
        )

    def postprocess(self, model_outputs):
        return model_outputs
