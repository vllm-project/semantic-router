"""The ``decision`` pipeline for d3 models (``trust_remote_code=True``).

``pipeline("decision", model=repo, trust_remote_code=True)`` loads the model with ``AutoModel`` and answers
``{"state": ..., "questions": {...}}`` requests, optionally with ``"images": [...]`` (0 to 4 images per
request), or ``state=..., questions=..., images=...`` keywords, or a list of requests, with the model's
``system_one`` response. The model batches the questions of one request itself.
"""

from transformers import Pipeline

_UNSET = object()
REQUEST_KEYS = {"state", "questions"}
OPTIONAL_KEYS = {"images"}


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
        self, inputs=None, *, state=_UNSET, questions=_UNSET, images=_UNSET, **kwargs
    ):
        if state is not _UNSET or questions is not _UNSET or images is not _UNSET:
            if inputs is not None:
                raise TypeError("Pass one request, or state=, questions= and images=")
            inputs = {
                "state": None if state is _UNSET else state,
                "questions": None if questions is _UNSET else questions,
            }
            if images is not _UNSET:
                inputs["images"] = images
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
                'with optional "images": [...]'
            )
        return {
            "state": inputs["state"],
            "questions": inputs["questions"],
            "images": inputs.get("images"),
        }

    def _forward(self, model_inputs):
        return self.model.system_one(
            state=model_inputs["state"],
            questions=model_inputs["questions"],
            images=model_inputs["images"],
        )

    def postprocess(self, model_outputs):
        return model_outputs
