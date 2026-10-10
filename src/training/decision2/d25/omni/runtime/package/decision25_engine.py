"""Decision Index engine for a Decision 2.5 checkpoint (the kit's one-request-at-a-time path).

    hf download vllm-sr/Decision-2.5-Vega-27B --local-dir vega25
    PYTHONPATH=vega25 python -m decision_index run --engine decision25_engine:Decision25Engine \
        --option model=vega25 --option device=cuda:0 --out runs/vega25 --compact

Options: ``model`` (package directory or Hub id), ``revision``, ``device`` (default cuda:0), ``batch_size``
(questions per forward pass, default 8), ``verify`` (fast | full | none), ``model_name``. A request with a
question over the checkpoint's input limit is ``Unsupported`` (nothing is truncated).

Image requests: ``engine(state, questions, images=[...])`` with up to 4 images (PIL images, paths, http(s)
or data URLs) that every question sees; more than 4 images are ``Unsupported``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from decision_index.engines import Engine, Unsupported

# Not resolve(): in a Hugging Face cache snapshot this file is a link into the hash-named blobs directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Keep Triton autotune results on disk, so later processes reuse them (read when the kernels are imported).
os.environ.setdefault("TRITON_CACHE_AUTOTUNING", "1")

from decision25_runtime import (  # noqa: E402
    DEFAULT_BATCH_SIZE,
    Decision25,
    ImageLimitExceeded,
)


class Decision25Engine(Engine):
    name = "decision-2.5"
    latency = (
        "Device-synchronized in-process request wall time including prompt rendering and tokenization (and, "
        "for image requests, image decoding and preprocessing); one request per call, its questions in request "
        "order in batches of batch_size; excludes model loading."
    )

    def __init__(
        self,
        model: str | None = None,
        revision: str | None = None,
        device: str = "cuda:0",
        batch_size: int = DEFAULT_BATCH_SIZE,
        verify: str = "fast",
        model_name: str | None = None,
        **options,
    ):
        if options:
            raise TypeError(f"unknown engine options {sorted(options)}")
        if not model:
            raise ValueError("pass --option model=<package dir or Hub id>")
        super().__init__(
            model=model,
            revision=revision,
            device=device,
            batch_size=batch_size,
            verify=verify,
            model_name=model_name,
        )
        self.decision = Decision25.from_pretrained(
            model,
            revision=revision,
            device=device,
            batch_size=int(batch_size),
            verify=verify,
            model_name=model_name,
        )
        self.provenance = self.decision.provenance()

    def warmup(self):
        super().warmup()
        self.warmup_seconds = self.decision.warmup()

    def runtime(self):
        return self.decision.runtime_info()

    def synchronize(self):
        self.decision.synchronize()

    def __call__(self, state, questions, images=None):
        try:
            prepared = self.decision.prepare(state, questions, images)
        except ImageLimitExceeded as exc:
            raise Unsupported(str(exc)) from exc
        over = [
            e["message"]
            for e in prepared.errors.values()
            if e["error"] == "max_length_exceeded"
        ]
        if over:
            raise Unsupported(over[0])
        if prepared.errors:
            raise ValueError(
                "invalid questions: "
                + "; ".join(f"{k}: {e['message']}" for k, e in prepared.errors.items())
            )
        probabilities, tokens = self.decision.run(prepared)
        response = self.decision.respond(prepared, probabilities, tokens)
        failed = {
            k: a["message"] for k, a in response["answers"].items() if "error" in a
        }
        if failed:
            raise ValueError(f"invalid model output: {failed}")
        return response, None
