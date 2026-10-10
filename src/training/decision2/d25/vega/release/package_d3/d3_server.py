"""System One HTTP server for a d3 checkpoint: ``POST /v1/systemone``.

    pip install fastapi uvicorn
    python d3_server.py --model <package dir or Hub id> [--device cuda:0] [--host 127.0.0.1] [--port 8000]

Request ``{"model", "state", "questions", "images"}``, response ``{"model", "answers", "usage"}``: the wire
format of the Decision Index ``http`` engine. ``images`` (optional) lists any number of base64 PNG, JPEG or WebP
data URLs (``data:image/png;base64,...``) that every question sees, each at most 8,000,000 bytes and 16,000,000
pixels (the model reads it at up to 1.6 MP). A question over the input limit refuses the whole request with
HTTP 422 naming the maximum context length (the Index records it as unsupported; nothing is truncated);
malformed requests and invalid images also get 422. Requests are served one at a time. With
``DECISION_API_KEY`` set, requests need ``Authorization: Bearer <key>``. ``GET /health`` and
``GET /v1/models`` describe the loaded model.

The server design is adapted from perplexity-ai/pplx-decider-v1.1-27b, Copyright Perplexity AI,
Apache License 2.0.
"""

import argparse
import hmac
import os
import sys
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

# Not resolve(): in a Hugging Face cache snapshot this file is a link into the hash-named blobs directory.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Keep Triton autotune results on disk, so later processes reuse them (read when the kernels are imported).
os.environ.setdefault("TRITON_CACHE_AUTOTUNING", "1")

from d3_runtime import (  # noqa: E402
    DEFAULT_BATCH_SIZE,
    IMAGE_MAX_PIXELS,
    D3,
)

REQUEST_FIELDS = {"model", "state", "questions", "images"}


def reads_images(model: D3) -> bool:
    return getattr(model, "image_unavailable", "unknown") is None


def modalities(model: D3) -> list[str]:
    return ["text", "image"] if reads_images(model) else ["text"]


class Service:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.model: D3 | None = None
        self.lock = threading.Lock()


def build_app(args: argparse.Namespace):
    from fastapi import Depends, FastAPI, Header, HTTPException, Request
    from fastapi.responses import JSONResponse
    from starlette.concurrency import run_in_threadpool

    service = Service(args)

    @asynccontextmanager
    async def lifespan(app):
        model = await run_in_threadpool(
            D3.from_pretrained,
            args.model,
            revision=args.revision,
            device=args.device,
            batch_size=args.batch_size,
            verify=args.verify,
            model_name=args.name,
        )
        if not args.no_warmup:
            await run_in_threadpool(model.warmup)
        service.model = model
        try:
            yield
        finally:
            service.model = None

    app = FastAPI(title="d3 System One", version="1.0", lifespan=lifespan)

    def authenticate(authorization: str | None = Header(default=None)) -> None:
        key = os.getenv("DECISION_API_KEY")
        if key and not hmac.compare_digest(
            (authorization or "").encode(), f"Bearer {key}".encode()
        ):
            raise HTTPException(
                401,
                "Missing or invalid API key.",
                headers={"WWW-Authenticate": "Bearer"},
            )

    @app.middleware("http")
    async def timing(request: Request, call_next):
        started, identifier = time.perf_counter(), uuid.uuid4().hex
        response = await call_next(request)
        response.headers["x-request-id"] = identifier
        response.headers["server-timing"] = (
            f"total;dur={(time.perf_counter() - started) * 1000:.1f}"
        )
        return response

    @app.get("/health")
    def health() -> dict[str, Any]:
        model = service.model
        return {
            "status": "ready" if model is not None else "loading",
            "model": model.model_name if model else None,
            "max_input_tokens": model.max_length if model else None,
            "modalities": modalities(model) if model else None,
            "authentication": bool(os.getenv("DECISION_API_KEY")),
        }

    @app.get("/v1/models", dependencies=[Depends(authenticate)])
    def models() -> dict[str, Any]:
        model = service.model
        if model is None:
            raise HTTPException(503, "The model is not ready.")
        entry = {
            "name": model.model_name,
            "description": "d3 typed decisions (choice, noul, score).",
            "max_input_tokens": model.max_length,
            "modalities": modalities(model),
        }
        if reads_images(model):
            entry["image_max_pixels"] = IMAGE_MAX_PIXELS
        return {"models": [entry]}

    def decode_images(model: D3, images: Any) -> list[Any]:
        if not isinstance(images, list):
            raise ValueError("images must be a list of base64 data URLs")
        if not images:
            return []
        if not reads_images(model):
            raise ValueError("This model reads text only; images are not supported.")
        return model.load_images(images, strict=True)

    def decide(body: dict[str, Any], images: list[Any]) -> dict[str, Any]:
        model = service.model
        with service.lock:
            if images:
                prepared = model.prepare(
                    body.get("state"), body.get("questions"), images
                )
            else:
                prepared = model.prepare(body.get("state"), body.get("questions"))
            over = [
                e
                for e in prepared.errors.values()
                if e["error"] == "max_length_exceeded"
            ]
            if over:
                raise HTTPException(422, over[0]["message"])
            invalid = {k: e["message"] for k, e in prepared.errors.items()}
            if invalid:
                raise HTTPException(422, {"invalid_questions": invalid})
            probabilities, tokens = model.run(prepared)
            return model.respond(prepared, probabilities, tokens)

    @app.post("/v1/systemone", dependencies=[Depends(authenticate)])
    async def system_one(request: Request):
        if service.model is None:
            raise HTTPException(503, "The model is not ready.")
        try:
            body = await request.json()
        except ValueError as exc:
            raise HTTPException(422, "The request body must be JSON.") from exc
        if not isinstance(body, dict):
            raise HTTPException(422, "The request body must be a JSON object.")
        unknown = set(body) - REQUEST_FIELDS
        if unknown:
            raise HTTPException(422, f"Unknown request fields: {sorted(unknown)}")
        try:
            images = (
                await run_in_threadpool(decode_images, service.model, body["images"])
                if body.get("images") is not None
                else []
            )
            return await run_in_threadpool(decide, body, images)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    return app


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--model",
        default=os.getenv("DECISION_MODEL", os.path.dirname(os.path.abspath(__file__))),
        help="package directory or Hub repository (default: this file's directory)",
    )
    ap.add_argument("--revision")
    ap.add_argument("--device")
    ap.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--verify", default="fast", choices=("fast", "full", "none"))
    ap.add_argument(
        "--name", help="served model name (default: the package's model name)"
    )
    ap.add_argument(
        "--no-warmup",
        action="store_true",
        help="skip compiling the kernels for every batch size at start",
    )
    ap.add_argument("--host", default=os.getenv("HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")))
    args = ap.parse_args(argv)
    import uvicorn

    uvicorn.run(build_app(args), host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
