"""Development server entry point for the backend-neutral Decision API."""

from __future__ import annotations

import importlib
import os

from .api import create_app
from .backend import ModelDescriptor
from .engine import DecisionEngine
from .fake_backend import FakeDecisionBackend
from .scheduler import ModelScheduler


def create_default_app():
    """Create a deterministic contract server; it is not a real model server."""

    model_names = tuple(
        name.strip()
        for name in os.getenv("VLLM_SR_DECISION_MODELS", "decision-fake").split(",")
        if name.strip()
    )
    if not model_names:
        raise ValueError("VLLM_SR_DECISION_MODELS must name at least one model")
    models = [
        ModelDescriptor(
            name=name,
            description="Deterministic contract-development backend",
            release_date="1970-01-01",
        )
        for name in model_names
    ]
    engine = DecisionEngine(FakeDecisionBackend(models))
    scheduler = ModelScheduler(
        model_names,
        max_concurrency=_positive_env("VLLM_SR_DECISION_CONCURRENCY", 1),
        max_queue=_nonnegative_env("VLLM_SR_DECISION_QUEUE", 8),
    )
    return create_app(engine, scheduler=scheduler)


def _positive_env(name: str, default: int) -> int:
    value = _integer_env(name, default)
    if value < 1:
        raise ValueError(f"{name} must be positive")
    return value


def _nonnegative_env(name: str, default: int) -> int:
    value = _integer_env(name, default)
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _integer_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


app = create_default_app()


def main() -> None:
    """Run the optional Uvicorn server."""

    try:
        uvicorn = importlib.import_module("uvicorn")
    except ImportError as exc:  # pragma: no cover - packaging guard
        raise SystemExit(
            "Install the server dependencies with "
            "`pip install 'vllm-sr[decision-runtime]'`."
        ) from exc
    uvicorn.run(
        "decision_runtime.server:app",
        host=os.getenv("VLLM_SR_DECISION_HOST", "127.0.0.1"),
        port=_positive_env("VLLM_SR_DECISION_PORT", 8000),
        workers=1,
    )


if __name__ == "__main__":
    main()
