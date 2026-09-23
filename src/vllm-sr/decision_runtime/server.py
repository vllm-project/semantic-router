"""Production HTTP server for one verified, resident Decision model."""

from __future__ import annotations

from contextlib import asynccontextmanager

from .api import create_app
from .engine import DecisionEngine
from .metrics import RuntimeMetrics
from .runtime_factory import RuntimeLaunchConfig, assemble_runtime


def create_runtime_app(config: RuntimeLaunchConfig):
    """Load one pinned model before the server can report readiness."""

    metrics = RuntimeMetrics()
    assembled = assemble_runtime(config, metrics=metrics)
    app = create_app(
        DecisionEngine(assembled.backend),
        scheduler=assembled.scheduler,
        metrics=metrics,
        artifact_provenance=getattr(assembled, "artifact_provenance", None),
    )

    @asynccontextmanager
    async def lifespan(_app):
        try:
            yield
        finally:
            try:
                await assembled.backend.aclose()
            finally:
                app.state.decision_backend_closed = True

    app.router.lifespan_context = lifespan
    app.state.decision_backend = assembled.backend
    app.state.decision_backend_closed = False
    return app


def run_server(config: RuntimeLaunchConfig) -> None:
    """Serve the already-verified app with one model-owning worker."""

    import asyncio  # noqa: PLC0415 - preserve import at server startup

    import uvicorn  # noqa: PLC0415 - optional HTTP server dependency

    app = create_runtime_app(config)
    try:
        uvicorn.run(app, host=config.host, port=config.port, workers=1)
    finally:
        # Uvicorn can fail before entering ASGI lifespan.
        if not app.state.decision_backend_closed:
            asyncio.run(app.state.decision_backend.aclose())
