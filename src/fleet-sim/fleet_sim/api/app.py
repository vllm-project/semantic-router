"""FastAPI application for the vllm-sr-sim service."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from .routes import fleets, jobs, traces, workloads
from .trace_ingest import seed_example_traces_if_enabled


@asynccontextmanager
async def lifespan(_: FastAPI):
    seed_example_traces_if_enabled()
    yield


app = FastAPI(
    title="vllm-sr-sim API",
    description=(
        "HTTP API for the fleet-level LLM inference simulator. "
        "Supports trace management, fleet configuration, simulation jobs, "
        "and result retrieval for dashboard integration."
    ),
    version="0.1.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def apply_forwarded_prefix_root_path(request: Request, call_next):
    """Honor dashboard proxy prefixes so Swagger/OpenAPI assets resolve correctly."""

    forwarded_prefix = request.headers.get("x-forwarded-prefix", "").rstrip("/")
    if forwarded_prefix:
        request.scope["root_path"] = forwarded_prefix
    return await call_next(request)


# ── API routes ────────────────────────────────────────────────────────────────
app.include_router(traces.router, prefix="/api")
app.include_router(workloads.router, prefix="/api")
app.include_router(fleets.router, prefix="/api")
app.include_router(jobs.router, prefix="/api")


@app.get("/healthz", tags=["System"], summary="Health check")
async def healthz():
    return {"status": "ok", "service": "vllm-sr-sim"}


@app.get("/", include_in_schema=False)
async def root():
    return {
        "service": "vllm-sr-sim",
        "status": "ok",
        "docs": "/api/docs",
        "openapi": "/api/openapi.json",
    }
