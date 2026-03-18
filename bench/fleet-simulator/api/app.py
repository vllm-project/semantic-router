"""FastAPI application for the inference-fleet-sim dashboard API."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .routes import traces, fleets, workloads, jobs

DASHBOARD_DIR = Path(__file__).parent.parent / "dashboard"

app = FastAPI(
    title="inference-fleet-sim API",
    description=(
        "REST API for the fleet-level LLM inference simulator. "
        "Supports trace management, fleet configuration, simulation jobs, "
        "and result retrieval for dashboard visualisation."
    ),
    version="0.1.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── API routes ────────────────────────────────────────────────────────────────
app.include_router(traces.router,    prefix="/api")
app.include_router(workloads.router, prefix="/api")
app.include_router(fleets.router,    prefix="/api")
app.include_router(jobs.router,      prefix="/api")


# ── Dashboard static files ───────────────────────────────────────────────────
if DASHBOARD_DIR.exists():
    app.mount("/static", StaticFiles(directory=DASHBOARD_DIR), name="static")

    @app.get("/", include_in_schema=False)
    async def serve_dashboard():
        return FileResponse(DASHBOARD_DIR / "index.html")
