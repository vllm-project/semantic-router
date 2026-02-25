#!/usr/bin/env python3
"""
ML Training Service - HTTP API wrapping benchmark.py and train.py.

Provides REST endpoints for:
- POST /api/benchmark  - Run benchmark (streams progress via SSE)
- POST /api/train      - Run training  (streams progress via SSE)
- GET  /api/health     - Health check

This service is designed to run as a sidecar container alongside the
Go dashboard backend. The Go backend calls these HTTP endpoints instead
of spawning Python subprocesses directly, making the architecture
production-ready without requiring Python in the dashboard container.

Usage:
    python server.py --port 8686 --data-dir /app/data

Environment variables:
    ML_SERVICE_PORT      - Port to listen on (default: 8686)
    ML_SERVICE_DATA_DIR  - Data directory for job outputs (default: ./data)
"""

import argparse
import json
import logging
import os
import queue
import sys
import threading
import traceback
from pathlib import Path
from typing import List, Optional

# --- FastAPI / Uvicorn ---
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel, Field
    import uvicorn
except ImportError:
    print(
        "Error: FastAPI and Uvicorn required.\n"
        "Install with: pip install fastapi uvicorn[standard]"
    )
    sys.exit(1)

# --- Local imports (benchmark + train modules) ---
# These are the existing modules in src/training/ml_model_selection/
from benchmark import run_benchmark_pipeline
from train import run_training_pipeline
from models import TORCH_AVAILABLE

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ml-service")

# ---------------------------------------------------------------------------
# Pydantic request models
# ---------------------------------------------------------------------------


class BenchmarkRequest(BaseModel):
    """Request body for POST /api/benchmark."""

    queries_path: str = Field(..., description="Path to JSONL queries file")
    models_yaml_path: str = Field(..., description="Path to models YAML config")
    output_dir: str = Field(..., description="Directory to write output files")
    concurrency: int = Field(1, ge=0, le=64, description="0 = use default (1)")
    max_tokens: int = Field(1024, ge=0, description="0 = use default (1024)")
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    concise: bool = False
    limit: int = Field(0, ge=0, description="0 = no limit")


class TrainRequest(BaseModel):
    """Request body for POST /api/train."""

    data_file: str = Field(..., description="Path to benchmark JSONL output")
    output_dir: str = Field(..., description="Directory to write model files")
    algorithms: List[str] = Field(default=["knn", "kmeans", "svm", "mlp"])
    device: str = Field("cpu")
    embedding_model: str = Field("qwen3")
    cache_dir: str = Field("", description="Cache dir for embeddings")
    quality_weight: float = Field(0.9, ge=0.0, le=1.0)
    batch_size: int = Field(32, ge=0, description="0 = use default (32)")
    # KNN
    knn_k: int = Field(5, ge=0, description="0 = use default (5)")
    # KMeans
    kmeans_clusters: int = Field(8, ge=0, description="0 = use default (8)")
    # SVM
    svm_kernel: str = Field("rbf")
    svm_gamma: float = Field(1.0, ge=0.0)
    # MLP
    mlp_hidden_sizes: str = Field("256,128")
    mlp_epochs: int = Field(100, ge=0, description="0 = use default (100)")
    mlp_learning_rate: float = Field(0.001, ge=0.0)
    mlp_dropout: float = Field(0.1, ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Progress helper
# ---------------------------------------------------------------------------


class ProgressReporter:
    """Thread-safe progress reporter that queues SSE events."""

    def __init__(self):
        self.q: queue.Queue = queue.Queue()
        self._done = False

    def send(self, percent: int, step: str, message: str):
        if self._done:
            return
        event = {
            "percent": percent,
            "step": step,
            "message": message,
        }
        self.q.put(event)

    def done(
        self, success: bool, message: str = "", output_files: Optional[List[str]] = None
    ):
        event = {
            "percent": 100,
            "step": "completed" if success else "failed",
            "message": message,
            "done": True,
            "success": success,
            "output_files": output_files or [],
        }
        self.q.put(event)
        self._done = True

    def stream(self):
        """Generator that yields SSE-formatted events."""
        while True:
            try:
                event = self.q.get(timeout=60)
            except queue.Empty:
                # Send keepalive
                yield "event: keepalive\ndata: {}\n\n"
                continue

            data = json.dumps(event)
            yield f"data: {data}\n\n"

            if event.get("done"):
                break


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="ML Training Service",
    description="Sidecar service for ML benchmark and training operations",
    version="1.0.0",
)


@app.get("/api/health")
def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "torch_available": TORCH_AVAILABLE,
        "python_version": sys.version,
    }


@app.post("/api/benchmark")
def api_benchmark(req: BenchmarkRequest):
    """
    Run benchmark and stream progress as SSE.

    Delegates to the shared run_benchmark_pipeline() from benchmark.py,
    passing an on_progress callback for real-time SSE updates.
    """
    if not Path(req.queries_path).exists():
        raise HTTPException(400, f"Queries file not found: {req.queries_path}")
    if not Path(req.models_yaml_path).exists():
        raise HTTPException(400, f"Models config not found: {req.models_yaml_path}")

    os.makedirs(req.output_dir, exist_ok=True)

    progress = ProgressReporter()
    output_file = os.path.join(req.output_dir, "benchmark_output.jsonl")

    def _run():
        try:
            # Apply defaults for zero values (Go sends 0 for unset fields)
            concurrency = req.concurrency if req.concurrency > 0 else 4
            max_tokens = req.max_tokens if req.max_tokens > 0 else 1024

            results = run_benchmark_pipeline(
                queries_path=req.queries_path,
                models_yaml_path=req.models_yaml_path,
                output_path=output_file,
                concurrency=concurrency,
                max_tokens=max_tokens,
                temperature=req.temperature,
                concise=req.concise,
                limit=req.limit,
                show_progress=False,  # We use on_progress callback instead
                on_progress=lambda pct, step, msg: progress.send(pct, step, msg),
            )
            progress.done(
                True,
                f"Benchmark complete: {len(results)} results",
                output_files=[output_file],
            )
        except Exception as e:
            logger.error(f"Benchmark failed: {traceback.format_exc()}")
            progress.done(False, str(e))

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    return StreamingResponse(
        progress.stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/train")
def api_train(req: TrainRequest):
    """
    Run ML training and stream progress as SSE.

    Delegates to the shared run_training_pipeline() from train.py,
    calling it once per algorithm (or once with "all") and passing
    an on_progress callback for real-time SSE updates.
    """
    if not Path(req.data_file).exists():
        raise HTTPException(400, f"Data file not found: {req.data_file}")

    os.makedirs(req.output_dir, exist_ok=True)

    progress = ProgressReporter()

    def _run():
        try:
            algorithms = req.algorithms or ["knn", "kmeans", "svm", "mlp"]
            all_four = set(algorithms) == {"knn", "kmeans", "svm", "mlp"}
            mlp_hidden_sizes = [int(x.strip()) for x in req.mlp_hidden_sizes.split(",")]
            cache_dir = req.cache_dir or os.path.join(req.output_dir, ".cache")

            # Apply defaults for zero values (Go sends 0 for unset fields)
            batch_size = req.batch_size if req.batch_size > 0 else 32
            knn_k = req.knn_k if req.knn_k > 0 else 5
            kmeans_clusters = req.kmeans_clusters if req.kmeans_clusters > 0 else 8
            mlp_epochs = req.mlp_epochs if req.mlp_epochs > 0 else 100
            quality_weight = req.quality_weight if req.quality_weight > 0 else 0.9
            svm_gamma = req.svm_gamma if req.svm_gamma > 0 else 1.0
            mlp_lr = req.mlp_learning_rate if req.mlp_learning_rate > 0 else 0.001
            mlp_dropout = req.mlp_dropout if req.mlp_dropout > 0 else 0.1
            device = req.device if req.device else "cpu"
            embedding_model = req.embedding_model if req.embedding_model else "qwen3"
            svm_kernel = req.svm_kernel if req.svm_kernel else "rbf"

            # Determine runs: single "all" or one per algorithm
            runs = ["all"] if all_four else algorithms
            total_runs = len(runs)

            all_output_files = []
            for i, alg in enumerate(runs):
                # Scale progress across runs: each run gets an equal share of 5-90%
                base_pct = 5 + (i * 85 // total_runs)

                def on_progress(
                    pct, step, msg, _base=base_pct, _share=85 // total_runs
                ):
                    # Map pipeline's 0-100% into this run's slice of overall progress
                    scaled = _base + (pct * _share // 100)
                    progress.send(min(scaled, 95), step, msg)

                output_files = run_training_pipeline(
                    data_file=req.data_file,
                    output_dir=req.output_dir,
                    embedding_model=embedding_model,
                    cache_dir=cache_dir,
                    quality_weight=quality_weight,
                    batch_size=batch_size,
                    device=device,
                    knn_k=knn_k,
                    kmeans_clusters=kmeans_clusters,
                    svm_kernel=svm_kernel,
                    svm_gamma=svm_gamma,
                    mlp_hidden_sizes=mlp_hidden_sizes,
                    mlp_epochs=mlp_epochs,
                    mlp_learning_rate=mlp_lr,
                    mlp_dropout=mlp_dropout,
                    algorithm=alg,
                    on_progress=on_progress,
                )
                all_output_files.extend(output_files)

            if not all_output_files:
                progress.done(False, "No model files were generated")
                return

            progress.done(
                True,
                f"Training complete: {len(all_output_files)} model(s) generated",
                output_files=all_output_files,
            )

        except Exception as e:
            logger.error(f"Training failed: {traceback.format_exc()}")
            progress.done(False, str(e))

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    return StreamingResponse(
        progress.stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="ML Training Service")
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("ML_SERVICE_PORT", "8686")),
        help="Port to listen on (default: 8686)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=os.environ.get("ML_SERVICE_HOST", "0.0.0.0"),
        help="Host to bind to (default: 0.0.0.0)",
    )
    args = parser.parse_args()

    logger.info(f"Starting ML Training Service on {args.host}:{args.port}")
    logger.info(f"PyTorch available: {TORCH_AVAILABLE}")
    logger.info(f"Python version: {sys.version}")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
