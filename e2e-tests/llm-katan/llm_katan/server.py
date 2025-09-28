"""
FastAPI server implementation for LLM Katan

Provides OpenAI-compatible endpoints for lightweight LLM serving.

Signed-off-by: Yossi Ovadia <yovadia@redhat.com>
"""

import asyncio
import json
import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Union

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel

from .config import ServerConfig
from .model import ModelBackend, create_backend

logger = logging.getLogger(__name__)


# Pydantic models for request/response
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    stream: Optional[bool] = False


class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[Dict]
    usage: Optional[Dict] = None


class ModelInfo(BaseModel):
    id: str
    object: str
    created: int
    owned_by: str


class ModelsResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


class HealthResponse(BaseModel):
    status: str
    model: str
    backend: str


class MetricsResponse(BaseModel):
    total_requests: int
    total_tokens_generated: int
    average_response_time: float
    model: str
    backend: str


# Global backend instance and metrics
backend: Optional[ModelBackend] = None
metrics = {
    "total_requests": 0,
    "total_tokens_generated": 0,
    "response_times": [],
    "start_time": time.time(),
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global backend
    config = app.state.config

    logger.info(f"🚀 Starting LLM Katan server with model: {config.model_name}")
    logger.info(f"🔧 Backend: {config.backend}")
    logger.info(f"📛 Served model name: {config.served_model_name}")

    # Create and load model backend
    backend = create_backend(config)
    await backend.load_model()

    logger.info("✅ LLM Katan server started successfully")
    yield

    logger.info("🛑 Shutting down LLM Katan server")
    backend = None


def create_app(config: ServerConfig) -> FastAPI:
    """Create FastAPI application"""
    app = FastAPI(
        title="LLM Katan - Lightweight LLM Server",
        description="A lightweight LLM serving package for testing and development",
        version="0.1.4",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # Store config in app state
    app.state.config = config

    @app.get("/health", response_model=HealthResponse)
    async def health():
        """Health check endpoint"""
        return HealthResponse(
            status="ok",
            model=config.served_model_name,
            backend=config.backend,
        )

    @app.get("/v1/models", response_model=ModelsResponse)
    async def list_models():
        """List available models"""
        if backend is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        model_info = backend.get_model_info()
        return ModelsResponse(data=[ModelInfo(**model_info)])

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest, http_request: Request):
        """Chat completions endpoint (OpenAI compatible)"""
        if backend is None:
            raise HTTPException(status_code=503, detail="Model not loaded")

        start_time = time.time()
        client_ip = http_request.client.host

        # Log the incoming request with model and prompt info
        user_prompt = request.messages[-1].content if request.messages else "No prompt"
        logger.info(
            f"💬 Chat request from {client_ip} | Model: {config.served_model_name} | "
            f"Prompt: '{user_prompt[:100]}{'...' if len(user_prompt) > 100 else ''}'"
        )

        try:
            # Convert messages to dict format
            messages = [
                {"role": msg.role, "content": msg.content} for msg in request.messages
            ]

            # Update metrics
            metrics["total_requests"] += 1

            if request.stream:
                # Streaming response
                async def generate_stream():
                    async for chunk in backend.generate(
                        messages=messages,
                        max_tokens=request.max_tokens,
                        temperature=request.temperature,
                        stream=True,
                    ):
                        yield f"data: {json.dumps(chunk)}\n\n"
                    yield "data: [DONE]\n\n"

                return StreamingResponse(
                    generate_stream(),
                    media_type="text/plain",
                    headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
                )
            else:
                # Non-streaming response
                response_generator = backend.generate(
                    messages=messages,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    stream=False,
                )
                response = await response_generator.__anext__()

                # Log response and update metrics
                response_time = time.time() - start_time
                metrics["response_times"].append(response_time)
                if "choices" in response and response["choices"]:
                    generated_text = (
                        response["choices"][0].get("message", {}).get("content", "")
                    )
                    token_count = len(generated_text.split())  # Rough token estimate
                    metrics["total_tokens_generated"] += token_count

                    logger.info(
                        f"✅ Response sent | Model: {config.served_model_name} | "
                        f"Tokens: ~{token_count} | Time: {response_time:.2f}s | "
                        f"Response: '{generated_text[:100]}{'...' if len(generated_text) > 100 else ''}'"
                    )

                return response

        except Exception as e:
            response_time = time.time() - start_time
            logger.error(
                f"❌ Error in chat completions | Model: {config.served_model_name} | "
                f"Time: {response_time:.2f}s | Error: {str(e)}"
            )
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/metrics")
    async def get_metrics():
        """Prometheus-style metrics endpoint"""
        avg_response_time = (
            sum(metrics["response_times"]) / len(metrics["response_times"])
            if metrics["response_times"]
            else 0.0
        )

        uptime = time.time() - metrics["start_time"]

        # Return Prometheus-style metrics
        prometheus_metrics = f"""# HELP llm_katan_requests_total Total number of requests processed
# TYPE llm_katan_requests_total counter
llm_katan_requests_total{{model="{config.served_model_name}",backend="{config.backend}"}} {metrics["total_requests"]}

# HELP llm_katan_tokens_generated_total Total number of tokens generated
# TYPE llm_katan_tokens_generated_total counter
llm_katan_tokens_generated_total{{model="{config.served_model_name}",backend="{config.backend}"}} {metrics["total_tokens_generated"]}

# HELP llm_katan_response_time_seconds Average response time in seconds
# TYPE llm_katan_response_time_seconds gauge
llm_katan_response_time_seconds{{model="{config.served_model_name}",backend="{config.backend}"}} {avg_response_time:.4f}

# HELP llm_katan_uptime_seconds Server uptime in seconds
# TYPE llm_katan_uptime_seconds gauge
llm_katan_uptime_seconds{{model="{config.served_model_name}",backend="{config.backend}"}} {uptime:.2f}
"""

        return PlainTextResponse(content=prometheus_metrics, media_type="text/plain")

    @app.get("/")
    async def root():
        """Root endpoint"""
        return {
            "message": "LLM Katan - Lightweight LLM Server",
            "version": "0.1.4",
            "model": config.served_model_name,
            "backend": config.backend,
            "docs": "/docs",
            "metrics": "/metrics",
        }

    return app


async def run_server(config: ServerConfig):
    """Run the server with uvicorn"""
    import uvicorn

    app = create_app(config)

    uvicorn_config = uvicorn.Config(
        app,
        host=config.host,
        port=config.port,
        log_level="info",
        access_log=True,
    )

    server = uvicorn.Server(uvicorn_config)
    await server.serve()
