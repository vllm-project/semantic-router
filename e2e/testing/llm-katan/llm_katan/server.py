"""
FastAPI server implementation for LLM Katan

Provides OpenAI-compatible endpoints for lightweight LLM serving.

Signed-off-by: Yossi Ovadia <yovadia@redhat.com>
"""

import json
import logging
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from http import HTTPStatus

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from .config import ServerConfig

try:
    from importlib.metadata import PackageNotFoundError, version

    __version__ = version("llm-katan")
except PackageNotFoundError:
    __version__ = "unknown"
from .model import ModelBackend, create_backend

logger = logging.getLogger(__name__)

LOG_PREVIEW_CHARS = 100
INVALID_REQUEST_STATUSES = frozenset(
    {
        HTTPStatus.BAD_REQUEST,
        HTTPStatus.NOT_FOUND,
        HTTPStatus.CONFLICT,
        HTTPStatus.UNPROCESSABLE_ENTITY,
    }
)


# Pydantic models for request/response
class ChatMessage(BaseModel):
    role: str
    content: str | list[dict[str, object]] | None = None


class ChatCompletionRequest(BaseModel):
    model: str = Field(min_length=1)
    messages: list[ChatMessage] = Field(min_length=1)
    max_tokens: int | None = None
    temperature: float | None = None
    stream: bool = False


class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: list[dict]
    usage: dict | None = None


class ModelInfo(BaseModel):
    id: str
    object: str
    created: int
    owned_by: str


class ModelsResponse(BaseModel):
    object: str = "list"
    data: list[ModelInfo]


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


metrics = {
    "total_requests": 0,
    "total_tokens_generated": 0,
    "response_times": [],
    "start_time": time.time(),
}


def _openai_error(
    status_code: int, message: str, param: str | None = None
) -> JSONResponse:
    """Return the error envelope exposed by OpenAI-compatible model servers."""
    if status_code == HTTPStatus.UNAUTHORIZED:
        error_type = "authentication_error"
    elif status_code == HTTPStatus.FORBIDDEN:
        error_type = "permission_error"
    elif status_code == HTTPStatus.TOO_MANY_REQUESTS:
        error_type = "rate_limit_error"
    elif status_code in INVALID_REQUEST_STATUSES:
        error_type = "invalid_request_error"
    else:
        error_type = "server_error"
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": error_type,
                "param": param,
                "code": None,
            }
        },
    )


def _validation_parameter(error: dict) -> str | None:
    location = [str(component) for component in error.get("loc", ())]
    if location and location[0] == "body":
        location = location[1:]
    return ".".join(location) or None


def _content_text(content: str | list[dict[str, object]] | None) -> str:
    """Render valid Chat content parts for the text-only Katan backends."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content

    rendered = []
    for part in content:
        part_type = part.get("type")
        if part_type in {"text", "input_text"} and isinstance(part.get("text"), str):
            rendered.append(part["text"])
        elif part_type == "refusal" and isinstance(part.get("refusal"), str):
            rendered.append(part["refusal"])
        else:
            rendered.append(json.dumps(part, sort_keys=True, separators=(",", ":")))
    return "\n".join(rendered)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    config = app.state.config

    logger.info(f"🚀 Starting LLM Katan server with model: {config.model_name}")
    logger.info(f"🔧 Backend: {config.backend}")
    logger.info(f"📛 Served model name: {config.served_model_name}")

    # Create and load model backend
    app.state.backend = create_backend(config)
    await app.state.backend.load_model()

    logger.info("✅ LLM Katan server started successfully")
    yield

    logger.info("🛑 Shutting down LLM Katan server")
    app.state.backend = None


def _loaded_backend(app: FastAPI) -> ModelBackend:
    model_backend = getattr(app.state, "backend", None)
    if model_backend is None:
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            detail="Model not loaded",
        )
    return model_backend


async def _request_validation_error_handler(
    _request: Request, exc: RequestValidationError
) -> JSONResponse:
    first_error = exc.errors()[0] if exc.errors() else {}
    message = str(first_error.get("msg") or "Request validation failed")
    return _openai_error(
        HTTPStatus.BAD_REQUEST,
        message,
        _validation_parameter(first_error),
    )


async def _http_error_handler(
    _request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    return _openai_error(exc.status_code, str(exc.detail))


async def _stream_chunks(
    model_backend: ModelBackend,
    messages: list[dict[str, str]],
    request: ChatCompletionRequest,
) -> AsyncGenerator[str, None]:
    async for chunk in model_backend.generate(
        messages=messages,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        stream=True,
    ):
        yield f"data: {json.dumps(chunk)}\n\n"
    yield "data: [DONE]\n\n"


def _streaming_response(
    model_backend: ModelBackend,
    messages: list[dict[str, str]],
    request: ChatCompletionRequest,
) -> StreamingResponse:
    return StreamingResponse(
        _stream_chunks(model_backend, messages, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type",
        },
    )


async def _buffered_response(
    model_backend: ModelBackend,
    messages: list[dict[str, str]],
    request: ChatCompletionRequest,
    config: ServerConfig,
    start_time: float,
) -> dict:
    response_generator = model_backend.generate(
        messages=messages,
        max_tokens=request.max_tokens,
        temperature=request.temperature,
        stream=False,
    )
    response = await response_generator.__anext__()

    response_time = time.time() - start_time
    metrics["response_times"].append(response_time)
    if response.get("choices"):
        generated_text = response["choices"][0].get("message", {}).get("content", "")
        token_count = len(generated_text.split())
        metrics["total_tokens_generated"] += token_count
        preview = generated_text[:LOG_PREVIEW_CHARS]
        ellipsis = "..." if len(generated_text) > LOG_PREVIEW_CHARS else ""
        logger.info(
            f"✅ Response sent | Model: {config.served_model_name} | "
            f"Tokens: ~{token_count} | Time: {response_time:.2f}s | "
            f"Response: '{preview}{ellipsis}'"
        )
    return response


async def _chat_completion(
    request: ChatCompletionRequest,
    http_request: Request,
    config: ServerConfig,
    model_backend: ModelBackend,
):
    start_time = time.time()
    user_prompt = _content_text(request.messages[-1].content)
    prompt_preview = user_prompt[:LOG_PREVIEW_CHARS]
    ellipsis = "..." if len(user_prompt) > LOG_PREVIEW_CHARS else ""
    client_host = http_request.client.host if http_request.client else "unknown"
    logger.info(
        f"💬 Chat request from {client_host} | "
        f"Model: {config.served_model_name} | Prompt: '{prompt_preview}{ellipsis}'"
    )

    messages = [
        {"role": message.role, "content": _content_text(message.content)}
        for message in request.messages
    ]
    metrics["total_requests"] += 1

    try:
        if request.stream:
            return _streaming_response(model_backend, messages, request)
        return await _buffered_response(
            model_backend,
            messages,
            request,
            config,
            start_time,
        )
    except Exception as error:
        response_time = time.time() - start_time
        logger.error(
            f"❌ Error in chat completions | Model: {config.served_model_name} | "
            f"Time: {response_time:.2f}s | Error: {error!s}"
        )
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail=str(error),
        ) from error


def create_app(config: ServerConfig) -> FastAPI:
    """Create FastAPI application"""
    app = FastAPI(
        title="LLM Katan - Lightweight LLM Server",
        description="A lightweight LLM serving package for testing and development",
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # Store config in app state
    app.state.config = config
    app.add_exception_handler(
        RequestValidationError,
        _request_validation_error_handler,
    )
    app.add_exception_handler(StarletteHTTPException, _http_error_handler)

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
        model_info = _loaded_backend(app).get_model_info()
        return ModelsResponse(data=[ModelInfo(**model_info)])

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest, http_request: Request):
        """Chat completions endpoint (OpenAI compatible)"""
        return await _chat_completion(
            request,
            http_request,
            config,
            _loaded_backend(app),
        )

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
            "version": __version__,
            "model": config.served_model_name,
            "backend": config.backend,
            "docs": "/docs",
            "metrics": "/metrics",
        }

    return app


async def run_server(config: ServerConfig):
    """Run the server with uvicorn"""
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
