"""Compose independent provider protocols and optional fixture scenarios."""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from . import chat, classify, images, looper, messages, provider_boundary, responses
from .cache import SessionCacheTracker
from .provider_boundary import RequestStore
from .settings import Settings
from .shadow_control import ShadowControl
from .shadow_control import router as shadow_router


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    instance = FastAPI(title="provider-mocker")
    instance.state.settings = settings
    instance.state.request_store = RequestStore()
    instance.state.cache_tracker = SessionCacheTracker()
    instance.state.shadow_control = ShadowControl() if settings.shadow_control else None
    instance.state.dispatch_counts = {}
    for router in (
        provider_boundary.router,
        classify.router,
        chat.router,
        responses.router,
        messages.router,
        images.router,
    ):
        instance.include_router(router)
    if settings.shadow_control:
        instance.include_router(shadow_router)
    if settings.scenario == "looper":
        instance.include_router(looper.router)

    if settings.scenario == "cli":

        @instance.post("/{path:path}")
        async def cli_prefixed_chat(path: str, request: Request):
            if not path.endswith("/chat/completions"):
                return JSONResponse(status_code=404, content={"error": "not_found"})
            return await chat.chat_completions(request)

    @instance.middleware("http")
    async def authorize_canary(request: Request, call_next):
        if request.method == "POST" and (
            request.url.path.startswith("/v1/") or settings.scenario == "cli"
        ):
            expected = settings.expected_authorization
            if expected is not None:
                if request.headers.get("Authorization", "") != f"Bearer {expected}":
                    return JSONResponse(
                        status_code=401,
                        content={
                            "error": {
                                "type": "authentication_error",
                                "message": "invalid authorization",
                            }
                        },
                    )
                print("authorization-canary-received", flush=True)
            if settings.scenario == "cli":
                raw_path = request.scope.get(
                    "raw_path", request.url.path.encode("utf-8")
                )
                target = raw_path.decode("latin-1")
                query = request.scope.get("query_string", b"")
                if query:
                    target += "?" + query.decode("latin-1")
                print(target, flush=True)
        return await call_next(request)

    return instance


app = create_app()
