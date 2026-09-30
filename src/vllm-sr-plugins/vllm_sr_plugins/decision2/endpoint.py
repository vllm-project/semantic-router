"""``POST /v1/decisions`` on the vLLM OpenAI-compatible server.

A ``vllm.endpoint_plugins`` entry point; vLLM loads it only when its name is in
``VLLM_PLUGINS``. At startup it opens the served package (``--model``, or
``VLLM_SR_DECISION2_PACKAGE``) with the package's own runtime and checks that
the engine runs ``Decision2Qwen3_5ForScoring``. Each request body is a System
One call, ``{"state": ..., "questions": {...}}``, and the response is the
runtime's ``{"model", "answers", "usage"}``; malformed requests get HTTP 400.
``/v1/system_one`` is an alias with the same contract.
"""

# No `from __future__ import annotations`: FastAPI must see the real Request
# class on the route handler, which is imported lazily below.
import asyncio
import os
import uuid
from argparse import Namespace
from typing import Any

from ..compat import check_vllm_version, get_logger
from .gather import POOLING_TASK, POSITIONS_KEY
from .package import describe, load_package
from .service import SystemOneService

logger = get_logger("decision2.endpoint")

ARCHITECTURE_NAME = "Decision2Qwen3_5ForScoring"
PACKAGE_ENV = "VLLM_SR_DECISION2_PACKAGE"
STATE_KEY = "vllm_sr_decisions"
ROUTES = ("/v1/decisions", "/v1/system_one")
REQUEST_KEYS = {"state", "questions", "model"}


def engine_encoder(engine_client: Any):
    from vllm import PoolingParams

    try:
        from vllm.inputs import tokens_input
    except ImportError:  # older builds accept a raw TokensPrompt

        def tokens_input(ids: list[int]) -> dict[str, Any]:
            return {"prompt_token_ids": ids}

    async def encode(
        ids: list[int], positions: dict[str, Any], request_id: str
    ) -> list[float]:
        params = PoolingParams(task=POOLING_TASK, extra_kwargs=positions)
        final = None
        async for output in engine_client.encode(tokens_input(ids), params, request_id):
            final = output
        if final is None:
            raise RuntimeError("engine returned no output")
        return final.outputs.data.float().tolist()

    return encode


class DecisionsEndpoint:
    name = "vllm_sr_decisions"
    required_tasks = (POOLING_TASK,)

    def attach_router(self, app: Any) -> None:
        from fastapi import APIRouter, Request
        from fastapi.responses import JSONResponse

        async def decisions(raw: Request):
            service: SystemOneService | str | None = getattr(
                raw.app.state, STATE_KEY, None
            )
            if not isinstance(service, SystemOneService):
                return JSONResponse(
                    {"error": service or "the decisions endpoint is not initialized"},
                    status_code=503,
                )
            try:
                body = await raw.json()
            except ValueError:
                return JSONResponse(
                    {"error": "request body must be JSON"}, status_code=400
                )
            if (
                not isinstance(body, dict)
                or not {"state", "questions"} <= set(body) <= REQUEST_KEYS
            ):
                return JSONResponse(
                    {"error": "request must be an object with state and questions"},
                    status_code=400,
                )
            if body.get("model") not in (None, service.package.model_name):
                return JSONResponse(
                    {"error": f"this server answers for {service.package.model_name}"},
                    status_code=404,
                )
            try:
                result = await service.system_one(
                    state=body["state"],
                    questions=body["questions"],
                    request_id=f"decisions-{uuid.uuid4().hex}",
                )
            except ValueError as exc:
                return JSONResponse({"error": str(exc)}, status_code=400)
            return JSONResponse(result)

        router = APIRouter()
        for path in ROUTES:
            router.add_api_route(path, decisions, methods=["POST"])
        app.include_router(router)

    async def init_state(self, engine_client: Any, state: Any, args: Namespace) -> None:
        if engine_client is None:
            return
        architecture = getattr(engine_client.model_config, "architecture", None)
        if architecture != ARCHITECTURE_NAME:
            # Installed next to other models: leave their servers running.
            reason = (
                f"{ROUTES[0]} needs a {ARCHITECTURE_NAME} engine, not {architecture}"
            )
            setattr(state, STATE_KEY, reason)
            logger.warning("%s; the route answers 503", reason)
            return
        check_vllm_version()
        path = os.environ.get(PACKAGE_ENV) or getattr(args, "model", None)
        if not path:
            raise RuntimeError(f"set --model or {PACKAGE_ENV} to the served package")
        package = await asyncio.to_thread(load_package, path)
        served = os.path.realpath(engine_client.model_config.model)
        if served != str(package.root):
            raise RuntimeError(
                f"engine serves {served}, the endpoint opened {package.root}"
            )
        setattr(
            state, STATE_KEY, SystemOneService(package, engine_encoder(engine_client))
        )
        logger.info(
            "Decisions endpoint ready on %s (%s): %s",
            ", ".join(ROUTES),
            POSITIONS_KEY,
            describe(package),
        )
