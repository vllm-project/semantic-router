"""``POST /v1/system_one`` on the vLLM OpenAI-compatible server.

A ``vllm.endpoint_plugins`` entry point; vLLM loads it only when its name is in
``VLLM_PLUGINS``. At startup it opens the served package (``--model``, or
``VLLM_SR_DECISION2_PACKAGE``) with the package's own runtime and checks that
the engine runs ``Decision2Qwen3_5ForScoring``. Each request body is
``{"state": ..., "questions": {...}}`` and the response is the runtime's
``{"model", "answers", "usage"}``; malformed requests get HTTP 400.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from argparse import Namespace
from typing import Any

from ..compat import check_vllm_version, get_logger
from .gather import POSITIONS_KEY
from .package import describe, load_package
from .service import SystemOneService

logger = get_logger("decision2.endpoint")

ARCHITECTURE_NAME = "Decision2Qwen3_5ForScoring"
PACKAGE_ENV = "VLLM_SR_DECISION2_PACKAGE"
STATE_KEY = "vllm_sr_system_one"
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
        params = PoolingParams(task="plugin", extra_kwargs=positions)
        final = None
        async for output in engine_client.encode(tokens_input(ids), params, request_id):
            final = output
        if final is None:
            raise RuntimeError("engine returned no output")
        return final.outputs.data.float().tolist()

    return encode


class SystemOneEndpoint:
    name = "vllm_sr_system_one"
    required_tasks = ("plugin",)

    def attach_router(self, app: Any) -> None:
        from fastapi import APIRouter, Request
        from fastapi.responses import JSONResponse

        router = APIRouter()

        @router.post("/v1/system_one")
        async def system_one(raw: Request):
            service: SystemOneService | None = getattr(raw.app.state, STATE_KEY, None)
            if service is None:
                return JSONResponse(
                    {"error": "System One is not initialized"}, status_code=503
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
                    request_id=f"sysone-{uuid.uuid4().hex}",
                )
            except ValueError as exc:
                return JSONResponse({"error": str(exc)}, status_code=400)
            return JSONResponse(result)

        app.include_router(router)

    async def init_state(self, engine_client: Any, state: Any, args: Namespace) -> None:
        if engine_client is None:
            return
        check_vllm_version()
        architecture = getattr(engine_client.model_config, "architecture", None)
        if architecture != ARCHITECTURE_NAME:
            raise RuntimeError(
                f"/v1/system_one needs a {ARCHITECTURE_NAME} engine, not {architecture}"
            )
        path = os.environ.get(PACKAGE_ENV) or getattr(args, "model", None)
        if not path:
            raise RuntimeError(f"set --model or {PACKAGE_ENV} to the served package")
        package = await asyncio.to_thread(load_package, path)
        served = os.path.realpath(engine_client.model_config.model)
        if served != str(package.root):
            raise RuntimeError(
                f"engine serves {served}, System One opened {package.root}"
            )
        setattr(
            state, STATE_KEY, SystemOneService(package, engine_encoder(engine_client))
        )
        logger.info("System One ready (%s): %s", POSITIONS_KEY, describe(package))
