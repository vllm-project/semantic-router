"""Expose completed connector results on buffered OpenAI HTTP responses."""

from __future__ import annotations

import json

_MAX_RESPONSE_BYTES = 1024 * 1024
_STATUSES = {"applied", "fallback_reprefill", "unsupported"}


class KVTransferResponseMiddleware:
    """vLLM --middleware adapter for non-streaming Chat and Completion responses."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope.get("path") not in (
            "/v1/chat/completions",
            "/v1/completions",
        ):
            return await self.app(scope, receive, send)
        start = None
        chunks = []
        size = 0

        async def capture(message):
            nonlocal start, size
            if message["type"] == "http.response.start":
                if any(
                    key.lower() == b"content-type"
                    and value.split(b";", 1)[0].strip().lower() == b"application/json"
                    for key, value in message["headers"]
                ):
                    start = message
                    return
            elif message["type"] == "http.response.body" and start is not None:
                chunks.append(message)
                size += len(message.get("body", b""))
                if size <= _MAX_RESPONSE_BYTES and message.get("more_body", False):
                    return
                if size <= _MAX_RESPONSE_BYTES:
                    try:
                        body = json.loads(b"".join(c.get("body", b"") for c in chunks))
                        result = (body.get("kv_transfer_params") or {}).get(
                            "vsr_kv_transfer"
                        )
                        if (
                            isinstance(result, dict)
                            and result.get("status") in _STATUSES
                        ):
                            params = dict(body["kv_transfer_params"])
                            del params["vsr_kv_transfer"]
                            body["kv_transfer_params"] = params or None
                            encoded = json.dumps(body).encode()
                            chunks[:] = [
                                {"type": "http.response.body", "body": encoded}
                            ]
                            start["headers"] = [
                                (key, value)
                                for key, value in start["headers"]
                                if key.lower() != b"content-length"
                            ] + [(b"content-length", str(len(encoded)).encode())]
                            start["headers"].extend(
                                [
                                    (
                                        b"x-vsr-kv-transfer-status",
                                        result["status"].encode(),
                                    ),
                                    (
                                        b"x-vsr-kv-transfer-reason",
                                        str(result.get("reason", "unknown")).encode(),
                                    ),
                                ]
                            )
                    except (ValueError, AttributeError, TypeError):
                        pass
                await send(start)
                start = None
                for chunk in chunks:
                    await send(chunk)
                chunks.clear()
                return
            await send(message)

        await self.app(scope, receive, capture)
