"""Prove Operator reconciliation produced a working routed data-plane request."""

import argparse
import json
import time
import uuid
from http import HTTPStatus
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def validate_response(payload, headers, *, model, nonce):
    if headers.get("x-vsr-selected-decision") != "operator_ci_route":
        raise ValueError("router did not execute the declared Operator route")
    if headers.get("x-vsr-response-path") != "upstream":
        raise ValueError("Operator request did not reach the upstream provider")
    if headers.get("x-vsr-selected-model") != model:
        raise ValueError("router did not report the expected selected backend model")
    if payload.get("model") != model:
        raise ValueError("response model does not match the selected model")
    echo = json.loads(payload["choices"][0]["message"]["content"])
    if (
        echo.get("mock") != "provider-mocker"
        or echo.get("protocol") != "chat_completions"
    ):
        raise ValueError("response did not come from the request-observing backend")
    if echo.get("model") != model or echo.get("user") != [nonce]:
        raise ValueError(
            "backend did not receive the rewritten model and this request payload"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    nonce = "operator-reconciliation-" + uuid.uuid4().hex
    request = Request(
        args.url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(
            {"model": "auto", "messages": [{"role": "user", "content": nonce}]}
        ).encode(),
        headers={"Content-Type": "application/json", "x-vsr-debug": "true"},
        method="POST",
    )
    result = {"id": "operator-routed-request", "status": "failed"}
    try:
        deadline = time.monotonic() + 120
        while True:
            try:
                with urlopen(request, timeout=60) as response:
                    payload = json.load(response)
                    validate_response(
                        payload, response.headers, model=args.model, nonce=nonce
                    )
                break
            except HTTPError as error:
                body = error.read().decode("utf-8", errors="replace")
                if (
                    error.code < HTTPStatus.INTERNAL_SERVER_ERROR
                    or time.monotonic() >= deadline
                ):
                    raise RuntimeError(
                        f"Operator routed request returned HTTP {error.code}: {body}"
                    ) from error
                time.sleep(1)
            except URLError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(1)
        result["status"] = "passed"
        result["selected_model"] = args.model
    except Exception as error:
        result["error"] = str(error)
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps({"cases": [result], "expected_cases": [result["id"]]}, indent=2)
            + "\n"
        )


if __name__ == "__main__":
    main()
