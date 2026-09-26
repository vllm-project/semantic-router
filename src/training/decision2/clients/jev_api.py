"""Collect pinned TypeSafe System One responses without storing credentials.

Input JSONL rows contain id, state, and questions. The bearer token is read
once from stdin, so it does not appear in arguments, environment or receipts.
Run this collector only in the accelerator evaluation environment.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
import time
from pathlib import Path
from urllib import error, request

API_URL = "https://api.typesafe.ai/v1/systemone"
RETRYABLE = {429, 529, 500, 502, 503, 504}


def canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()


def run(
    input_path: Path,
    output_path: Path,
    model: str,
    token: str,
    max_requests: int,
    resume: bool = False,
) -> None:
    if not token:
        raise ValueError("Empty API token")
    if output_path.exists() and not resume:
        raise FileExistsError(f"Refusing to overwrite {output_path}")
    rows = [
        json.loads(line) for line in input_path.read_text().splitlines() if line.strip()
    ]
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Input must have unique IDs and at least one row")
    rows = rows[:max_requests] if max_requests else rows
    if resume and output_path.exists():
        completed = [
            json.loads(line)
            for line in output_path.read_text().splitlines()
            if line.strip()
        ]
        if len(completed) > len(rows):
            raise ValueError("Receipt count exceeds input count")
        for index, receipt in enumerate(completed):
            row = rows[index]
            body = {
                "state": row["state"],
                "model": model,
                "questions": row["questions"],
            }
            if (
                receipt.get("id") != row["id"]
                or receipt.get("input_sha256")
                != hashlib.sha256(canonical(body)).hexdigest()
                or receipt.get("requested_model") != model
                or receipt.get("returned_model") != model
                or receipt.get("http_status") != 200
            ):
                raise ValueError(
                    f"Receipt {index + 1} does not match a completed input"
                )
        rows = rows[len(completed) :]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    with output_path.open("a" if resume else "x", encoding="utf-8") as output:
        for row in rows:
            body = {
                "state": row["state"],
                "model": model,
                "questions": row["questions"],
            }
            data = canonical(body)
            req = request.Request(
                API_URL,
                data=data,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            response_body = None
            status = None
            latency = None
            for attempt in range(6):
                tick = time.perf_counter()
                try:
                    with request.urlopen(req, timeout=90) as response:
                        status = response.status
                        response_body = json.load(response)
                    latency = time.perf_counter() - tick
                    break
                except error.HTTPError as exc:
                    status = exc.code
                    latency = time.perf_counter() - tick
                    # Error pages can be empty or HTML, and may echo request
                    # headers. Never persist their body or credentials.
                    exc.read()
                    response_body = {"error": {"http_status": status}}
                    if status not in RETRYABLE or attempt == 5:
                        break
                    retry_after = exc.headers.get("Retry-After")
                    try:
                        delay = (
                            max(0.0, min(60.0, float(retry_after)))
                            if retry_after
                            else 0.5 * 2**attempt
                        )
                    except ValueError:
                        delay = 0.5 * 2**attempt
                    time.sleep(delay)
                except (error.URLError, TimeoutError) as exc:
                    status = "transport_error"
                    latency = time.perf_counter() - tick
                    response_body = {"error": type(exc).__name__}
                    if attempt == 5:
                        break
                    time.sleep(0.5 * 2**attempt)
            if status != 200 or not isinstance(response_body, dict):
                raise RuntimeError(
                    f"{row['id']}: request failed after retries (status={status})"
                )
            receipt = {
                "id": row["id"],
                "input_sha256": hashlib.sha256(data).hexdigest(),
                "requested_model": model,
                "returned_model": (
                    response_body.get("model")
                    if isinstance(response_body, dict)
                    else None
                ),
                "http_status": status,
                "latency_seconds": latency,
                "response": response_body,
                "run_started_utc": started,
            }
            output.write(
                json.dumps(receipt, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
            output.flush()
            print(
                json.dumps(
                    {
                        "id": row["id"],
                        "status": status,
                        "model": receipt["returned_model"],
                    }
                ),
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="jev-1.13.0")
    parser.add_argument("--max-requests", type=int, default=0)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Validate and continue completed HTTP 200 receipts",
    )
    args = parser.parse_args()
    if args.max_requests < 0:
        parser.error("--max-requests must be nonnegative")
    token = sys.stdin.readline().strip()
    run(args.input, args.output, args.model, token, args.max_requests, args.resume)


if __name__ == "__main__":
    main()
