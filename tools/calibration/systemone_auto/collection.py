"""Explicit native/Chat collection. Stores observations, never fake timings."""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from http import HTTPStatus
from pathlib import Path

from .artifacts import canonical, digest, file_digest, read_json, read_jsonl, write_json
from .dataset import validate_dataset
from .metrics import evaluate_response, features

DECISION_MODELS = {
    f"vllm-sr/Decision-2.0-{name}"
    for name in ("Kai-0.6B", "Eos-0.8B", "Sol-2B", "Nox-4B", "Lux-9B", "Vega-27B")
}
CHAT_MODEL = "Qwen/Qwen3.8-Flash-Next"
MAX_TIMEOUT_SECONDS = 3600


def validate_targets(targets: list[dict]) -> None:
    names = set()
    for target in targets:
        if target["name"] in names:
            raise ValueError("duplicate model/action name")
        names.add(target["name"])
        allowed = DECISION_MODELS if target["protocol"] == "systemone" else {CHAT_MODEL}
        if (
            target["protocol"] not in {"systemone", "chat"}
            or target["model_id"] not in allowed
        ):
            raise ValueError(
                "collection matrix permits Decision 2.0 and Qwen Flash Next only"
            )
        if not re.fullmatch(r"[0-9a-f]{40}", target["revision"]):
            raise ValueError("each model requires an immutable artifact revision")
        address = urllib.parse.urlparse(target["endpoint"])
        if (
            address.scheme not in {"http", "https"}
            or not address.netloc
            or address.username
            or address.password
            or address.query
            or address.fragment
        ):
            raise ValueError(
                "endpoint must be HTTP(S), with no credentials/query/fragment"
            )
    if not names:
        raise ValueError("collection requires at least one explicit target")


def chat_payload(row: dict, model: str) -> dict:
    """Ask only for typed point answers; do not elicit fabricated confidence."""
    instructions = (
        "Answer every declared question using the supplied state. Return only a JSON object "
        "with an 'answers' object keyed by the original question IDs. Each answer must retain "
        "'type'. For choice include 'choice' with an exact criteria key; for noul include "
        "'noul' as a JSON boolean; for score include 'score' as a number on the zero-based "
        "ordered criteria scale. Do not return probabilities or explanations."
    )
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": instructions},
            {
                "role": "user",
                "content": canonical(
                    {key: row["request"][key] for key in ("state", "questions")}
                ),
            },
        ],
        "temperature": 0,
        "max_tokens": 1024,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "response_format": {"type": "json_object"},
    }


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("collection does not forward credentials through redirects")


def request_once(
    endpoint: str, key: str | None, payload: dict, timeout: float
) -> tuple[int, dict, float]:
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    request = urllib.request.Request(
        endpoint, data=canonical(payload).encode(), headers=headers, method="POST"
    )
    started = time.perf_counter()
    try:
        with urllib.request.build_opener(_NoRedirect).open(
            request, timeout=timeout
        ) as response:
            status, data = response.status, response.read(4 * 1024 * 1024 + 1)
    except urllib.error.HTTPError as error:
        # Error bodies may contain private paths; retain the status, not that body.
        return (
            error.code,
            {"error": {"code": "http_error"}},
            (time.perf_counter() - started) * 1000,
        )
    except (OSError, ValueError, TimeoutError):
        return (
            0,
            {"error": {"code": "transport_error"}},
            (time.perf_counter() - started) * 1000,
        )
    elapsed = (time.perf_counter() - started) * 1000
    if len(data) > 4 * 1024 * 1024:
        return status, {"error": {"code": "response_too_large"}}, elapsed
    try:
        value = json.loads(data)
        canonical(value)
        if not isinstance(value, dict):
            raise ValueError("non-object response")
    except (ValueError, TypeError):
        return status, {"error": {"code": "invalid_json"}}, elapsed
    return status, value, elapsed


def _decode_chat(raw: dict) -> dict:
    try:
        choice = raw["choices"][0]
        if choice.get("finish_reason") != "stop":
            raise ValueError("incomplete generation")
        result = json.loads(choice["message"]["content"])
        canonical(result)
        if not isinstance(result, dict):
            raise ValueError("non-object typed output")
        return result
    except (KeyError, IndexError, ValueError, TypeError):
        return {"error": {"code": "invalid_typed_chat_output"}}


def collect(
    dataset_path: Path,
    target_path: Path,
    output: Path,
    *,
    timeout: float = 120,
    limit: int | None = None,
    target_names: list[str] | None = None,
) -> dict:
    if not 0 < timeout <= MAX_TIMEOUT_SECONDS or (limit is not None and limit <= 0):
        raise ValueError("timeout/limit outside allowed bounds")
    dataset, targets = read_json(dataset_path), read_json(target_path)
    validate_dataset(dataset)
    validate_targets(targets)
    selected_targets = [
        target
        for target in targets
        if target_names is None or target["name"] in target_names
    ]
    if not selected_targets or (
        target_names
        and set(target_names) != {target["name"] for target in selected_targets}
    ):
        raise ValueError("selected target is absent from the immutable matrix")
    public_targets = [
        {key: target[key] for key in ("name", "protocol", "model_id", "revision")}
        for target in targets
    ]
    identity = {
        "collector_source_sha256": digest(
            {
                name: file_digest(Path(__file__).with_name(name))
                for name in (
                    "__init__.py",
                    "artifacts.py",
                    "collection.py",
                    "dataset.py",
                    "metrics.py",
                    "sources.py",
                )
            }
        ),
        "dataset_sha256": digest(dataset),
        "targets": public_targets,
        "timeout_seconds": timeout,
        "retries": 0,
        "concurrency": 1,
        "chat_settings": {
            "temperature": 0,
            "max_tokens": 1024,
            "enable_thinking": False,
        },
    }
    identity_hash = digest(identity)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path, trace_path = (
        output / "collection.json",
        output / "observations.jsonl",
    )
    if (
        manifest_path.exists()
        and read_json(manifest_path)["collection_identity"] != identity_hash
    ):
        raise ValueError("resume identity differs; use a new output directory")
    if trace_path.exists() and not manifest_path.exists():
        raise ValueError("orphan trace has no manifest")
    previous = read_jsonl(trace_path) if trace_path.exists() else []
    completed = {(row["record_id"], row["target"]) for row in previous}
    if len(completed) != len(previous):
        raise ValueError("duplicate observation in resume trace")
    records = {row["id"]: row for row in dataset["records"]}
    allowed_targets = {target["name"] for target in targets}
    for observation in previous:
        identifier = observation["record_id"]
        if identifier not in records or observation["target"] not in allowed_targets:
            raise ValueError("resume contains unknown record/target")
        if observation["collection_identity"] != identity_hash or observation[
            "record_request_sha256"
        ] != digest(records[identifier]["request"]):
            raise ValueError("resume observation identity mismatch")
    manifest = {
        "schema_version": "systemone-collection/v1",
        "collection_identity": identity_hash,
        **identity,
        "timing_semantics": "observed serial client wall time, including transport; not server compute or routed end-to-end timing",
    }
    write_json(manifest_path, manifest)
    rows = dataset["records"][:limit] if limit else dataset["records"]
    with trace_path.open("a") as stream:
        for row_index, row in enumerate(rows):
            # Rotate model order to reduce a systematic warmup/time-of-run confound.
            ordered = (
                selected_targets[row_index % len(selected_targets) :]
                + selected_targets[: row_index % len(selected_targets)]
            )
            for target in ordered:
                if (row["id"], target["name"]) in completed:
                    continue
                model = target.get("served_model_id", target["model_id"])
                native = target["protocol"] == "systemone"
                payload = (
                    {**row["request"], "model": model}
                    if native
                    else chat_payload(row, model)
                )
                key = (
                    os.environ[target["api_key_env"]]
                    if target.get("api_key_env")
                    else None
                )
                status, raw, elapsed = request_once(
                    target["endpoint"], key, payload, timeout
                )
                response = raw if native else _decode_chat(raw)
                result = evaluate_response(row, response, native=native)
                if not HTTPStatus.OK <= status < HTTPStatus.MULTIPLE_CHOICES:
                    result = evaluate_response(row, {}, native=native)
                    result["error"] = "request_failed"
                observation = {
                    "record_id": row["id"],
                    "group_id": row["group_id"],
                    "split": row["split"],
                    "target": target["name"],
                    "request_sha256": digest(payload),
                    "record_request_sha256": digest(row["request"]),
                    "collection_identity": identity_hash,
                    "observed_at": datetime.now(timezone.utc).isoformat(),
                    "http_status": status,
                    "client_elapsed_ms": elapsed,
                    "raw_response": raw,
                    "result": result,
                    "features": features(row, result) if native else None,
                }
                stream.write(canonical(observation) + "\n")
                stream.flush()
                completed.add((row["id"], target["name"]))
    manifest.update(
        observation_count=len(completed),
        expected_count=len(dataset["records"]) * len(targets),
        complete=len(completed) == len(dataset["records"]) * len(targets),
    )
    write_json(manifest_path, manifest)
    return manifest
