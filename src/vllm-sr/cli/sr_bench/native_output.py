"""Frozen native capacities and evidence for actual provider output budgets."""

from __future__ import annotations

import hashlib
import json
import socket
import threading
import time
from contextlib import suppress
from http import HTTPStatus

import requests

MAX_RENDER_BYTES = 32 * 1024 * 1024
MAX_RENDER_SECONDS = 30
MAX_TOKEN_HEADER_DIGITS = 20


class NativeOutputError(ValueError):
    """A controlled native-output contract error with no provider content."""


def _positive(value):
    return type(value) is int and value > 0


def _digest(value):
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(encoded.encode()).hexdigest()


def validate_limits(target, required=False):
    limits = target.get("native_limits")
    if limits is None and not required:
        return
    if not isinstance(limits, dict) or not limits:
        raise NativeOutputError("Native output requires frozen model native_limits")
    for model, values in limits.items():
        if (
            not isinstance(model, str)
            or not model
            or not isinstance(values, dict)
            or set(values) != {"context_window", "max_output_tokens"}
            or not all(_positive(v) for v in values.values())
            or values["max_output_tokens"] > values["context_window"]
        ):
            raise NativeOutputError("Invalid frozen native model limits")
    if target["kind"] == "single" and set(limits) != {
        target.get("expected_response_model") or target["model"]
    }:
        raise NativeOutputError(
            "Single native_limits must identify its physical response model"
        )


def capacity(target):
    return max(row["max_output_tokens"] for row in target["native_limits"].values())


def model_limits(manifest):
    models = {}
    for target in [
        *manifest["targets"],
        *manifest.get("auxiliary_targets", {}).values(),
    ]:
        for model, limits in target.get("native_limits", {}).items():
            if model in models and models[model] != limits:
                raise NativeOutputError(
                    "Frozen native limits conflict for the same model"
                )
            models[model] = limits
    return models


def configure(manifest, targets, limits):
    policy = manifest.setdefault("output_policy", "bounded")
    if not isinstance(policy, str) or policy not in {"bounded", "native"}:
        raise NativeOutputError("output_policy must be bounded or native")
    for target in targets:
        validate_limits(target, required=policy == "native")
    model_limits(manifest)
    if policy != "native":
        return
    if not isinstance(manifest.get("sampling", {}), dict) or any(
        not isinstance(target.get("request_params", {}), dict) for target in targets
    ):
        raise NativeOutputError("Native sampling and request_params must be objects")
    if "max_tokens" in manifest.get("sampling", {}) or any(
        "max_tokens" in target.get("request_params", {}) for target in targets
    ):
        raise NativeOutputError(
            "Native output forbids explicit sampling/target max_tokens"
        )
    for target in targets:
        if manifest.get("cost_policy") == "require_priced" and set(
            target["native_limits"]
        ) - set(target.get("prices", {})):
            raise NativeOutputError(
                "Native output requires frozen prices for every physical model"
            )
        if target["kind"] == "mom" and (
            target.get("max_inference_calls") != 1 or not target.get("capture_recipe")
        ):
            raise NativeOutputError(
                "Native MoM requires one dispatch and captured automatic-output recipe"
            )
    ceiling = max(capacity(target) for target in targets)
    requested = manifest.get("limits", {}).get("max_output_tokens", ceiling)
    if type(requested) is not int or requested != ceiling:
        raise NativeOutputError(
            "Native output evidence ceiling must equal frozen native capacity"
        )
    limits["max_output_tokens"] = ceiling


def validate_recipes(manifest, provenance):
    if manifest["output_policy"] != "native":
        return
    snapshots = provenance.get("recipe_snapshots", {})
    for target in manifest["targets"]:
        if target["kind"] != "mom":
            continue
        snapshot = snapshots.get(target["id"], {})
        matches = [
            row.get("recipe")
            for row in snapshot.get("entrypoints", [])
            if target["model"] in row.get("model_names", [])
        ]
        recipes = [
            row for row in snapshot.get("recipes", []) if row.get("name") in matches
        ]
        if (
            len(matches) != 1
            or len(recipes) != 1
            or snapshot.get("config_hash") != target["config_hash"]
        ):
            raise NativeOutputError(
                "Native MoM requires its captured entrypoint and active recipe"
            )
        decisions = recipes[0].get("routing", {}).get("decisions", [])
        if not decisions:
            raise NativeOutputError(
                "Native MoM recipe has no automatic-output decisions"
            )
        for decision in decisions:
            refs = decision.get("modelRefs", [])
            if not refs or any(
                ref.get("model") not in target["native_limits"] for ref in refs
            ):
                raise NativeOutputError(
                    "Native MoM limits must cover every configured physical candidate; alias mapping is unsupported"
                )
            params = [
                plugin.get("configuration", {})
                for plugin in decision.get("plugins", [])
                if plugin.get("type") == "request_params"
            ]
            if (
                len(params) != 1
                or params[0].get("default_max_tokens") != "auto"
                or params[0].get("max_tokens_limit") is not None
            ):
                raise NativeOutputError(
                    "Native MoM requires auto output without a configured output clip on every decision"
                )


def _evidence(target, model, input_tokens, max_tokens, source):
    limits = target["native_limits"].get(model)
    if limits is None or not _positive(input_tokens) or not _positive(max_tokens):
        raise NativeOutputError(
            "Native output model or token evidence is missing or invalid"
        )
    expected = min(limits["max_output_tokens"], limits["context_window"] - input_tokens)
    if expected <= 0 or max_tokens != expected:
        raise NativeOutputError(
            "Native output capacity differs from frozen model limits and actual input"
        )
    return {
        "policy": "native",
        "source": source,
        "model": model,
        "input_tokens": input_tokens,
        "max_output_tokens": max_tokens,
        "context_window": limits["context_window"],
        "configured_max_output_tokens": limits["max_output_tokens"],
    }


def resolve_single(target, body, headers, limits, cancelled, started):
    """Render the exact prompt-bearing payload; never perform generation here."""
    remaining = limits["total_timeout_s"] - (time.monotonic() - started)
    if remaining <= 0 or cancelled():
        raise NativeOutputError("Native render cancelled or request deadline exceeded")
    url = target["base_url"].rstrip("/") + "/chat/completions/render"
    deadline = time.monotonic() + min(remaining, MAX_RENDER_SECONDS)
    data = _render_json(url, body, headers, limits, cancelled, deadline)
    try:
        rendered = json.loads(data)
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise NativeOutputError("Native render returned invalid JSON") from exc
    if not isinstance(rendered, dict):
        raise NativeOutputError("Native render returned an invalid document")
    model = target.get("expected_response_model") or target["model"]
    ids = rendered.get("token_ids")
    sampling = rendered.get("sampling_params")
    if (
        rendered.get("model") != model
        or not isinstance(ids, list)
        or not ids
        or any(type(token) is not int or token < 0 for token in ids)
        or not isinstance(sampling, dict)
        or rendered.get("features") is not None
    ):
        raise NativeOutputError(
            "Native render requires matching model and text token evidence"
        )
    frozen = target["native_limits"][model]
    maximum = min(frozen["max_output_tokens"], frozen["context_window"] - len(ids))
    provider_max = sampling.get("max_tokens")
    if not _positive(provider_max) or provider_max < maximum:
        raise NativeOutputError(
            "Native render provider capacity is below frozen model capability"
        )
    evidence = _evidence(target, model, len(ids), maximum, "provider_render")
    effective = {**body, "max_tokens": maximum}
    evidence.update(
        provider_max_output_tokens=provider_max,
        render_request_sha256=_digest(body),
        render_response_sha256=hashlib.sha256(data).hexdigest(),
        effective_request_sha256=_digest(effective),
    )
    return effective, evidence


def _render_json(url, body, headers, limits, cancelled, deadline):
    stop = threading.Event()
    interrupted = threading.Event()
    remaining = max(0.01, deadline - time.monotonic())
    try:
        with requests.post(
            url,
            json=body,
            headers=headers,
            stream=True,
            allow_redirects=False,
            timeout=(min(10, remaining), min(5, limits["idle_timeout_s"], remaining)),
        ) as response:
            if response.status_code != HTTPStatus.OK:
                raise NativeOutputError(
                    f"Native render HTTP {response.status_code}; a vLLM Chat render endpoint is required"
                )

            def watch():
                while not stop.wait(0.05):
                    if cancelled() or time.monotonic() > deadline:
                        interrupted.set()
                        with suppress(AttributeError, OSError):
                            response.raw._fp.fp.raw._sock.shutdown(socket.SHUT_RDWR)
                        return

            threading.Thread(target=watch, daemon=True).start()
            data = bytearray()
            for chunk in response.iter_content(chunk_size=1):
                data.extend(chunk)
                if len(data) > MAX_RENDER_BYTES:
                    raise NativeOutputError(
                        "Native render response exceeds evidence bound"
                    )
                if cancelled() or time.monotonic() > deadline:
                    interrupted.set()
                    break
            if interrupted.is_set() or cancelled() or time.monotonic() > deadline:
                raise NativeOutputError(
                    "Native render cancelled or request deadline exceeded"
                )
            return data
    except requests.RequestException as exc:
        message = (
            "Native render cancelled or request deadline exceeded"
            if interrupted.is_set()
            else "Native render transport failed"
        )
        raise NativeOutputError(message) from exc
    finally:
        stop.set()


def from_headers(target, headers):
    model = headers.get("x-vsr-selected-model")
    values = []
    for name in ("x-vsr-effective-input-tokens", "x-vsr-effective-max-output-tokens"):
        value = headers.get(name, "")
        if (
            not isinstance(value, str)
            or not value.isascii()
            or not value.isdecimal()
            or len(value) > MAX_TOKEN_HEADER_DIGITS
        ):
            raise NativeOutputError(
                "Native MoM effective token acknowledgement is missing or invalid"
            )
        values.append(int(value))
    if headers.get("x-sr-bench-config-hash") != target.get(
        "config_hash"
    ) and headers.get("x-vsr-config-hash") != target.get("config_hash"):
        raise NativeOutputError("Native MoM configuration acknowledgement mismatched")
    if headers.get("x-vsr-inference-call-count") != "1":
        raise NativeOutputError("Native MoM requires an acknowledged single dispatch")
    return _evidence(target, model, *values, "router_dispatch")


def validate_response(evidence, model, usage):
    if model != evidence["model"]:
        raise NativeOutputError(
            "Native output requires matching selected and physical response model; alias mapping is unsupported"
        )
    validate_usage(evidence, usage)


def validate_usage(evidence, usage):
    if usage is not None and usage["output_tokens"] > evidence["max_output_tokens"]:
        raise NativeOutputError("Output tokens exceed the acknowledged native capacity")
    if (
        usage is not None
        and sum(
            usage[k]
            for k in ("input_tokens", "cached_input_tokens", "cache_write_tokens")
        )
        != evidence["input_tokens"]
    ):
        raise NativeOutputError(
            "Input token usage differs from the acknowledged rendered input"
        )
