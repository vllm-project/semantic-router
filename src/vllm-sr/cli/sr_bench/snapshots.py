"""Optional, server-observed recipe receipts from the declared Router origin."""

from __future__ import annotations

import ipaddress
import json
import os
import re
import time
from datetime import datetime, timezone
from http import HTTPStatus
from urllib.parse import urlsplit, urlunsplit

import requests

from .contracts import digest

MAX_CONFIG_BYTES = 4 * 1024 * 1024
CONFIG_DEADLINE_S = 15
SECRET_KEY = re.compile(
    r"(?:^|_)(?:api_?keys?|token|passwords?|passwd|secrets?|authorization|auth|cookies?|credentials?|private_key|client_key)(?:$|_)"
)
DEPLOYMENT_KEY = re.compile(
    r"(?:^|_)(?:headers?|metadata|address(?:es)?|addr|hosts?|hostnames?|ips?|uris?|urls?|paths?|directories|directory|dirs?|sockets?|endpoints?|listen|bind|ports?|connection|dsn|certificates?|cert|deployment|providers)(?:$|_)"
)
IPV4 = re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])")
HASH_KEYS = ("source_config_hash", "generated_runtime_hash", "active_runtime_hash")


def _private_string(value):
    if re.search(
        r"://|(?:^|\s)(?:/|~/|\.{1,2}/|[A-Za-z]:[\\/]|\\\\)|\b(?:Bearer|Basic)\s+|-----BEGIN .*PRIVATE KEY",
        value,
        re.IGNORECASE,
    ):
        return True
    if re.search(r"\b[\w.-]+\.(?:internal|local|lan|corp)\b", value):
        return True
    for candidate in [value, *IPV4.findall(value)]:
        try:
            ipaddress.ip_address(candidate)
            return True
        except ValueError:
            pass
    return False


def _public(value, redactions, path=""):
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            location = f"{path}.{key}" if path else key
            normalized = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", key)
            normalized = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", normalized)
            normalized = re.sub(r"[^a-z0-9]+", "_", normalized.lower()).strip("_")
            env_reference = (
                normalized.endswith(("_env", "_env_var"))
                and isinstance(item, str)
                and re.fullmatch(r"[A-Z_][A-Z0-9_]*", item)
            )
            secret = (
                SECRET_KEY.search(normalized) or normalized == "tokens"
            ) and not env_reference
            if secret or DEPLOYMENT_KEY.search(normalized):
                result[key] = "<redacted>"
                redactions.append(location)
            else:
                result[key] = _public(item, redactions, location)
        return result
    if isinstance(value, list):
        return [
            _public(item, redactions, f"{path}[{i}]") for i, item in enumerate(value)
        ]
    if isinstance(value, str) and _private_string(value):
        redactions.append(path)
        return "<redacted>"
    return value


def _origin(target):
    value = target.get("preview_url", "")
    if not isinstance(value, str) or any(
        char <= " " or char == "\x7f" for char in value
    ):
        raise ValueError("Recipe capture requires the canonical management preview_url")
    try:
        preview = urlsplit(value)
        _ = preview.port  # Validate the authority before attaching credentials.
    except ValueError as exc:
        raise ValueError(
            "Recipe capture requires the canonical management preview_url"
        ) from exc
    if (
        preview.scheme not in {"http", "https"}
        or not preview.hostname
        or preview.username
        or preview.password
        or preview.query
        or preview.fragment
        or "\\" in preview.netloc
        or "%" in preview.netloc
        or preview.path != "/api/v1/routing/preview"
    ):
        raise ValueError("Recipe capture requires the canonical management preview_url")
    return urlunsplit((preview.scheme, preview.netloc, "", "", ""))


def _read_json(url, headers, started):
    remaining = CONFIG_DEADLINE_S - (time.monotonic() - started)
    if remaining <= 0:
        raise ValueError("Recipe capture deadline exceeded")
    try:
        with requests.get(
            url,
            headers=headers,
            timeout=(min(5, remaining), min(5, remaining)),
            allow_redirects=False,
            stream=True,
        ) as response:
            if response.status_code != HTTPStatus.OK:
                raise ValueError(f"Recipe capture HTTP {response.status_code}")
            body = bytearray()
            # Byte reads enforce the absolute bound even for a trickling response.
            for chunk in response.iter_content(chunk_size=1):
                body.extend(chunk)
                if len(body) > MAX_CONFIG_BYTES:
                    raise ValueError("Recipe capture exceeds configuration size limit")
                if time.monotonic() - started > CONFIG_DEADLINE_S:
                    raise ValueError("Recipe capture deadline exceeded")
            if time.monotonic() - started > CONFIG_DEADLINE_S:
                raise ValueError("Recipe capture deadline exceeded")
            return json.loads(body), response.headers.get("ETag")
    except requests.RequestException as exc:
        raise ValueError("Recipe capture transport failed") from exc
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError(
            "Recipe capture returned an invalid configuration document"
        ) from exc


def _hashes(document, expected):
    if not isinstance(document, dict) or any(
        not isinstance(document.get(key), str)
        or not re.fullmatch(r"[0-9a-f]{64}", document[key])
        for key in HASH_KEYS
    ):
        raise ValueError("Recipe capture returned invalid configuration hashes")
    if (
        document.get("activation_status") != "active"
        or document["generated_runtime_hash"] != expected
        or document["active_runtime_hash"] != expected
    ):
        raise ValueError("Recipe capture active configuration acknowledgement mismatch")
    return {key: document[key] for key in HASH_KEYS}


def _capture(target):
    origin = _origin(target)
    expected = target.get("config_hash", "")
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("Recipe capture requires a frozen configuration hash")
    headers = {"Accept": "application/json"}
    credential = target.get("preview_api_key_env") or target.get("api_key_env")
    if credential:
        if not isinstance(credential, str) or not re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]*", credential
        ):
            raise ValueError(
                "Recipe capture credential must name an environment variable"
            )
        token = os.environ.get(credential)
        if not token:
            raise ValueError(
                "Recipe capture credential environment variable is not set"
            )
        headers["Authorization"] = "Bearer " + token
    started = time.monotonic()
    before, _ = _read_json(origin + "/api/v1/config/hash", headers, started)
    before = _hashes(before, expected)
    document, etag = _read_json(origin + "/api/v1/config", headers, started)
    if etag != f'"{before["source_config_hash"]}"':
        raise ValueError("Recipe capture source configuration acknowledgement mismatch")
    after, _ = _read_json(origin + "/api/v1/config/hash", headers, started)
    if _hashes(after, expected) != before:
        raise ValueError("Recipe capture configuration changed during capture")
    if not isinstance(document, dict) or not isinstance(document.get("recipes"), list):
        raise ValueError("Recipe capture returned an invalid canonical configuration")
    global_config = document.get("global", {})
    if not isinstance(global_config, dict) or not isinstance(
        global_config.get("router", {}), dict
    ):
        raise ValueError("Recipe capture returned an invalid canonical configuration")
    projection = {
        "recipes": document["recipes"],
        "entrypoints": document.get("entrypoints", []),
        "evaluation": document.get("evaluation", {}),
        "routing": document.get("routing", {}),
        "learning": global_config.get("router", {}).get("learning", {}),
    }
    redactions = []
    public = _public(projection, redactions)
    return {
        **public,
        **before,
        "config_hash": expected,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "recipe_sha256": digest(public),
        "redacted": True,
        "redactions": redactions,
        "source": "router_config_api_bracketed_hashes",
        "scope": "Sanitized routing snapshot; deployment wiring omitted. Source and active runtime hashes were stable across capture, not atomically locked. Each live request separately verifies the active configuration hash.",
    }


def capture_recipes(manifest):
    snapshots = {}
    for target in manifest["targets"]:
        enabled = target.get("capture_recipe", False)
        if not isinstance(enabled, bool):
            raise ValueError("capture_recipe must be boolean")
        if enabled:
            if target["kind"] != "mom":
                raise ValueError("Recipe capture is only available for MoM targets")
            snapshots[target["id"]] = _capture(target)
    return snapshots
