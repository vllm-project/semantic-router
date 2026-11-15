"""Point the Pi coding agent at a Router gateway, as `vllm-sr connect pi` does.

Pi reads custom providers from models.json in its agent directory
($PI_CODING_AGENT_DIR, by default ~/.pi/agent). The command owns one provider
there, named PROVIDER, and leaves every other provider and key as it was. The
file never holds a key: apiKey names the variable Pi reads it from ("$NAME"),
or the command it runs for it ("!command"), and Pi resolves either before
each request (Pi 1.5).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cli import harness_connect as hc

PROVIDER = hc.PROVIDER
AGENT_DIR_VARIABLE = "PI_CODING_AGENT_DIR"
OWNED = f"providers/{PROVIDER}"


def agent_dir(env: dict[str, str]) -> Path:
    return Path(env.get(AGENT_DIR_VARIABLE) or str(Path.home() / ".pi" / "agent"))


def provider(
    gateway: str,
    model: str,
    key_env: str | None,
    context_window: int | None,
    key_command: str | None = None,
) -> dict[str, Any]:
    """The provider entry. Chat Completions is the API Pi's docs use for an
    OpenAI-compatible endpoint, and there Pi keeps its own model id on each
    reply, with the model the Router picked beside it as responseModel. Pi
    sends its session id, as x-session-id, only when the compat asks for it,
    and the Router holds a session's model on that header. A listener without
    keys still gets a placeholder: Pi lists a model only once its provider has
    a key."""
    api_key = f"!{key_command}" if key_command else f"${key_env}" if key_env else None
    entry: dict[str, Any] = {"id": model, "maxTokens": int(hc.MAX_OUTPUT_TOKENS)}
    if context_window:
        entry["contextWindow"] = context_window
    return {
        "name": "vLLM Semantic Router",
        "baseUrl": gateway.rstrip("/") + "/v1",
        "api": "openai-completions",
        "apiKey": api_key or hc.NO_KEY,
        "compat": {
            "sendSessionAffinityHeaders": True,
            "sessionAffinityFormat": "openrouter",
        },
        "models": [entry],
    }


def models_change(home: Path, entry: dict[str, Any]) -> hc.Change:
    return hc.Change(home / "models.json", "json", {OWNED: entry})


def connections(manifest_dir: Path) -> list[dict[str, Any]]:
    """What the undo record in Pi's agent directory says connect set up: the
    provider written to models.json, and whether the file still holds it."""
    record = json.loads(hc.read(manifest_dir / hc.MANIFEST) or "{}")
    found = []
    for item in record.get("files") or []:
        written = (item["keys"].get(OWNED) or {}).get("written")
        if item["kind"] != "json" or not isinstance(written, dict):
            continue
        model = (written.get("models") or [{}])[0]
        api_key = str(written.get("apiKey") or "")
        key_env = api_key[1:] if api_key.startswith("$") else None
        key_command = api_key[1:] if api_key.startswith("!") else None
        found.append(
            {
                "harness": "pi",
                "provider": PROVIDER,
                "gateway": str(written.get("baseUrl") or "").removesuffix("/v1")
                or None,
                "model": model.get("id"),
                "context_window": model.get("contextWindow"),
                "login": (
                    "key command"
                    if key_command
                    else "gateway key" if key_env else "no key"
                ),
                "key_env": key_env,
                "key_command": key_command,
                "files": [hc.file_state(item)],
            }
        )
    return found
