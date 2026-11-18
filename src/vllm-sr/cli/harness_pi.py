"""Point the Pi coding agent at a Router gateway, as `vllm-sr connect pi` does.

Pi reads custom providers from models.json in its agent directory
($PI_CODING_AGENT_DIR, by default ~/.pi/agent). The command owns one provider
there, named PROVIDER, and leaves every other provider and key as it was. The
file never holds a key: apiKey names the variable Pi reads it from ("$NAME"),
or the command it runs for it ("!command"), and Pi resolves either before
each request (Pi 1.5). On a gateway that presents each client's own login
upstream, the command is the one Pi offers external clients for its stored
login, so the login stays in Pi's auth.json and Pi refreshes it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cli import harness_connect as hc

PROVIDER = hc.PROVIDER
AGENT_DIR_VARIABLE = "PI_CODING_AGENT_DIR"
OWNED = f"providers/{PROVIDER}"
# The Pi providers whose OAuth login each caller provider issued, the first one
# Pi holds taken: Claude's, and ChatGPT's for Codex or for the OpenAI API.
LOGINS = {"anthropic": ("anthropic",), "openai": ("openai-codex", "openai")}
# Prints the provider's stored OAuth login, refreshed first when it expires in
# less than 30 minutes (Pi 1.5.2).
LOGIN_COMMAND = "pi auth print-bearer-token --provider "


def agent_dir(env: dict[str, str]) -> Path:
    return Path(env.get(AGENT_DIR_VARIABLE) or str(Path.home() / ".pi" / "agent"))


def provider(
    gateway: str,
    model: str,
    key_env: str | None,
    context_window: int | None,
    key_command: str | None = None,
    max_tokens: int = int(hc.MAX_OUTPUT_TOKENS),
    login: str | None = None,
) -> dict[str, Any]:
    """The provider entry. Chat Completions is the API Pi's docs use for an
    OpenAI-compatible endpoint, and there Pi keeps its own model id on each
    reply, with the model the Router picked beside it as responseModel. Pi
    sends its session id, as x-session-id, only when the compat asks for it,
    and the Router holds a session's model on that header. A listener without
    keys still gets a placeholder: Pi lists a model only once its provider has
    a key. With login, Pi sends that login of its own, and an Anthropic one
    over Messages, the only API the Router takes it on. There Pi sends no
    session header on a login, and the Router refuses its default
    tools[].eager_input_streaming, which Pi then asks for with a beta header."""
    key_command = LOGIN_COMMAND + login if login else key_command
    api_key = f"!{key_command}" if key_command else f"${key_env}" if key_env else None
    entry: dict[str, Any] = {"id": model, "maxTokens": max_tokens}
    if context_window:
        entry["contextWindow"] = context_window
    messages = login == "anthropic"
    return {
        "name": "vLLM Semantic Router",
        # Pi's Anthropic client adds /v1/messages itself
        "baseUrl": gateway.rstrip("/") + ("" if messages else "/v1"),
        "api": "anthropic-messages" if messages else "openai-completions",
        "apiKey": api_key or hc.NO_KEY,
        "compat": (
            {"supportsEagerToolInputStreaming": False}
            if messages
            else {
                "sendSessionAffinityHeaders": True,
                "sessionAffinityFormat": "openrouter",
            }
        ),
        "models": [entry],
    }


STORED_KEY = (
    f"Pi holds a key of its own for {PROVIDER} in auth.json, which it sends in place"
    " of the one models.json names"
)


def stored_key(home: Path) -> bool:
    """Whether Pi's auth.json holds a credential for PROVIDER: Pi 1.5 sends a
    stored credential before it reads the provider's apiKey."""
    try:
        return PROVIDER in json.loads(hc.read(home / "auth.json") or "{}")
    except (ValueError, TypeError):
        return False


def login(home: Path, provider: str) -> str | None:
    """The Pi provider whose OAuth login, in auth.json, the caller provider
    issued, or None. Only each credential's type is read."""
    try:
        stored = json.loads(hc.read(home / "auth.json") or "{}")
        types = {p: (stored.get(p) or {}).get("type") for p in LOGINS.get(provider, ())}
    except (ValueError, AttributeError):
        return None
    return next((p for p, kind in types.items() if kind == "oauth"), None)


def models_change(home: Path, entry: dict[str, Any]) -> hc.Change:
    return hc.Change(home / "models.json", "json", {OWNED: entry})


def connections(manifest_dir: Path) -> list[dict[str, Any]]:
    """What the undo record in Pi's agent directory says connect set up: the
    provider written to models.json, and whether the file still holds it."""
    record = json.loads(hc.read(manifest_dir / hc.MANIFEST) or "{}")
    prefix = f"!{LOGIN_COMMAND}"
    found = []
    for item in record.get("files") or []:
        written = (item["keys"].get(OWNED) or {}).get("written")
        if item["kind"] != "json" or not isinstance(written, dict):
            continue
        model = (written.get("models") or [{}])[0]
        api_key = str(written.get("apiKey") or "")
        key_env = api_key[1:] if api_key.startswith("$") else None
        own = api_key[len(prefix) :] if api_key.startswith(prefix) else None
        # the login command is never run for a status: it prints the login
        key_command = api_key[1:] if api_key.startswith("!") and not own else None
        found.append(
            {
                "harness": "pi",
                "provider": PROVIDER,
                "gateway": str(written.get("baseUrl") or "").removesuffix("/v1")
                or None,
                "model": model.get("id"),
                "context_window": model.get("contextWindow"),
                "login": (
                    "own login"
                    if own
                    else (
                        "key command"
                        if key_command
                        else "gateway key" if key_env else "no key"
                    )
                ),
                "pi_login": own,
                "key_env": key_env,
                "key_command": key_command,
                "stored_key": stored_key(manifest_dir),
                "files": [hc.file_state(item)],
            }
        )
    return found
