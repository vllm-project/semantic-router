"""Click command: one-shot chat against the routed OpenAI-compatible API."""

from __future__ import annotations

import json
import logging
from http import HTTPStatus

import click
import requests

from cli.chat_client import (
    DEFAULT_CHAT_MODEL,
    build_chat_payload,
    chat_completions_url,
    extract_assistant_text,
    post_chat_completions,
    resolve_chat_base_url,
)
from cli.commands.common import exit_with_logged_error
from cli.deployment_backend import resolve_target
from cli.docker_backend import DockerBackend
from cli.utils import get_logger

log = get_logger(__name__)


@click.command("chat")
@click.argument("message", nargs=-1, required=False)
@click.option(
    "--prompt",
    default=None,
    help="User message (alternative to a positional prompt).",
)
@click.option(
    "--model",
    "model",
    default=DEFAULT_CHAT_MODEL,
    show_default=True,
    help="Model name sent to the router (default: MoM).",
)
@click.option(
    "--system",
    "system_prompt",
    default=None,
    help="Optional system message prepended to the conversation.",
)
@click.option(
    "--config",
    default="config.yaml",
    show_default=True,
    help="Config file used to resolve listener host port (Docker default only).",
)
@click.option(
    "--json",
    "json_output",
    is_flag=True,
    help="Print the raw JSON response instead of assistant text.",
)
@click.option(
    "--timeout",
    default=120.0,
    show_default=True,
    type=float,
    help="HTTP timeout in seconds for the completion request.",
)
@click.option(
    "--temperature",
    default=None,
    type=float,
    help="Optional sampling temperature passed through to the API.",
)
@click.option("--target", default=None, help="Deployment target (docker or k8s).")
@exit_with_logged_error(log)
def chat(
    message: tuple[str, ...],
    prompt: str | None,
    model: str,
    system_prompt: str | None,
    config: str,
    json_output: bool,
    timeout: float,
    temperature: float | None,
    target: str | None,
) -> None:
    """
    Send a one-shot chat completion through the Envoy-routed HTTP API.

    Uses the first listener port in config.yaml plus the stack port offset.
    Default model is MoM.

    Examples:

        vllm-sr chat "hello"

        vllm-sr chat --model MoM --prompt "Explain mixture of models"

        vllm-sr chat --json "hello"
    """
    user_text = (prompt or "").strip() or " ".join(message).strip()
    if not user_text:
        raise click.UsageError("Provide a prompt as arguments or use --prompt.")

    try:
        base = resolve_chat_base_url(config_path=config, target=target)
    except (OSError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc

    if resolve_target(target) == "docker":
        logging.getLogger("cli.docker_runtime").setLevel(logging.WARNING)
        backend = DockerBackend()
        if not backend.is_running():
            raise click.ClickException(
                "vLLM Semantic Router does not appear to be running locally "
                "(no managed Docker containers). Start the stack with "
                "`vllm-sr serve`."
            )

    url = chat_completions_url(base)
    payload = build_chat_payload(
        model=model,
        user_text=user_text,
        system_text=system_prompt,
        temperature=temperature,
    )

    try:
        resp = post_chat_completions(url=url, payload=payload, timeout=timeout)
    except requests.exceptions.ConnectionError as exc:
        raise click.ClickException(
            f"Could not reach {url}. Is Envoy listening and port-forwarded if needed? ({exc})"
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise click.ClickException(
            f"Request to {url} timed out after {timeout}s."
        ) from exc

    if resp.status_code >= HTTPStatus.BAD_REQUEST:
        body_preview = (resp.text or "")[:2000]
        raise click.ClickException(
            f"HTTP {resp.status_code} from {url}: {body_preview or '(empty body)'}"
        )

    try:
        data = resp.json()
    except json.JSONDecodeError as exc:
        raise click.ClickException(
            f"Response was not JSON (status {resp.status_code}): {resp.text[:500]}"
        ) from exc

    if json_output:
        click.echo(json.dumps(data, indent=2))
        return

    try:
        text = extract_assistant_text(data)
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(text)
