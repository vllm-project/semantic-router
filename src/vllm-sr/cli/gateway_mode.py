"""Where client traffic enters a served stack: the gateway mode.

``standalone`` (the default) is the Router serving the OpenAI-compatible API
on its listeners, with no Envoy. ``extproc`` puts an Envoy-based gateway in
front of the Router, which serves ext_proc: on the docker target, the Envoy
container the CLI starts, as every release before standalone mode did; on
the kubernetes target, the gateway the cluster runs.
"""

from __future__ import annotations

import os

from cli.utils import get_logger

log = get_logger(__name__)

GATEWAY_STANDALONE = "standalone"
GATEWAY_EXTPROC = "extproc"
VALID_GATEWAYS = (GATEWAY_STANDALONE, GATEWAY_EXTPROC)
DEFAULT_GATEWAY = GATEWAY_STANDALONE

# The Router container's entrypoint, and config realization inside the
# stack's containers, read the mode from this variable.
GATEWAY_ENV = "VLLM_SR_GATEWAY"

GATEWAY_HELP = (
    "Where client traffic enters: standalone (default; the Router serves the "
    "OpenAI-compatible API on the config's listeners, with no Envoy) or "
    "extproc (an Envoy-based gateway in front of the Router: the Envoy "
    "container on the docker target, your gateway on kubernetes)."
)


def resolve_gateway(gateway: str | None) -> str:
    """Validate the gateway mode; None is the default."""

    if gateway is None:
        return DEFAULT_GATEWAY
    normalized = gateway.strip().lower()
    if normalized not in VALID_GATEWAYS:
        raise ValueError(
            f"Invalid gateway '{gateway}'. Must be one of: {', '.join(VALID_GATEWAYS)}"
        )
    return normalized


def stack_gateway() -> str:
    """The gateway mode of the stack whose config is being realized.

    `serve` sets it for every container it starts; a stack started before
    standalone mode existed has none and runs Envoy.
    """

    return os.getenv(GATEWAY_ENV, "").strip().lower() or GATEWAY_EXTPROC


def apply_gateway_mode(env_vars: dict[str, str], gateway: str) -> None:
    """Hand the stack's gateway mode to the processes it starts."""

    env_vars[GATEWAY_ENV] = gateway


def log_gateway_mode(gateway: str, target: str, *, explicit: bool) -> None:
    """State the mode, and how to get the previous default back."""

    source = "--gateway" if explicit else "default"
    log.info(f"Gateway: {gateway} ({source}), target: {target}")
    if gateway == GATEWAY_STANDALONE and not explicit:
        log.info(
            "The Router serves the OpenAI-compatible API itself by default now; "
            "--gateway extproc puts Envoy in front of it, as earlier releases did"
        )


def runs_envoy(gateway: str) -> bool:
    """Whether the docker target starts an Envoy container for the mode."""

    return gateway == GATEWAY_EXTPROC
