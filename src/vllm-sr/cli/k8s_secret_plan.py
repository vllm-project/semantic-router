"""Pure planning primitives for CLI-managed Kubernetes runtime Secrets."""

from __future__ import annotations

import base64
import re
import secrets
from dataclasses import dataclass, field


@dataclass(frozen=True)
class SecretSnapshot:
    """Secret metadata and encoded data captured without logging its content."""

    name: str
    data: dict[str, str] = field(repr=False)
    immutable: bool = False
    cli_managed: bool = False
    data_known: bool = False


@dataclass(frozen=True)
class EnvSecretPlan:
    """Side-effect-free plan for one release-scoped credential generation."""

    active_name: str | None
    new_manifest: str | None = field(default=None, repr=False)
    recreate_for_looper_rotation: bool = False

    @property
    def creates_secret(self) -> bool:
        return self.new_manifest is not None


def desired_encoded_secret_data(
    env_vars: dict[str, str] | None,
    previous: SecretSnapshot | None,
) -> dict[str, str]:
    """Build the exact desired data, always with one release-wide Looper key."""
    from cli.commands.runtime_support import (  # noqa: PLC0415
        LOOPER_SHARED_SECRET_ENV,
        PASSTHROUGH_ENV_RULES,
    )

    source = env_vars or {}
    sensitive_names = {name for name, masked in PASSTHROUGH_ENV_RULES if masked}
    explicit = {
        key: value
        for key, value in source.items()
        if key in sensitive_names and key != LOOPER_SHARED_SECRET_ENV and value
    }
    encoded = encode_secret_data(explicit)

    if LOOPER_SHARED_SECRET_ENV in source:
        looper_secret = source[LOOPER_SHARED_SECRET_ENV]
        if (
            not isinstance(looper_secret, str)
            or re.fullmatch(r"[0-9a-fA-F]{64}", looper_secret) is None
        ):
            raise ValueError(
                f"{LOOPER_SHARED_SECRET_ENV} must be exactly 64 hexadecimal characters"
            )
        encoded[LOOPER_SHARED_SECRET_ENV] = encode_secret_data(
            {LOOPER_SHARED_SECRET_ENV: looper_secret}
        )[LOOPER_SHARED_SECRET_ENV]
        return encoded

    reusable = _reusable_looper_secret(previous)
    encoded[LOOPER_SHARED_SECRET_ENV] = (
        reusable
        or encode_secret_data({LOOPER_SHARED_SECRET_ENV: secrets.token_hex(32)})[
            LOOPER_SHARED_SECRET_ENV
        ]
    )
    return encoded


def _reusable_looper_secret(previous: SecretSnapshot | None) -> str | None:
    from cli.commands.runtime_support import LOOPER_SHARED_SECRET_ENV  # noqa: PLC0415

    if (
        previous is None
        or not previous.cli_managed
        or not previous.immutable
        or not previous.data_known
    ):
        return None
    encoded = previous.data.get(LOOPER_SHARED_SECRET_ENV)
    if not isinstance(encoded, str):
        return None
    try:
        decoded = base64.b64decode(encoded, validate=True).decode("ascii")
    except (ValueError, UnicodeDecodeError):
        return None
    if re.fullmatch(r"[0-9a-fA-F]{64}", decoded) is None:
        return None
    return encoded


def encode_secret_data(secret_data: dict[str, str]) -> dict[str, str]:
    """Encode Secret data without shell interpolation or command arguments."""
    return {
        key: base64.b64encode(value.encode()).decode()
        for key, value in secret_data.items()
    }
