"""Secure, idempotent bootstrap for the local managed Router stack."""

from __future__ import annotations

import base64
import json
import os
import secrets
import shutil
import stat
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path

import yaml

from cli.config_contract import DEFAULT_BACKEND_DISPATCH
from cli.consts import DEFAULT_LISTENER_PORT
from cli.runtime_stack import RuntimeStackLayout, resolve_runtime_stack

DEFAULT_OUTPUT_DIR_NAME = ".vllm-sr"
LOCAL_SECRETS_DIR_NAME = "secrets"
LOCAL_BOOTSTRAP_DIR_NAME = "bootstrap"
LOCAL_BOOTSTRAP_TOKEN_NAME = "router-token"
LOCAL_POSTGRES_DSN_ENV = "VLLM_SR_ACCESS_DATABASE_URL"
LOCAL_VALKEY_URL_ENV = "VLLM_SR_ACCESS_RUNTIME_URL"
LOCAL_REPLICA_ID_ENV = "VLLM_SR_REPLICA_ID"
_SECRET_FILE_MODE = 0o600
_SECRET_DIRECTORY_MODE = 0o700
_ED25519_PRIVATE_DER_PREFIX = bytes.fromhex("302e020100300506032b657004220420")
_ED25519_PUBLIC_DER_PREFIX = bytes.fromhex("302a300506032b6570032100")


@dataclass(frozen=True)
class BootstrapResult:
    """Result of ensuring one local managed workspace."""

    config_path: Path
    output_dir: Path
    secret_dir: Path | None = None
    created_config: bool = False
    created_output_dir: bool = False
    created_secrets: bool = False


def _b64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _symmetric_keyring() -> bytes:
    return (
        json.dumps(
            {
                "activeVersion": "v1",
                "keys": [{"version": "v1", "key": _b64url(secrets.token_bytes(32))}],
            },
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _run_openssl(arguments: list[str], *, input_bytes: bytes | None = None) -> bytes:
    executable = shutil.which("openssl")
    if executable is None:
        raise RuntimeError("OpenSSL is required to bootstrap local Management TLS")
    completed = subprocess.run(
        [executable, *arguments], input=input_bytes, capture_output=True, check=False
    )
    if completed.returncode != 0:
        raise RuntimeError("OpenSSL failed while generating local Management trust")
    return completed.stdout


def _signing_keyring() -> bytes:
    seed = secrets.token_bytes(32)
    public_der = _run_openssl(
        ["pkey", "-inform", "DER", "-pubout", "-outform", "DER"],
        input_bytes=_ED25519_PRIVATE_DER_PREFIX + seed,
    )
    if not public_der.startswith(_ED25519_PUBLIC_DER_PREFIX) or len(public_der) != 44:
        raise RuntimeError("OpenSSL returned invalid Ed25519 public material")
    public = public_der[len(_ED25519_PUBLIC_DER_PREFIX) :]
    return (
        json.dumps(
            {
                "activeVersion": "v1",
                "keys": [
                    {
                        "version": "v1",
                        "privateKey": _b64url(seed),
                        "publicKey": _b64url(public),
                    }
                ],
            },
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _ensure_private_directory(path: Path) -> bool:
    created = False
    if path.exists():
        metadata = path.lstat()
        if not stat.S_ISDIR(metadata.st_mode) or path.is_symlink():
            raise ValueError(f"local state path is not a private directory: {path}")
    else:
        path.mkdir(parents=True, mode=_SECRET_DIRECTORY_MODE)
        created = True
    path.chmod(_SECRET_DIRECTORY_MODE)
    return created


def _ensure_secret(path: Path, payload_factory) -> bool:
    """Create one immutable secret without overwriting a concurrent winner."""

    try:
        metadata = path.lstat()
    except FileNotFoundError:
        metadata = None
    if metadata is not None:
        if not stat.S_ISREG(metadata.st_mode) or path.is_symlink():
            raise ValueError(f"local secret path is not a regular file: {path}")
        path.chmod(_SECRET_FILE_MODE)
        if path.stat().st_size == 0:
            raise ValueError(f"local secret file is empty: {path}")
        return False

    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(temporary, flags, _SECRET_FILE_MODE)
    try:
        payload = payload_factory()
        if not isinstance(payload, bytes) or not payload:
            raise ValueError(f"secret payload is invalid for {path.name}")
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            if written <= 0:
                raise OSError(f"short write while creating {path.name}")
            offset += written
        os.fsync(descriptor)
        os.fchmod(descriptor, _SECRET_FILE_MODE)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        os.close(descriptor)
    try:
        os.link(temporary, path)
        created = True
    except FileExistsError:
        created = False
    finally:
        temporary.unlink(missing_ok=True)
    path.chmod(_SECRET_FILE_MODE)
    return created


def local_secret_directory(
    state_root_dir: str | Path, stack_layout: RuntimeStackLayout
) -> Path:
    return (
        Path(state_root_dir).expanduser().absolute()
        / DEFAULT_OUTPUT_DIR_NAME
        / LOCAL_SECRETS_DIR_NAME
        / stack_layout.stack_name
    )


def local_bootstrap_token_directory(
    state_root_dir: str | Path, stack_layout: RuntimeStackLayout
) -> Path:
    """Return the dedicated one-time authority directory.

    The Dashboard receives this directory as its only writable secret mount so
    completing first-admin provisioning cannot expose or mutate any long-lived
    Router or issuer key material.
    """

    return (
        local_secret_directory(state_root_dir, stack_layout) / LOCAL_BOOTSTRAP_DIR_NAME
    )


def _ensure_tls_server(
    secret_dir: Path, *, name: str, hostname: str
) -> tuple[Path, Path, bool]:
    private_key = secret_dir / f"{name}-tls-key.pem"
    certificate = secret_dir / f"{name}-tls-cert.pem"
    created_key = _ensure_secret(
        private_key, lambda: _run_openssl(["genpkey", "-algorithm", "ED25519"])
    )

    def certificate_payload() -> bytes:
        return _run_openssl(
            [
                "req",
                "-new",
                "-x509",
                "-key",
                str(private_key),
                "-days",
                "3650",
                "-subj",
                f"/CN={hostname}",
                "-addext",
                f"subjectAltName=DNS:{hostname},DNS:localhost,IP:127.0.0.1",
            ]
        )

    created = _ensure_secret(certificate, certificate_payload) or created_key
    return certificate, private_key, created


def local_dashboard_environment(
    state_root_dir: str | Path, stack_layout: RuntimeStackLayout
) -> dict[str, str]:
    """Resolve first-install Dashboard issuer settings from private local state."""

    secret_dir = local_secret_directory(state_root_dir, stack_layout)
    required = {
        "issuer_id": secret_dir / "dashboard-issuer-id",
        "signing_key": secret_dir / "dashboard-issuer-signing-key.pem",
        "tls_certificate": secret_dir / "dashboard-issuer-tls-cert.pem",
        "tls_key": secret_dir / "dashboard-issuer-tls-key.pem",
        "trust_bundle": secret_dir / "local-tls-trust-bundle.pem",
        "jwt_secret": secret_dir / "dashboard-jwt-secret",
    }
    if not all(path.is_file() for path in required.values()):
        return {}
    return {
        "DASHBOARD_ISSUER": (f"https://{stack_layout.dashboard_container_name}:8743"),
        "DASHBOARD_ISSUER_ID": required["issuer_id"]
        .read_text(encoding="utf-8")
        .strip(),
        "DASHBOARD_SIGNING_KEY_FILE": str(required["signing_key"]),
        "DASHBOARD_KEY_ID": "local-v1",
        "DASHBOARD_ISSUER_TLS_LISTEN_ADDR": ":8743",
        "DASHBOARD_ISSUER_TLS_CERT_FILE": str(required["tls_certificate"]),
        "DASHBOARD_ISSUER_TLS_KEY_FILE": str(required["tls_key"]),
        "DASHBOARD_ROUTER_BOOTSTRAP_TOKEN_FILE": str(
            local_bootstrap_token_directory(state_root_dir, stack_layout)
            / LOCAL_BOOTSTRAP_TOKEN_NAME
        ),
        "DASHBOARD_JWT_SECRET": required["jwt_secret"]
        .read_text(encoding="utf-8")
        .strip(),
        "SSL_CERT_FILE": str(required["trust_bundle"]),
    }


def _build_managed_config(secret_dir: Path, stack_layout: RuntimeStackLayout) -> dict:
    paths = {
        "api": secret_dir / "api-key-hmac.json",
        "delegation": secret_dir / "delegation-hmac.json",
        "reveal": secret_dir / "reveal-kek.json",
        "tenant": secret_dir / "tenant-signing.json",
        "provider": secret_dir / "provider-kek.json",
        "management": secret_dir / "management-signing.json",
        "service": secret_dir / "service-account-hmac.json",
        "invitation": secret_dir / "invitation-hmac.json",
        "control": secret_dir / "control-plane-hmac.json",
        "response": secret_dir / "response-kek.json",
        "bootstrap": secret_dir / LOCAL_BOOTSTRAP_DIR_NAME / LOCAL_BOOTSTRAP_TOKEN_NAME,
        "certificate": secret_dir / "management-tls-cert.pem",
        "private_key": secret_dir / "management-tls-key.pem",
    }
    return {
        "version": "v0.4",
        "listeners": [
            {
                "name": f"http-{DEFAULT_LISTENER_PORT}",
                "address": "0.0.0.0",
                "port": DEFAULT_LISTENER_PORT,
                "timeout": "300s",
            }
        ],
        "global": {
            "control_plane": {
                "mode": "managed",
                "provider_catalog": {
                    "replica_id_env": LOCAL_REPLICA_ID_ENV,
                    "rollout_groups": [
                        {"plane": "control", "id": "management"},
                        {"plane": "data", "id": "router"},
                    ],
                    "required_rollout_groups": [
                        {"plane": "control", "id": "management"},
                        {"plane": "data", "id": "router"},
                    ],
                },
            },
            "stores": {
                "access": {
                    "type": "postgres",
                    "postgres": {"dsn_env": LOCAL_POSTGRES_DSN_ENV},
                },
                "access_runtime": {
                    "type": "redis",
                    "redis": {"url_env": LOCAL_VALKEY_URL_ENV},
                },
            },
            "services": {
                "agent": {
                    "public_inference_endpoint": (
                        stack_layout.envoy_listener_service_url(DEFAULT_LISTENER_PORT)
                        + "/v1/chat/completions"
                    )
                },
                "access": {
                    "enabled": True,
                    "credentials": {
                        "api_key_hmac_keyring_file": str(paths["api"]),
                        "delegation_hmac_keyring_file": str(paths["delegation"]),
                        "reveal": {
                            "enabled": True,
                            "kek_keyring_file": str(paths["reveal"]),
                        },
                    },
                    "tenant_context": {"signing_key_file": str(paths["tenant"])},
                },
                "backend_credentials": {
                    "provider_kek_keyring_file": str(paths["provider"])
                },
                "backend_dispatch": dict(DEFAULT_BACKEND_DISPATCH),
                "backend_egress": {
                    "policy_file": "/app/config/backend-egress-policy.yaml"
                },
                "management_api": {
                    "bind_address": "0.0.0.0",
                    "port": 8080,
                    "remote_exposure": False,
                    "tls": {
                        "certificate_file": str(paths["certificate"]),
                        "private_key_file": str(paths["private_key"]),
                    },
                    "auth": {
                        "mode": "router",
                        "token_signing_keyring_file": str(paths["management"]),
                        "service_account_hmac_keyring_file": str(paths["service"]),
                        "invitation_hmac_keyring_file": str(paths["invitation"]),
                        "control_plane_hmac_keyring_file": str(paths["control"]),
                        "response_kek_keyring_file": str(paths["response"]),
                        "bootstrap": {
                            "token_file": str(paths["bootstrap"]),
                            "disable_after_first_cluster_admin": True,
                        },
                    },
                },
            },
        },
    }


def ensure_bootstrap_workspace(
    config_path: str | Path,
    *,
    state_root_dir: str | Path | None = None,
    stack_layout: RuntimeStackLayout | None = None,
) -> BootstrapResult:
    """Create the canonical local managed bootstrap once, preserving existing config."""

    path = Path(config_path).expanduser().absolute()
    stack_layout = stack_layout or resolve_runtime_stack()
    state_root = (
        Path(state_root_dir).expanduser().absolute()
        if state_root_dir is not None
        else path.parent
    )
    output_dir = state_root / DEFAULT_OUTPUT_DIR_NAME
    secret_dir = local_secret_directory(state_root, stack_layout)
    path.parent.mkdir(parents=True, exist_ok=True)
    created_output_dir = _ensure_private_directory(output_dir)
    if path.exists():
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"config path is not a regular file: {path}")
        return BootstrapResult(
            config_path=path,
            output_dir=output_dir,
            secret_dir=secret_dir,
            created_output_dir=created_output_dir,
        )
    _ensure_private_directory(output_dir / LOCAL_SECRETS_DIR_NAME)
    _ensure_private_directory(secret_dir)
    bootstrap_token_dir = local_bootstrap_token_directory(state_root, stack_layout)
    _ensure_private_directory(bootstrap_token_dir)

    created_secrets = False
    for name in (
        "api-key-hmac.json",
        "delegation-hmac.json",
        "reveal-kek.json",
        "provider-kek.json",
        "service-account-hmac.json",
        "invitation-hmac.json",
        "control-plane-hmac.json",
        "response-kek.json",
    ):
        created_secrets = (
            _ensure_secret(secret_dir / name, _symmetric_keyring) or created_secrets
        )
    for name in ("tenant-signing.json", "management-signing.json"):
        created_secrets = (
            _ensure_secret(secret_dir / name, _signing_keyring) or created_secrets
        )
    created_secrets = (
        _ensure_secret(
            bootstrap_token_dir / LOCAL_BOOTSTRAP_TOKEN_NAME,
            lambda: (secrets.token_urlsafe(48) + "\n").encode("utf-8"),
        )
        or created_secrets
    )
    management_certificate, _management_key, management_tls_created = (
        _ensure_tls_server(
            secret_dir,
            name="management",
            hostname=stack_layout.router_container_name,
        )
    )
    dashboard_certificate, _dashboard_key, dashboard_tls_created = _ensure_tls_server(
        secret_dir,
        name="dashboard-issuer",
        hostname=stack_layout.dashboard_container_name,
    )
    created_secrets = management_tls_created or dashboard_tls_created or created_secrets
    created_secrets = (
        _ensure_secret(
            secret_dir / "local-tls-trust-bundle.pem",
            lambda: (
                management_certificate.read_bytes() + dashboard_certificate.read_bytes()
            ),
        )
        or created_secrets
    )
    created_secrets = (
        _ensure_secret(
            secret_dir / "dashboard-issuer-signing-key.pem",
            lambda: _run_openssl(["genpkey", "-algorithm", "ED25519"]),
        )
        or created_secrets
    )
    created_secrets = (
        _ensure_secret(
            secret_dir / "dashboard-issuer-id",
            lambda: (str(uuid.uuid4()) + "\n").encode("utf-8"),
        )
        or created_secrets
    )
    created_secrets = (
        _ensure_secret(
            secret_dir / "dashboard-jwt-secret",
            lambda: (secrets.token_urlsafe(48) + "\n").encode("utf-8"),
        )
        or created_secrets
    )
    created_config = False
    if not path.exists():
        payload = yaml.safe_dump(
            _build_managed_config(secret_dir, stack_layout), sort_keys=False
        ).encode("utf-8")
        created_config = _ensure_secret(path, lambda: payload)

    return BootstrapResult(
        config_path=path,
        output_dir=output_dir,
        secret_dir=secret_dir,
        created_config=created_config,
        created_output_dir=created_output_dir,
        created_secrets=created_secrets,
    )
