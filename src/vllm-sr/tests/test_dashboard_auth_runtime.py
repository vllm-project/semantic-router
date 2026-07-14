import hashlib
from pathlib import Path

import pytest
from cli.dashboard_auth_runtime import (
    DASHBOARD_ADMIN_PASSWORD_ENV,
    DASHBOARD_JWT_SECRET_ENV,
    DASHBOARD_PASSWORD_BLOCKLIST_CONTAINER_PATH,
    DASHBOARD_PASSWORD_BLOCKLIST_PATH_ENV,
    DASHBOARD_PASSWORD_BLOCKLIST_SHA256_ENV,
    DASHBOARD_SECURITY_PROFILE_DEVELOPMENT,
    DASHBOARD_SECURITY_PROFILE_ENV,
    DASHBOARD_SECURITY_PROFILE_PRODUCTION,
    MINIMUM_PRODUCTION_BLOCKLIST_ENTRIES,
    build_dashboard_auth_runtime_plan,
    without_dashboard_auth_env,
)


def test_plan_resolves_blocklist_to_readonly_fixed_container_mount(tmp_path: Path):
    blocklist = tmp_path / "passwords.txt"
    blocklist.write_text("known-compromised-password\n")

    plan = build_dashboard_auth_runtime_plan(
        {
            DASHBOARD_PASSWORD_BLOCKLIST_PATH_ENV: str(blocklist),
            DASHBOARD_JWT_SECRET_ENV: "jwt-secret-value",
        },
        dashboard_enabled=True,
    )

    assert plan.container_env[DASHBOARD_PASSWORD_BLOCKLIST_PATH_ENV] == (
        DASHBOARD_PASSWORD_BLOCKLIST_CONTAINER_PATH
    )
    assert plan.mount_specs == (
        f"{blocklist.resolve()}:{DASHBOARD_PASSWORD_BLOCKLIST_CONTAINER_PATH}:ro,z",
    )
    assert plan.container_env[DASHBOARD_SECURITY_PROFILE_ENV] == (
        DASHBOARD_SECURITY_PROFILE_DEVELOPMENT
    )
    assert plan.inherited_secret_names == frozenset({DASHBOARD_JWT_SECRET_ENV})


@pytest.mark.parametrize("kind", ["missing", "directory"])
def test_plan_rejects_non_regular_blocklist_paths(tmp_path: Path, kind: str):
    path = tmp_path / kind
    if kind == "directory":
        path.mkdir()

    with pytest.raises(ValueError, match=DASHBOARD_PASSWORD_BLOCKLIST_PATH_ENV):
        build_dashboard_auth_runtime_plan(
            {DASHBOARD_PASSWORD_BLOCKLIST_PATH_ENV: str(path)},
            dashboard_enabled=True,
        )


def test_disabled_dashboard_ignores_invalid_blocklist_and_auth_values(tmp_path: Path):
    plan = build_dashboard_auth_runtime_plan(
        {
            DASHBOARD_PASSWORD_BLOCKLIST_PATH_ENV: str(tmp_path / "missing"),
            DASHBOARD_ADMIN_PASSWORD_ENV: "must-not-propagate",
        },
        dashboard_enabled=False,
    )

    assert plan.enabled is False
    assert plan.container_env == {}
    assert plan.mount_specs == ()
    assert plan.secret_process_env == {}


def test_plan_repr_and_sanitized_env_do_not_expose_secret_values():
    secret = "super-secret-jwt-value"
    password = "super-secret-admin-password"
    source = {
        DASHBOARD_JWT_SECRET_ENV: secret,
        DASHBOARD_ADMIN_PASSWORD_ENV: password,
        "SR_LOG_LEVEL": "info",
    }

    plan = build_dashboard_auth_runtime_plan(source, dashboard_enabled=True)

    assert secret not in repr(plan)
    assert password not in repr(plan)
    assert without_dashboard_auth_env(source) == {"SR_LOG_LEVEL": "info"}


def test_production_plan_verifies_digest_and_nfc_unique_entry_floor(tmp_path: Path):
    blocklist, digest = _write_production_blocklist(
        tmp_path, include_nfc_duplicate=True
    )

    plan = build_dashboard_auth_runtime_plan(
        {
            DASHBOARD_SECURITY_PROFILE_ENV: DASHBOARD_SECURITY_PROFILE_PRODUCTION,
            DASHBOARD_PASSWORD_BLOCKLIST_PATH_ENV: str(blocklist),
            DASHBOARD_PASSWORD_BLOCKLIST_SHA256_ENV: digest.upper(),
        },
        dashboard_enabled=True,
    )

    assert plan.container_env[DASHBOARD_SECURITY_PROFILE_ENV] == (
        DASHBOARD_SECURITY_PROFILE_PRODUCTION
    )
    assert plan.container_env[DASHBOARD_PASSWORD_BLOCKLIST_SHA256_ENV] == digest
    assert plan.mount_specs == (
        f"{blocklist.resolve()}:{DASHBOARD_PASSWORD_BLOCKLIST_CONTAINER_PATH}:ro,z",
    )


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (
            {DASHBOARD_SECURITY_PROFILE_ENV: "staging"},
            DASHBOARD_SECURITY_PROFILE_ENV,
        ),
        (
            {DASHBOARD_SECURITY_PROFILE_ENV: DASHBOARD_SECURITY_PROFILE_PRODUCTION},
            DASHBOARD_PASSWORD_BLOCKLIST_PATH_ENV,
        ),
        (
            {DASHBOARD_PASSWORD_BLOCKLIST_SHA256_ENV: "0" * 64},
            DASHBOARD_PASSWORD_BLOCKLIST_PATH_ENV,
        ),
    ],
)
def test_plan_rejects_incomplete_security_profile(source: dict[str, str], message: str):
    with pytest.raises(ValueError, match=message):
        build_dashboard_auth_runtime_plan(source, dashboard_enabled=True)


def test_production_plan_rejects_missing_mismatched_or_small_corpus(tmp_path: Path):
    small = tmp_path / "small.txt"
    small.write_text("one compromised credential\n")
    small_digest = hashlib.sha256(small.read_bytes()).hexdigest()

    cases = [
        (
            {
                DASHBOARD_SECURITY_PROFILE_ENV: DASHBOARD_SECURITY_PROFILE_PRODUCTION,
                DASHBOARD_PASSWORD_BLOCKLIST_PATH_ENV: str(small),
            },
            DASHBOARD_PASSWORD_BLOCKLIST_SHA256_ENV,
        ),
        (
            {
                DASHBOARD_SECURITY_PROFILE_ENV: DASHBOARD_SECURITY_PROFILE_PRODUCTION,
                DASHBOARD_PASSWORD_BLOCKLIST_PATH_ENV: str(small),
                DASHBOARD_PASSWORD_BLOCKLIST_SHA256_ENV: "0" * 64,
            },
            "does not match",
        ),
        (
            {
                DASHBOARD_SECURITY_PROFILE_ENV: DASHBOARD_SECURITY_PROFILE_PRODUCTION,
                DASHBOARD_PASSWORD_BLOCKLIST_PATH_ENV: str(small),
                DASHBOARD_PASSWORD_BLOCKLIST_SHA256_ENV: small_digest,
            },
            "at least 10000",
        ),
    ]
    for source, message in cases:
        with pytest.raises(ValueError, match=message):
            build_dashboard_auth_runtime_plan(source, dashboard_enabled=True)


def test_plan_rejects_invalid_unicode_before_runtime_mutation(tmp_path: Path):
    blocklist = tmp_path / "invalid-unicode.txt"
    blocklist.write_bytes(b"valid compromised password\n\xff\n")

    with pytest.raises(ValueError, match="invalid Unicode"):
        build_dashboard_auth_runtime_plan(
            {DASHBOARD_PASSWORD_BLOCKLIST_PATH_ENV: str(blocklist)},
            dashboard_enabled=True,
        )


def _write_production_blocklist(
    tmp_path: Path, *, include_nfc_duplicate: bool = False
) -> tuple[Path, str]:
    canary = "local breached café credential"
    entries = [canary]
    entries.extend(
        f"offline compromised corpus entry {index:05d}"
        for index in range(1, MINIMUM_PRODUCTION_BLOCKLIST_ENTRIES)
    )
    if include_nfc_duplicate:
        entries.append(canary.replace("é", "e\u0301"))
    raw = ("\n".join(entries) + "\n").encode()
    path = tmp_path / "production-passwords.txt"
    path.write_bytes(raw)
    return path, hashlib.sha256(raw).hexdigest()
