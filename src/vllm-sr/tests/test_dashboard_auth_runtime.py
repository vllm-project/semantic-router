from pathlib import Path

import pytest
from cli.dashboard_auth_runtime import (
    DASHBOARD_ADMIN_PASSWORD_ENV,
    DASHBOARD_JWT_SECRET_ENV,
    DASHBOARD_PASSWORD_BLOCKLIST_CONTAINER_PATH,
    DASHBOARD_PASSWORD_BLOCKLIST_PATH_ENV,
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
