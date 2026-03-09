from cli.commands.runtime_support import apply_runtime_mode_env_vars


def test_apply_runtime_mode_env_vars_sets_dashboard_readonly_when_requested():
    env_vars: dict[str, str] = {}

    apply_runtime_mode_env_vars(
        env_vars=env_vars,
        minimal=False,
        readonly=True,
        setup_mode=False,
        platform=None,
    )

    assert env_vars["DASHBOARD_READONLY"] == "true"


def test_apply_runtime_mode_env_vars_skips_dashboard_readonly_in_minimal_mode():
    env_vars: dict[str, str] = {}

    apply_runtime_mode_env_vars(
        env_vars=env_vars,
        minimal=True,
        readonly=True,
        setup_mode=False,
        platform=None,
    )

    assert env_vars["DISABLE_DASHBOARD"] == "true"
    assert "DASHBOARD_READONLY" not in env_vars
