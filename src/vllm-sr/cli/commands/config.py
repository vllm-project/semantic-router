"""Config command implementation."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

from cli import router_validation
from cli.commands.runtime_paths import resolve_state_root_dir
from cli.commands.validate import (
    ask_router,
    collect_config_errors,
    router_check_line,
    router_refusal,
    router_warnings,
)
from cli.config_generator import generate_envoy_config_from_user_config
from cli.config_schema import schema_document
from cli.config_schema.views import parse_surface_selector, schema_view
from cli.config_yaml import safe_load_router_config
from cli.gateway_mode import GATEWAY_ENV, GATEWAY_STANDALONE, resolve_gateway
from cli.parser import (
    ConfigParseError,
    parse_user_config,
    parse_user_config_data,
)
from cli.router_management_client import RouterManagementClient
from cli.router_validation import (
    ROUTER_BINARY_ENV,
    RouterVerdict,
    RouterWarning,
    with_moving_tag_hint,
)
from cli.terminal import echo, fields, heading, success, warning
from cli.utils import get_logger
from cli.validation_error import ValidationError
from cli.validator import (
    print_validation_errors,
    print_validation_warnings,
    validate_user_config,
)

log = get_logger(__name__)
# `config migrate` has no --image: the environment names its Router.
MIGRATE_NAMES_ROUTER = "`VLLM_SR_IMAGE`"
# A standalone Router names this key when it refuses identity headers that no
# listener trusts.
TRUST_HEADERS_KEY = "listeners[].identity.trust_headers"
UNTRUSTED_IDENTITY_HINT = (
    "Re-run with `--gateway extproc` when Envoy or another authenticating "
    f"gateway sits in front of the Router, or set `{TRUST_HEADERS_KEY}: true` "
    "only behind a trusted proxy."
)
CONFIG_TEMPLATE_PATH = (
    Path(__file__).resolve().parents[1] / "templates" / "config.template.yaml"
)


def init_config_command(
    output_path: str = "config.yaml",
    *,
    force: bool = False,
) -> Path:
    """Write the packaged minimal canonical configuration template."""

    destination = Path(output_path)
    if destination.exists():
        if destination.is_dir():
            raise ValueError(f"Config output path is a directory: {destination}")
        if not force:
            raise ValueError(
                f"Config file already exists: {destination}. Use --force to overwrite it."
            )
    try:
        template = CONFIG_TEMPLATE_PATH.read_text(encoding="utf-8")
    except OSError as exc:
        raise RuntimeError("packaged config template is unavailable") from exc
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(template, encoding="utf-8")

    success("Configuration template created")
    fields((("Output", destination),))
    return destination


def config_schema_command(
    endpoint: str | None = None,
    *,
    full: bool = False,
    section: str | None = None,
    surface: str | None = None,
    expanded: bool = False,
    timeout: float = 15,
    token_env: str = "VSR_MGMT_TOKEN",
) -> None:
    """Print one progressive local or deployed Router contract view."""

    selected = sum((full, section is not None, surface is not None))
    if selected > 1:
        raise ValueError("use only one of --full, --section, or --surface")
    if expanded and section is None:
        raise ValueError("--expanded requires --section")
    view = (
        "full" if full else "section" if section else "surface" if surface else "index"
    )
    surface_kind = None
    surface_name = None
    if surface:
        surface_kind, surface_name = parse_surface_selector(surface)

    if endpoint:
        document = (
            RouterManagementClient(
                endpoint,
                timeout=timeout,
                token_env=token_env,
            )
            .get_config_schema(
                view=view,
                path=section,
                surface_kind=surface_kind,
                surface_name=surface_name,
                expanded=expanded,
            )
            .payload
        )
        echo(json.dumps(document, indent=2, sort_keys=True) + "\n", nl=False)
        return

    document = schema_view(
        schema_document(),
        view=view,
        path=section,
        surface_kind=surface_kind,
        surface_name=surface_name,
        expanded=expanded,
    )
    echo(json.dumps(document, indent=2, sort_keys=True) + "\n", nl=False)


def config_command(config_type: str, config_path: str = "config.yaml"):
    """
    Print generated configuration.

    Args:
        config_type: Type of config to print ('envoy' or 'router')
        config_path: Path to user config.yaml (default: config.yaml)
    """
    if config_type not in ["envoy", "router"]:
        log.error(f"Invalid config type: {config_type}")
        log.error("Must be 'envoy' or 'router'")
        sys.exit(1)

    # Check if config file exists
    if not Path(config_path).exists():
        log.error(f"Config file not found: {config_path}")
        log.error(
            "Run 'vllm-sr serve' to bootstrap setup mode and create a config file"
        )
        log.error(
            "Or write a canonical v0.5 config.yaml using the docs examples if you want to hand-author it directly"
        )
        sys.exit(1)

    # Parse user config
    try:
        user_config = parse_user_config(config_path, log_summary=False)
    except ConfigParseError as e:
        log.error(f"Failed to parse configuration: {e}")
        sys.exit(1)

    # Validate user config
    errors = validate_user_config(user_config, log_summary=False)
    if errors:
        print_validation_errors(errors)
        sys.exit(1)

    if config_type == "router":
        # Router now reads canonical config.yaml directly.
        echo(Path(config_path).read_text(encoding="utf-8"), nl=False)

    elif config_type == "envoy":
        # Generate envoy config
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", delete=False
            ) as f:
                temp_path = f.name

            generate_envoy_config_from_user_config(
                user_config,
                temp_path,
                log_summary=False,
            )

            # Read and print
            with open(temp_path, encoding="utf-8") as f:
                echo(f.read(), nl=False)

            # Clean up
            Path(temp_path).unlink()

        except Exception as e:
            log.error(f"Failed to generate Envoy config: {e}")
            sys.exit(1)


def migrate_config_command(
    config_path: str = "config.yaml",
    output_path: str | None = None,
    force: bool = False,
    gateway: str | None = None,
) -> Path:
    """Migrate a config file to canonical v0.5 YAML.

    Every rewrite lives in the Router: the file goes to a local Router once
    with -migrate-config, and the output is written only when the CLI's
    structural checks and the Router's validation in the *gateway* mode
    (default: $VLLM_SR_GATEWAY or standalone) accept it.
    """

    gateway = resolve_gateway(gateway or os.getenv(GATEWAY_ENV))
    source_path = Path(config_path)
    if not source_path.exists():
        log.error(f"Config file not found: {config_path}")
        sys.exit(1)
    destination = (
        Path(output_path)
        if output_path
        else source_path.with_name(f"{source_path.stem}.migrated{source_path.suffix}")
    )
    if destination.exists() and not force:
        log.error(f"Output file already exists: {destination}")
        log.error("Use --force to overwrite the destination")
        sys.exit(1)

    migration, unavailable = ask_router(
        lambda: _router_verdict(source_path, config_path, gateway, migrate=True)
    )
    if migration is None:
        log.error(f"No Router could migrate the configuration: {unavailable}")
        log.error(f"Or set {ROUTER_BINARY_ENV} to a Router binary")
        sys.exit(1)
    if migration.migration_error or not migration.migrated:
        reason = migration.migration_error or "it wrote nothing"
        log.error(
            with_moving_tag_hint(
                f"The Router could not migrate the configuration: {reason}",
                migration,
                MIGRATE_NAMES_ROUTER,
            )
        )
        log.error("No output was written")
        sys.exit(1)
    rendered = migration.migrated

    verdict = _validate_migrated_config(rendered, config_path, gateway)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(rendered, encoding="utf-8")

    success("Configuration migrated")
    heading("Files")
    fields(
        (
            ("Source", source_path),
            ("Output", destination),
        )
    )
    echo(router_check_line(verdict, "", ""))
    print_validation_warnings(router_warnings(verdict))
    _print_migration_notes(migration.notes)
    return destination


def _print_migration_notes(notes: tuple[RouterWarning, ...]) -> None:
    """The Router's notes: what migrate rewrote (migration_change) and what an
    operator must do for the file to work as before (migration_action)."""

    changes = [note for note in notes if note.code != "migration_action"]
    actions = [note for note in notes if note.code == "migration_action"]
    if changes:
        heading("Changes to review")
        for note in changes:
            echo(f"  {note.field or 'configuration'}")
            echo(f"      {note.message}")
    for note in actions:
        warning(f"{note.field or 'configuration'}: {note.message}")


def _validate_migrated_config(
    rendered: str, config_path: str, gateway: str
) -> RouterVerdict:
    """Hold migrated output to the `config validate` standard before writing.

    The rendered YAML is reloaded so the checks see exactly what would land on
    disk, and the Router that migrated it validates it again, in the *gateway*
    mode, for the warnings it would log.
    """

    try:
        user_config = parse_user_config_data(
            safe_load_router_config(rendered), log_summary=False
        )
    except ConfigParseError as e:
        log.error(f"Migrated configuration is not canonical: {e}")
        log.error("No output was written")
        sys.exit(1)

    errors = collect_config_errors(user_config)
    with tempfile.TemporaryDirectory(prefix="vllm-sr-migrate-") as directory:
        candidate = Path(directory) / "config.yaml"
        candidate.write_text(rendered, encoding="utf-8")
        verdict, unavailable = ask_router(
            lambda: _router_verdict(candidate, config_path, gateway)
        )
    if verdict is None:
        errors.append(
            ValidationError(f"the Router could not validate the output: {unavailable}")
        )
    refusal = router_refusal(verdict, MIGRATE_NAMES_ROUTER)
    if refusal and gateway == GATEWAY_STANDALONE and TRUST_HEADERS_KEY in verdict.error:
        refusal[0].hint = UNTRUSTED_IDENTITY_HINT
    errors.extend(refusal)
    if errors:
        print_validation_errors(errors)
        print_validation_warnings(router_warnings(verdict))
        log.error("No output was written")
        sys.exit(1)
    return verdict


def _router_verdict(
    candidate: Path, config_path: str, gateway: str, *, migrate: bool = False
) -> RouterVerdict:
    """A local Router's verdict on a file in the *gateway* mode, with its
    migration when migrate is set; model paths resolve in the models directory
    `serve` mounts."""

    return router_validation.local_router_verdict(
        candidate,
        gateway=gateway,
        models_dir=Path(resolve_state_root_dir(config_path)) / "models",
        migrate=migrate,
    )
