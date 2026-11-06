"""Config command implementation."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import yaml

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
from cli.config_migration import migrate_config_data
from cli.config_migration_notes import MigrationNotes
from cli.config_schema import schema_document
from cli.config_schema.views import parse_surface_selector, schema_view
from cli.config_yaml import safe_load_router_config
from cli.gateway_mode import GATEWAY_ENV, resolve_gateway
from cli.parser import (
    ConfigParseError,
    load_config_file,
    parse_user_config,
    parse_user_config_data,
)
from cli.router_management_client import RouterManagementClient
from cli.router_validation import ROUTER_BINARY_ENV, RouterVerdict
from cli.terminal import echo, fields, heading, success, warning
from cli.utils import get_logger
from cli.validator import (
    print_validation_errors,
    print_validation_warnings,
    validate_user_config,
)

log = get_logger(__name__)
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
            "Or write a canonical v0.3 config.yaml using the docs examples if you want to hand-author it directly"
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
    offline: bool = False,
) -> Path:
    """Migrate a legacy or mixed config file to canonical v0.3 YAML."""

    source_path = Path(config_path)
    if not source_path.exists():
        log.error(f"Config file not found: {config_path}")
        sys.exit(1)

    try:
        data = load_config_file(config_path)
    except ConfigParseError as e:
        log.error(f"Failed to read configuration: {e}")
        sys.exit(1)

    notes = MigrationNotes()
    migrated = migrate_config_data(data, notes)
    rendered = yaml.safe_dump(migrated, sort_keys=False, allow_unicode=True)
    rendered = _migrate_with_router(rendered, config_path, notes, offline=offline)

    destination = (
        Path(output_path)
        if output_path
        else source_path.with_name(f"{source_path.stem}.migrated{source_path.suffix}")
    )
    if destination.exists() and not force:
        log.error(f"Output file already exists: {destination}")
        log.error("Use --force to overwrite the destination")
        sys.exit(1)

    verdict, unavailable = _validate_migrated_config(
        rendered, config_path, offline=offline
    )

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
    echo(
        router_check_line(
            verdict,
            unavailable,
            f"set {ROUTER_BINARY_ENV}, or run `vllm-sr config validate` on the "
            "output, to ask it",
        )
    )
    print_validation_warnings(router_warnings(verdict))
    _print_migration_notes(notes)
    return destination


def _print_migration_notes(notes: MigrationNotes) -> None:
    changes = [note for note in notes if not note.action_required]
    actions = [note for note in notes if note.action_required]
    if changes:
        heading("Changes to review")
        for note in changes:
            echo(f"  {note.path}")
            echo(f"      {note.message}")
    for note in actions:
        warning(f"{note.path}: {note.message}")


def _migrate_with_router(
    rendered: str, config_path: str, notes: MigrationNotes, *, offline: bool
) -> str:
    """The document with the forms the Router no longer reads (removed and
    renamed forms, deprecated model fields) in the forms that replace them.

    Every rewrite lives in the Router: the CLI's output goes to a local Router
    once with -migrate-config. Without one (or with offline) those forms stay,
    and the notes say how to rewrite them.
    """

    path = "global.model_catalog"
    if offline:
        notes.changed(
            "configuration",
            "removed and renamed forms and deprecated model fields were not "
            "rewritten (--offline); run `vllm-sr config migrate` where a Router "
            "can run to rewrite them",
        )
        return rendered
    verdict, unavailable = ask_router(
        lambda: _migrated_verdict(rendered, config_path, migrate=True)
    )
    if verdict is None:
        notes.changed(
            "configuration",
            "removed and renamed forms and deprecated model fields were not "
            f"rewritten: {unavailable}; set {ROUTER_BINARY_ENV} to a Router "
            "binary to rewrite them",
        )
        return rendered
    # The Router rewrites a file it refuses as written (removed forms), so
    # only its own migration error stops the rewrite.
    if verdict.migration_error or not verdict.migrated:
        if verdict.migration_error:
            notes.action(
                path,
                f"the Router could not migrate the configuration: {verdict.migration_error}",
            )
        return rendered
    for note in verdict.kept:
        # migration_change is a form the Router rewrote; migration_action says
        # what to do for the file to work as before; every other note is a
        # deprecated field with no replacement.
        if note.code == "migration_change":
            notes.changed(note.field or path, note.message)
            continue
        message = (
            note.message if note.code == "migration_action" else f"kept: {note.message}"
        )
        notes.action(note.field or path, message)
    return verdict.migrated


def _validate_migrated_config(
    rendered: str, config_path: str, *, offline: bool = False
) -> tuple[RouterVerdict | None, str]:
    """Hold migrated output to the `config validate` standard before writing.

    The rendered YAML is reloaded so the checks see exactly what would land on
    disk. Fields migration carries over without a canonical form fail here with
    their field paths instead of at serve time. Returns the Router's verdict,
    or why no Router could give one (nothing with offline).
    """

    try:
        user_config = parse_user_config_data(
            safe_load_router_config(rendered), config_path, log_summary=False
        )
    except ConfigParseError as e:
        log.error(f"Migrated configuration is not canonical: {e}")
        log.error("No output was written")
        sys.exit(1)

    errors = collect_config_errors(user_config)
    verdict, unavailable = ask_router(
        None if offline else lambda: _migrated_verdict(rendered, config_path)
    )
    errors.extend(router_refusal(verdict))
    if errors:
        print_validation_errors(errors)
        print_validation_warnings(router_warnings(verdict))
        log.error("No output was written")
        sys.exit(1)
    return verdict, unavailable


def _migrated_verdict(
    rendered: str, config_path: str, *, migrate: bool = False
) -> RouterVerdict:
    """A local Router's verdict on the output before it is written, with the
    Router's migration of its model fields when migrate is set."""

    with tempfile.TemporaryDirectory(prefix="vllm-sr-migrate-") as directory:
        candidate = Path(directory) / "config.yaml"
        candidate.write_text(rendered, encoding="utf-8")
        return router_validation.local_router_verdict(
            candidate,
            gateway=resolve_gateway(os.getenv(GATEWAY_ENV)),
            models_dir=Path(resolve_state_root_dir(config_path)) / "models",
            migrate=migrate,
        )
