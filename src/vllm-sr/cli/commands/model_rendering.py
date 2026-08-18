"""Human-readable rendering for model discovery commands."""

from __future__ import annotations

import textwrap
from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import Any

from cli.terminal import echo, heading, output_width

_WIDE_LIST_MIN_WIDTH = 96
_MAX_TABLE_STATUS_WIDTH = 24
_MIN_TRAITS_COLUMN_WIDTH = 20


def render_builtin_models(models: Iterable[Any]) -> None:
    """Render installed catalog models without runtime log decoration."""
    model_list = list(models)
    width = output_width()
    heading(f"Built-in models ({len(model_list)})")
    if not model_list:
        echo("No compatible models are installed.")
        return

    grouped: dict[tuple[str, str], list[Any]] = defaultdict(list)
    for model in model_list:
        grouped[(model.catalog_version, model.channel)].append(model)

    for index, ((version, channel), catalog_models) in enumerate(grouped.items()):
        echo()
        heading(f"Catalog: {_catalog_label(version, channel)}")
        if _use_stacked_list(catalog_models, width):
            _render_stacked_models(catalog_models, width)
        else:
            _render_model_table(catalog_models, width)

        if index + 1 < len(grouped):
            echo()

    echo()
    heading("Next step")
    _render_paragraph(
        "Serve one or more catalog entrypoints: " "vllm-sr serve <MODEL> [MODEL]...",
        width,
    )
    _render_paragraph(
        "Physical inference engines must already be running. Connect user-owned "
        "model endpoints with:",
        width,
    )
    _render_paragraph("vllm-sr serve --config <PATH>", width)


def render_model_details(details: dict[str, Any]) -> None:
    """Render one catalog model as grouped, scan-friendly details."""
    width = output_width()
    heading(details["display_name"])
    _render_paragraph(str(details["id"]), width)
    description = details.get("description")
    if description:
        echo()
        _render_paragraph(str(description), width)

    echo()
    _render_section(
        "Overview",
        (
            (
                "Catalog",
                _catalog_label(details["catalog_version"], details["channel"]),
            ),
            ("Policy", details["policy_version"]),
            (
                "Type",
                f"{details['kind']} ({details['family']} v{details['generation']})",
            ),
            ("Recipe", details["recipe"]),
            (
                "Status",
                f"{details['verification']['status']}; "
                f"{details['compatibility_reason']}",
            ),
            ("Verified by", details["verification"]["authority"]),
            ("Asset digest", details["verification"]["asset_sha256"]),
        ),
        width,
    )

    echo()
    _render_section(
        "Interfaces",
        (
            ("Entrypoint", details["entrypoint"]),
            ("Protocols", ", ".join(details["protocols"])),
            ("Traits", ", ".join(details["traits"])),
        ),
        width,
    )

    echo()
    heading("Backend requirements")
    for role_index, role in enumerate(details["roles"]):
        if role_index:
            echo()
        heading(role["name"], indent=2)
        requirement = "required" if role["required"] else "optional"
        _render_fields(
            (
                (
                    "Requirement",
                    f"{requirement}; minimum {role['minimum_candidates']}",
                ),
                ("Traits", ", ".join(role["traits"])),
                ("Recommended", ", ".join(role["recommended_pool"])),
            ),
            width,
            indent=4,
        )


def render_configured_models(configured: dict[str, Any]) -> None:
    """Render provider models and model cards from one runtime config."""
    width = output_width()
    heading("Configured models")
    _render_fields((("Source", configured["source"]),), width)

    _render_provider_models(configured["provider_models"], width)
    _render_model_cards(configured["model_cards"], width)


def _render_provider_models(
    provider_models: Sequence[dict[str, Any]], width: int
) -> None:
    """Render the provider-model half of one configured-model payload."""

    echo()
    heading(f"Provider models ({len(provider_models)})")
    if not provider_models:
        echo("  None configured")
    for index, model in enumerate(provider_models):
        if index:
            echo()
        suffix = " (default)" if model["default"] else ""
        heading(f"{model['id']}{suffix}", indent=2)
        fields = []
        if model["provider_model_id"] and model["provider_model_id"] != model["id"]:
            fields.append(("Provider model", model["provider_model_id"]))
        if model["reasoning_family"]:
            fields.append(("Reasoning", model["reasoning_family"]))
        if model["api_format"]:
            fields.append(("API format", model["api_format"]))
        if fields:
            _render_fields(fields, width, indent=4)
        for backend in model["backends"]:
            _render_fields(
                (("Backend", backend["name"] or "(unnamed)"),),
                width,
                indent=4,
            )
            _render_fields(
                (
                    ("Provider", backend["provider"] or "-"),
                    ("Base URL", backend["base_url"] or "-"),
                    ("Protocol", backend["protocol"]),
                    ("Weight", str(backend["weight"])),
                ),
                width,
                indent=6,
            )


def _render_model_cards(model_cards: Sequence[dict[str, Any]], width: int) -> None:
    """Render the model-card half of one configured-model payload."""
    echo()
    heading(f"Model cards ({len(model_cards)})")
    if not model_cards:
        echo("  None configured")
    for index, card in enumerate(model_cards):
        if index:
            echo()
        heading(card["id"], indent=2)
        fields = []
        if card["modality"]:
            fields.append(("Modality", card["modality"]))
        if card["param_size"]:
            fields.append(("Parameters", card["param_size"]))
        if card["context_window_size"]:
            fields.append(("Context window", str(card["context_window_size"])))
        if card["capabilities"]:
            fields.append(("Capabilities", ", ".join(card["capabilities"])))
        if card["tags"]:
            fields.append(("Tags", ", ".join(card["tags"])))
        if card["loras"]:
            fields.append(("LoRAs", ", ".join(card["loras"])))
        if fields:
            _render_fields(fields, width, indent=4)


def _catalog_label(version: str, channel: str) -> str:
    if version == channel:
        return version
    return f"{version} ({channel})"


def _use_stacked_list(models: Sequence[Any], width: int) -> bool:
    if width < _WIDE_LIST_MIN_WIDTH:
        return True
    longest_status = max(len(_model_status(model)) for model in models)
    return longest_status > _MAX_TABLE_STATUS_WIDTH


def _render_model_table(models: Sequence[Any], width: int) -> None:
    model_width = max(len("MODEL"), *(len(model.id) for model in models))
    default_width = len("CATALOG DEFAULT")
    status_width = max(len("STATUS"), *(len(_model_status(model)) for model in models))
    fixed_width = model_width + default_width + status_width + 6
    traits_width = width - fixed_width
    if traits_width < _MIN_TRAITS_COLUMN_WIDTH:
        _render_stacked_models(models, width)
        return

    echo(
        f"{'MODEL':<{model_width}}  {'CATALOG DEFAULT':<{default_width}}  "
        f"{'STATUS':<{status_width}}  TRAITS"
    )
    for model in models:
        default = "yes" if model.default else "no"
        status = _model_status(model)
        traits = _wrap_value(", ".join(model.traits), traits_width)
        echo(
            f"{model.id:<{model_width}}  {default:<{default_width}}  "
            f"{status:<{status_width}}  {traits[0]}"
        )
        continuation = " " * fixed_width
        for line in traits[1:]:
            echo(f"{continuation}{line}")


def _render_stacked_models(models: Sequence[Any], width: int) -> None:
    for index, model in enumerate(models):
        if index:
            echo()
        suffix = " (catalog default)" if model.default else ""
        heading(f"{model.id}{suffix}")
        _render_fields(
            (
                ("Status", _model_status(model)),
                ("Traits", ", ".join(model.traits)),
            ),
            width,
        )


def _model_status(model: Any) -> str:
    if not model.compatibility.compatible:
        return model.compatibility.reason
    if model.verified:
        return "verified"
    return "unverified"


def _render_section(
    title: str,
    fields: Sequence[tuple[str, str]],
    width: int,
) -> None:
    heading(title)
    _render_fields(fields, width)


def _render_fields(
    fields: Sequence[tuple[str, str]],
    width: int,
    *,
    indent: int = 2,
) -> None:
    if not fields:
        return
    label_width = max(len(label) for label, _ in fields)
    value_width = max(20, width - indent - label_width - 2)
    prefix = " " * indent
    for label, raw_value in fields:
        value_lines = _wrap_value(str(raw_value), value_width)
        echo(f"{prefix}{label:<{label_width}}  {value_lines[0]}")
        continuation = " " * (indent + label_width + 2)
        for line in value_lines[1:]:
            echo(f"{continuation}{line}")


def _render_paragraph(value: str, width: int) -> None:
    for line in _wrap_value(value, width):
        echo(line)


def _wrap_value(value: str, width: int) -> list[str]:
    return textwrap.wrap(
        value,
        width=max(1, width),
        break_long_words=True,
        break_on_hyphens=False,
    ) or [""]
