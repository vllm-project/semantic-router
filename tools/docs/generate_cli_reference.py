#!/usr/bin/env python3
"""Generate the website CLI reference from the registered Click command tree."""

from __future__ import annotations

import argparse
import html
import inspect
import re
import sys
import textwrap
from pathlib import Path

import click

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "website/docs/api/cli.md"


def load_cli() -> click.Group:
    """Import source directly, without package build hooks or command invocation."""
    sys.path.insert(0, str(ROOT / "src/vllm-sr"))
    from cli.main import main

    return main


def command_tree(command: click.Command, name: str = "vllm-sr", parent=None):
    """Walk public commands in stable order without parsing or invoking them."""
    context = click.Context(
        command,
        info_name=name,
        parent=parent,
        terminal_width=100,
        max_content_width=100,
        resilient_parsing=True,
    )
    yield context
    if isinstance(command, click.Group):
        for child_name in sorted(command.list_commands(context)):
            child = command.get_command(context, child_name)
            if child is not None and not child.hidden:
                yield from command_tree(child, child_name, context)


def prose(value: str) -> str:
    """Keep help text literal in MDX, including types such as <runtime>."""
    return html.escape(value, quote=False).replace("{", "&#123;").replace("}", "&#125;")


def cell(value: str) -> str:
    return prose(value).replace("|", "&#124;").replace("\n", "<br />")


def code(value: str) -> str:
    delimiter = "`" * (
        max((len(part) for part in re.findall(r"`+", value)), default=0) + 1
    )
    return delimiter + value.replace("|", "\\|").replace("\n", " ") + delimiter


def anchor(context: click.Context) -> str:
    return context.command_path.replace(" ", "-")


def help_blocks(value: str | None) -> str:
    """Retain Click's explicit preformatted paragraphs and command examples."""
    if not value:
        return ""
    blocks = []
    for paragraph in inspect.cleandoc(value).split("\f", 1)[0].split("\n\n"):
        if "\b" in paragraph:
            literal = textwrap.dedent(paragraph.replace("\b\n", "")).strip()
            blocks.append("```text\n" + literal + "\n```")
        elif paragraph.startswith("Examples:\n"):
            example = textwrap.dedent(paragraph.split("\n", 1)[1]).strip()
            blocks.append("Examples:\n\n```bash\n" + example + "\n```")
        elif paragraph.startswith("    "):
            blocks.append("```text\n" + textwrap.dedent(paragraph).strip() + "\n```")
        else:
            blocks.append(prose(paragraph))
    return "\n\n".join(blocks)


def default_text(parameter: click.Parameter, context: click.Context) -> str | None:
    value = parameter.get_default(context, call=False)
    if value is None or value is getattr(click.core, "UNSET", None):
        return None
    if callable(value):
        return "computed at runtime"
    if isinstance(value, (tuple, list)):
        return ", ".join(str(item) for item in value) if value else "none"
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def parameters(context: click.Context) -> list[str]:
    rows = []
    for parameter in context.command.get_params(context):
        if getattr(parameter, "hidden", False):
            continue
        if isinstance(parameter, click.Option):
            record = parameter.get_help_record(context)
            if record is None:
                continue
            label, description = record
            if isinstance(parameter.type, click.Choice):
                label = label.replace(parameter.make_metavar(context), "CHOICE")
        else:
            label = parameter.metavar or (parameter.name or "argument").upper()
            if not parameter.required:
                label = f"[{label}]"
            if parameter.nargs != 1:
                label += "..."
            description = (
                "Required argument." if parameter.required else "Optional argument."
            )
            description += f" Type: {parameter.type.name}."
        if isinstance(parameter.type, click.Choice):
            description += (
                " Choices: "
                + ", ".join(str(choice) for choice in parameter.type.choices)
                + "."
            )
        default = default_text(parameter, context)
        if default is not None and "[default:" not in description:
            description += f" Default: {default}."
        if parameter.multiple:
            description += " May be repeated."
        if parameter.nargs == -1:
            description += " Accepts multiple values."
        if parameter.envvar:
            names = parameter.envvar
            if not isinstance(names, str):
                names = ", ".join(names)
            description += f" Environment: {names}."
        rows.append(f"| {code(label)} | {cell(description.strip()) or '—'} |")
    return rows


def render(command: click.Command) -> str:
    contexts = list(command_tree(command))
    lines = [
        "---",
        "title: CLI Commands",
        "description: Generated reference for every public vllm-sr command and option.",
        "toc_max_heading_level: 2",
        "---",
        "",
        "<!-- Generated by tools/docs/generate_cli_reference.py. Do not edit directly. -->",
        "",
        "# CLI Commands",
        "",
        "Use `vllm-sr` to configure and run the router, manage recipes, send requests, "
        "and evaluate routing. See [Installation](/docs/installation/) to install the CLI "
        "and [Local Docker deployment](/docs/installation/docker) to start a router.",
        "",
        "This reference is generated from the registered CLI commands. Command descriptions, "
        "arguments, options, and declared defaults match the source. An option without a "
        "declared default is omitted until supplied; commands may resolve it from configuration "
        "or the environment as described in their help. Repeatable options collect values. "
        "Run `vllm-sr COMMAND --help` for help from your installed version.",
        "",
        "## Command index",
        "",
        "| Command | Description |",
        "| --- | --- |",
    ]
    for context in contexts:
        summary = (
            (context.command.short_help or context.command.help or "")
            .strip()
            .split("\n", 1)[0]
        )
        lines.append(
            f"| [{code(context.command_path)}](#{anchor(context)}) | {cell(summary)} |"
        )
    for context in contexts:
        level = (
            "##" if context.parent is None or context.parent.parent is None else "###"
        )
        lines.extend(
            [
                "",
                f"{level} `{context.command_path}` {{#{anchor(context)}}}",
                "",
                "```text",
                context.get_usage(),
                "```",
            ]
        )
        description = help_blocks(context.command.help)
        if description:
            lines.extend(["", description])
        rows = parameters(context)
        if rows:
            lines.extend(["", "| Parameter | Description |", "| --- | --- |", *rows])
        if context.command.epilog:
            lines.extend(["", help_blocks(context.command.epilog)])
    return "\n".join(lines) + "\n"


def sync(output: Path, content: str, *, check: bool) -> bool:
    """Return whether content differs; check mode never creates or modifies files."""
    if output.is_file() and output.read_text(encoding="utf-8") == content:
        return False
    if not check:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(content, encoding="utf-8")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="fail on drift without writing"
    )
    parser.add_argument(
        "--output", type=Path, default=OUTPUT, help="reference Markdown path"
    )
    args = parser.parse_args(argv)
    content = render(load_cli())
    changed = sync(args.output, content, check=args.check)
    if args.check and changed:
        print(
            "CLI reference is missing or outdated; run make docs-cli.",
            file=sys.stderr,
        )
        return 1
    print(f"CLI reference {'checked' if args.check else 'generated'}: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
