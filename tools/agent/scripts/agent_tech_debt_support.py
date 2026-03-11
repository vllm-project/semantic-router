#!/usr/bin/env python3
"""Shared helpers for indexed technical debt docs."""

from __future__ import annotations

import re
from pathlib import Path

from agent_support import REPO_ROOT

TECH_DEBT_DIR = REPO_ROOT / "docs" / "agent" / "tech-debt"
TECH_DEBT_REGISTER_DOC = REPO_ROOT / "docs" / "agent" / "tech-debt-register.md"
TECH_DEBT_FILENAME_PATTERN = re.compile(r"^td-(\d{3})-[a-z0-9-]+\.md$")
TECH_DEBT_HEADING_PREFIX = "# TD"
TECH_DEBT_ENTRY_REQUIRED_SECTIONS = [
    "## Status",
    "## Scope",
    "## Summary",
    "## Evidence",
    "## Why It Matters",
    "## Desired End State",
    "## Exit Criteria",
]


def collect_tech_debt_doc_paths() -> list[Path]:
    if not TECH_DEBT_DIR.exists():
        return []
    return sorted(
        path for path in TECH_DEBT_DIR.glob("*.md") if path.name != "README.md"
    )


def collect_tech_debt_entries() -> list[dict[str, str | Path]]:
    return [parse_tech_debt_entry(path) for path in collect_tech_debt_doc_paths()]


def collect_open_tech_debt_items() -> list[str]:
    items: list[str] = []
    for entry in collect_tech_debt_entries():
        status = str(entry.get("status", "")).strip().lower()
        if status != "open":
            continue
        item_id = str(entry.get("id", "")).strip()
        title = str(entry.get("title", "")).strip()
        if item_id and title:
            items.append(f"{item_id}: {title}")
    return items


def parse_tech_debt_entry(path: Path) -> dict[str, str | Path]:
    text = path.read_text(encoding="utf-8")
    item_id, title = parse_tech_debt_heading(text)
    return {
        "path": path,
        "text": text,
        "id": item_id,
        "title": title,
        "status": first_markdown_section_value(text, "## Status"),
        "scope": first_markdown_section_value(text, "## Scope"),
    }


def first_markdown_section_value(text: str, heading: str) -> str:
    for line in read_markdown_section_lines(text, heading):
        stripped = line.strip()
        if stripped:
            return stripped.removeprefix("- ").strip()
    return ""


def read_markdown_section_lines(text: str, heading: str) -> list[str]:
    lines = text.splitlines()
    collected: list[str] = []
    in_section = False
    for line in lines:
        if line.strip() == heading:
            in_section = True
            continue
        if in_section and line.startswith("## "):
            break
        if in_section and line.startswith("# "):
            break
        if in_section:
            collected.append(line)
    return collected


def validate_tech_debt_inventory_and_template(
    repo_manifest: dict, errors: list[str]
) -> None:
    texts = load_tech_debt_index_texts(errors)
    if texts is None:
        return

    register_text, debt_readme_text = texts
    validate_tech_debt_register_landing_page(register_text, errors)
    entries = collect_tech_debt_entries()
    if not entries:
        errors.append("docs/agent/tech-debt must contain at least one TD entry")
        return

    manifest_docs = set(repo_manifest.get("docs", []))
    validate_tech_debt_entries(entries, manifest_docs, debt_readme_text, errors)


def load_tech_debt_index_texts(errors: list[str]) -> tuple[str, str] | None:
    if not TECH_DEBT_REGISTER_DOC.exists():
        errors.append("Missing docs/agent/tech-debt-register.md")
        return None
    if not TECH_DEBT_DIR.exists():
        errors.append("Missing docs/agent/tech-debt directory")
        return None

    debt_readme = TECH_DEBT_DIR / "README.md"
    if not debt_readme.exists():
        errors.append("Missing docs/agent/tech-debt/README.md")
        return None

    return (
        TECH_DEBT_REGISTER_DOC.read_text(encoding="utf-8"),
        debt_readme.read_text(encoding="utf-8"),
    )


def validate_tech_debt_entries(
    entries: list[dict[str, str | Path]],
    manifest_docs: set[str],
    debt_readme_text: str,
    errors: list[str],
) -> None:
    entry_ids: list[str] = []
    filename_indices: list[str] = []
    for entry in entries:
        item_id, filename_index = validate_single_tech_debt_entry(
            entry, manifest_docs, debt_readme_text, errors
        )
        if filename_index:
            filename_indices.append(filename_index)
        if not item_id:
            continue
        entry_ids.append(item_id)

    duplicate_entry_ids = sorted(
        {item_id for item_id in entry_ids if entry_ids.count(item_id) > 1}
    )
    if duplicate_entry_ids:
        errors.append(
            "docs/agent/tech-debt has duplicate debt IDs: "
            + ", ".join(duplicate_entry_ids)
        )

    duplicate_filename_indices = sorted(
        {index for index in filename_indices if filename_indices.count(index) > 1}
    )
    if duplicate_filename_indices:
        errors.append(
            "docs/agent/tech-debt has duplicate filename indices: "
            + ", ".join(duplicate_filename_indices)
        )


def validate_single_tech_debt_entry(
    entry: dict[str, str | Path],
    manifest_docs: set[str],
    debt_readme_text: str,
    errors: list[str],
) -> tuple[str, str]:
    path = entry.get("path")
    if not isinstance(path, Path):
        return "", ""

    relative_path = path.relative_to(REPO_ROOT).as_posix()
    validate_tech_debt_entry_inventory(
        path, relative_path, manifest_docs, debt_readme_text, errors
    )

    text = str(entry.get("text", ""))
    validate_tech_debt_entry_template(relative_path, text, errors)

    item_id = str(entry.get("id", "")).strip()
    if not item_id:
        errors.append(
            f"Tech debt entry '{relative_path}' must use a heading like '# TD001: Title'"
        )
        return "", ""

    filename_index = parse_tech_debt_filename_index(path.name)
    expected_item_id = f"TD{filename_index}" if filename_index else ""
    if expected_item_id and item_id != expected_item_id:
        errors.append(
            f"Tech debt entry '{relative_path}' filename index '{filename_index}' must match heading ID '{item_id}'"
        )
    return item_id, filename_index


def validate_tech_debt_entry_inventory(
    path: Path,
    relative_path: str,
    manifest_docs: set[str],
    debt_readme_text: str,
    errors: list[str],
) -> None:
    if relative_path not in manifest_docs:
        errors.append(
            f"repo-manifest docs is missing tech debt entry '{relative_path}'"
        )

    if not TECH_DEBT_FILENAME_PATTERN.match(path.name):
        errors.append(
            f"Tech debt entry '{relative_path}' must use a slug like 'td-001-example.md'"
        )

    if f"({path.name})" not in debt_readme_text:
        errors.append(
            f"docs/agent/tech-debt/README.md must link to tech debt entry '{relative_path}'"
        )


def validate_tech_debt_entry_template(
    relative_path: str, text: str, errors: list[str]
) -> None:
    if not text.startswith(TECH_DEBT_HEADING_PREFIX):
        errors.append(f"Tech debt entry '{relative_path}' must start with '# TD'")
    for section in TECH_DEBT_ENTRY_REQUIRED_SECTIONS:
        if section not in text:
            errors.append(
                f"Tech debt entry '{relative_path}' is missing required section '{section}'"
            )


def validate_tech_debt_register_landing_page(
    register_text: str, errors: list[str]
) -> None:
    if "(tech-debt/README.md)" not in register_text:
        errors.append(
            "docs/agent/tech-debt-register.md must link to docs/agent/tech-debt/README.md"
        )
    if "### TD" in register_text:
        errors.append(
            "docs/agent/tech-debt-register.md must stay a landing page and must not duplicate per-item TD headings"
        )


def parse_tech_debt_heading(text: str) -> tuple[str, str]:
    first_line = text.splitlines()[0].strip() if text.splitlines() else ""
    if not first_line.startswith(TECH_DEBT_HEADING_PREFIX):
        return "", ""
    if ": " not in first_line:
        return "", ""
    item_id = first_line.removeprefix("# ").split(":", 1)[0].strip()
    title = first_line.split(": ", 1)[1].strip()
    return item_id, title


def parse_tech_debt_filename_index(filename: str) -> str:
    match = TECH_DEBT_FILENAME_PATTERN.match(filename)
    return match.group(1) if match else ""
