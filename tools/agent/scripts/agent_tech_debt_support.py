#!/usr/bin/env python3
"""Shared helpers for indexed technical debt docs."""

from __future__ import annotations

import re
from pathlib import Path

from agent_support import REPO_ROOT

TECH_DEBT_DIR = REPO_ROOT / "docs" / "agent" / "tech-debt"
TECH_DEBT_REGISTER_DOC = REPO_ROOT / "docs" / "agent" / "tech-debt-register.md"
TECH_DEBT_ITEM_PATTERN = re.compile(r"^### (TD\d{3})\b", re.MULTILINE)
TECH_DEBT_FILENAME_PATTERN = re.compile(r"^(TD\d{3})-[a-z0-9-]+\.md$")
TECH_DEBT_HEADING_PATTERN = re.compile(r"^# (TD\d{3}): (.+)$", re.MULTILINE)
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
    heading_match = TECH_DEBT_HEADING_PATTERN.search(text)
    item_id = heading_match.group(1) if heading_match else ""
    title = heading_match.group(2).strip() if heading_match else ""
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
    entries = collect_tech_debt_entries()
    if not entries:
        errors.append("docs/agent/tech-debt must contain at least one TD entry")
        return

    manifest_docs = set(repo_manifest.get("docs", []))
    entry_ids, entry_map = validate_tech_debt_entries(
        entries, manifest_docs, debt_readme_text, register_text, errors
    )
    validate_tech_debt_register_index(register_text, entry_ids, entry_map, errors)


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
    register_text: str,
    errors: list[str],
) -> tuple[list[str], dict[str, dict[str, str | Path]]]:
    entry_ids: list[str] = []
    entry_map: dict[str, dict[str, str | Path]] = {}
    for entry in entries:
        item_id = validate_single_tech_debt_entry(
            entry, manifest_docs, debt_readme_text, register_text, errors
        )
        if not item_id:
            continue
        entry_ids.append(item_id)
        entry_map[item_id] = entry

    duplicate_entry_ids = sorted(
        {item_id for item_id in entry_ids if entry_ids.count(item_id) > 1}
    )
    if duplicate_entry_ids:
        errors.append(
            "docs/agent/tech-debt has duplicate debt IDs: "
            + ", ".join(duplicate_entry_ids)
        )
    return entry_ids, entry_map


def validate_single_tech_debt_entry(
    entry: dict[str, str | Path],
    manifest_docs: set[str],
    debt_readme_text: str,
    register_text: str,
    errors: list[str],
) -> str:
    path = entry.get("path")
    if not isinstance(path, Path):
        return ""

    relative_path = path.relative_to(REPO_ROOT).as_posix()
    validate_tech_debt_entry_inventory(
        path, relative_path, manifest_docs, debt_readme_text, register_text, errors
    )

    text = str(entry.get("text", ""))
    validate_tech_debt_entry_template(relative_path, text, errors)

    item_id = str(entry.get("id", "")).strip()
    if not item_id:
        errors.append(
            f"Tech debt entry '{relative_path}' must use a heading like '# TD001: Title'"
        )
        return ""

    if not path.name.startswith(f"{item_id}-"):
        errors.append(
            f"Tech debt entry '{relative_path}' filename must start with '{item_id}-'"
        )
    return item_id


def validate_tech_debt_entry_inventory(
    path: Path,
    relative_path: str,
    manifest_docs: set[str],
    debt_readme_text: str,
    register_text: str,
    errors: list[str],
) -> None:
    if relative_path not in manifest_docs:
        errors.append(
            f"repo-manifest docs is missing tech debt entry '{relative_path}'"
        )

    if not TECH_DEBT_FILENAME_PATTERN.match(path.name):
        errors.append(
            f"Tech debt entry '{relative_path}' must use a TD slug like 'TD001-example.md'"
        )

    if f"({path.name})" not in debt_readme_text:
        errors.append(
            f"docs/agent/tech-debt/README.md must link to tech debt entry '{relative_path}'"
        )
    if f"(tech-debt/{path.name})" not in register_text:
        errors.append(
            f"docs/agent/tech-debt-register.md must link to tech debt entry '{relative_path}'"
        )


def validate_tech_debt_entry_template(
    relative_path: str, text: str, errors: list[str]
) -> None:
    if not text.startswith("# TD"):
        errors.append(f"Tech debt entry '{relative_path}' must start with '# TD'")
    for section in TECH_DEBT_ENTRY_REQUIRED_SECTIONS:
        if section not in text:
            errors.append(
                f"Tech debt entry '{relative_path}' is missing required section '{section}'"
            )


def validate_tech_debt_register_index(
    register_text: str,
    entry_ids: list[str],
    entry_map: dict[str, dict[str, str | Path]],
    errors: list[str],
) -> None:
    register_ids = TECH_DEBT_ITEM_PATTERN.findall(register_text)
    if not register_ids:
        errors.append(
            "docs/agent/tech-debt-register.md must contain at least one TD item"
        )
        return

    duplicate_register_ids = sorted(
        {item_id for item_id in register_ids if register_ids.count(item_id) > 1}
    )
    if duplicate_register_ids:
        errors.append(
            "docs/agent/tech-debt-register.md has duplicate debt IDs: "
            + ", ".join(duplicate_register_ids)
        )

    summary_map = collect_register_summaries(register_text, errors)
    report_tech_debt_index_mismatches(entry_ids, register_ids, errors)
    for item_id, summary in summary_map.items():
        entry = entry_map.get(item_id)
        if entry is None:
            continue
        compare_register_and_entry(item_id, summary, entry, errors)


def report_tech_debt_index_mismatches(
    entry_ids: list[str], register_ids: list[str], errors: list[str]
) -> None:
    missing_from_register = sorted(set(entry_ids) - set(register_ids))
    if missing_from_register:
        errors.append(
            "docs/agent/tech-debt-register.md is missing summaries for: "
            + ", ".join(missing_from_register)
        )

    missing_entry_files = sorted(set(register_ids) - set(entry_ids))
    if missing_entry_files:
        errors.append(
            "docs/agent/tech-debt is missing entry files for: "
            + ", ".join(missing_entry_files)
        )


def collect_register_summaries(
    register_text: str, errors: list[str]
) -> dict[str, dict[str, str]]:
    summary_map: dict[str, dict[str, str]] = {}
    headings = re.findall(r"^### (TD\d{3}\b.*)$", register_text, flags=re.MULTILINE)
    chunks = re.split(r"^### TD\d{3}\b.*$", register_text, flags=re.MULTILINE)[1:]
    for heading, chunk in zip(headings, chunks, strict=False):
        for marker in ("- Status:", "- Scope:", "- Summary:", "- Entry:"):
            if marker not in chunk:
                errors.append(
                    f"docs/agent/tech-debt-register.md item '{heading}' is missing '{marker}'"
                )
        item_id = heading.split()[0]
        summary_map[item_id] = {
            "status": extract_register_marker_value(chunk, "Status"),
            "scope": extract_register_marker_value(chunk, "Scope"),
        }
    return summary_map


def compare_register_and_entry(
    item_id: str,
    summary: dict[str, str],
    entry: dict[str, str | Path],
    errors: list[str],
) -> None:
    register_status = summary.get("status", "").strip().lower()
    entry_status = str(entry.get("status", "")).strip().lower()
    if register_status and entry_status and register_status != entry_status:
        errors.append(
            f"docs/agent/tech-debt-register.md item '{item_id}' status does not match its entry file"
        )

    register_scope = summary.get("scope", "").strip().lower()
    entry_scope = str(entry.get("scope", "")).strip().lower()
    if register_scope and entry_scope and register_scope != entry_scope:
        errors.append(
            f"docs/agent/tech-debt-register.md item '{item_id}' scope does not match its entry file"
        )


def extract_register_marker_value(chunk: str, marker: str) -> str:
    match = re.search(rf"^- {marker}:\s*(.+)$", chunk, flags=re.MULTILINE)
    return match.group(1).strip() if match else ""
