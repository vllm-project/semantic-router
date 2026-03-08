#!/usr/bin/env python3
"""Validate public website docs reachability and locale structure."""

from __future__ import annotations

import re
import sys
from pathlib import Path

WEBSITE_ROOT = Path(__file__).resolve().parents[1]
DOCS_ROOT = WEBSITE_ROOT / "docs"
SIDEBAR_PATH = WEBSITE_ROOT / "sidebars.ts"
I18N_ROOT = WEBSITE_ROOT / "i18n"

# Locale-only sections must be narrow, documented exceptions rather than silent drift.
LOCALE_ONLY_TOP_LEVEL_ALLOWLIST: dict[str, set[str]] = {
    "zh-Hans": {"cookbook"},
}


def collect_doc_ids(root: Path) -> set[str]:
    doc_ids: set[str] = set()
    for path in root.rglob("*.md"):
        rel = path.relative_to(root).as_posix()
        if rel == "intro.md":
            doc_ids.add("intro")
        else:
            doc_ids.add(rel[:-3])
    return doc_ids


def collect_sidebar_doc_ids(path: Path) -> set[str]:
    text = path.read_text(encoding="utf-8")
    all_single_quoted = set(re.findall(r"'([^']+)'", text))
    known_top_levels = collect_top_level_sections(DOCS_ROOT) - {"__root__"}
    doc_like_refs: set[str] = set()
    for value in all_single_quoted:
        if value == "intro":
            doc_like_refs.add(value)
            continue
        top_level = value.split("/", 1)[0]
        if top_level in known_top_levels:
            doc_like_refs.add(value)
    return doc_like_refs


def collect_top_level_sections(root: Path) -> set[str]:
    sections: set[str] = set()
    for path in root.rglob("*.md"):
        rel_parts = path.relative_to(root).parts
        if not rel_parts:
            continue
        if rel_parts[0] == "intro.md":
            sections.add("__root__")
        else:
            sections.add(rel_parts[0])
    return sections


def validate_sidebar_reachability(errors: list[str]) -> None:
    docs = collect_doc_ids(DOCS_ROOT)
    sidebar_docs = collect_sidebar_doc_ids(SIDEBAR_PATH)

    missing_from_sidebar = sorted(docs - sidebar_docs)
    unknown_sidebar_refs = sorted(sidebar_docs - docs)

    if missing_from_sidebar:
        errors.append(
            "Public docs are missing from website/sidebars.ts: "
            + ", ".join(missing_from_sidebar)
        )

    if unknown_sidebar_refs:
        errors.append(
            "website/sidebars.ts references docs that do not exist: "
            + ", ".join(unknown_sidebar_refs)
        )


def validate_locale_structure(errors: list[str]) -> None:
    english_sections = collect_top_level_sections(DOCS_ROOT)

    for locale_dir in sorted(
        I18N_ROOT.glob("*/docusaurus-plugin-content-docs/current")
    ):
        locale = locale_dir.parts[-3]
        locale_sections = collect_top_level_sections(locale_dir)
        allowed_extras = LOCALE_ONLY_TOP_LEVEL_ALLOWLIST.get(locale, set())
        unexpected_extras = sorted(locale_sections - english_sections - allowed_extras)
        if unexpected_extras:
            errors.append(
                f"Locale '{locale}' has undocumented top-level docs sections: "
                + ", ".join(unexpected_extras)
            )


def main() -> int:
    errors: list[str] = []

    validate_sidebar_reachability(errors)
    validate_locale_structure(errors)

    if errors:
        print("Website docs structure check failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print("Website docs structure check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
