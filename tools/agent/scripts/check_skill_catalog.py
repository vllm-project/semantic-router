#!/usr/bin/env python3
"""Check the shipped skill inventory and its public-source mapping."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from urllib.parse import unquote, urlsplit

import yaml

ROOT = Path(__file__).resolve().parents[3]
CATALOG = Path("tools/agent/skills/catalog.yaml")
README = Path("tools/agent/skills/README.md")
SKILL_ROOTS = (
    Path("tools/agent/skills"),
    Path("src/agent-harness/claude-code/skills"),
)
PUBLIC_SOURCE = Path("tools/agent/skills/vllm-sr-agent-operations/SKILL.md")
PUBLIC_COPY = Path("website/static/install/agent/vllm-sr/SKILL.md")
PUBLIC_URL = "https://vllm-sr.ai/install/agent/vllm-sr/SKILL.md"
FIELDS = (
    "id",
    "name",
    "audience",
    "source",
    "distribution",
    "owner",
    "trigger",
    "prerequisites",
    "inputs",
    "outputs",
    "handoff",
    "support",
)
LINK = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
NAME = re.compile(r"\A---\n(?:[^\n]*\n)*?name:\s*([^\n]+)\n", re.MULTILINE)


def check(root: Path = ROOT) -> list[str]:
    """Return actionable errors; leave the repository unchanged."""
    errors: list[str] = []
    data = yaml.safe_load((root / CATALOG).read_text())
    skills = data.get("skills") if isinstance(data, dict) else None
    if not isinstance(skills, list):
        return [f"{CATALOG}: expected a skills list"]

    ids: set[str] = set()
    sources: set[Path] = set()
    copies: set[Path] = set()
    for index, entry in enumerate(skills, 1):
        label = f"{CATALOG}: skills[{index}]"
        if not isinstance(entry, dict):
            errors.append(f"{label}: expected a mapping")
            continue
        missing = [
            key
            for key in FIELDS
            if not isinstance(entry.get(key), str) or not entry[key].strip()
        ]
        if missing:
            errors.append(f"{label}: missing {', '.join(missing)}")
            continue
        skill_id = entry["id"]
        if skill_id in ids:
            errors.append(f"{label}: duplicate id {skill_id}")
        ids.add(skill_id)

        source = Path(entry["source"])
        if (
            source.is_absolute()
            or ".." in source.parts
            or not source.as_posix().endswith("/SKILL.md")
        ):
            errors.append(f"{label}: source must be a repository SKILL.md path")
            continue
        if source in sources:
            errors.append(f"{label}: duplicate source {source}")
        sources.add(source)
        source_file = root / source
        if not source_file.is_file():
            errors.append(f"{label}: missing source {source}")
        else:
            match = NAME.search(source_file.read_text())
            if not match or match.group(1).strip() != entry["name"]:
                errors.append(f"{label}: frontmatter name differs from {entry['name']}")

        distribution = entry["distribution"]
        if distribution not in {"repository", "public-website", "claude-code-plugin"}:
            errors.append(f"{label}: unknown distribution {distribution}")
        if distribution == "public-website":
            copy = Path(entry.get("published", ""))
            if copy.is_absolute() or ".." in copy.parts or not (root / copy).is_file():
                errors.append(f"{label}: missing public copy {copy}")
            if copy in copies:
                errors.append(f"{label}: duplicate public copy {copy}")
            copies.add(copy)
            if (source, copy, entry.get("public_url")) != (
                PUBLIC_SOURCE,
                PUBLIC_COPY,
                PUBLIC_URL,
            ):
                errors.append(
                    f"{label}: public source, copy, and URL must match the sync contract"
                )
        elif "published" in entry or "public_url" in entry:
            errors.append(
                f"{label}: only a public-website skill may declare a public copy"
            )
        if distribution == "claude-code-plugin" and not source.is_relative_to(
            SKILL_ROOTS[1]
        ):
            errors.append(f"{label}: plugin skill must live in its plugin directory")

    actual = {
        path.relative_to(root)
        for skill_root in SKILL_ROOTS
        for path in (root / skill_root).rglob("SKILL.md")
    }
    for source in sorted(actual - sources):
        errors.append(f"{CATALOG}: unlisted skill {source}")
    for source in sorted(sources - actual):
        errors.append(f"{CATALOG}: listed skill outside inventory {source}")
    if copies != {PUBLIC_COPY}:
        errors.append(f"{CATALOG}: expected exactly the operations public copy")

    readme = (root / README).read_text()
    linked_sources: set[Path] = set()
    for target in LINK.findall(readme):
        parsed = urlsplit(target)
        if parsed.scheme or parsed.netloc or target.startswith("#"):
            continue
        path = (root / README.parent / unquote(parsed.path)).resolve()
        if not path.is_file() or not path.is_relative_to(root.resolve()):
            errors.append(f"{README}: broken local link {target}")
        elif path.name == "SKILL.md":
            linked_sources.add(path.relative_to(root.resolve()))
    for source in sorted(sources - linked_sources):
        errors.append(f"{README}: missing discoverable entry for {source}")
    return errors


def main() -> int:
    errors = check()
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print("Skill catalog: checked shipped sources, public mapping, and local links")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
