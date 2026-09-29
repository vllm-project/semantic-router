#!/usr/bin/env python3
"""Publish the operations skill from its single repository-authored source."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "tools/agent/skills/vllm-sr-agent-operations"
DESTINATION = ROOT / "website/static/install/agent/vllm-sr"
PUBLIC_ORIGIN = "https://vllm-sr.ai/install/agent/vllm-sr/"
MARKDOWN_LINK = re.compile(
    r"(!?\[[^\]\n]*\]\([ \t]*)(<[^>\n]+>|[^\s)]+)"
    r"((?:[ \t]+(?:\"[^\"\n]*\"|'[^'\n]*'|\([^\)\n]*\)))?[ \t]*\))"
)
REFERENCE_LINK = re.compile(
    r"(^[ \t]{0,3}\[[^\]\n]+\]:[ \t]*)(<?[^\s>]+>?)(.*$)", re.MULTILINE
)


def absolute_target(target: str, path: Path, documents: dict[Path, str]) -> str:
    """Resolve links without relying on where a downloaded skill is installed."""
    parsed = urlsplit(target)
    if parsed.scheme:
        published = target
    else:
        published = urljoin(urljoin(PUBLIC_ORIGIN, path.as_posix()), target)

    # Validate same-site skill references, including already-absolute links and
    # fragments. Other site paths and external URLs retain their destinations.
    published_path = urlsplit(published)
    origin = urlsplit(PUBLIC_ORIGIN)
    if published_path.netloc == origin.netloc and published_path.path.startswith(
        origin.path
    ):
        relative = Path(unquote(published_path.path[len(origin.path) :]))
        if relative not in documents:
            raise ValueError(
                f"{path}: reference is not a published skill document: {target}"
            )
    elif not parsed.scheme and not parsed.netloc and not target.startswith("/"):
        raise ValueError(f"{path}: reference escapes skill source: {target}")
    return published


def published_files(source: Path) -> dict[Path, str]:
    """Render all authored Markdown, rejecting unresolved local references."""
    paths = [
        Path("SKILL.md"),
        *sorted(p.relative_to(source) for p in (source / "references").rglob("*.md")),
    ]
    documents = {path: (source / path).read_text() for path in paths}
    expected = {}
    for path, authored in documents.items():
        content = authored
        if path == Path("SKILL.md"):
            content, count = re.subn(
                r"\A---\nname: vllm-sr-agent-operations\n",
                "---\nname: vllm-sr\n",
                content,
                count=1,
            )
            if count != 1:
                raise ValueError(
                    "canonical skill must declare name: vllm-sr-agent-operations"
                )

        def rewrite_link(match: re.Match[str], path: Path = path) -> str:
            target = match[2]
            wrapped = target.startswith("<") and target.endswith(">")
            published = absolute_target(
                target[1:-1] if wrapped else target, path, documents
            )
            if wrapped:
                published = f"<{published}>"
            return f"{match[1]}{published}{match[3]}"

        content = MARKDOWN_LINK.sub(rewrite_link, content)
        expected[path] = REFERENCE_LINK.sub(rewrite_link, content)
    return expected


def sync(source: Path, destination: Path, *, check: bool) -> list[str]:
    """Return drift; check mode never mutates the public tree."""
    expected = published_files(source)
    actual = {path.relative_to(destination) for path in destination.rglob("*.md")}
    drift = []
    for path, content in expected.items():
        target = destination / path
        if not target.exists() or target.read_text() != content:
            drift.append(f"out of date: {path}")
            if not check:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content)
    for path in sorted(actual - expected.keys()):
        drift.append(f"unexpected generated document: {path}")
        if not check:
            (destination / path).unlink()
    return drift


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="fail on drift without writing"
    )
    args = parser.parse_args()
    try:
        drift = sync(SOURCE, DESTINATION, check=args.check)
    except (OSError, ValueError) as exc:
        print(f"Public skill: {exc}", file=sys.stderr)
        return 1
    if args.check and drift:
        print("Public skill drift; run make agent-skill-sync:", file=sys.stderr)
        print("\n".join(drift), file=sys.stderr)
        return 1
    print(
        f"Public skill {'checked' if args.check else 'synchronized'} from {SOURCE.relative_to(ROOT)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
