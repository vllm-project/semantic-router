#!/usr/bin/env python3
"""Publish standalone agent skills from their repository-authored sources."""

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
SERVE_SOURCE = ROOT / "tools/agent/skills/vllm-sr-serve"
SERVE_DESTINATION = ROOT / "website/static/install/agent/vllm-sr-serve"
SERVE_PUBLIC_ORIGIN = "https://vllm-sr.ai/install/agent/vllm-sr-serve/"
DECISION_SOURCE = ROOT / "tools/agent/skills/vllm-sr-decision-model"
DECISION_DESTINATION = ROOT / "website/static/install/agent/vllm-sr-decision-model"
DECISION_PUBLIC_ORIGIN = "https://vllm-sr.ai/install/agent/vllm-sr-decision-model/"
CONFIGURE_SOURCE = ROOT / "tools/agent/skills/vllm-sr-configure-recipe"
CONFIGURE_DESTINATION = ROOT / "website/static/install/agent/vllm-sr-configure-recipe"
CONFIGURE_PUBLIC_ORIGIN = "https://vllm-sr.ai/install/agent/vllm-sr-configure-recipe/"
DIAGNOSE_SOURCE = ROOT / "tools/agent/skills/vllm-sr-diagnose-routing"
DIAGNOSE_DESTINATION = ROOT / "website/static/install/agent/vllm-sr-diagnose-routing"
DIAGNOSE_PUBLIC_ORIGIN = "https://vllm-sr.ai/install/agent/vllm-sr-diagnose-routing/"
BENCHMARK_SOURCE = ROOT / "tools/agent/skills/vllm-sr-benchmark"
BENCHMARK_DESTINATION = ROOT / "website/static/install/agent/vllm-sr-benchmark"
BENCHMARK_PUBLIC_ORIGIN = "https://vllm-sr.ai/install/agent/vllm-sr-benchmark/"
COMPARE_SOURCE = ROOT / "tools/agent/skills/vllm-sr-compare-configurations"
COMPARE_DESTINATION = (
    ROOT / "website/static/install/agent/vllm-sr-compare-configurations"
)
COMPARE_PUBLIC_ORIGIN = (
    "https://vllm-sr.ai/install/agent/vllm-sr-compare-configurations/"
)
PUBLICATIONS = (
    (SOURCE, DESTINATION, PUBLIC_ORIGIN, "vllm-sr-agent-operations", "vllm-sr"),
    (
        SERVE_SOURCE,
        SERVE_DESTINATION,
        SERVE_PUBLIC_ORIGIN,
        "vllm-sr-serve",
        "vllm-sr-serve",
    ),
    (
        DECISION_SOURCE,
        DECISION_DESTINATION,
        DECISION_PUBLIC_ORIGIN,
        "vllm-sr-decision-model",
        "vllm-sr-decision-model",
    ),
    (
        CONFIGURE_SOURCE,
        CONFIGURE_DESTINATION,
        CONFIGURE_PUBLIC_ORIGIN,
        "vllm-sr-configure-recipe",
        "vllm-sr-configure-recipe",
    ),
    (
        DIAGNOSE_SOURCE,
        DIAGNOSE_DESTINATION,
        DIAGNOSE_PUBLIC_ORIGIN,
        "vllm-sr-diagnose-routing",
        "vllm-sr-diagnose-routing",
    ),
    (
        BENCHMARK_SOURCE,
        BENCHMARK_DESTINATION,
        BENCHMARK_PUBLIC_ORIGIN,
        "vllm-sr-benchmark",
        "vllm-sr-benchmark",
    ),
    (
        COMPARE_SOURCE,
        COMPARE_DESTINATION,
        COMPARE_PUBLIC_ORIGIN,
        "vllm-sr-compare-configurations",
        "vllm-sr-compare-configurations",
    ),
)
MARKDOWN_LINK = re.compile(
    r"(!?\[[^\]\n]*\]\([ \t]*)(<[^>\n]+>|[^\s)]+)"
    r"((?:[ \t]+(?:\"[^\"\n]*\"|'[^'\n]*'|\([^\)\n]*\)))?[ \t]*\))"
)
REFERENCE_LINK = re.compile(
    r"(^[ \t]{0,3}\[[^\]\n]+\]:[ \t]*)(<?[^\s>]+>?)(.*$)", re.MULTILINE
)


def absolute_target(
    target: str,
    path: Path,
    documents: dict[Path, str],
    public_origin: str = PUBLIC_ORIGIN,
) -> str:
    """Resolve links without relying on where a downloaded skill is installed."""
    parsed = urlsplit(target)
    if parsed.scheme:
        published = target
    else:
        published = urljoin(urljoin(public_origin, path.as_posix()), target)

    # Validate same-site skill references, including already-absolute links and
    # fragments. Other site paths and external URLs retain their destinations.
    published_path = urlsplit(published)
    origin = urlsplit(public_origin)
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


def published_files(
    source: Path,
    *,
    public_origin: str = PUBLIC_ORIGIN,
    source_name: str = "vllm-sr-agent-operations",
    public_name: str = "vllm-sr",
) -> dict[Path, str]:
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
                rf"\A---\nname: {re.escape(source_name)}\n",
                f"---\nname: {public_name}\n",
                content,
                count=1,
            )
            if count != 1:
                raise ValueError(f"canonical skill must declare name: {source_name}")

        def rewrite_link(match: re.Match[str], path: Path = path) -> str:
            target = match[2]
            wrapped = target.startswith("<") and target.endswith(">")
            published = absolute_target(
                target[1:-1] if wrapped else target,
                path,
                documents,
                public_origin,
            )
            if wrapped:
                published = f"<{published}>"
            return f"{match[1]}{published}{match[3]}"

        content = MARKDOWN_LINK.sub(rewrite_link, content)
        expected[path] = REFERENCE_LINK.sub(rewrite_link, content)
    return expected


def sync(
    source: Path,
    destination: Path,
    *,
    check: bool,
    public_origin: str = PUBLIC_ORIGIN,
    source_name: str = "vllm-sr-agent-operations",
    public_name: str = "vllm-sr",
) -> list[str]:
    """Return drift; check mode never mutates the public tree."""
    expected = published_files(
        source,
        public_origin=public_origin,
        source_name=source_name,
        public_name=public_name,
    )
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
    failed = False
    for source, destination, origin, source_name, public_name in PUBLICATIONS:
        try:
            drift = sync(
                source,
                destination,
                check=args.check,
                public_origin=origin,
                source_name=source_name,
                public_name=public_name,
            )
        except (OSError, ValueError) as exc:
            print(f"Public skill {source.relative_to(ROOT)}: {exc}", file=sys.stderr)
            failed = True
            continue
        if args.check and drift:
            print(
                f"Public skill drift in {destination.relative_to(ROOT)}; run make agent-skill-sync:",
                file=sys.stderr,
            )
            print("\n".join(drift), file=sys.stderr)
            failed = True
        else:
            print(
                f"Public skill {'checked' if args.check else 'synchronized'} from {source.relative_to(ROOT)}"
            )
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
