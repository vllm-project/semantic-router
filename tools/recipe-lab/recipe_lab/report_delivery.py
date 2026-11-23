"""Publish portable Markdown reports from explicitly selected local evidence."""

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

from .user_report import render_report


def _publish(path, data):
    """Publish complete bytes once. Concurrent identical writes are idempotent."""
    path.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    stream = tempfile.NamedTemporaryFile(dir=path.parent, delete=False)
    staged = Path(stream.name)
    try:
        with stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            if os.name == "nt":
                # Windows rename is atomic and refuses an existing destination,
                # including on volumes that do not implement hard links.
                staged.rename(path)
            else:
                os.link(staged, path)
        except FileExistsError:
            if path.is_symlink() or path.read_bytes() != data:
                raise ValueError(f"refusing to replace existing output: {path}")
    finally:
        staged.unlink(missing_ok=True)


def write_report(path, evidence, *, background=None):
    if path.suffix.lower() != ".md":
        raise ValueError("report output must have a .md extension")
    data = render_report(evidence, background=background).encode("utf-8")
    _publish(path, data)
    return str(path.resolve())


def snapshot(workspace, record):
    """Keep each result and its report together, even for concurrent studies."""
    evidence = (
        json.dumps(
            record, sort_keys=True, allow_nan=False, ensure_ascii=False, indent=2
        ).encode("utf-8")
        + b"\n"
    )
    data = render_report([record]).encode("utf-8")
    # A newer renderer may describe identical evidence differently. Keep both
    # snapshots so upgrading the tool never overwrites an earlier handoff.
    fingerprint = hashlib.sha256(evidence + b"\0" + data).hexdigest()
    folder = workspace / "reports" / fingerprint
    _publish(folder / "evidence.json", evidence)
    path = folder / "report.md"
    _publish(path, data)
    return str(path.resolve())


def main(argv=None):
    parser = argparse.ArgumentParser(prog="recipe-lab report", description=__doc__)
    parser.add_argument(
        "--evidence",
        action="append",
        required=True,
        type=Path,
        help="one saved JSON manifest or receipt; repeat in workflow order",
    )
    parser.add_argument(
        "--out", required=True, type=Path, help="destination Markdown file"
    )
    parser.add_argument("--background", help="user-supplied context for this handoff")
    args = parser.parse_args(argv)
    try:
        evidence = [json.loads(path.read_bytes()) for path in args.evidence]
        report = write_report(args.out, evidence, background=args.background)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"recipe-lab report: {exc}\n")
    print(json.dumps({"report": report, "evidence_count": len(evidence)}))
    return 0
