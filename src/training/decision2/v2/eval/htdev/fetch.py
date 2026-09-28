"""Download the pinned HT-DEV source snapshots and verify every file against pins.json.

    python3 -m v2.eval.htdev.fetch fetch [--key KEY ...] [--dest /data/dev2/private/htdev/sources]
    python3 -m v2.eval.htdev.fetch pin --dest <scratch dir> --output <new pins.json> [--key KEY ...]

`fetch` (node A) writes `<dest>/<key>/<relpath>` for every pinned file and a
`<dest>/<key>/SNAPSHOT.json` (repository, full revision, file sha256s, licence evidence).
Hugging Face files come through `huggingface_hub` with `HF_HUB_CACHE` (default
/data/dev2/hf-cache); the token is read by huggingface_hub from its token file, never
from argv. GitHub, Zenodo and other URL files are fetched over HTTPS. A file whose
sha256 differs from its pin is refused and nothing is written for it; an existing
snapshot file is re-verified, never overwritten.

`fetch --reuse <dir>` first looks for each pinned file, by sha256, among the files of
`<dir>/<key>/` (or the isolation worker's key name, `ISO_KEYS`) and hard-links (else
copies) a match instead of downloading it.

`pin` fetches the public files anonymously (HF `resolve/<revision>` URLs) into a scratch
directory and writes a new pins file with the measured sha256s; it refuses to change a
sha256 that is already pinned.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

PINS = Path(__file__).with_name("pins.json")
DEFAULT_DEST = Path("/data/dev2/private/htdev/sources")
DEFAULT_HF_CACHE = "/data/dev2/hf-cache"
SNAPSHOT_SCHEMA = "dev2-htdev-snapshot/1"
TIMEOUT = 120
ISO_KEYS = {
    "claim_stance": "claimstance",
    "empathic_reactions": "empathic",
    "wic_tsv": "wictsv",
    "moral_stories": "moralstories",
}


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_pins(path: Path = PINS) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def file_url(spec: dict[str, Any], relative: str) -> str:
    quoted = urllib.parse.quote(relative)
    if spec["kind"] == "hf":
        return (
            f"https://huggingface.co/datasets/{spec['repo']}/resolve/"
            f"{spec['revision']}/{quoted}"
        )
    if spec["kind"] == "github":
        return (
            f"https://raw.githubusercontent.com/{spec['repo']}/"
            f"{spec['revision']}/{quoted}"
        )
    return spec["urls"][relative]


def download_url(url: str, target: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "dev2-htdev-fetch/1"})
    with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
        with target.open("wb") as stream:
            shutil.copyfileobj(response, stream, 8 << 20)


def download_hub(spec: dict[str, Any], relative: str, target: Path) -> None:
    os.environ.setdefault("HF_HUB_CACHE", DEFAULT_HF_CACHE)
    from huggingface_hub import hf_hub_download

    cached = hf_hub_download(
        repo_id=spec["repo"],
        filename=relative,
        repo_type="dataset",
        revision=spec["revision"],
    )
    shutil.copyfile(cached, target)


def reuse_index(key: str, dirs: list[Path]) -> dict[str, Path]:
    """sha256 -> path of every file under <dir>/<key>/ (or its isolation key name)."""
    index: dict[str, Path] = {}
    for base in dirs:
        for name in dict.fromkeys((key, ISO_KEYS.get(key, key))):
            root = base / name
            if not root.is_dir():
                continue
            for path in sorted(root.rglob("*")):
                if path.is_file() and not path.is_symlink():
                    index.setdefault(sha_file(path), path)
    return index


def place(
    spec: dict[str, Any],
    relative: str,
    root: Path,
    expected: str | None,
    hub: bool,
    reuse: dict[str, Path] | None = None,
) -> str:
    """Fetch one file into root/relative; returns its sha256 (refuses a mismatch)."""
    target = root / relative
    if target.exists():
        digest = sha_file(target)
        if expected and digest != expected:
            raise ValueError(f"{relative}: existing file sha256 differs from the pin")
        return digest
    target.parent.mkdir(parents=True, exist_ok=True)
    if expected and reuse and expected in reuse:
        try:
            os.link(reuse[expected], target)
        except OSError:
            shutil.copyfile(reuse[expected], target)
        if sha_file(target) != expected:
            target.unlink()
            raise ValueError(f"{relative}: reused file sha256 differs from the pin")
        return expected
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as handle:
        partial = Path(handle.name)
    try:
        if hub and spec["kind"] == "hf":
            download_hub(spec, relative, partial)
        else:
            download_url(file_url(spec, relative), partial)
        digest = sha_file(partial)
        if expected and digest != expected:
            raise ValueError(f"{relative}: downloaded sha256 differs from the pin")
        os.link(partial, target)
    finally:
        partial.unlink(missing_ok=True)
    return digest


def snapshot(key: str, spec: dict[str, Any], files: dict[str, str]) -> dict[str, Any]:
    return {
        "schema": SNAPSHOT_SCHEMA,
        "key": key,
        "kind": spec["kind"],
        "repo": spec["repo"],
        "revision": spec["revision"],
        "licence": spec["licence"],
        "licence_evidence": spec["licence_evidence"],
        "files": dict(sorted(files.items())),
    }


def fetch_source(
    key: str,
    spec: dict[str, Any],
    dest: Path,
    hub: bool,
    reuse: list[Path] | None = None,
) -> dict:
    if spec.get("unavailable"):
        raise ValueError(f"{key}: unavailable ({spec['unavailable']})")
    missing = [name for name, digest in spec["files"].items() if not digest]
    if missing:
        raise ValueError(f"{key}: {len(missing)} files have no pinned sha256")
    root = dest / key
    root.mkdir(parents=True, exist_ok=True)
    index = reuse_index(key, reuse) if reuse else None
    files = {
        relative: place(spec, relative, root, digest, hub, index)
        for relative, digest in sorted(spec["files"].items())
    }
    record = snapshot(key, spec, files)
    path = root / "SNAPSHOT.json"
    data = json.dumps(record, indent=1, sort_keys=True) + "\n"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != record:
            raise ValueError(f"{key}: SNAPSHOT.json exists with different content")
    else:
        path.write_text(data, encoding="utf-8")
    return record


def fetch(args: argparse.Namespace) -> int:
    pins = load_pins(args.pins)
    keys = args.key or sorted(pins["sources"])
    report = {}
    for key in keys:
        spec = pins["sources"][key]
        if spec.get("unavailable"):
            report[key] = {"status": "unavailable"}
            continue
        before = (
            {p for p in (args.dest / key).rglob("*") if p.is_file()}
            if (args.dest / key).is_dir()
            else set()
        )
        record = fetch_source(key, spec, args.dest, hub=True, reuse=args.reuse)
        linked = sum(
            (args.dest / key / r).stat().st_nlink > 1
            and (args.dest / key / r) not in before
            for r in record["files"]
        )
        report[key] = {"status": "ok", "files": len(record["files"]), "linked": linked}
    print(json.dumps(report, indent=1, sort_keys=True))
    return 0


def pin(args: argparse.Namespace) -> int:
    pins = load_pins(args.pins)
    keys = args.key or sorted(pins["sources"])
    for key in keys:
        spec = pins["sources"][key]
        if spec.get("unavailable"):
            continue
        root = args.dest / key
        for relative, expected in sorted(spec["files"].items()):
            spec["files"][relative] = place(spec, relative, root, expected, hub=False)
        record = snapshot(key, spec, spec["files"])
        (root / "SNAPSHOT.json").write_text(
            json.dumps(record, indent=1, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps({key: len(spec["files"])}))
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(pins, indent=1, sort_keys=True) + "\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    one = commands.add_parser("fetch")
    one.add_argument("--pins", type=Path, default=PINS)
    one.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    one.add_argument("--key", action="append")
    one.add_argument("--reuse", type=Path, action="append")
    two = commands.add_parser("pin")
    two.add_argument("--pins", type=Path, default=PINS)
    two.add_argument("--dest", type=Path, required=True)
    two.add_argument("--output", type=Path, required=True)
    two.add_argument("--key", action="append")
    args = parser.parse_args(argv)
    return {"fetch": fetch, "pin": pin}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
