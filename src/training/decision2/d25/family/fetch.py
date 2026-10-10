"""Copy a directory tree from a ``d25.family.k8s xfer`` server to node disk (resumable).

    python -m d25.family.fetch --url http://<pod ip>:8080/M2-v5/ --out /data/d25/omni/family/data/v1/M2-v5

Walks the server's directory listings, skips files whose size already matches, downloads to a
``.part`` file and renames it, then writes ``FETCHED`` with the file list and sizes.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import urllib.parse
import urllib.request
from pathlib import Path


def listing(url: str) -> list[str]:
    with urllib.request.urlopen(url, timeout=60) as response:
        html = response.read().decode("utf-8", "replace")
    names = [urllib.parse.unquote(h) for h in re.findall(r'href="([^"?#]+)"', html)]
    return [n for n in names if not n.startswith(("/", "..", "http"))]


def walk(url: str, prefix: str = "") -> list[str]:
    files: list[str] = []
    for name in listing(url):
        if name.endswith("/"):
            files += walk(url + urllib.parse.quote(name), prefix + name)
        else:
            files.append(prefix + name)
    return files


def fetch(url: str, out: Path) -> dict[str, int]:
    url = url if url.endswith("/") else url + "/"
    sizes: dict[str, int] = {}
    for rel in walk(url):
        source = url + urllib.parse.quote(rel)
        target = out / rel
        with urllib.request.urlopen(source, timeout=600) as response:
            size = int(response.headers["Content-Length"])
            if target.exists() and target.stat().st_size == size:
                sizes[rel] = size
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            part = target.with_name(target.name + ".part")
            with part.open("wb") as stream:
                shutil.copyfileobj(response, stream, 1 << 22)
        if part.stat().st_size != size:
            raise SystemExit(f"{rel}: got {part.stat().st_size} of {size} bytes")
        part.replace(target)
        sizes[rel] = size
        print(json.dumps({"fetched": rel, "bytes": size}), flush=True)
    (out / "FETCHED").write_text(json.dumps({"url": url, "files": sizes}, indent=1))
    return sizes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    sizes = fetch(args.url, Path(args.out))
    print(json.dumps({"files": len(sizes), "bytes": sum(sizes.values())}))


if __name__ == "__main__":
    main()
