"""Copy a directory tree listed in its SHA256SUMS from another node's read-only HTTP server.

    python -m d25.omni.k8s.pull --url http://<pod-ip>:8080/vision-0.3.1 --dest /data/d25/omni/suite/vision-0.3.1

Files already present with the right digest are skipped, every download is verified, and a
``PULLED`` marker is written last, so the step can be re-run after a restart.
"""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(url: str, dest: Path, expected: str) -> int:
    if dest.exists() and sha256(dest) == expected:
        return 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    for attempt in range(4):
        with urllib.request.urlopen(url, timeout=120) as response, open(
            tmp, "wb"
        ) as handle:
            while block := response.read(1 << 20):
                handle.write(block)
        if sha256(tmp) == expected:
            tmp.replace(dest)
            return dest.stat().st_size
    raise RuntimeError(f"digest mismatch after retries: {url}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", required=True)
    parser.add_argument("--dest", required=True)
    parser.add_argument("--threads", type=int, default=32)
    args = parser.parse_args()
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(f"{args.url}/SHA256SUMS", timeout=120) as response:
        sums = response.read()
    entries = [
        line.split(None, 1) for line in sums.decode().splitlines() if line.strip()
    ]
    with ThreadPoolExecutor(args.threads) as pool:
        sizes = list(
            pool.map(
                lambda e: fetch(
                    f"{args.url}/{e[1].strip()}", dest / e[1].strip(), e[0]
                ),
                entries,
            )
        )
    (dest / "SHA256SUMS").write_bytes(sums)
    (dest / "PULLED").write_text(
        f"{args.url} files {len(entries)} bytes {sum(sizes)}\n"
    )
    print(
        f"pulled {len(entries)} files, {sum(sizes)} new bytes into {dest}", flush=True
    )


if __name__ == "__main__":
    main()
