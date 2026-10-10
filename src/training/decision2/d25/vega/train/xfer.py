"""Cross-node export transfer over the pod network (sha256-verified, resumable).

Every training node runs one CPU-only server Job (``d25-vega-train-xfer-<NN>`` with a Service of the
same name, port 8080; ``train_job.py xfer --node NN``) that

* serves finished exports read-only: ``GET /manifest/<rel>`` lists files with size and sha256,
  ``GET /file/<rel>/<file>`` streams a file (HTTP Range for resume). ``<rel>`` is an export root:
  ``ckpt/<arm>/step-NNNNNN`` or ``ckpt/soups/<name>``. A root is served once it has
  ``decision_config.json``, no ``<root>.partial`` sibling and no file younger than two minutes;
* pulls what other nodes serve, in the background (no GPUs), for every spec in
  ``/data/d25/vega/xfer/wanted/*.json``: ``{"src_node": "06", "path": "ckpt/w1-g2-nc/step-005135"}``
  (optional ``dest``; default ``/data/d25/vega/imports/<src_node>/<arm>/<step>``). Status:
  ``/data/d25/vega/xfer/status/<spec>.json``.

``pull`` does the same from any pod (e.g. a runner tool step) and returns once the copy is verified:

    python -m d25.vega.train.xfer pull --src-node 06 --path ckpt/w1-g2-nc/step-005135 [--wait-h 6]

A verified copy carries ``XFER_VERIFIED.json`` (the source manifest). Files land in ``<dest>.partial/``
(interrupted files resume from ``*.part``) and the directory is renamed when every file matches.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(os.environ.get("D25_VEGA_ROOT", "/data/d25/vega"))
XFER = ROOT / "xfer"
PORT = 8080
SERVICE = "d25-vega-train-xfer-{node}.semantic-router.svc.cluster.local"
ROOT_PATTERNS = (
    re.compile(r"ckpt/[a-z0-9][a-z0-9.-]{0,62}/step-\d{6}"),
    re.compile(r"ckpt/soups/[a-z0-9][a-z0-9.-]{0,62}"),
)
STABLE_S = 120
CHUNK = 1 << 22
MARKER = "XFER_VERIFIED.json"


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(message: str) -> None:
    print(f"{now()} [xfer] {message}", flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(CHUNK):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------------------------------------
# Server side
# ---------------------------------------------------------------------------------------------


def export_root(rel: str) -> str | None:
    """The allowed export root a request path belongs to (or None)."""
    rel = rel.strip("/")
    if ".." in rel.split("/"):
        return None
    for pattern in ROOT_PATTERNS:
        match = pattern.match(rel)
        if match and (len(rel) == match.end() or rel[match.end()] == "/"):
            return match.group(0)
    return None


def root_files(root: Path) -> list[Path]:
    return sorted(
        p
        for p in root.rglob("*")
        if p.is_file() and not p.name.endswith((".part", ".lock"))
    )


class Manifests:
    """Per-root manifests (size + sha256 per file), cached in memory and under xfer/manifests/."""

    def __init__(self, base: Path):
        self.base = base
        self.locks: dict[str, threading.Lock] = {}
        self.guard = threading.Lock()
        (XFER / "manifests").mkdir(parents=True, exist_ok=True)

    def get(self, rel: str) -> tuple[int, dict]:
        root = self.base / rel
        if not (root / "decision_config.json").exists():
            return 404, {"error": "not found (no decision_config.json yet)"}
        if root.with_name(root.name + ".partial").exists():
            return 404, {"error": "export still being written"}
        files = root_files(root)
        if any(time.time() - p.stat().st_mtime < STABLE_S for p in files):
            return 404, {"error": "export modified within the last two minutes"}
        signature = [
            [str(p.relative_to(root)), p.stat().st_size, p.stat().st_mtime_ns]
            for p in files
        ]
        key = hashlib.sha256(rel.encode()).hexdigest()[:24]
        cache = XFER / "manifests" / f"{key}.json"
        with self.guard:
            lock = self.locks.setdefault(rel, threading.Lock())
        with lock:
            if cache.exists():
                cached = json.loads(cache.read_text())
                if cached.get("signature") == signature:
                    return 200, cached["manifest"]
            began = time.time()
            with ThreadPoolExecutor(max_workers=8) as pool:
                digests = list(pool.map(sha256_file, files))
            manifest = {
                "root": rel,
                "files": [
                    {"name": s[0], "size": s[1], "sha256": d}
                    for s, d in zip(signature, digests)
                ],
                "bytes": sum(s[1] for s in signature),
                "created": now(),
                "hash_seconds": round(time.time() - began, 1),
            }
            tmp = cache.with_suffix(".tmp")
            tmp.write_text(json.dumps({"signature": signature, "manifest": manifest}))
            os.replace(tmp, cache)
            log(
                f"manifest {rel}: {len(files)} files, {manifest['bytes'] / 1e9:.1f} GB, {manifest['hash_seconds']} s"
            )
            return 200, manifest


def make_handler(manifests: Manifests):
    base = manifests.base

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # noqa: N802 - quiet access log
            return

        def reply_json(self, code: int, body: dict) -> None:
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):  # noqa: N802
            try:
                path = urllib.request.unquote(self.path.split("?", 1)[0])
                if path == "/health":
                    return self.reply_json(200, {"ok": True, "time": now()})
                if path.startswith("/manifest/"):
                    rel = path[len("/manifest/") :].strip("/")
                    if export_root(rel) != rel:
                        return self.reply_json(403, {"error": "not an export root"})
                    code, body = manifests.get(rel)
                    return self.reply_json(code, body)
                if path.startswith("/file/"):
                    return self.send_file(path[len("/file/") :].strip("/"))
                return self.reply_json(404, {"error": "unknown endpoint"})
            except (BrokenPipeError, ConnectionResetError):
                return None
            except Exception as exc:  # noqa: BLE001
                log(f"error {self.path}: {exc}")
                try:
                    self.reply_json(500, {"error": str(exc)})
                except Exception:  # noqa: BLE001
                    pass

        def send_file(self, rel: str) -> None:
            root = export_root(rel)
            if root is None or root == rel:
                return self.reply_json(
                    403, {"error": "not a file inside an export root"}
                )
            target = (base / rel).resolve()
            if (
                not str(target).startswith(str((base / root).resolve()) + os.sep)
                or not target.is_file()
            ):
                return self.reply_json(404, {"error": "no such file"})
            size = target.stat().st_size
            start, end = 0, size - 1
            ranged = self.headers.get("Range")
            if ranged:
                match = re.fullmatch(r"bytes=(\d+)-(\d*)", ranged.strip())
                if not match:
                    return self.reply_json(416, {"error": "bad range"})
                start = int(match.group(1))
                end = int(match.group(2)) if match.group(2) else size - 1
                if start >= size or end < start:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return None
                end = min(end, size - 1)
            length = end - start + 1
            self.send_response(206 if ranged else 200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(length))
            if ranged:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            with target.open("rb") as handle:
                self.wfile.flush()
                self.connection.sendfile(handle, offset=start, count=length)
            return None

    return Handler


# ---------------------------------------------------------------------------------------------
# Client side
# ---------------------------------------------------------------------------------------------


def server_url(node: str) -> str:
    override = os.environ.get(f"D25_XFER_URL_{node}")
    return (
        override.rstrip("/")
        if override
        else f"http://{SERVICE.format(node=node)}:{PORT}"
    )


def default_dest(src_node: str, rel: str) -> Path:
    parts = rel.strip("/").split("/")
    return ROOT / "imports" / src_node / "/".join(parts[1:])


class NotReady(Exception):
    pass


def fetch_manifest(url: str, rel: str) -> dict:
    try:
        with urllib.request.urlopen(f"{url}/manifest/{rel}", timeout=900) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise NotReady(
                json.loads(exc.read() or b"{}").get("error", "not ready")
            ) from exc
        raise
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
        raise NotReady(f"source unreachable: {exc}") from exc


def download(
    url: str, rel: str, entry: dict, work: Path, verified: dict, lock: threading.Lock
) -> int:
    """One file into ``work`` (resume from ``.part``); returns bytes transferred."""
    target = work / entry["name"]
    if (
        verified.get(entry["name"]) == entry["sha256"]
        and target.exists()
        and target.stat().st_size == entry["size"]
    ):
        return 0
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    digest = hashlib.sha256()
    have = part.stat().st_size if part.exists() else 0
    if have > entry["size"]:
        part.unlink()
        have = 0
    if have:
        with part.open("rb") as handle:
            while block := handle.read(CHUNK):
                digest.update(block)
    moved = 0
    for attempt in range(8):
        if have >= entry["size"]:
            break
        request = urllib.request.Request(
            f"{url}/file/{rel}/{entry['name']}", headers={"Range": f"bytes={have}-"}
        )
        try:
            with urllib.request.urlopen(request, timeout=300) as response, part.open(
                "ab"
            ) as out:
                while block := response.read(CHUNK):
                    out.write(block)
                    digest.update(block)
                    have += len(block)
                    moved += len(block)
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
            log(
                f"{entry['name']}: transfer interrupted at {have} bytes ({exc}); retry {attempt + 1}"
            )
            time.sleep(min(60, 5 * (attempt + 1)))
    if have != entry["size"] or digest.hexdigest() != entry["sha256"]:
        part.unlink(missing_ok=True)
        raise IOError(
            f"{entry['name']}: size {have}/{entry['size']} or sha256 mismatch; partial removed"
        )
    os.replace(part, target)
    with lock:
        verified[entry["name"]] = entry["sha256"]
        tmp = work / ".verified.json.tmp"
        tmp.write_text(json.dumps(verified))
        os.replace(tmp, work / ".verified.json")
    return moved


def is_verified(dest: Path) -> bool:
    return (dest / MARKER).exists()


def pull(
    src_node: str,
    rel: str,
    dest: Path | None = None,
    wait_h: float = 0.0,
    parallel: int = 6,
) -> dict:
    """Copy export root ``rel`` from ``src_node`` into ``dest``; waits up to ``wait_h`` for the source."""
    rel = rel.strip("/")
    if export_root(rel) != rel:
        raise ValueError(
            f"{rel!r} is not an export root (ckpt/<arm>/step-NNNNNN or ckpt/soups/<name>)"
        )
    dest = Path(dest) if dest else default_dest(src_node, rel)
    if is_verified(dest):
        return {"dest": str(dest), "state": "verified", "bytes": 0}
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = server_url(src_node)
    with open(dest.with_name(dest.name + ".lock"), "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        if is_verified(dest):
            return {"dest": str(dest), "state": "verified", "bytes": 0}
        deadline = time.time() + wait_h * 3600
        while True:
            try:
                manifest = fetch_manifest(url, rel)
                break
            except NotReady as exc:
                if time.time() >= deadline:
                    raise
                log(f"waiting for {src_node}:{rel}: {exc}")
                time.sleep(120)
        work = dest.with_name(dest.name + ".partial")
        work.mkdir(parents=True, exist_ok=True)
        state_file = work / ".verified.json"
        verified = json.loads(state_file.read_text()) if state_file.exists() else {}
        began = time.time()
        guard = threading.Lock()
        entries = sorted(manifest["files"], key=lambda e: -e["size"])
        with ThreadPoolExecutor(max_workers=parallel) as pool:
            moved = sum(
                pool.map(
                    lambda e: download(url, rel, e, work, verified, guard), entries
                )
            )
        expected = {e["name"] for e in manifest["files"]}
        for path in root_files(work):
            if str(path.relative_to(work)) not in expected:
                path.unlink()
        (work / ".verified.json").unlink(missing_ok=True)
        record = {
            "source_node": src_node,
            "source": rel,
            "manifest": manifest,
            "verified": now(),
        }
        (work / MARKER).write_text(json.dumps(record, indent=1))
        for path in work.rglob("*"):
            if path.is_file():
                with path.open("rb") as handle:
                    os.fsync(handle.fileno())
        if dest.exists():
            shutil.rmtree(dest)
        os.replace(work, dest)
        seconds = time.time() - began
        summary = {
            "dest": str(dest),
            "state": "verified",
            "bytes": moved,
            "seconds": round(seconds, 1),
            "MBps": round(moved / max(seconds, 1e-6) / 1e6, 1),
            "files": len(entries),
        }
        log(f"pulled {src_node}:{rel} -> {dest}: {summary}")
        return summary


def watch_wanted(stop: threading.Event) -> None:
    """Background puller for ``xfer/wanted/*.json`` specs (one transfer at a time)."""
    wanted, status_dir = XFER / "wanted", XFER / "status"
    wanted.mkdir(parents=True, exist_ok=True)
    status_dir.mkdir(parents=True, exist_ok=True)
    while not stop.is_set():
        for spec_path in sorted(wanted.glob("*.json")):
            status_path = status_dir / spec_path.name
            try:
                spec = json.loads(spec_path.read_text())
                dest = (
                    Path(spec["dest"])
                    if spec.get("dest")
                    else default_dest(spec["src_node"], spec["path"])
                )
                if is_verified(dest):
                    if (
                        not status_path.exists()
                        or json.loads(status_path.read_text()).get("state")
                        != "verified"
                    ):
                        status_path.write_text(
                            json.dumps(
                                {"state": "verified", "dest": str(dest), "time": now()}
                            )
                        )
                    continue
                result = pull(
                    spec["src_node"],
                    spec["path"],
                    dest,
                    wait_h=0,
                    parallel=int(spec.get("parallel", 6)),
                )
                status_path.write_text(json.dumps({**result, "time": now()}))
            except NotReady as exc:
                status_path.write_text(
                    json.dumps({"state": "waiting", "reason": str(exc), "time": now()})
                )
            except Exception as exc:  # noqa: BLE001
                status_path.write_text(
                    json.dumps({"state": "error", "reason": str(exc), "time": now()})
                )
                log(f"wanted {spec_path.name}: {traceback.format_exc()[-600:]}")
        stop.wait(60)


def serve(port: int) -> None:
    manifests = Manifests(ROOT)
    stop = threading.Event()
    threading.Thread(target=watch_wanted, args=(stop,), daemon=True).start()
    server = ThreadingHTTPServer(("0.0.0.0", port), make_handler(manifests))
    server.daemon_threads = True
    log(f"serving export roots of {ROOT} on :{port}; watching {XFER / 'wanted'}")
    try:
        server.serve_forever(poll_interval=1.0)
    finally:
        stop.set()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--port", type=int, default=PORT)
    p = sub.add_parser("pull")
    p.add_argument("--src-node", required=True)
    p.add_argument(
        "--path", required=True, help="ckpt/<arm>/step-NNNNNN or ckpt/soups/<name>"
    )
    p.add_argument("--dest")
    p.add_argument("--wait-h", type=float, default=0.0)
    p.add_argument("--parallel", type=int, default=6)
    w = sub.add_parser(
        "want", help="queue a background pull on this node's xfer server"
    )
    w.add_argument("--src-node", required=True)
    w.add_argument("--path", required=True)
    w.add_argument("--dest")
    args = parser.parse_args()
    if args.cmd == "serve":
        serve(args.port)
        return 0
    if args.cmd == "want":
        (XFER / "wanted").mkdir(parents=True, exist_ok=True)
        name = re.sub(
            r"[^a-z0-9.-]+", "-", f"{args.src_node}-{args.path}".lower()
        ).strip("-")
        spec = {
            "src_node": args.src_node,
            "path": args.path,
            **({"dest": args.dest} if args.dest else {}),
        }
        (XFER / "wanted" / f"{name}.json").write_text(json.dumps(spec))
        print(json.dumps({"wanted": name, **spec}))
        return 0
    try:
        result = pull(
            args.src_node,
            args.path,
            Path(args.dest) if args.dest else None,
            args.wait_h,
            args.parallel,
        )
    except NotReady as exc:
        print(json.dumps({"state": "not-ready", "reason": str(exc)}))
        return 2
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
