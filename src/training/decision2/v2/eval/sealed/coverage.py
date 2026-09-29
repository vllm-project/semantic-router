"""Build one node's overlap-scanner manifest of every training text file under roots.

    python3 -m v2.eval.sealed.coverage build --node NAME --root LABEL=PATH ... \
        [--exclude PATH_OR_GLOB ...] --work DIR --output <manifest.json> \
        --receipt <receipt.json> [--hot-minutes 90] [--workers 16] [--dry-run]
    python3 -m v2.eval.sealed.coverage hf-delta --repo REPO --repo-type dataset|model \
        --base REV --token-file PATH --output DIR --receipt <receipt.json> \
        [--head REV] [--endpoint URL]

Each root is walked without descending into directory symlinks; a symlink to a regular
file counts as that file (Hugging Face snapshots link into blobs). Never walked or
listed: an --exclude path and everything under it, a path matching an --exclude glob
(fnmatch on the absolute path, a directory also with a trailing `/`), directories named
.git, __pycache__, .cache, .locks or node_modules or with `triton` in the name, WORK,
and another root (listed under its own label). Nothing whose absolute path contains
`/private/sealed/` (also with symlinks resolved, or as a member's place in WORK) is
read, listed or copied, whatever the arguments. Model metadata files (config.json,
tokenizer files, trainer_state.json, *.index.json, ...) are skipped. By lowercased name
a file is

    scan        .jsonl .jsonl.gz .json .parquet .csv .tsv: listed in place; one
                modified within --hot-minutes of the build start is copied to WORK
                and the copy listed, so that a writer cannot change it under the scan
    text        .txt .md: copied to WORK with `.jsonl` appended, so that the scanner
                reads every line as text
    compressed  <inner>.gz .xz .bz2 .zst with <inner> .json .jsonl .csv .tsv .txt .md:
                decompressed to WORK under the inner name (text also gets `.jsonl`);
                zstd through the zstandard module, the standard library's
                compression.zstd (Python 3.14+) or the zstd binary, else unsupported
    archive     .tar .tar.gz .tgz .tar.xz .tar.bz2 .zip: members of the three classes
                above are extracted (compressed ones decompressed) to WORK under
                `<archive name>.d/`; absolute, `..`, link and device members are
                rejected, nested archives are counted, not extracted
    sniffed     a name without any dot (content-addressed blobs): its first 64 bytes
                decide. Parquet (PAR1) is copied as `<name>.parquet`; gzip, xz, bz2
                or zstd is decompressed and its head sniffed again (only JSON and text
                are kept); zip is an archive; safetensors (a little-endian u64 header
                length below the size, then `{`) counts as weights; a UTF-8 head
                starting with `{` or `[` becomes `<name>.json` (read as JSON lines
                when it is not one document), other UTF-8 without NUL becomes
                `<name>.txt.jsonl`; anything else counts as binary. Always copied.
                --dry-run opens no file and counts these as unsniffed
    other       not listed; counted by suffix

WORK mirrors the roots as WORK/<label>/<path under the root> (a derived name that is
already taken moves under `<source name>.d/`), with new mode-600 files in mode-700
directories; derived files keep their root's label. The sha256 and size of every listed
file come from a process pool (derived files are hashed as they are written). The
manifest (c1-corpora/1, labels and files sorted) and the receipt are new mode-600
files. The receipt holds counts only; its only paths are the roots, the excludes and
files that failed to read or decompress. --dry-run classifies by name (it opens no
file) and writes just the receipt. A full WORK disk stops the build.

`hf-delta` downloads, from a (private) Hugging Face repo, every file version that
first appears after --base up to --head (default main), intermediate versions that a
later commit overwrote included: it lists the commits after --base (refusing when
--base is not in the head's history), lists the tree of --base and of every later
commit, and fetches each (path, blob or LFS id) absent from the base tree from the
earliest commit holding it, to DIR/<commit[:12]>/<path> (mode 600, directories 700).
A download must match its LFS sha256 or git blob sha1, else it is removed and the
command exits 1. The token is sent only as an unredirected Authorization header (a
redirect to a CDN drops it), pagination links must stay on the endpoint's host, and
the token is never printed or written. The receipt (dev2-c1-hf-delta/1) holds the
commits and counts. Standard library only; the endpoint is --endpoint, $HF_ENDPOINT
or https://huggingface.co.
"""

from __future__ import annotations

import argparse
import bz2
import contextlib
import errno
import fnmatch
import functools
import gzip
import hashlib
import itertools
import json
import lzma
import multiprocessing
import os
import posixpath
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import IO, Any, NamedTuple

from v2.eval.sealed.overlap import file_kind

try:
    import zstandard
except ImportError:
    zstandard = None
try:
    from compression import zstd as stdzstd
except ImportError:
    stdzstd = None

SCHEMA = "dev2-c1-coverage/1"
BLOCK = 8 << 20
BATCH_BYTES = 64 << 20
BATCH_JOBS = 256
SEALED = "/private/sealed/"
METADATA = frozenset(
    {
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "added_tokens.json",
        "vocab.json",
        "merges.txt",
        "trainer_state.json",
        "adapter_config.json",
        "preprocessor_config.json",
    }
)
PRUNED = frozenset({".git", "__pycache__", ".cache", ".locks", "node_modules"})
TEXT = (".txt", ".md")
INNER = (".json", ".jsonl", ".csv", ".tsv", *TEXT)
CODECS = (".gz", ".xz", ".bz2", ".zst")
ARCHIVES = (".tar", ".tar.gz", ".tgz", ".tar.xz", ".tar.bz2", ".zip")
CLASSES = (
    "scan",
    "hot",
    "text",
    "compressed",
    "archive",
    "sniffed",
    "unsniffed",
    "metadata",
    "other",
    "listed",
)
HEAD = 64
MAGIC = (
    (b"PAR1", "parquet"),
    (b"\x1f\x8b", "gz"),
    (b"\xfd7zXZ\x00", "xz"),
    (b"BZh", "bz2"),
    (b"\x28\xb5\x2f\xfd", "zst"),
    (b"PK\x03\x04", "zip"),
)
SNIFFED = {"parquet": ".parquet", "json": ".json", "text": ".txt.jsonl"}
REJECTED = ("absolute", "parent", "empty", "link", "device", "duplicate")
LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+@-]*")
TOP_SUFFIXES = 50
HF_SCHEMA = "dev2-c1-hf-delta/1"
HF_ENDPOINT = "https://huggingface.co"
HF_TIMEOUT = 120
REPO = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*(/[A-Za-z0-9][A-Za-z0-9._-]*)?")
REVISION = re.compile(r"[0-9a-f]{4,40}")
NEXT = re.compile(r'<([^>]+)>\s*;\s*rel="?next"?')
_STATE: dict[str, Any] = {}


class Refusal(Exception):
    """hf-delta cannot go on; the message never holds the token."""


class Root(NamedTuple):
    label: str
    path: str
    real: str


class Scope(NamedTuple):
    """What a walk prunes; `paths` holds the exclude paths as given and resolved,
    `stops` maps the roots and WORK to their prune reason."""

    excludes: tuple[str, ...]
    paths: frozenset[str]
    globs: tuple[str, ...]
    stops: dict[str, str]

    def reason(self, forms: tuple[str, ...], name: str | None = None) -> str | None:
        """Why a directory (given its name) or a file is pruned; None keeps it."""
        for path in forms:
            if path in self.paths:
                return "exclude"
            if path in self.stops:
                return self.stops[path]
        if name is not None:
            forms += tuple(path + "/" for path in forms)
        match = fnmatch.fnmatchcase
        if any(match(path, glob) for path in forms for glob in self.globs):
            return "glob"
        if name is None:
            return None
        if name in PRUNED:
            return name
        return "triton" if "triton" in name.lower() else None


class Entry(NamedTuple):
    label: str
    path: str
    rel: str
    kind: str
    name: str
    size: int
    mtime: float


class Job(NamedTuple):
    kind: str
    label: str
    source: str
    dest: str
    size: int
    codec: str = ""


class Tally:
    """Counts of one build for the receipt."""

    def __init__(self, labels: Sequence[str]) -> None:
        self.classes = {label: {kind: [0, 0] for kind in CLASSES} for label in labels}
        self.other: dict[str, list[int]] = {}
        self.metadata: Counter = Counter()
        self.pruned: dict[str, Counter] = {"dirs": Counter(), "files": Counter()}
        self.counts: Counter = Counter()
        self.failures: dict[str, list[dict[str, Any]]] = {
            "decompression": [],
            "read": [],
        }

    def add(self, label: str, kind: str, size: int) -> None:
        cell = self.classes[label][kind]
        cell[0] += 1
        cell[1] += size


def classify(name: str) -> tuple[str, str]:
    """(class, name in WORK) of a file or archive member name."""
    lower = name.lower()
    if lower in METADATA or lower.endswith(".index.json"):
        return "metadata", name
    if file_kind(lower):
        return "scan", name
    if lower.endswith(TEXT):
        return "text", name + ".jsonl"
    if lower.endswith(ARCHIVES):
        return "archive", name
    stem, codec = os.path.splitext(name)
    if codec.lower() in CODECS and stem.lower().endswith(INNER):
        kind, inner = classify(stem)
        return ("metadata", name) if kind == "metadata" else ("compressed", inner)
    return "other", name


def sniff(head: bytes, size: int) -> str:
    """Kind of a file of `size` bytes from its first HEAD bytes: parquet, gz, xz, bz2,
    zst, zip, weights, json, text or binary."""
    for magic, kind in MAGIC:
        if head.startswith(magic):
            return kind
    header = int.from_bytes(head[:8], "little")
    if len(head) > 8 and 0 < header < size and head[8:9] == b"{":
        return "weights"
    try:
        head.decode("utf-8")
    except UnicodeDecodeError as error:
        if error.reason != "unexpected end of data" or len(head) >= size:
            return "binary"
    if head.lstrip()[:1] in (b"{", b"["):
        return "json"
    return "binary" if b"\0" in head else "text"


def sealed(*paths: str) -> bool:
    return any(SEALED in path + "/" for path in paths)


def zstd_method() -> str | None:
    if zstandard is not None:
        return "zstandard"
    if stdzstd is not None:
        return "compression.zstd"
    return "zstd" if shutil.which("zstd") else None


def _under(path: str, base: str) -> bool:
    return path == base or path.startswith(base.rstrip("/") + "/")


def _ancestors(path: str) -> Iterator[str]:
    while path != os.path.dirname(path):
        yield path
        yield path + "/"
        path = os.path.dirname(path)


def setup(
    specs: Sequence[str], excludes: Sequence[str], work: str
) -> tuple[list[Root], Scope, str]:
    """Roots, prune scope and WORK of a build; ValueError for a refused argument."""
    globs = tuple(item for item in excludes if any(char in item for char in "*?["))
    given = sorted({os.path.abspath(item) for item in excludes if item not in globs})
    paths = frozenset(given) | {os.path.realpath(path) for path in given}
    roots: list[Root] = []
    for spec in specs:
        label, separator, path = spec.partition("=")
        if not separator or not path:
            raise ValueError(f"root must be LABEL=PATH, got {spec!r}")
        if not LABEL.fullmatch(label):
            raise ValueError(f"root label {label!r} must match {LABEL.pattern}")
        if not os.path.isdir(path):
            raise ValueError(f"root {label}: {path} is not a directory")
        roots.append(Root(label, os.path.abspath(path), os.path.realpath(path)))
    if len({root.label for root in roots}) < len(roots):
        raise ValueError("duplicate root label")
    if len({root.real for root in roots}) < len(roots):
        raise ValueError("two roots are the same directory")
    for root in roots:
        for form in (root.path, root.real):
            if any(_under(form, path) for path in paths) or any(
                fnmatch.fnmatchcase(path, glob)
                for path in _ancestors(form)
                for glob in globs
            ):
                raise ValueError(f"root {root.label} is under an exclude")
    work = os.path.abspath(work)
    if os.path.lexists(work) and not (os.path.isdir(work) and not os.listdir(work)):
        raise ValueError(f"WORK {work} must be a new or empty directory")
    stops = {form: "root" for root in roots for form in (root.path, root.real)}
    stops.update(dict.fromkeys((work, os.path.realpath(work)), "work"))
    return roots, Scope(tuple(given), paths, globs, stops), work


def walk(root: Root, scope: Scope, tally: Tally) -> Iterator[Entry]:
    """The files of one root to list or derive; everything else is counted."""
    prefix = root.path.rstrip("/") + "/"
    real = root.real.rstrip("/") + "/"

    def forms(path: str) -> tuple[str, ...]:
        return (path,) if real == prefix else (path, real + path[len(prefix) :])

    def failed(error: OSError) -> None:
        tally.failures["read"].append(
            {"path": error.filename, "error": type(error).__name__}
        )

    for top, dirs, files in os.walk(root.path, onerror=failed, followlinks=False):
        kept = []
        for name in dirs:
            path = os.path.join(top, name)
            if os.path.islink(path):
                tally.counts["symlink_dirs"] += 1
                continue
            paths = forms(path)
            if sealed(*paths):
                tally.counts["sealed_guard"] += 1
            elif reason := scope.reason(paths, name):
                tally.pruned["dirs"][reason] += 1
            else:
                kept.append(name)
        dirs[:] = kept
        for name in files:
            path = os.path.join(top, name)
            paths = forms(path)
            if sealed(*paths):
                tally.counts["sealed_guard"] += 1
                continue
            reason = scope.reason(paths)
            if reason:
                tally.pruned["files"][reason] += 1
                continue
            try:
                info = os.lstat(path)
            except OSError as error:
                failed(error)
                continue
            if stat.S_ISLNK(info.st_mode):
                if sealed(os.path.realpath(path)):
                    tally.counts["sealed_guard"] += 1
                    continue
                try:
                    info = os.stat(path)
                except OSError:
                    tally.counts["broken_links"] += 1
                    continue
            if not stat.S_ISREG(info.st_mode):
                tally.counts["special_files"] += 1
                continue
            kind, derived = classify(name)
            if kind == "other" and "." not in name:
                kind = "sniffed"
            tally.add(root.label, kind, info.st_size)
            if kind == "metadata":
                tally.metadata[name.lower()] += 1
            elif kind == "other":
                suffix = os.path.splitext(name.lower())[1]
                cell = tally.other.setdefault(suffix, [0, 0])
                cell[0] += 1
                cell[1] += info.st_size
            else:
                rel = path[len(prefix) :]
                yield Entry(
                    root.label, path, rel, kind, derived, info.st_size, info.st_mtime
                )


def plan(
    entries: Sequence[Entry], work: str, cutoff: float, zstd: str | None, tally: Tally
) -> list[Job]:
    """Jobs for the sorted entries: hash in place, copy, decompress or extract."""
    jobs: list[Job] = []
    moved: list[tuple[Entry, str, str]] = []
    for entry in entries:
        if entry.kind == "archive":
            kind = "zip" if entry.path.lower().endswith(".zip") else "tar"
            dest = os.path.join(work, entry.label, entry.rel + ".d")
            jobs.append(Job(kind, entry.label, entry.path, dest, entry.size))
            continue
        codec = ""
        if entry.kind == "compressed":
            codec = os.path.splitext(entry.path)[1].lower()
            if codec == ".zst" and zstd is None:
                tally.counts["zst_unsupported"] += 1
                continue
        elif entry.kind == "scan":
            if entry.mtime < cutoff:
                jobs.append(Job("hash", entry.label, entry.path, "", entry.size))
                continue
            tally.add(entry.label, "hot", entry.size)
        place = os.path.join(work, entry.label, os.path.dirname(entry.rel), entry.name)
        moved.append((entry, place, codec))
    moved.sort(key=lambda item: item[0].kind == "sniffed")
    plain = {name for entry, place, _ in moved for name in _claims(entry, place)}
    taken: set[str] = set()
    for entry, place, codec in moved:
        if taken.intersection(_claims(entry, place)):
            place = os.path.join(work, entry.label, entry.rel + ".d", entry.name)
            if plain.intersection(_claims(entry, place)):
                raise ValueError(f"no free WORK name for {entry.path}")
            tally.counts["renamed"] += 1
        taken.update(_claims(entry, place))
        if entry.kind == "sniffed":
            kind = "sniff"
        else:
            kind = "decompress" if codec else "copy"
        jobs.append(Job(kind, entry.label, entry.path, place, entry.size, codec))
    return jobs


def _claims(entry: Entry, place: str) -> list[str]:
    """WORK names a moved entry may write: a sniffed one any of its derived names."""
    if entry.kind != "sniffed":
        return [place]
    return [place + suffix for suffix in (*SNIFFED.values(), ".part")]


def _init(zstd: str | None) -> None:
    _STATE["zstd"] = zstd


def _fatal(error: BaseException) -> bool:
    return isinstance(error, OSError) and error.errno in (errno.ENOSPC, errno.EDQUOT)


def _makedirs(path: str) -> None:
    """Create `path` and its missing parents with mode 700 (os.makedirs gives the
    parents the default mode)."""
    if not path or os.path.isdir(path):
        return
    _makedirs(os.path.dirname(path))
    try:
        os.mkdir(path, 0o700)
    except FileExistsError:
        if not os.path.isdir(path):
            raise


def _digest(path: str) -> tuple[str, int]:
    digest, size = hashlib.sha256(), 0
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(BLOCK), b""):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


@contextlib.contextmanager
def _zstd_process(raw: IO[bytes]) -> Iterator[IO[bytes]]:
    process = subprocess.Popen(
        ["zstd", "-dcq"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    errors: list[Exception] = []

    def feed() -> None:
        try:
            for block in iter(lambda: raw.read(BLOCK), b""):
                process.stdin.write(block)
        except BrokenPipeError:
            pass
        except Exception as error:
            errors.append(error)
        finally:
            with contextlib.suppress(BrokenPipeError):
                process.stdin.close()

    thread = threading.Thread(target=feed, daemon=True)
    thread.start()
    try:
        yield process.stdout
    finally:
        process.stdout.close()
        thread.join()
        code = process.wait()
    if errors:
        raise errors[0]
    if code:
        raise subprocess.CalledProcessError(code, "zstd -dcq")


def _decoder(codec: str, raw: IO[bytes]) -> Any:
    if not codec:
        return contextlib.nullcontext(raw)
    if codec == ".gz":
        return gzip.GzipFile(fileobj=raw, mode="rb")
    if codec == ".xz":
        return lzma.LZMAFile(raw)
    if codec == ".bz2":
        return bz2.BZ2File(raw)
    if _STATE["zstd"] == "zstandard":
        decompressor = zstandard.ZstdDecompressor()
        return decompressor.stream_reader(raw, read_across_frames=True)
    if _STATE["zstd"] == "compression.zstd":
        return stdzstd.ZstdFile(raw)
    return _zstd_process(raw)


def _write(
    dest: str, open_source: Callable[[], IO[bytes]], codec: str = ""
) -> tuple[str, int]:
    """Stream a source (decompressed by `codec`) into a new mode-600 file: (sha256,
    bytes). A failed write leaves no file behind."""
    _makedirs(os.path.dirname(dest))
    descriptor = os.open(dest, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    digest, size = hashlib.sha256(), 0
    try:
        with os.fdopen(descriptor, "wb") as out, open_source() as raw:
            with _decoder(codec, raw) as stream:
                for block in iter(lambda: stream.read(BLOCK), b""):
                    digest.update(block)
                    out.write(block)
                    size += len(block)
    except BaseException:
        os.unlink(dest)
        raise
    return digest.hexdigest(), size


def _member_path(name: str) -> tuple[str, str]:
    """(safe relative path, "") of an archive member name, or ("", rejection reason)."""
    if name.startswith(("/", "\\")):
        return "", "absolute"
    parts = [part for part in re.split(r"[/\\]", name) if part not in ("", ".")]
    if ".." in parts:
        return "", "parent"
    return ("/".join(parts), "") if parts else ("", "empty")


def _target(
    job: Job, name: str, size: int, reject: str, taken: set[str], counts: Counter
) -> tuple[str, str] | None:
    """(destination, codec) of an archive member to extract; None counts it instead."""
    path, reason = ("", reject) if reject else _member_path(name)
    if reason:
        counts[f"members_rejected_{reason}"] += 1
        return None
    kind, derived = classify(posixpath.basename(path))
    codec = os.path.splitext(path)[1].lower() if kind == "compressed" else ""
    dest = os.path.join(job.dest, posixpath.dirname(path), derived)
    if kind == "archive":
        counts["members_nested"] += 1
    elif kind == "metadata":
        counts["members_metadata"] += 1
    elif kind == "other":
        counts["members_other_files"] += 1
        counts["members_other_bytes"] += size
    elif codec == ".zst" and not _STATE["zstd"]:
        counts["zst_unsupported"] += 1
    elif sealed(dest):
        counts["sealed_guard"] += 1
    elif dest in taken:
        counts["members_rejected_duplicate"] += 1
    else:
        taken.add(dest)
        return dest, codec
    return None


def _extract(
    job: Job,
    files: list[tuple[str, str, int]],
    counts: Counter,
    failures: list[tuple[str, dict[str, Any]]],
) -> None:
    taken: set[str] = set()

    def extract(
        name: str,
        target: tuple[str, str] | None,
        open_source: Callable[[], IO[bytes]],
    ) -> None:
        if target is None:
            return
        dest, codec = target
        try:
            digest, size = _write(dest, open_source, codec)
        except Exception as error:
            if _fatal(error):
                raise
            error_type = type(error).__name__
            record = {"path": job.source, "member": name, "error": error_type}
            failures.append(("decompression", record))
            return
        files.append((dest, digest, size))
        counts["members_extracted_files"] += 1
        counts["members_extracted_bytes"] += size

    if job.kind == "zip":
        with zipfile.ZipFile(job.source) as archive:
            for info in archive.infolist():
                mode = stat.S_IFMT(info.external_attr >> 16)
                if info.is_dir() or mode == stat.S_IFDIR:
                    continue
                if mode == stat.S_IFLNK:
                    reject = "link"
                else:
                    reject = "" if mode in (0, stat.S_IFREG) else "device"
                name, size = info.filename, info.file_size
                target = _target(job, name, size, reject, taken, counts)
                extract(name, target, functools.partial(archive.open, info))
        return
    with tarfile.open(job.source, "r:*") as archive:
        while (member := archive.next()) is not None:
            # TarFile.next() keeps every member it has read.
            archive.members.clear()
            if member.isdir():
                continue
            if member.issym() or member.islnk():
                reject = "link"
            else:
                reject = "" if member.isreg() else "device"
            name, size = member.name, member.size
            target = _target(job, name, size, reject, taken, counts)
            extract(name, target, functools.partial(archive.extractfile, member))


def _unpack(
    job: Job, codec: str, files: list[tuple[str, str, int]], counts: Counter
) -> str:
    """Sniffed kind of a compressed file, keeping it decompressed when JSON or text."""
    if codec == "zst" and not _STATE["zstd"]:
        counts["zst_unsupported"] += 1
        return "unsupported.zst"
    opener = functools.partial(open, job.source, "rb")
    head = b""
    try:
        with opener() as raw, _decoder("." + codec, raw) as stream:
            head = stream.read(HEAD)
    except subprocess.CalledProcessError:
        # The zstd binary fails when its output is closed after the head.
        if len(head) < HEAD:
            raise
    inner = sniff(head, 1 << 64)
    if inner in ("json", "text", "weights"):
        part = job.dest + ".part"
        try:
            digest, size = _write(part, opener, "." + codec)
            with open(part, "rb") as stream:
                inner = sniff(stream.read(HEAD), size)
            if inner in ("json", "text"):
                dest = job.dest + SNIFFED[inner]
                os.link(part, dest)
                files.append((dest, digest, size))
        finally:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(part)
    return f"{inner}.{codec}"


def _sniff(
    job: Job,
    files: list[tuple[str, str, int]],
    counts: Counter,
    failures: list[tuple[str, dict[str, Any]]],
) -> None:
    with open(job.source, "rb") as stream:
        kind = sniff(stream.read(HEAD), job.size)
    try:
        if kind == "zip":
            _extract(
                job._replace(kind="zip", dest=job.dest + ".d"), files, counts, failures
            )
        elif kind not in (*SNIFFED, "weights", "binary"):
            kind = _unpack(job, kind, files, counts)
    except Exception as error:
        if _fatal(error):
            raise
        record = {"path": job.source, "error": type(error).__name__, "member": None}
        failures.append(("decompression", record))
        kind = f"broken.{kind}"
    if kind in SNIFFED:
        opener = functools.partial(open, job.source, "rb")
        files.append(
            (job.dest + SNIFFED[kind], *_write(job.dest + SNIFFED[kind], opener))
        )
    counts[f"sniffed:{kind}:files"] += 1
    counts[f"sniffed:{kind}:bytes"] += job.size


def _run(
    item: tuple[int, Job],
) -> tuple[int, list[tuple[str, str, int]], Counter, list[tuple[str, dict[str, Any]]]]:
    """(job number, listed (path, sha256, bytes), counts, (stage, failure) records)."""
    number, job = item
    files: list[tuple[str, str, int]] = []
    counts: Counter = Counter()
    failures: list[tuple[str, dict[str, Any]]] = []
    try:
        if job.kind == "hash":
            files.append((job.source, *_digest(job.source)))
        elif job.kind in ("copy", "decompress"):
            opener = functools.partial(open, job.source, "rb")
            files.append((job.dest, *_write(job.dest, opener, job.codec)))
        elif job.kind == "sniff":
            _sniff(job, files, counts, failures)
        else:
            _extract(job, files, counts, failures)
    except Exception as error:
        if _fatal(error):
            raise
        record: dict[str, Any] = {"path": job.source, "error": type(error).__name__}
        if job.kind == "sniff":
            counts["sniffed:unreadable:files"] += 1
            counts["sniffed:unreadable:bytes"] += job.size
        if job.kind in ("hash", "copy", "sniff"):
            failures.append(("read", record))
        else:
            failures.append(("decompression", {**record, "member": None}))
    return number, files, counts, failures


def _run_batch(batch: list[tuple[int, Job]]) -> list[tuple[Any, ...]]:
    return [_run(item) for item in batch]


def batches(
    jobs: Sequence[Job], size: int = BATCH_BYTES, count: int = BATCH_JOBS
) -> list[list[tuple[int, Job]]]:
    """Numbered jobs, largest first, grouped into pool tasks of about `size` bytes or
    `count` jobs (a large job is a task of its own)."""
    order = sorted(enumerate(jobs), key=lambda item: (-item[1].size, item[0]))
    grouped: list[list[tuple[int, Job]]] = []
    batch: list[tuple[int, Job]] = []
    total = 0
    for item in order:
        batch.append(item)
        total += item[1].size
        if total >= size or len(batch) >= count:
            grouped.append(batch)
            batch, total = [], 0
    return grouped + [batch] if batch else grouped


def execute(
    jobs: Sequence[Job],
    work: str,
    workers: int,
    zstd: str | None,
    tally: Tally,
    log: Any = None,
) -> list[tuple[str, str, str, int]]:
    """(label, path, sha256, bytes) of every listed file; the largest jobs go first."""
    _makedirs(work)
    tasks = batches(jobs)
    total = sum(job.size for job in jobs)
    listed: list[tuple[str, str, str, int]] = []
    done, seen, last = 0, 0, time.time()
    processes = min(workers, len(tasks))
    pool = None
    try:
        if processes > 1:
            pool = multiprocessing.get_context("fork").Pool(processes, _init, (zstd,))
            results: Iterator[Any] = pool.imap_unordered(_run_batch, tasks)
        else:
            _init(zstd)
            results = map(_run_batch, tasks)
        for number, files, counts, failures in itertools.chain.from_iterable(results):
            job = jobs[number]
            listed.extend((job.label, *item) for item in files)
            tally.counts.update(counts)
            for stage, record in failures:
                tally.failures[stage].append(record)
            done, seen = done + 1, seen + job.size
            if log and (time.time() - last > 30 or done == len(jobs)):
                last = time.time()
                log(f"{done}/{len(jobs)} jobs, {seen / 1e9:.1f}/{total / 1e9:.1f} GB")
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()
        _STATE.clear()
    return listed


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _cells(cells: dict[str, list[int]]) -> dict[str, dict[str, int]]:
    return {kind: {"files": cell[0], "bytes": cell[1]} for kind, cell in cells.items()}


def _order(record: dict[str, Any]) -> tuple[str, str]:
    return record["path"] or "", record.get("member") or ""


def _receipt(
    roots: Sequence[Root],
    scope: Scope,
    tally: Tally,
    run: dict[str, Any],
) -> dict[str, Any]:
    totals = {kind: [0, 0] for kind in CLASSES}
    for cells in tally.classes.values():
        for kind, (files, size) in cells.items():
            totals[kind][0] += files
            totals[kind][1] += size
    labels: dict[str, dict[str, Any]] = {
        label: _cells(cells) for label, cells in sorted(tally.classes.items())
    }
    summed: dict[str, Any] = _cells(totals)
    dry = run["dry_run"]
    if dry:
        for cells in (summed, *labels.values()):
            cells["listed"] = None
    others = sorted(tally.other.items(), key=lambda item: (-item[1][1], item[0]))
    rest = others[TOP_SUFFIXES:]
    counts = tally.counts
    members = {
        "extracted": {
            "files": counts["members_extracted_files"],
            "bytes": counts["members_extracted_bytes"],
        },
        "rejected": {key: counts[f"members_rejected_{key}"] for key in REJECTED},
        "nested": counts["members_nested"],
        "metadata": counts["members_metadata"],
        "other": {
            "files": counts["members_other_files"],
            "bytes": counts["members_other_bytes"],
        },
    }
    sniffed: dict[str, dict[str, int]] = {}
    for key, value in sorted(counts.items()):
        if key.startswith("sniffed:"):
            _, kind, unit = key.split(":")
            sniffed.setdefault(kind, {})[unit] = value
    return {
        "schema": SCHEMA,
        **run,
        "roots": {root.label: root.path for root in roots},
        "excludes": {"paths": list(scope.excludes), "globs": list(scope.globs)},
        "labels": labels,
        "totals": summed,
        "metadata_skipped": dict(sorted(tally.metadata.items())),
        "pruned": {kind: dict(sorted(c.items())) for kind, c in tally.pruned.items()},
        "sealed_guard": counts["sealed_guard"],
        "symlink_dirs": counts["symlink_dirs"],
        "broken_links": counts["broken_links"],
        "special_files": counts["special_files"],
        "renamed": counts["renamed"],
        "zst_unsupported": counts["zst_unsupported"],
        "other_suffixes": {
            suffix: {"files": files, "bytes": size}
            for suffix, (files, size) in others[:TOP_SUFFIXES]
        },
        "other_suffixes_rest": {
            "suffixes": len(rest),
            "files": sum(files for _, (files, _) in rest),
            "bytes": sum(size for _, (_, size) in rest),
        },
        "archive_members": None if dry else members,
        "sniffed": None if dry else sniffed,
        "decompression_failures": sorted(tally.failures["decompression"], key=_order),
        "read_failures": sorted(tally.failures["read"], key=_order),
        "listed": summed["listed"],
        "manifest_sha256": None,
    }


def build(
    roots: Sequence[Root],
    scope: Scope,
    work: str,
    *,
    node: str,
    hot_minutes: int = 90,
    workers: int = 1,
    dry_run: bool = False,
    log: Any = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """(manifest, None for a dry run; receipt without the manifest's sha256)."""
    started, utc_start = time.time(), _utc()
    zstd = zstd_method()
    tally = Tally([root.label for root in roots])
    entries: list[Entry] = []
    for root in roots:
        if sealed(root.path, root.real):
            tally.counts["sealed_guard"] += 1
        else:
            entries.extend(walk(root, scope, tally))
    if dry_run:
        for cells in tally.classes.values():
            cells["unsniffed"], cells["sniffed"] = cells["sniffed"], [0, 0]
    entries.sort()
    if log:
        log(f"{len(entries)} files to list, walked in {time.time() - started:.0f}s")
    jobs = plan(entries, work, started - hot_minutes * 60, zstd, tally)
    manifest = None
    if not dry_run:
        files: dict[str, list[dict[str, Any]]] = {root.label: [] for root in roots}
        for label, path, digest, size in sorted(
            execute(jobs, work, workers, zstd, tally, log)
        ):
            files[label].append({"path": path, "sha256": digest, "bytes": size})
            tally.add(label, "listed", size)
        manifest = {
            "schema": "c1-corpora/1",
            "labels": {
                label: {"kind": "training", "files": files[label]}
                for label in sorted(files)
            },
        }
    run = {
        "node": node,
        "utc_start": utc_start,
        "utc_end": _utc(),
        "dry_run": dry_run,
        "hot_minutes": hot_minutes,
        "workers": workers,
        "zstd": zstd,
    }
    return manifest, _receipt(roots, scope, tally, run)


def summary(receipt: dict[str, Any]) -> dict[str, Any]:
    failures = receipt["decompression_failures"] + receipt["read_failures"]
    return {
        "node": receipt["node"],
        "dry_run": receipt["dry_run"],
        "files": {kind: c and c["files"] for kind, c in receipt["totals"].items()},
        "listed": receipt["listed"],
        "failures": len(failures),
        "sealed_guard": receipt["sealed_guard"],
        "zst_unsupported": receipt["zst_unsupported"],
        "manifest_sha256": receipt["manifest_sha256"],
    }


def _log(message: str) -> None:
    print(f"[coverage] {message}", file=sys.stderr, flush=True)


def _write_new(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)


def _open(url: str, token: str) -> Any:
    request = urllib.request.Request(url)
    request.add_unredirected_header("Authorization", f"Bearer {token}")
    return urllib.request.urlopen(request, timeout=HF_TIMEOUT)


def _fetch(url: str, token: str) -> tuple[bytes, Any]:
    with _open(url, token) as response:
        return response.read(), response.headers


def _pages(url: str, token: str, origin: tuple[str, str]) -> Iterator[Any]:
    """Decoded JSON pages of a listing, following `Link: <...>; rel="next"` on the
    endpoint's host only."""
    while url:
        data, headers = _fetch(url, token)
        yield json.loads(data)
        match = NEXT.search(headers.get("Link") or "")
        if not match:
            return
        url = urllib.parse.urljoin(url, match.group(1))
        if tuple(urllib.parse.urlsplit(url)[:2]) != origin:
            raise Refusal(f"a pagination link leaves {origin[1]}")


def _tree(
    api: str, commit: str, token: str, origin: tuple[str, str]
) -> dict[tuple[str, str], tuple[int, bool]]:
    """{(path, LFS sha256 or blob sha1): (bytes, is LFS)} of a commit's files."""
    versions: dict[tuple[str, str], tuple[int, bool]] = {}
    url = f"{api}/tree/{commit}?recursive=true&expand=false"
    for page in _pages(url, token, origin):
        for entry in page:
            if entry["type"] != "file":
                continue
            lfs = entry.get("lfs")
            if lfs:
                versions[entry["path"], lfs["oid"]] = (lfs["size"], True)
            else:
                versions[entry["path"], entry["oid"]] = (entry["size"], False)
    return versions


def _download(url: str, token: str, dest: str, size: int, oid: str, lfs: bool) -> int:
    """Stream a file version into a new mode-600 file and check it against its LFS
    sha256 or git blob sha1; a failed or mismatching download leaves no file."""
    _makedirs(os.path.dirname(dest))
    descriptor = os.open(dest, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    digest = hashlib.sha256() if lfs else hashlib.sha1(b"blob %d\0" % size)
    count = 0
    try:
        with os.fdopen(descriptor, "wb") as out, _open(url, token) as response:
            for block in iter(lambda: response.read(BLOCK), b""):
                digest.update(block)
                out.write(block)
                count += len(block)
        if count != size or digest.hexdigest() != oid:
            kind = "LFS sha256" if lfs else "git blob sha1"
            raise Refusal(f"{dest} does not match its {kind} {oid}")
    except BaseException:
        os.unlink(dest)
        raise
    return count


def hf_delta(
    repo: str,
    repo_type: str,
    base: str,
    token: str,
    output: str,
    *,
    head: str | None = None,
    endpoint: str | None = None,
    log: Any = None,
) -> dict[str, Any]:
    """Download the file versions that appeared after `base`; the receipt."""
    utc_start = _utc()
    endpoint = (endpoint or os.environ.get("HF_ENDPOINT") or HF_ENDPOINT).rstrip("/")
    origin = tuple(urllib.parse.urlsplit(endpoint)[:2])
    api = f"{endpoint}/api/{repo_type}s/{repo}"
    files = (
        f"{endpoint}/datasets/{repo}"
        if repo_type == "dataset"
        else f"{endpoint}/{repo}"
    )
    revision = urllib.parse.quote(head or "main", safe="")
    data, _ = _fetch(f"{api}/revision/{revision}", token)
    head_sha = json.loads(data)["sha"]
    later: list[str] = []
    base_sha = None
    for page in _pages(f"{api}/commits/{head_sha}", token, origin):
        for commit in page:
            if commit["id"].startswith(base):
                base_sha = commit["id"]
                break
            later.append(commit["id"])
        if base_sha:
            break
    if base_sha is None:
        raise Refusal(f"base {base} is not in the history of {head_sha}")
    later.reverse()
    known = set(_tree(api, base_sha, token, origin))
    first: dict[tuple[str, str], tuple[str, int, bool]] = {}
    for commit in later:
        for version, (size, lfs) in _tree(api, commit, token, origin).items():
            if version not in known and version not in first:
                first[version] = (commit, size, lfs)
    for path, _ in first:
        dest = os.path.join(output, path)
        if _member_path(path) != (path, "") or sealed(dest):
            raise Refusal(f"unsafe path {path!r} in the repo tree")
    if log:
        log(f"{len(later)} commits after {base_sha[:12]}, {len(first)} new versions")
    _makedirs(output)
    total = 0
    tops: Counter = Counter()
    order = {commit: number for number, commit in enumerate(later)}
    for (path, oid), (commit, size, lfs) in sorted(
        first.items(), key=lambda item: (order[item[1][0]], item[0])
    ):
        url = f"{files}/resolve/{commit}/{urllib.parse.quote(path)}"
        dest = os.path.join(output, commit[:12], path)
        total += _download(url, token, dest, size, oid, lfs)
        tops[path.split("/")[0] if "/" in path else "."] += 1
    return {
        "schema": HF_SCHEMA,
        "repo": repo,
        "repo_type": repo_type,
        "base": base_sha,
        "head": head_sha,
        "later_commits": later,
        "downloaded": {"versions": len(first), "bytes": total},
        "top_level": dict(sorted(tops.items())),
        "utc_start": utc_start,
        "utc_end": _utc(),
    }


def _hf_delta_main(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    if not REPO.fullmatch(args.repo):
        parser.error(f"--repo must be NAME or OWNER/NAME: {args.repo}")
    if not REVISION.fullmatch(args.base):
        parser.error("--base must be a commit sha or a prefix of 4+ lowercase hex")
    for path in (args.output, args.receipt):
        if sealed(os.path.abspath(path), os.path.realpath(path)):
            parser.error(f"refusing to write under {SEALED}: {path}")
    if args.receipt.exists() or args.receipt.is_symlink():
        parser.error(f"refusing to overwrite {args.receipt}")
    if (
        args.output.is_symlink()
        or args.output.exists()
        and (not args.output.is_dir() or any(args.output.iterdir()))
    ):
        parser.error(f"--output must be a new or empty directory: {args.output}")
    try:
        token = args.token_file.read_text().strip()
    except OSError as error:
        parser.error(f"cannot read {args.token_file}: {error.strerror}")
    if not token:
        parser.error(f"empty token file {args.token_file}")
    try:
        receipt = hf_delta(
            args.repo,
            args.repo_type,
            args.base,
            token,
            str(args.output),
            head=args.head,
            endpoint=args.endpoint,
            log=_log,
        )
    except (Refusal, OSError, ValueError, KeyError, TypeError) as error:
        message = str(error).replace(token, "<token>")
        print(f"hf-delta refused: {type(error).__name__}: {message}", file=sys.stderr)
        return 1
    text = json.dumps(receipt, indent=1, sort_keys=True) + "\n"
    _write_new(args.receipt, text.encode())
    brief = {key: receipt[key] for key in ("repo", "base", "head")}
    brief["later_commits"] = len(receipt["later_commits"])
    print(json.dumps({**brief, **receipt["downloaded"]}, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("build", help="write one node's corpus manifest")
    command.add_argument("--node", required=True)
    command.add_argument("--root", action="append", required=True, metavar="LABEL=PATH")
    command.add_argument(
        "--exclude", action="append", default=[], metavar="PATH_OR_GLOB"
    )
    command.add_argument("--work", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--receipt", type=Path, required=True)
    command.add_argument("--hot-minutes", type=int, default=90)
    command.add_argument("--workers", type=int, default=16)
    command.add_argument("--dry-run", action="store_true")
    delta = commands.add_parser(
        "hf-delta", help="download the file versions a Hugging Face repo gained"
    )
    delta.add_argument("--repo", required=True)
    delta.add_argument("--repo-type", required=True, choices=("dataset", "model"))
    delta.add_argument("--base", required=True)
    delta.add_argument("--head")
    delta.add_argument("--token-file", type=Path, required=True)
    delta.add_argument("--output", type=Path, required=True)
    delta.add_argument("--receipt", type=Path, required=True)
    delta.add_argument("--endpoint")
    args = parser.parse_args(argv)
    if args.command == "hf-delta":
        return _hf_delta_main(parser, args)
    if not args.node.strip() or args.workers < 1 or args.hot_minutes < 0:
        parser.error("need a --node name, --workers >= 1 and --hot-minutes >= 0")
    if args.output.resolve() == args.receipt.resolve():
        parser.error("manifest and receipt must be different files")
    for path in (args.output, args.receipt):
        if path.exists() or path.is_symlink():
            parser.error(f"refusing to overwrite {path}")
    for path in (args.work, args.output, args.receipt):
        if sealed(os.path.abspath(path), os.path.realpath(path)):
            parser.error(f"refusing to write under {SEALED}: {path}")
    try:
        roots, scope, work = setup(args.root, args.exclude, str(args.work))
    except ValueError as error:
        parser.error(str(error))
    manifest, receipt = build(
        roots,
        scope,
        work,
        node=args.node,
        hot_minutes=args.hot_minutes,
        workers=args.workers,
        dry_run=args.dry_run,
        log=_log,
    )
    if manifest is not None:
        data = (json.dumps(manifest, indent=1, sort_keys=True) + "\n").encode()
        receipt["manifest_sha256"] = hashlib.sha256(data).hexdigest()
        _write_new(args.output, data)
    text = json.dumps(receipt, indent=1, sort_keys=True) + "\n"
    _write_new(args.receipt, text.encode())
    print(json.dumps(summary(receipt), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
