"""bf16z: lossless byte-plane zstd storage for safetensors weight files (ZipNN-style).

A ``<name>.safetensors.bf16z`` file restores ``<name>.safetensors`` byte for byte.
Each tensor's bytes are split into byte planes (BF16: the low mantissa byte and the
sign / exponent byte; FP32: all four bytes), and each plane is stored with zstd or,
when zstd does not shrink it, raw. The original safetensors header is stored
uncompressed, so parameter counts need no decompression. Decompression checks the
restored file's SHA-256 against the one recorded at compression.

Layout: ``MAGIC``, u64 LE index length, the JSON index, the exact original
safetensors prefix (u64 header length + header), then the plane blobs.

    python -m v2.release.runtime.bf16z compress --source CKPT --output NEW --receipt R.json
    python -m v2.release.runtime.bf16z verify --dir CKPT_OR_PACKAGE [--receipt R.json]

Needs numpy and zstandard (BSD).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mmap
import os
import shutil
import struct
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

FORMAT = "bf16z/1"
MAGIC = b"BF16Z\x00\x01\n"
SUFFIX = ".bf16z"
RECEIPT_SCHEMA = "dev2-release-bf16z/1"
ITEMSIZE = {
    "BF16": 2,
    "F16": 2,
    "F32": 4,
    "F64": 8,
    "I64": 8,
    "U64": 8,
    "I32": 4,
    "U32": 4,
    "I16": 2,
    "U16": 2,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
}
DEFAULT_LEVEL = 19
PROBE_BYTES = 4 << 20
PROBE_RAW_RATIO = 0.97
BIG_PLANE = 64 << 20


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_prefix(path: Path) -> tuple[bytes, dict[str, Any]]:
    """The exact safetensors prefix (u64 length + header bytes) and the parsed header."""
    with Path(path).open("rb") as stream:
        raw = stream.read(8)
        (size,) = struct.unpack("<Q", raw)
        if not 2 <= size <= 256 << 20:
            raise ValueError(f"Invalid safetensors header: {path}")
        header_bytes = stream.read(size)
    header = json.loads(header_bytes)
    if not isinstance(header, dict):
        raise ValueError(f"Invalid safetensors header: {path}")
    return raw + header_bytes, header


def _tensors(header: dict[str, Any], data_bytes: int) -> list[dict[str, Any]]:
    """Tensors in data order; the data region must be covered exactly, without holes."""
    rows = []
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        begin, end = meta["data_offsets"]
        rows.append({"name": name, "dtype": meta["dtype"], "begin": begin, "end": end})
    rows.sort(key=lambda r: (r["begin"], r["end"]))
    cursor = 0
    for row in rows:
        if row["begin"] != cursor or row["end"] < row["begin"]:
            raise ValueError("safetensors data region has a hole or an overlap")
        cursor = row["end"]
    if cursor != data_bytes:
        raise ValueError("safetensors data region does not end at the file end")
    return rows


def _planes(buffer: memoryview, itemsize: int) -> list[bytes]:
    import numpy as np

    if itemsize == 1 or len(buffer) % itemsize:
        return [bytes(buffer)]
    view = np.frombuffer(buffer, dtype=np.uint8).reshape(-1, itemsize)
    return [np.ascontiguousarray(view[:, j]).tobytes() for j in range(itemsize)]


def _compress_plane(plane: bytes, level: int, threads: int) -> tuple[str, bytes]:
    import zstandard

    if not plane:
        return "raw", plane
    probe = plane[:PROBE_BYTES]
    if len(zstandard.ZstdCompressor(level=3).compress(probe)) >= PROBE_RAW_RATIO * len(
        probe
    ):
        return "raw", plane
    workers = threads if len(plane) >= BIG_PLANE else 0
    packed = zstandard.ZstdCompressor(level=level, threads=workers).compress(plane)
    if len(packed) >= len(plane):
        return "raw", plane
    return "zstd", packed


def compress_file(
    source: Path, target: Path, *, level: int = DEFAULT_LEVEL, pool=None, threads=8
) -> dict[str, Any]:
    """Write ``target`` (a .bf16z file) that restores ``source`` exactly."""
    prefix, header = read_prefix(source)
    total = Path(source).stat().st_size
    rows = _tensors(header, total - len(prefix))
    with Path(source).open("rb") as stream, mmap.mmap(
        stream.fileno(), 0, access=mmap.ACCESS_READ
    ) as mapped:
        start = len(prefix)

        def work(row: dict[str, Any]) -> list[tuple[str, bytes, int]]:
            buffer = memoryview(mapped)[start + row["begin"] : start + row["end"]]
            try:
                size = ITEMSIZE.get(row["dtype"], 1)
                return [
                    (*_compress_plane(p, level, threads), len(p))
                    for p in _planes(buffer, size)
                ]
            finally:
                buffer.release()

        own = pool is None
        pool = pool or ThreadPoolExecutor(max_workers=os.cpu_count() or 8)
        try:
            results = list(pool.map(work, rows))
        finally:
            if own:
                pool.shutdown()
    offset = 0
    index_rows = []
    for row, planes in zip(rows, results):
        entry = dict(row)
        entry["itemsize"] = ITEMSIZE.get(row["dtype"], 1)
        entry["planes"] = []
        for codec, blob, size in planes:
            entry["planes"].append(
                {"codec": codec, "offset": offset, "length": len(blob), "size": size}
            )
            offset += len(blob)
        index_rows.append(entry)
    index = {
        "format": FORMAT,
        "source": {
            "name": Path(source).name,
            "bytes": total,
            "sha256": _sha_file(source),
        },
        "prefix_bytes": len(prefix),
        "level": level,
        "tensors": index_rows,
    }
    encoded = json.dumps(index, sort_keys=True, separators=(",", ":")).encode()
    pending = Path(str(target) + ".pending")
    with pending.open("wb") as out:
        out.write(MAGIC + struct.pack("<Q", len(encoded)) + encoded + prefix)
        for planes in results:
            for _, blob, _ in planes:
                out.write(blob)
    pending.rename(target)
    return {
        "source_sha256": index["source"]["sha256"],
        "source_bytes": total,
        "sha256": _sha_file(target),
        "bytes": Path(target).stat().st_size,
    }


def read_index(path: Path) -> tuple[dict[str, Any], bytes, int]:
    """(index, exact original prefix, offset of the first plane blob)."""
    with Path(path).open("rb") as stream:
        if stream.read(len(MAGIC)) != MAGIC:
            raise ValueError(f"Not a {FORMAT} file: {path}")
        (size,) = struct.unpack("<Q", stream.read(8))
        if not 2 <= size <= 1 << 30:
            raise ValueError(f"Invalid {FORMAT} index: {path}")
        index = json.loads(stream.read(size))
        if index.get("format") != FORMAT:
            raise ValueError(f"Unknown bf16z format in {path}")
        prefix = stream.read(index["prefix_bytes"])
    return index, prefix, len(MAGIC) + 8 + size + len(prefix)


def original_header(path: Path) -> dict[str, Any]:
    """The restored file's safetensors header, read without decompressing weights."""
    _, prefix, _ = read_index(path)
    (size,) = struct.unpack("<Q", prefix[:8])
    return json.loads(prefix[8 : 8 + size])


def decompress_file(source: Path, target: Path | None = None) -> dict[str, Any]:
    """Restore the original file (or only hash it when ``target`` is None); SHA-256 checked."""
    import numpy as np
    import zstandard

    index, prefix, base = read_index(source)
    digest = hashlib.sha256(prefix)
    written = len(prefix)
    pending = Path(str(target) + ".pending") if target is not None else None
    out = pending.open("wb") if pending is not None else None
    decompressor = zstandard.ZstdDecompressor()
    try:
        if out is not None:
            out.write(prefix)
        with Path(source).open("rb") as stream, mmap.mmap(
            stream.fileno(), 0, access=mmap.ACCESS_READ
        ) as mapped:
            for row in index["tensors"]:
                planes = []
                for plane in row["planes"]:
                    blob = mapped[
                        base
                        + plane["offset"] : base
                        + plane["offset"]
                        + plane["length"]
                    ]
                    if plane["codec"] == "zstd":
                        blob = decompressor.decompress(
                            blob, max_output_size=plane["size"]
                        )
                    elif plane["codec"] != "raw":
                        raise ValueError(f"Unknown plane codec {plane['codec']}")
                    if len(blob) != plane["size"]:
                        raise ValueError(f"Plane size differs in {row['name']}")
                    planes.append(blob)
                if len(planes) == 1:
                    data = memoryview(planes[0])
                else:
                    stacked = np.empty(
                        (row["planes"][0]["size"], len(planes)), dtype=np.uint8
                    )
                    for j, blob in enumerate(planes):
                        stacked[:, j] = np.frombuffer(blob, dtype=np.uint8)
                    data = memoryview(stacked.reshape(-1))
                if data.nbytes != row["end"] - row["begin"]:
                    raise ValueError(f"Tensor size differs in {row['name']}")
                digest.update(data)
                written += data.nbytes
                if out is not None:
                    out.write(data)
    finally:
        if out is not None:
            out.close()
    sha = digest.hexdigest()
    if sha != index["source"]["sha256"] or written != index["source"]["bytes"]:
        if pending is not None:
            pending.unlink(missing_ok=True)
        raise ValueError(f"{source} does not restore its recorded source bytes")
    if pending is not None:
        pending.rename(target)
    return {"sha256": sha, "bytes": written}


def restored_name(name: str) -> str:
    if not name.endswith(SUFFIX):
        raise ValueError(f"Not a {SUFFIX} name: {name}")
    return name[: -len(SUFFIX)]


def cache_root() -> Path:
    value = os.environ.get("DECISION2_CACHE")
    return Path(value) if value else Path.home() / ".cache" / "decision2"


def materialize(
    root: Path, manifest: dict[str, Any], cache: Path | None = None
) -> Path:
    """A plain checkpoint directory with every model file restored, reused when present.

    The caller checks the restored checkpoint's identity against the manifest.
    """
    storage = manifest["storage"]
    if storage.get("codec") != FORMAT:
        raise ValueError(f"Unknown weight storage codec: {storage.get('codec')}")
    identity = manifest["identity"]["model_sha256"]
    target = (cache or cache_root()) / "bf16z" / identity
    if (target / ".complete").is_file():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    pending = target.parent / f".{identity}.{os.getpid()}.{time.time_ns()}"
    pending.mkdir()
    try:
        jobs = []
        for name in manifest["model_files"]:
            if name.endswith(SUFFIX):
                entry = storage["files"][name]
                out = pending / restored_name(name)
                out.parent.mkdir(parents=True, exist_ok=True)
                jobs.append((root / name, out, entry))
            else:
                out = pending / name
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(root / name, out)

        def restore(job: tuple[Path, Path, dict[str, Any]]) -> None:
            source, out, entry = job
            done = decompress_file(source, out)
            if done["sha256"] != entry["restored_sha256"]:
                raise ValueError(
                    f"{source.name} restores other bytes than the manifest"
                )

        with ThreadPoolExecutor(max_workers=min(16, len(jobs) or 1)) as pool:
            list(pool.map(restore, jobs))
        (pending / ".complete").write_text(identity + "\n", encoding="utf-8")
        try:
            pending.rename(target)
        except OSError:
            if not (target / ".complete").is_file():
                raise
            shutil.rmtree(pending, ignore_errors=True)
    except BaseException:
        shutil.rmtree(pending, ignore_errors=True)
        raise
    return target


def compress_checkpoint(
    source: Path, output: Path, *, level: int = DEFAULT_LEVEL
) -> dict[str, Any]:
    """Copy a checkpoint, storing every backbone safetensors shard as .bf16z."""
    if output.exists():
        raise FileExistsError(output)
    pending = output.with_name(output.name + ".pending")
    shards = sorted((source / "backbone").glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"No backbone safetensors under {source}")
    skip = {s.name for s in shards}
    shutil.copytree(
        source,
        pending,
        ignore=lambda d, names: (
            skip & set(names)
            if Path(d).resolve() == (source / "backbone").resolve()
            else set()
        ),
    )
    files = {}
    started = time.time()
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 8) as pool:
        for shard in shards:
            name = f"backbone/{shard.name}{SUFFIX}"
            files[name] = compress_file(shard, pending / name, level=level, pool=pool)
    pending.rename(output)
    for name, entry in files.items():
        entry["restored"] = restored_name(name)
    return {"files": files, "seconds": time.time() - started}


def verify_dir(root: Path, entries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Hash every .bf16z file and the bytes it restores (nothing written)."""

    def check(item: tuple[str, dict[str, Any]]) -> tuple[str, dict[str, Any]]:
        name, entry = item
        path = root / name
        restored = decompress_file(path)
        sha = _sha_file(path)
        return name, {
            "sha256": sha,
            "restored_sha256": restored["sha256"],
            "restored_bytes": restored["bytes"],
            "ok": sha == entry["sha256"]
            and restored["sha256"] == entry["restored_sha256"],
        }

    with ThreadPoolExecutor(max_workers=16) as pool:
        results = dict(pool.map(check, sorted(entries.items())))
    return {"files": results, "passed": all(r["ok"] for r in results.values())}


def make_receipt(result: dict[str, Any], level: int, source: Path) -> dict[str, Any]:
    """The compression receipt the release builder pins (``bf16z`` spec entry)."""
    import zstandard

    files = result["files"]
    return {
        "schema": RECEIPT_SCHEMA,
        "format": FORMAT,
        "level": level,
        "zstandard": zstandard.__version__,
        "source": str(source),
        "seconds": result["seconds"],
        "files": {
            name: {
                "sha256": e["sha256"],
                "bytes": e["bytes"],
                "restored": e["restored"],
                "restored_sha256": e["source_sha256"],
                "restored_bytes": e["source_bytes"],
            }
            for name, e in files.items()
        },
        "bytes": sum(e["bytes"] for e in files.values()),
        "restored_bytes": sum(e["source_bytes"] for e in files.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("compress")
    c.add_argument("--source", type=Path, required=True)
    c.add_argument("--output", type=Path, required=True)
    c.add_argument("--receipt", type=Path, required=True)
    c.add_argument("--level", type=int, default=DEFAULT_LEVEL)
    v = sub.add_parser("verify")
    v.add_argument("--dir", type=Path, required=True)
    v.add_argument("--receipt", type=Path)
    v.add_argument("--output", type=Path, required=True)
    r = sub.add_parser("restore")
    r.add_argument("--package", type=Path, required=True)
    r.add_argument("--cache", type=Path)
    args = parser.parse_args()
    if args.command == "compress":
        if args.receipt.exists():
            raise FileExistsError(args.receipt)
        result = compress_checkpoint(args.source, args.output, level=args.level)
        receipt = make_receipt(result, args.level, args.source)
        args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        print(
            json.dumps({k: receipt[k] for k in ("bytes", "restored_bytes", "seconds")})
        )
        return
    if args.command == "restore":
        manifest = json.loads((args.package / "MODEL_MANIFEST.json").read_text())
        print(materialize(args.package, manifest, args.cache))
        return
    if args.receipt:
        entries = json.loads(args.receipt.read_text())["files"]
    else:
        manifest = json.loads((args.dir / "MODEL_MANIFEST.json").read_text())
        entries = manifest["storage"]["files"]
    report = verify_dir(args.dir, entries)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": report["passed"], "files": len(report["files"])}))
    sys.exit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
