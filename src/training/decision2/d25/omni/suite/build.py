"""Build the public Vision-board 0.3.1 suite from fetched sources.

    python -m d25.omni.suite.build --sources /data/d25/omni/suite/sources \
        --out /data/d25/omni/suite/vision-0.3.1 [--only CV-Bench,BLINK] [--rebuild]

Each benchmark is built into ``parts/<slug>.jsonl.gz`` plus ``parts/<slug>.json`` (stats) and its
variants into ``variants/<name>.jsonl.gz``; finished parts are skipped, so the build resumes. The
final step writes ``rows.jsonl.gz`` (benchmarks in board order), ``manifest.json`` (inventory and
board targets), ``BUILD.json`` (sources, revisions, file digests, counts per benchmark and subtask,
option and image-count distributions, lattice checks, construction notes) and ``SHA256SUMS``.
Outputs are byte-deterministic for the same sources and code.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import subprocess
import time
from collections import Counter
from pathlib import Path

from d25.omni.suite import lattice, score
from d25.omni.suite import sources as src
from d25.omni.suite.benchmarks import MODULES, builder
from d25.omni.suite.rows import Context

EDITION = "0.3.1"


def slug(benchmark: str) -> str:
    return MODULES[benchmark]


def dumps(row: dict) -> str:
    return json.dumps(row, ensure_ascii=False, separators=(",", ":"))


def write_jsonl_gz(path: Path, rows) -> str:
    """Deterministic gzip (mtime 0, no name); returns the sha256 of the file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as raw, gzip.GzipFile(
        filename="", mode="wb", fileobj=raw, mtime=0
    ) as f:
        for row in rows:
            f.write((dumps(row) + "\n").encode("utf-8"))
    tmp.replace(path)
    return src.sha256_file(path)


def read_jsonl_gz(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_json(path: Path, value) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(value, indent=1, ensure_ascii=False, sort_keys=False) + "\n"
    )
    tmp.replace(path)


def stats(rows: list[dict]) -> dict:
    chance_sum = sum(score.row_chance(r) for r in rows)
    return {
        "rows": len(rows),
        "chance_sum": round(chance_sum, 6),
        "chance": round(chance_sum / len(rows), 8) if rows else None,
        "options": dict(
            sorted(Counter(r["metadata"]["n_options"] for r in rows).items())
        ),
        "images_per_row": dict(
            sorted(Counter(r["metadata"]["n_images"] for r in rows).items())
        ),
        "subtasks": dict(
            sorted(Counter(str(r["metadata"]["subtask"]) for r in rows).items())
        ),
        "gold": dict(sorted(Counter(r["expected"]["q1"] for r in rows).items())),
        "splits": dict(sorted(Counter(r["split"] for r in rows).items())),
    }


def build_part(benchmark: str, ctx: Context, out: Path, rebuild: bool = False) -> dict:
    meta_path = out / "parts" / f"{slug(benchmark)}.json"
    rows_path = out / "parts" / f"{slug(benchmark)}.jsonl.gz"
    if meta_path.exists() and rows_path.exists() and not rebuild:
        meta = json.loads(meta_path.read_text())
        if meta.get("rows_sha256") == src.sha256_file(rows_path):
            return meta
    started = time.time()
    module = builder(benchmark)
    result = module.build(ctx)
    rows = result["rows"]
    ids = [r["id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise RuntimeError(f"{benchmark}: duplicate row ids")
    variants = {}
    for name, vrows in sorted(result["variants"].items()):
        variants[name] = {
            "rows": len(vrows),
            "sha256": write_jsonl_gz(out / "variants" / f"{name}.jsonl.gz", vrows),
            **{k: v for k, v in stats(vrows).items() if k != "rows"},
        }
    meta = {
        "benchmark": benchmark,
        "board_rows": score.PUBLIC_ROWS[benchmark],
        **stats(rows),
        "lattice_misses": (
            lattice.check(
                len(rows),
                sum(score.row_chance(r) for r in rows),
                lattice.board_values()[benchmark],
            )
            if rows
            else None
        ),
        "notes": result["notes"],
        "sources": list(module.SOURCES),
        "variants": variants,
        "image_index": {d: ctx.images[d] for d in sorted(ctx.images)},
        "seconds": round(time.time() - started, 1),
        "rows_sha256": write_jsonl_gz(rows_path, rows),
    }
    write_json(meta_path, meta)
    ctx.images.clear()
    return meta


def code_version() -> dict:
    here = Path(__file__).resolve().parent
    files = sorted(p for p in here.rglob("*.py") if "tests" not in p.parts)
    digest = hashlib.sha256()
    for p in files:
        digest.update(p.relative_to(here).as_posix().encode() + b"\0" + p.read_bytes())
    out = {
        "suite_code_sha256": digest.hexdigest(),
        "tag": os.environ.get("D25_SRC_TAG"),
    }
    try:
        out["git"] = (
            subprocess.run(
                ["git", "-C", str(here), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout.strip()
            or None
        )
    except Exception:
        out["git"] = None
    return out


def finalize(out: Path, parts: dict[str, dict], sources_root: Path) -> dict:
    rows_sha = write_jsonl_gz(
        out / "rows.jsonl.gz",
        (
            r
            for b in score.BENCHMARKS
            if b in parts
            for r in read_jsonl_gz(out / "parts" / f"{slug(b)}.jsonl.gz")
        ),
    )
    images = {}
    for meta in parts.values():
        images.update(meta["image_index"])
    used = set()
    for b in parts:
        for r in read_jsonl_gz(out / "parts" / f"{slug(b)}.jsonl.gz"):
            used.update(r["images"])
    keys = sorted({k for m in parts.values() for k in m["sources"]})
    fetch = {k: src.record(k, sources_root) for k in keys}
    benchmarks = {
        b: {k: v for k, v in parts[b].items() if k != "image_index"}
        | {"exact_count": parts[b]["rows"] == score.PUBLIC_ROWS[b]}
        for b in score.BENCHMARKS
        if b in parts
    }
    total = sum(m["rows"] for m in parts.values())
    manifest = {
        "suite": f"JEV Decision Index Vision board {EDITION}, public part (rebuild)",
        "rows_file": "rows.jsonl.gz",
        "rows_sha256": rows_sha,
        "rows": total,
        "board_rows": sum(score.PUBLIC_ROWS.values()),
        "complete": set(parts) == set(score.BENCHMARKS),
        "images_used": len(used),
        "images_stored": len(images),
        "benchmarks": {
            b: {
                "rows": v["rows"],
                "board_rows": v["board_rows"],
                "chance": v["chance"],
                "weight": score.WEIGHTS[b],
                "lattice_ok": not v["lattice_misses"],
                "rows_sha256": v["rows_sha256"],
            }
            for b, v in benchmarks.items()
        },
        "variants": {
            name: {
                "file": f"variants/{name}.jsonl.gz",
                **{k: info[k] for k in ("rows", "sha256")},
            }
            for v in benchmarks.values()
            for name, info in v["variants"].items()
        },
        "row_format": "kit row + images; see d25/omni/suite/rows.py and d25/omni/common/vision_format.py",
        "licence_note": "Benchmark data stays on our nodes. Winoground: Meta Images Research License, evaluation only.",
    }
    build = {
        "edition": EDITION,
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "code": code_version(),
        "sources": {
            k: {
                "repo": f.get("repo") or f.get("url"),
                "revision": f.get("revision"),
                "licence": f.get("licence"),
                "files": f["files"],
            }
            for k, f in fetch.items()
        },
        "benchmarks": benchmarks,
        "totals": {
            "rows": total,
            "options": dict(
                sorted(
                    sum(
                        (
                            Counter({int(k): v for k, v in m["options"].items()})
                            for m in parts.values()
                        ),
                        Counter(),
                    ).items()
                )
            ),
            "images_per_row": dict(
                sorted(
                    sum(
                        (
                            Counter({int(k): v for k, v in m["images_per_row"].items()})
                            for m in parts.values()
                        ),
                        Counter(),
                    ).items()
                )
            ),
            "images_used": len(used),
        },
        "image_index": {
            d: images[d] for d in sorted(images) if images[d]["path"] in used
        },
    }
    write_json(out / "manifest.json", manifest)
    write_json(out / "BUILD.json", build)
    write_sums(out)
    return manifest


def write_sums(out: Path) -> None:
    lines = []
    for path in sorted(p for p in out.rglob("*") if p.is_file()):
        rel = path.relative_to(out).as_posix()
        if rel == "SHA256SUMS" or rel.startswith("parts/") or rel.endswith(".tmp"):
            continue
        name = path.name
        digest = (
            name.split(".")[0]
            if rel.startswith("images/") and re.fullmatch(r"[0-9a-f]{64}\.\w+", name)
            else src.sha256_file(path)
        )
        lines.append(f"{digest}  {rel}")
    (out / "SHA256SUMS").write_text("\n".join(lines) + "\n")


def verify(out: Path) -> list[str]:
    """Files whose sha256 does not match SHA256SUMS (recomputes every digest, images included)."""
    bad = []
    for line in (out / "SHA256SUMS").read_text().splitlines():
        digest, rel = line.split("  ", 1)
        path = out / rel
        if not path.exists() or src.sha256_file(path) != digest:
            bad.append(rel)
    return bad


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sources", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--only", default="", help="comma-separated board benchmark names"
    )
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument(
        "--verify", action="store_true", help="recheck SHA256SUMS after finalizing"
    )
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ctx = Context(args.sources, out)
    wanted = [b for b in args.only.split(",") if b] or list(score.BENCHMARKS)
    parts = {}
    for benchmark in score.BENCHMARKS:
        meta_path = out / "parts" / f"{slug(benchmark)}.json"
        if benchmark in wanted:
            meta = build_part(benchmark, ctx, out, args.rebuild)
            print(
                json.dumps(
                    {
                        "benchmark": benchmark,
                        "rows": meta["rows"],
                        "board": meta["board_rows"],
                        "chance_sum": meta["chance_sum"],
                        "lattice_misses": len(meta["lattice_misses"] or []),
                        "seconds": meta["seconds"],
                    }
                ),
                flush=True,
            )
            parts[benchmark] = meta
        elif meta_path.exists():
            parts[benchmark] = json.loads(meta_path.read_text())
    manifest = finalize(out, parts, Path(args.sources))
    print(
        json.dumps(
            {k: manifest[k] for k in ("rows", "board_rows", "complete", "images_used")}
        ),
        flush=True,
    )
    if args.verify:
        bad = verify(out)
        print(json.dumps({"verify_failures": bad[:20], "n": len(bad)}), flush=True)
        if bad:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
