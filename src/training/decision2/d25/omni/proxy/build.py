"""Build one proxy set into ``$D25_OMNI_PROXY/<set>/`` in the vision-suite layout.

    python -m d25.omni.proxy.build charts --n 420 --seed 20261009 [--out DIR] [--workers 8]

A set module exposes ``NAME``, ``BENCHMARK``, ``VERSION``, ``build_item(index, seed, context)`` and,
optionally, ``prepare(work, seed[, n]) -> context`` (source downloads, indexing and the per-item
candidate plan, cached under ``work``), ``worker_init()`` / ``worker_close()`` (per-process resources such as a browser) and
``sources(context) -> {key: info}`` for the manifest and ``size(context)`` when a source caps the
item count. Finished items are kept as ``parts/<index>.json``
so a restarted Job resumes; the final rows, manifest and checksums are written once every item exists.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from d25.omni.proxy.rows import ProxyWriter

SETS = {
    "charts": "d25.omni.proxy.charts",
    "kie": "d25.omni.proxy.kie",
    "infographics": "d25.omni.proxy.infographics",
    "web": "d25.omni.proxy.web",
    "correspondence": "d25.omni.proxy.correspondence",
    "cvbench": "d25.omni.proxy.cvbench",
    "realworld": "d25.omni.proxy.realworld",
    "moderation": "d25.omni.proxy.moderation",
    "winoground": "d25.omni.proxy.winoground",
}

_module: Any = None
_context: Any = None
_root: Path | None = None
_seed = 0


def _init(module_name: str, context: Any, root: str, seed: int) -> None:
    global _module, _context, _root, _seed
    _module = importlib.import_module(module_name)
    _context, _root, _seed = context, Path(root), seed
    if hasattr(_module, "worker_init"):
        _module.worker_init()


def _work(index: int) -> tuple[int, str | None]:
    part = _root / "parts" / f"{index:05d}.json"
    if part.exists():
        return index, None
    try:
        item = _module.build_item(index, _seed, _context)
        writer = ProxyWriter(
            _root, _module.NAME, _module.BENCHMARK, _module.VERSION, _seed
        )
        images = [writer.image(payload, ext) for payload, ext in item.pop("payloads")]
        item["images"] = images
        tmp = part.with_suffix(".tmp")
        tmp.write_text(json.dumps(item, ensure_ascii=False))
        tmp.replace(part)
        return index, None
    except Exception:
        return index, traceback.format_exc(limit=6)


def build(
    name: str, n: int, seed: int, out: Path, workers: int, limit_errors: int = 50
) -> dict[str, Any]:
    module = importlib.import_module(SETS[name])
    out.mkdir(parents=True, exist_ok=True)
    (out / "parts").mkdir(exist_ok=True)
    work = out.parent / "_work" / name
    work.mkdir(parents=True, exist_ok=True)
    context = None
    if hasattr(module, "prepare"):
        params = inspect.signature(module.prepare).parameters
        context = (
            module.prepare(work, seed, n)
            if "n" in params
            else module.prepare(work, seed)
        )
    if hasattr(module, "size"):
        n = min(n, module.size(context))
    todo = [i for i in range(n) if not (out / "parts" / f"{i:05d}.json").exists()]
    print(
        f"{name}: {n - len(todo)} items done, {len(todo)} to build with {workers} workers",
        flush=True,
    )
    errors: list[str] = []
    started = time.time()
    if todo:
        ctx = mp.get_context("fork")
        with ctx.Pool(
            workers, initializer=_init, initargs=(SETS[name], context, str(out), seed)
        ) as pool:
            for done, (index, error) in enumerate(
                pool.imap_unordered(_work, todo, chunksize=1), 1
            ):
                if error:
                    errors.append(f"item {index}: {error}")
                    print(errors[-1], flush=True)
                    if len(errors) >= limit_errors:
                        pool.terminate()
                        break
                if done % 50 == 0 or done == len(todo):
                    rate = done / max(1e-6, time.time() - started)
                    print(
                        f"{name}: {done}/{len(todo)} ({rate:.2f} items/s)", flush=True
                    )
    missing = [i for i in range(n) if not (out / "parts" / f"{i:05d}.json").exists()]
    if missing:
        raise SystemExit(
            f"{name}: {len(missing)} items missing (first errors above); rerun to resume"
        )
    writer = ProxyWriter(out, module.NAME, module.BENCHMARK, module.VERSION, seed)
    for key, info in (
        module.sources(context) if hasattr(module, "sources") else {}
    ).items():
        writer.source(key, **info)
    for i in range(n):
        item = json.loads((out / "parts" / f"{i:05d}.json").read_text())
        writer.add(
            item["item_id"],
            item["subtask"],
            [tuple(x) for x in item["images"]],
            item["instructions"],
            item["criteria"],
            item["answer"],
            state=item.get("state", ""),
            provenance=item.get("provenance", []),
            extra=item.get("extra"),
        )
    manifest = writer.close(command=sys.argv, code_tag=os.environ.get("D25_CODE_TAG"))
    print(
        json.dumps(
            {
                k: manifest[k]
                for k in (
                    "proxy",
                    "rows",
                    "images",
                    "subtasks",
                    "answer_keys",
                    "mean_chance",
                )
            },
            indent=1,
        )
    )
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("set", choices=sorted(SETS))
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20261009)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--workers", type=int, default=int(os.environ.get("D25_WORKERS", "4"))
    )
    args = parser.parse_args(argv)
    root = Path(os.environ.get("D25_OMNI_PROXY", "/data/d25/omni/proxy"))
    out = args.out or root / importlib.import_module(SETS[args.set]).NAME
    build(args.set, args.n, args.seed, out, args.workers)


if __name__ == "__main__":
    main()
