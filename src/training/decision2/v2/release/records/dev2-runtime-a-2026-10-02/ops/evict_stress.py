"""Graph-eviction stress of a released Decision 2.0 package on the Index panel (record section 8).

The released runtime (the package's own decision2/, or --runtime DIR's modules in their place) answers the Index
panel (panel-3, in panel order) in groups of --batch consecutive requests (--shard i --shards n takes every n-th
group). Every group runs through up to three paths in one process, as the native engine's crash run did (exact,
batching and max_speed), so input shapes recur across paths (the fast path captures a shape's HIP graph on its
second use) and large captured graphs alternate with large eager batches:

  exact    every request alone through system_one (the released default path),
  batched  the group's questions coalesced into shared padded batches over the forward budget (as ixbatch.py),
  shared   every request through system_one(share_context=True), where the package has the switch.

Answers are not scored here: each record keeps the request's status per path and a digest of its answers. Before
each path of each group one trace line is printed (flushed), and every --log-every groups the fast path's graph
statistics (captures, replays, eager, evicted or full, cached graphs, retained output bytes). A GPU memory access
fault kills the process (exit 139); the last trace line names the group and path it was in. Writes
records.jsonl.gz and summary.json to $RC_RUN_DIR; the panel and the records stay private on the node.

    python3 -I -B evict_stress.py --package DIR [--runtime MIRROR/v2/release/runtime] --panel DIR \\
        --shard I --shards 6 --site /opt/decision-fla
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib
import importlib.util
import json
import os
import sys
import time
import traceback
from pathlib import Path

# The modules a release runtime ships under decision2/ (a --runtime overlay replaces these and nothing else).
OVERLAY = ("fast_kernels", "fast", "shared_ctx", "qwen", "api")


def canonical(value) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def load(
    package: Path,
    runtime: Path | None,
    device: str = "cuda:0",
    base_path: str | None = None,
    threads: int = 4,
):
    """The package's own runtime, with ``runtime``'s modules in place of its own (vendored model sources stay)."""
    sys.path.insert(0, str(package.resolve()))
    import decision2  # noqa: F401  (the package's runtime package)

    if runtime is not None:
        for name in OVERLAY:
            path = runtime / f"{name}.py"
            if not path.is_file():
                continue
            spec = importlib.util.spec_from_file_location(f"decision2.{name}", path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[f"decision2.{name}"] = module
            spec.loader.exec_module(module)
            setattr(sys.modules["decision2"], name, module)
    api = (
        sys.modules["decision2.api"]
        if runtime is not None
        else importlib.import_module("decision2.api")
    )
    return api.Decision2.from_pretrained(
        package, device=device, base_path=base_path, threads=threads
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", type=Path, required=True)
    ap.add_argument(
        "--runtime",
        type=Path,
        help="a mirror's v2/release/runtime, loaded over the package's modules",
    )
    ap.add_argument("--panel", type=Path, required=True)
    ap.add_argument("--paths", default="exact,batched,shared")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--base-path")
    ap.add_argument("--site", action="append", default=[])
    a = ap.parse_args()
    out_dir = Path(os.environ["RC_RUN_DIR"])
    for site in reversed(a.site):
        sys.path.insert(0, site)
    import torch

    if a.site:
        import causal_conv1d  # noqa: F401
        import fla  # noqa: F401

    rt = load(a.package, a.runtime, base_path=a.base_path)
    backend = rt.backend
    qwen = sys.modules["decision2.qwen"]
    fast_module = sys.modules.get("decision2.fast")
    from decision2._vendor.dev2model.decision_model import collate, encode
    from decision2._vendor.dev2model.infer import (
        _api_json_payload,
        product_answer,
        question_to_row,
    )

    graphs = getattr(backend.fast, "graphs", None) if backend.fast is not None else None

    def graph_stats():
        if graphs is None:
            return None
        return {
            **graphs.stats,
            "cached": len(graphs.graphs),
            "output_bytes": graphs.output_bytes,
        }

    paths = a.paths.split(",")
    if "shared" in paths and not hasattr(backend, "share_context"):
        paths.remove("shared")
    print(
        json.dumps(
            {
                "package": str(a.package),
                "runtime": str(a.runtime) if a.runtime else "package",
                "fast_py_sha256": (
                    hashlib.sha256(Path(fast_module.__file__).read_bytes()).hexdigest()
                    if fast_module
                    else None
                ),
                "graphs": (
                    None
                    if graphs is None
                    else {
                        "max_graphs": graphs.max_graphs,
                        "max_tokens": graphs.max_tokens,
                        "max_output_bytes": graphs.max_output_bytes,
                    }
                ),
                "batch_tokens": backend.batch_tokens,
                "paths": paths,
                "receipt": backend.fast.receipt() if backend.fast is not None else None,
            },
            default=str,
        )[:2000],
        flush=True,
    )

    pad_id = backend.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = backend.tokenizer.eos_token_id
    panel = json.loads((a.panel / "panel.json").read_text())
    rows = []
    for shard in panel["shards"]:
        with gzip.open(a.panel / shard["file"], "rt", encoding="utf-8") as stream:
            rows.extend(json.loads(line) for line in stream if line.strip())

    def digest(answers) -> str:
        return hashlib.sha256(canonical(answers).encode()).hexdigest()[:16]

    def exact(row, share=None):
        try:
            response = rt.system_one(
                state=row["state"], questions=row["questions"], share_context=share
            )
            errors = sum(
                1
                for v in response["answers"].values()
                if isinstance(v, dict) and "error" in v
            )
            return ["ok", errors, digest(response["answers"])]
        except Exception as exc:  # a request error, not a crash
            return [
                "error",
                f"{type(exc).__name__}: {str(exc)[:160]}",
                traceback.format_exc()[-400:],
            ]

    def plan(row):
        state, questions = row["state"], row["questions"]
        if (
            not isinstance(questions, dict)
            or not questions
            or not _api_json_payload(state)
        ):
            return None
        item = {"id": "request", "state": state}
        answers, jobs = {}, []
        for qid, question in questions.items():
            try:
                r = question_to_row(item, qid, question)
                enc = (backend.encode_fn or encode)(r, backend.tokenizer, backend.cap)
            except ValueError:
                answers[qid] = {"error": "invalid"}
                continue
            jobs.append((qid, r, enc))
        return answers, jobs

    def batched(block):
        try:
            plans = [plan(row) for row in block]
            flat = [
                (i, job) for i, p in enumerate(plans) if p is not None for job in p[1]
            ]
            lengths = [len(job[2]["ids"]) for _, job in flat]
            logits = [None] * len(flat)
            for batch_rows in (
                qwen.micro_batches(lengths, backend.batch_tokens) if flat else []
            ):
                batch = {
                    k: v.to(backend.device) if torch.is_tensor(v) else v
                    for k, v in collate(
                        [flat[r][1][2] for r in batch_rows], pad_id
                    ).items()
                }
                with torch.inference_mode(), torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16
                ):
                    if backend.fast is None:
                        output = backend.model(**batch)
                    else:
                        with backend.fast.forward([lengths[r] for r in batch_rows]):
                            output = backend.model(**batch)
                for r, values in zip(batch_rows, output):
                    logits[r] = values
            for (i, (qid, r, enc)), values in zip(flat, logits):
                try:
                    values = values[: len(enc["keys"])].float().cpu().tolist()
                    plans[i][0][qid] = product_answer(
                        r["task_type"],
                        enc["keys"],
                        values,
                        backend.temperatures[r["task_type"]],
                        [o["description"] for o in r["options"]],
                    )
                except ValueError:
                    plans[i][0][qid] = {"error": "invalid_model_output"}
            return [
                (
                    ["ok", 0, digest(p[0])]
                    if p is not None
                    else ["error", "invalid request", ""]
                )
                for p in plans
            ]
        except Exception as exc:
            message = f"{type(exc).__name__}: {str(exc)[:160]}"
            return [["error", message, traceback.format_exc()[-400:]] for _ in block]

    groups = [rows[i : i + a.batch] for i in range(0, len(rows), a.batch)]
    blocks = groups[a.shard :: a.shards]
    print(
        json.dumps(
            {"requests": len(rows), "groups": len(groups), "shard_groups": len(blocks)}
        ),
        flush=True,
    )
    # Kept open across groups and flushed every --log-every groups, so a crash keeps what was written.
    records = out_dir / "records.jsonl.gz"
    sink = gzip.open(records, "wt", encoding="utf-8")  # noqa: SIM115
    totals = {"requests": 0, "groups": 0, "errors": dict.fromkeys(paths, 0)}
    seconds = dict.fromkeys(paths, 0.0)
    t0 = time.time()
    for k, block in enumerate(blocks):
        by = {}
        for name in paths:
            print(
                json.dumps(
                    {
                        "trace": k,
                        "path": name,
                        "first": block[0]["_evaluation"]["run_id"][:70],
                    }
                ),
                flush=True,
            )
            torch.cuda.synchronize()
            t = time.perf_counter()
            if name == "exact":
                by[name] = [exact(row) for row in block]
            elif name == "shared":
                by[name] = [exact(row, share=True) for row in block]
            else:
                by[name] = batched(block)
            torch.cuda.synchronize()
            seconds[name] += time.perf_counter() - t
            totals["errors"][name] += sum(1 for r in by[name] if r[0] != "ok")
        for i, row in enumerate(block):
            sink.write(
                json.dumps(
                    {
                        "run_id": row["_evaluation"]["run_id"],
                        "q": len(row["questions"]),
                        "by": {name: results[i] for name, results in by.items()},
                    }
                )
                + "\n"
            )
        totals["requests"] += len(block)
        totals["groups"] = k + 1
        if k % a.log_every == 0 or k == len(blocks) - 1:
            sink.flush()
            print(
                json.dumps(
                    {
                        "block": k + 1,
                        "of": len(blocks),
                        "elapsed_s": round(time.time() - t0),
                        **totals,
                        "seconds": {n: round(s, 1) for n, s in seconds.items()},
                        "graphs": graph_stats(),
                        "allocated_gib": round(
                            torch.cuda.memory_allocated() / 2**30, 2
                        ),
                        "reserved_gib": round(torch.cuda.memory_reserved() / 2**30, 2),
                    }
                ),
                flush=True,
            )
    sink.close()
    summary = {
        **totals,
        "seconds": seconds,
        "elapsed_s": round(time.time() - t0),
        "graphs": graph_stats(),
    }
    with open(out_dir / "summary.json", "x") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({"finished": True, **summary}), flush=True)


if __name__ == "__main__":
    main()
