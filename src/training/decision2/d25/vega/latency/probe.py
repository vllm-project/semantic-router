"""Latency probes of the d3 runtime on one GPU, one request at a time as the kit runs it.

    python -m d25.vega.latency.probe env --out env.json
    python -m d25.vega.latency.probe time --package PKG --runtime DIR --rows latency-760.jsonl.gz \
        --design latency-760.jsonl.gz.json --out runs/x [--reference-runtime DIR]
    python -m d25.vega.latency.probe profile --package PKG --runtime DIR --rows ... --run-ids ID ... --out prof/x

``--runtime`` is a directory holding ``d3_runtime.py`` (and ``d3_format.py``); the weights always come from
``--package``. ``time`` writes ``results.jsonl`` in the kit's shape (run_id, catalog_id, status, response,
total_wall_ms), so ``d25.vega.release.compare`` reads it, and ``summary.json`` (latency, phases, tokens). The
time of a request is the kit's: device sync, the engine call (render, tokenize, forward passes, readout,
answers), device sync, response validation. With ``--reference-runtime`` a second copy of the model runs every
request through that runtime as well, in alternating order, timed the same way into ``reference/``, and the
answers are compared (A/B in one process on one GPU under the same conditions).
"""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any


def read_rows(path: str | Path) -> list[dict[str, Any]]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def load_runtime(directory: str | Path, name: str):
    """Import ``d3_runtime.py`` from ``directory`` under its own module name (two runtimes can share a process)."""
    directory = str(Path(directory).resolve())
    if directory not in sys.path:
        sys.path.insert(0, directory)
    spec = importlib.util.spec_from_file_location(
        name, Path(directory) / "d3_runtime.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def percentile(values: list[float], q: float) -> float:
    """Linear interpolation between closest ranks (compare.py's definition)."""
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q / 100
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    return {
        "n": len(values),
        "median_ms": round(statistics.median(values), 1),
        "mean_ms": round(statistics.fmean(values), 1),
        "p80_ms": round(percentile(values, 80), 1),
        "p95_ms": round(percentile(values, 95), 1),
        "max_ms": round(max(values), 1),
    }


def validator():
    try:
        from decision_index.engines import (
            validate,
        )  # the kit's own check, inside its timing window

        return validate
    except Exception:  # noqa: BLE001 - the probe also runs without the kit
        return lambda questions, response: None


def engine_call(model, state, questions, phases: dict[str, float]):
    """What ``d3_engine.D3Engine.__call__`` does, with the phase times recorded."""
    t0 = time.perf_counter()
    prepared = model.prepare(state, questions)
    if prepared.errors:
        raise ValueError(f"invalid or over-limit questions: {sorted(prepared.errors)}")
    t1 = time.perf_counter()
    probabilities, tokens = model.run(prepared)
    t2 = time.perf_counter()
    response = model.respond(prepared, probabilities, tokens)
    failed = {k: a["message"] for k, a in response["answers"].items() if "error" in a}
    if failed:
        raise ValueError(f"invalid model output: {failed}")
    t3 = time.perf_counter()
    phases.update(
        prepare_ms=(t1 - t0) * 1000,
        run_ms=(t2 - t1) * 1000,
        respond_ms=(t3 - t2) * 1000,
    )
    return response


def timed(model, row: dict[str, Any], validate) -> dict[str, Any]:
    e = row["_evaluation"]
    result: dict[str, Any] = {"run_id": e["run_id"], "catalog_id": e.get("catalog_id")}
    phases: dict[str, float] = {}
    t = time.perf_counter()
    model.synchronize()
    try:
        response = engine_call(model, row["state"], row["questions"], phases)
        model.synchronize()
        validate(row["questions"], response)
        result.update(status="ok", response=response)
    except Exception as exc:  # noqa: BLE001 - recorded like the kit records it
        result.update(status="error", error=f"{type(exc).__name__}: {exc}")
    result["total_wall_ms"] = (time.perf_counter() - t) * 1000
    result.update({k: round(v, 3) for k, v in phases.items()})
    if result["status"] == "ok":
        result["input_tokens"] = response["usage"]["input_tokens"]
        result["questions"] = len(response["answers"])
    return result


def summarize(results: list[dict[str, Any]], warm: set[str]) -> dict[str, Any]:
    timed_rows = [r for r in results if r["run_id"] not in warm]
    ok = [r for r in timed_rows if r["status"] == "ok"]
    out = {
        "timed_rows": len(timed_rows),
        "ok": len(ok),
        "errors": len(timed_rows) - len(ok),
        "latency": stats([r["total_wall_ms"] for r in ok]),
    }
    for phase in ("prepare_ms", "run_ms", "respond_ms"):
        out[phase] = stats([r[phase] for r in ok if phase in r])
    tokens = [r["input_tokens"] for r in ok]
    if tokens:
        out["input_tokens"] = {
            "median": statistics.median(tokens),
            "p80": percentile(tokens, 80),
            "max": max(tokens),
        }
        out["questions"] = {"median": statistics.median([r["questions"] for r in ok])}
    return out


def load_model(rt, package: str, device: str, **options):
    model = rt.D3.from_pretrained(package, device=device, verify="none", **options)
    return model


def cmd_time(args) -> int:
    rows = read_rows(args.rows)
    warm = (
        set(json.loads(Path(args.design).read_text())["warmup_run_ids"])
        if args.design
        else set()
    )
    if args.limit:
        rows = rows[: args.limit]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    validate = validator()
    rt = load_runtime(args.runtime, "d3_runtime_probe")
    model = load_model(rt, args.package, args.device)
    report: dict[str, Any] = {
        "runtime": str(args.runtime),
        "package": str(args.package),
    }
    report["warmup_seconds"] = round(model.warmup(), 1) if not args.no_warmup else None
    ref = None
    if args.reference_runtime:
        rt_ref = load_runtime(args.reference_runtime, "d3_runtime_reference")
        ref = load_model(rt_ref, args.package, args.device)
        report["reference_runtime"] = str(args.reference_runtime)
        report["reference_warmup_seconds"] = (
            round(ref.warmup(), 1) if not args.no_warmup else None
        )
    results, ref_results = [], []
    started = time.time()
    with open(out / "results.jsonl", "w", encoding="utf-8") as log:
        for i, row in enumerate(rows):
            order = (
                [("new", model), ("ref", ref)]
                if i % 2 == 0
                else [("ref", ref), ("new", model)]
            )
            for kind, m in order:
                if m is None:
                    continue
                r = timed(m, row, validate)
                if kind == "new":
                    results.append(r)
                    log.write(json.dumps(r, ensure_ascii=False) + "\n")
                else:
                    ref_results.append(r)
            if (i + 1) % 50 == 0:
                print(
                    json.dumps(
                        {"done": i + 1, "elapsed_s": round(time.time() - started)}
                    ),
                    flush=True,
                )
    report["summary"] = summarize(results, warm)
    if hasattr(model, "fast_report"):
        report["fast"] = model.fast_report()
    if ref is not None:
        (out / "reference").mkdir(exist_ok=True)
        with open(out / "reference" / "results.jsonl", "w", encoding="utf-8") as log:
            for r in ref_results:
                log.write(json.dumps(r, ensure_ascii=False) + "\n")
        report["reference_summary"] = summarize(ref_results, warm)
        from d25.vega.release.compare import answers

        report["answers_vs_reference"] = answers(
            {r["run_id"]: r for r in results}, {r["run_id"]: r for r in ref_results}
        )
    report["runtime_info"] = model.runtime_info()
    (out / "summary.json").write_text(json.dumps(report, indent=1) + "\n")
    print(
        json.dumps(
            {k: v for k, v in report.items() if k != "answers_vs_reference"}, indent=1
        )
    )
    if "answers_vs_reference" in report:
        a = report["answers_vs_reference"]
        print(json.dumps({k: a[k] for k in a if k != "flips"}, indent=1))
    return 0


def cmd_profile(args) -> int:
    import torch
    from torch.profiler import ProfilerActivity, profile

    rows = {r["_evaluation"]["run_id"]: r for r in read_rows(args.rows)}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rt = load_runtime(args.runtime, "d3_runtime_probe")
    model = load_model(rt, args.package, args.device)
    if not args.no_warmup:
        model.warmup()
    validate = validator()
    report: dict[str, Any] = {"requests": {}}
    run_ids = list(args.run_ids or [])
    if args.pick:
        sized = []
        for rid, row in rows.items():
            prepared = model.prepare(row["state"], row["questions"])
            tokens = sum(len(s) for s in prepared.sequences.values())
            sized.append((tokens, rid))
        sized.sort()
        for q in args.pick:
            run_ids.append(sized[min(len(sized) - 1, int(len(sized) * q / 100))][1])
    for rid in run_ids:
        row = rows[rid]
        for _ in range(args.warm):
            timed(model, row, validate)
        walls = [timed(model, row, validate) for _ in range(args.repeat)]
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(args.repeat):
                timed(model, row, validate)
        events = prof.key_averages()
        gpu_us = sum(e.self_device_time_total for e in events) / args.repeat
        table = events.table(sort_by="self_device_time_total", row_limit=args.top)
        cpu_table = events.table(sort_by="self_cpu_time_total", row_limit=25)
        (out / f"{rid.replace('/', '_')}.txt").write_text(
            table + "\n\n" + cpu_table + "\n"
        )
        kernels = (
            sum(e.count for e in events if e.device_type.name == "CUDA") / args.repeat
        )
        report["requests"][rid] = {
            "questions": walls[0].get("questions"),
            "input_tokens": walls[0].get("input_tokens"),
            "wall_ms": [round(w["total_wall_ms"], 1) for w in walls],
            "phases_ms": {
                k: walls[-1].get(k) for k in ("prepare_ms", "run_ms", "respond_ms")
            },
            "gpu_busy_ms_profiled": round(gpu_us / 1000, 1),
            "device_kernels_per_request": round(kernels),
        }
        print(rid, json.dumps(report["requests"][rid]), flush=True)
    if hasattr(model, "fast_report"):
        report["fast"] = model.fast_report()
    report["runtime_info"] = model.runtime_info()
    report["kernels"] = model.kernels
    (out / "profile.json").write_text(json.dumps(report, indent=1) + "\n")
    torch.cuda.synchronize()
    return 0


def image_requests(pack: Path, kind: str, n: int, max_pixels: int) -> list[tuple]:
    """(id, state, questions, data URLs): ``one`` = one image of at least 1.6 MP per request (the image's own
    questions, rows in sha256 order of their id, as ``d25.omni.runtime.latency``); ``suite`` = natural rows
    spread over the families."""
    from PIL import Image

    from d25.omni.runtime.common import data_url, sha_key, stratified

    rows = read_rows(pack / "rows.jsonl.gz")
    if kind == "suite":
        chosen = stratified(rows, n)
    else:
        chosen = []
        for row in sorted(rows, key=lambda r: sha_key(r["id"])):
            if len(row["images"]) != 1:
                continue
            with Image.open(pack / row["images"][0]) as image:
                width, height = image.size
            if width * height >= max_pixels and max(width, height) <= 20 * min(
                width, height
            ):
                chosen.append(row)
            if len(chosen) >= n:
                break
    return [
        (
            f"{r['id']}#{i}",
            r.get("state"),
            r["questions"],
            [data_url(pack / p) for p in r["images"]],
        )
        for i, r in ((i, chosen[i % len(chosen)]) for i in range(n))
    ]


def cmd_image(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rt = load_runtime(args.runtime, "d3_runtime_probe")
    model = rt.D3.from_pretrained(
        args.package, device=args.device, verify="none", max_length=1 << 20
    )
    requests = image_requests(
        Path(args.pack), args.kind, args.warmup + args.n, rt.IMAGE_MAX_PIXELS
    )
    models = [("new", model)]
    if args.reference_runtime:
        rt_ref = load_runtime(args.reference_runtime, "d3_runtime_reference")
        models.append(
            (
                "ref",
                rt_ref.D3.from_pretrained(
                    args.package, device=args.device, verify="none", max_length=1 << 20
                ),
            )
        )
    report: dict[str, Any] = {
        "kind": args.kind,
        "requests": len(requests) - args.warmup,
    }
    for name, m in models:
        report[f"{name}_warmup_seconds"] = round(m.warmup(), 1)
    times = {name: [] for name, _ in models}
    phases = {name: {"prepare_ms": [], "run_ms": []} for name, _ in models}
    probs = {name: {} for name, _ in models}
    tokens = []
    for i, (rid, state, questions, images) in enumerate(requests):
        order = models if i % 2 == 0 else models[::-1]
        for name, m in order:
            m.synchronize()
            t0 = time.perf_counter()
            prepared = m.prepare(state, questions, images)
            t1 = time.perf_counter()
            p, used = m.run(prepared)
            m.respond(prepared, p, used)
            m.synchronize()
            t2 = time.perf_counter()
            if i >= args.warmup:
                times[name].append((t2 - t0) * 1000)
                phases[name]["prepare_ms"].append((t1 - t0) * 1000)
                phases[name]["run_ms"].append((t2 - t1) * 1000)
                probs[name][rid] = p
                if name == "new":
                    tokens.append(used)
    for name, _ in models:
        if not times[name]:
            continue
        report[name] = {
            **stats(times[name]),
            **{k: round(statistics.median(v), 1) for k, v in phases[name].items()},
        }
    report["input_tokens_median"] = statistics.median(tokens)
    report["fast"] = model.fast_report() if hasattr(model, "fast_report") else None
    if args.reference_runtime:
        exact = flips = questions_n = 0
        worst = 0.0
        for rid, ans in probs["ref"].items():
            for key, want in ans.items():
                have = probs["new"][rid][key]
                questions_n += 1
                exact += list(have) == list(want)
                worst = max(worst, max(abs(a - b) for a, b in zip(have, want)))
                flips += max(range(len(have)), key=have.__getitem__) != max(
                    range(len(want)), key=want.__getitem__
                )
        report["parity"] = {
            "questions": questions_n,
            "exact": exact,
            "argmax_changes": flips,
            "max_abs_dp": worst,
        }
    (out / f"image-{args.kind}.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report, indent=1))
    return 0


def cmd_env(args) -> int:
    import torch

    info: dict[str, Any] = {
        "torch": torch.__version__,
        "hip": getattr(torch.version, "hip", None),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "arch": (
            torch.cuda.get_device_properties(0).gcnArchName
            if torch.cuda.is_available()
            else None
        ),
    }
    for name in (
        "transformers",
        "triton",
        "fla",
        "flash_attn",
        "aiter",
        "causal_conv1d",
        "torchvision",
    ):
        try:
            module = __import__(name)
            info[name] = getattr(module, "__version__", "present")
        except Exception as exc:  # noqa: BLE001
            info[name] = f"missing ({type(exc).__name__}: {str(exc)[:120]})"
    try:
        from flash_attn import flash_attn_varlen_func  # noqa: F401

        info["flash_attn_varlen"] = "importable"
    except Exception as exc:  # noqa: BLE001
        info["flash_attn_varlen"] = f"missing ({type(exc).__name__})"
    # Which SDPA backends serve the d3 full-attention shape (head dim 256, boolean key mask) on this GPU.
    from torch.nn.attention import SDPBackend, sdpa_kernel

    dev = "cuda:0"
    q = torch.randn(2, 24, 300, 256, device=dev, dtype=torch.bfloat16)
    k = torch.randn(2, 4, 300, 256, device=dev, dtype=torch.bfloat16)
    mask = torch.ones(2, 1, 1, 300, device=dev, dtype=torch.bool)
    mask[1, ..., :40] = False
    backends = {}
    for backend in (
        SDPBackend.FLASH_ATTENTION,
        SDPBackend.EFFICIENT_ATTENTION,
        SDPBackend.MATH,
    ):
        for label, m in (("mask", mask), ("nomask", None)):
            try:
                with sdpa_kernel([backend]):
                    torch.nn.functional.scaled_dot_product_attention(
                        q, k, k, attn_mask=m, enable_gqa=True
                    )
                backends[f"{backend.name}/{label}"] = "ok"
            except Exception as exc:  # noqa: BLE001
                backends[f"{backend.name}/{label}"] = f"no ({str(exc)[:100]})"
    info["sdpa_hd256"] = backends
    info["env"] = {
        k: os.environ.get(k)
        for k in ("PYTORCH_TUNABLEOP_ENABLED", "HIP_VISIBLE_DEVICES")
    }
    text = json.dumps(info, indent=1)
    if args.out:
        Path(args.out).write_text(text + "\n")
    print(text)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    env = sub.add_parser("env")
    env.add_argument("--out")
    for name in ("time", "profile"):
        p = sub.add_parser(name)
        p.add_argument("--package", required=True)
        p.add_argument("--runtime", required=True)
        p.add_argument("--rows", required=True)
        p.add_argument("--device", default="cuda:0")
        p.add_argument("--no-warmup", action="store_true")
        p.add_argument("--out", required=True)
    t = sub.choices["time"]
    t.add_argument("--design")
    t.add_argument("--limit", type=int)
    t.add_argument("--reference-runtime")
    pr = sub.choices["profile"]
    pr.add_argument("--run-ids", nargs="*")
    pr.add_argument(
        "--pick",
        nargs="*",
        type=float,
        help="input-token quantiles (0-100) of requests to profile",
    )
    pr.add_argument("--warm", type=int, default=2)
    pr.add_argument("--repeat", type=int, default=3)
    pr.add_argument("--top", type=int, default=40)
    im = sub.add_parser("image")
    im.add_argument("--package", required=True)
    im.add_argument("--runtime", required=True)
    im.add_argument("--reference-runtime")
    im.add_argument("--pack", required=True)
    im.add_argument("--kind", choices=("one", "suite"), default="one")
    im.add_argument("--n", type=int, default=100)
    im.add_argument("--warmup", type=int, default=10)
    im.add_argument("--device", default="cuda:0")
    im.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    return {
        "env": cmd_env,
        "time": cmd_time,
        "profile": cmd_profile,
        "image": cmd_image,
    }[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
