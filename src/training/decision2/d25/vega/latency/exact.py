"""Exactness of the d3 fast path against the plain runtime path on real prompts (one process, one model).

    python -m d25.vega.latency.exact --package PKG --runtime DIR --rows latency-760.jsonl.gz --requests 40 \
        --out exact.json

For the batches of the chosen requests (spread over input sizes) it compares, bit for bit, the last-token
hidden states and the answer probabilities of: ``plain`` (the v3.0.2 forward through the backbone),
``layers`` (the synchronization-free pass through the eager decoder layers), ``fused`` (the fused kernels)
and ``graph`` (the HIP graph of the bucketed shape, right-padded), each against ``plain``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from d25.vega.latency.probe import load_runtime, read_rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--package", required=True)
    ap.add_argument("--runtime", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--requests", type=int, default=40)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    rt = load_runtime(args.runtime, "d3_runtime_exact")
    model = rt.D3.from_pretrained(args.package, device=args.device, verify="none")
    torch = model.torch
    fast = model.fast
    if fast is None:
        raise SystemExit(f"fast path off: {model.fast_skipped}")
    rows = read_rows(args.rows)
    sized = []
    for row in rows:
        prepared = model.prepare(row["state"], row["questions"])
        keys = prepared.runnable
        sized.append((sum(len(prepared.sequences[k]) for k in keys), row, prepared))
    sized.sort(key=lambda x: x[0])
    step = max(1, len(sized) // args.requests)
    chosen = sized[::step][: args.requests]
    fused = fast.fused
    report = {"fast": fast.report(), "batches": [], "summary": {}}
    totals = {
        k: {
            "exact_hidden": 0,
            "exact_probs": 0,
            "argmax_changes": 0,
            "max_abs_dp": 0.0,
            "max_abs_dh": 0.0,
        }
        for k in ("layers", "fused", "graph")
    }
    n = 0
    started = time.time()
    for tokens, row, prepared in chosen:
        keys = prepared.runnable
        for start in range(0, len(keys), model.batch_size):
            chunk = keys[start : start + model.batch_size]
            seqs = [prepared.sequences[k] for k in chunk]
            counts = [len(prepared.questions[k].keys) for k in chunk]
            width = max(map(len, seqs))
            ids = torch.full((len(seqs), width), model.pad_id, dtype=torch.long)
            mask = torch.zeros((len(seqs), width), dtype=torch.long)
            for i, s in enumerate(seqs):
                ids[i, width - len(s) :] = torch.as_tensor(s)
                mask[i, width - len(s) :] = 1
            ids, mask = ids.to(model.device), mask.to(model.device)
            padded = any(len(s) != width for s in seqs)
            with torch.inference_mode():
                h_plain = model.backbone(
                    input_ids=ids, attention_mask=mask, use_cache=False
                ).last_hidden_state[:, -1]
                fast.fused = None
                h_layers = fast.hidden(ids, mask, mask if padded else None)[:, -1]
                fast.fused = fused
                h_fused = (
                    fast.hidden(ids, mask, mask if padded else None)[:, -1]
                    if fused is not None
                    else None
                )
            saved = model.fast
            model.fast = None
            p_plain = model.probabilities(seqs, counts)
            model.fast = saved
            size = fast.bucket_of(len(seqs), width)
            graph_ok = size is not None and fast.capture(len(seqs), size)
            outs = {"layers": (h_layers, None), "fused": (h_fused, None)}
            if fused is not None:
                fast.fused = fused
            p_fast = fast.probabilities(seqs, counts) if graph_ok else None
            outs["graph"] = (None, p_fast)
            entry = {
                "tokens": tokens,
                "rows": len(seqs),
                "width": width,
                "bucket": size,
                "padded": padded,
            }
            for name, (h, p) in outs.items():
                t = totals[name]
                if h is not None:
                    same = bool(torch.equal(h, h_plain))
                    t["exact_hidden"] += same
                    dh = float((h.float() - h_plain.float()).abs().max())
                    t["max_abs_dh"] = max(t["max_abs_dh"], dh)
                    entry[f"{name}_hidden_exact"] = same
                    if name == "fused":
                        with torch.inference_mode():
                            p = (
                                fast.probabilities_of(
                                    h, torch.as_tensor(counts, device=model.device)
                                )
                                .cpu()
                                .tolist()
                            )
                        p = [x[:c] for x, c in zip(p, counts)]
                if p is None:
                    continue
                exact = all(a == b for a, b in zip(p, p_plain))
                t["exact_probs"] += exact
                dp = max(
                    max(abs(x - y) for x, y in zip(a, b)) for a, b in zip(p, p_plain)
                )
                t["max_abs_dp"] = max(t["max_abs_dp"], dp)
                flips = sum(
                    max(range(len(a)), key=a.__getitem__)
                    != max(range(len(b)), key=b.__getitem__)
                    for a, b in zip(p, p_plain)
                )
                t["argmax_changes"] += flips
                entry[f"{name}_dp"] = dp
            n += 1
            report["batches"].append(entry)
    report["summary"] = {
        "batches": n,
        "seconds": round(time.time() - started, 1),
        **totals,
    }
    Path(args.out).write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report["summary"], indent=1))
    print(json.dumps(fast.report()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
