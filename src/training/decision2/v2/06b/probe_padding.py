"""Padded versus one-row micro-batches: same rows and weights, same loss and gradients?

Rebuilds one arm's trainer family and its first logical batches exactly as
`train.run` does, then for every multi-row micro-batch compares the trainer's
padded micro-batch with the same rows processed one at a time: per-row logits,
hidden states at real tokens, the summed loss and every parameter gradient.
Settings cross the compute precision (BF16 autocast as trained, or FP32) with
the attention kernel (default SDPA dispatch, forced SDPA math or efficient
kernel, eager). A synthetic check runs the SDPA kernels alone on the same
shapes against an FP32 math reference. Gold enters only the local loss.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import time
from pathlib import Path
from typing import Any

from . import train
from .common import (
    file_sha256,
    load_rights_clean,
    native_records,
    read_jsonl,
    schedule,
    write_json,
)


def model_of(family: Any) -> Any:
    return family.native.model if hasattr(family, "native") else family.model


def real_lengths(family: Any, records: list[dict[str, Any]]) -> list[int] | None:
    if hasattr(family, "packer"):
        return [family.packer.encode(r)["input_tokens"] for r in records]
    if hasattr(family, "encoded"):
        return [len(family.encoded(r)["ids"]) for r in records]
    return None


def logits_of(
    family: Any, records: list[dict[str, Any]], device: str, exact: bool
) -> Any:
    """The family's training forward; `exact` drops the causal collate's pad-to-8 tail."""
    import torch

    if hasattr(family, "native"):
        batch, _ = family.native.collator(
            [dict(r) for r in records], labeled=True, device=device
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return (
                family.api.forward_for_training(family.native, batch),
                batch["valid_candidates"],
            )
    if hasattr(family, "packer"):
        batch = family.packer.collate(
            [family.packer.encode(r) for r in records], device
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return family.model(batch), batch["valid_candidates"]
    batch = family.batch(records)
    if exact:
        if len(records) != 1:
            raise ValueError("exact mode is one row")
        n = len(family.encoded(records[0])["ids"])
        batch = {
            **batch,
            "input_ids": batch["input_ids"][:, :n],
            "attention_mask": batch["attention_mask"][:, :n],
        }
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return family.model(**batch), batch["candidate_mask"]


def loss_of(
    family: Any,
    records: list[dict[str, Any]],
    teacher: list[Any],
    device: str,
    exact: bool,
) -> Any:
    if not exact:
        return family.loss(records, teacher, device)[0]
    # Same loss as CausalQwenFamily.loss on the unpadded single row.
    original = family.batch

    def unpadded(rows: list[dict[str, Any]]) -> dict[str, Any]:
        batch = original(rows)
        n = len(family.encoded(rows[0])["ids"])
        return {
            **batch,
            "input_ids": batch["input_ids"][:, :n],
            "attention_mask": batch["attention_mask"][:, :n],
        }

    family.batch = unpadded
    try:
        return family.loss(records, teacher, device)[0]
    finally:
        family.batch = original


def grads(model: Any) -> dict[str, Any]:
    return {
        name: p.grad.detach().float().cpu().clone()
        for name, p in model.named_parameters()
        if p.grad is not None
    }


def group_of(name: str) -> str:
    if name.startswith(("head.", "ordinal.")) or ".head." in name:
        return "head"
    parts = name.split(".")
    if "layers" in parts:
        i = parts.index("layers")
        sub = parts[i + 2] if len(parts) > i + 2 else ""
        return f"layer{int(parts[i + 1]):02d}.{sub}"
    if "embed" in name:
        return "embed"
    return "other:" + ".".join(parts[-2:])


def compare(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """a vs reference b: relative L2 error, cosine and norm ratio per group and in total."""
    import torch

    groups: dict[str, list[float]] = {}
    for name in sorted(set(a) | set(b)):
        x = a.get(name)
        y = b.get(name)
        if x is None:
            x = torch.zeros_like(y)
        if y is None:
            y = torch.zeros_like(x)
        key = group_of(name)
        acc = groups.setdefault(key, [0.0, 0.0, 0.0, 0.0])
        acc[0] += float((x - y).double().square().sum())
        acc[1] += float(y.double().square().sum())
        acc[2] += float((x.double() * y.double()).sum())
        acc[3] += float(x.double().square().sum())

    def summary(keys: list[str]) -> dict[str, float]:
        d = sum(groups[k][0] for k in keys)
        yy = sum(groups[k][1] for k in keys)
        xy = sum(groups[k][2] for k in keys)
        xx = sum(groups[k][3] for k in keys)
        return {
            "rel_err": math.sqrt(d / yy) if yy else float("nan"),
            "cosine": xy / math.sqrt(xx * yy) if xx and yy else float("nan"),
            "norm_ratio": math.sqrt(xx / yy) if yy else float("nan"),
            "norm_ref": math.sqrt(yy),
        }

    keys = sorted(groups)
    per = {k: summary([k]) for k in keys}
    worst = sorted(
        per.items(), key=lambda kv: -kv[1]["rel_err"] if kv[1]["norm_ref"] > 0 else 0
    )[:6]
    return {
        "total": summary(keys),
        "head": summary([k for k in keys if k == "head"]) if "head" in groups else None,
        "backbone": summary([k for k in keys if k != "head"]),
        "worst_groups": {k: v for k, v in worst},
    }


@contextlib.contextmanager
def setting(family: Any, name: str):
    """precision-kernel, e.g. bf16-default, fp32-math, bf16-efficient, bf16-eager."""
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel

    precision, kernel = name.split("-", 1)
    stack = contextlib.ExitStack()
    original_autocast = torch.autocast
    backbone = getattr(model_of(family), "backbone", None)
    try:
        if precision == "fp32":

            def disabled(*args: Any, **kwargs: Any) -> Any:
                kwargs["enabled"] = False
                return original_autocast(*args, **kwargs)

            torch.autocast = disabled
        if kernel == "math":
            stack.enter_context(sdpa_kernel([SDPBackend.MATH]))
        elif kernel == "efficient":
            stack.enter_context(sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]))
        elif kernel == "eager":
            backbone.set_attn_implementation("eager")
        elif kernel != "default":
            raise ValueError(name)
        yield
    finally:
        torch.autocast = original_autocast
        stack.close()
        if kernel == "eager":
            backbone.set_attn_implementation("sdpa")


def hidden_capture(family: Any) -> tuple[dict[str, Any], Any]:
    store: dict[str, Any] = {}
    backbone = getattr(model_of(family), "backbone", None)
    if backbone is None:
        return store, None

    def hook(_module: Any, _inputs: Any, output: Any) -> None:
        store["hidden"] = (
            (
                output.last_hidden_state
                if hasattr(output, "last_hidden_state")
                else output[0]
            )
            .detach()
            .float()
        )

    return store, backbone.register_forward_hook(hook)


def forward_compare(
    family: Any, records: list[dict[str, Any]], device: str, exact: bool
) -> dict[str, Any]:
    import torch

    store, handle = hidden_capture(family)
    lengths = real_lengths(family, records)
    try:
        with torch.no_grad():
            padded, valid = logits_of(family, records, device, False)
            padded = padded.float()
            hidden_pad = store.get("hidden")
            rows = []
            for i, record in enumerate(records):
                one, one_valid = logits_of(family, [record], device, exact)
                one = one.float()
                width = int(one_valid[0].sum())
                if int(valid[i].sum()) != width:
                    raise AssertionError(
                        "candidate count differs between padded and one-row"
                    )
                a, b = padded[i, :width], one[0, :width]
                row = {
                    "logit_max_abs": float((a - b).abs().max()),
                    "prob_max_abs": float((a.softmax(-1) - b.softmax(-1)).abs().max()),
                }
                if hidden_pad is not None and lengths is not None:
                    n = lengths[i]
                    h1 = store["hidden"][0, :n]
                    hp = hidden_pad[i, :n]
                    row["hidden_rel"] = float((hp - h1).norm() / h1.norm())
                    row["hidden_max_abs"] = float((hp - h1).abs().max())
                rows.append(row)
    finally:
        if handle is not None:
            handle.remove()
    return {
        "logit_max_abs": max(r["logit_max_abs"] for r in rows),
        "prob_max_abs": max(r["prob_max_abs"] for r in rows),
        "hidden_rel_max": max((r.get("hidden_rel", 0.0) for r in rows), default=None),
        "hidden_max_abs": max(
            (r.get("hidden_max_abs", 0.0) for r in rows), default=None
        ),
        "rows": rows,
    }


def backward(
    family: Any, micro: list[dict[str, Any]], teacher: list[Any], device: str, mode: str
) -> tuple[float, dict[str, Any]]:
    model = model_of(family)
    model.zero_grad(set_to_none=True)
    if mode == "padded":
        loss = loss_of(family, micro, teacher, device, False)
        loss.backward()
        total = float(loss.detach())
    else:
        total = 0.0
        for record, t in zip(micro, teacher):
            loss = loss_of(family, [record], [t], device, mode == "exact")
            loss.backward()
            total += float(loss.detach())
    out = grads(model)
    model.zero_grad(set_to_none=True)
    return total, out


def kernel_check(
    lengths: list[int], heads: int, kv_heads: int, dim: int, seed: int = 0
) -> dict[str, Any]:
    """SDPA alone: right-padded causal batch vs unpadded rows vs an FP32 math reference."""
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel

    torch.manual_seed(seed)
    device = "cuda:0"
    width = math.ceil(max(lengths) / 8) * 8
    b = len(lengths)
    q = torch.randn(b, heads, width, dim, device=device)
    k = torch.randn(b, kv_heads, width, dim, device=device).repeat_interleave(
        heads // kv_heads, 1
    )
    v = torch.randn(b, kv_heads, width, dim, device=device).repeat_interleave(
        heads // kv_heads, 1
    )
    upstream = torch.randn(b, heads, width, dim, device=device)
    keep = torch.zeros(b, width, dtype=torch.bool, device=device)
    for i, n in enumerate(lengths):
        keep[i, :n] = True
        upstream[i, :, n:] = 0
    causal = torch.ones(width, width, dtype=torch.bool, device=device).tril()
    mask = causal[None, None] & keep[:, None, None, :]

    def run(
        dtype: Any, backends: list[Any] | None, batched: bool
    ) -> list[tuple[Any, ...]]:
        results = []
        ctx = sdpa_kernel(backends) if backends else contextlib.nullcontext()
        with ctx:
            if batched:
                xs = [t.detach().to(dtype).requires_grad_() for t in (q, k, v)]
                out = torch.nn.functional.scaled_dot_product_attention(
                    *xs, attn_mask=mask
                )
                (out.float() * upstream).sum().backward()
                for i, n in enumerate(lengths):
                    results.append(
                        tuple(
                            t[i, :, :n].float()
                            for t in (out.detach(), *(x.grad for x in xs))
                        )
                    )
            else:
                for i, n in enumerate(lengths):
                    xs = [
                        t[i : i + 1, :, :n].detach().to(dtype).requires_grad_()
                        for t in (q, k, v)
                    ]
                    out = torch.nn.functional.scaled_dot_product_attention(
                        *xs, is_causal=True
                    )
                    (out.float() * upstream[i : i + 1, :, :n]).sum().backward()
                    results.append(
                        tuple(
                            t[0].float() for t in (out.detach(), *(x.grad for x in xs))
                        )
                    )
        return results

    reference = run(torch.float32, [SDPBackend.MATH], False)

    def err(results: list[tuple[Any, ...]]) -> dict[str, float]:
        names = ("out", "dq", "dk", "dv")
        return {
            name: max(
                float((r[j] - ref[j]).norm() / ref[j].norm())
                for r, ref in zip(results, reference)
            )
            for j, name in enumerate(names)
        }

    report: dict[str, Any] = {
        "lengths": lengths,
        "heads": heads,
        "kv_heads": kv_heads,
        "dim": dim,
    }
    for label, backends in (
        ("default", None),
        ("math", [SDPBackend.MATH]),
        ("efficient", [SDPBackend.EFFICIENT_ATTENTION]),
        ("flash", [SDPBackend.FLASH_ATTENTION]),
    ):
        for batched in (True, False):
            key = f"bf16-{label}-{'padded' if batched else 'rows'}"
            try:
                report[key] = err(run(torch.bfloat16, backends, batched))
            except RuntimeError as exc:  # a backend may reject masks or shapes
                report[key] = {"error": str(exc).splitlines()[0][:200]}
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--steps", type=int, default=2, help="logical batches from the plan"
    )
    parser.add_argument("--micro-rows", type=int, default=8)
    parser.add_argument(
        "--settings",
        default="bf16-default,fp32-default,bf16-math,bf16-efficient,fp32-math",
    )
    parser.add_argument("--kernel-check", action="store_true")
    args = parser.parse_args()
    import torch

    started = time.monotonic()
    from . import kai8k

    kai8k.runtime_flags(torch)
    torch.use_deterministic_algorithms(True, warn_only=True)
    spec = train.load_spec(args.spec)
    device = "cuda:0"
    splits = load_rights_clean(spec["data"]["parent"])
    records = native_records(splits["train"], spec["data"]["converter_bundle"])
    if spec["family"] == "kai-native":
        family = train.KaiFamily(spec, device)
    elif spec["family"] == "qwen-causal":
        family = train.CausalQwenFamily(
            spec, device, {row["id"]: row for row in splits["train"]}
        )
    else:
        family = train.EncoderFamily(spec, device)
    teacher: list[Any] = [None] * len(records)
    if spec["teacher"] is not None:
        path = Path(spec["teacher"]["path"])
        if file_sha256(path) != spec["teacher"]["sha256"]:
            raise ValueError("Teacher file differs from its frozen hash")
        by_id = {row["source_row_id"]: row["probabilities"] for row in read_jsonl(path)}
        teacher = [by_id[r["source_row_id"]] for r in records]
    kinds = [r["question"]["type"].lower() for r in records]
    plan = schedule(
        [r["source_row_id"] for r in records], spec["logical_batch"], spec["seed"]
    )
    lengths = {}
    micros = []
    for rows in plan[: args.steps]:
        for i in rows:
            lengths[i] = family.length(records[i])
        for micro in train.micro_batches(
            rows, kinds, lengths, args.micro_rows, spec["micro_token_budget"]
        ):
            if len(micro) > 1:
                micros.append(micro)
    family.train_mode()
    exact_mode = spec["family"] == "qwen-causal"
    settings = [s for s in args.settings.split(",") if s]
    if spec["family"] != "qwen-causal":
        settings = [s for s in settings if not s.endswith("eager")]
    report: dict[str, Any] = {
        "spec": str(args.spec),
        "spec_sha256": file_sha256(args.spec),
        "family": spec["family"],
        "start": family.identity,
        "environment": train.environment(),
        "micro_batches": [
            {"rows": len(m), "kind": kinds[m[0]], "lengths": [lengths[i] for i in m]}
            for m in micros
        ],
        "settings": {},
    }
    reference: dict[int, dict[str, Any]] = {}
    # FP32 math one-row gradients are the reference every other path is measured against.
    order = ["fp32-math"] + [s for s in settings if s != "fp32-math"]
    for name in order:
        entry: list[dict[str, Any]] = []
        with setting(family, name):
            for index, micro in enumerate(micros):
                rows = [records[i] for i in micro]
                ts = [teacher[i] for i in micro]
                item: dict[str, Any] = {"micro": index}
                try:
                    item["forward_padded_vs_rows"] = forward_compare(
                        family, rows, device, False
                    )
                    loss_pad, g_pad = backward(family, rows, ts, device, "padded")
                    loss_one, g_one = backward(family, rows, ts, device, "rows")
                    item["loss_padded"], item["loss_rows"] = loss_pad, loss_one
                    item["loss_rel_diff"] = abs(loss_pad - loss_one) / abs(loss_one)
                    item["grad_padded_vs_rows"] = compare(g_pad, g_one)
                    if name == "fp32-math":
                        reference[index] = g_one
                    elif index in reference:
                        item["grad_padded_vs_ref"] = compare(g_pad, reference[index])[
                            "total"
                        ]
                        item["grad_rows_vs_ref"] = compare(g_one, reference[index])[
                            "total"
                        ]
                    if exact_mode:
                        # One row cut to its real length: no pad tail, so no attention mask at all.
                        item["forward_padded_vs_exact_rows"] = forward_compare(
                            family, rows, device, True
                        )
                        loss_x, g_x = backward(family, rows, ts, device, "exact")
                        item["loss_exact_rows"] = loss_x
                        item["grad_exact_rows_vs_rows"] = compare(g_x, g_one)["total"]
                        item["grad_padded_vs_exact_rows"] = compare(g_pad, g_x)["total"]
                        if index in reference and name != "fp32-math":
                            item["grad_exact_rows_vs_ref"] = compare(
                                g_x, reference[index]
                            )["total"]
                        del g_x
                    del g_pad, g_one
                except RuntimeError as exc:  # a forced kernel may reject one shape
                    item["error"] = str(exc).splitlines()[0][:300]
                entry.append(item)
                print(
                    json.dumps(
                        {
                            "setting": name,
                            "micro": index,
                            "rows": len(micro),
                            "error": item.get("error"),
                            "logit_max_abs": item.get("forward_padded_vs_rows", {}).get(
                                "logit_max_abs"
                            ),
                            "hidden_rel_max": item.get(
                                "forward_padded_vs_rows", {}
                            ).get("hidden_rel_max"),
                            "loss_rel_diff": item.get("loss_rel_diff"),
                            "grad_total": item.get("grad_padded_vs_rows", {}).get(
                                "total"
                            ),
                            "grad_head": item.get("grad_padded_vs_rows", {}).get(
                                "head"
                            ),
                            "grad_padded_vs_ref": item.get("grad_padded_vs_ref"),
                            "grad_rows_vs_ref": item.get("grad_rows_vs_ref"),
                            "grad_exact_rows_vs_rows": item.get(
                                "grad_exact_rows_vs_rows"
                            ),
                        }
                    ),
                    flush=True,
                )
        report["settings"][name] = entry
        write_json(args.output, report)
    if args.kernel_check:
        config = getattr(getattr(model_of(family), "backbone", None), "config", None)
        heads = getattr(config, "num_attention_heads", 16)
        kv = getattr(config, "num_key_value_heads", heads) or heads
        dim = getattr(config, "head_dim", None) or config.hidden_size // heads
        biggest = max(micros, key=lambda m: len(m) * max(lengths[i] for i in m))
        report["kernel_check"] = kernel_check(
            [lengths[i] for i in biggest], heads, kv, dim
        )
        print(json.dumps({"kernel_check": report["kernel_check"]}), flush=True)
    report["elapsed_seconds"] = time.monotonic() - started
    write_json(args.output, report)


if __name__ == "__main__":
    main()
