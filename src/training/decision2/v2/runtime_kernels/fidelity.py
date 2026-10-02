"""Element-level fidelity metrics of a kernel output against the reference output."""

from __future__ import annotations

from typing import Any


def compare(torch: Any, ref: Any, out: Any) -> dict[str, float]:
    """Bitwise element match, relative L2, max |difference| and the largest ULP distance.

    ULP distance is measured on the reference dtype's bit patterns (monotone integer
    mapping of BF16 / FP32), so 1 means "one representable value apart".
    """
    if ref.shape != out.shape:
        raise ValueError(f"shape mismatch {tuple(ref.shape)} vs {tuple(out.shape)}")
    out = out.to(ref.dtype)
    r, o = ref.float(), out.float()
    diff = (o - r).double()
    norm = r.double().norm().item()
    result = {
        "elements": ref.numel(),
        "match": (bits(torch, ref) == bits(torch, out)).double().mean().item(),
        "rel_l2": diff.norm().item() / norm if norm > 0 else float(diff.norm().item()),
        "max_abs": diff.abs().max().item() if ref.numel() else 0.0,
        "max_ulp": ulp_distance(torch, ref, out).max().item() if ref.numel() else 0,
    }
    finite = torch.isfinite(o).all().item()
    result["finite"] = bool(finite)
    return result


def bits(torch: Any, x: Any) -> Any:
    view = {
        torch.bfloat16: torch.int16,
        torch.float16: torch.int16,
        torch.float32: torch.int32,
    }[x.dtype]
    return x.contiguous().view(view)


def ulp_distance(torch: Any, a: Any, b: Any) -> Any:
    """|ordinal(a) - ordinal(b)| with the usual sign-magnitude to two's-complement mapping."""

    def ordinal(x: Any) -> Any:
        i = bits(torch, x).long()
        width = 16 if x.dtype in (torch.bfloat16, torch.float16) else 32
        sign = 1 << (width - 1)
        i = torch.where(i < 0, i + (1 << width), i)
        return torch.where(i >= sign, sign - i, i)

    return (ordinal(a) - ordinal(b)).abs()
