"""Which SDPA backends serve the full-attention shapes of a 27B Qwen3.8 code-readout model, and how fast.

    python -m d25.vega.release.sdpa_probe --out sdpa.json

For batch sizes, lengths and both masks the model uses (``noncausal``: a [B, 1, 1, L] boolean padding mask;
``causal``: a [B, 1, L, L] boolean causal + padding mask), each backend is forced in turn; the report gives
the mean milliseconds per call or the reason the backend refused, plus what the default selection costs.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

HEADS, HEAD_DIM = 24, 256


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--batches", default="1,8")
    ap.add_argument("--lengths", default="512,2048,8192")
    ap.add_argument("--repeat", type=int, default=5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    import torch
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel

    backends = {
        "flash": SDPBackend.FLASH_ATTENTION,
        "efficient": SDPBackend.EFFICIENT_ATTENTION,
        "math": SDPBackend.MATH,
        "cudnn": SDPBackend.CUDNN_ATTENTION,
    }
    rows = []
    for batch in map(int, args.batches.split(",")):
        for length in map(int, args.lengths.split(",")):
            q, k, v = (
                torch.randn(
                    batch, HEADS, length, HEAD_DIM, device="cuda", dtype=torch.bfloat16
                )
                for _ in range(3)
            )
            pad = torch.ones(batch, length, dtype=torch.bool, device="cuda")
            pad[0, : length // 4] = False
            masks = {
                "noncausal": pad[:, None, None, :],
                "causal": torch.tril(
                    torch.ones(length, length, dtype=torch.bool, device="cuda")
                )[None, None]
                & pad[:, None, None, :],
            }
            for mask_name, mask in masks.items():
                choices = {
                    "default": list(backends.values()),
                    "runtime": [
                        backends["flash"],
                        backends["efficient"],
                        backends["math"],
                    ],
                    **{name: [backend] for name, backend in backends.items()},
                }
                for name, allowed in choices.items():
                    row = {
                        "batch": batch,
                        "length": length,
                        "mask": mask_name,
                        "backend": name,
                    }
                    try:
                        context = sdpa_kernel(allowed)
                        with context:
                            F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
                            torch.cuda.synchronize()
                            started = time.perf_counter()
                            for _ in range(args.repeat):
                                F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
                            torch.cuda.synchronize()
                        row["ms"] = round(
                            (time.perf_counter() - started) * 1000 / args.repeat, 3
                        )
                    except Exception as exc:  # noqa: BLE001 - a refusal is the result
                        row["error"] = (
                            f"{type(exc).__name__}: {str(exc).splitlines()[0][:160]}"
                        )
                    torch.cuda.empty_cache()
                    rows.append(row)
            del q, k, v
    report = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "flash_sdp_enabled": torch.backends.cuda.flash_sdp_enabled(),
        "cudnn_sdp_enabled": torch.backends.cuda.cudnn_sdp_enabled(),
        "rows": rows,
    }
    with open(args.out, "w") as stream:
        json.dump(report, stream, indent=1)
    for row in rows:
        print(json.dumps(row))
    return 0


if __name__ == "__main__":
    sys.exit(main())
