#!/usr/bin/env python3
"""
Step 3 of probe-2026-05-25-image-drift-isolation.

Loads the three 384-d embeddings produced by step1 (Python) and step2 (Go test),
reports cosine similarities, L2 norms, and max abs diffs.

Decision logic:
  - cos(Python, Candle-PIL-pipe)   = model-forward drift only (same preproc)
  - cos(Candle-PIL, Candle-Go-pipe) = preprocessing drift only (same Rust forward)
  - cos(Python, Candle-Go-pipe)    = full pipeline drift (the on-record 0.992)
"""

import json
import sys
from pathlib import Path

import numpy as np

PROBE_DIR = Path(__file__).parent.resolve()


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def maxdiff(a, b):
    return float(np.max(np.abs(a - b)))


def meandiff(a, b):
    return float(np.mean(np.abs(a - b)))


def main():
    p_python = PROBE_DIR / "embedding_python.npy"
    p_pilpipe = PROBE_DIR / "embedding_candle_pilpipe.bin"
    p_gopipe = PROBE_DIR / "embedding_candle_gopipe.bin"

    missing = [p for p in (p_python, p_pilpipe, p_gopipe) if not p.exists()]
    if missing:
        print("ERROR: missing artifacts. Run step1 + step2 first.", file=sys.stderr)
        for m in missing:
            print(f"  missing: {m}", file=sys.stderr)
        sys.exit(1)

    e_python = np.load(p_python).astype(np.float32)
    e_pilpipe = np.fromfile(p_pilpipe, dtype=np.float32)
    e_gopipe = np.fromfile(p_gopipe, dtype=np.float32)

    for name, arr in [
        ("Python (PIL preproc + torch forward)", e_python),
        ("Candle PIL-pipe (PIL preproc + Rust forward)", e_pilpipe),
        ("Candle Go-pipe  (Go bilinear + Rust forward)", e_gopipe),
    ]:
        if arr.shape != (384,):
            print(f"ERROR: {name} shape {arr.shape} != (384,)", file=sys.stderr)
            sys.exit(1)

    norms = {
        "python": float(np.linalg.norm(e_python)),
        "pilpipe": float(np.linalg.norm(e_pilpipe)),
        "gopipe": float(np.linalg.norm(e_gopipe)),
    }

    cos_python_pil = cos(e_python, e_pilpipe)
    cos_python_go = cos(e_python, e_gopipe)
    cos_pil_go = cos(e_pilpipe, e_gopipe)

    md_python_pil = maxdiff(e_python, e_pilpipe)
    md_python_go = maxdiff(e_python, e_gopipe)
    md_pil_go = maxdiff(e_pilpipe, e_gopipe)

    mn_python_pil = meandiff(e_python, e_pilpipe)
    mn_python_go = meandiff(e_python, e_gopipe)
    mn_pil_go = meandiff(e_pilpipe, e_gopipe)

    print()
    print("=" * 78)
    print("Image-drift isolation report - 2026-05-25 - inrule_identifier_passport.jpg")
    print("=" * 78)
    print()
    print("L2 norms (all three should be ~1.0 if L2-normalize is applied at end):")
    for k, v in norms.items():
        print(f"  {k:8s}: {v:.6f}")
    print()
    print("Pairwise comparisons:")
    print()
    print(
        f"  [Model-forward drift]   Python  vs  Candle-PIL : cos={cos_python_pil:.6f}  max_abs={md_python_pil:.6f}  mean_abs={mn_python_pil:.6f}"
    )
    print(
        f"  [Full pipeline drift]   Python  vs  Candle-Go  : cos={cos_python_go:.6f}  max_abs={md_python_go:.6f}  mean_abs={mn_python_go:.6f}"
    )
    print(
        f"  [Preprocessing drift]   Candle-PIL vs Candle-Go: cos={cos_pil_go:.6f}  max_abs={md_pil_go:.6f}  mean_abs={mn_pil_go:.6f}"
    )
    print()

    # Verdict heuristic
    print("Verdict:")
    if cos_python_pil > 0.9995 and cos_pil_go < cos_python_pil - 0.003:
        print("  >>> PREPROCESSING IS THE DOMINANT DRIFT SOURCE.")
        print("      Model forward (Python vs Candle-PIL) cosine is essentially 1.0,")
        print("      so the Rust port of the mmes vision forward is clean.")
        print(
            "      The residual ~0.8% lives in Go's 4-tap bilinear vs PIL bicubic+antialias."
        )
        print(
            "      Fix path: replace decodeAndResizeImage with a PIL-equivalent resize."
        )
    elif cos_python_pil < 0.995 and cos_pil_go > 0.999:
        print("  >>> MODEL-FORWARD IS THE DOMINANT DRIFT SOURCE.")
        print("      Preprocessing is fine; the Rust port of mmes has a numerical bug.")
        print("      Next step: per-stage activation comparison to localize.")
    elif cos_python_pil > 0.999 and cos_pil_go > 0.999:
        print("  >>> NEGLIGIBLE DRIFT (under measurement noise). Recheck setup.")
    else:
        print("  >>> MIXED. Both preprocessing AND model forward contribute.")
        print("      Run per-stage comparison next.")
    print()

    report = {
        "probe_date": "2026-05-25",
        "image": "inrule_identifier_passport.jpg",
        "norms": norms,
        "cosine": {
            "python_vs_candle_pil": cos_python_pil,
            "python_vs_candle_go": cos_python_go,
            "candle_pil_vs_candle_go": cos_pil_go,
        },
        "max_abs_diff": {
            "python_vs_candle_pil": md_python_pil,
            "python_vs_candle_go": md_python_go,
            "candle_pil_vs_candle_go": md_pil_go,
        },
        "mean_abs_diff": {
            "python_vs_candle_pil": mn_python_pil,
            "python_vs_candle_go": mn_python_go,
            "candle_pil_vs_candle_go": mn_pil_go,
        },
    }
    out_path = PROBE_DIR / "report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
