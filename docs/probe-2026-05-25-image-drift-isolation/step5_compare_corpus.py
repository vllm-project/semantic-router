#!/usr/bin/env python3
"""
Step 5 of probe-2026-05-25-image-drift-isolation: corpus-wide comparison.

Loads:
  - python_corpus_embeddings.npy  (from step4, shape [20, 384])
  - candle_corpus_gopipe.bin       (from Go test, shape [20, 384] as float32 LE)
  - {python,candle}_corpus_index.json (image names in row order, must match)

Reports per-image cosine sim + max abs diff + mean abs diff plus corpus
summary (mean / min / max cosine across the 20 images). Writes
corpus_report.json with the canonical numbers.
"""

import json
import statistics
import sys
from pathlib import Path

import numpy as np

PROBE_DIR = Path(__file__).parent.resolve()


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main():
    py_npy = PROBE_DIR / "python_corpus_embeddings.npy"
    py_idx = PROBE_DIR / "python_corpus_index.json"
    cd_bin = PROBE_DIR / "candle_corpus_gopipe.bin"
    cd_idx = PROBE_DIR / "candle_corpus_index.json"

    missing = [p for p in (py_npy, py_idx, cd_bin, cd_idx) if not p.exists()]
    if missing:
        print("ERROR: missing artifacts.", file=sys.stderr)
        for m in missing:
            print(f"  missing: {m}", file=sys.stderr)
        sys.exit(1)

    py_emb = np.load(py_npy).astype(np.float32)
    cd_flat = np.fromfile(cd_bin, dtype=np.float32)
    if cd_flat.size != py_emb.size:
        print(
            f"ERROR: candle binary has {cd_flat.size} floats, python has {py_emb.size}",
            file=sys.stderr,
        )
        sys.exit(1)
    cd_emb = cd_flat.reshape(py_emb.shape)

    with open(py_idx) as f:
        py_idx_data = json.load(f)
    with open(cd_idx) as f:
        cd_idx_data = json.load(f)
    if py_idx_data["images"] != cd_idx_data["images"]:
        print(
            "ERROR: image order mismatch between python and candle indices",
            file=sys.stderr,
        )
        for i, (p, c) in enumerate(zip(py_idx_data["images"], cd_idx_data["images"])):
            mark = " " if p == c else "*"
            print(f"  {mark} [{i}] python={p:50s} candle={c}", file=sys.stderr)
        sys.exit(1)

    names = py_idx_data["images"]
    n = len(names)

    print()
    print("=" * 90)
    print(f"Corpus drift report - 2026-05-25 - {n} images x 384-d embeddings")
    print("=" * 90)
    print()
    print(f"{'image':50s} {'cos':>10s} {'max_abs':>10s} {'mean_abs':>10s}")
    print("-" * 90)

    rows = []
    for i, name in enumerate(names):
        c = cos(py_emb[i], cd_emb[i])
        mx = float(np.max(np.abs(py_emb[i] - cd_emb[i])))
        mn = float(np.mean(np.abs(py_emb[i] - cd_emb[i])))
        rows.append({"image": name, "cos": c, "max_abs": mx, "mean_abs": mn})
        print(f"{name:50s} {c:10.6f} {mx:10.6f} {mn:10.6f}")

    cosines = [r["cos"] for r in rows]
    max_abs = [r["max_abs"] for r in rows]
    mean_abs = [r["mean_abs"] for r in rows]

    summary = {
        "image_count": n,
        "cosine": {
            "min": float(min(cosines)),
            "max": float(max(cosines)),
            "mean": float(statistics.mean(cosines)),
            "median": float(statistics.median(cosines)),
        },
        "max_abs_diff": {
            "min": float(min(max_abs)),
            "max": float(max(max_abs)),
            "mean": float(statistics.mean(max_abs)),
            "median": float(statistics.median(max_abs)),
        },
        "mean_abs_diff": {
            "min": float(min(mean_abs)),
            "max": float(max(mean_abs)),
            "mean": float(statistics.mean(mean_abs)),
            "median": float(statistics.median(mean_abs)),
        },
    }

    print()
    print("Corpus summary:")
    print(
        f"  cosine     : min={summary['cosine']['min']:.6f} max={summary['cosine']['max']:.6f} mean={summary['cosine']['mean']:.6f}"
    )
    print(
        f"  max_abs    : min={summary['max_abs_diff']['min']:.6f} max={summary['max_abs_diff']['max']:.6f} mean={summary['max_abs_diff']['mean']:.6f}"
    )
    print(
        f"  mean_abs   : min={summary['mean_abs_diff']['min']:.6f} max={summary['mean_abs_diff']['max']:.6f} mean={summary['mean_abs_diff']['mean']:.6f}"
    )
    print()

    # Verdict heuristic - same band as single-image step3
    if summary["cosine"]["min"] >= 0.999:
        print(
            ">>> CORPUS-WIDE PARITY ACHIEVED. All 20 images at cos >= 0.999 vs Python reference."
        )
    elif summary["cosine"]["min"] >= 0.995:
        print(
            ">>> CORPUS GOOD. All 20 images cos >= 0.995. Some images have residual drift; outliers worth inspecting."
        )
    else:
        print(">>> CORPUS HAS OUTLIERS. Investigate images below cos 0.995.")

    report = {
        "probe_date": "2026-05-25",
        "fixture_set": "docs/probe-2026-05-20-calibration/fixtures (20 images)",
        "summary": summary,
        "per_image": rows,
    }
    out = PROBE_DIR / "corpus_report.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
