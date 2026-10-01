"""Decoder M16: successor items 1-8 for a formal finalist against two bars (prereg dec-m16-prereg-2026-10-01.md,
"Formal, successor items, hand-offs"): M14's two-bar evaluation (ops/m14/m14_successor.py, unchanged rule) with the
M16 bar notes and the 2B tier (bar-t1 = the stored DEV2.0-2B formal run; bar-b = its node-B M16 collection).

    python3 m16_successor.py --tier 2b|08b|4b --run RUN --types TYPES.json \
        --bar NAME=MLX-PAIRED.json,GATE.json --bar NAME=MLX-PAIRED.json,GATE.json \
        --overlap overlap-effects.json --exposure RECEIPT [--c1 SUMMARY.json] --output PREFIX
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "m14_successor", Path(__file__).resolve().parents[1] / "m14" / "m14_successor.py"
)
m14 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m14)

m14.SCHEMA = "dec-m16-successor/1"
m14.BAR_NOTES.clear()
m14.BAR_NOTES.update(
    {
        "4b": "bar-lh = the released LH's stored formal run (T = 1); bar-b = its node-B M16 collection",
        "2b": "bar-t1 = the stored DEV2.0-2B formal run; bar-b = its node-B M16 collection",
        "08b": "bar-t1 = the stored DEV2.0-0.8B formal run; bar-b = its node-B M16 collection",
    }
)

if __name__ == "__main__":
    sys.exit(m14.main())
