"""Print the first preferred GPU with low use and enough free VRAM (rocm-smi JSON on stdin).

  rocm-smi --showuse --showmeminfo vram --json | python3 pick_gpu.py <need GB> <gpu> [<gpu> ...]

Exits 1 when none qualifies; the caller shares that GPU under its own lease file.
"""

from __future__ import annotations

import json
import sys

need_gb, prefer = float(sys.argv[1]), [int(g) for g in sys.argv[2:]]
cards = json.load(sys.stdin)
for gpu in prefer:
    card = cards.get(f"card{gpu}")
    if not card:
        continue
    use = float(card.get("GPU use (%)", 100))
    free = (
        int(card["VRAM Total Memory (B)"]) - int(card["VRAM Total Used Memory (B)"])
    ) / 1e9
    print(f"gpu{gpu} use {use:.0f}% free {free:.1f} GB", file=sys.stderr)
    if use < 50 and free >= need_gb:
        print(gpu)
        sys.exit(0)
sys.exit(1)
