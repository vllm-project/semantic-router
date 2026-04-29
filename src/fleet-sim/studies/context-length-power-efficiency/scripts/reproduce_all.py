"""
reproduce_all.py — run all five table scripts in sequence and print results.

Usage:
    make tables
    python scripts/reproduce_all.py

Expected runtime: ~10 seconds (fleet analysis dominates).
"""

import subprocess, sys, os

SCRIPTS = [
    ("Table 1 — ctx window vs n_max / tok/W",          "table1_ctx_nmax.py"),
    ("Table 2 — single-GPU tok/W by model architecture","table2_arch_tpw.py"),
    ("Table 3 — fleet tok/W by topology & GPU",         "table3_fleet_tpw.py"),
    ("Table 4 — context vs semantic routing",            "table4_routing.py"),
    ("Table 5 — GPU generation comparison",              "table5_gen_compare.py"),
]

_dir = os.path.dirname(os.path.abspath(__file__))

for title, fname in SCRIPTS:
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)
    result = subprocess.run(
        [sys.executable, fname],
        cwd=_dir,
    )
    if result.returncode != 0:
        print(f"[ERROR] {fname} exited with code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)

print("\n" + "=" * 72)
print("  All tables reproduced successfully.")
print("=" * 72)
