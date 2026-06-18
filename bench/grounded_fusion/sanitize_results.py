"""Strip personal/machine-specific data from benchmark result JSON in place.

Result files (summary_*.json, ab_report.json) and per-sample JSONL may embed the
absolute dataset path or home-rooted output dirs. This rewrites them so the files
are safe to share/attach: the dataset path becomes its basename and any
home-rooted path becomes ``~``-relative. Benchmark content (DRACO problems,
rubric scores, model answers) is left untouched.

Usage:
    .venv-bench/bin/python -m bench.grounded_fusion.sanitize_results \
        bench/grounded_fusion/results bench/grounded_fusion/results_sweep
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HOME = str(Path.home())


def _mask(obj):
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k == "draco_path" and isinstance(v, str):
                out[k] = Path(v).name
            else:
                out[k] = _mask(v)
        return out
    if isinstance(obj, list):
        return [_mask(v) for v in obj]
    if isinstance(obj, str) and obj.startswith(HOME):
        return "~" + obj[len(HOME) :]
    return obj


def _sanitize_json(path: Path) -> bool:
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    masked = _mask(data)
    if masked != data:
        path.write_text(json.dumps(masked, indent=2))
        return True
    return False


def _sanitize_jsonl(path: Path) -> bool:
    changed = False
    lines = []
    for line in path.read_text().splitlines():
        if not line.strip():
            lines.append(line)
            continue
        obj = json.loads(line)
        masked = _mask(obj)
        changed = changed or masked != obj
        lines.append(json.dumps(masked))
    if changed:
        path.write_text("\n".join(lines) + "\n")
    return changed


def main(argv: list[str]) -> None:
    roots = [Path(a) for a in argv] or [Path("bench/grounded_fusion")]
    n = 0
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*.json"):
            if _sanitize_json(p):
                print(f"sanitized {p}")
                n += 1
        for p in root.rglob("*.jsonl"):
            if _sanitize_jsonl(p):
                print(f"sanitized {p}")
                n += 1
    print(f"done ({n} file(s) modified)")


if __name__ == "__main__":
    main(sys.argv[1:])
