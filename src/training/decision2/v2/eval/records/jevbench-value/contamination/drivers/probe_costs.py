"""Probe cost inputs from prior collections (CPU, stdlib): per-panel wall seconds, model path, adapter.

Usage (stdin driver, node A): ssh NODE_A "cd /tmp && python3 - '<json>'" < probe_costs.py
  json: {"models": [{"name", "dir"}]}
Reads <dir>/COLLECT.json (panels[].wall_seconds, output_rows; model path; adapter) and
<dir>/GPU-TIME.json, and checks whether the recorded model path still exists.
"""

import json
import os
import sys
from pathlib import Path


def find(obj, keys):
    stack = [obj]
    while stack:
        x = stack.pop()
        if isinstance(x, dict):
            for k, v in x.items():
                if k in keys and isinstance(v, (str, int, float)):
                    return v
                stack.append(v)
        elif isinstance(x, list):
            stack.extend(x)
    return None


args = json.loads(sys.argv[1])
out = {}
for m in args["models"]:
    d = Path(m["dir"])
    e = {}
    c = d / "COLLECT.json"
    if c.exists():
        col = json.loads(c.read_text())
        e["panels"] = {
            p["panel"]: {
                "wall_s": round(p.get("wall_seconds") or 0, 1),
                "rows": p.get("output_rows"),
            }
            for p in col.get("panels", [])
            if isinstance(p, dict) and "panel" in p
        }
        mp = find(col, {"model_path", "model_dir"})
        e["model_path"] = mp
        e["model_path_exists"] = bool(mp) and os.path.exists(str(mp))
        e["adapter"] = find(col, {"adapter", "adapter_name", "backend"})
        e["revision"] = find(col, {"revision"})
    else:
        e["collect"] = "missing"
        rep = d / "REPORT.json"
        if rep.exists():
            r = json.loads(rep.read_text())
            e["report_gpu_wall_s"] = r.get("gpu_wall_seconds")
            e["reused"] = r.get("reused")
            mp = find(r, {"model_path", "model_dir", "path"})
            e["report_model_path"] = mp
    g = d / "GPU-TIME.json"
    if g.exists():
        gt = json.loads(g.read_text())
        e["gpu_time"] = {
            k: gt.get(k)
            for k in ("gpu", "wall_seconds", "shared", "image_id")
            if k in gt
        }
    out[m["name"]] = e
print(json.dumps(out, indent=1))
