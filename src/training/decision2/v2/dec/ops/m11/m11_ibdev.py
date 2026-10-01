"""Decoder M11 stage-2 IB DEV panel (prereg dec-m11-stage2-prereg-2026-10-01.md, "Retention probes and IB DEV").

  prompts  IB1 DEV then IB2 DEV (each hash-checked), every row one gold-free prompt rendered exactly as M9's HR2 DEV
           slice (ops/m9/m9_hr2dev.py: ops/m8/m8_data.to_prompt, one question); writes <out>/ib-dev.dev.jsonl (the
           concatenated rows), ib-dev.prompts.jsonl, ib-dev.gold.jsonl and manifest.json. The caller moves the gold
           and the rows under /data/dev2/private (mode 600).
  score    per-family accuracy (invalid answers count as wrong), the IB DEV macro over every family, the transfer
           macro without the --exclude families, and with --reference the paired group bootstrap of both macros'
           differences (M9's: 2,000 draws, seed 20261001).

The breadth signal of stage 2 (gate 3: IB DEV macro delta vs LH >= 0); never Index rows.

usage: m11_ibdev.py prompts --ib1 F --ib1-sha S --ib2 F --ib2-sha S --output DIR
       m11_ibdev.py score --gold G --predictions P --name N [--reference R --reference-name RN] \
         --exclude isarc,hover,gsm2 --output OUT
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

SCHEMA = "dec-m11-ibdev/1"
OPS = Path(__file__).resolve().parents[1]


def hr2dev() -> Any:
    spec = importlib.util.spec_from_file_location(
        "m9_hr2dev", OPS / "m9" / "m9_hr2dev.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")


def block(h: Any, gold: list[dict], left: dict, right: dict | None) -> dict[str, Any]:
    out = h.summarize(gold, left)
    if right is not None:
        ref = h.summarize(gold, right)
        out["reference_family_macro"] = ref["family_macro"]
        out["reference_families"] = {
            k: v["accuracy"] for k, v in ref["families"].items()
        }
        out["delta_family_macro"] = out["family_macro"] - ref["family_macro"]
        out["paired"] = h.paired(gold, left, right)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("prompts")
    for name in ("ib1", "ib2"):
        b.add_argument(f"--{name}", type=Path, required=True)
        b.add_argument(f"--{name}-sha", required=True)
    b.add_argument("--output", type=Path, required=True)
    s = sub.add_parser("score")
    s.add_argument("--gold", type=Path, required=True)
    s.add_argument("--predictions", type=Path, required=True)
    s.add_argument("--name", required=True)
    s.add_argument("--reference", type=Path)
    s.add_argument("--reference-name")
    s.add_argument("--exclude", required=True)
    s.add_argument("--output", type=Path, required=True)
    a = p.parse_args(argv)
    h = hr2dev()
    if a.cmd == "prompts":
        lines = []
        for name in ("ib1", "ib2"):
            path = getattr(a, name)
            if h.sha(path) != getattr(a, f"{name}_sha"):
                raise ValueError(f"{path}: sha256 differs from --{name}-sha")
            lines += [
                line
                for line in path.read_bytes().splitlines(keepends=True)
                if line.strip()
            ]
        rows = [json.loads(line) for line in lines]
        if len({r["id"] for r in rows}) != len(rows):
            raise ValueError("repeated id across IB1 / IB2 DEV")
        prompts, gold = h.build_prompts(rows)
        a.output.mkdir(parents=True, exist_ok=False)
        (a.output / "ib-dev.dev.jsonl").write_bytes(b"".join(lines))
        write_jsonl(a.output / "ib-dev.prompts.jsonl", prompts)
        write_jsonl(a.output / "ib-dev.gold.jsonl", gold)
        families: dict[str, int] = {}
        for g in gold:
            families[g["family"]] = families.get(g["family"], 0) + 1
        manifest = {
            "schema": SCHEMA,
            "ib1_dev_sha256": a.ib1_sha,
            "ib2_dev_sha256": a.ib2_sha,
            "rows": len(rows),
            "dev_sha256": h.sha(a.output / "ib-dev.dev.jsonl"),
            "prompts_sha256": h.sha(a.output / "ib-dev.prompts.jsonl"),
            "gold_sha256": h.sha(a.output / "ib-dev.gold.jsonl"),
            "families": dict(sorted(families.items())),
        }
        (a.output / "manifest.json").write_text(
            json.dumps(manifest, indent=1, sort_keys=True) + "\n"
        )
        print(json.dumps({k: manifest[k] for k in ("rows", "prompts_sha256")}))
        return 0
    exclude = set(a.exclude.split(","))
    gold = h.read_jsonl(a.gold)
    left = h.outcomes(gold, {r["id"]: r for r in h.read_jsonl(a.predictions)})
    right = None
    if a.reference is not None:
        right = h.outcomes(gold, {r["id"]: r for r in h.read_jsonl(a.reference)})
    transfer = [g for g in gold if g["family"] not in exclude]
    out = {
        "schema": SCHEMA,
        "role": "stage-2 breadth signal (gate 3: IB DEV macro delta vs the reference >= 0); transfer macro report only",
        "name": a.name,
        "reference_name": a.reference_name,
        "files": {
            "gold": h.sha(a.gold),
            "predictions": h.sha(a.predictions),
            "reference": None if a.reference is None else h.sha(a.reference),
        },
        "excluded_from_transfer": sorted(exclude),
        "ib_dev": block(h, gold, left, right),
        "transfer": block(h, transfer, left, right),
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with open(a.output, "x") as f:
        json.dump(out, f, indent=1, sort_keys=True)
        f.write("\n")
    print(
        json.dumps(
            {
                "name": a.name,
                "ib_dev_macro": round(out["ib_dev"]["family_macro"], 4),
                "delta": out["ib_dev"].get("delta_family_macro"),
                "transfer_macro": round(out["transfer"]["family_macro"], 4),
                "transfer_delta": out["transfer"].get("delta_family_macro"),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
