"""Own-model values for ``card_input.py --own``, from measured results only.

The public index and its areas come from the checkpoint's kit-scored ``result.json`` (``ckpt_eval --what full``).
Full, same-skill and new-domain are ws-measure's paired estimates against Perplexity Decider v1.1 (official pplx
Full plus the paired difference on the same calibrated measurements); ``--se`` is the paired standard error of
the Full estimate. Nothing here invents a value: a missing or incomplete public run is an error.

    python -m d25.vega.release.card_own --result result.json --full 63.7 --full-unadjusted 64.3 \\
        --paired-full 62.75 [--se 0.8] [--same-skill S --new-domain O] --gate gate-decisions.json --out own.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from d25.vega.release.card_input import AREAS


def own_values(
    result: dict[str, Any],
    full: float,
    se: float | None = None,
    same_skill: float | None = None,
    new_domain: float | None = None,
    full_unadjusted: float | None = None,
    paired_with: dict[str, Any] | None = None,
    min_coverage: float = 0.95,
) -> dict[str, Any]:
    public = result.get("public") or {}
    index = public.get("index")
    coverage = [v.get("coverage") for v in (public.get("per_benchmark") or {}).values()]
    if (
        index is None
        or not coverage
        or None in coverage
        or min(coverage) < min_coverage
    ):
        raise ValueError(
            "result.json carries no kit-scored public index with full benchmark coverage"
        )
    areas = {str(k).lower(): float(v) for k, v in (public.get("areas") or {}).items()}
    missing = [a for a in AREAS if a not in areas]
    if missing:
        raise ValueError(f"result.json public areas lack {missing}")
    for name, value in (
        ("full", full),
        ("same_skill", same_skill),
        ("new_domain", new_domain),
        ("full_unadjusted", full_unadjusted),
    ):
        if value is not None and not 0 < value <= 100:
            raise ValueError(f"{name}={value} is not an Index value (0-100]")
    if se is not None and not 0 < se < 20:
        raise ValueError(f"paired SE {se} out of range")
    return {
        "status": "estimate",
        "full": round(full, 2),
        "public": round(100 * index if index <= 1 else index, 2),
        "same_skill": None if same_skill is None else round(same_skill, 2),
        "new_domain": None if new_domain is None else round(new_domain, 2),
        "uncertainty": None if se is None else round(se, 2),
        "full_unadjusted": (
            None if full_unadjusted is None else round(full_unadjusted, 2)
        ),
        "paired_with": paired_with,
        "areas": {a: round(areas[a], 2) for a in AREAS},
        "source": {
            "arm": result.get("arm"),
            "step": result.get("step"),
            "result_created": result.get("created"),
            "public_benchmarks": len(coverage),
            "public_min_coverage": min(coverage),
            "method": "paired estimate vs Perplexity Decider v1.1 (ws-measure gate)",
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--result",
        type=Path,
        required=True,
        help="ckpt_eval result.json with a full public run",
    )
    ap.add_argument("--full", type=float, required=True, help="paired Full estimate")
    ap.add_argument(
        "--se",
        type=float,
        help="paired standard error of the Full estimate (omit until published)",
    )
    ap.add_argument("--same-skill", type=float)
    ap.add_argument("--new-domain", type=float)
    ap.add_argument(
        "--full-unadjusted",
        type=float,
        help="paired estimate without the in-distribution adjustment",
    )
    ap.add_argument("--paired-name", default="Perplexity Decider v1.1")
    ap.add_argument(
        "--paired-full",
        type=float,
        required=True,
        help="official board Full of the paired model",
    )
    ap.add_argument(
        "--gate",
        type=Path,
        help="gate decision file, recorded by sha256 for provenance",
    )
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    values = own_values(
        json.loads(args.result.read_text()),
        args.full,
        args.se,
        args.same_skill,
        args.new_domain,
        args.full_unadjusted,
        {"name": args.paired_name, "full": args.paired_full},
    )
    if args.gate:
        from d25.vega.release.build import sha256_file

        values["source"]["gate_sha256"] = sha256_file(args.gate)
    args.out.write_text(json.dumps(values, indent=1) + "\n")
    print(
        json.dumps(
            {"out": str(args.out), "full": values["full"], "public": values["public"]}
        )
    )


if __name__ == "__main__":
    main()
