"""Dump DRACO items as JSONL for the fusioneval Go driver.

The cached-panel evaluator splits generation (Go: cmd/fusioneval, which needs the
candle NLI + Ollama) from DRACO parsing (Python: datasets.py). This writes one
item per line — ``{id, domain, question, context}`` — which the Go driver reads to
generate the panel and synthesize the arms. ``context`` is populated for
context-grounded datasets (gold passages); empty for DRACO panel-mode.

    python -m bench.grounded_fusion.items --draco-path draco.json \
        --domains Medicine,Law --max-samples 100 --out results/items.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .datasets import get_dataset


def dump_items(
    path: str,
    out: str,
    domains: str | None = None,
    max_samples: int | None = None,
    seed: int = 42,
    dataset: str = "draco",
) -> int:
    samples = get_dataset(
        dataset,
        path=path,
        max_samples=max_samples,
        domains=domains.split(",") if domains else None,
        seed=seed,
    )
    outp = Path(out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with outp.open("w") as fh:
        for s in samples:
            context = ""
            if isinstance(s.metadata, dict):
                context = str(s.metadata.get("context", "") or "")
            fh.write(
                json.dumps(
                    {
                        "id": s.id,
                        "domain": s.domain,
                        "question": s.problem,
                        "context": context,
                    }
                )
                + "\n"
            )
    return len(samples)


def main():
    ap = argparse.ArgumentParser(description="Dump DRACO items JSONL for fusioneval")
    ap.add_argument("--draco-path", required=True)
    ap.add_argument("--out", default="bench/grounded_fusion/results/items.jsonl")
    ap.add_argument("--dataset", default="draco")
    ap.add_argument("--domains", default=None, help="comma-separated domain filter")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    n = dump_items(
        args.draco_path,
        args.out,
        domains=args.domains,
        max_samples=args.max_samples,
        seed=args.seed,
        dataset=args.dataset,
    )
    print(f"wrote {n} items -> {args.out}")


if __name__ == "__main__":
    main()
