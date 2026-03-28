"""Trace-based workload replay.

Replays real request traces from CSV or JSON-Lines files.

Supported formats
-----------------
azure_csv          : Azure LLM Inference Trace CSV
                     Columns: TIMESTAMP, ContextTokens, GeneratedTokens

splitwise          : Splitwise two-phase-commit CSV
                     Columns: timestamp, num_prefill_tokens, num_decode_tokens

semantic_router    : Pre-labeled trace produced by a semantic router
                     (e.g. vLLM semantic router, RouterDC, AutoMix, complexity
                     classifier, embedding-based selection).

                     Each record carries the routing decision made by the
                     semantic router alongside token counts.  The simulator
                     replays those decisions via ModelRouter so you can
                     evaluate fleet sizing and SLO compliance without
                     re-running the classifier.

                     Both JSONL (one JSON object per line) and CSV are
                     accepted.  Column names are configurable via ``field_map``
                     to match whatever your router logs.

                     Default field_map for JSONL::

                         {
                           "timestamp":      "timestamp",
                           "l_in":           "prompt_tokens",
                           "l_out":          "generated_tokens",
                           "model_id":       "selected_model",   # routing decision
                           "complexity":     "complexity",       # optional signal
                           "category":       "category",         # optional signal
                         }

                     Set ``model_id_field`` (shorthand for field_map["model_id"])
                     to match your router's field name, e.g.:
                       ``model_id_field="model"``             (OpenAI-style body)
                       ``model_id_field="x_vsr_selected_model"``
                       ``model_id_field="routed_to"``
                       ``model_id_field="selected_model"``    (default)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from ..core.request import Request

# Default column names for the semantic_router format
_SR_DEFAULTS: dict[str, str] = {
    "timestamp": "timestamp",
    "l_in": "prompt_tokens",
    "l_out": "generated_tokens",
    "model_id": "selected_model",
    "complexity": "complexity",
    "category": "category",
}


class TraceWorkload:
    """Replay a real request trace.

    Parameters
    ----------
    path        : path to trace file (CSV or JSONL)
    fmt         : "azure_csv" | "splitwise" | "semantic_router"
    scale_lam   : if set, re-timestamps requests to achieve this avg arrival
                  rate (req/s), overriding the original timestamps
    max_reqs    : truncate after this many requests (None = all)
    seed        : RNG seed for re-scaling
    field_map   : (semantic_router only) override default column-name mapping.
                  Keys are simulator fields; values are column names in the
                  file.  Unspecified keys use the defaults shown above.
    model_id_field : shorthand for field_map["model_id"].  Convenience
                  parameter so callers only need to name the routing-decision
                  column without building a full field_map.
    default_model_id : (semantic_router only) model_id to assign when the
                  routing-decision column is absent or empty.
    """

    def __init__(
        self,
        path: str,
        fmt: str = "azure_csv",
        scale_lam: float | None = None,
        max_reqs: int | None = None,
        seed: int = 42,
        field_map: dict[str, str] | None = None,
        model_id_field: str | None = None,
        default_model_id: str | None = None,
    ):
        self.path = Path(path)
        self.fmt = fmt
        self.scale_lam = scale_lam
        self.max_reqs = max_reqs
        self.seed = seed
        self.default_model_id = default_model_id

        # Build effective field map for semantic_router format
        self._fmap = dict(_SR_DEFAULTS)
        if field_map:
            self._fmap.update(field_map)
        if model_id_field:
            self._fmap["model_id"] = model_id_field

    def generate(self) -> list[tuple[float, Request]]:
        if self.fmt == "azure_csv":
            raw = self._load_azure_csv()
        elif self.fmt == "splitwise":
            raw = self._load_splitwise()
        elif self.fmt == "semantic_router":
            raw = self._load_semantic_router()
        else:
            raise ValueError(
                f"Unknown trace format: {self.fmt!r}.  "
                f"Supported: 'azure_csv', 'splitwise', 'semantic_router'."
            )

        if self.max_reqs:
            raw = raw[: self.max_reqs]

        if self.scale_lam is not None:
            import random

            rng = random.Random(self.seed)
            t = 0.0
            arrivals: list = []
            for _, req in raw:
                t += rng.expovariate(self.scale_lam)
                req.arrival_time = t
                arrivals.append((t, req))
            return arrivals

        return raw

    # ── format loaders ────────────────────────────────────────────────────────

    def _load_azure_csv(self) -> list:
        arrivals = []
        with open(self.path, newline="") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                try:
                    ts = float(row.get("TIMESTAMP", i))
                    l_in = int(float(row.get("ContextTokens", 512)))
                    l_out = int(float(row.get("GeneratedTokens", 128)))
                except (ValueError, KeyError):
                    continue
                req = Request(
                    req_id=i,
                    arrival_time=ts,
                    l_in=max(1, l_in),
                    l_out=max(1, l_out),
                    category="prose",
                )
                arrivals.append((ts, req))
        arrivals.sort(key=lambda x: x[0])
        return arrivals

    def _load_splitwise(self) -> list:
        arrivals = []
        with open(self.path, newline="") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                try:
                    ts = float(row.get("timestamp", i))
                    l_in = int(float(row.get("num_prefill_tokens", 512)))
                    l_out = int(float(row.get("num_decode_tokens", 128)))
                except (ValueError, KeyError):
                    continue
                req = Request(
                    req_id=i,
                    arrival_time=ts,
                    l_in=max(1, l_in),
                    l_out=max(1, l_out),
                    category="prose",
                )
                arrivals.append((ts, req))
        arrivals.sort(key=lambda x: x[0])
        return arrivals

    def _load_semantic_router(self) -> list:
        """Load a pre-labeled trace from the semantic router.

        Detects format automatically: JSONL if the file ends with .jsonl or
        the first non-empty line starts with '{'; CSV otherwise.
        """
        suffix = self.path.suffix.lower()
        if suffix == ".jsonl":
            rows = self._parse_jsonl()
        elif suffix in (".csv", ".tsv"):
            rows = self._parse_csv(delimiter="\t" if suffix == ".tsv" else ",")
        else:
            # Auto-detect: peek at first non-empty line
            with open(self.path) as f:
                first = next((l.strip() for l in f if l.strip()), "")
            rows = self._parse_jsonl() if first.startswith("{") else self._parse_csv()

        fm = self._fmap
        ts_key = fm["timestamp"]
        lin_key = fm["l_in"]
        lout_key = fm["l_out"]
        mid_key = fm["model_id"]
        cat_key = fm.get("category", "category")
        cpx_key = fm.get("complexity", "complexity")

        arrivals = []
        for i, row in enumerate(rows):
            try:
                ts = float(row.get(ts_key, i))
                l_in = int(float(row.get(lin_key, 512)))
                l_out = int(float(row.get(lout_key, 128)))
            except (ValueError, KeyError, TypeError):
                continue

            model_id = (row.get(mid_key) or self.default_model_id) or None

            # Use semantic router's category as request category if present
            category = row.get(cat_key, "prose") or "prose"

            req = Request(
                req_id=i,
                arrival_time=ts,
                l_in=max(1, l_in),
                l_out=max(1, l_out),
                category=str(category),
                model_id=str(model_id) if model_id else None,
            )

            # Attach complexity signal as a custom attribute for analysis
            cpx = row.get(cpx_key)
            if cpx is not None:
                req._complexity = str(cpx)

            arrivals.append((ts, req))

        arrivals.sort(key=lambda x: x[0])
        return arrivals

    def _parse_jsonl(self) -> list[dict]:
        rows = []
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return rows

    def _parse_csv(self, delimiter: str = ",") -> list[dict]:
        rows = []
        with open(self.path, newline="") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            for row in reader:
                rows.append(dict(row))
        return rows
