"""Episode identity helpers shared by the comparison slice.

`episode_ids` fingerprints a measurement's physical execution (its paired
result and request receipts) so a relabelled re-import of the same episodes
cannot be counted as an independent comparison. Extracted from the scheduler
(the agent loop lands later) so the comparison slice can pin physical
episodes without importing the scheduler.
"""

from __future__ import annotations

import json
from pathlib import Path

from .loop_evidence import digest


def canonical(doc):
    return json.dumps(doc, sort_keys=True, separators=(",", ":"), allow_nan=False)


def episode_ids(row):
    seen = set()
    for name, value in row["evidence"].items():
        path = Path(name)
        if path.name != "result.json":
            continue
        requests = row["evidence"].get(str(path.with_name("requests.jsonl")))
        if requests:
            seen.add(
                digest(
                    canonical(
                        {"result.json": value, "requests.jsonl": requests}
                    ).encode()
                )
            )
    return seen
