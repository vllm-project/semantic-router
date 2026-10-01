"""Result rows of the forward-token-budget record from the fetched receipts (run from src/training/decision2).

python3 v2/release/records/dev2-budget-2026-10-01/ops/summary.py

Receipts: <key>/extra/receipts/*.json and <key>/release/receipts/*.json (ops/fetch_receipts.sh).
"""

from __future__ import annotations

import json
from pathlib import Path

R = Path("v2/release/records/dev2-budget-2026-10-01")
TIERS = (("0.8B", "0p8b"), ("2B", "2b"), ("9B", "9b"), ("27B", "27b"))


def load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.is_file() else None


def panels(receipt: dict | None) -> str:
    if not receipt:
        return "—"
    ps = list(receipt["panels"].values())
    slots = sum(p.get("slots", p.get("prompts", 0)) for p in ps)
    changes = sum(p["category_changes"] + p["missing"] for p in ps)
    drift = max(p.get("max_abs_drift", 0.0) for p in ps)
    return (
        f"{changes} of {slots:,} ({drift:.0e})"
        if drift
        else f"{changes} of {slots:,} (0)"
    )


def main() -> None:
    print(
        "| Model | Parity pre / post: answers changed (max drift) | AutoModel vs native | Long-input regression "
        "| New `main` | Final decision (gate items) | Hub smoke 5.17 / 5.18 |"
    )
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for tier, key in TIERS:
        rel = R / key / "release" / "receipts"
        ext = R / key / "extra" / "receipts"
        gate = load(rel / "gate.json")
        upload = load(rel / "upload.json")
        long = load(ext / "long-request.json")
        hub = [load(rel / f"automap-hub{s}.json") for s in ("", "-tf518")]
        lr = "—"
        if long:
            ties = len(long.get("ties_vs_alone", {}))
            lr = (
                f"{'pass' if long['pass'] else 'FAIL'}: {long['questions']} q, forwards "
                + " + ".join(f"{r}×{w:,}" for r, w in long["forwards"])
                + f", {len(long['mismatched_vs_alone'])} mismatched"
                + (f", {ties} tie" if ties else "")
                + f", drift {long['max_drift_vs_alone']:.3f}"
            )
        items = ""
        if gate:
            items = f" ({sum(1 for i in gate['items'].values() if i.get('passed'))} / {len(gate['items'])})"
        print(
            f"| DEV2.0-{tier} | {panels(load(rel / 'parity-pre.json'))} / {panels(load(rel / 'parity-post.json'))} "
            f"| {panels(load(rel / 'automap-vs-native-pre.json'))} | {lr} "
            f"| {'`' + upload['revision'] + '`' if upload else '—'} "
            f"| {'`' + gate['decision_sha256'][:8] + '…`' + items if gate else '—'} "
            f"| {' / '.join(('pass' if h.get('passed') else 'FAIL') if h else '—' for h in hub)} |"
        )
    print()
    print(
        "| Model | Requests | Bit-identical | p50 ms old → new | p95 ms old → new | Request peak GiB old → new |"
    )
    print("| --- | ---: | ---: | --- | --- | --- |")
    for tier, key in TIERS:
        c = load(R / key / "extra" / "receipts" / "bench-compare.json")
        if not c:
            continue
        lat, mem = c["latency_ms"], c["memory_gib"]["request_peak"]
        print(
            f"| DEV2.0-{tier} | {c['items']} | {c['bit_identical_items']} / {c['items']} "
            f"| {lat['p50']['old']:.1f} → {lat['p50']['new']:.1f} | {lat['p95']['old']:.1f} → {lat['p95']['new']:.1f} "
            f"| {mem['old']:.2f} → {mem['new']:.2f} |"
        )


if __name__ == "__main__":
    main()
