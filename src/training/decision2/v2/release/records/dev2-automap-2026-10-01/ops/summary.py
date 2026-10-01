"""Result rows of the auto_map record from the fetched receipts (run from src/training/decision2).

python3 v2/release/records/dev2-automap-2026-10-01/ops/summary.py
"""

from __future__ import annotations

import json
from pathlib import Path

R = Path("v2/release/records/dev2-automap-2026-10-01")
TIERS = (
    ("0.6B", "0p6b"),
    ("0.8B", "0p8b"),
    ("2B", "2b"),
    ("4B", "4b"),
    ("9B", "9b"),
    ("27B", "27b"),
)


def load(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.is_file() else None


def compared(receipt: dict | None) -> tuple[int, int, int, float] | None:
    """Answers compared (question slots), answer changes, missing, maximum drift."""
    if not receipt:
        return None
    panels = list(receipt["panels"].values())
    return (
        sum(p["slots"] for p in panels),
        sum(p["category_changes"] for p in panels),
        sum(p["missing"] for p in panels),
        receipt["max_abs_drift"],
    )


def cell(result) -> str:
    if result is None:
        return "—"
    answers, changes, missing, drift = result
    return f"{changes} of {answers:,} ({drift:.2g})" + (
        f", {missing} missing" if missing else ""
    )


def main() -> None:
    print(
        "| Model | AutoModel vs native, Transformers 5.17: answers changed (max drift) | 5.18 | New `main` | "
        "Final decision (gate items) | Hub smoke 5.17 / 5.18 |"
    )
    print("| --- | --- | --- | --- | --- | --- |")
    for tier, key in TIERS:
        base = R / key
        verify = load(base / "verify/receipts/automap-vs-native-pre.json")
        mlx = load(base / "mlx/receipts/automap-vs-native-pre.json")
        release = load(base / "release/receipts/automap-vs-native-pre.json")
        main = compared(release or verify)
        if mlx and main:
            extra = compared(mlx)
            main = (
                main[0] + extra[0],
                main[1] + extra[1],
                main[2] + extra[2],
                max(main[3], extra[3]),
            )
        tf518 = compared(load(base / "tf518/receipts/automap-tf518-vs-native.json"))
        upload = load(base / "release/receipts/upload.json") or {}
        gate = load(base / "release/receipts/gate.json") or {}
        hubs = [
            load(base / f"release/receipts/automap-hub{s}.json") for s in ("", "-tf518")
        ]
        smoke = " / ".join(
            "pass" if h and h["passed"] else ("fail" if h else "—") for h in hubs
        )
        items = gate.get("items") or {}
        decision = (
            f"`{gate['decision_sha256'][:8]}…` ({sum(i['passed'] for i in items.values())} / {len(items)})"
            if gate
            else "—"
        )
        revision = f"`{upload['revision']}`" if upload.get("revision") else "—"
        print(
            f"| DEV2.0-{tier} | {cell(main)} | {cell(tf518)} | {revision} | {decision} | {smoke} |"
        )


if __name__ == "__main__":
    main()
