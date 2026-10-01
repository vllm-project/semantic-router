"""Markdown tables of the BF16-resident rollout from the copied receipts (run from the records directory).

python3 dev2-bf16-resident-2026-10-01/ops/tables.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TIERS = (
    ("0p6b", "0.6B"),
    ("0p8b", "0.8B"),
    ("2b", "2B"),
    ("4b", "4B"),
    ("9b", "9B"),
    ("27b", "27B"),
)
PANELS = ("typed-final", "css15", "public231", "mlx-diag")


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def drift(value: float) -> str:
    return "0" if value == 0 else f"{value:.1e}"


def main() -> None:
    result, release, panels = [], [], []
    for key, tier in TIERS:
        r = ROOT / key / "release" / "receipts"
        x = ROOT / key / "release" / "extra"
        gate, bench = load(r / "gate.json"), load(ROOT / key / "bench" / "compare.json")
        decision = ROOT / f"DEV2.0-{tier}.decision.bf16r.json"
        if gate is None:
            result.append(f"| DEV2.0-{tier} | not published | | | | |")
            continue
        assert (
            gate["decision_sha256"] == hashlib.sha256(decision.read_bytes()).hexdigest()
        )
        replaced = json.loads(decision.read_text(encoding="utf-8"))["supersedes"][
            "released_as"
        ]
        pre, post = load(r / "parity-pre.json"), load(r / "parity-post.json")
        mlx = load(ROOT / key / "mlx-parity" / "receipts" / "parity-pre.json")
        merged = {**pre["panels"], **((mlx or {}).get("panels") or {})}
        changes = sum(p["category_changes"] + p["missing"] for p in merged.values())
        worst = max(p["max_abs_drift"] for p in merged.values())
        answers = sum(p["slots"] for p in merged.values())
        lat, mem = bench["latency_ms"], bench["memory_gib"]["request_peak"]
        result.append(
            f"| DEV2.0-{tier} | `{gate['revision']}` | `{gate['decision_sha256'][:8]}…` "
            f"({gate.get('gate_profile') or 'own 1.0'}, {sum(i['passed'] for i in gate['items'].values())} / "
            f"{len(gate['items'])}) | `{replaced.split('@')[1][:8]}` | {changes} of {answers:,} ({drift(worst)}) | "
            f"{lat['p50']['old']:.1f} → {lat['p50']['new']:.1f} ms; {mem['old']:.1f} → {mem['new']:.1f} GiB |"
        )
        cells = []
        for name in PANELS:
            side = [p for p in (pre, post, mlx) if p and name in p["panels"]]
            cells.append(
                " / ".join(
                    f"{s['panels'][name]['category_changes']} ({drift(s['panels'][name]['max_abs_drift'])})"
                    for s in side
                )
            )
        panels.append(f"| DEV2.0-{tier} | " + " | ".join(cells) + " |")
        diff, ex = load(x / "runtime-diff.json"), load(x / "examples-vs-released.json")
        tree, post_run, links = (
            load(r / "tree.json"),
            load(r / "post.json"),
            load(x / "hub-links.json"),
        )
        summary = load(ROOT / key / "release" / "RELEASE-RECEIPT.json") or {}
        release.append(
            f"| DEV2.0-{tier} | {tree['files']} | {post_run['loaded_parameters']:,} | "
            f"{diff['weight_files']} files, {diff['weight_bytes'] / 1e9:.2f} GB: "
            f"{'identical' if diff['weights_byte_identical'] else 'CHANGED'} | "
            f"{', '.join(n.removeprefix('decision2/') for n in diff['changed_runtime_files'])} + "
            f"{', '.join(diff['changed_card_files'])} | "
            f"{'identical' if ex['bit_identical_answers'] else 'differ'} | "
            f"{links['checked'] - len(links['failed'])} / {links['checked']} | "
            f"{summary.get('wall_seconds', 0):.0f} s |"
        )
    print("RESULT\n" + "\n".join(result))
    print(
        "\nPANELS (answer changes (max drift): pre / post [/ mlx-only])\n"
        + "\n".join(panels)
    )
    print("\nRELEASE\n" + "\n".join(release))


if __name__ == "__main__":
    main()
