from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from v2.eval.sealed import candidates as runner
from v2.eval.sealed.schema import validate
from v2.eval.sealed.sources import factbutcher

PROVERENO = "https://provereno.media/blog/{}/example-article/"


def row(claim_id, verdict, reference, acceptable=None, url=None):
    return {
        "claim_id": claim_id,
        "claim": f"Утверждение номер {claim_id} о погоде в Москве.",
        "gold_verdict": verdict,
        "acceptable_verdicts": acceptable or [verdict],
        "benchmark_component": (
            "provereno_media" if url else "factbutcher_human_benchmark"
        ),
        "reference_date": reference,
        "source_name": "Provereno.Media" if url else "FactButcher Human Benchmark",
        "source_url": url,
        "source_license": "CC BY 4.0" if url else None,
        "source_license_url": None,
    }


ROWS = [
    row("cand_1", "TRUE", "2026-06-10"),
    row("p-2", "FALSE", "2026-06-20", url=PROVERENO.format("2026/06/20")),
    row("cand_3", "MIXED", "2026-06-11", acceptable=["MIXED", "FALSE"]),
    row("h1_r4", "FALSE", None),
    row("cand_5", "TRUE", "2026-05-30"),
    row("p-6", "FALSE", "2026-06-02", url=PROVERENO.format("2026/05/28")),
    row("cand_7", "MIXED", "2026-07-01"),
    row("cand_8", "INSUFFICIENT_EVIDENCE", "2026-07-02"),
]


def snapshot(tmp: str) -> Path:
    root = Path(tmp) / "snap"
    (root / "data").mkdir(parents=True)
    lines = [json.dumps(r, ensure_ascii=False) for r in ROWS]
    (root / "data" / "factbutcher_benchmark_v1.jsonl").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return root


class FactButcherTest(unittest.TestCase):
    def test_date_and_ambiguity_filters(self):
        with tempfile.TemporaryDirectory() as tmp:
            items = list(factbutcher.candidates(snapshot(tmp)))
        self.assertEqual([c.source_item_id for c in items], ["cand_1", "p-2", "cand_7"])
        self.assertEqual([c.gold for c in items], ["accurate", "inaccurate", "mixed"])
        for item in items:
            self.assertEqual(validate(item, factbutcher.SPEC), [])
            self.assertEqual(set(item.state), {"claim", "reference_date"})
            self.assertEqual(item.date, item.state["reference_date"])
            self.assertEqual(item.balance_label, item.gold)
        self.assertEqual(items[0].group_id, "cand_1")
        self.assertEqual(items[1].group_id, PROVERENO.format("2026/06/20"))
        self.assertEqual(
            list(items[0].question["criteria"]), ["accurate", "inaccurate", "mixed"]
        )

    def test_runner_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = snapshot(tmp)
            receipt = runner.run("factbutcher", root, Path(tmp) / "out.jsonl")
        self.assertEqual(receipt["invalid"], {})
        self.assertEqual(receipt["valid"], 3)
        self.assertEqual(receipt["tasks"]["factbutcher/verdict"]["label:mixed"], 1)


if __name__ == "__main__":
    unittest.main()
