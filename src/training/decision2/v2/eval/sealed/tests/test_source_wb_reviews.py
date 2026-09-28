from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from v2.eval.sealed import candidates as runner
from v2.eval.sealed.schema import validate
from v2.eval.sealed.sources import wb_reviews

HEADER = (
    "review_id",
    "product_id",
    "category",
    "category_label",
    "rating",
    "text",
    "pros",
    "cons",
    "date",
)
ROWS = [
    (
        "1",
        "10",
        "home",
        "Дом",
        "5",
        "Отличный чайник, кипятит быстро.",
        "Цена",
        "",
        "2026-06-15",
    ),
    ("2", "10", "home", "Дом", "1", "", "", "Сломался через день.", "2026-07-01"),
    (
        "3",
        "11",
        "food",
        "Еда",
        "3",
        "Обычный вкус, ничего особенного.",
        "",
        "",
        "2026-05-31",
    ),
    ("4", "12", "food", "Еда", "4", "Вкусно, но дорого.", "", "", ""),
    ("5", "12", "food", "Еда", "0", "Нормально.", "", "", "2026-06-02"),
    ("6", "13", "kids", "Детям", "2", " ", "", "", "2026-06-03"),
    (
        "7",
        "14",
        "sport",
        "Спорт",
        "2",
        "Мяч сдулся,\nвернула продавцу.",
        "",
        "Качество",
        "2026-07-20",
    ),
    ("8", "15", "sport", "Спорт", "4", "очень " * 5_000, "", "", "2026-07-21"),
]


def snapshot(tmp: str) -> Path:
    root = Path(tmp) / "snap"
    root.mkdir()
    with (root / "filtered.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(HEADER)
        writer.writerows(ROWS)
    return root


class WbReviewsTest(unittest.TestCase):
    def test_filters_levels_and_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            items = list(wb_reviews.candidates(snapshot(tmp)))
        self.assertEqual([c.source_item_id for c in items], ["1", "2", "7"])
        for item in items:
            self.assertEqual(validate(item, wb_reviews.SPEC), [])
            self.assertEqual(item.balance_label, str(item.gold))
            self.assertGreaterEqual(item.date, "2026-06-01")
            self.assertEqual(item.overlap_texts, list(item.state.values()))
        first, second, third = items
        self.assertEqual((first.gold, second.gold, third.gold), (4, 0, 1))
        self.assertEqual(
            first.state, {"review": "Отличный чайник, кипятит быстро.", "pros": "Цена"}
        )
        self.assertEqual(second.state, {"cons": "Сломался через день."})
        self.assertEqual(third.state["review"], "Мяч сдулся,\nвернула продавцу.")
        self.assertEqual((first.group_id, second.group_id), ("10", "10"))
        self.assertEqual(len(first.question["criteria"]), 5)

    def test_runner_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = snapshot(tmp)
            receipt = runner.run("wb_reviews", root, Path(tmp) / "out.jsonl")
        self.assertEqual(receipt["invalid"], {})
        self.assertEqual(receipt["valid"], 3)
        task = receipt["tasks"]["wb_reviews/star_rating"]
        self.assertEqual((task["groups"], task["label:0"], task["label:4"]), (2, 1, 1))


if __name__ == "__main__":
    unittest.main()
