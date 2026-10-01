from __future__ import annotations

import collections
import unittest

from training.model.data import validate_row
from v2.data.ib3 import audit, families as fam, index_guard


def counts(rows):
    return collections.Counter(
        (r["audit_metadata"]["ib3"]["cell"], r["label"]) for r in rows
    )


class CellBalance(unittest.TestCase):
    def assert_balanced(self, rows):
        seen = counts(rows)
        for cell in {c for c, _ in seen}:
            self.assertEqual(seen[(cell, 0)], seen[(cell, 1)], cell)

    def test_wpd_shape_cells_and_filters(self):
        records = [
            {"url": "http://example.com/a/login.php", "status": "phishing"},
            {"url": "http://example.org/a/about.php", "status": "legitimate"},
            {"url": "http://203.0.113.5/x", "status": "phishing"},
            {"url": "https://bit.ly/abc", "status": "phishing"},
            {"url": "https://www.only-legit.com", "status": "legitimate"},
        ]
        report = collections.Counter()
        rows = fam.wpd(records, report)
        self.assertEqual(len(rows), 2)
        self.assert_balanced(rows)
        self.assertEqual(report["drop_ipv4"], 1)
        self.assertEqual(report["drop_shortener"], 1)

    def test_phiu_shared_shape_only(self):
        records = [
            {"URL": "https://www.shop-a.com", "Title": "Shop A", "label": "1"},
            {"URL": "https://www.login-b.com", "Title": "Sign in", "label": "0"},
            {"URL": "https://login-c.com/x", "Title": "Sign in", "label": "0"},
            {"URL": "https://www.d.com", "Title": "404 Not Found", "label": "0"},
        ]
        report = collections.Counter()
        rows = fam.phiu(records, report)
        self.assertEqual(sorted(r["label"] for r in rows), [0, 1])
        self.assertEqual(report["drop_shape_cell"], 1)
        self.assertEqual(report["drop_title"], 1)

    def test_haluqa_drops_sentence_answers(self):
        records = [
            {
                "knowledge": "K one.",
                "question": "Q1?",
                "right_answer": "Paris",
                "hallucinated_answer": "Lyon",
            },
            {
                "knowledge": "K two.",
                "question": "Q2?",
                "right_answer": "Rome",
                "hallucinated_answer": "It is Milan.",
            },
        ]
        report = collections.Counter()
        rows = fam.haluqa(records, report)
        self.assertEqual(len(rows), 2)
        self.assert_balanced(rows)
        self.assertTrue(
            all(r["state"]["answer"] in ("Paris", "Lyon", "Rome") for r in rows)
        )

    def test_mqa_formula_verified_key(self):
        record = {
            "Problem": "a train running at the speed of 60 km / hr crosses a pole in 9 seconds . what is the length of the train ?",
            "options": "a ) 120 metres , b ) 180 metres , c ) 324 metres , d ) 150 metres , e ) 140 metres",
            "correct": "d",
            "linear_formula": "multiply(n0,const_0_2778)|multiply(n1,#0)|",
        }
        report = collections.Counter()
        rows = fam.mqa([record], report)
        self.assertEqual(report["verified"], 1)
        self.assertEqual(sorted(r["label"] for r in rows), [0, 1])
        gold = [r for r in rows if r["label"] == 1][0]
        self.assertEqual(gold["state"]["proposed_answer"], "150 metres")
        self.assertTrue(gold["state"]["problem"].endswith("length of the train?"))
        wrong = dict(record, correct="a")
        report = collections.Counter()
        self.assertEqual(fam.mqa([wrong], report), [])
        self.assertEqual(report["drop_key_not_verified"], 1)

    def test_maud_option_cells(self):
        records = []
        for i, answer in enumerate(["Yes", "No", "Yes", "No"]):
            records.append(
                {
                    "data_type": "main",
                    "contract_name": f"c{i}",
                    "text": f"excerpt {i}",
                    "answer": answer,
                    "question": "Q",
                    "subquestion": "<NONE>",
                }
            )
        report = collections.Counter()
        rows = fam.maud(records, report)
        self.assertEqual(len(rows), 8)
        self.assert_balanced(rows)


class UrlGuard(unittest.TestCase):
    def test_normalization_and_perturbation(self):
        raw = "http://www.Example.com/Login?x=1"
        self.assertEqual(
            index_guard.norm_url(raw), ("example.com/login?x=1", "example.com")
        )
        self.assertEqual(
            index_guard.norm_url(index_guard.perturb(raw))[0], "example.com/login?x=1"
        )
        ref = {"url": {index_guard.h64("example.com/login?x=1", "url"): 0}, "host": {}}
        row = {
            "state": {"url": index_guard.perturb(raw)},
            "instructions": "",
            "options": [],
        }
        self.assertTrue(index_guard.flags(index_guard.candidate_urls(row), ref))


class AuditCopy(unittest.TestCase):
    def test_hypothesis_field_renamed(self):
        rows = fam.haluqa(
            [
                {
                    "knowledge": "K.",
                    "question": "Q?",
                    "right_answer": "A",
                    "hallucinated_answer": "B",
                }
            ],
            collections.Counter(),
        )
        for item in rows:
            copy = audit.audit_copy(item)
            self.assertEqual(list(copy["state"]), ["passages", "question", "claim"])
            validate_row(copy, "train")
        url_row = {"family": "wpd", "state": {"url": "http://x.com"}}
        self.assertEqual(audit.audit_copy(url_row)["state"], {"url": "http://x.com"})


if __name__ == "__main__":
    unittest.main()
