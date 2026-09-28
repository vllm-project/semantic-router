from __future__ import annotations

import hashlib
import json
import os
import random
import stat
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

from training.data import audit_vitaminc_score as legacy
from training.model.data import canonical
from v2.data import overlap
from v2.data.textnorm import (
    compact,
    decode_canonical_json,
    is_cjk_heavy,
    normalize,
    text_leaves,
    word_tokens,
)

SYLLABLES = [c + v for c in "bcdfghjklmnprstvz" for v in "aeiou"]


def _vocabulary(size: int = 4000) -> list[str]:
    rng = random.Random(0)
    words: set[str] = set()
    while len(words) < size:
        words.add("".join(rng.choice(SYLLABLES) for _ in range(rng.randint(2, 4))))
    return sorted(words)


WORDS = _vocabulary()


def prose(seed: int, chars: int) -> str:
    rng = random.Random(seed)
    words: list[str] = []
    length = -1
    while length < chars:
        words.append(rng.choice(WORDS))
        length += len(words[-1]) + 1
    return " ".join(words)


def candidate(
    row_id: str, group: str, state: object, instructions: str = "Pick one."
) -> dict:
    return {
        "id": row_id,
        "group_id": group,
        "state": state,
        "instructions": instructions,
        "options": [
            {"key": "A", "description": "yes"},
            {"key": "B", "description": "no"},
        ],
    }


def protected(**roles: list[dict]) -> overlap.ProtectedLeaves:
    return overlap.protected_from_rows(roles)


class TextNormTest(unittest.TestCase):
    def test_normalize_and_compact(self) -> None:
        self.assertEqual(normalize("  Ｈｅｌｌｏ\tWORLD\n\u00a0ß  "), "hello world ss")
        self.assertEqual(
            compact("Hello, World! 你好，世界。 x-1"), "helloworld你好世界x1"
        )
        self.assertEqual(normalize("ﬁ"), "fi")

    def test_matches_legacy_rules(self) -> None:
        for value in ["Ｆｕｌｌ　width", "Straße  NAÏVE", "  a\u2028b  ", "数据 集"]:
            self.assertEqual(normalize(value), legacy.normalize(value))
            self.assertEqual(compact(value), legacy.compact(value))
        nested = {"a": ["x", {"k": "y"}], "b": 3, "c": None}
        self.assertEqual(text_leaves(nested), legacy.text_leaves(nested))

    def test_text_leaves_ignores_keys_and_decodes_canonical_json(self) -> None:
        value = {
            "question text": "value one",
            "n": 4,
            "list": ["two", {"deep": "three"}],
        }
        self.assertEqual(text_leaves(value), ["value one", "two", "three"])
        encoded = canonical([{"description": "alpha", "key": "A"}])
        self.assertEqual(text_leaves(encoded), [encoded])
        self.assertEqual(text_leaves(encoded, decode_json=True), ["alpha", "A"])
        self.assertIsNone(decode_canonical_json('{"b": 1, "a": 2}'))
        self.assertIsNone(decode_canonical_json("[not json"))
        self.assertEqual(decode_canonical_json('{"a":2,"b":1}'), {"a": 2, "b": 1})

    def test_word_tokens_strip_edge_punctuation(self) -> None:
        self.assertEqual(
            word_tokens("“Hello,” said (the) U.S. — «Été»! ... can't 3.5%"),
            ["hello", "said", "the", "u.s", "été", "can't", "3.5"],
        )

    def test_cjk_heavy(self) -> None:
        self.assertTrue(is_cjk_heavy("这是一个中文句子 with a few words"))
        self.assertTrue(is_cjk_heavy("日本語のテキストです"))
        self.assertTrue(is_cjk_heavy("한국어 문장입니다"))
        self.assertFalse(is_cjk_heavy("An English sentence with 中"))
        self.assertFalse(is_cjk_heavy("!!!"))


class WindowTest(unittest.TestCase):
    def test_last_window_is_end_aligned(self) -> None:
        self.assertEqual(overlap.window_starts(350, 400, 200), [0])
        self.assertEqual(overlap.window_starts(1000, 400, 200), [0, 200, 400, 600])
        self.assertEqual(overlap.window_starts(1050, 400, 200), [0, 200, 400, 600, 650])

    def test_candidate_windows_must_fit_inside_protected_windows(self) -> None:
        with self.assertRaises(ValueError):
            overlap.Params(candidate_window=250)


class ScanMethodTest(unittest.TestCase):
    def test_long_passage_caught_by_l_and_n_not_s(self) -> None:
        document = prose(1, 3000)
        tokens = document.split()
        reference = protected(
            typed_dev=[
                {
                    "id": "p-long",
                    "state": document,
                    "questions": {"q": {"type": "choice", "instructions": "Pick one."}},
                }
            ]
        )
        for fraction in (0.0, 0.07, 0.21, 0.4, 0.63, 0.8):
            start = int(fraction * (len(tokens) - 90))
            end = start + 1
            while len(" ".join(tokens[start:end])) < 450:
                end += 1
            passage = " ".join(tokens[start:end])
            self.assertGreaterEqual(len(passage), 450)
            self.assertIn(passage, document)
            private, public = overlap.scan([candidate("c1", "g1", passage)], reference)
            record = private["groups"]["g1"]
            self.assertEqual(record["methods"], ["L", "N"], fraction)
            self.assertEqual(record["protected_ids"], {"typed_dev": ["p-long"]})
            self.assertEqual(record["counts"]["S"], 0)
            self.assertGreater(record["counts"]["N_word"], 0)
            self.assertGreater(record["counts"]["N_char"], 0)
            self.assertEqual(
                public["protected"]["distinct_leaves_by_length"]["gt_800"], 1
            )
            self.assertGreater(public["protected"]["long_leaf_windows_scanned"], 10)
            self.assertEqual(
                public["protected_leaves_over_800_chars_not_near_scanned"], 0
            )

    def test_near_copy_of_long_leaf_with_edits_is_caught_by_l(self) -> None:
        document = prose(2, 4000)
        tokens = document.split()
        copied = tokens[200:330]
        for index in (10, 60, 110):
            copied[index] = "zzzqqq"
        reference = protected(css_pilot=[{"id": "p", "state": document}])
        private, _ = overlap.scan([candidate("c", "g", " ".join(copied))], reference)
        self.assertIn("L", private["groups"]["g"]["methods"])

    def test_unrelated_text_is_not_flagged(self) -> None:
        reference = protected(
            typed_dev=[
                {"id": f"p{i}", "state": prose(100 + i, 900 if i % 2 else 500)}
                for i in range(6)
            ]
        )
        rows = [candidate(f"c{i}", f"g{i}", prose(500 + i, 1200)) for i in range(4)]
        private, public = overlap.scan(rows, reference)
        self.assertEqual(private["groups"], {})
        self.assertEqual(overlap.quarantine_groups(private), [])
        self.assertEqual(public["flagged"]["quarantine"], {"groups": 0, "rows": 0})

    def test_exact_leaf_and_whole_state(self) -> None:
        passage = prose(3, 300)
        state = {"table": [{"x": 1, "y": "two"}, {"x": 3, "y": "four"}], "note": "tiny"}
        other = {"table": [{"x": 2, "y": "two"}, {"x": 3, "y": "four"}], "note": "tiny"}
        reference = protected(
            rights_clean_select=[
                {"id": "s1", "state": passage},
                {"id": "s2", "state": canonical(state), "instructions": "Pick one."},
            ]
        )
        rows = [
            candidate("c1", "g-leaf", {"passage": passage.upper()}),
            candidate("c2", "g-state", state),
            candidate("c3", "g-none", other),
        ]
        private, _ = overlap.scan(rows, reference)
        self.assertEqual(private["groups"]["g-leaf"]["methods"], ["E", "N"])
        self.assertEqual(private["groups"]["g-leaf"]["counts"]["E_leaf"], 1)
        self.assertEqual(private["groups"]["g-leaf"]["counts"]["S"], 0)
        self.assertEqual(private["groups"]["g-state"]["methods"], ["E"])
        self.assertEqual(private["groups"]["g-state"]["counts"]["E_state"], 1)
        self.assertEqual(
            private["groups"]["g-state"]["protected_ids"],
            {"rights_clean_select": ["s2"]},
        )
        self.assertNotIn("g-none", private["groups"])

    def test_short_near_duplicate_is_s(self) -> None:
        original = prose(4, 400)
        tokens = original.split()
        tokens[5] = "qqqzzz"
        reference = protected(jevbench_public231=[{"id": "j", "state": original}])
        private, _ = overlap.scan([candidate("c", "g", " ".join(tokens))], reference)
        self.assertEqual(private["groups"]["g"]["counts"]["S"], 1)
        self.assertEqual(private["groups"]["g"]["counts"]["E_leaf"], 0)
        self.assertIn("S", private["groups"]["g"]["methods"])

    def test_s_rule_matches_legacy_screen(self) -> None:
        leaves = [("role_a", prose(1000 + i, 120 + 25 * i)) for i in range(18)]
        leaves += [("role_b", prose(2000 + i, 200 + 30 * i)) for i in range(18)]
        leaves += [("role_b", prose(2999, 2400))]
        rng = random.Random(7)
        rows, legacy_rows = [], []
        for page in range(60):
            role, source = leaves[page % len(leaves)]
            tokens = source.split()
            if page < 16:
                for _ in range(rng.randint(1, 3)):
                    tokens[rng.randrange(len(tokens))] = rng.choice(WORDS)
                evidence = " ".join(tokens)
            elif page < 24:
                evidence = source
            elif page < 32:
                evidence = prose(3000 + page, 300) + " " + source
            else:
                evidence = prose(4000 + page, rng.randint(60, 900))
            claim = prose(5000 + page, 80)
            legacy_rows.append(
                {"claim": claim, "evidence": evidence, "page": f"p{page}"}
            )
            rows.append(
                {
                    "id": f"r{page}",
                    "group_id": f"p{page}",
                    "state": evidence,
                    "instructions": claim,
                    "options": [],
                }
            )
        normalized = [
            (role, legacy.normalize(text))
            for role, text in leaves
            if len(legacy.compact(text)) >= 20
        ]
        expected = legacy.overlap_screen(legacy_rows, normalized)
        reference = overlap.protected_from_rows(
            {
                role: [
                    {"id": f"{role}-{i}", "state": text}
                    for i, (owner, text) in enumerate(leaves)
                    if owner == role
                ]
                for role in ("role_a", "role_b")
            }
        )
        private, _ = overlap.scan(rows, reference)
        observed: dict[str, dict[str, set[str]]] = {}
        for group, record in private["groups"].items():
            for method, kind in (("E", "exact"), ("S", "near")):
                for role in record["by_method"].get(method, {}):
                    observed.setdefault(role, {"exact": set(), "near": set()})[
                        kind
                    ].add(group)
        self.assertEqual(
            {
                role: {kind: len(pages) for kind, pages in kinds.items()}
                for role, kinds in observed.items()
            },
            expected["suspected_page_groups_by_role"],
        )
        self.assertGreater(
            expected["suspected_page_groups_by_role"]["role_a"]["near"], 5
        )


class BoilerplateTest(unittest.TestCase):
    def test_gram_in_five_protected_rows_never_quarantines(self) -> None:
        disclaimer = prose(9, 160)
        reference = protected(
            css15_goldfree=[
                {"id": f"p{i}", "state": prose(200 + i, 420) + " " + disclaimer}
                for i in range(5)
            ]
        )
        rows = [candidate("c", "g", prose(300, 420) + " " + disclaimer)]
        private, public = overlap.scan(rows, reference)
        self.assertEqual(private["groups"], {})
        self.assertGreater(private["boilerplate"]["N_word"]["units"], 0)
        self.assertEqual(private["boilerplate"]["N_word"]["groups"], 1)
        self.assertEqual(public["flagged"]["quarantine"]["groups"], 0)

    def test_gram_in_four_protected_rows_quarantines(self) -> None:
        shared = prose(9, 160)
        reference = protected(
            css15_goldfree=[
                {"id": f"p{i}", "state": prose(200 + i, 420) + " " + shared}
                for i in range(4)
            ]
        )
        rows = [candidate("c", "g", prose(300, 420) + " " + shared)]
        private, _ = overlap.scan(rows, reference)
        self.assertEqual(private["groups"]["g"]["methods"], ["N"])
        self.assertEqual(
            private["groups"]["g"]["protected_ids"],
            {"css15_goldfree": ["p0", "p1", "p2", "p3"]},
        )

    def test_gram_in_five_candidate_groups_never_quarantines(self) -> None:
        shared = prose(10, 160)
        reference = protected(
            typed_dev=[{"id": "p", "state": prose(400, 420) + " " + shared}]
        )
        rows = [
            candidate(f"c{i}", f"g{i}", prose(600 + i, 420) + " " + shared)
            for i in range(5)
        ]
        private, _ = overlap.scan(rows, reference)
        self.assertEqual(private["groups"], {})
        self.assertEqual(private["boilerplate"]["N_char"]["groups"], 5)
        private, _ = overlap.scan(rows[:4], reference)
        self.assertEqual(overlap.quarantine_groups(private), ["g0", "g1", "g2", "g3"])

    def test_template_leaf_in_five_candidate_groups_is_boilerplate(self) -> None:
        template = "Choose the option that is best supported by the passage above."
        reference = protected(
            rights_clean_train=[
                {"id": "t", "state": prose(1, 300), "instructions": template}
            ]
        )
        rows = [
            candidate(f"c{i}", f"g{i}", prose(700 + i, 300), template) for i in range(5)
        ]
        private, _ = overlap.scan(rows, reference)
        self.assertEqual(private["groups"], {})
        self.assertEqual(private["boilerplate"]["E_leaf"], {"units": 1, "groups": 5})
        self.assertEqual(private["boilerplate_units"][0]["kind"], "E_leaf")
        self.assertEqual(private["boilerplate_units"][0]["candidate_groups"], 5)


class InventoryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)

    def tearDown(self) -> None:
        self.directory.cleanup()

    def _file(self, name: str, rows: list[dict]) -> Path:
        path = self.root / name
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        return path

    def _inventory(self, entries: list[tuple[str, Path]], tamper: bool = False) -> Path:
        payload = [
            {
                "role": role,
                "path": path.name,
                "sha256": (
                    ("0" * 64)
                    if tamper
                    else hashlib.sha256(path.read_bytes()).hexdigest()
                ),
            }
            for role, path in entries
        ]
        inventory = self.root / "inventory.json"
        inventory.write_text(json.dumps(payload), encoding="utf-8")
        return inventory

    def test_projected_and_native_rows_load_and_match(self) -> None:
        option = "The ledger balance stays positive after the final transfer."
        projected = self._file(
            "select.jsonl",
            [
                {
                    "id": "select-row-one",
                    "state": prose(20, 200),
                    "instructions": canonical(
                        {"task_type": "choice", "instructions": "Pick."}
                    ),
                    "options": canonical([{"key": "A", "description": option}]),
                }
            ],
        )
        native = self._file(
            "dev.jsonl",
            [
                {
                    "id": "dev-row-one",
                    "family": "ledger_family",
                    "state": {
                        "target": {"entity": "x", "item": "y"},
                        "text": prose(21, 200),
                    },
                    "questions": {
                        "label": {"type": "noul", "instructions": prose(22, 120)}
                    },
                }
            ],
        )
        leaves = overlap.load_protected_inventory(
            self._inventory([("rights_clean_select", projected), ("typed_dev", native)])
        )
        self.assertEqual(
            leaves.refs,
            (("rights_clean_select", "select-row-one"), ("typed_dev", "dev-row-one")),
        )
        self.assertIn(option, [text for _, text in leaves.leaves])
        self.assertEqual(len(leaves.inventory_sha256), 64)
        row = candidate("candidate-row", "candidate-group", prose(23, 200))
        row["options"] = [
            {"key": "A", "description": option.upper()},
            {"key": "B", "description": "no"},
        ]
        private, public = overlap.scan([row], leaves)
        self.assertEqual(
            private["groups"]["candidate-group"]["by_method"]["E"],
            {"rights_clean_select": ["select-row-one"]},
        )
        text = json.dumps(public)
        for secret in ("row-one", "candidate-", "ledger", "positive after"):
            self.assertNotIn(secret, text)

    def test_hash_mismatch_rejected(self) -> None:
        path = self._file("a.jsonl", [{"id": "a", "state": "text"}])
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            overlap.load_protected_inventory(
                self._inventory([("typed_dev", path)], tamper=True)
            )

    def test_missing_file_rejected(self) -> None:
        path = self._file("a.jsonl", [{"id": "a", "state": "text"}])
        inventory = self._inventory([("typed_dev", path)])
        path.unlink()
        with self.assertRaisesRegex(ValueError, "missing"):
            overlap.load_protected_inventory(inventory)

    def test_answer_bearing_keys_rejected(self) -> None:
        for bad in (
            {"id": "a", "state": "text", "label": 1},
            {"id": "a", "state": "text", " Gold ": "A"},
            {"id": "a", "state": "text", "teacher_probs": {"A": 1.0}},
            {
                "id": "a",
                "state": "s",
                "questions": {"q": {"type": "choice", "answer": "A"}},
            },
        ):
            path = self._file("bad.jsonl", [bad])
            with self.assertRaisesRegex(ValueError, "answer-bearing"):
                overlap.load_protected_inventory(self._inventory([("typed_dev", path)]))

    def test_duplicate_ids_rejected(self) -> None:
        path = self._file(
            "dup.jsonl", [{"id": "a", "state": "x"}, {"id": "a", "state": "y"}]
        )
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            overlap.load_protected_inventory(self._inventory([("typed_dev", path)]))


class SelfScanTest(unittest.TestCase):
    def test_within_set_pairs_only_cross_groups(self) -> None:
        passage = prose(30, 700)
        rows = [
            candidate("a1", "ga", passage),
            candidate("a2", "ga", passage),
            candidate("b1", "gb", {"text": passage}),
            candidate("c1", "gc", prose(31, 700)),
        ]
        private, public = overlap.self_scan(rows)
        self.assertEqual(private["mode"], "within")
        self.assertEqual([(p["a"], p["b"]) for p in private["pairs"]], [("ga", "gb")])
        self.assertIn("E", private["pairs"][0]["methods"])
        self.assertEqual(public["pairs"], 1)
        self.assertEqual(public["candidate_groups_in_pairs"], 2)

    def test_across_sets(self) -> None:
        document = prose(32, 2500)
        tokens = document.split()
        train = [
            candidate("t1", "gt", " ".join(tokens[50:130])),
            candidate("t2", "gu", prose(33, 400)),
        ]
        held_out = [candidate("h1", "gh", document)]
        private, public = overlap.self_scan(train, held_out)
        self.assertEqual(private["mode"], "across")
        self.assertEqual([(p["a"], p["b"]) for p in private["pairs"]], [("gt", "gh")])
        self.assertEqual(private["pairs"][0]["methods"], ["L", "N"])
        self.assertEqual(
            public["by_method"]["L"], {"pairs": 1, "a_groups": 1, "b_groups": 1}
        )


class DeterminismAndCliTest(unittest.TestCase):
    def _fixture(self) -> tuple[list[dict], dict[str, list[dict]]]:
        documents = [prose(40 + i, 1500 + 200 * i) for i in range(6)]
        roles = {
            "typed_dev": [{"id": f"d{i}", "state": documents[i]} for i in range(3)],
            "css_pilot": [
                {
                    "id": f"c{i}",
                    "state": documents[i],
                    "instructions": prose(60 + i, 90),
                }
                for i in range(3, 6)
            ],
        }
        rows = []
        for i in range(24):
            source = documents[i % 6].split()
            state = " ".join(source[i : i + 60]) if i % 3 == 0 else prose(80 + i, 900)
            rows.append(candidate(f"r{i}", f"g{i // 2}", state, prose(60 + i % 4, 90)))
        return rows, roles

    def test_workers_do_not_change_receipts(self) -> None:
        rows, roles = self._fixture()
        reference = overlap.protected_from_rows(roles)
        single = overlap.scan(rows, reference, workers=1)
        double = overlap.scan(rows, reference, workers=2)
        self.assertEqual(canonical(single[0]), canonical(double[0]))
        self.assertEqual(canonical(single[1]), canonical(double[1]))
        self.assertTrue(single[0]["groups"])
        self.assertEqual(
            canonical(overlap.self_scan(rows, workers=2)),
            canonical(overlap.self_scan(rows)),
        )

    def test_cli_writes_private_files_and_refuses_overwrite(self) -> None:
        rows, roles = self._fixture()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entries = []
            for role, members in roles.items():
                path = root / f"{role}.jsonl"
                path.write_text(
                    "".join(json.dumps(row) + "\n" for row in members), encoding="utf-8"
                )
                entries.append(
                    {
                        "role": role,
                        "path": str(path),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                )
            inventory = root / "inventory.json"
            inventory.write_text(json.dumps(entries), encoding="utf-8")
            first, second = root / "a.jsonl", root / "b.jsonl"
            first.write_text(
                "".join(json.dumps(row) + "\n" for row in rows[:12]), encoding="utf-8"
            )
            second.write_text(
                "".join(json.dumps(row) + "\n" for row in rows[12:]), encoding="utf-8"
            )
            private_path, public_path = (
                root / "out.private.json",
                root / "out.public.json",
            )
            argv = [
                "--candidates",
                str(first),
                "--candidates",
                str(second),
                "--protected-inventory",
                str(inventory),
                "--private-receipt",
                str(private_path),
                "--public-receipt",
                str(public_path),
                "--workers",
                "2",
            ]
            with redirect_stdout(StringIO()):
                self.assertEqual(overlap.main(argv), 0)
            for path in (private_path, public_path):
                self.assertEqual(stat.S_IMODE(os.stat(path).st_mode), 0o600)
            private = json.loads(private_path.read_text(encoding="utf-8"))
            public = json.loads(public_path.read_text(encoding="utf-8"))
            expected, _ = overlap.scan(
                rows, overlap.load_protected_inventory(inventory)
            )
            self.assertEqual(private["groups"], expected["groups"])
            self.assertEqual(
                [item["rows"] for item in public["candidate_files"]], [12, 12]
            )
            self.assertNotIn(str(root), json.dumps(public))
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                with self.assertRaises(SystemExit):
                    overlap.main(argv)
            self_private, self_public = root / "self.private.json", root / "self.json"
            with redirect_stdout(StringIO()):
                overlap.main(
                    [
                        "--self-scan",
                        "--candidates",
                        str(first),
                        "--candidates",
                        str(second),
                        "--private-receipt",
                        str(self_private),
                        "--public-receipt",
                        str(self_public),
                    ]
                )
            expected_self, _ = overlap.self_scan(rows[:12], rows[12:])
            written = json.loads(self_private.read_text(encoding="utf-8"))
            self.assertEqual(written["pairs"], expected_self["pairs"])
            self.assertEqual(written["mode"], "across")


if __name__ == "__main__":
    unittest.main()
