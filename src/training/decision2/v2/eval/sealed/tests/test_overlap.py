from __future__ import annotations

import contextlib
import gzip
import hashlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any

from v2.eval.sealed import overlap
from v2.eval.sealed.overlap import (
    Params,
    expand,
    load_protected,
    scan,
    shingle_key,
    shingles,
    string_leaves,
    tokens,
    verdict,
)
from v2.eval.sealed.schema import normalized

try:
    import pyarrow
    import pyarrow.parquet
except ImportError:
    pyarrow = None


def words(stem: str, count: int, start: int = 0) -> str:
    return " ".join(f"{stem}{i}" for i in range(start, start + count))


TEXT = words("tok", 30)


def candidate(item: str, state: Any, overlap_texts: tuple[str, ...] = ()) -> dict:
    return {
        "source": "fake",
        "task": "fake/flag",
        "source_item_id": item,
        "group_id": f"g-{item}",
        "balance_label": "True",
        "language": "en",
        "state": state,
        "question": {
            "type": "noul",
            "instructions": "Decide.",
            "criteria": {"true": "Yes", "false": "No"},
        },
        "gold": True,
        "overlap_texts": list(overlap_texts),
        "date": "2026-07-01",
    }


class TextTest(unittest.TestCase):
    def test_normalised_tokens(self):
        self.assertEqual(
            tokens(normalized("Ｈｅｌｌｏ,  WORLD\tfoo_bar")),
            ["hello", "world", "foo_bar"],
        )

    def test_cjk_characters_are_tokens(self):
        self.assertEqual(tokens("東京は晴れ・です"), list("東京は晴れです"))
        self.assertEqual(tokens("abc東京def"), ["abc", "東", "京", "def"])
        self.assertEqual(tokens("abc東京def", Params(cjk_split=False)), ["abc東京def"])
        self.assertEqual(tokens("안녕하세요 세계"), ["안녕하세요", "세계"])

    def test_shingle_counts(self):
        self.assertEqual(len(shingles(words("w", 20).split())), 13)
        self.assertEqual(len(shingles(words("w", 8).split())), 1)
        seven = words("w", 7).split()
        self.assertEqual(shingles(seven), {shingle_key(seven)})
        self.assertEqual(len(shingles(words("w", 5).split())), 1)
        self.assertEqual(shingles(words("w", 4).split()), set())

    def test_string_leaves(self):
        value = {
            "key text": ["a", {"b": 1, "c": "d"}],
            "json": json.dumps({"inner": ["e"], "n": 2}),
            "broken": "{not json}",
        }
        self.assertEqual(list(string_leaves(value)), ["a", "d", "e", "{not json}"])

    def test_verdict_thresholds(self):
        self.assertEqual(verdict(0.5), "OVERLAP")
        self.assertEqual(verdict(0.4999), "REVIEW")
        self.assertEqual(verdict(0.2), "REVIEW")
        self.assertEqual(verdict(0.1999), "CLEAN")
        self.assertEqual(verdict(0.0, 3), "OVERLAP")
        strict = Params(exact_min_tokens=8)
        self.assertEqual(verdict(0.0, 3, strict), "REVIEW")
        self.assertEqual(verdict(0.0, 8, strict), "OVERLAP")
        self.assertEqual(verdict(0.6, 3, strict), "OVERLAP")

    def test_params_validation(self):
        for bad in (
            {"min_tokens": 9},
            {"contains_tokens": 9},
            {"review": 0.6},
            {"min_chars": 0},
            {"exact_min_tokens": -1},
        ):
            with self.assertRaises(ValueError):
                Params(**bad)


class Case(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.runs = 0

    def tearDown(self):
        self.tmp.cleanup()

    def write_jsonl(self, name: str, rows: list[Any]) -> Path:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        return path

    def run_scan(self, candidates, corpora, **kwargs):
        self.runs += 1
        protected = self.write_jsonl(f"candidates-{self.runs}.jsonl", candidates)
        receipt, hits = scan(protected, corpora, **kwargs)
        return receipt, {hit["id"]: hit for hit in hits}


class ProtectedTest(Case):
    def test_texts_are_deduplicated_and_counted(self):
        path = self.write_jsonl(
            "c.jsonl",
            [
                candidate("a", {"t": TEXT, "u": "short"}, (TEXT, " " + TEXT.upper())),
                candidate("b", "tiny"),
            ],
        )
        protected = load_protected(path)
        self.assertEqual(protected.ids, ["fake/flag|a", "fake/flag|b"])
        self.assertEqual(protected.sizes, [23, 0])
        self.assertEqual(protected.stats["texts"], 3)
        self.assertEqual(protected.stats["unscreenable"], 1)
        self.assertEqual(protected.stats["contains_texts"], 1)

    def test_bad_candidate_files(self):
        for name, rows in (
            ("dup.jsonl", [candidate("a", TEXT), candidate("a", TEXT)]),
            ("noid.jsonl", [{"task": "t", "state": TEXT}]),
        ):
            with self.assertRaises(ValueError):
                load_protected(self.write_jsonl(name, rows))
        blank = self.root / "blank.jsonl"
        blank.write_text(json.dumps(candidate("a", TEXT)) + "\n\n")
        with self.assertRaises(ValueError):
            load_protected(blank)


class ScanTest(Case):
    def test_containment_is_per_single_row(self):
        text = words("alpha", 20)
        corpus = self.write_jsonl(
            "rows.jsonl",
            [{"t": words("alpha", 12)}, {"t": words("alpha", 14, start=6)}],
        )
        receipt, hits = self.run_scan(
            [candidate("a", {"t": text})], [("train", [str(corpus)])]
        )
        hit = hits["fake/flag|a"]
        self.assertEqual((hit["matched"], hit["shingles"], hit["row"]), (7, 13, 1))
        self.assertEqual(hit["verdict"], "OVERLAP")
        self.assertIsNone(hit["exact"])
        self.assertEqual(receipt["verdicts"], {"OVERLAP": 1, "REVIEW": 0, "CLEAN": 0})
        self.assertEqual(receipt["overlap_reasons"]["containment_only"], 1)

    def test_thresholds_per_label(self):
        text = words("beta", 17)
        corpora = []
        for label, count in (("one", 8), ("two", 9), ("five", 12)):
            path = self.write_jsonl(f"{label}.jsonl", [{"t": words("beta", count)}])
            corpora.append((label, [str(path)]))
        receipt, hits = self.run_scan([candidate("a", text)], corpora)
        hit = hits["fake/flag|a"]
        labels = hit["labels"]
        self.assertEqual(labels["one"]["verdict"], "CLEAN")
        self.assertEqual(labels["two"]["verdict"], "REVIEW")
        self.assertEqual(labels["two"]["containment"], 0.2)
        self.assertEqual(labels["five"]["verdict"], "OVERLAP")
        self.assertEqual((hit["label"], hit["containment"]), ("five", 0.5))
        self.assertEqual(receipt["corpora"]["two"]["verdicts"]["REVIEW"], 1)

    def test_exact_equal_for_texts_without_shingles(self):
        title, rule = "Supercalifragilistic Expialidocious", "-" * 24
        corpus = self.write_jsonl(
            "rows.jsonl",
            [
                {"t": "supercalifragilistic   EXPIALIDOCIOUS"},
                {"t": title + " again", "r": rule},
            ],
        )
        receipt, hits = self.run_scan(
            [
                candidate("a", {"title": title}),
                candidate("b", {"title": title}),
                candidate("c", {"rule": rule}),
            ],
            [("peer", [str(corpus)])],
        )
        hit = hits["fake/flag|a"]
        self.assertEqual((hit["shingles"], hit["verdict"]), (0, "OVERLAP"))
        self.assertEqual(
            hit["exact"],
            {
                "label": "peer",
                "kind": "equal",
                "file": str(corpus),
                "row": 0,
                "tokens": 2,
                "shared": 2,
            },
        )
        self.assertEqual(hits["fake/flag|c"]["verdict"], "CLEAN")
        self.assertEqual(receipt["protected"]["unscreenable"], 1)
        reasons = receipt["overlap_reasons"]
        self.assertEqual(
            (reasons["exact_only"], reasons["exact_only_shared_text"]), (2, 2)
        )
        self.assertEqual(reasons["exact_only_tokens"]["0-4"], 2)
        receipt, hits = self.run_scan(
            [candidate("a", {"title": title})],
            [("peer", [str(corpus)])],
            params=Params(exact_min_tokens=8),
        )
        self.assertEqual(hits["fake/flag|a"]["verdict"], "REVIEW")
        self.assertEqual(receipt["overlap_reasons"]["review_short_exact"], 1)

    def test_longest_exact_match_is_kept(self):
        short, long = "Quite a short sentence here", words("long", 14)
        corpus = self.write_jsonl(
            "rows.jsonl", [{"t": short}, {"t": "noise"}, {"t": long + " more"}]
        )
        _, hits = self.run_scan(
            [candidate("a", {"a": short, "b": long, "c": words("pad", 60)})],
            [("train", [str(corpus)])],
            params=Params(exact_min_tokens=8),
        )
        hit = hits["fake/flag|a"]
        self.assertEqual(hit["verdict"], "OVERLAP")
        self.assertEqual(
            [hit["exact"][key] for key in ("kind", "tokens", "row")],
            ["contains", 14, 2],
        )

    def test_contains_needs_twelve_tokens(self):
        other = words("other", 40)
        twelve, eleven = words("gamma", 12), words("delta", 11)
        corpus = self.write_jsonl(
            "rows.jsonl", [{"t": f"zz{twelve} tail"}, {"t": f"zz{eleven} tail"}]
        )
        _, hits = self.run_scan(
            [
                candidate("a", {"a": twelve, "b": other}),
                candidate("b", {"a": eleven, "b": other}),
            ],
            [("train", [str(corpus)])],
        )
        first, second = hits["fake/flag|a"], hits["fake/flag|b"]
        self.assertEqual((first["matched"], first["shingles"]), (4, 38))
        self.assertEqual(first["verdict"], "OVERLAP")
        self.assertEqual(
            [first["exact"][key] for key in ("kind", "tokens", "shared")],
            ["contains", 12, 1],
        )
        self.assertEqual(second["verdict"], "CLEAN")
        self.assertIsNone(second["exact"])

    def test_short_texts(self):
        corpus = self.write_jsonl(
            "rows.jsonl",
            [
                {"t": "one two three four"},
                {"t": "a b c d e f"},
                {"t": "One, two, three, four, five, six!"},
                {"t": "zero one two three four five six seven"},
                {"t": "東京は晴れですね"},
                {"t": "東京は晴れです"},
            ],
        )
        _, hits = self.run_scan(
            [
                candidate("four", {"t": "one two three four"}),
                candidate("tiny", {"t": "a b c d e f"}),
                candidate("six", {"t": "one two three four five six"}),
                candidate("cjk", {"t": "東京は晴れです"}),
            ],
            [("train", [str(corpus)])],
        )
        self.assertEqual(hits["fake/flag|four"]["verdict"], "CLEAN")
        for item, row in (("tiny", 1), ("six", 2), ("cjk", 5)):
            with self.subTest(item=item):
                hit = hits[f"fake/flag|{item}"]
                self.assertEqual((hit["verdict"], hit["row"]), ("OVERLAP", row))
                self.assertEqual((hit["shingles"], hit["exact"]), (1, None))

    def reader_corpus(self) -> tuple[list[tuple[str, list[str]]], dict[str, int]]:
        root = self.root / "corpus"
        root.mkdir()
        files: dict[str, tuple[str, int]] = {}

        def add(name: str, data: str | bytes, row: int) -> None:
            path = root / name
            path.write_bytes(data if isinstance(data, bytes) else data.encode())
            files[name] = (str(path), row)

        lines = lambda rows: "".join(json.dumps(r) + "\n" for r in rows)  # noqa: E731
        add(
            "a.jsonl",
            lines([{"x": "nothing to see in this row"}])
            + "\n"
            + lines([{"deep": {"list": [1, {"t": TEXT}]}}]),
            2,
        )
        add("b.jsonl", '{"x": 1}\ngarbage {' + TEXT + "\n", 1)
        add("c.jsonl.gz", gzip.compress(lines([{"a": "b"}, {"t": TEXT}]).encode()), 1)
        add("d.json", json.dumps([{"a": "short"}, {"t": [TEXT]}]), 1)
        add("e.json", json.dumps({"train": [{"a": 1}], "test": [{}, {"t": TEXT}]}), 2)
        add("f.json", json.dumps({"id1": {"t": "x"}, "id2": {"t": TEXT}}), 1)
        add("g.json", json.dumps({"t": TEXT, "n": 3}), 0)
        add("h.json", lines([{"a": 1}, {"t": TEXT}]), 1)
        add("i.csv", f'id,text\n1,other\n2,"prefix, {TEXT}"\n', 2)
        add("j.tsv", f"id\ttext\n1\t{TEXT}\n", 1)
        add("k.jsonl", lines([{"state": json.dumps({"p": TEXT, "q": "more"})}]), 0)
        corpora = [(name, [path]) for name, (path, _) in files.items()]
        return corpora, {name: row for name, (_, row) in files.items()}

    def test_reader_formats(self):
        corpora, rows = self.reader_corpus()
        receipt, hits = self.run_scan([candidate("a", {"t": TEXT})], corpora)
        labels = hits["fake/flag|a"]["labels"]
        self.assertEqual(sorted(labels), sorted(rows))
        for name, row in rows.items():
            with self.subTest(name=name):
                self.assertEqual(labels[name]["row"], row)
                self.assertEqual(labels[name]["containment"], 1.0)
                self.assertEqual(labels[name]["verdict"], "OVERLAP")
        self.assertEqual(labels["b.jsonl"]["exact"], "contains")
        self.assertEqual(labels["k.jsonl"]["exact"], "equal")
        self.assertEqual(receipt["corpora"]["b.jsonl"]["unparsed_lines"], 1)
        self.assertEqual(receipt["corpora"]["a.jsonl"]["rows"], 2)
        self.assertEqual(receipt["corpora"]["i.csv"]["rows"], 3)

    def test_parallel_matches_serial(self):
        corpora, _ = self.reader_corpus()
        candidates = [candidate("a", {"t": TEXT}), candidate("b", words("tok", 12))]
        serial = self.run_scan(candidates, corpora)[1]
        parallel = self.run_scan(candidates, corpora, workers=2)[1]
        self.assertEqual(serial, parallel)

    def test_chunked_jsonl_keeps_file_rows(self):
        rows = [{"n": i, "t": words("pad", 6, start=i)} for i in range(40)]
        rows[29] = {"n": 29, "t": TEXT}
        corpus = self.write_jsonl("big.jsonl", rows)
        results = []
        for chunk in (100, 16 << 20):
            receipt, hits = self.run_scan(
                [candidate("a", TEXT)],
                [("train", [str(corpus)])],
                params=Params(chunk_bytes=chunk),
            )
            results.append(
                (hits["fake/flag|a"]["row"], receipt["corpora"]["train"]["rows"])
            )
            tasks = receipt["resources"]["tasks"]
        self.assertEqual(results, [(29, 40), (29, 40)])
        self.assertEqual(tasks, 1)

    def test_split_gzip_keeps_file_rows(self):
        rows = [{"n": i, "t": words("pad", 6, start=i)} for i in range(40)]
        rows[29] = {"n": 29, "t": TEXT}
        corpus = self.root / "big.jsonl.gz"
        corpus.write_bytes(
            gzip.compress("".join(json.dumps(r) + "\n" for r in rows).encode())
        )
        results = []
        for chunk in (10, 16 << 20):
            receipt, hits = self.run_scan(
                [candidate("a", TEXT)],
                [("peer", [str(corpus)])],
                params=Params(chunk_bytes=chunk),
                workers=2,
            )
            results.append(
                (
                    hits["fake/flag|a"]["row"],
                    receipt["corpora"]["peer"]["rows"],
                    receipt["resources"]["tasks"] > 1,
                )
            )
        self.assertEqual(results, [(29, 40, True), (29, 40, False)])

    @unittest.skipIf(pyarrow is None, "pyarrow not installed")
    def test_parquet_rows(self):
        table = pyarrow.table(
            {
                "id": list(range(5)),
                "text": ["x"] * 5,
                "blob": [b"\0" * 64] * 5,
                "nested": [[{"a": "y"}]] * 4 + [[{"a": TEXT}]],
            }
        )
        path = self.root / "p.parquet"
        pyarrow.parquet.write_table(table, path, row_group_size=3)
        for chunk in (1, 16 << 20):
            _, hits = self.run_scan(
                [candidate("a", TEXT)],
                [("peer", [str(path)])],
                params=Params(chunk_bytes=chunk),
            )
            self.assertEqual(hits["fake/flag|a"]["row"], 4)

    def test_expand(self):
        root = self.root / "tree"
        (root / "sub").mkdir(parents=True)
        (root / ".cache").mkdir()
        for name in ("sub/a.jsonl", "b.parquet", "c.md", ".cache/d.jsonl", "e.JSON"):
            (root / name).write_text("[]")
        self.assertEqual(
            [Path(p).relative_to(root).as_posix() for p in expand(str(root))],
            ["b.parquet", "e.JSON", "sub/a.jsonl"],
        )
        self.assertEqual(len(expand(str(root / "**" / "*"))), 3)
        with self.assertRaises(ValueError):
            overlap.parse_corpora([f"x={root}/missing*.jsonl"])
        with self.assertRaises(ValueError):
            overlap.plan([("x", [str(root / "c.md")])], Params())


class CliTest(Case):
    def test_scan_command(self):
        protected = self.write_jsonl(
            "candidates.jsonl",
            [
                candidate("a", {"t": TEXT}),
                candidate("b", {"t": words("zeta", 30)}),
                candidate("c", {"t": "tiny"}),
            ],
        )
        corpus = self.write_jsonl("corpus/x.jsonl", [{"t": TEXT}])
        self.write_jsonl("corpus/.cache/y.jsonl", [{"t": words("zeta", 30)}])
        (self.root / "corpus" / "readme.md").write_text(words("zeta", 30))
        receipt_path, hits_path = (
            self.root / "out/receipt.json",
            self.root / "out/hits.jsonl",
        )
        argv = [
            "scan",
            "--protected",
            str(protected),
            "--corpus",
            f"train={corpus.parent}",
            "--workers",
            "2",
            "--output",
            str(receipt_path),
            "--hits",
            str(hits_path),
        ]
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            self.assertEqual(overlap.main(argv), 0)
        receipt = json.loads(receipt_path.read_text())
        self.assertEqual(receipt["verdicts"], {"OVERLAP": 1, "REVIEW": 0, "CLEAN": 2})
        self.assertEqual(receipt["corpora"]["train"]["files"], 1)
        self.assertEqual(receipt["protected"]["unscreenable"], 1)
        self.assertEqual(
            receipt["by_source"]["fake"]["labels"],
            {"train": {"OVERLAP": 1, "REVIEW": 0}},
        )
        data = hits_path.read_bytes()
        self.assertEqual(receipt["hits_sha256"], hashlib.sha256(data).hexdigest())
        self.assertEqual(len(data.splitlines()), 3)
        for text in (data.decode(), receipt_path.read_text()):
            self.assertNotIn("tok0", text)
            self.assertNotIn("zeta0", text)
        self.assertEqual(os.stat(hits_path).st_mode & 0o777, 0o600)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            overlap.main(argv)

    def test_manifest_labels_and_verification(self):
        protected = self.write_jsonl("candidates.jsonl", [candidate("a", TEXT)])
        train = self.write_jsonl("train.jsonl", [{"t": TEXT}])
        peer = self.write_jsonl("peer.jsonl", [{"t": "unrelated text of some length"}])
        entry = lambda path, sha=None: {  # noqa: E731
            "path": str(path),
            "sha256": sha or hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
        }
        good = self.root / "manifest.json"
        good.write_text(
            json.dumps(
                {
                    "labels": {
                        "train-a": {"files": [entry(train)]},
                        "agg-b": {"files": [entry(peer)]},
                    }
                }
            )
        )
        bad = self.root / "bad.json"
        bad.write_text(
            json.dumps({"labels": {"train-a": {"files": [entry(train, "0" * 64)]}}})
        )

        def run(manifest: Path, name: str, *extra: str) -> dict:
            out = self.root / name
            argv = ["scan", "--protected", str(protected), "--manifest", str(manifest)]
            argv += [*extra, "--output", str(out), "--hits", str(out) + ".hits"]
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                io.StringIO()
            ):
                overlap.main(argv)
            return json.loads(out.read_text())

        receipt = run(good, "all.json")
        self.assertEqual(sorted(receipt["corpora"]), ["agg-b", "train-a"])
        self.assertEqual(receipt["resources"]["manifest_files_verified"], 2)
        self.assertEqual(
            receipt["manifest_sha256"], hashlib.sha256(good.read_bytes()).hexdigest()
        )
        self.assertEqual(
            list(run(good, "train.json", "--labels", "train-*")["corpora"]), ["train-a"]
        )
        with self.assertRaises(ValueError):
            run(bad, "bad-out.json")
        self.assertEqual(
            run(bad, "unverified.json", "--no-verify")["verdicts"]["OVERLAP"], 1
        )


if __name__ == "__main__":
    unittest.main()
