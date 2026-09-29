from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import json
import math
import stat
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from typing import Any, Callable

from v2.eval.sealed import independence, retire
from v2.eval.sealed.schema import Candidate


def words(prefix: str, count: int) -> str:
    return " ".join(f"{prefix}{i}" for i in range(count))


PASSAGE = words("aa", 40)
QUOTE, WIDE, SHORT = words("qq", 30), words("ww", 200), words("s", 7)
CONTAINED = f"{words('aa', 20)} {words('cc', 20)}"
TRANSCRIPT = f"{words('ff', 200)} {QUOTE} {words('gq', 200)}"
ALPHA = [
    {"doc_id": "a-1", "text": PASSAGE},
    {"doc_id": "a-2", "note": "tiny"},
    {"doc_id": "a-3", "q": QUOTE, "w": WIDE, "short": SHORT},
    {"doc_id": "a-4", "text": words("zz", 30)},
    {"id": 12, "text": words("ii", 25)},
]
BETA = [
    ("K9", words("bb", 30), "yes"),
    ("K10", words("bc", 30), "no"),
    ("M4 ", words("bm", 30), "yes"),
]
GAMMA = [
    {"item_id": "g-1", "body": words("gb", 30)},
    {"item_id": "g-99", "body": words("lone", 30)},
]
FLAGS = {
    "alpha/data/train.jsonl|0": "OVERLAP",
    "alpha/data/train.jsonl|2": "REVIEW",
    "alpha/data/train.jsonl|4": "REVIEW",
    "beta/cases.csv|0": "OVERLAP",
    "beta/cases.csv|2": "REVIEW",
    "gamma/items.json|1": "REVIEW",
}
BASELINE = {
    "alpha/data/train.jsonl|0": "REVIEW",
    "beta/cases.csv|1": "REVIEW",
    "beta/cases.csv|2": "REVIEW",
}
CANDIDATES: dict[str, tuple[str, dict[str, str]]] = {
    "alpha/judge|c1": ("gc1", {"text": CONTAINED}),
    "alpha/rank|c1b": ("gc1", {"text": words("cb", 20)}),
    "alpha/judge|t2": ("gt2", {"transcript": TRANSCRIPT}),
    "alpha/judge|a-1:x": ("a-1", {"text": PASSAGE}),
    "alpha/judge|12": ("g12", {"text": words("tw", 20)}),
    "alpha/judge|n1": ("gn1", {"text": words("nn", 30)}),
    "alpha/judge|n2": ("gn2", {"passage": words("xx", 200), "quote": SHORT}),
    "alpha/judge|n3": ("gn3", {"text": words("nt", 20)}),
    "alpha/judge|z1": ("gz1", {"text": words("zz", 30)}),
    "alpha/rank|K9": ("K9", {"text": words("ak", 20)}),
    "alpha/rank|r2": ("gr2", {"text": words("ar", 20)}),
    "beta/function|K9:p1": ("K9", {"text": words("kp", 20)}),
    "beta/function|K9:p2": ("K9", {"text": words("kq", 20)}),
    "beta/verdict|K9": ("K9", {"text": words("kv", 20)}),
    "beta/function|M4:s1": ("gM", {"text": words("ms", 20)}),
    "beta/verdict|gm-2": ("gM", {"text": words("mv", 20)}),
    "beta/function|K10:p1": ("K10", {"text": words("kx", 20)}),
    "beta/function|b-n": ("gbn", {"text": words("bn", 20)}),
    "beta/verdict|b-v": ("gbv", {"text": words("bv", 20)}),
    "gamma/rate|g-1": ("G1", {"text": f"Gamma prologue. {PASSAGE}"}),
    "gamma/rate|g-2": ("G1", {"text": words("gr", 20)}),
    "gamma/rate|g-3": ("K9", {"text": words("gk", 20)}),
    "gamma/rate|g-4": ("gg4", {"text": words("gf", 20)}),
}
TEXT = {"alpha/judge|c1", "alpha/judge|t2", "gamma/rate|g-1"}
BOTH = {"alpha/judge|a-1:x"}
ID = {
    "alpha/judge|12",
    "beta/function|K9:p1",
    "beta/function|K9:p2",
    "beta/verdict|K9",
    "beta/function|M4:s1",
}
GROUP_ONLY = {"alpha/rank|c1b", "beta/verdict|gm-2", "gamma/rate|g-2"}
RETIRED = TEXT | BOTH | ID | GROUP_ONLY
BUILD = {
    "alpha/judge|t2": ("OVERLAP", 0.9, 10),
    "alpha/judge|a-1:x": ("CLEAN", 0.0, 0),
    "beta/function|K9:p2": ("REVIEW", 0.1, 10),
    "beta/function|M4:s1": ("REVIEW", 0.3, 10),
}
NOT_SCANNED = "alpha/judge|12"
TASKS = {
    "alpha/judge": (1, 5),
    "alpha/rank": (2, 3),
    "beta/function": (2, 4),
    "beta/verdict": (1, 3),
    "gamma/extra": (1, 0),
    "gamma/rate": (1, 2),
}
BOUNDS = {
    "alpha/judge": 1,
    "alpha/rank": 1,
    "beta/function": 2,
    "beta/verdict": 2,
    "gamma/extra": 0,
    "gamma/rate": 1,
}


def candidate(key: str, group: str, state: dict[str, str]) -> dict[str, Any]:
    task, item = key.split("|")
    return Candidate(
        source=task.split("/")[0],
        task=task,
        source_item_id=item,
        group_id=group,
        balance_label="True",
        language="en",
        state=state,
        question={
            "type": "noul",
            "instructions": "Decide.",
            "criteria": {"true": "Yes.", "false": "No."},
        },
        gold=True,
        overlap_texts=list(state.values()),
    ).to_json()


def scan_hit(key: str, verdict: str) -> dict[str, Any]:
    task = key.split("|")[0]
    return {
        "id": key,
        "source": task.split("/")[0],
        "task": task,
        "verdict": verdict,
        "containment": {"OVERLAP": 0.6, "REVIEW": 0.3}.get(verdict, 0.0),
        "matched": 3,
        "shingles": 10,
        "label": "corpus",
        "file": "corpus.jsonl",
        "row": 0,
        "exact": None,
        "labels": {},
    }


def build_hit(key: str) -> dict[str, Any]:
    verdict, containment, shingles = BUILD.get(key, ("CLEAN", 0.0, 10))
    return {
        "id": key,
        "verdict": verdict,
        "containment": containment,
        "shingles": shingles,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    data = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode()
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def append(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row) + "\n")


def equal_variance(pairs: list[tuple[int, int]]) -> float:
    return math.sqrt(sum(1 / (n - k) for n, k in pairs) / sum(1 / n for n, _ in pairs))


class World:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.sources = root / "sources"
        self.candidates = root / "candidates"
        alpha = self.sources / "alpha" / "data"
        for directory in (alpha, self.sources / "beta", self.sources / "gamma"):
            directory.mkdir(parents=True)
        self.candidates.mkdir()
        lines = [json.dumps(row) for row in ALPHA]
        lines.insert(2, "not a json row")
        (alpha / "train.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")
        path = self.sources / "beta" / "cases.csv"
        with path.open("w", encoding="utf-8", newline="") as stream:
            csv.writer(stream).writerows([("case_id", "passage", "label"), *BETA])
        (self.sources / "gamma" / "items.json").write_text(json.dumps(GAMMA))
        self.protected = root / "protected.jsonl"
        with contextlib.redirect_stdout(io.StringIO()):
            independence.rows(
                Namespace(
                    sources_dir=self.sources,
                    source=["alpha", "beta", "gamma"],
                    output=self.protected,
                )
            )
        self.protected_ids = [
            f"{row['task']}|{row['source_item_id']}"
            for row in map(json.loads, self.protected.read_text().splitlines())
        ]
        self.hits = self.write_hits("hits.jsonl", FLAGS)
        self.baseline = self.write_hits("baseline.jsonl", BASELINE)
        rows = [candidate(key, *value) for key, value in CANDIDATES.items()]
        digests = {
            source: write_jsonl(
                self.candidates / f"{source}.jsonl",
                [row for row in rows if row["source"] == source],
            )
            for source in ("alpha", "beta", "gamma")
        }
        self.build_hits = root / "build-hits.jsonl"
        hits = [build_hit(key) for key in CANDIDATES if key != NOT_SCANNED]
        manifest = {
            "candidates_sha256": digests,
            "overlap_hits_sha256": write_jsonl(self.build_hits, hits),
            "config": {
                "sources": sorted(digests),
                "tasks": {
                    task: {"cap": 10, "group_cap": cap}
                    for task, (cap, _) in TASKS.items()
                },
            },
            "tasks": {task: {"selected": n} for task, (_, n) in TASKS.items()},
            "items": sum(n for _, n in TASKS.values()),
        }
        self.manifest = root / "manifest.json"
        self.manifest.write_text(json.dumps(manifest, indent=1))

    def write_hits(self, name: str, flags: dict[str, str]) -> Path:
        keys = dict.fromkeys([*self.protected_ids, *flags])
        path = self.root / name
        write_jsonl(path, [scan_hit(key, flags.get(key, "CLEAN")) for key in keys])
        return path

    def write_extract(self, name: str, protected_sha: str) -> Path:
        rows = [json.loads(line) for line in self.hits.read_text().splitlines()]
        value = {
            "schema": "dev2-c1-scan-extract/1",
            "protected_sha256": protected_sha,
            "non_clean": [row for row in rows if row["verdict"] != "CLEAN"],
        }
        path = self.root / name
        path.write_text(json.dumps(value, indent=1))
        return path

    def argv(self, name: str, *extra: str, hits: Path | None = None) -> list[str]:
        options = {
            "--hits": hits or self.hits,
            "--protected": self.protected,
            "--sources-dir": self.sources,
            "--candidates-dir": self.candidates,
            "--build-manifest": self.manifest,
            "--build-hits": self.build_hits,
            "--version": "v1.2",
            "--output": self.root / f"{name}.RETIRED.json",
            "--receipt": self.root / f"{name}.receipt.json",
        }
        return ["build", *(str(x) for pair in options.items() for x in pair), *extra]

    def run(
        self, name: str, *extra: str, hits: Path | None = None
    ) -> tuple[bytes, dict[str, Any], dict[str, Any]]:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = retire.main(self.argv(name, *extra, hits=hits))
        assert code == 0
        data = (self.root / f"{name}.RETIRED.json").read_bytes()
        receipt = json.loads((self.root / f"{name}.receipt.json").read_text())
        return data, receipt, json.loads(out.getvalue())


class RuleTest(unittest.TestCase):
    def shares(
        self, row_texts: list[str], state: dict[str, str]
    ) -> tuple[float, float, bool]:
        row, row_own = retire.features({"overlap_texts": row_texts, "state": {}})
        item, item_own = retire.features({"overlap_texts": [], "state": state})
        shared = len(row_own & item_own)
        linked = bool(shared) and retire.text_linked(row, item, shared)
        return shared / row.size, shared / item.size, linked

    def test_containment_either_way_links(self) -> None:
        row_share, item_share, linked = self.shares([PASSAGE], {"text": CONTAINED})
        self.assertGreaterEqual(min(row_share, item_share), 0.2)
        self.assertLess(max(row_share, item_share), 0.5)
        self.assertTrue(linked)
        wider = f"{words('aa', 31)} {words('xz', 200)}"
        row_share, item_share, linked = self.shares([PASSAGE], {"text": wider})
        self.assertGreaterEqual(row_share, 0.2)
        self.assertLess(item_share, 0.2)
        self.assertTrue(linked)
        narrow = f"{words('qq', 20)} zq0"
        row_share, item_share, linked = self.shares(
            [QUOTE, WIDE, SHORT], {"text": narrow}
        )
        self.assertLess(row_share, 0.2)
        self.assertGreaterEqual(item_share, 0.2)
        self.assertTrue(linked)

    def test_long_text_inside_the_other_side_links(self) -> None:
        row_share, item_share, linked = self.shares(
            [QUOTE, WIDE, SHORT], {"transcript": TRANSCRIPT}
        )
        self.assertLess(max(row_share, item_share), 0.2)
        self.assertTrue(linked)

    def test_shared_text_under_eight_tokens_does_not_link(self) -> None:
        row_share, item_share, linked = self.shares(
            [QUOTE, WIDE, SHORT], CANDIDATES["alpha/judge|n2"][1]
        )
        self.assertGreater(row_share, 0.0)
        self.assertLess(max(row_share, item_share), 0.2)
        self.assertFalse(linked)

    def test_id_values(self) -> None:
        row = {
            "ID": " 7 ",
            "case_id": "K9",
            "Parent_Id": 3,
            "grid": "g",
            "label": "x",
            "none_id": None,
            "blank_id": " ",
            None: ["extra", "fields"],
        }
        self.assertEqual(retire.id_values(row), {"7", "K9", "3"})
        self.assertEqual(retire.id_values(["id", "K9"]), set())


class RetireTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.world = World(self.root / "world")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_text_links_cross_sources_and_skip_unrelated_items(self) -> None:
        retired = set(json.loads(self.world.run("v12")[0])["candidates"])
        self.assertLessEqual(TEXT | BOTH, retired)
        for key in ("alpha/judge|n1", "alpha/judge|n2", "alpha/judge|z1"):
            self.assertNotIn(key, retired)

    def test_id_links_stay_within_the_source(self) -> None:
        retired = set(json.loads(self.world.run("v12")[0])["candidates"])
        self.assertLessEqual(ID, retired)
        for key in ("alpha/rank|K9", "gamma/rate|g-3", "beta/function|K10:p1"):
            self.assertNotIn(key, retired)

    def test_group_closure(self) -> None:
        data, receipt, _ = self.world.run("v12")
        retired = json.loads(data)
        self.assertEqual(set(retired["candidates"]), RETIRED)
        self.assertEqual(retired["protected_rows"], sorted(FLAGS))
        self.assertEqual(
            receipt["linked"]["by_kind"],
            {"text": len(TEXT), "id": len(ID), "both": len(BOTH)},
        )
        self.assertEqual(receipt["linked"]["candidates"], len(RETIRED - GROUP_ONLY))

    def test_receipt_counts_without_ids_or_text(self) -> None:
        _, receipt, summary = self.world.run(
            "v12", "--baseline", str(self.world.baseline)
        )
        text = (self.world.root / "v12.receipt.json").read_text()
        self.assertEqual(
            receipt["flagged"],
            {
                "rows": 6,
                "by_verdict": {"OVERLAP": 2, "REVIEW": 4},
                "by_source": {"alpha": 3, "beta": 2, "gamma": 1},
                "recurring": 2,
                "new": 4,
                "without_link": 1,
            },
        )
        self.assertEqual(
            receipt["linked"]["by_source"], {"alpha": 4, "beta": 4, "gamma": 1}
        )
        retired = receipt["retired"]
        self.assertEqual(
            retired["groups_by_source"], {"alpha": 4, "beta": 2, "gamma": 1}
        )
        self.assertEqual(
            retired["candidates_by_task"],
            {
                "alpha/judge": 4,
                "alpha/rank": 1,
                "beta/function": 3,
                "beta/verdict": 2,
                "gamma/rate": 2,
            },
        )
        self.assertEqual((retired["candidates"], retired["protected_rows"]), (12, 6))
        self.assertIsNone(retired["previous"])
        inputs = receipt["inputs_sha256"]
        self.assertEqual(len(inputs["baselines"]), 1)
        self.assertEqual(sorted(inputs["candidates"]), ["alpha", "beta", "gamma"])
        self.assertEqual(
            {k: len(v) for k, v in inputs["snapshot_files"].items()},
            {"alpha": 1, "beta": 1, "gamma": 1},
        )
        for key in [*CANDIDATES, *self.world.protected_ids]:
            self.assertNotIn(key, text)
        for passage in (PASSAGE, QUOTE, WIDE, TRANSCRIPT):
            self.assertNotIn(passage, text)
        for name in ("v12.RETIRED.json", "v12.receipt.json"):
            mode = (self.world.root / name).stat().st_mode
            self.assertEqual(stat.S_IMODE(mode), 0o600)
        self.assertEqual(summary["retired_candidates"], 12)
        self.assertEqual(
            summary["receipt_sha256"], hashlib.sha256(text.encode()).hexdigest()
        )

    def test_bounds_and_power(self) -> None:
        power = self.world.run("v12")[1]["power"]
        for task, bound in BOUNDS.items():
            items = TASKS[task][1]
            factor = math.sqrt(items / (items - bound)) if bound < items else None
            cell = power["tasks"][task]
            self.assertEqual(
                (cell["items"], cell["retired_upper_bound"]), (items, bound)
            )
            if factor is None:
                self.assertIsNone(cell["se_inflation_upper_bound"])
            else:
                self.assertAlmostEqual(
                    cell["se_inflation_upper_bound"], factor, delta=1e-4
                )
        self.assertEqual(
            power["totals"],
            {
                "items_v_prev": 17,
                "build_manifest_items": 17,
                "retired_items_upper_bound": 7,
                "items_lower_bound": 10,
            },
        )
        self.assertAlmostEqual(
            power["c1_se_inflation_upper_bound"], math.sqrt(3), delta=1e-4
        )
        self.assertAlmostEqual(
            power["c1_se_inflation_equal_variance"],
            equal_variance([(5, 1), (3, 1), (4, 2), (3, 2), (2, 1)]),
            delta=1e-4,
        )

    def test_previous_is_carried_into_output_and_bounds(self) -> None:
        previous = self.world.root / "v11.RETIRED.json"
        previous.write_text(
            json.dumps(
                {
                    "schema": "dev2-c1-retired/1",
                    "version": "v1.1",
                    "candidates": ["gamma/rate|g-3", "gamma/rate|g-4", "legacy/t|x"],
                    "protected_rows": ["legacy/rows.jsonl|3"],
                }
            )
        )
        data, receipt, _ = self.world.run("v12", "--previous", str(previous))
        retired = json.loads(data)
        carried = {"gamma/rate|g-3", "gamma/rate|g-4", "legacy/t|x"}
        self.assertEqual(set(retired["candidates"]), RETIRED | carried)
        self.assertEqual(
            set(retired["protected_rows"]), {*FLAGS, "legacy/rows.jsonl|3"}
        )
        self.assertEqual(
            receipt["retired"]["previous"],
            {"candidates": 3, "protected_rows": 1, "candidates_not_in_build": 1},
        )
        self.assertEqual(receipt["retired"]["candidates_by_task"]["gamma/rate"], 4)
        power = receipt["power"]
        self.assertEqual(
            power["tasks"]["gamma/rate"],
            {"items": 2, "retired_upper_bound": 2, "se_inflation_upper_bound": None},
        )
        self.assertIsNone(power["c1_se_inflation_upper_bound"])
        self.assertAlmostEqual(
            power["c1_se_inflation_equal_variance"],
            equal_variance([(5, 1), (3, 1), (4, 2), (3, 2)]),
            delta=1e-4,
        )

    def test_extract_input_and_canonical_bytes(self) -> None:
        first, _, summary = self.world.run("a")
        again, _, _ = self.world.run("b")
        protected_sha = hashlib.sha256(self.world.protected.read_bytes()).hexdigest()
        extract = self.world.write_extract("extract.json", protected_sha)
        extracted, receipt, _ = self.world.run("c", hits=extract)
        self.assertEqual(first, again)
        self.assertEqual(first, extracted)
        canonical = json.dumps(json.loads(first), sort_keys=True, indent=1) + "\n"
        self.assertEqual(first, canonical.encode())
        self.assertEqual(summary["retired_sha256"], hashlib.sha256(first).hexdigest())
        self.assertEqual(receipt["flagged"]["rows"], len(FLAGS))

    def test_refusals_write_nothing(self) -> None:
        def changed_candidates(world: World) -> None:
            append(
                world.candidates / "beta.jsonl", candidate("beta/verdict|new", "g", {})
            )

        def changed_build_hits(world: World) -> None:
            append(world.build_hits, build_hit("beta/verdict|new"))

        def index_out_of_range(world: World) -> None:
            row = {
                "source": "alpha",
                "task": "alpha/data/train.jsonl",
                "source_item_id": "7",
                "state": {},
                "overlap_texts": [words("oo", 10)],
            }
            append(world.protected, row)
            flags = {**FLAGS, "alpha/data/train.jsonl|7": "REVIEW"}
            world.hits = world.write_hits("more.jsonl", flags)

        def unprotected(world: World) -> None:
            flags = {**FLAGS, "alpha/data/train.jsonl|1": "REVIEW"}
            world.hits = world.write_hits("more.jsonl", flags)

        def changed_snapshot(world: World) -> None:
            path = world.sources / "alpha" / "data" / "train.jsonl"
            path.write_text(path.read_text().replace(PASSAGE, words("aa", 41)))

        def other_protected_rows(world: World) -> None:
            world.hits = world.write_extract("extract.json", "0" * 64)

        cases: list[tuple[Callable[[World], None], str]] = [
            (changed_candidates, "SHA-256 differs from the build manifest"),
            (changed_build_hits, "SHA-256 differs from the build manifest"),
            (index_out_of_range, "row index out of range"),
            (unprotected, "1 flagged ids are not in the protected rows"),
            (changed_snapshot, "a raw row differs from its protected row"),
            (other_protected_rows, "an extract of other protected rows"),
        ]
        for mutate, message in cases:
            with self.subTest(mutate.__name__):
                world = World(self.root / mutate.__name__)
                mutate(world)
                with self.assertRaises(SystemExit) as caught:
                    world.run("v12")
                self.assertIn(message, str(caught.exception.code))
                self.assertFalse((world.root / "v12.RETIRED.json").exists())
                self.assertFalse((world.root / "v12.receipt.json").exists())

    def test_refuses_to_overwrite(self) -> None:
        output = self.world.root / "v12.RETIRED.json"
        output.write_text("{}\n")
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                self.world.run("v12")
        self.assertEqual(caught.exception.code, 2)
        self.assertEqual(output.read_text(), "{}\n")
        self.assertFalse((self.world.root / "v12.receipt.json").exists())


if __name__ == "__main__":
    unittest.main()
