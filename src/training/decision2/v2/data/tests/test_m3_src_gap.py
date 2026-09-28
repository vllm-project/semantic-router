"""Tests for v2.data.m3.src_gap on small synthetic fixtures (no tokenizer:
``TokenCounter(None)`` counts ``len(text) // 4``)."""

from __future__ import annotations

import collections
import json
import random
import tempfile
import unittest
import zipfile
from pathlib import Path

from training.model.data import validate_row
from v2.data.m2 import common
from v2.data.m3 import src_gap as g
from v2.data.textnorm import normalize

COUNT = g.TokenCounter(None)
FILLER = "The committee met again and recorded routine business about roads and parks."


def paragraph(index: int, words: int = 60) -> str:
    return " ".join([f"Section {index} notes."] + [FILLER] * (words // 12))


def window_record(**overrides):
    paragraphs = [paragraph(i) for i in range(30)]
    paragraphs[7] = paragraph(7) + " The bridge was designed by Miriam Okafor in 1911."
    record = {
        "local_id": "ex-1",
        "group_key": "who designed the bridge",
        "language": "en",
        "title": "Harbour Bridge",
        "question": "who designed the bridge",
        "paragraphs": paragraphs,
        "gold": 7,
        "answers": ["Miriam Okafor"],
    }
    record.update(overrides)
    return record


def claim_record(hops: int = 3, pool: int = 60):
    gold = [
        (f"Person {i}", f"Person {i} was born in Town {i}. " + FILLER * 3)
        for i in range(hops)
    ]
    distractors = [
        (f"Other {i}", f"Other {i} is a village. " + FILLER * 4) for i in range(pool)
    ]
    return {
        "local_id": "claim-1",
        "group_key": "which town",
        "claim": "Person 0 and Person 1 were born in neighbouring towns.",
        "verdict": "SUPPORTED",
        "hops": hops,
        "gold": gold,
        "distractors": distractors,
    }


def paragraphs_of(state: str) -> list[str]:
    return [line.split(": ", 1)[0].split(" — ", 1)[1] for line in state.split("\n\n")]


class WindowTest(unittest.TestCase):
    def build(self, record=None, **kwargs):
        return g.window_twins(
            record or window_record(),
            source="src",
            family="fam",
            namespace="ns",
            template="t",
            seed="seed",
            count=COUNT,
            **kwargs,
        )

    def test_twins_share_paragraphs_but_one(self) -> None:
        rows, reason = self.build()
        self.assertIsNone(reason)
        complete, removed = rows
        self.assertEqual([complete["label"], removed["label"]], [1, 0])
        self.assertEqual(complete["group_id"], removed["group_id"])
        self.assertEqual(complete["instructions"], removed["instructions"])
        first = complete["state"].split("\n\n")
        second = removed["state"].split("\n\n")
        self.assertEqual(len(first), len(second))
        self.assertIn("Miriam Okafor", complete["state"])
        self.assertNotIn("Miriam Okafor", removed["state"])
        self.assertEqual(len(set(first) ^ set(second)), 2)
        for row in rows:
            validate_row(row, "train")
            self.assertGreaterEqual(COUNT(row["state"]), g.MIN_WINDOW_TOKENS - 100)

    def test_deterministic(self) -> None:
        self.assertEqual(self.build()[0], self.build()[0])

    def test_guards(self) -> None:
        cases = {
            "answer_in_lead_paragraph": window_record(gold=0),
            "answer_in_title": window_record(title="Miriam Okafor Bridge"),
            "answer_names_page_subject": window_record(title="Okafor"),
            "answer_too_long": window_record(
                answers=["one two three four five six seven eight nine"]
            ),
            "unusable_answer": window_record(answers=["7"]),
        }
        for reason, record in cases.items():
            with self.subTest(reason=reason):
                self.assertEqual(self.build(record), ([], reason))
        repeated = window_record()
        repeated["paragraphs"][9] += " Miriam Okafor also built a school."
        rows, reason = self.build(repeated)
        self.assertIn(reason, (None, "answer_left_after_removal"))
        if reason is None:
            self.assertNotIn("Miriam Okafor", rows[1]["state"])

    def test_short_page_dropped(self) -> None:
        record = window_record(paragraphs=[paragraph(i, 12) for i in range(5)], gold=2)
        record["paragraphs"][2] += " Miriam Okafor designed it."
        self.assertEqual(self.build(record), ([], "window_under_min_tokens"))

    def test_grow_window_bounds(self) -> None:
        sizes = [100] * 50
        for seed in range(20):
            indices, total = g.grow_window(
                sizes.__getitem__, len(sizes), 10, 2000, random.Random(seed)
            )
            self.assertIn(10, indices)
            self.assertEqual(indices, list(range(indices[0], indices[-1] + 1)))
            self.assertEqual(total, 100 * len(indices))
            self.assertGreaterEqual(total, 2000)
        self.assertIsNone(
            g.grow_window([g.STATE_CAP + 1].__getitem__, 1, 0, 10, random.Random(0))
        )


class ClaimTest(unittest.TestCase):
    def test_twins_one_edit_and_budget(self) -> None:
        rows, reason = g.claim_twins(
            claim_record(),
            source="s",
            family="f",
            namespace="multihop",
            template="t",
            seed="seed",
            count=COUNT,
        )
        self.assertIsNone(reason)
        complete, removed = (paragraphs_of(row["state"]) for row in rows)
        self.assertEqual(len(complete), len(removed))
        gold = {f"Person {i}" for i in range(3)}
        self.assertEqual(gold & set(complete), gold)
        self.assertEqual(len(gold & set(removed)), 2)
        self.assertEqual(len(set(complete) ^ set(removed)), 2)
        for row in rows:
            self.assertLessEqual(COUNT(row["state"]), g.STATE_CAP + 200)
            validate_row(row, "train")

    def test_coverage_group(self) -> None:
        rows, reason = g.claim_coverage(
            claim_record(hops=4),
            source="s",
            family="f",
            namespace="multihop",
            template="t",
            seed="seed",
            count=COUNT,
        )
        self.assertIsNone(reason)
        self.assertEqual(sorted(row["label"] for row in rows), [0, 1, 2, 3, 4])
        self.assertEqual(len({row["instructions"] for row in rows}), 1)
        self.assertEqual(len({json.dumps(row["options"]) for row in rows}), 1)
        sizes = set()
        for row in rows:
            names = paragraphs_of(row["state"])
            sizes.add(len(names))
            self.assertEqual(
                sum(name.startswith("Person") for name in names), row["label"]
            )
        self.assertEqual(len(sizes), 1)

    def test_too_few_distractors(self) -> None:
        record = claim_record(pool=1)
        for build in (g.claim_twins, g.claim_coverage):
            rows, reason = build(
                record,
                source="s",
                family="f",
                namespace="multihop",
                template="t",
                seed="seed",
                count=COUNT,
            )
            self.assertEqual((rows, reason), ([], "too_few_distractors"))


class PassageTest(unittest.TestCase):
    RECORD = {
        "local_id": "es:1",
        "group_key": "q",
        "language": "es",
        "question": "¿Dónde nació?",
        "positives": [("A", "Nació en Lima.")],
        "negatives": [(f"N{i}", f"Texto {i}.") for i in range(6)],
    }

    def test_pool_twins(self) -> None:
        rows, reason = g.pool_twins(
            self.RECORD, source="s", family="f", namespace="ns", template="t", seed="x"
        )
        self.assertIsNone(reason)
        complete, removed = rows
        self.assertEqual(
            complete["state"].count("\n\n"), removed["state"].count("\n\n")
        )
        self.assertIn("Lima", complete["state"])
        self.assertNotIn("Lima", removed["state"])

    def test_relevance_twins(self) -> None:
        rows, reason = g.relevance_twins(
            self.RECORD, source="s", family="f", namespace="ns", template="t", seed="x"
        )
        self.assertIsNone(reason)
        self.assertEqual([row["label"] for row in rows], [1, 0])
        self.assertEqual(rows[0]["language"], "es")


class SourceParsingTest(unittest.TestCase):
    def test_sentimix_blocks(self) -> None:
        text = (
            "meta\t1\tpositive\nSo\tlang1\nmeta\tlang2\nbien\tlang2\n\n"
            "meta\t2\tneutral\nok\tlang1\n\nmeta\t3\tangry\nno\tlang1\n"
        )
        items, skipped = g.sentimix_items(text)
        self.assertEqual(
            items, [("1", "So meta bien", "positive"), ("2", "ok", "neutral")]
        )
        self.assertEqual(skipped["empty_or_unknown_label"], 1)

    def test_sentimix_family_balanced(self) -> None:
        blocks = []
        for index in range(30):
            label = ("negative", "neutral", "positive")[index % 3 if index < 21 else 2]
            blocks.append(f"meta\t{index}\t{label}\nword{index}\tlang1\n")
        with tempfile.TemporaryDirectory() as tmp:
            with zipfile.ZipFile(
                Path(tmp, "Semeval_2020_task9_data.zip"), "w"
            ) as archive:
                archive.writestr(g.SENTIMIX_MEMBER, "\n".join(blocks))
            rows, report = g.build_sentimix(
                {"sentimix": Path(tmp)}, family="sm", seed="s"
            )
        counts = collections.Counter(row["label"] for row in rows)
        self.assertLessEqual(max(counts.values()), min(counts.values()) * 6 // 5)
        self.assertEqual({row["language"] for row in rows}, {"es-en"})
        self.assertEqual(report["rows"], len(rows))

    def test_html_text(self) -> None:
        fragment = '<P>The <b>Beatles</b> were a band<sup id="c"><a>[1]</a></sup> &amp; more.</P>'
        self.assertEqual(g.html_text(fragment), "The Beatles were a band & more.")

    def test_nq_example(self) -> None:
        page = "<H1>T</H1><P>Lead text here.</P><Table><Tr>x</Tr></Table><P>The answer is Zed Q.</P>"
        data = page.encode()
        spans = [(data.index(b"<P>"), data.index(b"</P>") + 4)]
        second = data.index(b"<P>", spans[0][1])
        spans.append((second, len(data)))
        spans.insert(1, (data.index(b"<Table>"), data.index(b"</Table>") + 8))
        answer = data.index(b"Zed Q")
        row = {
            "id": 5,
            "title": "T",
            "html": page,
            "text": "what is the answer",
            "long_answer_candidates": {
                "start_byte": [s for s, _ in spans],
                "end_byte": [e for _, e in spans],
                "top_level": [True, True, True],
            },
            "annotations": {
                "id": ["a"],
                "long_answer": [{"candidate_index": 2}],
                "yes_no_answer": [-1],
                "short_answers": [{"start_byte": [answer], "end_byte": [answer + 5]}],
            },
        }
        example, reason = g.nq_example(row)
        self.assertIsNone(reason)
        self.assertEqual(
            example["paragraphs"], ["Lead text here.", "The answer is Zed Q."]
        )
        self.assertEqual((example["gold"], example["answers"]), (1, ["Zed Q"]))
        row["annotations"]["long_answer"] = [{"candidate_index": 1}]
        self.assertEqual(g.nq_example(row), (None, "long_answer_not_paragraph"))
        row["annotations"]["yes_no_answer"] = [1]
        self.assertEqual(g.nq_example(row), (None, "yes_no_answer"))


class BuildTest(unittest.TestCase):
    def test_isolation_then_cap(self) -> None:
        def rows_for(dirs):
            rows = []
            for group in range(10):
                for twin in range(2):
                    rows.append(
                        common.make_row(
                            source="src",
                            family="fam",
                            task_type="noul",
                            language="en",
                            namespace="ns",
                            group_key=f"g{group}",
                            local_id=f"{group}:{twin}",
                            state=f"state {group} {twin}",
                            instructions="q?",
                            options=common.noul_options("en"),
                            label=twin,
                            template="t",
                        )
                    )
            return rows, {"candidates": 10}

        family = g.GapFamily("h8", "fam", "src", rows_for, 6, "seed")
        taken = common.group_id("ns", "g3")
        rows, report = g.build("h8", {}, existing=({taken}, set()), families=[family])
        self.assertEqual(len(rows), 6)
        self.assertNotIn(taken, {row["group_id"] for row in rows})
        self.assertEqual(report["families"]["fam"]["isolation"]["groups_dropped"], 1)
        self.assertEqual(
            collections.Counter(row["group_id"] for row in rows).most_common(1)[0][1], 2
        )

    def test_isolate_by_input_hash(self) -> None:
        row = common.make_row(
            source="s",
            family="f",
            task_type="noul",
            language="en",
            namespace="ns",
            group_key="k",
            local_id="1",
            state="x",
            instructions="q",
            options=common.noul_options("en"),
            label=1,
            template="t",
        )
        kept, report = g.isolate([row], set(), {row["input_sha256"]})
        self.assertEqual((kept, report["groups_dropped"]), ([], 1))

    def test_registry(self) -> None:
        names = [item.family for item in g.FAMILIES]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual({item.arm for item in g.FAMILIES}, set(g.ARMS))
        self.assertNotIn("en", g.MIRACL_LANGUAGES)

    def test_length_baseline_uninformative(self) -> None:
        rows = [
            {
                "state": "x" * (50 + group),
                "label": label,
                "group_id": f"g{group}",
                "task_type": "noul",
            }
            for group in range(200)
            for label in (0, 1)
        ]
        result = g.length_baseline(rows)["noul"]
        self.assertLessEqual(result["accuracy"], 0.55)

    def test_budget(self) -> None:
        rows = [
            common.make_row(
                source="s",
                family="f",
                task_type="noul",
                language="en",
                namespace="ns",
                group_key=f"k{i // 2}",
                local_id=str(i),
                state=f"s{i}",
                instructions="q",
                options=common.noul_options("en"),
                label=i % 2,
                template="t",
            )
            for i in range(4)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            source, tokens = Path(tmp, "a.jsonl"), Path(tmp, "t.jsonl")
            source.write_text("".join(json.dumps(r) + "\n" for r in rows))
            tokens.write_text(
                "".join(
                    json.dumps(
                        {"id": r["id"], "native": 9000 if i == 0 else 2500, "kai": 900}
                    )
                    + "\n"
                    for i, r in enumerate(rows)
                )
            )
            report = g.apply_budget(
                source, tokens, Path(tmp, "b.jsonl"), Path(tmp, "r.json")
            )
        self.assertEqual((report["groups_over_limit"], report["rows_kept"]), (1, 2))
        self.assertEqual(report["long_evidence_share"], 1.0)


def _row(**overrides):
    fields = dict(
        source="s",
        family="f",
        task_type="score",
        language="ko",
        namespace="ns",
        group_key="k",
        local_id="1",
        state="x",
        instructions="How similar are the two sentences?",
        options=common.score_options(["low", "mid", "high"]),
        label=1,
        template="t",
    )
    fields.update(overrides)
    return common.make_row(**fields)


class ProtectedFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        protected = self.tmp / "p.jsonl"
        protected.write_text(json.dumps({"id": "p1", "state": "held"}) + "\n")
        self.base = self.tmp / "manifest.json"
        self.base.write_text(
            json.dumps([{"role": "old", "path": str(protected), "sha256": "0" * 64}])
        )
        self.extra = self.tmp / "extra.jsonl"
        self.extra.write_text(json.dumps(_row()) + "\n")

    def tearDown(self) -> None:
        import shutil

        shutil.rmtree(self.tmp)


class ProtectedTest(ProtectedFixture):
    def test_verified_manifest_and_flags(self) -> None:
        digest = g.file_sha256(self.extra)
        report = g.project_protected(
            self.base,
            [("new", self.extra)],
            self.tmp / "pi",
            base_sha256=g.file_sha256(self.base),
            expected={"new": digest},
            report_only=["old"],
        )
        self.assertTrue(report["base_verified"])
        self.assertEqual(report["added"][0]["origin_sha256"], digest)
        self.assertTrue(report["added"][0]["origin_verified"])
        self.assertEqual(report["report_only_roles"], ["old"])
        manifest = json.loads((self.tmp / "pi/manifest.json").read_text())
        self.assertEqual([entry["role"] for entry in manifest], ["old", "new"])
        projected = json.loads((self.tmp / "pi/new.jsonl").read_text())
        self.assertNotIn("label", projected)

    def test_hash_mismatch_writes_nothing(self) -> None:
        cases = [
            {"expected": {"new": "1" * 64}},
            {"base_sha256": "2" * 64},
            {"expected": {"other": "3" * 64}},
            {"report_only": ["missing"]},
        ]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    g.project_protected(
                        self.base, [("new", self.extra)], self.tmp / "pi", **kwargs
                    )
                self.assertFalse((self.tmp / "pi").exists())


class QuarantiningManifestTest(ProtectedFixture):
    def test_drops_report_only_roles(self) -> None:
        report = g.project_protected(
            self.base, [("new", self.extra)], self.tmp / "pi", report_only=["new"]
        )
        out = self.tmp / "pi/manifest.quarantining.json"
        subset = g.quarantining_manifest(
            self.tmp / "pi/manifest.json", self.tmp / "pi/receipt.json", out
        )
        entries = json.loads(out.read_text())
        self.assertEqual([entry["role"] for entry in entries], ["old"])
        self.assertEqual(subset["from_manifest_sha256"], report["manifest_sha256"])
        self.assertEqual(subset["dropped_report_only_roles"], ["new"])
        receipt = json.loads(
            (self.tmp / "pi/manifest.quarantining.receipt.json").read_text()
        )
        self.assertEqual(receipt["manifest_sha256"], g.file_sha256(out))
        with self.assertRaises(ValueError):
            g.quarantining_manifest(
                self.base, self.tmp / "pi/receipt.json", self.tmp / "other.json"
            )


class PairCheckTest(unittest.TestCase):
    def test_pairs_found_by_hash_pair_and_sentence(self) -> None:
        a7k = [
            _row(local_id=str(i), state=f"Sentence 1: {first}\nSentence 2: {second}")
            for i, (first, second) in enumerate(
                [("Rain  today.", "It rains."), ("A cat.", "A dog."), ("Sun.", "Moon.")]
            )
        ]
        a6h2 = [
            _row(
                local_id="a",
                state={"sentence_1": "it rains.", "sentence_2": "Rain today."},
            ),
            _row(local_id="b", state={"sentence_1": "A cat.", "sentence_2": "A bird."}),
            _row(local_id="c", state={"passage": "Sun."}),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            pairs, against = Path(tmp, "a7k.jsonl"), Path(tmp, "h6.jsonl")
            pairs.write_text("".join(json.dumps(r) + "\n" for r in a7k))
            against.write_text("".join(json.dumps(r) + "\n" for r in a6h2 + [a7k[2]]))
            report = g.pair_check([pairs], [against])
        found = report["pairs"][0]
        self.assertEqual(
            {key: found[key] for key in ("rows", "not_a_pair", "input_sha256", "pair")},
            {"rows": 3, "not_a_pair": 0, "input_sha256": 1, "pair": 2},
        )
        self.assertEqual(found["sentence_shared"], 3)
        self.assertEqual(report["against_pairs"], 3)
        self.assertIsNone(g.pair_texts({"state": "Sentence 1: only one line"}))


class StatsTest(unittest.TestCase):
    def test_slice_stats(self) -> None:
        rows = [
            _row(local_id="1", group_key="a", label=0),
            _row(local_id="2", group_key="a", label=2),
            _row(
                local_id="3",
                group_key="b",
                task_type="noul",
                options=common.noul_options("en"),
                label=1,
            ),
        ]
        tokens = {
            rows[0]["id"]: {"native": 2500, "kai": 1100},
            rows[1]["id"]: {"native": 1500, "kai": 900},
            rows[2]["id"]: {"native": 100, "kai": 90},
        }
        stats = g.slice_stats(rows, tokens)
        self.assertEqual((stats["rows"], stats["groups"]), (3, 2))
        self.assertEqual(
            (stats["native_tokens"], stats["long_evidence_tokens"]), (4100, 2500)
        )
        self.assertEqual((stats["rows_ge_2000"], stats["kai_over_1024"]), (1, 1))
        self.assertEqual(stats["score_levels"], {"L3": {"0": 1, "1": 0, "2": 1}})
        self.assertEqual(stats["noul_labels"], {"1": 1})


class NormalizationTest(unittest.TestCase):
    def test_group_keys_are_normalized_questions(self) -> None:
        rows, _ = g.window_twins(
            window_record(group_key=normalize("Who designed the BRIDGE")),
            source="s",
            family="f",
            namespace="nq",
            template="t",
            seed="s",
            count=COUNT,
        )
        self.assertEqual(
            rows[0]["group_id"], common.group_id("nq", "who designed the bridge")
        )


if __name__ == "__main__":
    unittest.main()
