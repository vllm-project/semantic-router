"""Tests for v2.data.m2.src_multihop on small synthetic fixtures that mirror the
pinned HotpotQA, 2WikiMultihopQA and MuSiQue-Full layouts."""

from __future__ import annotations

import collections
import dataclasses
import hashlib
import itertools
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

from training.model.data import canonical, validate_row
from v2.data.m2 import build, common, constructions, spec
from v2.data.m2 import src_multihop as sm
from v2.data.sources import musique
from v2.data.textnorm import normalize

PACKAGE_ROOT = Path(__file__).resolve().parents[3]
WORDS = ("amber", "birch", "cedar", "dune", "elm", "fern", "glen", "heath")
KIND = {
    sm.HOTPOTQA_TWINS: "twins",
    sm.HOTPOTQA_COVERAGE: "coverage",
    sm.HOTPOTQA_ABSTENTION: "abstention",
    sm.TWOWIKI_TWINS: "twins",
    sm.TWOWIKI_COVERAGE: "coverage",
    sm.TWOWIKI_ABSTENTION: "abstention",
    sm.MUSIQUE_TWINS: "native_twins",
    sm.MUSIQUE_COVERAGE: "coverage",
}
PADDING = {sm.HOTPOTQA_COVERAGE: 2, sm.TWOWIKI_COVERAGE: 2, sm.MUSIQUE_COVERAGE: 3}
DIGEST_SCRIPT = """
import hashlib, sys
from pathlib import Path
from training.model.data import canonical
from v2.data.m2 import build, spec, src_multihop
dirs = dict(spec.resolve(Path(sys.argv[1])), **{build.V1_ROWS_KEY: Path(sys.argv[2])})
digest = hashlib.sha256()
for item in src_multihop.FAMILIES:
    rows, report = item.build(dirs)
    digest.update(canonical({"family": item.family, "rows": rows, "report": report}).encode())
print(digest.hexdigest())
"""

Sentences = list[str]
Paragraphs = list[tuple[str, Sentences]]


@dataclasses.dataclass(frozen=True)
class Item:
    source: str
    local_id: str
    question: str
    answer: str
    gold: tuple[tuple[str, str], ...]
    family: str
    reason: str | None = None


def pick(source: str, prefix: str, low: bool, share: int, count: int = 1) -> list[str]:
    found = []
    for number in itertools.count():
        local_id = f"{prefix}{number:03d}"
        if (sm.alloc(source, local_id) < share) == low:
            found.append(local_id)
            if len(found) == count:
                return found
    raise AssertionError("unreachable")


def joined(sentences: Sentences) -> str:
    return " ".join("".join(sentences).split())


def fillers(tag: str, count: int) -> Paragraphs:
    return [
        (
            f"Filler {tag} {k}",
            [
                f"Filler {tag} {k} is a quiet {WORDS[k]} village.",
                " It lies near a river.",
            ],
        )
        for k in range(count)
    ]


def bridge(tag: str, answer: str, leak: bool = False) -> tuple[str, Paragraphs]:
    film, person = f"Film {tag}", f"Person {tag}"
    first = [f"{film} is a drama film.", f"  It  was directed by {person}."]
    if leak:
        first.append(f" It was shot in {answer}.")
    second = [f"{person} is a director.", f" {person} was born in {answer}."]
    return f"Where was the director of {film} born?", [(film, first), (person, second)]


def comparison(tag: str) -> Paragraphs:
    return [
        (f"Alpha {tag}", [f"Alpha {tag} is a river.", " It is long."]),
        (f"Beta {tag}", [f"Beta {tag} is a river.", " It is short."]),
    ]


def bridge_comparison(tag: str) -> Paragraphs:
    return [
        (f"Film {tag} A", [f"Film {tag} A is a film.", f" It is by Director {tag} A."]),
        (f"Film {tag} B", [f"Film {tag} B is a film.", f" It is by Director {tag} B."]),
        (f"Director {tag} A", [f"Director {tag} A is a Norwegian director."]),
        (f"Director {tag} B", [f"Director {tag} B is a Chilean director."]),
    ]


def hotpot_row(
    local_id: str,
    kind: str,
    question: str,
    answer: str,
    gold: Paragraphs,
    supporting: list[str] | None = None,
) -> dict[str, Any]:
    distractors = fillers(local_id, 8)
    context = [distractors[0], gold[1], *distractors[1:5], gold[0], *distractors[5:]]
    titles = supporting or [gold[0][0], gold[1][0], gold[1][0]]
    return {
        "id": local_id,
        "type": kind,
        "level": "medium",
        "question": question,
        "answer": answer,
        "supporting_facts": {"title": titles, "sent_id": list(range(len(titles)))},
        "context": {
            "title": [title for title, _ in context],
            "sentences": [sentences for _, sentences in context],
        },
    }


def wiki_row(
    local_id: str,
    kind: str,
    question: str,
    answer: str,
    gold: Paragraphs,
    supporting: list[str] | None = None,
    duplicate: bool = False,
) -> dict[str, Any]:
    distractors = fillers(local_id, 10 - len(gold) - int(duplicate))
    if duplicate:
        distractors.append(distractors[0])
    context = [*distractors[:2], *reversed(gold), *distractors[2:]]
    titles = supporting or [title for title, _ in gold]
    return {
        "_id": local_id,
        "type": kind,
        "question": question,
        "answer": answer,
        "context": json.dumps(
            [[title, [s.strip() for s in sentences]] for title, sentences in context]
        ),
        "supporting_facts": json.dumps([[title, 0] for title in titles]),
        "evidences": json.dumps([]),
    }


def musique_twins(
    key: str,
    question: str,
    hops: int,
    supporting: int | None = None,
    other_question: str | None = None,
    gold_copy: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    support = [3 + 4 * k for k in range(hops if supporting is None else supporting)]
    paragraphs = [
        {
            "idx": k,
            "title": f"Page {key} {k % 7}",
            "paragraph_text": f"Paragraph {k} of {key}   describes a town.",
            "is_supporting": k in support,
        }
        for k in range(20)
    ]
    if gold_copy:
        paragraphs[19] = dict(paragraphs[support[0]], idx=19, is_supporting=False)
    steps = [
        {
            "id": k,
            "question": f"hop {k}",
            "answer": f"bridge {k}",
            "paragraph_support_idx": support[k] if k < len(support) else None,
        }
        for k in range(hops)
    ]
    answerable = {
        "id": key,
        "paragraphs": paragraphs,
        "question": question,
        "question_decomposition": steps,
        "answer": f"Final {key}",
        "answer_aliases": [],
        "answerable": True,
    }
    unanswerable = dict(
        answerable,
        answerable=False,
        question=other_question or question,
        paragraphs=[
            (
                dict(p, is_supporting=False, paragraph_text=f"Replacement {p['idx']}.")
                if p["is_supporting"]
                else p
            )
            for p in paragraphs
        ],
        question_decomposition=[dict(s, paragraph_support_idx=None) for s in steps],
    )
    return answerable, unanswerable


def gold_of(paragraphs: Paragraphs) -> tuple[tuple[str, str], ...]:
    return tuple((title, joined(sentences)) for title, sentences in paragraphs)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def write_hotpotqa(root: Path, items: list[Item]) -> str:
    rows = []
    low = pick(sm.HOTPOTQA, "hp-low-", True, sm.TWIN_SHARE, 4)
    high = pick(sm.HOTPOTQA, "hp-high-", False, sm.TWIN_SHARE, 3)
    for local_id in low + high:
        leak, missing = local_id == low[-1], local_id == high[-1]
        question, gold = bridge(local_id, "Zorvath", leak)
        supporting = [gold[0][0], "Ghost page"] if missing else None
        rows.append(
            hotpot_row(local_id, "bridge", question, "Zorvath", gold, supporting)
        )
        items.append(
            Item(
                sm.HOTPOTQA,
                local_id,
                question,
                "Zorvath",
                gold_of(gold),
                sm.HOTPOTQA_TWINS if local_id in low else sm.HOTPOTQA_COVERAGE,
                (
                    "answer_in_incomplete_state"
                    if leak
                    else "missing_supporting_title" if missing else None
                ),
            )
        )
    cases = [
        ("hp-yn-0", "yes", sm.HOTPOTQA_TWINS, None),
        ("hp-yn-1", " No ", sm.HOTPOTQA_TWINS, None),
        ("hp-cmp-0", "Alpha hp-cmp-0", sm.HOTPOTQA_ABSTENTION, None),
        ("hp-cmp-1", "beta HP-CMP-1", sm.HOTPOTQA_ABSTENTION, None),
        ("hp-cmp-2", "Alpha hp-cmp-2", sm.HOTPOTQA_ABSTENTION, None),
        ("hp-cmp-3", "Paris", sm.HOTPOTQA_ABSTENTION, "answer_not_an_entity"),
        ("hp-other-0", "Alpha hp-other-0", sm.UNALLOCATED, None),
    ]
    for local_id, answer, family, reason in cases:
        question = f"Which river is longer, Alpha {local_id} or Beta {local_id}?"
        gold = comparison(local_id)
        kind = "other" if family == sm.UNALLOCATED else "comparison"
        rows.append(hotpot_row(local_id, kind, question, answer, gold))
        items.append(
            Item(sm.HOTPOTQA, local_id, question, answer, gold_of(gold), family, reason)
        )
    base = root / spec.SOURCE_DIRS["hotpotqa"] / "distractor"
    write_jsonl(base / "train-00000-of-00002.jsonl", rows[:7])
    write_jsonl(base / "train-00001-of-00002.jsonl", [*rows[7:], rows[0]])
    question, gold = bridge("DEVSPLIT", "Zorvath")
    decoy = hotpot_row("DEVSPLIT", "bridge", question, "Zorvath", gold)
    write_jsonl(base / "validation-00000-of-00001.jsonl", [decoy])
    return rows[0]["question"]


def write_twowiki(root: Path, items: list[Item], shared_question: str) -> None:
    rows = []
    low = pick(sm.TWOWIKI, "tw-low-", True, sm.TWIN_SHARE, 3)
    high = pick(sm.TWOWIKI, "tw-high-", False, sm.TWIN_SHARE, 3)
    for local_id in low + high:
        kind = "inference" if local_id in (low[2], high[2]) else "compositional"
        question, gold = bridge(local_id, "Quendra")
        if local_id == high[1]:
            question = "  " + shared_question.upper().replace(" ", "  ")
        rows.append(wiki_row(local_id, kind, question, "Quendra", gold))
        family = sm.TWOWIKI_TWINS if local_id in low else sm.TWOWIKI_COVERAGE
        items.append(
            Item(sm.TWOWIKI, local_id, question, "Quendra", gold_of(gold), family)
        )
    for n in range(3):
        local_id = f"tw-bc-{n}"
        gold = bridge_comparison(local_id)
        question = f"Are the directors of Film {local_id} A and B from one country?"
        rows.append(
            wiki_row(
                local_id, "bridge_comparison", question, "no", gold, duplicate=n == 2
            )
        )
        reason = "too_few_distractors" if n == 2 else None
        items.append(
            Item(
                sm.TWOWIKI,
                local_id,
                question,
                "no",
                gold_of(gold),
                sm.TWOWIKI_COVERAGE,
                reason,
            )
        )
    cases = [
        ("tw-cmp-0", "Beta tw-cmp-0", None, None),
        ("tw-cmp-1", "Alpha tw-cmp-1", None, None),
        ("tw-cmp-yn", "no", None, "yes_no_answer"),
        ("tw-cmp-miss", "Alpha tw-cmp-miss", ["Ghost"], "missing_supporting_title"),
    ]
    for local_id, answer, ghost, reason in cases:
        question = f"Which river is longer, Alpha {local_id} or Beta {local_id}?"
        gold = comparison(local_id)
        supporting = [gold[0][0], *ghost] if ghost else None
        rows.append(
            wiki_row(local_id, "comparison", question, answer, gold, supporting)
        )
        items.append(
            Item(
                sm.TWOWIKI,
                local_id,
                question,
                answer,
                gold_of(gold),
                sm.TWOWIKI_ABSTENTION,
                reason,
            )
        )
    write_jsonl(root / spec.SOURCE_DIRS["twowiki"] / sm.TWOWIKI_FILE, rows)


def write_musique(
    root: Path, items: list[Item]
) -> tuple[dict[str, tuple[dict[str, Any], dict[str, Any]]], Path]:
    def one(prefix: str, low: bool) -> str:
        return pick(sm.MUSIQUE, prefix, low, sm.MUSIQUE_TWIN_SHARE)[0]

    shared = "Which town hosts the yearly fair?"
    cases = [
        (one("2hop__t", True), 2, {}, sm.MUSIQUE_TWINS, None),
        (one("3hop1__t", True), 3, {}, sm.MUSIQUE_TWINS, None),
        (one("4hop1__t", True), 4, {}, sm.MUSIQUE_TWINS, None),
        (one("2hop__inc", True), 2, {}, sm.MUSIQUE_TWINS, "incomplete_twin_pair"),
        (
            one("2hop__mis", True),
            2,
            {"other_question": "Which river is it?"},
            sm.MUSIQUE_TWINS,
            "twin_question_mismatch",
        ),
        (one("2hop__c", False), 2, {"shared": True}, sm.MUSIQUE_COVERAGE, None),
        (one("3hop2__c", False), 3, {"gold_copy": True}, sm.MUSIQUE_COVERAGE, None),
        (one("4hop3__c", False), 4, {}, sm.MUSIQUE_COVERAGE, None),
        (
            one("3hop1__bad", False),
            3,
            {"supporting": 2},
            sm.MUSIQUE_COVERAGE,
            "supporting_count_not_hops",
        ),
        (one("2hop__v", False), 2, {"shared": True}, sm.V1_EXCLUDED, None),
    ]
    pairs = {}
    for key, hops, options, family, reason in cases:
        question = shared if options.pop("shared", False) else f"Question about {key}?"
        answerable, unanswerable = musique_twins(key, question, hops, **options)
        pairs[key] = (answerable, unanswerable)
        gold = tuple(
            (p["title"], " ".join(p["paragraph_text"].split()))
            for p in answerable["paragraphs"]
            if p["is_supporting"]
        )
        items.append(
            Item(sm.MUSIQUE, key, question, answerable["answer"], gold, family, reason)
        )
    incomplete = cases[3][0]
    rows = [pair[0] for pair in pairs.values()]
    rows += [pair[1] for key, pair in pairs.items() if key != incomplete]
    write_jsonl(root / spec.SOURCE_DIRS["musique"] / musique.DATA, rows)
    v1_key = cases[-1][0]
    v1_rows = root / "v1" / "a3.train.jsonl"
    write_jsonl(
        v1_rows,
        [
            {
                "source": sm.MUSIQUE,
                "audit_metadata": {"source_local_id": f"{v1_key}:{t}"},
            }
            for t in ("answerable", "unanswerable")
        ]
        + [
            {
                "source": "musique_ans_v1.0_train",
                "audit_metadata": {"source_local_id": f"{cases[0][0]}:answerable"},
            }
        ],
    )
    manifest = root / "v1-rows.json"
    manifest.write_text(json.dumps([str(v1_rows)]), encoding="utf-8")
    return pairs, manifest


def write_fixture(root: Path) -> tuple[list[Item], dict[str, Any], Path]:
    items: list[Item] = []
    shared = write_hotpotqa(root, items)
    write_twowiki(root, items, shared)
    pairs, manifest = write_musique(root, items)
    return items, pairs, manifest


def build_all(dirs: dict[str, Path]) -> dict[str, tuple[list[dict], dict]]:
    return {item.family: item.build(dirs) for item in sm.FAMILIES}


def digest_all(built: dict[str, tuple[list[dict], dict]]) -> str:
    digest = hashlib.sha256()
    for item in sm.FAMILIES:
        rows, report = built[item.family]
        payload = {"family": item.family, "rows": rows, "report": report}
        digest.update(canonical(payload).encode())
    return digest.hexdigest()


def state_paragraphs(state: str) -> list[str]:
    return [part.split(" — ", 1)[1] for part in state.split("\n\n")]


def present(state: str, gold: tuple[tuple[str, str], ...]) -> list[bool]:
    paragraphs = set(state_paragraphs(state))
    return [f"{title}: {text}" in paragraphs for title, text in gold]


class MultihopFamiliesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.items, cls.pairs, cls.manifest = write_fixture(cls.root)
        cls.by_id = {(item.source, item.local_id): item for item in cls.items}
        cls.dirs = {**spec.resolve(cls.root), build.V1_ROWS_KEY: cls.manifest}
        cls.built = build_all(cls.dirs)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.tmp.cleanup()

    def item_of(self, row: dict[str, Any]) -> Item:
        local_id = row["audit_metadata"]["source_local_id"].split(":")[0]
        return self.by_id[(row["source"], local_id)]

    def grouped(self, family: str) -> dict[str, list[dict[str, Any]]]:
        groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
        for row in self.built[family][0]:
            groups[self.item_of(row).local_id].append(row)
        return groups

    def test_registry(self) -> None:
        expected = {
            sm.HOTPOTQA_TWINS: ("h3", sm.HOTPOTQA, 16000, "h3-hotpotqa-v1"),
            sm.HOTPOTQA_COVERAGE: ("h6", sm.HOTPOTQA, 9000, "h6-hotpotqa-coverage-v1"),
            sm.HOTPOTQA_ABSTENTION: (
                "e11",
                sm.HOTPOTQA,
                4000,
                "e11-hotpotqa-abstention-v1",
            ),
            sm.TWOWIKI_TWINS: ("h3", sm.TWOWIKI, 16000, "h3-twowiki-v1"),
            sm.TWOWIKI_COVERAGE: ("h6", sm.TWOWIKI, 9000, "h6-twowiki-coverage-v1"),
            sm.TWOWIKI_ABSTENTION: (
                "e11",
                sm.TWOWIKI,
                4000,
                "e11-twowiki-abstention-v1",
            ),
            sm.MUSIQUE_TWINS: ("h3", "musique_full_v1.0_train", 16000, "h3-musique-v2"),
            sm.MUSIQUE_COVERAGE: (
                "h6",
                "musique_full_v1.0_train",
                6000,
                "h6-musique-coverage-v1",
            ),
        }
        self.assertEqual(
            {f.family: (f.arm, f.source, f.cap_rows, f.seed) for f in sm.FAMILIES},
            expected,
        )
        self.assertEqual(
            [f.family for f in build.families(("src_multihop",))],
            [f.family for f in sm.FAMILIES],
        )

    def test_rows_are_valid_and_tagged(self) -> None:
        for item in sm.FAMILIES:
            rows, report = self.built[item.family]
            with self.subTest(item.family):
                self.assertTrue(rows)
                self.assertEqual(report["rows"], len(rows))
                local_ids = [row["audit_metadata"]["source_local_id"] for row in rows]
                self.assertEqual(len(set(local_ids)), len(local_ids))
                for row in rows:
                    validate_row(row, "train")
                    self.assertEqual(row["family"], item.family)
                    self.assertEqual(row["source"], item.source)
                    self.assertEqual(row["language"], "en")
                    self.assertEqual(
                        row["render_template"], sm.TEMPLATES[KIND[item.family]]
                    )

    def test_kept_items_and_drops_match_the_fixture(self) -> None:
        for item in sm.FAMILIES:
            rows, report = self.built[item.family]
            mine = [i for i in self.items if i.family == item.family]
            with self.subTest(item.family):
                self.assertEqual(report["candidates"], len(mine))
                self.assertEqual(
                    report["drops"],
                    dict(
                        sorted(
                            collections.Counter(
                                i.reason for i in mine if i.reason
                            ).items()
                        )
                    ),
                )
                kept = {i.local_id for i in mine if i.reason is None}
                self.assertEqual({self.item_of(row).local_id for row in rows}, kept)
                self.assertEqual(report["kept_items"], len(kept))
                self.assertEqual(
                    report["candidates"],
                    report["kept_items"] + sum(report["drops"].values()),
                )
                self.assertEqual(
                    sum(report["qtype_histogram"].values()), len(rows), report
                )

    def test_allocation_is_a_disjoint_partition(self) -> None:
        for source in (sm.HOTPOTQA, sm.TWOWIKI, sm.MUSIQUE):
            families = [f.family for f in sm.FAMILIES if f.source == source]
            expected = collections.Counter(
                i.family for i in self.items if i.source == source
            )
            if source == sm.HOTPOTQA:
                expected[sm.DUPLICATE] += 1
            reports = [self.built[family][1] for family in families]
            with self.subTest(source):
                for report in reports:
                    self.assertEqual(
                        report["allocation"], dict(sorted(expected.items()))
                    )
                seen: dict[str, str] = {}
                for family in families:
                    for local_id in self.grouped(family):
                        self.assertNotIn(local_id, seen)
                        seen[local_id] = family
        cases = [
            (sm.hotpotqa_family, {"type": "comparison", "id": "x", "answer": " Yes"}),
            (
                sm.hotpotqa_family,
                {"type": "comparison", "id": "x", "answer": "Yesterday"},
            ),
            (sm.hotpotqa_family, {"type": "other", "id": "x", "answer": "no"}),
            (sm.twowiki_family, {"type": "bridge_comparison", "_id": "x"}),
            (sm.twowiki_family, {"type": "comparison", "_id": "x"}),
            (sm.twowiki_family, {"type": "unknown", "_id": "x"}),
        ]
        self.assertEqual(
            [allocate(item) for allocate, item in cases],
            [
                sm.HOTPOTQA_TWINS,
                sm.HOTPOTQA_ABSTENTION,
                sm.UNALLOCATED,
                sm.TWOWIKI_COVERAGE,
                sm.TWOWIKI_ABSTENTION,
                sm.UNALLOCATED,
            ],
        )

    def test_group_is_the_normalized_question_in_one_namespace(self) -> None:
        for item in sm.FAMILIES:
            for row in self.built[item.family][0]:
                question = self.item_of(row).question
                self.assertEqual(
                    row["group_id"], common.group_id("multihop", normalize(question))
                )
        hotpot = {row["group_id"] for row in self.built[sm.HOTPOTQA_TWINS][0]}
        wiki = {row["group_id"] for row in self.built[sm.TWOWIKI_COVERAGE][0]}
        self.assertEqual(len(hotpot & wiki), 1)

    def test_answerability_twins(self) -> None:
        for family in (sm.HOTPOTQA_TWINS, sm.TWOWIKI_TWINS):
            rows, report = self.built[family]
            self.assertEqual(
                report["label_histogram"], {"0": len(rows) // 2, "1": len(rows) // 2}
            )
            for local_id, pair in self.grouped(family).items():
                item = self.by_id[(pair[0]["source"], local_id)]
                with self.subTest(family=family, item=local_id):
                    by_label = {row["label"]: row for row in pair}
                    self.assertEqual(sorted(by_label), [0, 1])
                    complete, removed = by_label[1]["state"], by_label[0]["state"]
                    self.assertEqual(
                        present(complete, item.gold), [True] * len(item.gold)
                    )
                    self.assertEqual(
                        sum(present(removed, item.gold)), len(item.gold) - 1
                    )
                    self.assertEqual(
                        len(state_paragraphs(complete)), len(state_paragraphs(removed))
                    )
                    self.assertEqual(
                        len(state_paragraphs(complete)), len(item.gold) + 4
                    )
                    if not sm.yes_no(item.answer):
                        self.assertNotIn(item.answer, removed)
                    self.assertEqual(
                        by_label[1]["instructions"],
                        constructions.ANSWERABLE_INSTRUCTIONS.format(
                            " ".join(item.question.split())
                        ),
                    )

    def test_musique_native_twins_follow_v1(self) -> None:
        rows, report = self.built[sm.MUSIQUE_TWINS]
        self.assertEqual(report["label_histogram"], {"0": 3, "1": 3})
        self.assertEqual(report["qtype_histogram"], {"2hop": 2, "3hop1": 2, "4hop1": 2})
        for row in rows:
            key, twin = row["audit_metadata"]["source_local_id"].split(":")
            answerable, unanswerable = self.pairs[key]
            source = answerable if twin == "answerable" else unanswerable
            self.assertEqual(row["label"], int(twin == "answerable"))
            self.assertEqual(row["state"], musique.render(source["paragraphs"]))
            self.assertEqual(
                row["instructions"],
                constructions.ANSWERABLE_INSTRUCTIONS.format(answerable["question"]),
            )
            self.assertEqual(row["options"], common.noul_options("en"))
            self.assertEqual(row["render_template"], "m2/musique_answerable/v1")
            self.assertEqual(row["audit_metadata"]["musique_id"], key)

    def test_coverage_grades_are_exactly_balanced(self) -> None:
        for family in (sm.HOTPOTQA_COVERAGE, sm.TWOWIKI_COVERAGE, sm.MUSIQUE_COVERAGE):
            rows, report = self.built[family]
            levels: dict[int, collections.Counter[int]] = collections.defaultdict(
                collections.Counter
            )
            for local_id, group in self.grouped(family).items():
                item = self.by_id[(group[0]["source"], local_id)]
                needed = len(item.gold)
                with self.subTest(family=family, item=local_id):
                    self.assertEqual(
                        sorted(row["label"] for row in group), list(range(needed + 1))
                    )
                    self.assertEqual(len({row["instructions"] for row in group}), 1)
                    self.assertEqual(
                        len({canonical(row["options"]) for row in group}), 1
                    )
                    self.assertEqual(len({row["group_id"] for row in group}), 1)
                    for row in group:
                        self.assertEqual(len(row["options"]), needed + 1)
                        self.assertEqual(
                            sum(present(row["state"], item.gold)), row["label"]
                        )
                        self.assertEqual(
                            len(state_paragraphs(row["state"])),
                            needed + PADDING[family],
                        )
                        levels[needed + 1][row["label"]] += 1
            for level, grades in levels.items():
                self.assertEqual(len(set(grades.values())), 1, (family, level, grades))
            self.assertEqual(
                report["grades_by_level"],
                {
                    str(level): {str(g): n for g, n in sorted(grades.items())}
                    for level, grades in sorted(levels.items())
                },
            )
        self.assertEqual(
            set(self.built[sm.TWOWIKI_COVERAGE][1]["grades_by_level"]), {"3", "5"}
        )
        self.assertEqual(
            set(self.built[sm.MUSIQUE_COVERAGE][1]["grades_by_level"]), {"3", "4", "5"}
        )

    def test_musique_coverage_drops_exact_copies_of_gold(self) -> None:
        for row in self.built[sm.MUSIQUE_COVERAGE][0]:
            item = self.item_of(row)
            paragraphs = state_paragraphs(row["state"])
            self.assertEqual(len(paragraphs), len(set(paragraphs)))
            if row["label"] == 0:
                self.assertFalse(any(present(row["state"], item.gold)))

    def test_abstention_is_half_abstain(self) -> None:
        for family in (sm.HOTPOTQA_ABSTENTION, sm.TWOWIKI_ABSTENTION):
            rows, report = self.built[family]
            half = len(rows) // 2
            self.assertEqual(report["gold_kind"], {"abstain": half, "entity": half})
            for local_id, pair in self.grouped(family).items():
                item = self.by_id[(pair[0]["source"], local_id)]
                names = [title for title, _ in item.gold]
                by_twin = {row["audit_metadata"]["twin"]: row for row in pair}
                complete, removed = by_twin["complete"], by_twin["removed"]
                with self.subTest(family=family, item=local_id):
                    self.assertEqual(
                        complete["options"][complete["label"]][
                            "description"
                        ].casefold(),
                        item.answer.strip().casefold(),
                    )
                    self.assertEqual(
                        removed["options"][removed["label"]]["description"],
                        common.ABSTAIN,
                    )
                    self.assertEqual(
                        present(complete["state"], item.gold), [True, True]
                    )
                    self.assertEqual(sum(present(removed["state"], item.gold)), 1)
                    self.assertEqual(
                        sorted(o["description"] for o in complete["options"]),
                        sorted([names[0], names[1], common.ABSTAIN]),
                    )
                    self.assertEqual(
                        [o["key"] for o in complete["options"]], ["o1", "o2", "o3"]
                    )

    def test_v1_ids_are_excluded_and_reported(self) -> None:
        for family in (sm.MUSIQUE_TWINS, sm.MUSIQUE_COVERAGE):
            report = self.built[family][1]
            self.assertEqual(report["allocation"][sm.V1_EXCLUDED], 1)
            self.assertEqual(report["v1_ids"], 1)
        self.assertEqual(
            self.built[sm.MUSIQUE_COVERAGE][1]["v1_shared_question_groups"], 1
        )
        self.assertEqual(
            self.built[sm.MUSIQUE_TWINS][1]["v1_shared_question_groups"], 0
        )
        dirs = {k: v for k, v in self.dirs.items() if k != build.V1_ROWS_KEY}
        spec_ = {f.family: f for f in sm.FAMILIES}[sm.MUSIQUE_COVERAGE]
        rows, report = spec_.build(dirs)
        self.assertNotIn(sm.V1_EXCLUDED, report["allocation"])
        self.assertEqual(report["v1_ids"], 0)
        self.assertEqual(report["candidates"], 5)
        self.assertEqual(len(rows), len(self.built[sm.MUSIQUE_COVERAGE][0]) + 3)

    def test_only_train_files_are_read(self) -> None:
        for family in (sm.HOTPOTQA_TWINS, sm.HOTPOTQA_COVERAGE, sm.HOTPOTQA_ABSTENTION):
            rows, report = self.built[family]
            self.assertEqual(
                sorted(report["inputs"]),
                [
                    "distractor/train-00000-of-00002.jsonl",
                    "distractor/train-00001-of-00002.jsonl",
                ],
            )
            self.assertNotIn("DEVSPLIT", json.dumps(rows))
        self.assertEqual(
            list(self.built[sm.TWOWIKI_TWINS][1]["inputs"]), ["train.jsonl"]
        )
        self.assertEqual(
            list(self.built[sm.MUSIQUE_TWINS][1]["inputs"]), [musique.DATA]
        )

    def test_orchestrator_builds_each_arm(self) -> None:
        for arm in ("h3", "h6", "e11"):
            rows, report = build.build(
                arm, self.dirs, workers=1, modules=("src_multihop",)
            )
            names = {f.family for f in sm.FAMILIES if f.arm == arm}
            self.assertEqual(set(report["families"]), names)
            self.assertEqual(len(rows), sum(len(self.built[n][0]) for n in names))
            for row in rows:
                validate_row(row, "train")

    def test_builds_are_deterministic(self) -> None:
        expected = digest_all(self.built)
        self.assertEqual(digest_all(build_all(self.dirs)), expected)
        for seed in ("0", "4242"):
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    DIGEST_SCRIPT,
                    str(self.root),
                    str(self.manifest),
                ],
                cwd=PACKAGE_ROOT,
                env={**os.environ, "PYTHONHASHSEED": seed},
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertEqual(result.stdout.strip(), expected)


class RecordTest(unittest.TestCase):
    def test_hotpotqa_joins_sentences_and_orders_gold_by_supporting_facts(self) -> None:
        item = {
            "id": "h1",
            "type": "bridge",
            "question": "Q?",
            "answer": "A",
            "supporting_facts": {"title": ["B", "A", "B"], "sent_id": [0, 1, 1]},
            "context": {
                "title": ["D1", "A", "B", "D2", "A"],
                "sentences": [
                    ["First.", " Second  one."],
                    ["a1.", "   a2."],
                    ["b1."],
                    [],
                    ["a copy."],
                ],
            },
        }
        record, reason = sm.hotpotqa_record(item)
        self.assertIsNone(reason)
        self.assertEqual(record["gold"], [("B", "b1."), ("A", "a1. a2.")])
        self.assertEqual(record["distractors"], [("D1", "First. Second one.")])
        self.assertEqual(record["group_key"], "q?")
        missing = dict(item, supporting_facts={"title": ["B", "Z"], "sent_id": [0, 0]})
        self.assertEqual(
            sm.hotpotqa_record(missing), (None, "missing_supporting_title")
        )

    def test_twowiki_parses_json_strings_and_dedupes(self) -> None:
        context = [
            ["D1", ["d one.", "d two."]],
            ["A", ["a1.", "a2."]],
            ["D1", ["d one.", "d two."]],
            ["B", ["b1."]],
            ["A", ["a1.", "a2."]],
            ["D2", ["  spaced   out ."]],
        ]
        item = {
            "_id": "w1",
            "type": "compositional",
            "question": "Q?",
            "answer": "A",
            "context": json.dumps(context),
            "supporting_facts": json.dumps([["B", 0], ["A", 1], ["A", 0]]),
        }
        record, reason = sm.twowiki_record(item)
        self.assertIsNone(reason)
        self.assertEqual(record["gold"], [("B", "b1."), ("A", "a1. a2.")])
        self.assertEqual(
            record["distractors"], [("D1", "d one. d two."), ("D2", "spaced out .")]
        )
        decoded = dict(item, context=context, supporting_facts=[["B", 0], ["A", 1]])
        self.assertEqual(sm.twowiki_record(decoded), (record, None))

    def test_musique_coverage_record_keeps_same_title_distractors(self) -> None:
        answerable, unanswerable = musique_twins("2hop__x", "Q?", 2, gold_copy=True)
        record, reason = sm.musique_coverage_record(
            ("2hop__x", {True: answerable, False: unanswerable})
        )
        self.assertIsNone(reason)
        self.assertEqual(
            [title for title, _ in record["gold"]], ["Page 2hop__x 3", "Page 2hop__x 0"]
        )
        titles = [title for title, _ in record["distractors"]]
        self.assertIn("Page 2hop__x 3", titles)
        self.assertEqual(len(record["distractors"]), 17)
        self.assertTrue(set(record["gold"]).isdisjoint(record["distractors"]))


if __name__ == "__main__":
    unittest.main()
