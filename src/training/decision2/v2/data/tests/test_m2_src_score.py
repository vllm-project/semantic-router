from __future__ import annotations

import collections
import csv
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import unittest
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from training.model.data import INPUT_FIELDS, canonical, validate_row
from v2.data.m2 import build, common, spec, src_score
from v2.data.sources import argq, jglue, klue, ordinal
from v2.data.textnorm import normalize

PACKAGE_ROOT = Path(__file__).resolve().parents[3]
MARKER = "MARKER"
MLQEPE_HEADER = (
    "index\toriginal\ttranslation\tscores\tmean\tz_scores\tz_mean\tmodel_scores"
)
EN_DE_MEMBER = "en-de-train/train.ende.df.short.tsv"  # codespell:ignore ende
EN_DE_ROWS = 1800
OTHER_PAIR_ROWS = 360
STS_ROWS = 4000
ARGQ_TOPICS = (
    "We should adopt a four-day week",
    "We should ban plastic bags",
    "We should fund public libraries",
)
ARGQ_PER_TOPIC = 1500
FULL_DESIGN = {
    "mlqepe_ende": src_score.MLQEPE_SCALE,
    "a6h2_klue_sts": src_score.STS_SCALE,
    "a6h2_jsts": src_score.STS_SCALE,
    "a6h2_argq": src_score.ARGQ_SCALE,
}
PRE_BINNING_DROPS = tuple(
    dict.fromkeys((*src_score.MLQEPE_DROPS, *src_score.A6H2_DROPS))
)
FIXTURE: dict[str, Any] = {}
VALUE_FIELD = {
    "mlqepe": "mean",
    "a6h2_klue_sts": "mean_rating",
    "a6h2_jsts": "mean_rating",
    "a6h2_argq": "wa",
}
DIGEST_SCRIPT = """
import hashlib, sys
from pathlib import Path
from training.model.data import canonical
from v2.data.m2 import spec, src_score
dirs = spec.resolve(Path(sys.argv[1]))
out = [item.build(dirs) for item in src_score.FAMILIES]
print(hashlib.sha256(canonical(out).encode("utf-8")).hexdigest())
"""


def write_tar(path: Path, members: Mapping[str, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "w:gz") as archive:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))


def mlqepe_mean(pair: int, index: int) -> float:
    return ((index * 7919 + pair * 131) % 1800) / 18


def mlqepe_line(
    index: str, original: str, translation: str, mean: str, scores: str
) -> str:
    return "\t".join(
        [index, original, translation, scores, mean, f'["{MARKER}"]', MARKER, MARKER]
    )


def mlqepe_texts(pair: str, index: int) -> tuple[str, str]:
    return (
        f"Source sentence {index} of the {pair} set mentions item {index % 97}.",
        f"Translated sentence {index} of the {pair} set with detail {index % 89}.",
    )


def mlqepe_tsv(pair: str, position: int, rows: int) -> str:
    lines = [MLQEPE_HEADER]
    for index in range(rows):
        mean = mlqepe_mean(position, index)
        lines.append(
            mlqepe_line(
                str(index),
                *mlqepe_texts(pair, index),
                repr(mean),
                json.dumps([mean] * 3),
            )
        )
    if pair == "en-de":
        n = rows
        lines += [
            "",
            mlqepe_line(
                str(n),
                '"Quoted opening," the report said.',
                '"Zitierte Eröffnung", sagte der Bericht.',
                "55.5",
                "[55, 56]",
            ),
            "\t".join([str(n + 1), "only", "seven", "[1]", "1.0", "[]", "0.0"]),
            mlqepe_line(
                str(n + 2), "A broken mean.", "Ein Satz.", "not-a-number", "[1]"
            ),
            mlqepe_line(
                str(n + 3), "A mean above.", "Noch ein Satz.", "100.5", "[100]"
            ),
            mlqepe_line(str(n + 4), "No translation.", "   ", "40.0", "[40]"),
            mlqepe_line("7", "Reused index seven.", "Wieder sieben.", "40.0", "[40]"),
            mlqepe_line(str(n + 5), *mlqepe_texts(pair, 10), "12.25", "[12.25]"),
            mlqepe_line(
                str(n + 6),
                "  " + mlqepe_texts(pair, 11)[0].upper() + "  ",
                "A different translation of sentence eleven.",
                "61.0",
                "[61]",
            ),
            mlqepe_line(str(n + 7), "Mean differs.", "Anders.", "30.0", "[10, 20]"),
        ]
    return "\n".join(lines) + "\n"


def make_mlqepe(root: Path) -> None:
    for position, pair in enumerate(src_score.MLQEPE_PAIRS):
        compact = pair.replace("-", "")
        rows = EN_DE_ROWS if pair == "en-de" else OTHER_PAIR_ROWS
        broken = f"{MARKER}\t\t\n".encode()
        write_tar(
            root / src_score.MLQEPE_DIR / f"{pair}-train.tar.gz",
            {
                f"{pair}-train/train.{compact}.df.short.tsv": mlqepe_tsv(
                    pair, position, rows
                ).encode("utf-8"),
                f"{pair}-train/train.doc_ids": b"doc_id\nMARKER article\n",
                f"{pair}-train/word-probas/mt.train.{compact}": broken,
                f"{pair}-train/word-probas/word_probas.train.{compact}": broken,
            },
        )
        for other in ("data/direct-assessments/dev", "data/post-editing/train"):
            (root / other).mkdir(parents=True, exist_ok=True)
            (root / other / f"{pair}-train.tar.gz").write_bytes(b"never opened")


ONESTOP_FILES: dict[str, bytes] = {
    "Ele-Txt/Library-News-ele.txt": "\ufeffThe town’s new library opened on Monday – "
    "many people came.  \nIt has “lots” of books for children. \n".encode(),
    "Int-Txt/Library-News-int.txt": b"Intermediate\nThe towns new library opened on "
    b"Monday  many people came to see it.\nIt has lots of books for children and "
    b"adults.\n\n\n",
    "Adv-Txt/Library-News-adv.txt": "\ufeffThe town’s new library was inaugurated on "
    "Monday – drawing large crowds.\n\nIts collection, worth £2m, is “vast”.\n".encode(),
    "Ele-Txt/River-Clean-Up-ele.txt": "\ufeffVolunteers cleaned the river on "
    "Sunday.\n".encode(),
    "Int-Txt/River-Clean-Up-int.txt": b"Intermediate\r\nVolunteers spent Sunday "
    b"cleaning the river banks.\r\n",
    "Adv-Txt/River-Clean-Up-adv.txt": b"Volunteers devoted Sunday to clearing debris "
    b"from the river\xff banks.\n",
    "Ele-Txt/Solar-Farm-ele.txt": b"A solar farm is being built.\n",
    "Adv-Txt/Solar-Farm-adv.txt": b"Construction of a solar farm has begun.\n",
    "Ele-Txt/Old-Bridge-ele.txt": b"The old bridge will close.\n",
    "Int-Txt/Old-Bridge-int.txt": b"Intermediate\nThe old  bridge will close.\n",
    "Adv-Txt/Old-Bridge-adv.txt": b"The old bridge is to be closed for repairs.\n",
    "Ele-Txt/Night-Bus-ele.txt": b"A new night bus starts soon.\n",
    "Int-Txt/Night-Bus-int.txt": b"Intermediate\nA new night bus service starts soon.\n",
    "Adv-Txt/Night-Bus-adv.txt": b" \n\n",
    "Ele-Txt/Tea-Shop-ele.txt": b"Advanced\nThe tea shop sells cake.\n",
    "Int-Txt/Tea-Shop-int.txt": b"Intermediate\nThe tea shop sells cakes and tea.\n",
    "Adv-Txt/Tea-Shop-adv.txt": b"The tea shop offers an assortment of pastries.\n",
    "Ele-Txt/Kite Day-ele.txt": b"Kites fly.\n",
    "Ele-Txt/Kite  Day-ele.txt": b"Kites fly high.\n",
    "Int-Txt/Kite Day-int.txt": b"Kites fly high in the wind.\n",
    "Adv-Txt/Kite Day-adv.txt": b"Kites soar on the strong wind.\n",
    "Ele-Txt/notes.txt": b"Not an article.\n",
    "Ele-Txt/.DS_Store": b"\x00\x01",
    "Ele-Txt/._Library-News-ele.txt": b"\x00\x05\x16\x07\xff",
    "Int-Txt/.DS_Store": b"\x00\x01",
    "Adv-Txt/.DS_Store": b"\x00\x01",
    "Int-Txt/README.md": b"not a text\n",
}
ONESTOP_STATES = {
    "Library-News": (
        "The towns new library opened on Monday many people came.\n"
        "It has lots of books for children.",
        "The towns new library opened on Monday many people came to see it.\n"
        "It has lots of books for children and adults.",
        "The towns new library was inaugurated on Monday drawing large crowds.\n"
        "Its collection, worth 2m, is vast.",
    ),
    "River-Clean-Up": (
        "Volunteers cleaned the river on Sunday.",
        "Volunteers spent Sunday cleaning the river banks.",
        "Volunteers devoted Sunday to clearing debris from the river banks.",
    ),
}


def make_onestop(root: Path) -> None:
    for name, data in ONESTOP_FILES.items():
        path = root / src_score.ONESTOP_DIR / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def sts_value(index: int, offset: int = 0) -> float:
    return ((index * 7919 + offset) % STS_ROWS) / 800


def klue_record(index: int, value: float, sentences: int) -> dict[str, Any]:
    return {
        "guid": f"klue-sts-v1_train_{index:05d}",
        "source": f"{MARKER}-{'rtt' if index % 2 else 'sampled'}",
        "sentence1": f"문장 {sentences}: 숙소는 역에서 {sentences % 9 + 3}분 거리에 있습니다.",
        "sentence2": f"문장 {sentences}의 짝: 역에서 숙소까지 도보 {sentences % 7 + 2}분입니다.",
        "labels": {
            "label": round(value, 1),
            "real-label": value,
            "binary-label": int(value >= 3),
        },
    }


def make_klue(root: Path) -> None:
    records = [klue_record(i, sts_value(i), i) for i in range(STS_ROWS)]
    records.append(klue_record(STS_ROWS, 2.5, 3))
    write_jsonl(root / klue.STS_FILE, records)
    (root / "sts/validation.jsonl").write_text("never opened\n", encoding="utf-8")


def jsts_image(index: int) -> str:
    base = str(6000 + index // 3)
    return base if index % 10 else f"{base}_{9000 + index}"


def make_jglue(root: Path) -> None:
    write_jsonl(
        root / jglue.JSTS_FILE,
        [
            {
                "sentence_pair_id": str(i),
                "yjcaptions_id": f"{jsts_image(i)}-{i}-{i + 5000}",
                "sentence1": f"{i}番目の写真では人が公園を歩いています。",
                "sentence2": f"{i}番目の写真の別の説明です。",
                "label": sts_value(i, 3),
            }
            for i in range(STS_ROWS)
        ],
    )
    (root / "datasets/jsts-v1.3/valid-v1.3.json").write_text(
        "never opened\n", encoding="utf-8"
    )


def argq_values(topic: int, index: int) -> tuple[float, float]:
    wa = ((index * 7919 + topic * 17) % ARGQ_PER_TOPIC) / ARGQ_PER_TOPIC
    return wa, (round((wa + 0.5) % 1, 9) if index % 10 == 0 else wa)


def argq_text(topic: int, index: int) -> str:
    return (
        f"Argument {index} about {ARGQ_TOPICS[topic].lower()}: it would change "
        f"daily habits for {index % 13} groups."
    )


def make_argq(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with (root / argq.FILE).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(argq.COLUMNS)
        for t, topic in enumerate(ARGQ_TOPICS):
            for i in range(ARGQ_PER_TOPIC):
                wa, mace = argq_values(t, i)
                writer.writerow(
                    [argq_text(t, i), topic, "train", repr(wa), repr(mace), 1, MARKER]
                )
        writer.writerow(
            [argq_text(0, 5).upper(), ARGQ_TOPICS[1], "train", "0.5", "0.5", -1, MARKER]
        )
        for split in ("dev", "test"):
            writer.writerow(["Skipped.", ARGQ_TOPICS[0], split, "0.5", "0.5", 1, 1.0])


def make_all(root: Path) -> dict[str, Path]:
    dirs = spec.resolve(root)
    make_mlqepe(dirs["mlqepe"])
    make_onestop(dirs["onestop"])
    make_klue(dirs["klue"])
    make_jglue(dirs["jglue"])
    make_argq(dirs["argq"])
    return dirs


def inputs_text(row: Mapping[str, Any]) -> str:
    return canonical({field: row[field] for field in INPUT_FIELDS})


def value_of(row: Mapping[str, Any]) -> float:
    family = "mlqepe" if row["family"].startswith("mlqepe_") else row["family"]
    return row["audit_metadata"][VALUE_FIELD[family]]


def cuts_of(report: Mapping[str, Any], row: Mapping[str, Any]) -> list[float]:
    size = f"L{len(row['options'])}"
    if row["family"] == "a6h2_argq":
        return report["cut_points"][row["state"]["topic"]][size]
    return report["cut_points"][size]


def level_cells(rows: Sequence[Mapping[str, Any]]) -> dict[int, list[int]]:
    counts: dict[int, collections.Counter[int]] = collections.defaultdict(
        collections.Counter
    )
    for row in rows:
        counts[len(row["options"])][row["label"]] += 1
    return {
        size: [counts[size][level] for level in range(size)] for size in sorted(counts)
    }


def setUpModule() -> None:
    tmp = tempfile.TemporaryDirectory()
    root = Path(tmp.name) / "sources"
    dirs = make_all(root)
    results = {item.family: item.build(dirs) for item in src_score.FAMILIES}
    FIXTURE.update(tmp=tmp, root=root, dirs=dirs, results=results)


def tearDownModule() -> None:
    FIXTURE.pop("tmp").cleanup()


class FixtureCase(unittest.TestCase):
    root: Path
    dirs: dict[str, Path]
    results: dict[str, tuple[list[dict[str, Any]], dict[str, Any]]]

    @classmethod
    def setUpClass(cls) -> None:
        cls.root, cls.dirs = FIXTURE["root"], FIXTURE["dirs"]
        cls.results = FIXTURE["results"]


class RegistryTest(unittest.TestCase):
    def test_families_arms_sources_caps_and_seeds(self) -> None:
        expected = [
            (f"mlqepe_{p.replace('-', '')}", "mlqepe_train", 3000, f"h6-mlqepe-{p}-v1")
            for p in ("en-de", "en-zh", "et-en", "ne-en", "ro-en", "ru-en", "si-en")
        ] + [
            ("onestop_level", "onestop_english", 1000, "h6-onestop-v1"),
            ("a6h2_klue_sts", "klue_sts_train", 1800, "h6-a6h2_klue_sts-v1"),
            ("a6h2_jsts", "jglue_jsts_v1.3_train", 1800, "h6-a6h2_jsts-v1"),
            ("a6h2_argq", "argq30k_train", 1800, "h6-a6h2_argq-v1"),
        ]
        self.assertEqual(
            [
                (item.family, item.source, item.cap_rows, item.seed)
                for item in src_score.FAMILIES
            ],
            expected,
        )
        self.assertEqual({item.arm for item in src_score.FAMILIES}, {"h6"})
        self.assertIn("src_score", build.SOURCE_MODULES)

    def test_percent_rounds_half_up_and_descriptions_state_bands(self) -> None:
        self.assertEqual([src_score.percent(k, 3) for k in range(4)], [0, 33, 67, 100])
        self.assertEqual(
            [src_score.percent(k, 8) for k in range(9)],
            [0, 13, 25, 38, 50, 63, 75, 88, 100],
        )
        self.assertEqual(
            src_score.describe([40.25, 71.0], src_score.MLQEPE_SCALE),
            [
                "Level 1 of 3 (ranked 0–33% from the bottom among translations in "
                "this language pair): mean human quality rating 0.0–40.2 on a 0–100 "
                "scale",
                "Level 2 of 3 (ranked 33–67% from the bottom among translations in "
                "this language pair): mean human quality rating 40.2–71.0 on a 0–100 "
                "scale",
                "Level 3 of 3 (ranked 67–100% from the bottom among translations in "
                "this language pair): mean human quality rating 71.0–100.0 on a "
                "0–100 scale",
            ],
        )
        self.assertEqual(
            src_score.describe([0.625], src_score.ARGQ_SCALE)[1],
            "Level 2 of 2 (ranked 50–100% from the bottom among arguments on this "
            "topic): weighted human quality score 0.62–1.00 on a 0–1 scale",
        )


class BinnedTest(unittest.TestCase):
    def items(self, values: Sequence[float]) -> list[src_score.Item]:
        return [
            src_score.Item(
                local_id=str(i),
                group_key=f"g{i // 2}",
                key=(str(i),),
                state={"text": f"item {i}"},
                value=value,
                audit={"value": value},
            )
            for i, value in enumerate(values)
        ]

    def run_binned(
        self, items: Sequence[src_score.Item], cap: int
    ) -> tuple[list[dict[str, Any]], dict[str, Any], collections.Counter[str]]:
        drops: collections.Counter[str] = collections.Counter()
        rows, report = src_score.binned(
            items,
            items,
            family="t_score",
            source="t_score_train",
            language="en",
            namespace="t",
            scale=src_score.MLQEPE_SCALE,
            seed="t-seed",
            cap=cap,
            drops=drops,
        )
        return rows, report, drops

    def test_cap_then_balance_keeps_levels_within_six_fifths(self) -> None:
        items = self.items([((i * 7919) % 3000) / 30 for i in range(3000)])
        rows, report, drops = self.run_binned(items, 400)
        self.assertLessEqual(len(rows), 400)
        self.assertGreater(drops["cap"], 0)
        self.assertEqual(len(items), sum(drops.values()) + len(rows))
        self.assertEqual(
            report["rows_before_cap"] - drops["cap"],
            len(rows) + drops["post_cap_balance"],
        )
        for counts in level_cells(rows).values():
            self.assertLessEqual(max(counts), min(counts) * 6 // 5)
        again = common.cap_groups(rows, 400, "t-seed")
        self.assertEqual(sorted(r["id"] for r in again), sorted(r["id"] for r in rows))
        self.assertEqual(report["levels_after_cap"], src_score.level_table(rows))

    def test_ties_at_the_scale_minimum_empty_high_level_cells(self) -> None:
        values = [0.0 if i % 5 < 2 else 1 + (i % 97) for i in range(2000)]
        rows, report, drops = self.run_binned(self.items(values), 10_000)
        self.assertEqual(report["cut_points"]["L3"][0], 0.0)
        self.assertIn("L3", report["empty_level_cells"])
        self.assertNotIn("L2", report["empty_level_cells"])
        self.assertEqual({len(row["options"]) for row in rows}, {2})
        self.assertEqual(
            report["empty_level_cells_after_cap"], [f"L{k}" for k in range(3, 11)]
        )
        self.assertEqual(report["balance_cells"]["L3"]["after"], [0, 0, 0])
        self.assertEqual(drops["cap"], 0)


class MlqepeTest(FixtureCase):
    def test_tar_member_parsing_and_drops(self) -> None:
        rows, report = self.results["mlqepe_ende"]
        self.assertEqual(
            list(report["inputs"]), ["data/direct-assessments/train/en-de-train.tar.gz"]
        )
        self.assertEqual(report["member"]["name"], EN_DE_MEMBER)
        self.assertEqual(report["candidates"], EN_DE_ROWS + 9)
        drops = report["drops"]
        self.assertEqual(
            {k: drops[k] for k in src_score.MLQEPE_DROPS},
            {
                "malformed_line": 1,
                "duplicate_index": 1,
                "empty_text": 1,
                "invalid_mean": 2,
                "duplicate_state": 1,
            },
        )
        self.assertEqual(
            (report["blank_lines"], report["mean_differs_from_scores"]), (1, 1)
        )
        self.assertEqual(report["candidates"], sum(drops.values()) + len(rows))
        self.assertEqual({row["language"] for row in rows}, {"de"})
        self.assertEqual(self.results["mlqepe_eten"][1]["language"], "et")
        self.assertEqual(
            {row["language"] for row in self.results["mlqepe_enzh"][0]}, {"zh"}
        )

    def test_unquoted_fields_group_keys_and_states(self) -> None:
        data = src_score.read_member(
            self.dirs["mlqepe"] / src_score.MLQEPE_DIR / "en-de-train.tar.gz",
            EN_DE_MEMBER,
        )
        items, _ = src_score.mlqepe_items(data, "en-de", "t", collections.Counter())
        by_id = {item.local_id: item for item in items}
        quoted = by_id[f"en-de:{EN_DE_ROWS}"]
        self.assertEqual(quoted.state["source"], '"Quoted opening," the report said.')
        self.assertEqual(quoted.audit["raters"], 2)
        self.assertEqual(
            by_id[f"en-de:{EN_DE_ROWS + 6}"].group_key, by_id["en-de:11"].group_key
        )
        self.assertNotIn(f"en-de:{EN_DE_ROWS + 1}", by_id)
        for row in self.results["mlqepe_ende"][0]:
            self.assertEqual(set(row["state"]), {"source", "translation"})
            self.assertEqual(
                row["group_id"],
                common.group_id("mlqepe", normalize(row["state"]["source"])),
            )
            self.assertTrue(
                row["audit_metadata"]["source_local_id"].startswith("en-de:")
            )
            self.assertEqual(row["render_template"], "m2/mlqepe_ende/v1")

    def test_only_the_train_tsv_member_is_read(self) -> None:
        for item in src_score.FAMILIES:
            rows, _ = self.results[item.family]
            for row in rows:
                self.assertNotIn(MARKER, inputs_text(row), row["id"])


class OnestopTest(FixtureCase):
    def test_rows_normalization_and_drops(self) -> None:
        rows, report = self.results["onestop_level"]
        self.assertEqual(len(rows), 6)
        self.assertEqual(
            report["drops"],
            {
                "duplicate_article": 4,
                "empty_text": 1,
                "header_mismatch": 3,
                "identical_versions": 3,
                "missing_level": 4,
                "unexpected_name": 1,
            },
        )
        self.assertEqual(report["candidates"], 22)
        self.assertEqual(
            report["candidates"], sum(report["drops"].values()) + len(rows)
        )
        self.assertEqual(report["ignored_files"], 5)
        self.assertEqual(report["level_histogram"], {"0": 2, "1": 2, "2": 2})
        self.assertEqual(
            report["utf8_replacement"],
            {
                "files": [
                    "Texts-SeparatedByReadingLevel/Adv-Txt/River-Clean-Up-adv.txt"
                ],
                "characters": 1,
            },
        )
        self.assertEqual(report["header_lines_removed"], {"ele": 1, "int": 5, "adv": 0})
        self.assertEqual(
            report["non_ascii_removed"],
            {
                "ele": {"rows_with_removals": 1, "characters": 4},
                "int": {"rows_with_removals": 0, "characters": 0},
                "adv": {"rows_with_removals": 2, "characters": 6},
            },
        )
        states = {
            (row["audit_metadata"]["article"], row["label"]): row["state"]
            for row in rows
        }
        for article, texts in ONESTOP_STATES.items():
            for label, text in enumerate(texts):
                self.assertEqual(states[article, label], text)
        for row in rows:
            self.assertTrue(row["state"].isascii())
            self.assertEqual(
                [option["description"] for option in row["options"]],
                list(src_score.ONESTOP_OPTIONS),
            )
            self.assertEqual(row["instructions"], src_score.ONESTOP_INSTRUCTIONS)
            self.assertEqual(row["task_type"], "score")

    def test_one_group_per_article_with_one_row_per_level(self) -> None:
        rows, _ = self.results["onestop_level"]
        groups: dict[str, list[int]] = collections.defaultdict(list)
        for row in rows:
            groups[row["group_id"]].append(row["label"])
            article = row["audit_metadata"]["article"]
            self.assertEqual(
                row["group_id"], common.group_id("onestop", normalize(article))
            )
        self.assertEqual(
            sorted(sorted(labels) for labels in groups.values()), [[0, 1, 2]] * 2
        )
        self.assertEqual(
            sorted(row["audit_metadata"]["source_local_id"] for row in rows),
            sorted(
                f"{article}:{code}"
                for article in ONESTOP_STATES
                for code in ("ele", "int", "adv")
            ),
        )


class A6h2Test(FixtureCase):
    def test_local_ids_values_and_groups_match_v1(self) -> None:
        cases = (
            (
                klue.sts,
                src_score.klue_sts_items,
                "klue",
                "mean_rating",
                "klue_sts_train",
            ),
            (
                jglue.jsts,
                src_score.jsts_items,
                "jglue",
                "mean_rating",
                "jglue_jsts_v1.3_train",
            ),
            (argq.argq, src_score.argq_items, "argq", "wa", "argq30k_train"),
        )
        for v1_loader, parse, name, field, source in cases:
            v1_rows, _ = v1_loader(self.dirs[name])
            items = {item.local_id: item for item in parse(self.dirs[name])[0]}
            self.assertGreater(len(v1_rows), 100)
            for row in v1_rows:
                item = items[row["audit_metadata"]["source_local_id"]]
                self.assertEqual(item.value, row["audit_metadata"][field])
                self.assertEqual(
                    row["group_id"], f"a6h:{source}:{common.sha(item.group_key)[:20]}"
                )
        family_rows = self.results["a6h2_jsts"][0]
        for row in family_rows:
            image = jsts_image(int(row["audit_metadata"]["source_local_id"]))
            self.assertEqual(row["group_id"], common.group_id("a6h2", image))
        self.assertLess(len({row["group_id"] for row in family_rows}), len(family_rows))

    def test_states_languages_and_instructions(self) -> None:
        expected = {
            "a6h2_klue_sts": ("ko", {"sentence_1", "sentence_2"}, src_score.STS_SCALE),
            "a6h2_jsts": ("ja", {"sentence_1", "sentence_2"}, src_score.STS_SCALE),
            "a6h2_argq": ("en", {"topic", "argument"}, src_score.ARGQ_SCALE),
        }
        for family, (language, keys, scale) in expected.items():
            rows, report = self.results[family]
            self.assertEqual(report["v1_local_ids"], 0)
            for row in rows:
                self.assertEqual((row["language"], set(row["state"])), (language, keys))
                self.assertEqual(row["instructions"], scale.instructions)
                self.assertNotIn(str(value_of(row)), canonical(row["state"]))
        klue_report = self.results["a6h2_klue_sts"][1]
        self.assertEqual(klue_report["drops"]["duplicate_state"], 1)
        self.assertEqual(
            sorted(klue_report["strata"]), [f"{MARKER}-rtt", f"{MARKER}-sampled"]
        )
        argq_report = self.results["a6h2_argq"][1]
        self.assertEqual(
            (argq_report["non_train_rows_skipped"], argq_report["topics"]), (2, 3)
        )
        self.assertGreater(argq_report["drops"]["mace_p_disagreement"], 0)
        self.assertEqual(
            sum(argq_report["mace_p_disagreement_drops"].values()),
            argq_report["drops"]["mace_p_disagreement"],
        )

    def test_v1_items_groups_and_states_are_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            v1_file = Path(tmp) / "v1.jsonl"
            write_jsonl(
                v1_file,
                [
                    {"source": source, "audit_metadata": {"source_local_id": local_id}}
                    for source, local_id in (
                        ("klue_sts_train", "klue-sts-v1_train_00003"),
                        ("jglue_jsts_v1.3_train", "7"),
                        ("argq30k_train", "row5"),
                        ("other_train", "row6"),
                    )
                ],
            )
            manifest = Path(tmp) / "v1-rows.json"
            manifest.write_text(json.dumps([str(v1_file)]), encoding="utf-8")
            dirs = {**self.dirs, build.V1_ROWS_KEY: manifest}
            expected = {
                "a6h2_klue_sts": (
                    "klue-sts-v1_train_00003",
                    {"v1_item": 1, "v1_state": 1},
                ),
                "a6h2_jsts": ("7", {"v1_item": 1, "v1_group": 2}),
                "a6h2_argq": ("row5", {"v1_item": 1, "v1_group": 1}),
            }
            for item in src_score.FAMILIES:
                if item.family not in expected:
                    continue
                local_id, counts = expected[item.family]
                rows, report = item.build(dirs)
                self.assertEqual(report["v1_local_ids"], 1)
                for reason in ("v1_item", "v1_group", "v1_state", "duplicate_state"):
                    self.assertEqual(
                        report["drops"][reason], counts.get(reason, 0), reason
                    )
                ids = {row["audit_metadata"]["source_local_id"] for row in rows}
                self.assertNotIn(local_id, ids)
                self.assertEqual(
                    report["candidates"], sum(report["drops"].values()) + len(rows)
                )
            write_jsonl(
                v1_file,
                [
                    {
                        "source": "klue_sts_train",
                        "audit_metadata": {"source_local_id": "x"},
                    }
                ],
            )
            klue_spec = next(
                i for i in src_score.FAMILIES if i.family == "a6h2_klue_sts"
            )
            with self.assertRaisesRegex(ValueError, "not in the TRAIN file"):
                klue_spec.build(dirs)


class C10FixtureTest(FixtureCase):
    def test_every_level_count_present_and_balanced(self) -> None:
        for family in FULL_DESIGN:
            rows, report = self.results[family]
            cells = level_cells(rows)
            self.assertEqual(sorted(cells), list(src_score.LEVELS), family)
            for size, counts in cells.items():
                self.assertGreater(min(counts), 0, (family, size))
                self.assertLessEqual(max(counts), min(counts) * 6 // 5, (family, size))
            self.assertEqual(report["empty_level_cells"], [], family)
            self.assertEqual(report["empty_level_cells_after_cap"], [], family)
            self.assertEqual(report["levels_after_cap"], src_score.level_table(rows))
            self.assertEqual(
                sum(report["levels_assigned"].values()),
                report["candidates"]
                - sum(report["drops"].get(reason, 0) for reason in PRE_BINNING_DROPS),
            )

    def test_levels_follow_hash_cuts_and_guard_band(self) -> None:
        for family, scale in FULL_DESIGN.items():
            rows, report = self.results[family]
            options: dict[tuple[str, int], list[dict[str, str]]] = {}
            for row in rows:
                local_id = row["audit_metadata"]["source_local_id"]
                size = len(row["options"])
                self.assertEqual(
                    size, ordinal.level_count(family, local_id, src_score.LEVELS)
                )
                self.assertEqual(row["audit_metadata"]["levels"], size)
                cuts = cuts_of(report, row)
                self.assertEqual(row["label"], ordinal.bin_index(value_of(row), cuts))
                self.assertFalse(ordinal.near_cut(value_of(row), cuts, scale.guard))
                self.assertEqual(
                    [option["description"] for option in row["options"]],
                    src_score.describe(cuts, scale),
                )
                topic = row["state"].get("topic", "") if family == "a6h2_argq" else ""
                self.assertEqual(
                    options.setdefault((topic, size), row["options"]), row["options"]
                )

    def test_cut_points_are_train_quantiles_of_every_item(self) -> None:
        data = src_score.read_member(
            self.dirs["mlqepe"] / src_score.MLQEPE_DIR / "en-de-train.tar.gz",
            EN_DE_MEMBER,
        )
        items, _ = src_score.mlqepe_items(data, "en-de", "t", collections.Counter())
        report = self.results["mlqepe_ende"][1]
        for size in src_score.LEVELS:
            self.assertEqual(
                report["cut_points"][f"L{size}"],
                ordinal.quantile_cuts([item.value for item in items], size),
            )
        argq_report = self.results["a6h2_argq"][1]
        for t, topic in enumerate(ARGQ_TOPICS):
            values = [argq_values(t, i)[0] for i in range(ARGQ_PER_TOPIC)]
            if t == 1:
                values.append(0.5)
            self.assertEqual(
                argq_report["cut_points"][topic]["L7"], ordinal.quantile_cuts(values, 7)
            )
        klue_report = self.results["a6h2_klue_sts"][1]
        values = [sts_value(i) for i in range(STS_ROWS)] + [2.5]
        self.assertEqual(
            klue_report["cut_points"]["L4"], ordinal.quantile_cuts(values, 4)
        )

    def test_rows_valid_and_drops_accounted(self) -> None:
        for item in src_score.FAMILIES:
            rows, report = self.results[item.family]
            self.assertTrue(rows, item.family)
            self.assertEqual(
                report["candidates"], sum(report["drops"].values()) + len(rows)
            )
            self.assertLessEqual(len(rows), item.cap_rows)
            ids = [row["id"] for row in rows]
            self.assertEqual(len(ids), len(set(ids)))
            json.dumps(report, allow_nan=False)
            for row in rows:
                validate_row(row, "train")
                self.assertEqual(
                    (row["family"], row["source"]), (item.family, item.source)
                )
                self.assertEqual(
                    [option["key"] for option in row["options"]],
                    [str(k) for k in range(len(row["options"]))],
                )


class DeterminismTest(FixtureCase):
    def digest(self) -> str:
        out = [self.results[item.family] for item in src_score.FAMILIES]
        return hashlib.sha256(canonical(out).encode("utf-8")).hexdigest()

    def test_rebuild_is_identical(self) -> None:
        again = {item.family: item.build(self.dirs) for item in src_score.FAMILIES}
        self.assertEqual(canonical(again), canonical(self.results))

    def test_independent_of_hash_seed(self) -> None:
        expected = self.digest()
        for seed in ("0", "4242"):
            result = subprocess.run(
                [sys.executable, "-c", DIGEST_SCRIPT, str(self.root)],
                cwd=PACKAGE_ROOT,
                env={**os.environ, "PYTHONHASHSEED": seed},
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertEqual(result.stdout.strip(), expected, seed)

    def test_orchestrator_cap_is_a_no_op_and_arm_writes(self) -> None:
        rows, report = build.build(
            "h6", dict(self.dirs), workers=1, modules=("src_score",)
        )
        self.assertEqual(sorted(report["families"]), sorted(self.results))
        for family, entry in report["families"].items():
            self.assertEqual(entry["after_cap"], len(self.results[family][0]), family)
            self.assertEqual(entry["candidates"], entry["after_cap"], family)
        self.assertEqual(
            sorted(row["id"] for row in rows),
            sorted(row["id"] for rows_, _ in self.results.values() for row in rows_),
        )
        with tempfile.TemporaryDirectory() as out:
            manifest = common.write_arm(rows, Path(out), "h6", report)
            self.assertEqual(
                sum(manifest[part]["rows"] for part in common.SLICES), len(rows)
            )


if __name__ == "__main__":
    unittest.main()
