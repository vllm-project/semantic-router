from __future__ import annotations

import contextlib
import csv
import io
import json
import statistics
import tempfile
import unittest
from collections import Counter
from collections.abc import Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

from training.model.data import INPUT_FIELDS, canonical, file_sha256, validate_row
from v2.data import build_a5_a6h as build
from v2.data.shortcut import hypothesis_text
from v2.data.sources import argq, jglue, klue, ordinal, saf
from v2.data.sources.common import choice_options, rotate, sha

MARKER = "MARKER"
YNAT_TITLES = (
    "스마트폰 새 운영체제 업데이트 다음 달 배포",
    "기준금리 동결…시장 금리 소폭 하락",
    "초등학교 급식 식중독 의심 신고 잇따라",
    "주말 전국 맑고 일교차 커…건강 관리 유의",
    "유럽연합, 새 기후 협약 초안 공개",
    "프로축구 개막전 만원 관중…홈팀 2-1 승리",
    "여야, 예산안 처리 시한 앞두고 막판 협상",
)
KLUE_NLI = (
    (
        "도서관은 평일 오전 9시부터 오후 6시까지 문을 연다.",
        (
            "도서관은 평일 낮에 이용할 수 있다.",
            "도서관은 주말에도 문을 연다.",
            "도서관은 평일에 문을 열지 않는다.",
        ),
    ),
    (
        "민수는 어제 친구와 함께 영화를 보러 극장에 갔다.",
        (
            "민수는 어제 외출했다.",
            "민수가 본 영화는 코미디였다.",
            "민수는 어제 하루 종일 집에 있었다.",
        ),
    ),
    (
        "이 식당은 예약한 손님만 받는다.",
        (
            "예약하지 않으면 이 식당에서 식사할 수 없다.",
            "이 식당은 저녁에만 영업한다.",
            "이 식당은 예약 없이도 누구나 들어갈 수 있다.",
        ),
    ),
)
KLUE_REVERSED = ("회의는 오후 3시에 시작했다.", "회의는 오후에 시작했다.")
KLUE_MRC = (
    (
        "지역 봄꽃 축제 개막…사흘간 열려",
        "올해 봄꽃 축제가 12일 시청 앞 광장에서 개막했다. 축제는 사흘 동안 열리며 지역 농산물 "
        "장터와 거리 공연이 함께 진행된다. 주최 측은 첫날에만 약 2만 명이 다녀간 것으로 추산했다.",
        (
            ("축제는 며칠 동안 열리는가?", False),
            ("축제 첫날 방문객은 약 몇 명인가?", False),
            ("축제 입장료는 얼마인가?", True),
        ),
    ),
    (
        "도심 자전거 도로 확대",
        "시는 내년까지 도심 자전거 도로를 50킬로미터 늘리겠다고 밝혔다. 새 도로는 주요 지하철역과 "
        "공원을 연결하며, 공사는 다음 달 착공한다.",
        (
            ("자전거 도로는 얼마나 늘어나는가?", False),
            ("공사는 언제 착공하는가?", False),
            ("공사 예산은 얼마인가?", True),
        ),
    ),
    (
        "신인 작가 장편소설 베스트셀러 1위",
        "한 신인 작가의 장편소설이 출간 2주 만에 온라인 서점 베스트셀러 1위에 올랐다. 출판사는 "
        "이미 3쇄를 찍었다고 밝혔다.",
        (
            ("소설은 출간 몇 주 만에 1위에 올랐는가?", False),
            ("출판사는 몇 쇄를 찍었는가?", False),
            ("작가의 나이는 몇 살인가?", True),
        ),
    ),
)
JCQA = (
    (
        "雨の日に出かけるとき、手に持っていくものは？",
        ("扇風機", "傘", "冷蔵庫", "布団", "鉛筆"),
        1,
    ),
    ("魚を焼くときに使う道具は？", ("まな板", "掃除機", "グリル", "洗濯機", "本棚"), 2),
    (
        "手紙を送るときに封筒に貼るものは？",
        ("切手", "付箋", "絆創膏", "名札", "ボタン"),
        0,
    ),
    ("夜空に光って見えるものは？", ("雲", "砂", "石", "葉", "星"), 4),
    ("果物はどれ？", ("りんご", "りんご", "にんじん", "たまねぎ", "ごぼう"), 0),
)
JNLI = (
    (
        "男性が公園のベンチに座って新聞を読んでいます。",
        "男性が屋外で座っています。",
        "entailment",
    ),
    (
        "男性が屋外で座っています。",
        "男性が公園のベンチに座って新聞を読んでいます。",
        "neutral",
    ),
    (
        "女の子が犬と一緒に砂浜を走っています。",
        "女の子が部屋の中で眠っています。",
        "contradiction",
    ),
    (
        "女の子が部屋の中で眠っています。",
        "女の子が犬と一緒に砂浜を走っています。",
        "contradiction",
    ),
    (
        "テーブルの上にケーキと紅茶が置かれています。",
        "テーブルの上に食べ物があります。",
        "entailment",
    ),
    (
        "テーブルの上にケーキと紅茶が置かれています。",
        "誕生日のお祝いをしています。",
        "neutral",
    ),
)
JNLI_IMAGES = ("100124", "100124", "200001", "200002", "300001", "300001")
ARGQ_TOPICS = ("We should ban fast food", "We should adopt a four-day work week")
SAF_ITEMS = {
    "en": (
        (
            "What is the purpose of a checksum in a network packet?",
            "A checksum lets the receiver detect whether the packet was corrupted in transit.",
            (
                (
                    "It lets the receiver detect corrupted packets by recomputing the value.",
                    "Correct",
                    1.0,
                ),
                (
                    "It detects errors, and it also encrypts the packet.",
                    "Partially correct",
                    0.5,
                ),
                ("It tells the router which path to take.", "Incorrect", 0.0),
            ),
        ),
        (
            "Name one advantage of TCP over UDP.",
            "TCP provides reliable, ordered delivery through acknowledgements and retransmission.",
            (
                (
                    "TCP retransmits lost segments, so delivery is reliable.",
                    "Correct",
                    1.0,
                ),
                ("TCP is connection-based.", "Partially correct", 0.5),
                ("TCP is faster because it needs no handshake.", "Incorrect", 0.0),
            ),
        ),
    ),
    "de": (
        (
            "Frage 1: Was müssen Sie fotografieren, bevor Sie den Laden betreten?",
            "Es muss ein Foto des Eingangsbereichs mit dem Ladenschild angefertigt werden (0.5 P). "  # codespell:ignore foto
            "Das Foto muss bei Tageslicht aufgenommen werden (0.5 P).",  # codespell:ignore foto
            (
                (
                    "Ich fotografiere den Eingang mit dem Schild, solange es hell ist.",
                    "Correct",
                    1.0,
                ),
                (
                    "Ich mache ein Foto vom Eingang.",  # codespell:ignore foto
                    "Partially correct",
                    0.5,
                ),  # codespell:ignore foto
                ("Ich fotografiere die Kasse.", "Incorrect", 0.0),
            ),
        ),
        (
            "Frage 2: Was tun Sie, wenn das Geschäft geschlossen ist?",
            "Ein Foto der geschlossenen Tür mit den Öffnungszeiten anfertigen (0.5 P) und den "  # codespell:ignore foto
            "Auftrag im Fragebogen als nicht durchführbar markieren (0.5 P).",
            (
                (
                    "Ich fotografiere die Tür mit den Öffnungszeiten und vermerke es im Fragebogen.",
                    "Correct",
                    1.0,
                ),
                (
                    "Ich mache ein Foto von der Tür.",  # codespell:ignore foto
                    "Partially correct",
                    0.5,
                ),  # codespell:ignore foto
                ("Ich warte, bis jemand aufmacht.", "Incorrect", 0.0),
            ),
        ),
    ),
}


def write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def klue_sts_mean(index: int) -> float:
    return ((index * 11) % 36) / 7


def jsts_image(index: int) -> str:
    return (
        f"{4000 + index // 4}_{5000 + index // 4}"
        if index % 8 == 7
        else str(4000 + index // 4)
    )


def jsts_label(index: int) -> float:
    return ((index * 7) % 26) / 5


def make_klue(root: Path) -> None:
    write_jsonl(
        root / klue.YNAT_FILE,
        [
            {
                "guid": f"ynat-v1_train_{i:05d}",
                "title": YNAT_TITLES[i % 7] + (" (2보)" if i >= 7 else ""),
                "label": i % 7,
                "url": f"https://news.example.invalid/{MARKER}-url-{i}",
                "date": f"{MARKER}-date-{i}",
            }
            for i in range(14)
        ],
    )
    nli = [
        (premise, hypothesis, label)
        for premise, hypotheses in KLUE_NLI
        for label, hypothesis in enumerate(hypotheses)
    ]
    nli += [(*KLUE_REVERSED, 0), (*reversed(KLUE_REVERSED), 1)]
    write_jsonl(
        root / klue.NLI_FILE,
        [
            {
                "guid": f"klue-nli-v1_train_{i:05d}",
                "source": f"{MARKER}-nli-source",
                "premise": premise,
                "hypothesis": hypothesis,
                "label": label,
            }
            for i, (premise, hypothesis, label) in enumerate(nli)
        ],
    )
    mrc = [
        (title, context, question, impossible)
        for title, context, questions in KLUE_MRC
        for question, impossible in questions
    ]
    write_jsonl(
        root / klue.MRC_FILE,
        [
            {
                "title": title,
                "context": context,
                "news_category": f"{MARKER}-category",
                "source": f"{MARKER}-mrc-source",
                "guid": f"klue-mrc-v1_train_{i:05d}",
                "is_impossible": impossible,
                "question_type": 3 if impossible else 1,
                "question": question,
                "answers": {"answer_start": [0], "text": [f"{MARKER}-answer"]},
            }
            for i, (title, context, question, impossible) in enumerate(mrc)
        ],
    )
    write_jsonl(
        root / klue.STS_FILE,
        [
            {
                "guid": f"klue-sts-v1_train_{i:05d}",
                "source": f"{MARKER}-stratum-{'rtt' if i % 2 else 'sampled'}",
                "sentence1": f"숙소는 역에서 걸어서 {i % 9 + 3}분 거리에 있고 방이 {i % 4 + 1}개 있습니다.",
                "sentence2": f"역에서 숙소까지 도보로 약 {i % 9 + 3}분 걸리며 근처에 편의점이 있습니다.",
                "labels": {
                    "label": round(klue_sts_mean(i), 1),
                    "real-label": klue_sts_mean(i),
                    "binary-label": int(klue_sts_mean(i) >= 3),
                },
            }
            for i in range(160)
        ],
    )
    (root / "ynat/validation.jsonl").write_text("never opened\n", encoding="utf-8")


def make_jglue(root: Path) -> None:
    write_jsonl(
        root / jglue.JCQA_FILE,
        [
            {
                "q_id": i,
                "question": question,
                **{f"choice{k}": c for k, c in enumerate(choices)},
                "label": label,
            }
            for i, (question, choices, label) in enumerate(JCQA)
        ],
    )
    write_jsonl(
        root / jglue.JNLI_FILE,
        [
            {
                "sentence_pair_id": str(i),
                "yjcaptions_id": f"{JNLI_IMAGES[i]}-{9000 + i}-{9100 + i}",
                "sentence1": first,
                "sentence2": second,
                "label": label,
            }
            for i, (first, second, label) in enumerate(JNLI)
        ],
    )
    write_jsonl(
        root / jglue.JSTS_FILE,
        [
            {
                "sentence_pair_id": str(i),
                "yjcaptions_id": f"{jsts_image(i)}-{i}-{i + 1000}",
                "sentence1": f"{i % 5 + 2}人の子どもが公園で遊んでいます。",
                "sentence2": f"公園で子どもたちが{('ボール', '縄跳び', '鬼ごっこ')[i % 3]}をしています。",
                "label": jsts_label(i),
            }
            for i in range(160)
        ],
    )
    (root / "datasets/jnli-v1.3/valid-v1.3.json").write_text(
        "never opened\n", encoding="utf-8"
    )


def argq_values(topic: int, index: int) -> tuple[float, float]:
    wa = round(((index * 37 + topic * 11) % 60) / 59, 9)
    return wa, round(1 - wa, 9) if index % 10 == 0 else wa


def make_argq(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with (root / argq.FILE).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(argq.COLUMNS)
        for t, topic in enumerate(ARGQ_TOPICS):
            for i in range(60):
                wa, mace = argq_values(t, i)
                writer.writerow(
                    [
                        f"Point {i}: the proposal that {topic.lower()} would change daily habits for the better.",
                        topic,
                        "train",
                        repr(wa),
                        repr(mace),
                        1 if i % 2 else -1,
                        1.0,
                    ]
                )
        for split in ("dev", "test"):
            writer.writerow(
                [
                    f"A {split} argument that must be skipped.",
                    ARGQ_TOPICS[0],
                    split,
                    "0.5",
                    "0.5",
                    1,
                    1.0,
                ]
            )


def make_saf(root: Path, language: str, verdict_override: str | None = None) -> None:
    records = []
    for q, (question, reference, answers) in enumerate(SAF_ITEMS[language]):
        for a, (answer, verdict, score) in enumerate(answers):
            records.append(
                {
                    "id": sha(f"saf-{language}-{q}-{a}")[:32],
                    "question": question,
                    "reference_answer": reference,
                    "provided_answer": answer,
                    "answer_feedback": f"{MARKER}-feedback-{q}-{a}",
                    "verification_feedback": verdict_override or verdict,
                    "score": score,
                }
            )
    write_jsonl(root / saf.FILE, records)


def make_all(root: Path) -> dict[str, Path]:
    dirs = {name: root / name for name in ("klue", "jglue", "argq", "saf_en", "saf_de")}
    make_klue(dirs["klue"])
    make_jglue(dirs["jglue"])
    make_argq(dirs["argq"])
    make_saf(dirs["saf_en"], "en")
    make_saf(dirs["saf_de"], "de")
    return dirs


def inputs_text(row: Mapping[str, Any]) -> str:
    return canonical({field: row[field] for field in INPUT_FIELDS})


def gold(row: Mapping[str, Any]) -> str:
    return row["options"][row["label"]]["description"]


class FixtureCase(unittest.TestCase):
    dirs: dict[str, Path]

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.dirs = make_all(Path(cls._tmp.name))

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()


class OrdinalTest(unittest.TestCase):
    def test_level_count_is_seeded_sha256(self) -> None:
        drawn = [ordinal.level_count("fam", str(i), (3, 4, 5, 6)) for i in range(400)]
        self.assertEqual(
            drawn,
            [ordinal.level_count("fam", str(i), (3, 4, 5, 6)) for i in range(400)],
        )
        self.assertEqual(set(drawn), {3, 4, 5, 6})
        self.assertEqual(drawn[7], (3, 4, 5, 6)[int(sha("fam:7"), 16) % 4])

    def test_quantile_cuts_match_linear_interpolation(self) -> None:
        self.assertEqual(ordinal.quantile_cuts([4, 0, 3, 1, 2], 4), [1.0, 2.0, 3.0])
        self.assertEqual(ordinal.quantile_cuts([0, 0, 0, 1], 2), [0.0])
        values = [((i * 13) % 29) / 7 for i in range(83)]
        for levels in (3, 4, 5):
            expected = statistics.quantiles(values, n=levels, method="inclusive")
            for mine, theirs in zip(
                ordinal.quantile_cuts(values, levels), expected, strict=True
            ):
                self.assertAlmostEqual(mine, theirs, places=12)
        with self.assertRaises(ValueError):
            ordinal.quantile_cuts([], 3)

    def test_anchor_levels_and_guard_band(self) -> None:
        cuts = ordinal.anchor_cuts(5)
        self.assertEqual(cuts, [0.5, 1.5, 2.5, 3.5, 4.5])
        self.assertEqual(
            [
                ordinal.anchor_level(v, 5)
                for v in (-0.2, 0.0, 0.3, 0.7, 2.5, 4.8, 5.0, 5.4)
            ],
            [0, 0, 0, 1, 3, 5, 5, 5],
        )
        for value in (0.0, 0.29, 1.2, 2.8, 3.8, 4.71, 5.0):
            self.assertFalse(ordinal.near_cut(value, cuts, 0.2), value)
            self.assertEqual(
                ordinal.anchor_level(value, 5), ordinal.bin_index(value, cuts)
            )
        for value in (0.5, 0.3, 0.31, 0.7, 3.3, 3.7, 4.69):
            self.assertTrue(ordinal.near_cut(value, cuts, 0.2), value)
        self.assertTrue(ordinal.near_cut(0.52, [0.5], 0.02))
        self.assertFalse(ordinal.near_cut(0.521, [0.5], 0.02))
        self.assertEqual(
            [ordinal.bin_index(v, [1.0, 2.0]) for v in (0.5, 1.0, 1.5, 2.5)],
            [0, 1, 1, 2],
        )

    def test_bands_and_numbers(self) -> None:
        self.assertEqual(
            ordinal.bands([1.0, 2.5], 0, 5), [(0, 1.0), (1.0, 2.5), (2.5, 5)]
        )
        self.assertEqual(
            [ordinal.number(v) for v in (0, 5.0, 0.62, 4.5, 3.349)],
            ["0", "5", "0.6", "4.5", "3.3"],
        )

    def test_balance_keeps_six_fifths_of_rarest_level(self) -> None:
        items = [
            ("c", level, f"i{n}")
            for n, level in enumerate([0] * 10 + [1] * 30 + [2] * 50)
        ]
        kwargs = dict(
            cell=lambda x: x[0],
            level=lambda x: x[1],
            levels=lambda x: 3,
            ident=lambda x: x[2],
        )
        kept, report = ordinal.balance(items, seed="s", **kwargs)
        self.assertEqual(
            report["c"], {"before": [10, 30, 50], "after": [10, 12, 12], "limit": 12}
        )
        self.assertEqual(Counter(x[1] for x in kept), {0: 10, 1: 12, 2: 12})
        self.assertEqual(kept, ordinal.balance(items, seed="s", **kwargs)[0])
        self.assertNotEqual(kept, ordinal.balance(items, seed="other", **kwargs)[0])
        exact, _ = ordinal.balance(items, seed="s", ratio=Fraction(1), **kwargs)
        self.assertEqual(Counter(x[1] for x in exact), {0: 10, 1: 10, 2: 10})
        missing = [("c", level, f"m{n}") for n, level in enumerate([1] * 5 + [2] * 5)]
        emptied, report = ordinal.balance(missing, seed="s", **kwargs)
        self.assertEqual(
            (emptied, report["c"]["limit"], report["c"]["before"]), ([], 0, [0, 5, 5])
        )


class KlueTest(FixtureCase):
    def test_ynat_sections_and_state(self) -> None:
        rows, report = klue.ynat(self.dirs["klue"])
        self.assertEqual(len(rows), 14)
        self.assertEqual(report["train_class_histogram"]["스포츠"], 2)
        for row in rows:
            original = row["audit_metadata"]["original_label"]
            self.assertEqual(gold(row), klue.YNAT_SECTIONS[original])
            guid = row["audit_metadata"]["source_local_id"]
            row_id = f"a5-klue_ynat-{sha('klue_ynat_train|' + guid)[:20]}"
            self.assertEqual(row["id"], row_id)
            seed = f"a5-v1:{row_id}"
            self.assertEqual(
                (row["options"], row["label"]),
                rotate(choice_options(klue.YNAT_SECTIONS), original, seed),
            )
            self.assertTrue(row["state"].startswith(YNAT_TITLES[original]))
            self.assertEqual(
                (row["task_type"], row["language"], len(row["options"])),
                ("choice", "ko", 7),
            )
        self.assertEqual(len({row["group_id"] for row in rows}), 14)
        self.assertGreater(len({row["label"] for row in rows}), 1)

    def test_nli_relations_and_premise_groups(self) -> None:
        rows, report = klue.nli(self.dirs["klue"])
        self.assertEqual(
            report["train_class_histogram"],
            {"entailment": 4, "neutral": 4, "contradiction": 3},
        )
        pairs = [(p, h) for p, hs in KLUE_NLI for h in hs]
        pairs += [KLUE_REVERSED, tuple(reversed(KLUE_REVERSED))]
        for row, (premise, hypothesis) in zip(rows, pairs, strict=True):
            relation = klue.NLI_LABELS.index(row["audit_metadata"]["original_label"])
            self.assertEqual(gold(row), klue.NLI_TEXT["options"][relation])
            self.assertEqual(
                row["state"], {"premise": premise, "hypothesis": hypothesis}
            )
            self.assertEqual(hypothesis_text(row["state"]), hypothesis)
        groups = [row["group_id"] for row in rows]
        self.assertEqual(len(set(groups[:3])), 1)
        self.assertEqual(len(set(groups[:9])), 3)
        self.assertEqual(groups[9], groups[10])
        self.assertNotIn(groups[9], groups[:9])

    def test_mrc_answerable_labels(self) -> None:
        rows, report = klue.mrc(self.dirs["klue"])
        self.assertEqual(
            report["train_class_histogram"], {"answerable": 6, "unanswerable": 3}
        )
        for row in rows:
            self.assertEqual([o["key"] for o in row["options"]], ["false", "true"])
            self.assertEqual(
                row["label"], 0 if row["audit_metadata"]["is_impossible"] else 1
            )
            self.assertIn("\n질문: ", row["instructions"])
            self.assertTrue(row["state"].startswith("제목: "))
        self.assertEqual(len({row["group_id"] for row in rows}), 3)

    def test_sts_cut_points_guard_band_and_anchor_levels(self) -> None:
        rows, report = klue.sts(self.dirs["klue"])
        means = [klue_sts_mean(i) for i in range(160)]
        for levels in (3, 4, 5):
            self.assertEqual(
                report["cut_points"][f"L{levels}"], ordinal.quantile_cuts(means, levels)
            )
        self.assertEqual(report["cut_points"]["L6"], [0.5, 1.5, 2.5, 3.5, 4.5])
        by_id = {row["audit_metadata"]["source_local_id"]: row for row in rows}
        drops = Counter()
        for i, mean in enumerate(means):
            guid = f"klue-sts-v1_train_{i:05d}"
            levels = ordinal.level_count("klue_sts", guid, (3, 4, 5, 6))
            cuts = report["cut_points"][f"L{levels}"]
            if ordinal.near_cut(mean, cuts, 0.2):
                drops[f"L{levels}"] += 1
                self.assertNotIn(guid, by_id)
                continue
            row = by_id[guid]
            self.assertEqual(len(row["options"]), levels)
            expected = (
                min(5, max(0, round(mean)))
                if levels == 6
                else ordinal.bin_index(mean, cuts)
            )
            self.assertEqual(row["label"], expected)
            self.assertEqual(
                row["audit_metadata"]["stratum"],
                f"{MARKER}-stratum-{'rtt' if i % 2 else 'sampled'}",
            )
        self.assertEqual(
            report["guard_band_drops"], {f"L{k}": drops[f"L{k}"] for k in (3, 4, 5, 6)}
        )
        self.assertGreater(sum(drops.values()), 0)
        anchored = next(row for row in rows if len(row["options"]) == 6)
        self.assertEqual(
            anchored["options"][5]["description"],
            "두 문장의 의미가 완전히 같다 (4.5–5점)",
        )
        quantile = next(row for row in rows if len(row["options"]) == 3)
        low, high = report["cut_points"]["L3"]
        self.assertTrue(
            quantile["options"][1]["description"].endswith(
                f"({ordinal.number(low)}–{ordinal.number(high)}점)"
            )
        )

    def test_degenerate_quantile_cut_empties_a_level(self) -> None:
        pairs = [
            klue.RatedPair(str(i), "가", "나", 0.0 if i < 60 else 1 + (i % 40) / 10)
            for i in range(100)
        ]
        rows, report = klue.sts_rows(
            pairs,
            family="t_sts",
            source="t_sts_train",
            language="ko",
            texts=klue.STS_TEXT,
            template="t",
        )
        self.assertEqual(report["cut_points"]["L3"][0], 0.0)
        _, cells = ordinal.balance(
            rows,
            cell=build.score_cell,
            level=build._label,
            levels=build._size,
            ident=build._id,
            seed="s",
        )
        self.assertEqual(cells["t_sts_train|L3"]["before"][0], 0)
        self.assertEqual(cells["t_sts_train|L3"]["after"], [0, 0, 0])


class JglueTest(FixtureCase):
    def test_jcommonsenseqa_options_and_duplicate_drop(self) -> None:
        rows, report = jglue.jcommonsenseqa(self.dirs["jglue"])
        self.assertEqual((len(rows), report["dropped"]["duplicate_choices"]), (4, 1))
        for row, (question, choices, label) in zip(rows, JCQA, strict=False):
            self.assertEqual(row["state"], question)
            self.assertEqual(gold(row), choices[label])
            self.assertEqual(
                sorted(o["description"] for o in row["options"]), sorted(choices)
            )
            self.assertEqual(row["language"], "ja")

    def test_jnli_string_labels_and_reversed_pairs(self) -> None:
        rows, report = jglue.jnli(self.dirs["jglue"])
        self.assertEqual(
            report["train_class_histogram"],
            {"entailment": 2, "neutral": 2, "contradiction": 2},
        )
        for row, (first, second, label) in zip(rows, JNLI, strict=True):
            self.assertEqual(row["state"], {"premise": first, "hypothesis": second})
            self.assertEqual(
                gold(row), jglue.JNLI_TEXT["options"][klue.NLI_LABELS.index(label)]
            )
        groups = [row["group_id"] for row in rows]
        expected = [
            f"a5:jglue_jnli_v1.3_train:{sha(image)[:20]}"
            for image in ("100124", "100124", "200001", "200001", "300001", "300001")
        ]
        self.assertEqual(groups, expected)
        self.assertEqual(
            (report["image_ids"], report["groups_after_reversed_pair_merge"]), (4, 3)
        )

    def test_jnli_rejects_unknown_label(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            write_jsonl(
                Path(tmp) / jglue.JNLI_FILE,
                [
                    {
                        "sentence_pair_id": "0",
                        "yjcaptions_id": "x",
                        "sentence1": "a",
                        "sentence2": "b",
                        "label": "unrelated",
                    }
                ],
            )
            with self.assertRaisesRegex(ValueError, "unknown JNLI label"):
                jglue.jnli(Path(tmp))

    def test_jsts_levels(self) -> None:
        rows, report = jglue.jsts(self.dirs["jglue"])
        labels = [jsts_label(i) for i in range(160)]
        self.assertEqual(report["cut_points"]["L4"], ordinal.quantile_cuts(labels, 4))
        for row in rows:
            mean, levels = (
                row["audit_metadata"]["mean_rating"],
                row["audit_metadata"]["levels"],
            )
            if levels == 6:
                self.assertEqual(row["label"], round(mean))
                self.assertLessEqual(abs(mean - round(mean)), 0.3 + 1e-9)
            self.assertTrue(
                row["options"][0]["description"].startswith("2つの文の意味")
            )
            self.assertTrue(
                row["state"].startswith("文1：") and "\n文2：" in row["state"]
            )
            image = jsts_image(int(row["audit_metadata"]["source_local_id"]))
            self.assertEqual(
                row["group_id"], f"a6h:jglue_jsts_v1.3_train:{sha(image)[:20]}"
            )
        self.assertEqual(report["image_ids"], len({jsts_image(i) for i in range(160)}))
        self.assertLess(len({row["group_id"] for row in rows}), len(rows))


class ArgqTest(FixtureCase):
    def test_topic_cuts_guard_band_and_mace_agreement(self) -> None:
        rows, report = argq.argq(self.dirs["argq"])
        self.assertEqual(
            (report["non_train_rows_skipped"], report["candidates"], report["topics"]),
            (2, 120, 2),
        )
        kept, guard, disagree = {}, Counter(), Counter()
        for t, topic in enumerate(ARGQ_TOPICS):
            values = [argq_values(t, i) for i in range(60)]
            for i, (wa, mace) in enumerate(values):
                local_id = f"row{t * 60 + i}"
                levels = ordinal.level_count("argq30k", local_id, (3, 4, 5))
                wa_cuts = ordinal.quantile_cuts([v[0] for v in values], levels)
                mace_cuts = ordinal.quantile_cuts([v[1] for v in values], levels)
                self.assertEqual(
                    report["cut_points"]["WA"][topic][f"L{levels}"], wa_cuts
                )
                if ordinal.near_cut(wa, wa_cuts, 0.02):
                    guard[f"L{levels}"] += 1
                elif ordinal.bin_index(mace, mace_cuts) != ordinal.bin_index(
                    wa, wa_cuts
                ):
                    disagree[f"L{levels}"] += 1
                else:
                    kept[local_id] = ordinal.bin_index(wa, wa_cuts)
        self.assertEqual(
            {row["audit_metadata"]["source_local_id"]: row["label"] for row in rows},
            kept,
        )
        self.assertEqual(
            report["guard_band_drops"], {f"L{k}": guard[f"L{k}"] for k in (3, 4, 5)}
        )
        self.assertEqual(
            report["mace_p_disagreement_drops"],
            {f"L{k}": disagree[f"L{k}"] for k in (3, 4, 5)},
        )
        self.assertGreater(sum(disagree.values()), 0)
        row = rows[0]
        self.assertRegex(row["state"], r"^Topic: .+\nArgument: .+$")
        self.assertTrue(
            all(
                "on the same topic" in option["description"]
                for option in row["options"]
            )
        )


class SafTest(FixtureCase):
    def test_verdict_mapping_and_state(self) -> None:
        for language, source, label in (
            ("en", "saf_en_train", "Student answer: "),
            ("de", "saf_de_train", "Antwort des Lernenden: "),
        ):
            rows, report = saf.saf(self.dirs[f"saf_{language}"], language)
            self.assertEqual(
                report["train_class_histogram"],
                {"Incorrect": 2, "Partially correct": 2, "Correct": 2},
            )
            for row in rows:
                self.assertEqual(
                    row["label"],
                    {"Incorrect": 0, "Partially correct": 1, "Correct": 2}[
                        row["audit_metadata"]["verdict"]
                    ],
                )
                self.assertEqual(
                    (row["source"], row["language"], len(row["options"])),
                    (source, language, 3),
                )
                self.assertIn(label, row["state"])
        self.assertTrue(rows[0]["options"][0]["description"].startswith("Falsch:"))

    def test_unknown_verdicts_fail_loudly(self) -> None:
        for verdict in ("Mostly correct", "correct", "Correct "):
            with tempfile.TemporaryDirectory() as tmp:
                make_saf(Path(tmp), "en", verdict_override=verdict)
                with self.assertRaisesRegex(
                    ValueError, "unknown verification_feedback"
                ):
                    saf.saf(Path(tmp), "en")


class BuilderTest(FixtureCase):
    def test_a5_caps_mrc_balance_and_manifest(self) -> None:
        rows, manifest = build.build_a5(self.dirs)
        mrc = [row for row in rows if row["family"] == "klue_mrc_answerable"]
        self.assertEqual(Counter(row["label"] for row in mrc), {0: 3, 1: 3})
        families = manifest["families"]
        self.assertEqual(
            families["klue_mrc_answerable"]["noul_balance"]["before"], [3, 6]
        )
        self.assertEqual(families["klue_ynat"]["cap_seed"], "a5-klue-ynat-v1")
        self.assertEqual(families["jglue_jnli"]["shortfall"], 1000 - 6)
        nli = families["klue_nli"]["class_balance"]
        self.assertEqual(
            (nli["before"], nli["after"], nli["per_class"]),
            (
                {"entailment": 4, "neutral": 4, "contradiction": 3},
                {"entailment": 3, "neutral": 3, "contradiction": 3},
                3,
            ),
        )
        for family, size in (("klue_nli", 3), ("jglue_jnli", 3), ("klue_ynat", 7)):
            counts = Counter(
                row["audit_metadata"]["original_label"]
                for row in rows
                if row["family"] == family
            )
            self.assertEqual(len(counts), size)
            self.assertEqual(len(set(counts.values())), 1)
        self.assertNotIn("class_balance", families["jglue_jcommonsenseqa"])
        self.assertEqual(families["klue_nli"]["rows"], 9)
        self.assertEqual(families["jglue_jnli"]["groups"], 3)
        self.assertEqual(
            manifest["inputs"]["klue_ynat_train"]["sha256"],
            file_sha256(self.dirs["klue"] / klue.YNAT_FILE),
        )
        self.assertIn("v2/data/sources/klue.py", manifest["modules"])
        self.assertEqual(sum(f["rows"] for f in families.values()), len(rows))

    def test_a6h_balance_rule_and_reports(self) -> None:
        rows, manifest = build.build_a6h(self.dirs)
        for source, entry in manifest["sources"].items():
            final = Counter(
                (build.score_cell(row), row["label"])
                for row in rows
                if row["source"] == source
            )
            for cell, info in entry["balance_cells"].items():
                self.assertEqual(info["limit"], min(info["before"]) * 6 // 5)
                self.assertEqual(
                    info["after"], [min(n, info["limit"]) for n in info["before"]]
                )
                for level in range(len(info["before"])):
                    self.assertLessEqual(final[(cell, level)], info["limit"])
            self.assertEqual(entry["rows"], sum(final.values()))
            self.assertEqual(entry["cap_seed"], f"a6h-{source}-v1")
        klue_sts = manifest["sources"]["klue_sts_train"]
        self.assertEqual(
            sorted(klue_sts["balance_cells"]),
            [f"klue_sts_train|L{k}" for k in (3, 4, 5, 6)],
        )
        strata = [f"{MARKER}-stratum-rtt", f"{MARKER}-stratum-sampled"]
        for when in ("before", "after"):
            table = klue_sts[f"strata_by_level_{when}_balance"]
            totals = klue_sts[f"levels_{when}_balance"]
            for size, levels in table.items():
                for level, by_stratum in levels.items():
                    self.assertEqual(sorted(by_stratum), strata)
                    self.assertEqual(sum(by_stratum.values()), totals[size][level])
        self.assertNotIn(
            "strata_by_level_before_balance",
            manifest["sources"]["jglue_jsts_v1.3_train"],
        )
        self.assertIn("cut_points", manifest["sources"]["jglue_jsts_v1.3_train"])
        self.assertIn("mace_p_disagreement_drops", manifest["sources"]["argq30k_train"])
        self.assertEqual(
            set(manifest["label_histograms"]),
            {"klue_sts", "jglue_jsts", "argq30k", "saf"},
        )

    def test_states_never_carry_labels_or_provenance(self) -> None:
        rows = build.build_a5(self.dirs)[0] + build.build_a6h(self.dirs)[0]
        self.assertTrue(any(MARKER in canonical(row["audit_metadata"]) for row in rows))
        for row in rows:
            self.assertNotIn(MARKER, inputs_text(row), row["id"])
            if row["family"] in ("klue_sts", "jglue_jsts"):
                self.assertNotIn(
                    str(row["audit_metadata"]["mean_rating"]), row["state"]
                )
            if row["family"] == "saf":
                self.assertEqual(row["state"].count("\n"), 2)

    def test_duplicate_source_ids_fail_loudly(self) -> None:
        rows, report = klue.ynat(self.dirs["klue"])
        with self.assertRaisesRegex(ValueError, "duplicate source ids"):
            build.load(
                lambda root: (rows + rows[:1], report),
                self.dirs["klue"],
                "klue_ynat_train",
            )

    def test_rows_valid_and_deterministic(self) -> None:
        for builder in (build.build_a5, build.build_a6h):
            first, second = builder(self.dirs), builder(self.dirs)
            self.assertEqual(canonical(first), canonical(second))
            ids = [row["id"] for row in first[0]]
            self.assertEqual(len(ids), len(set(ids)))
            for row in first[0]:
                validate_row(row, "train")
                if row["task_type"] == "score":
                    self.assertEqual(
                        [o["key"] for o in row["options"]],
                        [str(k) for k in range(len(row["options"]))],
                    )

    def test_cli_writes_arms_and_refuses_overwrite(self) -> None:
        d = {name: str(path) for name, path in self.dirs.items()}
        with tempfile.TemporaryDirectory() as out:
            commands = {
                "a5": [
                    "--arm",
                    "a5",
                    "--klue",
                    d["klue"],
                    "--jglue",
                    d["jglue"],
                    "--out-dir",
                    out,
                ],
                "a6h": [
                    "--arm",
                    "a6h",
                    "--klue",
                    d["klue"],
                    "--jglue",
                    d["jglue"],
                    "--argq",
                    d["argq"],
                    "--saf-en",
                    d["saf_en"],
                    "--saf-de",
                    d["saf_de"],
                    "--out-dir",
                    out,
                ],
            }
            for arm, argv in commands.items():
                with contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(build.main(argv), 0)
                written = json.loads(
                    (Path(out) / f"{arm}.build.json").read_text(encoding="utf-8")
                )
                self.assertEqual(written["build"]["arm"], arm)
                aho = [
                    json.loads(line)
                    for line in (Path(out) / f"{arm}.aho.jsonl")
                    .read_text(encoding="utf-8")
                    .splitlines()
                ]
                self.assertTrue(all(row["split"] == "select" for row in aho))
                with (
                    contextlib.redirect_stderr(io.StringIO()),
                    self.assertRaises(SystemExit),
                ):
                    build.main(argv)
            with (
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                build.main(
                    [
                        "--arm",
                        "a6h",
                        "--klue",
                        d["klue"],
                        "--jglue",
                        d["jglue"],
                        "--out-dir",
                        out + "/x",
                    ]
                )


if __name__ == "__main__":
    unittest.main()
