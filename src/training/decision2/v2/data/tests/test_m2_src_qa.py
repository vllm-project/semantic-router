from __future__ import annotations

import collections
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from training.model.data import canonical, validate_row
from v2.data.m2 import build, spec, src_qa
from v2.data.m2.common import group_id, sha
from v2.data.m2.text import mentions
from v2.data.textnorm import normalize

PACKAGE_ROOT = Path(__file__).resolve().parents[3]
BEFORE_CONSTRUCTION = frozenset(
    {
        "unanswerable",
        "cannotanswer",
        "empty_text",
        "v1_item",
        "v1_passage",
        "duplicate_question",
        "duplicate_id",
    }
)

HARBOR = (
    "Pellmark is a small harbor town on the northern coast of the Veld Sea. "
    "The town was founded in 1742 by families of herring fishers. "
    "Its lighthouse, painted red and white, stands on a basalt cliff above the port. "
    "A narrow bridge links the old quarter to the island of Tessin. "
    "Every August the town holds a regatta for wooden sailing boats. "
    "Most residents today work in tourism or at the ferry terminal."
)
HAMLET = "Borl is a hamlet. It has one road. Nobody lives there in winter."
ISLAND = (
    "Tessin is an island. Tessin has a harbor. Ferries reach Tessin daily. "
    "The bridge to Tessin is old."
)
QUAC_LIFE = (
    "Orla Venn was born in the village of Brenholt in 1901. Her father kept bees and "
    "sold honey at the market. She learned to read from her older brother. At "
    "sixteen she moved to the city of Marrow to work in a bakery. She later "
    "described those years as the happiest of her life. CANNOTANSWER"
)
QUAC_CAREER = (
    "Orla Venn opened a small press in 1931. The press printed pamphlets about "
    "farming. Its first book was a guide to beekeeping. The press closed during "
    "the war. She never reopened it. CANNOTANSWER"
)
JA_TOWN = (
    "架空町は北の海岸にある小さな町である。町は1742年に漁師の家族によって作られた。"
    "灯台は赤と白に塗られている。細い橋が旧市街と島を結んでいる。毎年八月に帆船の競技会が開かれる。"
)
KO_HARBOR = (
    "해안 마을 벨마크는 북쪽 바다에 있다. 마을은 1742년에 어부 가족들이 세웠다. "
    "등대는 빨간색과 흰색으로 칠해져 있다. 좁은 다리가 구시가지와 섬을 잇는다. "
    "매년 팔월에 돛단배 경주가 열린다."
)
KO_VALLEY = (
    "산골 마을 토렌은 높은 고개 아래에 있다. 마을 사람들은 겨울마다 치즈를 만든다. "
    "오래된 물레방아가 개울 옆에 서 있다. 봄에는 산나물 장터가 열린다. "
    "마을 학교에는 학생이 열두 명 있다."
)
KO_PORT = (
    "항구 도시 카렌은 1905년에 세워졌다. 도시의 인구는 1905명이다. "
    "도시에는 큰 시장이 있다. 시장은 매주 월요일에 열린다. 항구에는 배가 많다."
)
ZH_CITY = "甲城位于北方海岸。城中有一座红白相间的灯塔。一座窄桥连接老城和小岛。每年八月举行帆船比赛。城里有一座古老的钟楼。"
ZH_TOWN = "乙镇在2019年建成。乙镇在2019年春天举行庆典。镇上有一座石桥。镇外有一片竹林。镇里有一所小学。"
ZHT_CITY = "丙城位於南方海岸。丙城的港口建於清朝。港口旁有一座紅色燈塔。舊城區有許多石板路。每年秋天舉行龍舟比賽。"
DE_CONTEXT = (
    "Beispiel_Stadt\n\n=== Geschichte ===\nBeispiel Stadt liegt an einem breiten "
    "Fluss. Die Stadt wurde im Jahr 1742 gegründet. Eine alte Brücke verbindet "  # codespell:ignore alte
    "die beiden Ufer. Jeden August findet ein Bootsrennen statt. Die meisten "
    "Einwohner arbeiten im Hafen."
)
FR_CONTEXT = (
    "Borvanne est une ville située au pied d'une colline. Elle a été fondée en "
    "1742 par des pêcheurs. Un vieux phare blanc domine le port. Un bac étroit "
    "traverse le lac jusqu'à l'île de Morn. Chaque été, une course de bateaux a "
    "lieu sur le lac."
)
ES_VILLAGE = (
    "Villa Rema es un pueblo situado junto a un lago. Fue fundado en 1742 por "
    "pescadores. Un faro rojo domina el puerto. Un puente estrecho une el casco "
    "antiguo con la isla de Morn. Cada verano se celebra una regata en el lago."
)
ES_MATCH = (
    "El equipo local ganó hoy la final del torneo regional. El partido terminó "
    "con un marcador de tres a uno. Miles de aficionados celebraron en la plaza. "
    "El entrenador agradeció el apoyo de la ciudad. La próxima temporada empieza "
    "en septiembre."
)
EN_DAM = (
    "The river Olm crosses the plain of Brask. Farmers there grow barley and flax. "
    "The plain floods almost every spring. A stone dam was built in 1874 near the "
    "village of Tarn. The dam holds back the spring water. Fishing is allowed "
    "above the dam."
)
EN_MILL = (
    "The Brask mill stands on the east bank. It was first recorded in 1650. "
    "The mill ground barley for three villages. Its wheel was replaced twice. "
    "Today it houses a small museum. Visitors can climb to the loft."
)


def write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def hf(ident: Any, context: str, question: str, answer: str | None, **extra: Any):
    answers = (
        {"text": [answer], "answer_start": [context.index(answer)]}
        if answer is not None
        else {"text": [], "answer_start": []}
    )
    return {
        "id": ident,
        "context": context,
        "question": question,
        "answers": answers,
        **extra,
    }


def squad_qa(ident: str, context: str, question: str, answer: str, offset: int = 0):
    return {
        "id": ident,
        "question": question,
        "answers": [{"text": answer, "answer_start": context.index(answer) + offset}],
    }


def routed(template: str, route: int) -> str:
    for number in range(1000):
        question = template.format(number)
        if int(sha("tydi-alloc:" + normalize(question)), 16) % 2 == route:
            return question
    raise AssertionError("no question with the requested route")


def ja_sentences(topic: str, count: int) -> str:
    return "".join(f"{topic}の第{n}記録には架空の港の話がある。" for n in range(count))


def tydi(
    language: str,
    question: str,
    title: str,
    paragraphs: Sequence[str],
    annotations: Sequence[tuple[int, str | None, str]],
    url: str | None = None,
) -> dict[str, Any]:
    document = "\n\n".join(paragraphs)
    starts, ends, position = [], [], 0
    for paragraph in paragraphs:
        size = len(paragraph.encode("utf-8"))
        starts.append(position)
        ends.append(position + size)
        position += size + 2
    columns: dict[str, list[Any]] = collections.defaultdict(list)
    for index, answer, kind in annotations:
        low = high = -1
        if answer is not None:
            offset = paragraphs[index].index(answer)
            low = starts[index] + len(paragraphs[index][:offset].encode("utf-8"))
            high = low + len(answer.encode("utf-8"))
        columns["passage_answer_candidate_index"].append(index)
        columns["minimal_answers_start_byte"].append(low)
        columns["minimal_answers_end_byte"].append(high)
        columns["yes_no_answer"].append(kind)
    return {
        "annotations": dict(columns),
        "document_plaintext": document,
        "document_title": title,
        "document_url": url or f"https://example.invalid/{sha(question)[:12]}",
        "language": language,
        "passage_answer_candidates": {
            "plaintext_start_byte": starts,
            "plaintext_end_byte": ends,
        },
        "question_text": question,
    }


def make_fixture(root: Path) -> dict[str, Any]:
    """Synthetic sources in the pinned layout; returns TyDi fixture details."""
    dirs = spec.resolve(root)
    write_jsonl(
        dirs["squad2"] / "squad_v2/train-00000-of-00001.jsonl",
        [
            hf(
                "s1",
                HARBOR,
                "When was Pellmark founded?",
                "1742",
                title="Pellmark_Harbor",
            ),
            hf(
                "s2",
                HARBOR,
                "What links the old quarter to Tessin?",
                "A narrow bridge",
                title="Pellmark_Harbor",
            ),
            hf(
                "s3",
                HARBOR,
                "When is the regatta held?",
                "Every August",
                title="Pellmark_Harbor",
            ),
            hf(
                "s4",
                HARBOR,
                "What is the town called?",
                "Pellmark",
                title="Pellmark_Harbor",
            ),
            hf(
                "s5",
                HARBOR,
                "Who painted the lighthouse?",
                None,
                title="Pellmark_Harbor",
            ),
            hf(
                "s6",
                HARBOR,
                "When was  Pellmark founded?",
                "1742",
                title="Pellmark_Harbor",
            ),
            hf(
                "s7", HAMLET, "How many roads does Borl have?", "one road", title="Borl"
            ),
            hf("s8", ISLAND, "Which island is it?", "Tessin", title="Tessin"),
        ],
    )
    write_json(
        dirs["quac"] / "train_v0.2.json",
        {
            "data": [
                {
                    "title": "Orla Venn",
                    "section_title": "Early life",
                    "background": "Orla Venn was a printer.",
                    "paragraphs": [
                        {
                            "id": "C_1_0",
                            "context": QUAC_LIFE,
                            "qas": [
                                quac_qa(
                                    "C_1_0_q#0",
                                    QUAC_LIFE,
                                    "Where was Orla Venn born?",
                                    "in the village of Brenholt",
                                ),
                                quac_qa(
                                    "C_1_0_q#1",
                                    QUAC_LIFE,
                                    "What did her father sell?",
                                    "honey",
                                ),
                            ],
                        }
                    ],
                },
                {
                    "title": "Orla Venn",
                    "section_title": "Career",
                    "background": "Orla Venn was a printer.",
                    "paragraphs": [
                        {
                            "id": "C_1_1",
                            "context": QUAC_CAREER,
                            "qas": [
                                quac_qa(
                                    "C_1_1_q#0",
                                    QUAC_CAREER,
                                    "Did she win awards?",
                                    "CANNOTANSWER",
                                ),
                                quac_qa(
                                    "C_1_1_q#1",
                                    QUAC_CAREER,
                                    "What was its first book?",
                                    "a guide to beekeeping",
                                ),
                            ],
                        }
                    ],
                },
            ]
        },
    )
    prefix = "架空町 [SEP] "
    ja_context = prefix + JA_TOWN
    write_json(
        dirs["jglue"] / "datasets/jsquad-v1.3/train-v1.3.json",
        {
            "data": [
                {
                    "title": "架空町",
                    "paragraphs": [
                        {
                            "context": ja_context,
                            "qas": [
                                {
                                    **squad_qa(
                                        "j1",
                                        ja_context,
                                        "町はいつ作られたか？",
                                        "1742年",
                                    ),
                                    "is_impossible": False,
                                },
                                {
                                    "id": "j2",
                                    "question": "町の名前は？",
                                    "answers": [{"text": "架空町", "answer_start": 0}],
                                    "is_impossible": False,
                                },
                                {
                                    **squad_qa(
                                        "j3", ja_context, "灯台は何色か？", "赤と白"
                                    ),
                                    "is_impossible": False,
                                },
                                {
                                    "id": "j4",
                                    "question": "町の人口は？",
                                    "answers": [],
                                    "is_impossible": True,
                                },
                            ],
                        }
                    ],
                }
            ]
        },
    )
    klue = [
        ("k1", "벨마크 소식", KO_HARBOR, "벨마크는 언제 세워졌나?", "1742년", False),
        ("k2", "벨마크 소식", KO_HARBOR, "등대는 누가 칠했나?", "어부", True),
        (
            "k3",
            "벨마크 소식",
            KO_HARBOR,
            "무엇이 구시가지와 섬을 잇나?",
            "좁은 다리",
            False,
        ),
        (
            "k4",
            "토렌 소식",
            KO_VALLEY,
            "토렌 사람들은 겨울마다 무엇을 만드나?",
            "치즈",
            False,
        ),
        (
            "k5",
            "토렌 소식",
            KO_VALLEY,
            "개울 옆에 무엇이 서 있나?",
            "오래된 물레방아",
            False,
        ),
        ("k6", "카렌 소식", KO_PORT, "카렌은 언제 세워졌나?", "1905", False),
    ]
    write_jsonl(
        dirs["klue"] / "mrc/train.jsonl",
        [
            {
                "guid": guid,
                "title": title,
                "context": context,
                "question": question,
                "answers": {"text": [answer], "answer_start": [context.index(answer)]},
                "is_impossible": impossible,
                "news_category": "종합",
                "source": "synthetic",
                "question_type": 1,
            }
            for guid, title, context, question, answer, impossible in klue
        ],
    )
    write_jsonl(
        dirs["cmrc2018"] / "data/train-00000-of-00001.jsonl",
        [
            hf("TRAIN_1_QUERY_0", ZH_CITY, "灯塔是什么颜色？", "红白相间"),
            hf("TRAIN_1_QUERY_1", ZH_CITY, "什么连接老城和小岛？", "一座窄桥"),
            hf("TRAIN_2_QUERY_0", ZH_TOWN, "乙镇哪一年建成？", "2019"),
        ],
    )
    write_json(
        dirs["drcd"] / "DRCD_training.json",
        {
            "version": "1.3",
            "data": [
                {
                    "title": "测试城",
                    "id": "9001",
                    "paragraphs": [
                        {
                            "context": ZHT_CITY,
                            "id": "9001-1",
                            "qas": [
                                squad_qa(
                                    "9001-1-1", ZHT_CITY, "丙城的港口建於何時？", "清朝"
                                ),
                                squad_qa(
                                    "9001-1-2", ZHT_CITY, "港口旁有什麼？", "紅色燈塔"
                                ),
                            ],
                        }
                    ],
                }
            ],
        },
    )
    write_jsonl(
        dirs["germanquad"] / "plain_text/train/0000.jsonl",
        [
            hf(101, DE_CONTEXT, "Wann wurde die Stadt gegründet?", "1742"),
            hf(102, DE_CONTEXT, "Wie heißt die Stadt?", "Beispiel Stadt"),
            hf(
                103, DE_CONTEXT, "Was verbindet die beiden Ufer?", "Eine alte Brücke"
            ),  # codespell:ignore alte
        ],
    )
    write_jsonl(
        dirs["piaf"] / "plain_text/train-00000-of-00001.jsonl",
        [
            hf(
                "p1",
                FR_CONTEXT,
                "Quand Borvanne a-t-elle été fondée ?",
                "1742",
                title="Ville imaginaire",
            ),
            hf(
                "p2",
                FR_CONTEXT,
                "Qu'est-ce qui traverse le lac ?",
                "Un bac étroit",
                title="Ville imaginaire",
            ),
        ],
    )
    write_json(
        dirs["sqac"] / "train.json",
        {
            "data": [
                {
                    "title": "Villa inventada",
                    "source": "wikipedia",
                    "paragraphs": [
                        {
                            "context": ES_VILLAGE,
                            "qas": [
                                squad_qa(
                                    "sq1",
                                    ES_VILLAGE,
                                    "¿Cuándo fue fundada Villa Rema?",
                                    "1742",
                                )
                            ],
                        }
                    ],
                },
                {
                    "title": "CESS-CAST-A_00001_20000101_rec.txt",
                    "source": "ancora",
                    "paragraphs": [
                        {
                            "context": ES_MATCH,
                            "qas": [
                                squad_qa(
                                    "sq2",
                                    ES_MATCH,
                                    "¿Cómo terminó el partido?",
                                    "tres a uno",
                                )
                            ],
                        }
                    ],
                },
            ]
        },
    )
    return tydi_fixture(dirs["tydiqa"])


def quac_qa(ident: str, context: str, question: str, answer: str) -> dict[str, Any]:
    span = {"text": answer, "answer_start": context.index(answer)}
    return {
        "id": ident,
        "question": question,
        "orig_answer": span,
        "answers": [span],
        "yesno": "x",
        "followup": "y",
    }


def tydi_fixture(root: Path) -> dict[str, Any]:
    removal_ja = routed("北浜港{}はどこにある？", 0)
    minimal_ja = routed("乙港{}はいつ開かれた？", 1)
    removal_en = routed("When was the Olm dam number {} built?", 0)
    relevance_en = routed("When was the Brask mill number {} built?", 1)
    gold_ja = " 北浜港は島の東側にある。港には小さな灯台が立っている。"
    window = [
        ja_sentences("東岬", 2),
        ja_sentences("西岬", 2),
        gold_ja,
        ja_sentences("南岬", 2),
        ja_sentences("中岬", 2),
    ]
    passage_only = tydi(
        "japanese",
        "庚港には灯台があるか？",
        "庚港",
        [
            ja_sentences("戊", 12),
            ja_sentences("己", 13),
            ja_sentences("庚", 11),
            "短い段落。",
            ja_sentences("辛", 100),
            ja_sentences("壬", 12),
            ja_sentences("癸", 12),
        ],
        [(2, None, "YES"), (5, None, "NONE")],
    )
    write_jsonl(
        root / "primary_task/train-00000-of-00002.jsonl",
        [
            tydi("japanese", removal_ja, "北浜港", window, [(2, "島の東側", "NONE")]),
            tydi(
                "japanese",
                minimal_ja,
                "乙港",
                [
                    ja_sentences("甲", 12),
                    "乙港は千九百年に開かれた。" + ja_sentences("乙", 11),
                    ja_sentences("丙", 12),
                    ja_sentences("丁", 12),
                ],
                [(1, "千九百年", "NONE")],
            ),
            tydi(
                "thai",
                "เมืองสมมติอยู่ที่ไหน",
                "เมืองสมมติ",
                ["เมืองสมมติ มีท่าเรือเล็ก และ ทะเลสาบใหญ่"],
                [(-1, None, "NONE")],
            ),
            tydi(
                "japanese",
                "丑港の人口は？",
                "丑港",
                [ja_sentences("丑", 12)],
                [(-1, None, "NONE")],
            ),
            tydi(
                "english", removal_en, "Olm (river)", [EN_DAM], [(0, "in 1874", "NONE")]
            ),
        ],
    )
    write_jsonl(
        root / "primary_task/train-00001-of-00002.jsonl",
        [
            passage_only,
            tydi(
                "japanese",
                "子港は古いか？",
                "子港",
                [
                    ja_sentences("子", 12),
                    ja_sentences("丑", 12),
                    ja_sentences("寅", 12),
                    "短い。",
                ],
                [(0, None, "NO")],
            ),
            tydi(
                "japanese",
                "卯港は大きいか？",
                "卯港",
                [
                    ja_sentences("卯", 100),
                    ja_sentences("辰", 12),
                    ja_sentences("巳", 12),
                    ja_sentences("午", 12),
                ],
                [(0, None, "YES")],
            ),
            passage_only,
            tydi(
                "english",
                relevance_en,
                "Brask mill",
                [EN_MILL],
                [(0, "in 1650", "NONE")],
            ),
        ],
    )
    return {
        "removal_ja": removal_ja,
        "minimal_ja": minimal_ja,
        "passage_only_ja": passage_only["question_text"],
        "gold_ja": gold_ja.strip(),
        "window_ja": window,
        "removal_en": removal_en,
        "relevance_en": relevance_en,
    }


def family(name: str) -> spec.FamilySpec:
    return next(item for item in src_qa.FAMILIES if item.family == name)


def build_all(dirs: dict[str, Path]) -> dict[str, tuple[list, dict]]:
    return {item.family: item.build(dirs) for item in src_qa.FAMILIES}


def digest(built: dict[str, tuple[list, dict]]) -> str:
    return hashlib.sha256(canonical(built).encode("utf-8")).hexdigest()


class FixtureCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.details = make_fixture(self.root)
        self.dirs = spec.resolve(self.root)
        src_qa._TYDI_CACHE.clear()
        self.addCleanup(src_qa._TYDI_CACHE.clear)

    def built(self, name: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        return family(name).build(self.dirs)


class RegistryTest(unittest.TestCase):
    def test_families(self) -> None:
        names = [item.family for item in src_qa.FAMILIES]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(len(names), 9 + 11 + 20)
        c1 = {item.family: item for item in src_qa.REMOVAL_SOURCES}
        self.assertEqual(
            {
                name: (item.arm, item.source, item.cap_rows, item.seed)
                for name, item in c1.items()
            },
            {
                "squad2_removal": ("e11", "squad2_train", 16000, "e11-squad2-v1"),
                "quac_removal": ("e11", "quac_train_v0.2", 10000, "e11-quac-v1"),
                "jsquad_removal": ("h5", "jsquad_v1.3_train", 6000, "h5-jsquad-v1"),
                "klue_mrc_removal": ("h5", "klue_mrc_train", 6000, "h5-klue-mrc-v1"),
                "cmrc2018_removal": ("h5", "cmrc2018_train", 6000, "h5-cmrc2018-v1"),
                "drcd_removal": ("h5", "drcd_train", 6000, "h5-drcd-v1"),
                "germanquad_removal": (
                    "h5",
                    "germanquad_train",
                    6000,
                    "h5-germanquad-v1",
                ),
                "piaf_removal": ("h5", "piaf_train", 6000, "h5-piaf-v1"),
                "sqac_removal": ("h5", "sqac_train", 6000, "h5-sqac-v1"),
            },
        )
        for item in src_qa.FAMILIES:
            self.assertIn(item.arm, spec.ARMS)
            if item.family.startswith("tydi_"):
                self.assertEqual(item.source, "tydiqa_primary_train")
        removal = {
            i.family: i for i in src_qa.FAMILIES if i.family.startswith("tydi_removal_")
        }
        self.assertEqual(len(removal), 11)
        self.assertEqual(
            (removal["tydi_removal_en"].arm, removal["tydi_removal_en"].seed),
            ("e11", "e11-tydi-removal-en-v1"),
        )
        self.assertEqual(
            (removal["tydi_removal_th"].arm, removal["tydi_removal_th"].cap_rows),
            ("h5", 4000),
        )
        self.assertEqual(removal["tydi_removal_th"].seed, "h5-tydi-removal-th-v1")
        for code in src_qa.TYDI_LANGUAGES:
            choice = [n for n in names if n == f"tydi_relevance_choice_{code}"]
            noul = [n for n in names if n == f"tydi_relevance_noul_{code}"]
            self.assertEqual(len(choice) + len(noul), 0 if code == "en" else 2)
        self.assertEqual(family("tydi_relevance_choice_ar").cap_rows, 1500)
        self.assertEqual(family("tydi_relevance_noul_ar").cap_rows, 3000)


class StatesTest(unittest.TestCase):
    def test_containment_and_underscores(self) -> None:
        self.assertFalse(mentions("ที่กรุงเทพมหานครมีคน", "กรุงเทพ"))
        self.assertTrue(src_qa.states("ที่กรุงเทพมหานครมีคน", "กรุงเทพ", "th"))
        self.assertFalse(mentions("1963年に作られた", "1963"))
        self.assertTrue(src_qa.states("1963年に作られた", "1963", "ja"))
        self.assertTrue(src_qa.states("인구는 1905명이다", "1905", "ko"))
        self.assertTrue(
            src_qa.states("Recht_der_Vereinigten_Staaten", "Vereinigten Staaten", "de")
        )
        self.assertFalse(src_qa.states("The start of it", "art", "en"))
        self.assertTrue(src_qa.states("The Art of it", "art", "en"))


class RemovalTest(FixtureCase):
    def assert_twins(self, rows: list[dict[str, Any]]) -> None:
        self.assertTrue(rows)
        labels = collections.Counter(row["label"] for row in rows)
        self.assertEqual(labels[0], labels[1])
        pairs: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
        for row in rows:
            validate_row(row, "train")
            self.assertEqual(row["task_type"], "noul")
            self.assertEqual(row["render_template"], "m2/removal_twins/v1")
            base, twin = row["audit_metadata"]["source_local_id"].rsplit(":", 1)
            self.assertEqual(twin, {1: "complete", 0: "removed"}[row["label"]])
            pairs[base].append(row)
        for twins in pairs.values():
            self.assertEqual(sorted(row["label"] for row in twins), [0, 1])
            self.assertEqual(len({row["group_id"] for row in twins}), 1)
            self.assertEqual(len({row["instructions"] for row in twins}), 1)
            units = {row["audit_metadata"]["units_removed"] for row in twins}
            self.assertEqual(len(units), 1)

    def test_every_c1_family_builds_twins(self) -> None:
        expected = {
            "squad2_removal": 3,
            "quac_removal": 1,
            "jsquad_removal": 2,
            "klue_mrc_removal": 4,
            "cmrc2018_removal": 2,
            "drcd_removal": 2,
            "germanquad_removal": 2,
            "piaf_removal": 2,
            "sqac_removal": 2,
        }
        for item in src_qa.REMOVAL_SOURCES:
            with self.subTest(item.family):
                rows, report = self.built(item.family)
                self.assert_twins(rows)
                self.assertEqual(report["pairs"], expected[item.family])
                self.assertEqual(len(rows), 2 * expected[item.family])
                self.assertEqual({row["language"] for row in rows}, {item.language})
                self.assertEqual({row["source"] for row in rows}, {item.source})
                drops = report["drops"]
                before = sum(n for r, n in drops.items() if r in BEFORE_CONSTRUCTION)
                after = sum(n for r, n in drops.items() if r not in BEFORE_CONSTRUCTION)
                self.assertEqual(report["records"], report["candidates"] + before)
                self.assertEqual(report["candidates"], report["pairs"] + after)
                root = self.dirs[item.directory]
                self.assertEqual(
                    sorted(report["inputs"]),
                    sorted(
                        p.relative_to(root).as_posix() for p in root.glob(item.pattern)
                    ),
                )

    def test_squad2_drops_groups_and_titles(self) -> None:
        rows, report = self.built("squad2_removal")
        self.assertEqual(
            report["drops"],
            {
                "answer_in_title": 1,
                "answer_too_widespread": 1,
                "duplicate_question": 1,
                "too_few_units": 1,
                "unanswerable": 1,
            },
        )
        self.assertEqual(
            {row["group_id"] for row in rows}, {group_id("squad", normalize(HARBOR))}
        )
        ids = {row["audit_metadata"]["source_local_id"].split(":")[0] for row in rows}
        self.assertEqual(ids, {"s1", "s2", "s3"})
        self.assertTrue(
            all(row["state"].startswith("Pellmark Harbor\n\n") for row in rows)
        )
        removed = {
            row["audit_metadata"]["source_local_id"]: row["state"]
            for row in rows
            if row["label"] == 0
        }
        self.assertNotIn("1742", removed["s1:removed"])

    def test_quac_uses_first_answerable_question_only(self) -> None:
        rows, report = self.built("quac_removal")
        self.assertEqual(report["drops"], {"cannotanswer": 1})
        self.assertEqual(report["records"], 2)
        self.assertEqual(
            {row["audit_metadata"]["source_local_id"] for row in rows},
            {"C_1_0_q#0:complete", "C_1_0_q#0:removed"},
        )
        for row in rows:
            self.assertNotIn("CANNOTANSWER", row["state"])
            self.assertTrue(row["state"].startswith("Orla Venn — Early life\n\n"))
            self.assertIn("Where was Orla Venn born?", row["instructions"])
        self.assertEqual(
            {row["group_id"] for row in rows},
            {group_id("quac", normalize(QUAC_LIFE[: -len("CANNOTANSWER")]))},
        )

    def test_jsquad_title_prefix(self) -> None:
        rows, report = self.built("jsquad_removal")
        self.assertEqual(report["drops"], {"answer_in_title": 1, "unanswerable": 1})
        for row in rows:
            self.assertNotIn("[SEP]", row["state"])
            self.assertTrue(row["state"].startswith("架空町\n\n"))
        self.assertEqual(
            {row["group_id"] for row in rows}, {group_id("jsquad", normalize(JA_TOWN))}
        )

    def test_germanquad_head_stays_in_both_twins(self) -> None:
        rows, report = self.built("germanquad_removal")
        self.assertEqual(report["drops"], {"answer_in_title": 1})
        head = "Beispiel_Stadt === Geschichte ===\n\n"
        self.assertTrue(all(row["state"].startswith(head) for row in rows))
        self.assertEqual(
            {row["audit_metadata"]["source_local_id"].split(":")[0] for row in rows},
            {"101", "103"},
        )
        body = DE_CONTEXT[DE_CONTEXT.index("Beispiel Stadt liegt") :]
        self.assertEqual(
            {row["group_id"] for row in rows}, {group_id("germanquad", normalize(body))}
        )

    def test_glued_answers_are_caught(self) -> None:
        rows, report = self.built("cmrc2018_removal")
        self.assertEqual(report["drops"], {"answer_left_after_removal": 1})
        self.assertEqual(
            {row["group_id"] for row in rows},
            {group_id("cmrc2018", normalize(ZH_CITY))},
        )
        rows, report = self.built("klue_mrc_removal")
        self.assertEqual(
            report["drops"], {"answer_left_after_removal": 1, "unanswerable": 1}
        )
        states = {
            row["audit_metadata"]["source_local_id"]: row["state"] for row in rows
        }
        self.assertTrue(src_qa.states(states["k1:complete"], "1742년", "ko"))
        self.assertFalse(src_qa.states(states["k1:removed"], "1742년", "ko"))
        self.assertNotIn("k6:removed", states)

    def test_titles_dropped_for_drcd_and_sqac_file_names(self) -> None:
        rows, _ = self.built("drcd_removal")
        self.assertEqual(len(rows), 4)
        self.assertTrue(all("测试城" not in row["state"] for row in rows))
        rows, _ = self.built("sqac_removal")
        states = {
            row["audit_metadata"]["source_local_id"]: row["state"] for row in rows
        }
        self.assertTrue(states["sq1:complete"].startswith("Villa inventada\n\n"))
        self.assertTrue(states["sq2:removed"].startswith("El equipo"))
        self.assertTrue(all(".txt" not in state for state in states.values()))

    def test_klue_v1_items_and_passages_excluded(self) -> None:
        v1_rows = self.root / "v1-a5.train.jsonl"
        write_jsonl(
            v1_rows,
            [
                {
                    "source": "klue_mrc_train",
                    "audit_metadata": {"source_local_id": "k4"},
                },
                {
                    "source": "klue_nli_train",
                    "audit_metadata": {"source_local_id": "k1"},
                },
            ],
        )
        manifest = self.root / "v1-rows.json"
        manifest.write_text(json.dumps([str(v1_rows)]), encoding="utf-8")
        rows, report = family("klue_mrc_removal").build(
            {**self.dirs, build.V1_ROWS_KEY: manifest}
        )
        self.assertEqual(report["v1_excluded"], {"items": 1, "passages": 1})
        self.assertEqual(report["drops"]["v1_item"], 1)
        self.assertEqual(report["drops"]["v1_passage"], 1)
        self.assertEqual(
            {row["audit_metadata"]["source_local_id"].split(":")[0] for row in rows},
            {"k1", "k3"},
        )
        _, plain = self.built("klue_mrc_removal")
        self.assertEqual(plain["v1_excluded"], {"items": 0, "passages": 0})


class TydiTest(FixtureCase):
    def local(self, rows: list[dict[str, Any]]) -> set[str]:
        return {
            row["audit_metadata"]["source_local_id"].rsplit(":", 1)[0] for row in rows
        }

    def test_allocation_and_drops(self) -> None:
        removal, report = self.built("tydi_removal_ja")
        choice, choice_report = self.built("tydi_relevance_choice_ja")
        noul, noul_report = self.built("tydi_relevance_noul_ja")
        self.assertEqual(report["records"], 7)
        self.assertEqual(
            report["allocation"],
            {
                "removal": 1,
                "relevance": 4,
                "dropped": {"duplicate_example": 1, "no_passage_answer": 1},
            },
        )
        self.assertEqual(
            report["drops"], {"duplicate_example": 1, "no_passage_answer": 1}
        )
        self.assertEqual(
            choice_report["drops"],
            {
                "duplicate_example": 1,
                "gold_passage_too_long": 1,
                "no_passage_answer": 1,
                "too_few_negatives": 1,
            },
        )
        self.assertEqual(choice_report["drops"], noul_report["drops"])
        self.assertEqual(self.local(removal), {"train-00000-of-00002:1"})
        self.assertEqual(
            self.local(choice), {"train-00000-of-00002:2", "train-00001-of-00002:1"}
        )
        self.assertEqual(self.local(noul), self.local(choice))
        self.assertFalse(self.local(removal) & self.local(choice))
        for rows in (removal, noul):
            self.assertEqual(
                collections.Counter(r["label"] for r in rows),
                {0: len(rows) // 2, 1: len(rows) // 2},
            )
        questions = {
            "train-00000-of-00002:1": self.details["removal_ja"],
            "train-00000-of-00002:2": self.details["minimal_ja"],
            "train-00001-of-00002:1": self.details["passage_only_ja"],
        }
        for row in removal + choice + noul:
            validate_row(row, "train")
            self.assertEqual(row["language"], "ja")
            base = row["audit_metadata"]["source_local_id"].rsplit(":", 1)[0]
            self.assertEqual(
                row["group_id"], group_id("tydi-miracl", normalize(questions[base]))
            )
        thai, thai_report = self.built("tydi_removal_th")
        self.assertEqual((thai, thai_report["records"]), ([], 1))
        self.assertEqual(thai_report["drops"], {"no_passage_answer": 1})

    def test_removal_context_window(self) -> None:
        rows, _ = self.built("tydi_removal_ja")
        texts = [text.strip() for text in self.details["window_ja"]]
        self.assertEqual(src_qa.tydi_window(texts, 2, "ja"), [1, 2, 3])
        complete = next(row for row in rows if row["label"] == 1)
        removed = next(row for row in rows if row["label"] == 0)
        self.assertTrue(complete["state"].startswith("北浜港\n\n"))
        self.assertIn("島の東側", complete["state"])
        self.assertNotIn("島の東側", removed["state"])
        self.assertEqual(complete["audit_metadata"]["units"], 6)
        self.assertNotIn("東岬", complete["state"] + removed["state"])
        self.assertIn(self.details["removal_ja"], complete["instructions"])

    def test_window_limits(self) -> None:
        texts = ["p0. p1.", "x" * 3000, "g0. g1.", "n0. n1.", "m0. m1.", "z0. z1."]
        self.assertEqual(src_qa.tydi_window(texts, 2, "en"), [2, 3, 4])
        texts = ["a" * 1000, "b" * 1000, "c" * 1500, "d" * 1000, "e" * 1000]
        self.assertEqual(src_qa.tydi_window(texts, 2, "en"), [1, 2, 3])
        texts = ["one. two. three.", "g0. g1. g2. g3. g4. g5.", "four. five."]
        self.assertEqual(src_qa.tydi_window(texts, 1, "en"), [1])

    def test_removal_record_offsets(self) -> None:
        paragraphs = [" 前の段落。", "  本文は「東の島」の話である。 ", "次の段落。"]
        line = tydi("japanese", "どこ？", "島", paragraphs, [(1, "東の島", "NONE")])
        document = line["document_plaintext"].encode("utf-8")
        starts = line["passage_answer_candidates"]["plaintext_start_byte"]
        ends = line["passage_answer_candidates"]["plaintext_end_byte"]
        passages = [
            document[s:e].decode("utf-8") for s, e in zip(starts, ends, strict=True)
        ]
        mark = tuple(
            line["annotations"][key][0]
            for key in (
                "passage_answer_candidate_index",
                "minimal_answers_start_byte",
                "minimal_answers_end_byte",
                "yes_no_answer",
            )
        )
        item, reason = src_qa.tydi_removal_record(
            document,
            passages,
            starts,
            ends,
            mark,
            local_id="x:1",
            group_key="どこ？",
            question="どこ？",
            title="島",
            language="ja",
        )
        self.assertIsNone(reason)
        start = item["answer_starts"][0]
        self.assertEqual(item["context"][start : start + len("東の島")], "東の島")
        self.assertEqual(
            item["context"], "前の段落。\n本文は「東の島」の話である。\n次の段落。"
        )
        long_gold = ["x" * 3001]
        item, reason = src_qa.tydi_removal_record(
            b"x" * 3001,
            long_gold,
            [0],
            [3001],
            (0, 10, 12, "NONE"),
            local_id="x:2",
            group_key="q",
            question="q",
            title="",
            language="en",
        )
        self.assertEqual((item, reason), (None, "gold_passage_too_long"))

    def test_relevance_rows(self) -> None:
        choice, report = self.built("tydi_relevance_choice_ja")
        noul, _ = self.built("tydi_relevance_noul_ja")
        self.assertEqual(len(ja_sentences("庚", 11)), 199)
        self.assertEqual(report["gold_under_200_chars"], 1)
        by_id = {row["audit_metadata"]["source_local_id"]: row for row in choice + noul}
        row = by_id["train-00001-of-00002:1:choice"]
        self.assertEqual(len(row["options"]), 4)
        self.assertEqual(
            row["options"][row["label"]]["description"], ja_sentences("庚", 11)
        )
        self.assertEqual(
            sorted(option["description"] for option in row["options"]),
            sorted(
                [
                    ja_sentences(t, n)
                    for t, n in (("戊", 12), ("己", 13), ("庚", 11), ("癸", 12))
                ]
            ),
        )
        self.assertEqual(row["state"], "Question: " + self.details["passage_only_ja"])
        answer = by_id["train-00001-of-00002:1:answer"]
        other = by_id["train-00001-of-00002:1:non_answer"]
        self.assertEqual((answer["label"], other["label"]), (1, 0))
        self.assertEqual(answer["state"], ja_sentences("庚", 11))
        self.assertIn(
            other["state"],
            [ja_sentences(t, n) for t, n in (("戊", 12), ("己", 13), ("癸", 12))],
        )
        for item in choice:
            self.assertEqual(item["render_template"], "m2/tydi_relevance/v1/choice")
        for item in noul:
            self.assertEqual(item["render_template"], "m2/tydi_relevance/v1/noul")

    def test_english_removal_only(self) -> None:
        rows, report = self.built("tydi_removal_en")
        self.assertEqual(report["allocation"]["removal"], 1)
        self.assertEqual(report["allocation"]["relevance"], 1)
        self.assertEqual(len(rows), 2)
        self.assertEqual({row["language"] for row in rows}, {"en"})
        self.assertEqual(
            {row["group_id"] for row in rows},
            {group_id("tydi-miracl", normalize(self.details["removal_en"]))},
        )

    def test_language_prefilter(self) -> None:
        path = self.root / "lines.jsonl"
        good = {"language": "japanese", "question_text": "a"}
        path.write_bytes(
            (json.dumps(good, sort_keys=True) + "\n").encode()
            + b'{"annotations": {"x": [1, 2]}, "language": "thai", "broken\n'
            + b"\n"
            + (
                json.dumps(dict(good, question_text="b"), separators=(",", ":")) + "\n"
            ).encode()
        )
        self.assertEqual(
            [(n, r["question_text"]) for n, r in src_qa.tydi_lines(path, "japanese")],
            [(1, "a"), (4, "b")],
        )

    def test_cache_does_not_change_results(self) -> None:
        noul_first = self.built("tydi_relevance_noul_ja")
        choice_cached = self.built("tydi_relevance_choice_ja")
        src_qa._TYDI_CACHE.clear()
        choice_fresh = self.built("tydi_relevance_choice_ja")
        src_qa._TYDI_CACHE.clear()
        self.assertEqual(canonical(choice_cached), canonical(choice_fresh))
        self.assertEqual(
            canonical(noul_first), canonical(self.built("tydi_relevance_noul_ja"))
        )
        self.assertLessEqual(len(src_qa._TYDI_CACHE), 1)


class DeterminismTest(FixtureCase):
    def test_rebuild_is_identical(self) -> None:
        first = digest(build_all(self.dirs))
        src_qa._TYDI_CACHE.clear()
        self.assertEqual(first, digest(build_all(self.dirs)))

    def test_hash_seed_independent(self) -> None:
        expected = digest(build_all(self.dirs))
        code = (
            "import hashlib, sys\n"
            "from pathlib import Path\n"
            "from training.model.data import canonical\n"
            "from v2.data.m2 import spec, src_qa\n"
            "dirs = spec.resolve(Path(sys.argv[1]))\n"
            "built = {i.family: i.build(dirs) for i in src_qa.FAMILIES}\n"
            "print(hashlib.sha256(canonical(built).encode('utf-8')).hexdigest())\n"
        )
        for seed in ("0", "4242"):
            result = subprocess.run(
                [sys.executable, "-c", code, str(self.root)],
                cwd=PACKAGE_ROOT,
                env={
                    **os.environ,
                    "PYTHONHASHSEED": seed,
                    "PYTHONPATH": str(PACKAGE_ROOT),
                },
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertEqual(result.stdout.strip(), expected)

    def test_orchestrator(self) -> None:
        for arm in ("e11", "h5"):
            rows, report = build.build(arm, self.dirs, workers=1, modules=("src_qa",))
            names = {i.family for i in src_qa.FAMILIES if i.arm == arm}
            self.assertEqual(set(report["families"]), names)
            for name, entry in report["families"].items():
                self.assertLessEqual(entry["after_cap"], entry["cap_rows"])
                self.assertEqual(
                    entry["after_cap"], sum(r["family"] == name for r in rows)
                )
            self.assertEqual(len({row["id"] for row in rows}), len(rows))
        pooled, _ = build.build("e11", self.dirs, workers=2, modules=("src_qa",))
        serial, _ = build.build("e11", self.dirs, workers=1, modules=("src_qa",))
        self.assertEqual(canonical(pooled), canonical(serial))


if __name__ == "__main__":
    unittest.main()
