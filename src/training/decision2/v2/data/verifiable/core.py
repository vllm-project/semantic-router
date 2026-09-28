"""Shared helpers for the deterministic, oracle-verified Decision 2.0 generator."""

from __future__ import annotations

import calendar as _calendar
import collections
import hashlib
import math
import random
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from fractions import Fraction
from typing import Any

from training.model.data import INPUT_FIELDS, digest, validate_row

GENERATOR = "decision2-verifiable-v2"
VERSION = "2.0.0"
SOURCE_PREFIX = "decision2_verifiable_v2_"
LANGS = ("en", "zh")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def make_rng(*parts: object) -> random.Random:
    material = "|".join(str(part) for part in parts).encode("utf-8")
    return random.Random(int.from_bytes(hashlib.sha256(material).digest(), "big"))


def group_hash(seed: str, family: str, lang: str, index: int) -> str:
    return sha256_text(f"{seed}|{family}|{lang}|{index}")[:16]


def is_zh(index: int, share: Fraction) -> bool:
    return math.floor((index + 1) * share) > math.floor(index * share)


def held_out(group_id: str) -> bool:
    return int(sha256_text(group_id), 16) % 10 == 0


# ---------------------------------------------------------------- options

NOUL_TEXT = {"en": ("No", "Yes"), "zh": ("否", "是")}


def noul_options(lang: str) -> list[dict[str, str]]:
    no, yes = NOUL_TEXT[lang]
    return [{"key": "false", "description": no}, {"key": "true", "description": yes}]


def choice_options(descriptions: Sequence[str]) -> list[dict[str, str]]:
    return [{"key": f"o{i}", "description": d} for i, d in enumerate(descriptions, 1)]


def score_options(descriptions: Sequence[str]) -> list[dict[str, str]]:
    return [{"key": str(i), "description": d} for i, d in enumerate(descriptions)]


# ---------------------------------------------------------------- dates

MONTHS_EN = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)
WEEKDAYS_EN = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)
WEEKDAYS_ZH = ("星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日")
WEEKDAYS = {"en": WEEKDAYS_EN, "zh": WEEKDAYS_ZH}
_M = "|".join(MONTHS_EN)
DATE_RX = {
    "en": rf"(?:(?:{_M}) \d{{1,2}}, \d{{4}}|\d{{1,2}} (?:{_M}) \d{{4}})",
    "zh": r"\d{4}年\d{1,2}月\d{1,2}日",
}
WEEKDAY_RX = {"en": "|".join(WEEKDAYS_EN), "zh": "|".join(WEEKDAYS_ZH)}


def fmt_date(value: date, lang: str, style: int = 0) -> str:
    if lang == "zh":
        return f"{value.year}年{value.month}月{value.day}日"
    if style == 1:
        return f"{value.day} {MONTHS_EN[value.month - 1]} {value.year}"
    return f"{MONTHS_EN[value.month - 1]} {value.day}, {value.year}"


def fmt_date_wd(value: date, lang: str, style: int = 0) -> str:
    if lang == "zh":
        return f"{fmt_date(value, lang)}（{WEEKDAYS_ZH[value.weekday()]}）"
    return f"{WEEKDAYS_EN[value.weekday()]}, {fmt_date(value, lang, style)}"


def to_date(text: str) -> date:
    text = text.strip()
    if m := re.fullmatch(r"(\d{4})年(\d{1,2})月(\d{1,2})日", text):
        return date(int(m[1]), int(m[2]), int(m[3]))
    if m := re.fullmatch(rf"({_M}) (\d{{1,2}}), (\d{{4}})", text):
        return date(int(m[3]), MONTHS_EN.index(m[1]) + 1, int(m[2]))
    if m := re.fullmatch(rf"(\d{{1,2}}) ({_M}) (\d{{4}})", text):
        return date(int(m[3]), MONTHS_EN.index(m[2]) + 1, int(m[1]))
    raise ValueError(f"not a date: {text!r}")


def month_end(year: int, month: int) -> date:
    return date(year, month, _calendar.monthrange(year, month)[1])


def add_months(value: date, months: int) -> date:
    index = value.year * 12 + value.month - 1 + months
    year, month = divmod(index, 12)
    return date(
        year, month + 1, min(value.day, _calendar.monthrange(year, month + 1)[1])
    )


def random_date(
    rng: random.Random, start: date = date(2025, 1, 6), span: int = 700
) -> date:
    return start + timedelta(days=rng.randrange(span))


# ---------------------------------------------------------------- numbers


def fmt_int(value: int, lang: str) -> str:
    return f"{value:,}" if lang == "en" else str(value)


def fmt_dec(value: float, places: int) -> str:
    text = f"{value:.{places}f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


ZH_DIGITS = "零一二三四五六七八九十"


def ordinal_suffix(value: int) -> str:
    suffix = (
        "th"
        if 10 <= value % 100 <= 20
        else {1: "st", 2: "nd", 3: "rd"}.get(value % 10, "th")  # codespell:ignore nd
    )
    return f"{value}{suffix}"


def zh_number(value: int) -> str:
    if value <= 10:
        return ZH_DIGITS[value]
    if value < 20:
        return "十" + ZH_DIGITS[value - 10]
    tens, ones = divmod(value, 10)
    return ZH_DIGITS[tens] + "十" + (ZH_DIGITS[ones] if ones else "")


# ---------------------------------------------------------------- names

EN_FIRST = (
    "Amara",
    "Lukas",
    "Priya",
    "Mateo",
    "Hiroshi",
    "Ingrid",
    "Tomasz",
    "Leila",
    "Kofi",
    "Sofia",
    "Dmitri",
    "Aiko",
    "Rafael",
    "Nadia",
    "Emeka",
    "Freya",
    "Arjun",
    "Camila",
    "Jonas",
    "Yara",
    "Wei",
    "Olga",
    "Tariq",
    "Elena",
    "Kwame",
    "Mei",
    "Bastian",
    "Zainab",
    "Anders",
    "Lucia",
    "Ravi",
    "Hanna",
    "Diego",
    "Ayesha",
    "Soren",
    "Chiara",
    "Femi",
    "Noor",
    "Pieter",
    "Ines",  # codespell:ignore ines
    "Kenji",
    "Marta",
    "Omar",
    "Greta",
    "Sanjay",
    "Beatriz",
    "Tobias",
    "Amina",
    "Viktor",
    "Rosa",
    "Hamid",
    "Linnea",
    "Andres",
    "Keiko",
    "Malik",
    "Paola",
    "Stefan",
    "Adaeze",
    "Luca",
    "Farida",
)
EN_LAST = (
    "Adebayo",
    "Brandt",
    "Iyer",
    "Salinas",
    "Tanaka",
    "Lindqvist",
    "Nowak",
    "Haddad",
    "Mensah",
    "Ferreira",
    "Volkov",
    "Moriyama",
    "Marchetti",
    "Karimi",
    "Eze",
    "Halvorsen",
    "Duarte",
    "Weber",
    "Nasser",
    "Zhou",
    "Lindahl",
    "Aziz",
    "Marin",
    "Asante",
    "Lin",
    "Vogel",
    "Bello",
    "Sorensen",
    "Rossi",
    "Prasad",
    "Almeida",
    "Fischer",
    "Qureshi",
    "Novak",
    "Moreau",
    "Castillo",
    "Kowalski",
    "Nakamura",
    "Bergstrom",
    "Rahman",
    "Oliveira",
    "Farouk",
    "Dubois",
    "Ncube",
    "Gallagher",
    "Takahashi",
    "Mendes",
    "Hussain",
    "Ortiz",
    "Varga",
    "Chowdhury",
    "Tamura",
    "Abara",
    "Morales",
    "Wojcik",
    "Yilmaz",
    "Horvat",
    "Quispe",
    "Banerjee",
    "Kruger",
    "Delgado",
    "Obi",
    "Svensson",
)
ZH_SURNAMES = (
    "王李张刘陈杨黄赵周吴徐孙朱胡郭何林高罗郑梁谢宋唐许韩冯邓曹彭曾田董潘袁蔡蒋余杜叶程魏苏"
    "吕丁沈任姚卢钟姜崔谭陆范汪廖石金贾夏付方邹熊白孟秦邱侯江尹薛段雷黎史陶贺毛郝顾龚邵万钱戴严"
)
ZH_GIVEN = (
    "思远",
    "慧敏",
    "嘉怡",
    "子涵",
    "浩然",
    "雨桐",
    "晓东",
    "婉清",
    "文博",
    "欣怡",
    "志强",
    "海燕",
    "佳琪",
    "宇航",
    "晨曦",
    "立新",
    "梦瑶",
    "振宇",
    "雅婷",
    "静怡",
    "博文",
    "书瑶",
    "可馨",
    "明轩",
    "秀兰",
    "春生",
    "玉洁",
    "若楠",
    "家豪",
    "诗涵",
    "泽宇",
    "美琳",
    "德明",
    "丽华",
    "建平",
    "少华",
    "嘉欣",
    "昊天",
    "心怡",
    "子墨",
    "晓光",
    "念慈",
    "逸飞",
    "安琪",
)
ZH_NAME_RX = f"[{ZH_SURNAMES}](?:{'|'.join(sorted(ZH_GIVEN, key=len, reverse=True))})"
EN_NAME_RX = r"[A-Z][a-z]+ [A-Z][a-z]+"

EN_COMPANY_STEMS = (
    "Brightwater",
    "Kestrel",
    "Tallowmere",
    "Quorvex",
    "Veltrana",
    "Marrowgate",
    "Ashcombe",
    "Pellucid",
    "Corvane",
    "Lumetta",
    "Fennimore",
    "Saltmarsh",
    "Wrenfield",
    "Cobaltine",
    "Nimbra",
    "Talbrook",
    "Harrowby",
    "Oakhollow",
    "Mistral Bay",
    "Greyfen",
    "Solvara",
    "Thistledown",
    "Ambergate",
    "Rookwood",
    "Calloway & Pike",
    "Juniper Row",
)
ZH_COMPANY_STEMS = (
    "澄湖",
    "北岚",
    "青禾",
    "远帆",
    "知行",
    "云栖",
    "松涧",
    "临溪",
    "墨石",
    "拾光",
    "鹿鸣",
    "清屿",
    "南枝",
    "橙田",
    "柏舟",
    "映山",
    "禾丰",
    "望川",
    "朗月",
    "寒汀",
    "栖霞谷",
    "石桥畔",
)


def people(
    rng: random.Random, lang: str, count: int, exclude: Iterable[str] = ()
) -> list[str]:
    excluded = list(exclude)
    blocked = {item.split()[0] for item in excluded} | set(excluded)
    result: list[str] = []
    while len(result) < count:
        if lang == "zh":
            first, surname = rng.choice(ZH_GIVEN), rng.choice(ZH_SURNAMES)
            name = surname + first
        else:
            first, surname = rng.choice(EN_FIRST), rng.choice(EN_LAST)
            name = f"{first} {surname}"
        if name in blocked or first in blocked or surname in blocked:
            continue
        blocked.update((name, first, surname))
        result.append(name)
    return result


def company(rng: random.Random, lang: str, suffixes: Sequence[str]) -> str:
    if lang == "zh":
        return rng.choice(ZH_COMPANY_STEMS) + rng.choice(suffixes)
    return f"{rng.choice(EN_COMPANY_STEMS)} {rng.choice(suffixes)}"


# ---------------------------------------------------------------- text

_CJK = re.compile(r"[\u4e00-\u9fff]")
_LATIN = re.compile(r"[A-Za-z0-9]+(?:[.,:][0-9]+)*")


def words(text: str, lang: str) -> int:
    if lang == "zh":
        return round(len(_CJK.findall(text)) / 1.5) + len(_LATIN.findall(text))
    return len(text.split())


def mentions(text: str, needle: str) -> bool:
    pattern = rf"(?<![0-9A-Za-z]){re.escape(needle)}(?![0-9A-Za-z])"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def presence_ok(
    state: str, options: Sequence[dict[str, Any]], exempt: Iterable[str] = ()
) -> bool:
    skipped = set(exempt)
    present = [
        mentions(state, option["description"])
        for option in options
        if option["description"] not in skipped
    ]
    return all(present) or not any(present)


def target_words(rng: random.Random) -> int:
    return min(900, max(60, round(math.exp(rng.gauss(math.log(185), 0.62)))))


def compose(
    title: str, sections: Sequence[tuple[str, Sequence[str]]], lang: str
) -> str:
    joiner = "" if lang == "zh" else " "
    parts = [title] if title else []
    for header, sentences in sections:
        if sentences:
            body = joiner.join(sentences)
            parts.append(f"{header}\n{body}" if header else body)
    return "\n\n".join(parts)


SECTION = {
    "en": {"background": "Background", "record": "Record", "notes": "Other notes"},
    "zh": {"background": "背景", "record": "记录", "notes": "其他备注"},
}

_ROOMS = {
    "en": (
        "small meeting room",
        "training room",
        "quiet room",
        "workshop space",
        "print room",
    ),
    "zh": ("小会议室", "培训室", "静音室", "工作坊", "文印室"),
}
_CITIES = {
    "en": (
        "Bergen",
        "Cork",
        "Graz",
        "Leuven",
        "Malmo",
        "Split",
        "Turku",
        "Aarhus",
        "Bilbao",
        "Coimbra",
    ),
    "zh": (
        "绍兴",
        "扬州",
        "泉州",
        "烟台",
        "桂林",
        "洛阳",
        "潍坊",
        "嘉兴",
        "威海",
        "芜湖",
    ),
}
_SUFFIX = {
    "en": (
        "Logistics",
        "Supply Co.",
        "Analytics",
        "Foods",
        "Outfitters",
        "Studios",
        "Works",
    ),
    "zh": ("物流", "科技", "贸易", "食品", "文化", "家居", "印务"),
}
DISTRACTORS = {
    "en": (
        "{p} mentioned that the {room} on the second floor is being repainted.",
        "The {co} newsletter went out to subscribers a little later than usual.",
        "Parking near the main entrance is limited during the morning rush.",
        "{p} has been with the company for {n} years and usually handles vendor questions.",
        "The quarterly all-hands meeting was moved to the larger auditorium.",
        "A fire drill is planned for next month, and staff will get an email beforehand.",
        "{p} asked whether the coffee machine could be replaced with a quieter model.",
        "The shared printer on the east side has been jamming again.",
        "Several people noted that the {city} office keeps its blinds closed in the afternoon.",
        "{co} changed its logo last spring, but the old letterhead is still in use.",
        "The team lunch on {wd} was well attended.",
        "{p} prefers to receive updates by email rather than by phone.",
        "The internal wiki page on travel expenses was rewritten to be shorter.",
        "Visitors must sign in at reception and wear a badge at all times.",
        "The heating in the meeting rooms tends to run warm in the afternoons.",
        "{p} is training a new colleague this quarter and has limited availability.",
        "A courier from {co} dropped off a box of brochures at the front desk.",
        "The {city} branch closes for a local festival once a year.",
        "Nobody raised concerns about the new seating plan.",
        "{p} volunteered to organise the book club this season.",
        "The phone system will be upgraded soon, which may cause short interruptions.",
        "The staff survey drew {n} written comments, mostly about lighting.",
        "{p} and {p2} share an office and usually coordinate their holidays.",
        "The recycling bins were moved closer to the kitchen.",
        "The office plants are watered every {wd} morning.",
        "{co} sponsors a youth football team in {city}.",
        "The lift in the north wing was serviced recently and works normally.",
        "A new water filter was installed in the kitchen.",
        "{p} keeps a spare umbrella at the front desk for visitors.",
        "The company style guide asks for plain language in all customer letters.",
        "There is a small shelf of reference books near the entrance.",
        "Some staff take the ferry to work when the bridge is busy.",
        "{p} suggested adding a second screen to each desk.",
        "The cafeteria now offers a vegetarian dish every day.",
        "{co} moved its paper archive to an off-site storage unit years ago.",
        "Meeting notes are kept in a shared folder that everyone can edit.",
        "The {room} is often booked by the design team.",
        "{p} recently came back from a trade fair in {city}.",
        "The window cleaners come round twice a year, weather permitting.",
        "{p2} thinks the reception area needs brighter lighting.",
        "The internet connection dropped briefly one afternoon, but nothing was lost.",
        "Everyone received a reminder to lock their screens when leaving their desks.",
        "{co} offered a discount on its spring catalogue to returning customers.",
        "The bicycle racks behind the building are usually full by nine.",
        "The {city} team has started a weekly language exchange over lunch.",
        "{p} is collecting suggestions for the next team outing.",
        "A local bakery delivers pastries to the office on special occasions.",
        "The old filing cabinets were donated to a nearby school.",
    ),
    "zh": (
        "{p}提到二楼的{room}最近在重新粉刷。",
        "{co}的内部通讯这期比平时晚发了两天。",
        "早高峰时，正门附近很难找到停车位。",
        "{p}在公司工作了{n}年，平时主要负责对接供应商。",
        "季度全员大会改到了更大的报告厅举行。",
        "下个月有一次消防演练，届时会提前发邮件通知大家。",
        "{p}问能不能把咖啡机换成声音小一点的型号。",
        "东侧那台共用打印机又开始卡纸了。",
        "不少同事说{city}办公室下午总是拉着百叶窗。",
        "{co}去年春天换了新标志，但旧信纸还在继续使用。",
        "{wd}的团队午餐来了很多人。",
        "{p}更希望通过邮件而不是电话了解进展。",
        "内部百科里关于差旅报销的页面被改写得更简洁了。",
        "访客需要在前台登记，并全程佩戴胸卡。",
        "会议室下午的暖气总是开得偏热。",
        "{p}这个季度在带一位新同事，时间比较紧。",
        "{co}的快递员往前台送来了一箱宣传册。",
        "{city}分部每年会因为当地的节庆活动闭馆一天。",
        "对于新的座位安排，大家没有提出异议。",
        "{p}主动报名组织这一季的读书会。",
        "电话系统近期要升级，其间可能会有短暂中断。",
        "员工调查一共收到{n}条书面意见，大多和照明有关。",
        "{p}和{p2}在同一间办公室，休假时间通常会互相协调。",
        "回收箱被挪到了离茶水间更近的地方。",
        "每到{wd}上午，行政同事会给办公室的绿植浇水。",
        "{co}赞助了{city}的一支青少年足球队。",
        "北楼的电梯最近做过保养，运行正常。",
        "茶水间新装了一台净水器。",
        "{p}在前台放了一把备用雨伞，方便来访的客人。",
        "公司的写作规范要求所有客户信函都使用平实的语言。",
        "入口附近有一个小书架，放着一些参考书。",
        "大桥拥堵的时候，有些同事会坐轮渡上班。",
        "{p}建议给每张办公桌再配一块显示器。",
        "食堂现在每天都会提供一道素菜。",
        "{co}多年前就把纸质档案搬到了外面的仓库。",
        "会议纪要都放在一个大家都能编辑的共享文件夹里。",
        "{room}经常被设计组预订。",
        "{p}最近刚从{city}的一个展会回来。",
        "保洁公司每年来擦两次窗户，遇到坏天气就顺延。",
        "{p2}觉得前台区域的灯光应该再亮一些。",
        "有一天下午网络短暂中断了一会儿，好在没有丢失任何资料。",
        "行政部提醒大家离开座位时要锁定电脑屏幕。",
        "{co}给老客户的春季目录打了折。",
        "楼后的自行车停车架通常不到九点就停满了。",
        "{city}团队开始在午饭时间举办每周一次的语言交流。",
        "{p}正在收集下次团建活动的建议。",
        "附近一家面包店会在特别的日子给办公室送点心。",
        "旧的文件柜捐给了附近的一所学校。",
    ),
}


def _stem(name: str, lang: str) -> str:
    return name[:2] if lang == "zh" else name.split()[0]


def distractor_pool(
    rng: random.Random,
    lang: str,
    exclude_names: Iterable[str] = (),
    avoid: Iterable[str] = (),
) -> list[str]:
    excluded = list(exclude_names)
    blocked = [item for item in avoid if item]
    stems = {_stem(name, lang) for name in excluded}
    names = people(rng, lang, 8, exclude=excluded)
    result = []
    for template in rng.sample(DISTRACTORS[lang], len(DISTRACTORS[lang])):
        co = company(rng, lang, _SUFFIX[lang])
        while _stem(co, lang) in stems:
            co = company(rng, lang, _SUFFIX[lang])
        sentence = template.format(
            p=names[rng.randrange(4)],
            p2=names[4 + rng.randrange(4)],
            co=co,
            city=rng.choice(_CITIES[lang]),
            wd=rng.choice(WEEKDAYS[lang]),
            room=rng.choice(_ROOMS[lang]),
            n=rng.randint(3, 19),
        )
        if not any(mentions(sentence, item) for item in blocked):
            result.append(sentence)
    return result


def take_words(pool: Sequence[str], lang: str, budget: int) -> list[str]:
    result, total = [], 0
    for sentence in pool:
        if total >= budget:
            break
        result.append(sentence)
        total += words(sentence, lang)
    return result


def split_filler(
    rng: random.Random, filler: Sequence[str]
) -> tuple[list[str], list[str]]:
    cut = rng.randint(0, len(filler)) if filler else 0
    return list(filler[:cut]), list(filler[cut:])


def filler(
    rng: random.Random,
    lang: str,
    base_words: int,
    exclude_names: Iterable[str] = (),
    avoid: Iterable[str] = (),
) -> tuple[list[str], list[str]]:
    budget = target_words(rng) - base_words
    pool = distractor_pool(rng, lang, exclude_names, avoid)
    return split_filler(rng, take_words(pool, lang, budget) if budget > 0 else [])


# ---------------------------------------------------------------- groups and rows


@dataclass
class Variant:
    state: str
    label: int
    facts: dict[str, Any]
    edit: str


@dataclass
class Group:
    task_type: str
    instructions: str
    options: list[dict[str, str]]
    variants: list[Variant]
    template: str
    subtype: str
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class A4Scenario:
    """One hard-negative scenario; distractor lists exclude the gold value.

    ``near_*``/``pool_*`` are split by side of the gold (nearest first for
    ``near_*``) so the gold's rank among the option values can be balanced.
    """

    state: str
    facts: dict[str, Any]
    subtype: str
    template: str
    question: str
    gold: str
    near_below: list[str]
    near_above: list[str]
    pool_below: list[str]
    pool_above: list[str]
    near_ordered: bool
    pool_ordered: bool
    noul_true: str
    noul_near: str
    noul_far: list[str]
    k: int = 4


@dataclass
class A4v2Render:
    state: str
    facts: dict[str, Any]
    subtype: str
    template: str
    question: str
    proposition: Callable[[str], str]


@dataclass
class A4v2Alternative:
    """One way to finish a hard-negative scenario: which option value becomes the gold."""

    key: int
    gold: str
    near: list[str]
    rand: list[str]
    render: Callable[[], A4v2Render | None]


@dataclass
class A4v2Plan:
    ordered: bool
    alternatives: list[A4v2Alternative]


def rank_windows(
    rng: random.Random, lo: int, hi: int, k: int = 4, width: int = 6
) -> list[tuple[int, list[int], list[int]]]:
    """Per gold rank r: (gold, near, random) value indices, both sets holding the gold at rank r.

    The near set is k consecutive indices; the random set is a random k-point shape
    within ``width`` translated so that its r-th point is the gold. Both are drawn
    before r is known, so option values carry no information about the gold's rank.
    """
    base = rng.randint(lo + width, hi - width - (k - 1))
    while True:
        shape = sorted(rng.sample(range(width + 1), k))
        if shape != list(range(shape[0], shape[0] + k)):
            break
    out = []
    for r in range(k):
        gold = base + r
        near = [base + i for i in range(k) if i != r]
        rand = [gold - shape[r] + s for i, s in enumerate(shape) if i != r]
        out.append((gold, near, rand))
    return out


_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def option_value(text: str) -> float | None:
    for rx in DATE_RX.values():
        if re.fullmatch(rx, text):
            return float(to_date(text).toordinal())
    found = _NUMBER.findall(text)
    return float(found[0].replace(",", "")) if len(found) == 1 else None


def mention_count(text: str, needle: str) -> int:
    return len(
        re.findall(
            rf"(?<![0-9A-Za-z]){re.escape(needle)}(?![0-9A-Za-z])",
            text,
            flags=re.IGNORECASE,
        )
    )


def _credit(scores: Sequence[float], label: int) -> float:
    best = max(scores)
    winners = [i for i, s in enumerate(scores) if s == best]
    return (label in winners) / len(winners)


def heuristic_credits(
    options: Sequence[dict[str, Any]],
    label: int,
    state: str | None,
    frequency: collections.Counter | None,
) -> dict[str, float]:
    """Option-only baselines with fractional credit on ties (chance = 1 / len(options))."""
    texts = [o["description"] for o in options]
    out = {"longest": _credit([len(t) for t in texts], label)}
    if frequency is not None:
        out["frequent"] = _credit([frequency[t] for t in texts], label)
    if state is not None:
        out["mentioned"] = _credit([mention_count(state, t) for t in texts], label)
    values = [option_value(t) for t in texts]
    if all(v is not None for v in values):
        out["argmin"] = _credit([-v for v in values], label)
        out["argmax"] = _credit(values, label)
        ordered = sorted(range(len(values)), key=lambda i: values[i])
        middle = ordered[(len(values) - 1) // 2 : len(values) // 2 + 1]
        out["median"] = (label in middle) / len(middle)
    return out


def template_regex(template: str, **groups: str) -> str:
    out = []
    for part in re.split(r"(\{\w+\})", template):
        slot = re.fullmatch(r"\{(\w+)\}", part)
        out.append(f"({groups[slot.group(1)]})" if slot else re.escape(part))
    return "".join(out)


def score_levels(options: Sequence[dict[str, Any]]) -> list[int]:
    return [int(option["key"]) for option in options]


def is_noul(options: Sequence[dict[str, Any]]) -> bool:
    return [option["key"] for option in options] == ["false", "true"]


def match_option(
    options: Sequence[dict[str, Any]], predicate: Callable[[str], bool]
) -> int | None:
    hits = [i for i, option in enumerate(options) if predicate(option["description"])]
    return hits[0] if len(hits) == 1 else None


def make_row(
    *,
    arm: str,
    family: str,
    lang: str,
    index: int,
    variant_index: int,
    group_id: str,
    state: str,
    instructions: str,
    options: list[dict[str, str]],
    label: int,
    task_type: str,
    template: str,
    edit: str,
    facts: dict[str, Any],
    reparse_label: int | None,
    meta: dict[str, Any],
    split: str = "train",
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": f"{arm}-{family}-{lang}-{index:05d}-v{variant_index}",
        "state": state,
        "instructions": instructions,
        "options": options,
        "label": label,
        "task_type": task_type,
        "family": family,
        "group_id": group_id,
        "language": lang,
        "split": split,
        "source": SOURCE_PREFIX + arm,
        "evaluation_role": split,
        "render_template": f"{family}/{template}/{lang}",
        "audit_metadata": {
            "generator": GENERATOR,
            "version": VERSION,
            "family": family,
            "variant_index": variant_index,
            "operative_edit": edit,
            "facts_sha256": digest(facts),
            "oracle_label": label,
            "reparse_label": reparse_label,
            **meta,
        },
    }
    row["input_sha256"] = digest({name: row[name] for name in INPUT_FIELDS})
    return validate_row(row, split)


def check_group(group: Group) -> None:
    labels = sorted(variant.label for variant in group.variants)
    if labels != list(range(len(group.options))):
        raise AssertionError(
            f"group labels {labels} do not cover the options exactly once"
        )
    if group.task_type == "noul" and len(group.options) != 2:
        raise AssertionError("noul groups need exactly two options")
    if len({variant.state for variant in group.variants}) != len(group.variants):
        raise AssertionError("counterfactual states must differ")
