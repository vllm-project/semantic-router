"""PN1 text rules (prereg ``v2/data/records/m4-pn1-prereg-2026-09-29.md``, sec. 2).

Identity checks compare ``norm`` forms: NFKC, casefold, whitespace and
punctuation (Unicode P* plus ASCII ``string.punctuation``) removed. The overlap
measures are computed on ``norm`` forms: character-bigram Jaccard,
character-unigram multiset Jaccard, the same-multiset flag and the length
ratio. Name swaps (``pn-name``) are pure string rules; twin edits (``pn-twin``)
are checked by ``twin_checks``. Everything here is standard library only.
"""

from __future__ import annotations

import collections
import functools
import json
import re
import unicodedata
from dataclasses import dataclass

from v2.data.textnorm import _punctuation

LANGS = ("ja", "ko", "zh", "es", "de", "fr", "ru", "ar")
ISO3 = {
    "ja": "jpn",
    "ko": "kor",
    "zh": "cmn",
    "es": "spa",
    "de": "deu",
    "fr": "fra",
    "ru": "rus",
    "ar": "ara",
}
LANGUAGE_NAMES = {
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese",
    "es": "Spanish",
    "de": "German",
    "fr": "French",
    "ru": "Russian",
    "ar": "Arabic",
}
FAMILIES = ("pn-hop", "pn-near", "pn-name", "pn-twin")
GROUP_OF = {
    "pn-hop": "natural",
    "pn-near": "natural",
    "pn-name": "swap",
    "pn-twin": "swap",
}
BIN_EDGES = (0.3, 0.5, 0.7, 0.85)
BIN_LABELS = ("[0,.3)", "[.3,.5)", "[.5,.7)", "[.7,.85)", "[.85,1]")
MAX_LENGTH_RATIO = 1.6
NEAR_MIN_JACCARD = 0.3
TWIN_MIN_MULTISET = 0.8
NEAR_DUPLICATE_CONTAINMENT = 0.8

INSTRUCTIONS = (
    "Do the two sentences have the same meaning?",
    "Is one sentence a paraphrase of the other, i.e. do both sentences say exactly the same thing?",
    "Do these two sentences mean the same thing, even though the wording or word order may differ?",
    "Would a fluent speaker say that both sentences express the same meaning?",
    "Are the two sentences paraphrases of each other?",
    "Does the second sentence convey exactly the same meaning as the first?",
    "Do both sentences state the same thing, with the same people and things in the same roles?",
    "Is the meaning of the two sentences identical?",
)
LAYOUTS = (
    "Sentence A: {a}\nSentence B: {b}",
    "First sentence: {a}\nSecond sentence: {b}",
    "Text 1: {a}\nText 2: {b}",
    "(1) {a}\n(2) {b}",
    "A: {a}\nB: {b}",
)


@functools.cache
def _dropped() -> frozenset[str]:
    return frozenset(_punctuation())


def norm(text: str) -> str:
    dropped = _dropped()
    return "".join(
        char
        for char in unicodedata.normalize("NFKC", text).casefold()
        if not char.isspace() and char not in dropped
    )


def usable(text: str) -> bool:
    """A sentence can enter a row: no control characters and a nonempty norm."""
    return (
        bool(text)
        and not any(unicodedata.category(char) == "Cc" for char in text)
        and bool(norm(text))
    )


def grams(value: str, n: int) -> set[str]:
    if len(value) < n:
        return {value} if value else set()
    return {value[i : i + n] for i in range(len(value) - n + 1)}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def bigram_jaccard(na: str, nb: str) -> float:
    return jaccard(grams(na, 2), grams(nb, 2))


def multiset_jaccard(na: str, nb: str) -> float:
    ca, cb = collections.Counter(na), collections.Counter(nb)
    union = sum((ca | cb).values())
    return sum((ca & cb).values()) / union if union else 1.0


def same_multiset(na: str, nb: str) -> bool:
    return collections.Counter(na) == collections.Counter(nb)


def length_ratio(na: str, nb: str) -> float:
    short, long = sorted((len(na), len(nb)))
    return long / max(short, 1)


def containment(na: str, nb: str, n: int = 4) -> float:
    """Share of ``na``'s character n-grams that occur in ``nb`` (0 below n characters)."""
    if len(na) < n:
        return 0.0
    mine = grams(na, n)
    return len(mine & grams(nb, n)) / len(mine)


def near_duplicate(na: str, nb: str) -> bool:
    return (
        na == nb
        or containment(na, nb) >= NEAR_DUPLICATE_CONTAINMENT
        or containment(nb, na) >= NEAR_DUPLICATE_CONTAINMENT
    )


def overlap_bin(value: float) -> int:
    return sum(value >= edge for edge in BIN_EDGES)


def pair_metrics(a: str, b: str) -> dict[str, object]:
    na, nb = norm(a), norm(b)
    value = bigram_jaccard(na, nb)
    return {
        "bigram_jaccard": round(value, 6),
        "multiset_jaccard": round(multiset_jaccard(na, nb), 6),
        "same_multiset": same_multiset(na, nb),
        "length_ratio": round(length_ratio(na, nb), 6),
        "bin": overlap_bin(value),
        "norm_equal": na == nb,
    }


def stratum(family: str, metrics: dict[str, object]) -> str:
    return (
        f"{GROUP_OF[family]}|ms{int(bool(metrics['same_multiset']))}|b{metrics['bin']}"
    )


# --- pn-name: person names, coordination and role swaps -------------------------------------

NAMES = {
    "ja": {"tom": ("トム",), "mary": ("メアリー",), "john": ("ジョン",)},
    "ko": {"tom": ("톰",), "mary": ("메리",), "john": ("존",)},
    "zh": {"tom": ("汤姆", "湯姆"), "mary": ("玛丽", "瑪麗"), "john": ("约翰", "約翰")},
    "es": {"tom": ("Tom",), "mary": ("Mary", "María"), "john": ("John",)},
    "de": {"tom": ("Tom",), "mary": ("Maria", "Mary"), "john": ("John",)},
    "fr": {"tom": ("Tom",), "mary": ("Marie", "Mary"), "john": ("John",)},
    "ru": {"tom": ("Том",), "mary": ("Мэри",), "john": ("Джон",)},
    "ar": {"tom": ("توم",), "mary": ("ماري",), "john": ("جون",)},
}
CONJUNCTIONS = {
    "ja": ("と",),
    "ko": ("과", "와", "하고", "이랑", "랑"),
    "zh": ("和", "跟", "与", "與"),
    "es": ("y",),
    "de": ("und",),
    "fr": ("et",),
    "ru": ("и",),
    "ar": ("و",),
}
COMITATIVE = {
    "es": ("con",),
    "de": ("mit",),
    "fr": ("avec",),
    "ru": ("с", "со"),
    "ar": ("مع",),
}
MIDDLE_DOTS = frozenset("·・•‧")
# Characters that continue a transliterated name (汤姆斯, 玛丽亚, 约翰逊, ...).
ZH_EXTENDERS = frozenset("斯森逊遜亚亞娅婭莲蓮尼内內林琳娜安克") | MIDDLE_DOTS
ZH_LINKERS = frozenset("和跟与與同")
KO_ALLOMORPHS = (
    ("이", "가"),
    ("은", "는"),
    ("을", "를"),
    ("과", "와"),
    ("이랑", "랑"),
    ("으로", "로"),
)
_KO_PAIR = {form: pair for pair in KO_ALLOMORPHS for form in pair}
KO_FIXED = (
    "에게서",
    "한테서",
    "에게",
    "한테",
    "에서",
    "보다",
    "처럼",
    "까지",
    "부터",
    "하고",
    "같이",
    "조차",
    "마저",
    "밖에",
    "의",
    "께",
    "도",
    "만",
    "에",
    "뿐",
)
_KO_BASES = tuple(sorted(set(_KO_PAIR) | set(KO_FIXED), key=lambda s: (-len(s), s)))
_KO_CLOSED = frozenset(("이", "가", "은", "는", "을", "를", "의"))
KO_TAILS = frozenset(("", "는", "은", "도", "만", "의", "을", "를", "이", "가", "요"))
KO_COORD = frozenset(CONJUNCTIONS["ko"])


def hangul_final(char: str) -> int:
    """Index of the final consonant of a Hangul syllable (0 = none, 8 = ㄹ)."""
    code = ord(char) - 0xAC00
    if not 0 <= code < 11172:
        raise ValueError(f"not a Hangul syllable: {char!r}")
    return code % 28


def ko_particle_after(base: str, name: str) -> str:
    """The allomorph of ``base`` that follows ``name`` (unchanged if not allomorphic)."""
    pair = _KO_PAIR.get(base)
    if pair is None:
        return base
    final = hangul_final(name[-1])
    if pair == ("으로", "로"):
        return "로" if final in (0, 8) else "으로"
    return pair[0] if final else pair[1]


def ko_split_particle(cont: str) -> tuple[str, str] | None:
    """(particle, tail) of the Hangul run that follows a name, or None if unlisted."""
    if cont == "":
        return "", ""
    if cont.startswith("씨"):
        return ("씨", cont[1:]) if ko_split_particle(cont[1:]) is not None else None
    for base in _KO_BASES:
        tail = cont[len(base) :]
        if cont.startswith(base) and (
            tail == "" or (base not in _KO_CLOSED and tail in KO_TAILS)
        ):
            return base, tail
    return None


def _hangul(char: str) -> bool:
    return (
        "\uac00" <= char <= "\ud7a3"
        or "\u1100" <= char <= "\u11ff"
        or "\u3130" <= char <= "\u318f"
    )


def _katakana(char: str) -> bool:
    return (
        "\u30a0" <= char <= "\u30ff"
        or "\u31f0" <= char <= "\u31ff"
        or "\uff65" <= char <= "\uff9f"
    )


def _word_char(char: str) -> bool:
    return char.isalnum() or char in "-'’_"


@dataclass(frozen=True)
class Mention:
    person: str
    name: str
    start: int
    end: int
    particle: str = ""
    tail: str = ""

    @property
    def stop(self) -> int:
        return self.end + len(self.particle) + len(self.tail)


def _mention(
    text: str, lang: str, person: str, name: str, start: int
) -> Mention | None:
    end = start + len(name)
    before = text[start - 1] if start else ""
    after = text[end] if end < len(text) else ""
    if lang in ("es", "de", "fr", "ru"):
        ok = not (before and _word_char(before)) and not (after and _word_char(after))
    elif lang == "ar":
        prefixed = before == "و" and (start == 1 or not text[start - 2].isalnum())
        ok = (not before or not before.isalnum() or prefixed) and not (
            after and after.isalnum()
        )
    elif lang == "ja":
        ok = not any(
            char and (_katakana(char) or char in MIDDLE_DOTS)
            for char in (before, after)
        )
    elif lang == "zh":
        ok = after not in ZH_EXTENDERS and before not in MIDDLE_DOTS
    elif lang == "ko":
        if before and _hangul(before):
            return None
        stop = end
        while stop < len(text) and _hangul(text[stop]):
            stop += 1
        split = ko_split_particle(text[end:stop])
        if split is None:
            return None
        return Mention(person, name, start, end, *split)
    else:
        raise ValueError(f"unsupported language {lang}")
    return Mention(person, name, start, end) if ok else None


def find_mentions(text: str, lang: str) -> list[Mention] | None:
    """Every occurrence of a listed name; None if any occurrence fails its boundary rule."""
    mentions = []
    for person, variants in NAMES[lang].items():
        for name in variants:
            start = text.find(name)
            while start != -1:
                mention = _mention(text, lang, person, name, start)
                if mention is None:
                    return None
                mentions.append(mention)
                start = text.find(name, start + 1)
    mentions.sort(key=lambda m: m.start)
    for left, right in zip(mentions, mentions[1:]):
        if left.stop > right.start:
            return None
    return mentions


def _coordinated(text: str, lang: str, first: Mention, second: Mention) -> bool:
    between = text[first.stop : second.start]
    if lang in ("es", "de", "fr", "ru"):
        return (
            re.fullmatch(rf"\s+(?:{'|'.join(CONJUNCTIONS[lang])})\s+", between)
            is not None
        )
    if lang == "ar":
        return re.fullmatch(r"\s+و\s?", between) is not None
    if lang == "ko":
        return first.particle in KO_COORD and not first.tail and between.strip() == ""
    return between in CONJUNCTIONS[lang]


def _linked(text: str, lang: str, mention: Mention) -> bool:
    """The name is coordinated with, or in a with-phrase of, something (role swaps exclude it)."""
    before = text[: mention.start]
    after = text[mention.stop :]
    if lang == "ja":
        return after.startswith("と")
    if lang == "ko":
        return mention.particle in KO_COORD
    if lang == "zh":
        return after[:1] in ZH_LINKERS or before[-1:] in ZH_LINKERS
    word = re.search(r"(\w+)\s+$", before)
    if word is None:
        return False
    previous = word.group(1).lower()
    markers = COMITATIVE[lang]
    return previous in markers or (
        lang == "ar" and previous[:1] == "و" and previous[1:] in markers
    )


def name_swap(text: str, lang: str) -> tuple[str, str] | None:
    """('coord', edited) for A CONJ B -> B CONJ A (yes), ('role', edited) for an in-place swap (no).

    The sentence must name exactly two of the three listed persons, each exactly once.
    Korean particles after both names are re-selected by allomorphy.
    """
    mentions = find_mentions(text, lang)
    if mentions is None or len(mentions) != 2:
        return None
    first, second = mentions
    if first.person == second.person:
        return None
    kind = "coord" if _coordinated(text, lang, first, second) else "role"
    if kind == "role" and (_linked(text, lang, first) or _linked(text, lang, second)):
        return None
    if lang == "ko":
        edited = (
            text[: first.start]
            + second.name
            + ko_particle_after(first.particle, second.name)
            + first.tail
            + text[first.stop : second.start]
            + first.name
            + ko_particle_after(second.particle, first.name)
            + second.tail
            + text[second.stop :]
        )
    else:
        edited = (
            text[: first.start]
            + second.name
            + text[first.end : second.start]
            + first.name
            + text[second.end :]
        )
    return (kind, edited) if edited != text else None


# --- scripts and twin edits -----------------------------------------------------------------

SCRIPT_CORE = {
    "ja": frozenset(("Han", "Hiragana", "Katakana")),
    "zh": frozenset(("Han",)),
    "ko": frozenset(("Hangul",)),
    "es": frozenset(("Latin",)),
    "de": frozenset(("Latin",)),
    "fr": frozenset(("Latin",)),
    "ru": frozenset(("Cyrillic",)),
    "ar": frozenset(("Arabic",)),
}
_SCRIPT_PREFIXES = (
    ("CJK UNIFIED IDEOGRAPH", "Han"),
    ("CJK COMPATIBILITY IDEOGRAPH", "Han"),
    ("IDEOGRAPHIC", "Han"),
    ("HIRAGANA", "Hiragana"),
    ("KATAKANA", "Katakana"),
    ("HALFWIDTH KATAKANA", "Katakana"),
    ("HANGUL", "Hangul"),
    ("HALFWIDTH HANGUL", "Hangul"),
    ("LATIN", "Latin"),
    ("FULLWIDTH LATIN", "Latin"),
    ("CYRILLIC", "Cyrillic"),
    ("ARABIC", "Arabic"),
)


def scripts(text: str) -> set[str]:
    found = set()
    for char in text:
        if not char.isalpha():
            continue
        name = unicodedata.name(char, "")
        found.add(
            next(
                (
                    script
                    for prefix, script in _SCRIPT_PREFIXES
                    if name.startswith(prefix)
                ),
                "Other",
            )
        )
    return found


def script_ok(edit: str, seed: str, lang: str) -> bool:
    """The edit uses the language's script and no script absent from both it and the seed."""
    used = scripts(edit)
    core = SCRIPT_CORE[lang]
    return bool(used & core) and used <= (core | scripts(seed))


def twin_eligible(text: str, lang: str) -> bool:
    """Seed length rule: >= 8 words (es de fr ru ar), >= 5 (ko), >= 16 characters (ja zh)."""
    if lang in ("ja", "zh"):
        return len(norm(text)) >= 16
    words = [word for word in text.split() if norm(word)]
    return len(words) >= (5 if lang == "ko" else 8)


def twin_prompt(lang: str, sentence: str) -> str:
    name = LANGUAGE_NAMES[lang]
    return (
        f"Here is a sentence in {name}:\n{sentence}\n\n"
        "Write two edited versions of this sentence, both in "
        f"{name}, and answer with only a JSON object of the form "
        '{"different": "...", "same": "..."}.\n'
        '- "different": swap the positions of exactly two words or short phrases (for '
        "example two people, two places or two objects, or the subject and the object) so "
        f"that the sentence stays grammatical and natural in {name} but its meaning "
        "changes. Keep every other word and do not add any negation.\n"
        '- "same": keep exactly the same meaning by only changing the word order or by '
        "replacing at most one word with an exact synonym. Keep every other word; the "
        f"sentence must be natural in {name}."
    )


def parse_twin(raw: str) -> dict[str, str] | None:
    """The model's {"different", "same"} object (a ```json fence is allowed), else None."""
    text = raw.strip()
    fence = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.S)
    if fence:
        text = fence.group(1)
    try:
        value = json.loads(text)
    except ValueError:
        return None
    if not isinstance(value, dict) or set(value) != {"different", "same"}:
        return None
    if not all(isinstance(value[key], str) and value[key].strip() for key in value):
        return None
    return {key: value[key].strip() for key in ("different", "same")}


def twin_checks(seed: str, edits: dict[str, str], lang: str) -> dict[str, str | None]:
    """Rejection reason per edit (None = kept)."""
    seed_norm = norm(seed)
    reasons: dict[str, str | None] = {}
    for kind in ("different", "same"):
        edit = edits[kind]
        edit_norm = norm(edit)
        if not usable(edit):
            reasons[kind] = "format"
        elif edit_norm == seed_norm:
            reasons[kind] = "identical"
        elif multiset_jaccard(edit_norm, seed_norm) < TWIN_MIN_MULTISET:
            reasons[kind] = "multiset_jaccard"
        elif not script_ok(edit, seed, lang):
            reasons[kind] = "script"
        else:
            reasons[kind] = None
    if norm(edits["different"]) == norm(edits["same"]):
        reasons = {kind: reasons[kind] or "edits_identical" for kind in reasons}
    return reasons


# --- judge prompts ----------------------------------------------------------------------------


def label_prompt(lang: str, a: str, b: str) -> str:
    return (
        f"Here are two sentences in {LANGUAGE_NAMES[lang]}.\n"
        f"Sentence A: {a}\nSentence B: {b}\n"
        "Do the two sentences have exactly the same meaning, i.e. are they paraphrases? "
        "Answer Yes or No."
    )


def fluency_prompt(lang: str, sentence: str) -> str:
    return (
        f"Is the following sentence natural and grammatical {LANGUAGE_NAMES[lang]}?\n"
        f"{sentence}\nAnswer Yes or No."
    )


def label_prompt_v2(lang: str, a: str, b: str) -> str:
    """Amendment 3 label prompt."""
    return (
        f"Here are two sentences in {LANGUAGE_NAMES[lang]}.\n"
        f"Sentence A: {a}\nSentence B: {b}\n"
        "Do the two sentences mean the same thing? Differences in wording, word order, "
        "politeness, formality or punctuation do not matter. Answer No if they differ in "
        "who does what to whom, in a person, place, object, time or quantity, or in any "
        "other fact. Answer Yes or No."
    )


def fluency_prompt_v2(lang: str, sentence: str) -> str:
    """Amendment 3 fluency prompt."""
    return (
        f"Here is a sentence in {LANGUAGE_NAMES[lang]}:\n{sentence}\n"
        "Is this a grammatical sentence that a native speaker could write? Minor "
        "awkwardness or unusual word order is fine; answer No only for a clear grammatical "
        "error or a sentence that makes no sense. Answer Yes or No."
    )


PROMPTS = {
    "v1": (label_prompt, fluency_prompt),
    "v2": (label_prompt_v2, fluency_prompt_v2),
}


def word_jaccard(a: str, b: str) -> float:
    from v2.data.textnorm import word_tokens

    return jaccard(set(word_tokens(a)), set(word_tokens(b)))


def edit_distance(na: str, nb: str) -> float:
    """Levenshtein distance of two norm forms over the longer length."""
    if not na and not nb:
        return 0.0
    previous = list(range(len(nb) + 1))
    for i, ca in enumerate(na, 1):
        current = [i]
        for j, cb in enumerate(nb, 1):
            current.append(
                min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb))
            )
        previous = current
    return previous[-1] / max(len(na), len(nb))


def extra_metrics(first: str, second: str) -> dict[str, float]:
    """Self-check features beyond the matching coordinates, in state order."""
    na, nb = norm(first), norm(second)
    return {
        "word_jaccard": round(word_jaccard(first, second), 6),
        "edit_distance": round(edit_distance(na, nb), 6),
        "containment_ab": round(containment(na, nb), 6),
        "containment_ba": round(containment(nb, na), 6),
    }
