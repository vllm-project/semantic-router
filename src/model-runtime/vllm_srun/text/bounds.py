"""Long inputs, tokenized only as far as an answer depends on them.

A model reads a bounded number of tokens of an input, but the tokenizer
spends memory and time on all of it: 160 to 400 bytes of tokenizer state per
token, 1.2 million tokens for 5 MiB of English. ``read`` tokenizes a prefix
instead and keeps only tokens that are exactly the whole text's first tokens:

- A prefix that ends before a cut point (whitespace, punctuation, a symbol, or
  a CJK, kana or Hangul character) keeps the tokens of its words except the
  last word and any word that starts within ``lookback`` characters of the
  cut: every built-in tokenizer starts a pre-tokenizer word at such a
  character or a few characters around it, and none normalizes across one.
- A run with no word boundary (an unbroken base64 or hex run, Chinese without
  spaces in a tokenizer that splits words only at spaces) is cut before a
  character NFC never composes with the one before it, and keeps the tokens
  that end ``RUN_MARGIN`` characters before the cut: the built-in tokenizers
  change at most about 100 characters before such a cut. A tokenizer that
  drops or folds characters never cuts inside a run: WordPiece reads a word
  of more than 100 characters as one unknown token, and a prefix of a word
  whose marks it strips can be shorter than that.

The prefix grows until it keeps the tokens a caller needs or reaches the end
of the text; realistic text runs at most about 4.5 characters per token, so
the first prefix (``CHARS_PER_TOKEN`` per needed token) usually suffices, and
a text that short is tokenized whole, as before. ``facts`` probes each
tokenizer once and enables only the cuts that keep its tokens exact on probe
texts; a tokenizer that passes none reads whole texts.

No token covers more than ``chars_per_token`` characters: the longest
vocabulary entry, times three when the tokenizer composes characters (NFC
turns up to three code points into one). So a prefix never needs more than
``need * chars_per_token`` characters past the margin, and ``surely_over``
tells, without tokenizing, that a text has more tokens than a budget. A
tokenizer that can drop characters (WordPiece drops whitespace and folds a
long word into one unknown token) has no such bound: it reads a text without
cut points whole and never rejects early.
"""

from __future__ import annotations

import bisect
import re
import threading
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

# Characters the first prefix reads per needed token: realistic text in the
# built-in tokenizers runs at most about 4.5 per token (English prose).
CHARS_PER_TOKEN = 6
# Characters added to every first prefix, so short budgets read whole words.
SLACK = 1024
# Words starting this close to a cut are dropped with the last word: a split
# pattern or an added token may join them across it.
LOOKBACK = 16
# A run with no cut point within this many characters is cut inside the run.
SEARCH = 4096
# Inside a run, the tokens kept end this many characters before the cut.
RUN_MARGIN = 1024
# Normalizers that compose characters, and how many code points become one.
COMPOSING = ('"NFC"', '"NFKC"', '"Precompiled"')
COMPOSED = 3
CUT_CLASSES = ("space", "whitespace", "punctuation", "symbol", "cjk")
CJK = (
    "\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7a3\uf900-\ufaff"
    "\U00020000-\U0002fa1f"
)
HANGUL_JAMO = ((0x1100, 0x11FF), (0xA960, 0xA97F), (0xD7B0, 0xD7FF))
# Cut points of each class in multilingual text, code and markup; full-width
# letters and punctuation are escaped.
PROBES = (
    "The quick  brown fox, jumps over the lazy dog's back; it's 3.14 or 42!",
    "line one\nline two\r\n\tindented   text\n\n  spaced  out  \u3000full\u3000width",
    "Ünïcödé café naïve e\u0301 a\u0300\u0323 ﬁ ½ \uff46\uff55\uff4c\uff4c ＡＢＣ①",
    "中文 混合 English 日本語の文章 한국어 문장 ภาษาไทย مرحبا",
    "我们的模型读取输入\uff0c并给出答案。它不会截断\uff1a“引号”和\uff08括号\uff09都保留\uff01"
    "日本語の文章です。カタカナとひらがな、漢字。한국어 문장입니다.",
    "emoji 👩‍👩‍👧‍👦 🏳️‍🌈 👍🏽 🚀 and <s> </s> [CLS] [SEP] <bos> <eos> <unk> <mask>",
    "def f(x):  return {'k': [x ** 2 for x in range(10)]}  # comment\n\tif a<=b: c+=1",
    "https://example.com/a?b=1&c=two  user@example.com  +1 (555) 010-0199",
    'price: $42.50 + €3 = 45.5%; a/b|c ~ d^e `code` <tag attr="v">x</tag> --- ***',
)
# Unbroken runs, for the cut inside a run (between the probe texts).
RUNS = (
    "QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVowMTIzNDU2Nzg5YWJjZGVmZ2hpamts" * 40,
    "iVBORw0KGgoAAAANSUhEUgAA+/8DAAAABJRU5ErkJggg/9j/4AAQSkZJRgAB+w==" * 40,
    "0123456789" * 260,
    "deadbeef0badf00d" * 160,
    "abcdefghijklmnopqrstuvwxyz" * 100,
    "我们的模型读取输入并给出答案它不会截断引号和括号都保留" * 100,
    "日本語の文章ですカタカナとひらがな漢字한국어문장입니다" * 100,
)
# Texts a tokenizer that drops or folds characters turns into fewer tokens than
# their length over the longest vocabulary entry.
FOLDS = tuple(
    char * 600
    for char in (
        " ",
        "\n",
        "\t",
        "\u3000",
        "\u00a0",
        "\u200b",
        "\u200d",
        "\ufeff",
        "\u00ad",
        "\ufe0f",
        "\x00",
        "\x07",
        "\x7f",
        "a",
        "é",
        "\u0301",
        "\ue000",
        "\U0001f600",
        "\u1100\u1161\u11a8",
    )
)


@dataclass(frozen=True)
class Read:
    """The content encoding of ``text[:end]``; its first ``tokens`` tokens are the whole text's first tokens.

    ``complete``: the whole text was tokenized (``tokens`` is all of it).
    """

    encoding: Any
    tokens: int
    complete: bool
    end: int


@dataclass(frozen=True)
class Facts:
    """Which cuts keep one tokenizer's tokens exact, found on probe texts.

    ``cuts``: the cut classes whose words before the cut are exact;
    ``in_run``: a cut inside a run keeps the tokens ``RUN_MARGIN`` characters
    before it; ``lookback``: the characters before a cut whose words are
    dropped (at least the longest added token); ``chars_per_token``: the most
    characters one token covers (None: the tokenizer drops or folds
    characters).
    """

    cuts: frozenset[str] = frozenset()
    in_run: bool = False
    lookback: int = LOOKBACK
    chars_per_token: int | None = None


WHOLE = Facts()

_facts: dict[int, tuple[Any, Facts]] = {}
_patterns: dict[frozenset[str], re.Pattern[str]] = {}
_lock = threading.Lock()
_classes: dict[str, str] = {}


def classes() -> dict[str, str]:
    """Each cut class as a regular-expression character set (built once)."""
    with _lock:
        if not _classes:
            punctuation, symbol = [], []
            for code in range(0x110000):
                kind = unicodedata.category(chr(code))[0]
                if kind == "P":
                    punctuation.append(code)
                elif kind == "S":
                    symbol.append(code)
            _classes.update(
                space=" ",
                whitespace=r"\s",
                punctuation=_ranges(punctuation),
                symbol=_ranges(symbol),
                cjk=CJK,
            )
        return _classes


def _ranges(codes: list[int]) -> str:
    out, start = [], 0
    for index, code in enumerate(codes):
        if index + 1 < len(codes) and codes[index + 1] == code + 1:
            continue
        first = codes[start]
        out.append(
            re.escape(chr(first))
            if first == code
            else f"{re.escape(chr(first))}-{re.escape(chr(code))}"
        )
        start = index + 1
    return "".join(out)


def class_of(char: str) -> str | None:
    """The cut class a character belongs to, or None."""
    for name, chars in classes().items():
        if re.fullmatch(f"[{chars}]", char):
            return name
    return None


def pattern(cuts: frozenset[str]) -> re.Pattern[str]:
    """One character set matching every cut point of ``cuts``."""
    compiled = _patterns.get(cuts)
    if compiled is None:
        sets = classes()
        compiled = re.compile("[" + "".join(sets[name] for name in sorted(cuts)) + "]")
        with _lock:
            _patterns[cuts] = compiled
    return compiled


def splits_run(char: str) -> bool:
    """Whether a cut before ``char`` leaves the text's normalization alone (NFC never composes it with what precedes)."""
    if unicodedata.category(char)[0] == "M" or unicodedata.combining(char):
        return False
    code = ord(char)
    return not any(low <= code <= high for low, high in HANGUL_JAMO)


def kept(encoding: Any, cut: int, lookback: int = LOOKBACK) -> int:
    """Tokens before the encoding's last word and before any word starting within ``lookback`` of ``cut``."""
    words, offsets = encoding.word_ids, encoding.offsets
    index = len(words)
    if not index:
        return 0
    last = words[-1]
    while index:
        word, first = words[index - 1], index - 1
        while first and words[first - 1] == word:
            first -= 1
        if word != last and offsets[first][0] < cut - lookback:
            return index
        index = first
    return 0


def ending_before(encoding: Any, limit: int) -> int:
    """Tokens that end at or before character ``limit``."""
    return bisect.bisect_right([end for _, end in encoding.offsets], limit)


def facts(tokenizer: Any) -> Facts:
    """Which cuts keep ``tokenizer``'s tokens exact, probed once and cached."""
    entry = _facts.get(id(tokenizer))
    if entry is not None and entry[0] is tokenizer:
        return entry[1]
    try:
        result = _probe(tokenizer)
    except Exception:
        result = WHOLE
    with _lock:
        _facts[id(tokenizer)] = (tokenizer, result)
    return result


def _encode(tokenizer: Any, text: str) -> Any:
    return tokenizer.encode(text, add_special_tokens=False)


def _probe(tokenizer: Any) -> Facts:
    added = [token.content for token in tokenizer.get_added_tokens_decoder().values()]
    lookback = max([LOOKBACK, *(len(content) + 1 for content in added)])
    cuts = set(CUT_CLASSES)
    # Every cut is probed with every kind of text after it, so a normalizer or
    # split pattern that looks ahead past the cut fails it; the second order
    # leaves out added-token text, which a tokenizer normalizes around.
    around = " ".join(PROBES)
    plain = [text for text in PROBES if not any(token in text for token in added)]
    for text in (around, " ".join(reversed(plain))):
        whole = _encode(tokenizer, text).ids
        for cut, char in enumerate(text):
            name = class_of(char) if cut else None
            if name not in cuts:
                continue
            part = _encode(tokenizer, text[:cut])
            count = kept(part, cut, lookback)
            if part.ids[:count] != whole[:count]:
                cuts.discard(name)
    chars = _chars_per_token(tokenizer)
    in_run = chars is not None
    for run in RUNS if in_run else ():
        text = f"{around} {run} {around}"
        whole = _encode(tokenizer, text).ids
        for cut in range(len(around) + RUN_MARGIN + 512, len(around) + len(run), 397):
            part = _encode(tokenizer, text[:cut])
            count = ending_before(part, cut - RUN_MARGIN)
            if not count or part.ids[:count] != whole[:count]:
                in_run = False
    return Facts(frozenset(cuts), in_run, lookback, chars)


def _chars_per_token(tokenizer: Any) -> int | None:
    longest = max(len(token) for token in tokenizer.get_vocab(with_added_tokens=True))
    normalizer = tokenizer.normalizer
    state = "" if normalizer is None else normalizer.__getstate__().decode()
    chars = longest * (COMPOSED if any(name in state for name in COMPOSING) else 1)
    for text in FOLDS:
        if len(_encode(tokenizer, text).ids) * chars < len(text):
            return None
    return chars


def reach(need: int) -> int:
    """Characters of the first prefix for ``need`` tokens; a text this short is tokenized whole."""
    return max(1, need) * CHARS_PER_TOKEN + SLACK


def surely_over(tokenizer: Any, text: str, limit: int) -> bool:
    """Whether ``text`` certainly has more than ``limit`` tokens, decided without tokenizing it."""
    if len(text) <= reach(limit):
        return False
    chars = facts(tokenizer).chars_per_token
    return chars is not None and len(text) > max(0, limit) * chars


def _cut(
    text: str, start: int, cuts: re.Pattern[str] | None, in_run: bool
) -> tuple[int, bool]:
    """The next cut at or after ``start``, and whether it is a cut point (else a cut inside a run); -1 for none."""
    if cuts is not None:
        found = cuts.search(text, start, start + SEARCH)
        if found is not None:
            return found.start(), True
    if in_run:
        for index in range(start, min(len(text), start + SEARCH)):
            if splits_run(text[index]):
                return index, False
    if cuts is not None:
        found = cuts.search(text, start + SEARCH)
        if found is not None:
            return found.start(), True
    return -1, False


def read(tokenizer: Any, text: str, need: int) -> Read:
    """``text``'s content encoding, or a prefix's that keeps at least ``need`` of the whole text's first tokens."""
    need = max(1, need)
    start = reach(need)
    known = facts(tokenizer) if len(text) > start else WHOLE
    cuts = pattern(known.cuts) if known.cuts else None
    while (cuts is not None or known.in_run) and start < len(text):
        cut, point = _cut(text, start, cuts, known.in_run)
        if cut < 0:
            break
        encoding = _encode(tokenizer, text[:cut])
        count = kept(encoding, cut, known.lookback) if point else 0
        if count < need and known.in_run and splits_run(text[cut]):
            count = max(count, ending_before(encoding, cut - RUN_MARGIN))
        if count >= need:
            return Read(encoding, count, False, cut)
        start = 2 * cut
    encoding = _encode(tokenizer, text)
    return Read(encoding, len(encoding.ids), True, len(text))


def read_batch(tokenizer: Any, texts: Sequence[str], need: int) -> list[Read]:
    """``read`` of every text; the texts short enough to read whole go through one ``encode_batch``."""
    whole = [index for index, text in enumerate(texts) if len(text) <= reach(need)]
    encodings = tokenizer.encode_batch(
        [texts[index] for index in whole], add_special_tokens=False
    )
    out = {
        index: Read(encoding, len(encoding.ids), True, len(texts[index]))
        for index, encoding in zip(whole, encodings, strict=True)
    }
    return [
        out[index] if index in out else read(tokenizer, text, need)
        for index, text in enumerate(texts)
    ]
