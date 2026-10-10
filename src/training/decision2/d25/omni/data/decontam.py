"""Text and image decontamination of Omni training rows (``omni/SPEC.md``, Decontamination).

Text (the Vega rule, ``vega/SPEC.md``, as calibrated by Vega's index v4): NFKC + casefold + ``\\w+``
tokens. A row is dropped when one of its text units (state, question with option texts, and every text
rendered into its images, kept in ``meta.image_text``) equals a protected unit, when its option set equals
a protected option set (>= 10 tokens, >= 6 alphabetic), or when it shares any non-boilerplate 13-gram with
a protected item; ``calibrate_text`` reports recall on planted items and the clean false-positive rate.
13-grams that occur in more than 50 protected items are boilerplate and ignored. Protected items are the
Vega text suite 0.3, the vision suite rows, every proxy row (Vega's protected proxy items included) and
the extra benchmark text pools (all splits of the board benchmarks). Exact matching needs at least
``EXACT_MIN_TOKENS`` tokens so that empty or one-word states do not collide.

Images: a row is dropped when any image matches the eval image bank (``bank.parquet`` columns
``sha256, phash64, dhash64, ...``) by sha256 of the stored or the original bytes, by perceptual hash
within the calibrated Hamming rule, or by SSCD cosine >= 0.95 once vectors exist. Hashes follow the
``imagehash`` definitions bit for bit (grayscale ``L``, Lanczos resize, 8x8 DCT low band against its
median; 9x8 horizontal gradient), packed row-major with the first bit most significant. The query side
also hashes a copy with uniform borders trimmed, which absorbs padding. ``calibrate_images`` plants
resized, re-encoded, cropped and padded copies of bank images and picks the loosest-needed rule that
catches all of them, and reports the false-positive rate on a clean sample.

Sources listed in ``shared/holdouts.json`` or ``omni/proxy-holdouts.json`` are dropped by registry
identity (whole dataset, split or hashed row slice), proxy item ids and image sha256.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import random
import re
import sys
import unicodedata
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

TOKEN = re.compile(r"\w+")
GRAM = 13
BOILERPLATE_ITEMS = 50
EXACT_MIN_TOKENS = 5
ANY_GRAM = 1e-9
SSCD_THRESHOLD = 0.95
MASK64 = (1 << 64) - 1


# ---------------------------------------------------------------- text


def tokens(text: str) -> list[str]:
    return TOKEN.findall(unicodedata.normalize("NFKC", text).casefold())


def describe(value: Any) -> str:
    if value is None:
        return ""
    return value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)


def question_text(question: dict[str, Any]) -> str:
    parts = [describe(question.get("instructions"))]
    criteria = question.get("criteria")
    if isinstance(criteria, dict):
        parts += [describe(value) for value in criteria.values() if value is not None]
    return "\n".join(part for part in parts if part)


def option_key(question: dict[str, Any] | None) -> int | None:
    """Order-free key of a question's option texts (Vega v4: >= 10 tokens, >= 6 alphabetic), else None."""
    criteria = (question or {}).get("criteria")
    if not isinstance(criteria, dict) or any(
        value is None for value in criteria.values()
    ):
        return None
    parts = sorted(" ".join(tokens(describe(value))) for value in criteria.values())
    words = " ".join(parts).split()
    if len(words) < 10 or sum(1 for w in words if w.isalpha()) < 6:
        return None
    return hash("\x1f".join(parts))


def protected_units(row: dict[str, Any]) -> Iterator[tuple[list[str], int | None]]:
    """(text units, option key) of one kit-format row (suite, proxy or benchmark pool), per question."""
    state = describe(row.get("state"))
    for question in (row.get("questions") or {}).values():
        yield [state, question_text(question)], option_key(question)
    if not row.get("questions") and row.get("question"):
        yield [state, question_text(row["question"])], option_key(row["question"])
    for extra in row.get("texts") or []:
        yield [describe(extra)], None


def training_units(row: dict[str, Any]) -> list[str]:
    units = [describe(row.get("state")), question_text(row["question"])]
    units += [
        describe(text) for text in (row.get("meta") or {}).get("image_text") or []
    ]
    return [unit for unit in units if unit]


def _grams(words: list[str]) -> set[int]:
    return {hash(" ".join(words[i : i + GRAM])) for i in range(len(words) - GRAM + 1)}


@dataclass
class TextIndex:
    exact: set[int] = field(default_factory=set)
    options: set[int] = field(default_factory=set)
    counts: Counter = field(default_factory=Counter)
    grams: frozenset[int] = frozenset()
    boilerplate: int = 0
    items: int = 0

    def add(self, units: Iterable[str], options: int | None = None) -> None:
        seen: set[int] = set()
        for unit in units:
            words = tokens(unit)
            if len(words) >= EXACT_MIN_TOKENS:
                self.exact.add(hash(" ".join(words)))
            seen |= _grams(words)
        if options is not None:
            self.options.add(options)
        self.counts.update(seen)
        self.items += 1

    def finalize(self) -> "TextIndex":
        self.grams = frozenset(
            g for g, n in self.counts.items() if n <= BOILERPLATE_ITEMS
        )
        self.boilerplate = len(self.counts) - len(self.grams)
        self.counts = Counter()
        return self

    def score(
        self, units: Iterable[str], options: int | None = None
    ) -> tuple[bool, float]:
        """(exact unit or option-set hit, highest share of non-boilerplate protected 13-grams over the units)."""
        exact, coverage = options is not None and options in self.options, 0.0
        for unit in units:
            words = tokens(unit)
            if len(words) >= EXACT_MIN_TOKENS and hash(" ".join(words)) in self.exact:
                exact = True
            grams = _grams(words)
            if grams:
                coverage = max(coverage, len(grams & self.grams) / len(grams))
        return exact, coverage


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def build_text_index(paths: Iterable[str | Path]) -> tuple[TextIndex, dict[str, int]]:
    index, per_file = TextIndex(), {}
    for path in paths:
        before = index.items
        for row in read_jsonl(path):
            for units, options in protected_units(row):
                index.add(units, options)
        per_file[str(path)] = index.items - before
    return index.finalize(), per_file


def plant_text(units: list[str], rng: random.Random) -> str:
    """Training-side disguises of a protected item: verbatim, re-cased, re-punctuated, wrapped, cut."""
    text = max(units, key=lambda unit: len(tokens(unit)))
    words = text.split()
    cut = " ".join(words[: max(GRAM + 1, int(len(words) * 0.6))])
    return [
        text,
        text.upper(),
        re.sub(r"\s+", "  ", text).replace(",", " ,").replace(".", " . "),
        f"Read the problem shown in the picture. {text} Pick the single best option.",
        cut,
    ][rng.randrange(5)]


def calibrate_text(
    index: TextIndex,
    protected_rows: list[dict[str, Any]],
    clean_units: list[list[str]],
    n_planted: int = 2000,
    seed: int = 20261009,
) -> dict[str, Any]:
    """Vega v4 rule (any non-boilerplate 13-gram, exact unit or option set): planted recall and clean FP."""
    rng = random.Random(seed)
    pool = [units for row in protected_rows for units, _ in protected_units(row)]
    planted = [
        plant_text(units, rng) for units in rng.sample(pool, min(n_planted, len(pool)))
    ]
    scores = [index.score([text]) for text in planted]
    long_unmatched = [
        cov
        for (exact, cov), text in zip(scores, planted)
        if not exact and len(tokens(text)) >= GRAM and cov > 0
    ]
    threshold = ANY_GRAM
    missed = [
        len(tokens(text)) >= GRAM
        for (exact, cov), text in zip(scores, planted)
        if not exact and cov < threshold
    ]
    clean = [index.score(units) for units in clean_units]
    flagged = sum(1 for exact, cov in clean if exact or cov >= threshold)
    return {
        "rule": "exact unit (>= %d tokens), exact option set, or any non-boilerplate 13-gram"
        % EXACT_MIN_TOKENS,
        "threshold": round(threshold, 4),
        "planted": len(planted),
        "planted_caught": len(planted) - len(missed),
        "planted_missed_short": sum(1 for is_long in missed if not is_long),
        "planted_missed_boilerplate_only": sum(1 for is_long in missed if is_long),
        "min_planted_coverage": (
            round(min(long_unmatched), 4) if long_unmatched else None
        ),
        "clean_sample": len(clean),
        "clean_flagged": flagged,
        "false_positive_rate": round(flagged / max(1, len(clean)), 6),
        "protected_items": index.items,
        "protected_grams": len(index.grams),
        "boilerplate_grams": index.boilerplate,
    }


# ---------------------------------------------------------------- images


def _pack(bits: np.ndarray) -> int:
    return int.from_bytes(np.packbits(bits.astype(np.uint8).flatten()).tobytes(), "big")


def phash64(image) -> int:
    """``imagehash.phash`` (hash_size 8, highfreq_factor 4) as an unsigned 64-bit integer."""
    import scipy.fftpack
    from PIL import Image

    pixels = np.asarray(image.convert("L").resize((32, 32), Image.Resampling.LANCZOS))
    dct = scipy.fftpack.dct(scipy.fftpack.dct(pixels, axis=0), axis=1)
    low = dct[:8, :8]
    return _pack(low > np.median(low))


def dhash64(image) -> int:
    """``imagehash.dhash`` (hash_size 8) as an unsigned 64-bit integer."""
    from PIL import Image

    pixels = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.LANCZOS))
    return _pack(pixels[:, 1:] > pixels[:, :-1])


def trim_borders(
    image, tolerance: int = 28, share: float = 0.97, min_keep: float = 0.5
):
    """Copy of ``image`` without near-uniform borders (robust to JPEG ringing; at most half a side)."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
    height, width = rgb.shape[:2]
    if height < 16 or width < 16:
        return image

    def uniform(line: np.ndarray, color: np.ndarray) -> bool:
        return bool((np.abs(line - color).max(axis=-1) <= tolerance).mean() >= share)

    top, bottom, left, right = 0, height, 0, width
    color = np.median(rgb[0], axis=0)
    while top < height * min_keep and uniform(rgb[top], color):
        top += 1
    color = np.median(rgb[-1], axis=0)
    while bottom > height * (1 - min_keep) and uniform(rgb[bottom - 1], color):
        bottom -= 1
    color = np.median(rgb[:, 0], axis=0)
    while left < width * min_keep and uniform(rgb[top:bottom, left], color):
        left += 1
    color = np.median(rgb[:, -1], axis=0)
    while right > width * (1 - min_keep) and uniform(rgb[top:bottom, right - 1], color):
        right -= 1
    if (
        (top, bottom, left, right) == (0, height, 0, width)
        or bottom - top < 8
        or right - left < 8
    ):
        return image
    return image.crop((left, top, right, bottom))


def image_hashes(image) -> list[tuple[int, int]]:
    """(phash64, dhash64) of the image and, when it differs, of its border-trimmed copy."""
    out = [(phash64(image), dhash64(image))]
    trimmed = trim_borders(image)
    if trimmed is not image:
        out.append((phash64(trimmed), dhash64(trimmed)))
    return out


BANK_CROPS: dict[str, tuple[float, float, float, float]] = {
    "c03": (0.03, 0.03, 0.03, 0.03),
    "c06": (0.06, 0.06, 0.06, 0.06),
    "l07": (0.07, 0, 0, 0),
    "t07": (0, 0.07, 0, 0),
    "r07": (0, 0, 0.07, 0),
    "b07": (0, 0, 0, 0.07),
    "l13": (0.13, 0, 0, 0),
    "t13": (0, 0.13, 0, 0),
    "r13": (0, 0, 0.13, 0),
    "b13": (0, 0, 0, 0.13),
    "lt05": (0.05, 0.05, 0, 0),
    "rt05": (0, 0.05, 0.05, 0),
    "lb05": (0.05, 0, 0, 0.05),
    "rb05": (0, 0, 0.05, 0.05),
}


def bank_variants(
    image, crops: dict[str, tuple[float, float, float, float]] = BANK_CROPS
):
    """Bank-side hashes: the full image plus fixed crops (fractions of left, top, right, bottom)."""
    rgb = image.convert("RGB")
    width, height = rgb.size
    out = [("full", phash64(rgb), dhash64(rgb))]
    for name, (left, top, right, bottom) in crops.items():
        box = (
            int(width * left),
            int(height * top),
            width - int(width * right),
            height - int(height * bottom),
        )
        if box[2] - box[0] >= 16 and box[3] - box[1] >= 16:
            part = rgb.crop(box)
            out.append((name, phash64(part), dhash64(part)))
    return out


def as_u64(value: Any) -> int:
    if isinstance(value, (bytes, bytearray)):
        return int.from_bytes(value, "big")
    if isinstance(value, str):
        return int(value, 16)
    return int(value) & MASK64


@dataclass
class Rule:
    """Match when phash distance <= ``phash`` and dhash distance <= ``dhash``, or phash <= ``strict``."""

    phash: int = 10
    dhash: int = 14
    strict: int = 4

    def match(self, pd: np.ndarray, dd: np.ndarray) -> np.ndarray:
        return ((pd <= self.phash) & (dd <= self.dhash)) | (pd <= self.strict)


class ImageIndex:
    """The eval image bank: sha256 set plus phash/dhash arrays for vectorised Hamming search."""

    def __init__(self, records: list[dict[str, Any]]):
        self.records = records
        self.sha = {r["sha256"] for r in records if r.get("sha256")}
        self.phash = np.array([as_u64(r["phash64"]) for r in records], dtype=np.uint64)
        self.dhash = np.array([as_u64(r["dhash64"]) for r in records], dtype=np.uint64)

    @classmethod
    def from_parquet(cls, paths: Iterable[str | Path]) -> "ImageIndex":
        import pyarrow.parquet as pq

        records: list[dict[str, Any]] = []
        for path in paths:
            table = pq.read_table(path)
            records += table.to_pylist()
        return cls(records)

    def __len__(self) -> int:
        return len(self.records)

    def distances(self, ph: int, dh: int) -> tuple[np.ndarray, np.ndarray]:
        pd = np.bitwise_count(self.phash ^ np.uint64(ph))
        dd = np.bitwise_count(self.dhash ^ np.uint64(dh))
        return pd, dd

    def frontier(self, hashes: list[tuple[int, int]]) -> np.ndarray:
        """``f[p]`` = smallest dhash distance among entries within phash distance ``p`` (99: none)."""
        out = np.full(65, 99, dtype=np.int16)
        for ph, dh in hashes:
            pd, dd = self.distances(ph, dh)
            best = np.full(65, 99, dtype=np.int16)
            np.minimum.at(best, pd.astype(np.int64), dd.astype(np.int16))
            out = np.minimum(out, np.minimum.accumulate(best))
        return out

    def match(self, hashes: list[tuple[int, int]], rule: Rule) -> int:
        """Bank row matched by any hash variant under ``rule``, or -1."""
        if not len(self):
            return -1
        for ph, dh in hashes:
            pd, dd = self.distances(ph, dh)
            hit = np.flatnonzero(rule.match(pd, dd))
            if hit.size:
                return int(hit[0])
        return -1


def _variants(image, rng: random.Random) -> list[tuple[str, Any]]:
    """Planted copies: resized, re-encoded, cropped, padded and combined edits."""
    from PIL import Image, ImageOps

    def jpeg(img, quality):
        buffer = io.BytesIO()
        img.convert("RGB").save(buffer, format="JPEG", quality=quality)
        return Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")

    def resize(img, scale):
        w, h = img.size
        return img.resize(
            (max(16, int(w * scale)), max(16, int(h * scale))), Image.Resampling.BICUBIC
        )

    def crop(img, fractions):
        w, h = img.size
        left, top, right, bottom = fractions
        return img.crop(
            (int(w * left), int(h * top), w - int(w * right), h - int(h * bottom))
        )

    def centre():
        f = rng.uniform(0.01, 0.07)
        return (f, f, f, f)

    def one_side():
        sides = [0.0, 0.0, 0.0, 0.0]
        sides[rng.randrange(4)] = rng.uniform(0.03, 0.15)
        return tuple(sides)

    def corner():
        sides = [0.0, 0.0, 0.0, 0.0]
        k = rng.randrange(4)
        sides[k] = rng.uniform(0.02, 0.07)
        sides[(k + 1) % 4] = rng.uniform(0.02, 0.07)
        return tuple(sides)

    def pad(img, frac, color):
        w, h = img.size
        border = (int(w * frac), int(h * frac))
        return ImageOps.expand(img.convert("RGB"), border=border, fill=color)

    def letterbox(img, color):
        w, h = img.size
        side = max(w, h)
        canvas = Image.new("RGB", (side, side), color)
        canvas.paste(img.convert("RGB"), ((side - w) // 2, (side - h) // 2))
        return canvas

    def cap(img):
        w, h = img.size
        if w * h <= 1_638_400:
            return img
        s = (1_638_400 / (w * h)) ** 0.5
        return img.resize((int(w * s), int(h * s)), Image.Resampling.LANCZOS)

    base = image.convert("RGB")
    color = rng.choice([(255, 255, 255), (0, 0, 0), (128, 128, 128)])
    return [
        ("resize_0.5", resize(base, 0.5)),
        ("resize_1.5_cap", cap(resize(base, 1.5))),
        ("jpeg_q40", jpeg(base, 40)),
        ("jpeg_q75_resize_0.75", jpeg(resize(base, 0.75), 75)),
        ("crop_centre_1_7pct", crop(base, centre())),
        ("crop_one_side_3_15pct", crop(base, one_side())),
        ("crop_corner_2_7pct_jpeg", jpeg(crop(base, corner()), 85)),
        ("pad_10pct", pad(base, 0.10, color)),
        ("letterbox", letterbox(base, color)),
        ("pad_5pct_resize_jpeg", jpeg(resize(pad(base, 0.05, color), 0.8), 80)),
    ]


def calibrate_images(
    planted_images: list[Any],
    clean_images: list[Any],
    bank: ImageIndex | None = None,
    seed: int = 20261009,
    crops: dict[str, tuple[float, float, float, float]] | None = None,
) -> dict[str, Any]:
    """Pick the Hamming rule that catches every planted copy with the fewest clean false positives.

    Planted originals enter the reference as bank entries with their crop variants (``crops``,
    default ``BANK_CROPS``); false positives are measured against those entries plus ``bank``.
    """
    rng = random.Random(seed)
    grid = BANK_CROPS if crops is None else crops
    originals = [
        [
            {"sha256": "", "phash64": ph, "dhash64": dh, "variant": v}
            for v, ph, dh in bank_variants(img, grid)
        ]
        for img in planted_images
    ]
    records = [record for entries in originals for record in entries]
    if bank is not None:
        records += bank.records
    reference = ImageIndex(records)
    names: list[str] = []
    planted: list[np.ndarray] = []
    for entries, image in zip(originals, planted_images):
        own = ImageIndex(entries)
        for name, variant in _variants(image, rng):
            names.append(name)
            planted.append(own.frontier(image_hashes(variant)))
    planted_f = np.stack(planted)
    clean_f = np.stack([reference.frontier(image_hashes(img)) for img in clean_images])

    def caught(frontiers: np.ndarray, rule: Rule) -> np.ndarray:
        return (frontiers[:, rule.phash] <= rule.dhash) | (
            frontiers[:, rule.strict] < 99
        )

    best: tuple[int, int, Rule] | None = None
    for strict in range(0, 9):
        loose = planted_f[:, strict] >= 99
        for ph in range(strict, 33):
            if not loose.any():
                need = 0
            else:
                values = planted_f[loose, ph]
                if (values >= 99).any():
                    continue
                need = int(values.max())
            if need > 32:
                continue
            rule = Rule(ph, need, strict)
            fp = int(caught(clean_f, rule).sum())
            if best is None or (fp, ph + need) < best[:2]:
                best = (fp, ph + need, rule)
    rule = best[2] if best else Rule(32, 32, 8)
    fp = int(caught(clean_f, rule).sum())
    hit = caught(planted_f, rule)
    per_variant: dict[str, dict[str, int]] = {}
    for name, frontier, ok in zip(names, planted_f, hit):
        stats = per_variant.setdefault(name, {"n": 0, "caught": 0, "max_min_phash": 0})
        stats["n"] += 1
        stats["caught"] += int(ok)
        stats["max_min_phash"] = max(
            stats["max_min_phash"], int(np.argmax(frontier < 99))
        )
    return {
        "rule": {"phash": rule.phash, "dhash": rule.dhash, "strict": rule.strict},
        "planted_originals": len(planted_images),
        "planted_copies": len(planted),
        "planted_caught": int(hit.sum()),
        "per_variant": per_variant,
        "reference_entries": len(reference),
        "clean_sample": len(clean_f),
        "clean_flagged": fp,
        "false_positive_rate": round(fp / max(1, len(clean_f)), 6),
    }


def sscd_matches(
    train: np.ndarray, bank: np.ndarray, threshold: float = SSCD_THRESHOLD
) -> np.ndarray:
    """Indices of training vectors with cosine >= ``threshold`` to any bank vector (SSCD descriptors)."""
    if train.size == 0 or bank.size == 0:
        return np.zeros(0, dtype=np.int64)
    bank = bank / np.linalg.norm(bank, axis=1, keepdims=True)
    hits = []
    for start in range(0, len(train), 4096):
        chunk = train[start : start + 4096]
        chunk = chunk / np.linalg.norm(chunk, axis=1, keepdims=True)
        best = (chunk @ bank.T).max(axis=1)
        hits.append(np.flatnonzero(best >= threshold) + start)
    return np.concatenate(hits)


# ---------------------------------------------------------------- holdouts


def slice_selected(name: str, key: Any, mod: int, keep: int) -> bool:
    """The shared holdouts.json row-slice rule."""
    digest = hashlib.sha256(f"d25-proxy-v1:{name}:{key}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % mod < keep


@dataclass
class Holdouts:
    """Exclusions from ``shared/holdouts.json`` and ``omni/proxy-holdouts.json``."""

    whole: set[str] = field(default_factory=set)
    splits: dict[str, set[str]] = field(default_factory=dict)
    slices: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    item_ids: set[str] = field(default_factory=set)
    image_sha256: set[str] = field(default_factory=set)
    files: dict[str, str] = field(default_factory=dict)

    @staticmethod
    def _norm(name: str) -> str:
        return name.strip().lower()

    def add_dataset(self, entry: dict[str, Any]) -> None:
        names = [
            entry.get("hf_id") or entry.get("source") or entry.get("dataset") or ""
        ]
        names += entry.get("aliases") or []
        scope = entry.get("scope", "all")
        for name in filter(None, names):
            name = self._norm(name)
            if scope == "splits":
                self.splits.setdefault(name, set()).update(entry.get("splits") or [])
            elif scope == "rows":
                self.slices.setdefault(name, []).append(
                    {"splits": set(entry.get("splits") or []), **entry["slice"]}
                )
            elif scope != "generator":
                self.whole.add(name)

    @classmethod
    def load(cls, paths: Iterable[str | Path]) -> "Holdouts":
        out = cls()
        for path in paths:
            path = Path(path)
            if not path.exists():
                out.files[str(path)] = "missing"
                continue
            data = json.loads(path.read_text())
            out.files[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            for entry in data.get("datasets") or data.get("sources") or []:
                out.add_dataset(entry if isinstance(entry, dict) else {"hf_id": entry})
            out.item_ids.update(str(i) for i in data.get("ids") or [])
            out.image_sha256.update(data.get("image_sha256") or [])
            for proxy in data.get("proxies") or []:
                for entry in proxy.get("sources") or []:
                    out.add_dataset(
                        entry if isinstance(entry, dict) else {"hf_id": entry}
                    )
                out.item_ids.update(str(i) for i in proxy.get("ids") or [])
                out.image_sha256.update(proxy.get("image_sha256") or [])
        return out

    def source_dropped(self, identities: Iterable[str]) -> bool:
        return any(self._norm(name) in self.whole for name in identities)

    def row_dropped(self, row: dict[str, Any], identities: Iterable[str]) -> str | None:
        meta = row.get("meta") or {}
        names = [self._norm(n) for n in identities]
        if any(n in self.whole for n in names):
            return "holdout_source"
        split = meta.get("source_split")
        if any(split in self.splits.get(n, ()) for n in names):
            return "holdout_split"
        for n in names:
            for rule in self.slices.get(n, []):
                if (not rule["splits"] or split in rule["splits"]) and slice_selected(
                    rule["name"], meta.get("source_key"), rule["mod"], rule["keep"]
                ):
                    return "holdout_rows"
        if str(meta.get("source_id")) in self.item_ids:
            return "holdout_item"
        digests = set(meta.get("image_sha256") or []) | set(
            meta.get("image_orig_sha256") or []
        )
        if digests & self.image_sha256:
            return "holdout_image"
        return None


# ---------------------------------------------------------------- row filter


@dataclass
class Decontaminator:
    text: TextIndex
    text_threshold: float
    images: ImageIndex
    rule: Rule
    holdouts: Holdouts
    root: Path

    def check(
        self, row: dict[str, Any], identities: list[str] | None = None
    ) -> tuple[str | None, dict[str, Any]]:
        """(drop reason or None, evidence) for one training row; ``identities`` name its sources."""
        from PIL import Image

        reason = self.holdouts.row_dropped(row, identities or [row["source"]])
        if reason:
            return reason, {}
        exact, coverage = self.text.score(
            training_units(row), option_key(row.get("question"))
        )
        if exact:
            return "text_exact", {"coverage": coverage}
        if coverage >= self.text_threshold:
            return "text_13gram", {"coverage": round(coverage, 4)}
        meta = row.get("meta") or {}
        digests = list(meta.get("image_sha256") or []) + [
            d for d in meta.get("image_orig_sha256") or [] if d
        ]
        if any(d in self.images.sha for d in digests):
            return "image_sha256", {}
        for ref in row.get("images") or []:
            with Image.open(self.root / ref) as image:
                hashes = image_hashes(image)
            j = self.images.match(hashes, self.rule)
            if j >= 0:
                hit = self.images.records[j]
                return "image_phash", {
                    "bank": f"{hit.get('benchmark')}/{hit.get('split')}/{hit.get('item_id')}"
                }
        return None, {}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate or report Omni decontamination."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    hashes = sub.add_parser(
        "hash", help="print phash64/dhash64 of images (bank parity checks)"
    )
    hashes.add_argument("images", nargs="+")
    args = parser.parse_args(argv)
    if args.cmd == "hash":
        from PIL import Image

        for path in args.images:
            with Image.open(path) as image:
                print(
                    json.dumps(
                        {
                            "path": path,
                            "phash64": f"{phash64(image):016x}",
                            "dhash64": f"{dhash64(image):016x}",
                        }
                    )
                )


if __name__ == "__main__":
    sys.exit(main())
