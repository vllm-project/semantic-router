"""Tatoeba export reader, translation-link graph and PN1 candidate construction.

Prereg ``v2/data/records/m4-pn1-prereg-2026-09-29.md`` sec. 2. Every Tatoeba
sentence id enters at most one candidate group, in the priority
name > twin > hop > near:

- ``pn-name``: every usable sentence with a valid name swap (``pn1_text.name_swap``).
- ``pn-twin`` seeds: sentences passing the length rule, in ``sha256("pn1-twin:" + id)``
  order, up to the planned count.
- ``pn-hop``: pivots (any other-language sentence; English pivots first, then
  ``sha256("pn1-pivot:" + id)`` order) linked to at least two free sentences of the
  language. Each pivot gives at most one pair: the first pair (``sha256("pn1-hop:" + id)``
  order) that is not normalization-equal, has length ratio <= 1.6 and falls in an
  overlap bin >= [.3, .5) whose per-bin cap is not reached. Pairs below .3 have no
  near-miss partner stratum and are not kept.
- ``pn-near``: queries in ``sha256("pn1-near:" + id)`` order over the remaining
  sentences; partners come from the query's rarest character bigrams. A partner is
  valid at bigram Jaccard >= .3, length ratio <= 1.6, not normalization-equal, with
  disjoint two-hop link neighbourhoods and no normalization-equal English
  translations. Quotas are per fine cell (0.05 Jaccard step x length-ratio class x
  length class) and equal the hop pairs' cell counts, so the near pairs follow the
  hop pairs' overlap and length distribution.

Readers and the CSR graph import numpy lazily; everything else is standard library.
"""

from __future__ import annotations

import array
import bz2
import collections
import hashlib
import io
import tarfile
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

from v2.data.m4.pn1_text import (
    ISO3,
    LANGS,
    MAX_LENGTH_RATIO,
    NEAR_MIN_JACCARD,
    grams,
    name_swap,
    norm,
    pair_metrics,
    twin_eligible,
    usable,
)

PIVOT = "en"
EXPORT_LANGUAGES = {**{lang: ISO3[lang] for lang in LANGS}, PIVOT: "eng"}
PLANNED_SEEDS = {
    "ja": 2600,
    "ko": 2200,
    "zh": 2000,
    "es": 1800,
    "de": 1000,
    "fr": 1000,
    "ru": 700,
    "ar": 700,
}
NEAR_RARE_BIGRAMS = 6
NEAR_POSTINGS = 200
NEAR_MAX_QUERIES = 250_000
NEAR_TRIES = 5


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def export_files() -> dict[str, str]:
    """Relative path of every export file PN1 reads, keyed by a short name."""
    files = {
        "links": "links.tar.bz2",
        "sentences_CC0": "sentences_CC0.tar.bz2",
    }
    for iso3 in EXPORT_LANGUAGES.values():
        files[f"{iso3}_sentences_detailed"] = (
            f"per_language/{iso3}/{iso3}_sentences_detailed.tsv.bz2"
        )
    return files


def read_sentences(path: Path) -> Iterator[tuple[int, str, str, str]]:
    """(id, iso3 language, text, username) from a ``*_sentences_detailed.tsv.bz2`` file."""
    with bz2.open(path, "rt", encoding="utf-8", newline="\n") as stream:
        for number, line in enumerate(stream, 1):
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6:
                raise ValueError(f"{path}:{number}: expected 6 tab-separated fields")
            yield int(parts[0]), parts[1], "\t".join(parts[2:-3]), parts[-3]


def _tar_member(path: Path, suffix: str) -> bytes:
    with tarfile.open(path, "r:bz2") as archive:
        members = [
            m for m in archive.getmembers() if m.isfile() and m.name.endswith(suffix)
        ]
        if len(members) != 1:
            raise ValueError(
                f"{path}: expected one *{suffix} member, found {len(members)}"
            )
        return archive.extractfile(members[0]).read()


def read_cc0_ids(path: Path) -> set[int]:
    data = _tar_member(path, ".csv")
    ids = set()
    for line in io.StringIO(data.decode("utf-8")):
        head = line.split("\t", 1)[0].strip()
        if head:
            ids.add(int(head))
    return ids


def load_link_graph(path: Path) -> LinkGraph:
    """The undirected graph of ``links.tar.bz2`` (numpy CSR; pure Python without numpy)."""
    data = _tar_member(path, ".csv")
    fields = data.split()
    if len(fields) % 2:
        raise ValueError(f"{path}: odd number of link fields")
    try:
        import numpy as np
    except ImportError:
        values = [int(value) for value in fields]
        return LinkGraph.from_pairs(zip(values[0::2], values[1::2]))
    return LinkGraph.from_array(np.array(fields, dtype=np.int64).reshape(-1, 2))


class LinkGraph:
    """Undirected translation links, over a dict (tests) or CSR arrays (``from_array``)."""

    def __init__(self, adjacency: dict[int, list[int]] | None = None) -> None:
        self._adjacency = adjacency
        self._indptr = None
        self._indices = None

    @classmethod
    def from_pairs(cls, pairs: Iterable[tuple[int, int]]) -> LinkGraph:
        adjacency: dict[int, set[int]] = collections.defaultdict(set)
        for a, b in pairs:
            if a != b:
                adjacency[a].add(b)
                adjacency[b].add(a)
        return cls({key: sorted(value) for key, value in adjacency.items()})

    @classmethod
    def from_array(cls, links: Any) -> LinkGraph:
        import numpy as np

        links = links[links[:, 0] != links[:, 1]]
        if links.size and (links.min() < 0 or links.max() >= 1 << 31):
            raise ValueError("sentence ids must be in [0, 2**31)")
        keys = np.unique(
            np.concatenate(
                [(links[:, 0] << 32) | links[:, 1], (links[:, 1] << 32) | links[:, 0]]
            )
        )
        sources, targets = keys >> 32, keys & 0xFFFFFFFF
        size = int(sources.max()) + 1 if keys.size else 0
        graph = cls()
        graph._indptr = np.concatenate(
            [[0], np.cumsum(np.bincount(sources, minlength=size))]
        ).astype(np.int64)
        graph._indices = targets.astype(np.int64)
        return graph

    @property
    def edges(self) -> int:
        if self._adjacency is not None:
            return sum(len(v) for v in self._adjacency.values()) // 2
        return int(self._indices.size) // 2

    def neighbours(self, sid: int) -> list[int]:
        if self._adjacency is not None:
            return self._adjacency.get(sid, [])
        if sid < 0 or sid + 1 >= self._indptr.size:
            return []
        return self._indices[self._indptr[sid] : self._indptr[sid + 1]].tolist()

    def within(self, ids: Iterable[int], hops: int = 2) -> set[int]:
        """The ids plus every sentence within ``hops`` links, in any language."""
        seen = set(ids)
        frontier = set(seen)
        for _ in range(hops):
            nxt = set()
            for sid in frontier:
                nxt.update(self.neighbours(sid))
            frontier = nxt - seen
            seen |= frontier
            if not frontier:
                break
        return seen


class Corpus:
    """Sentences of the PN1 languages and the English pivot, plus links and CC0 ids."""

    def __init__(
        self,
        texts: dict[str, dict[int, str]],
        users: dict[int, str],
        cc0: set[int],
        graph: LinkGraph,
    ) -> None:
        self.texts = texts
        self.users = users
        self.cc0 = cc0
        self.graph = graph
        self.lang_of = {sid: lang for lang, table in texts.items() for sid in table}
        self._english_norms: dict[int, str] = {}

    @classmethod
    def load(cls, root: Path) -> Corpus:
        files = export_files()
        texts: dict[str, dict[int, str]] = {}
        users: dict[int, str] = {}
        for lang, iso3 in EXPORT_LANGUAGES.items():
            table = {}
            for sid, code, text, user in read_sentences(
                root / files[f"{iso3}_sentences_detailed"]
            ):
                if code != iso3:
                    raise ValueError(f"sentence {sid} is {code}, not {iso3}")
                table[sid] = text
                if lang != PIVOT:
                    users[sid] = user
            texts[lang] = table
        graph = load_link_graph(root / files["links"])
        return cls(texts, users, read_cc0_ids(root / files["sentences_CC0"]), graph)

    def english_norms(self, sid: int) -> set[str]:
        english = self.texts.get(PIVOT, {})
        out = set()
        for other in self.graph.neighbours(sid):
            if other in english:
                if other not in self._english_norms:
                    self._english_norms[other] = norm(english[other])
                out.add(self._english_norms[other])
        return out

    def attribution(self, sid: int) -> dict[str, Any]:
        return {"user": self.users.get(sid, ""), "cc0": sid in self.cc0}


def candidate(
    corpus: Corpus | None,
    *,
    family: str,
    lang: str,
    cid: str,
    group_key: str,
    ids: list[int],
    texts: list[str],
    edited: list[bool],
    label: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record = {
        "cid": cid,
        "family": family,
        "language": lang,
        "label": label,
        "group_key": group_key,
        "ids": ids,
        "texts": texts,
        "edited": edited,
        "metrics": pair_metrics(texts[0], texts[1]),
        **(extra or {}),
    }
    if corpus is not None:
        record["attribution"] = {str(sid): corpus.attribution(sid) for sid in ids}
    return record


def name_candidates(
    lang: str, texts: dict[int, str], corpus: Corpus | None = None
) -> list[dict[str, Any]]:
    out = []
    for sid in sorted(texts):
        text = texts[sid]
        if not usable(text):
            continue
        swap = name_swap(text, lang)
        if swap is None or not usable(swap[1]):
            continue
        kind, edited = swap
        out.append(
            candidate(
                corpus,
                family="pn-name",
                lang=lang,
                cid=f"name:{sid}:{kind}",
                group_key=f"seed:{sid}",
                ids=[sid],
                texts=[text, edited],
                edited=[False, True],
                label=1 if kind == "coord" else 0,
                extra={"kind": kind, "editor": "rule"},
            )
        )
    return out


def twin_seeds(
    lang: str,
    texts: dict[int, str],
    used: set[int],
    count: int,
    corpus: Corpus | None = None,
) -> list[dict[str, Any]]:
    eligible = [
        sid
        for sid in texts
        if sid not in used and usable(texts[sid]) and twin_eligible(texts[sid], lang)
    ]
    eligible.sort(key=lambda sid: sha(f"pn1-twin:{sid}"))
    seeds = []
    for rank, sid in enumerate(eligible[:count]):
        seed = {"seed": sid, "language": lang, "rank": rank, "text": texts[sid]}
        if corpus is not None:
            seed["attribution"] = {str(sid): corpus.attribution(sid)}
        seeds.append(seed)
    return seeds


def hop_pairs(
    lang: str,
    texts: dict[int, str],
    graph: LinkGraph,
    lang_of: dict[int, str],
    used: set[int],
    cap_per_bin: int,
    corpus: Corpus | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    members: dict[int, list[int]] = collections.defaultdict(list)
    for sid in sorted(texts):
        if sid in used or not usable(texts[sid]):
            continue
        for pivot in graph.neighbours(sid):
            if lang_of.get(pivot) != lang:
                members[pivot].append(sid)
    pivots = sorted(
        (pivot for pivot, group in members.items() if len(group) >= 2),
        key=lambda p: (lang_of.get(p) != PIVOT, sha(f"pn1-pivot:{p}")),
    )
    taken: set[int] = set()
    per_bin: collections.Counter = collections.Counter()
    out = []
    for pivot in pivots:
        if all(per_bin[b] >= cap_per_bin for b in (1, 2, 3, 4)):
            break
        group = sorted(
            (sid for sid in members[pivot] if sid not in taken),
            key=lambda sid: sha(f"pn1-hop:{sid}"),
        )
        chosen = None
        for i, a in enumerate(group):
            for b in group[i + 1 :]:
                metrics = pair_metrics(texts[a], texts[b])
                if (
                    metrics["norm_equal"]
                    or metrics["length_ratio"] > MAX_LENGTH_RATIO
                    or metrics["bin"] == 0
                    or per_bin[metrics["bin"]] >= cap_per_bin
                ):
                    continue
                chosen = (a, b)
                break
            if chosen:
                break
        if chosen is None:
            continue
        a, b = sorted(chosen)
        taken.update((a, b))
        record = candidate(
            corpus,
            family="pn-hop",
            lang=lang,
            cid=f"hop:{a}:{b}",
            group_key=f"pair:{a}:{b}",
            ids=[a, b],
            texts=[texts[a], texts[b]],
            edited=[False, False],
            label=1,
            extra={"pivot": pivot, "pivot_lang": lang_of.get(pivot, "other")},
        )
        per_bin[record["metrics"]["bin"]] += 1
        out.append(record)
    stats = {
        "pivots_with_two_members": len(pivots),
        "pairs": len(out),
        "per_bin": {str(k): per_bin[k] for k in sorted(per_bin)},
        "english_pivots": sum(r["pivot_lang"] == PIVOT for r in out),
        "cap_per_bin": cap_per_bin,
    }
    return out, stats


def length_edges(records: list[dict[str, Any]]) -> list[float]:
    """Quartile edges of the pairs' mean normalized length."""
    values = sorted(
        (len(norm(r["texts"][0])) + len(norm(r["texts"][1]))) / 2 for r in records
    )
    if not values:
        return []
    return [values[len(values) * q // 4] for q in (1, 2, 3)]


def fine_cell(
    bigram: float, ratio: float, mean_length: float, edges: list[float]
) -> tuple[int, int, int]:
    step = min(int(bigram / 0.05), 19)
    ratio_class = 0 if ratio <= 1.1 else 1 if ratio <= 1.25 else 2
    return step, ratio_class, sum(mean_length >= edge for edge in edges)


def near_quotas(
    hops: list[dict[str, Any]], edges: list[float]
) -> dict[tuple[int, int, int], int]:
    quotas: collections.Counter = collections.Counter()
    for record in hops:
        a, b = (norm(t) for t in record["texts"])
        m = record["metrics"]
        quotas[
            fine_cell(
                m["bigram_jaccard"], m["length_ratio"], (len(a) + len(b)) / 2, edges
            )
        ] += 1
    return dict(quotas)


def near_pairs(
    lang: str,
    texts: dict[int, str],
    graph: LinkGraph,
    used: set[int],
    english_norms: Callable[[int], set[str]],
    quotas: dict[tuple[int, int, int], int],
    edges: list[float],
    *,
    rare: int = NEAR_RARE_BIGRAMS,
    postings: int = NEAR_POSTINGS,
    max_queries: int = NEAR_MAX_QUERIES,
    corpus: Corpus | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    order = sorted(
        (sid for sid in texts if sid not in used and usable(texts[sid])),
        key=lambda sid: sha(f"pn1-near:{sid}"),
    )
    norms = [norm(texts[sid]) for sid in order]
    lengths = [len(value) for value in norms]
    vocab: dict[str, int] = {}
    sets: list[frozenset[int]] = []
    index: dict[int, array.array] = collections.defaultdict(lambda: array.array("I"))
    for position, value in enumerate(norms):
        members = frozenset(vocab.setdefault(g, len(vocab)) for g in grams(value, 2))
        sets.append(members)
        for g in members:
            index[g].append(position)
    remaining = {cell: n for cell, n in quotas.items() if n > 0}
    taken = [False] * len(order)
    out = []
    queries = rejected_links = rejected_english = 0
    for position, query in enumerate(order):
        if not remaining or queries >= max_queries:
            break
        if taken[position]:
            continue
        queries += 1
        mine = sets[position]
        if not mine:
            continue
        rarest = sorted(mine, key=lambda g: (len(index[g]), g))[:rare]
        seen: set[int] = set()
        options = []
        for g in rarest:
            for other in index[g][:postings]:
                if other == position or taken[other] or other in seen:
                    continue
                seen.add(other)
                short, long = sorted((lengths[position], lengths[other]))
                if long > MAX_LENGTH_RATIO * max(short, 1):
                    continue
                theirs = sets[other]
                shared = len(mine & theirs)
                value = shared / (len(mine) + len(theirs) - shared)
                if value < NEAR_MIN_JACCARD or norms[other] == norms[position]:
                    continue
                cell = fine_cell(value, long / max(short, 1), (short + long) / 2, edges)
                if remaining.get(cell, 0) <= 0:
                    continue
                options.append((-remaining[cell] / quotas[cell], -value, other, cell))
        if not options:
            continue
        options.sort()
        mine_near = None
        for _, _, other, cell in options[:NEAR_TRIES]:
            if mine_near is None:
                mine_near = graph.within([query])
            partner = order[other]
            if mine_near & graph.within([partner]):
                rejected_links += 1
                continue
            if english_norms(query) & english_norms(partner):
                rejected_english += 1
                continue
            a, b = sorted((query, partner))
            out.append(
                candidate(
                    corpus,
                    family="pn-near",
                    lang=lang,
                    cid=f"near:{a}:{b}",
                    group_key=f"pair:{a}:{b}",
                    ids=[a, b],
                    texts=[texts[a], texts[b]],
                    edited=[False, False],
                    label=0,
                )
            )
            taken[position] = taken[other] = True
            remaining[cell] -= 1
            if remaining[cell] <= 0:
                del remaining[cell]
            break
    per_bin = collections.Counter(r["metrics"]["bin"] for r in out)
    stats = {
        "pool_sentences": len(order),
        "queries": queries,
        "pairs": len(out),
        "per_bin": {str(k): per_bin[k] for k in sorted(per_bin)},
        "quota_total": sum(quotas.values()),
        "quota_unfilled": sum(remaining.values()),
        "rejected_two_hop_links": rejected_links,
        "rejected_english_translation": rejected_english,
        "parameters": {
            "rare_bigrams": rare,
            "postings": postings,
            "max_queries": max_queries,
        },
    }
    return out, stats
