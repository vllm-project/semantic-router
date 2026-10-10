"""SK1: licence-clean multi-answer "select all that apply" decisions (d3 same-skill set).

    python -m d25.vega.data.sk1 fetch --raw DIR
    python -m d25.vega.data.sk1 build --raw DIR --index DIR --holdouts holdouts-v2.json \
        --tokenizer PATH --out DIR [--seed 20261011] [--workers 8]

An item is a text, a select-all question and 3-8 candidates with exact gold membership; every candidate becomes
one training row (one question per forward pass). 70% of the items use the Decision Index kit's SATA-Bench
layout (state {paragraph, question, options}; per candidate "Question: ...\\nDoes this candidate correctly answer
the question?\\nCandidate: ...", criteria {no: No, yes: Yes}), 30% natural variants (plain or other state,
other wording, criteria order, noul). A smaller part of rare-positive per-label detection rows (about 8% yes)
keeps per-label yes rates calibrated.

Sources (commercial-use licences on their cards; train splits only):
- GoEmotions (Apache-2.0): per-rater labels; gold = chosen by >= 2 raters, distractor = chosen by no rater.
- Civil Comments (CC0-1.0): attribute share >= 0.5 is gold, < 0.1 a distractor, anything between is not offered.
- arXiv abstracts 2021 (CC0-1.0): listed categories; distractors only from fields no listed category touches.
- DROP (CC BY-SA 4.0): multi-span answers; distractors are other names or numbers of the passage that share no
  token with a gold span.
SATA-Bench's own sources (MultiRC, RealToxicityPrompts, Reuters-21578, the PubMed MeSH multi-label set, EURLEX57K
and other EUR-Lex recasts, events_classification_biotech) are never used. Rows pass holdouts and the decontam
index (exact + 13-gram rule v4 against suite 0.3, GSM8K test and the proxy protected items); a flagged row drops
its whole item.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import statistics
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path
from typing import Any

from d25.vega.data.holdouts import Holdouts
from d25.vega.data.util import (
    choice_question,
    make_row,
    noul_question,
    rank,
    read_jsonl,
    rng_for,
    sha,
    sha256_file,
    write_json,
    write_jsonl,
)

SEED = 20261011
MAX_TOKENS = 8160
DEV_PER_MILLE = 20

SOURCES = {
    "goemotions": (
        "google-research-datasets/go_emotions",
        ["raw/train-*", "simplified/train-*"],
        "apache-2.0",
    ),
    "civil_comments": ("google/civil_comments", ["data/train-*"], "cc0-1.0"),
    "arxiv": ("gfissore/arxiv-abstracts-2021", ["arxiv-abstracts.jsonl.gz"], "cc0-1.0"),
    "drop": ("ucinlp/drop", ["data/train-*"], "cc-by-sa-4.0"),
}
ITEMS = {"goemotions": 1500, "civil_comments": 1100, "arxiv": 1300, "drop": 1200}
DETECT = {"goemotions": 1600, "civil_comments": 1600}

# SATA-Bench d1-d6 sources (paper appendix A) with mirrors and recasts.
SATA_EXCLUDED = [
    "multirc",
    "real-toxicity-prompts",
    "realtoxicityprompts",
    "reuters",
    "pubmed-multilabel",
    "pubmed_multilabel",
    "eurlex",
    "eur-lex",
    "multi_eurlex",
    "lex_glue",
    "events_classification_biotech",
    "sata-bench",
]

KIT_SATA = (
    "Question: {q}\nDoes this candidate correctly answer the question?\nCandidate: {o}"
)
YES_NO = {"no": "No", "yes": "Yes"}

EMOTIONS = {
    "admiration": "finding something or someone impressive or worthy of respect",
    "amusement": "finding something funny or entertaining",
    "anger": "a strong feeling of displeasure or hostility",
    "annoyance": "mild anger or irritation",
    "approval": "having or expressing a favourable opinion",
    "caring": "showing kindness and concern for someone",
    "confusion": "not understanding something, or feeling uncertain",
    "curiosity": "wanting to know or learn something",
    "desire": "strongly wanting something or wishing for something to happen",
    "disappointment": "sadness caused by unmet hopes or expectations",
    "disapproval": "having or expressing an unfavourable opinion",
    "disgust": "revulsion at something unpleasant or offensive",
    "embarrassment": "self-consciousness, shame or awkwardness",
    "excitement": "great enthusiasm and eagerness",
    "fear": "being afraid or worried about a threat",
    "gratitude": "thankfulness and appreciation",
    "grief": "intense sorrow, especially over a death or loss",
    "joy": "a feeling of pleasure and happiness",
    "love": "strong affection or warm attachment",
    "nervousness": "apprehension, worry or anxiety",
    "optimism": "hopefulness and confidence about the future",
    "pride": "satisfaction with one's own achievements or those of people close to one",
    "realization": "becoming aware of something",
    "relief": "reassurance after the end of anxiety or distress",
    "remorse": "regret or guilt about something one did",
    "sadness": "emotional pain or sorrow",
    "surprise": "being astonished by something unexpected",
    "neutral": "no particular emotion is expressed",
}
GO_Q = [
    "Which emotions does the comment above express?",
    "What emotions are expressed in the comment above?",
    "Which of the following emotions does the author of the comment above convey?",
]
GO_DEF_Q = [
    "Which of the emotion definitions above does the comment satisfy?",
    "Multi-label question: which of the defined emotions apply to the comment?",
]

CIVIL = {
    "toxicity": ("toxicity", "rude, hostile or disrespectful language toward others"),
    "severe_toxicity": (
        "severe toxicity",
        "extremely hateful, aggressive or abusive language",
    ),
    "obscene": ("obscenity", "swear words, cursing or other vulgar language"),
    "threat": ("threat", "a stated intention to hurt, injure or harm someone"),
    "insult": ("insult", "demeaning or belittling remarks aimed at a person or group"),
    "identity_attack": (
        "identity attack",
        "negative or hateful remarks about people because of an identity such as race, religion, "
        "gender or sexual orientation",
    ),
    "sexual_explicit": (
        "sexually explicit content",
        "references to sexual acts, body parts or other lewd content",
    ),
}
CIVIL_PHRASE = {
    "toxicity": "toxicity",
    "severe_toxicity": "severe toxicity",
    "obscene": "obscenity",
    "threat": "a threat",
    "insult": "an insult",
    "identity_attack": "an identity attack",
    "sexual_explicit": "sexually explicit content",
}
CIVIL_DEF_Q = [
    "Which of the categories defined above does the comment fall under?",
    "Multi-label question: which of the definitions above does this comment satisfy?",
]
CIVIL_Q = [
    "Which kinds of harmful content does the comment above contain?",
    "Which of the following moderation categories apply to the comment above?",
]

ARXIV_ALIAS = {
    "math.IT": "cs.IT",
    "math.NA": "cs.NA",
    "eess.SY": "cs.SY",
    "math.MP": "math-ph",
    "stat.TH": "math.ST",
    "q-fin.EC": "econ.GN",
}
ARXIV_NAMES = {
    "cs": "AI Artificial Intelligence|AR Hardware Architecture|CC Computational Complexity|CE Computational "
    "Engineering, Finance, and Science|CG Computational Geometry|CL Computation and Language|CR Cryptography "
    "and Security|CV Computer Vision and Pattern Recognition|CY Computers and Society|DB Databases|DC "
    "Distributed, Parallel, and Cluster Computing|DL Digital Libraries|DM Discrete Mathematics|DS Data "
    "Structures and Algorithms|ET Emerging Technologies|FL Formal Languages and Automata Theory|GR Graphics|GT "
    "Computer Science and Game Theory|HC Human-Computer Interaction|IR Information Retrieval|IT Information "
    "Theory|LG Machine Learning|LO Logic in Computer Science|MA Multiagent Systems|MM Multimedia|MS "
    "Mathematical Software|NA Numerical Analysis|NE Neural and Evolutionary Computing|NI Networking and "
    "Internet Architecture|OS Operating Systems|PF Performance|PL Programming Languages|RO Robotics|SC "
    "Symbolic Computation|SD Sound|SE Software Engineering|SI Social and Information Networks|SY Systems and "
    "Control",
    "math": "AC Commutative Algebra|AG Algebraic Geometry|AP Analysis of PDEs|AT Algebraic Topology|CA "
    "Classical Analysis and ODEs|CO Combinatorics|CT Category Theory|CV Complex Variables|DG Differential "
    "Geometry|DS Dynamical Systems|FA Functional Analysis|GM General Mathematics|GN General Topology|GR Group "
    "Theory|GT Geometric Topology|HO History and Overview|KT K-Theory and Homology|LO Logic|MG Metric "
    "Geometry|NT Number Theory|OA Operator Algebras|OC Optimization and Control|PR Probability|QA Quantum "
    "Algebra|RA Rings and Algebras|RT Representation Theory|SG Symplectic Geometry|SP Spectral Theory|ST "
    "Statistics Theory",
    "physics": "acc-ph Accelerator Physics|ao-ph Atmospheric and Oceanic Physics|app-ph Applied Physics|"
    "atm-clus Atomic and Molecular Clusters|atom-ph Atomic Physics|bio-ph Biological Physics|chem-ph Chemical "
    "Physics|class-ph Classical Physics|comp-ph Computational Physics|data-an Data Analysis, Statistics and "
    "Probability|ed-ph Physics Education|flu-dyn Fluid Dynamics|gen-ph General Physics|geo-ph Geophysics|"
    "hist-ph History and Philosophy of Physics|ins-det Instrumentation and Detectors|med-ph Medical Physics|"
    "optics Optics|plasm-ph Plasma Physics|pop-ph Popular Physics|soc-ph Physics and Society|space-ph Space "
    "Physics",
    "astro-ph": "CO Cosmology and Nongalactic Astrophysics|EP Earth and Planetary Astrophysics|GA "
    "Astrophysics of Galaxies|HE High Energy Astrophysical Phenomena|IM Instrumentation and Methods for "
    "Astrophysics|SR Solar and Stellar Astrophysics",
    "cond-mat": "dis-nn Disordered Systems and Neural Networks|mes-hall Mesoscale and Nanoscale Physics|"
    "mtrl-sci Materials Science|quant-gas Quantum Gases|soft Soft Condensed Matter|stat-mech Statistical "
    "Mechanics|str-el Strongly Correlated Electrons|supr-con Superconductivity",
    "nlin": "AO Adaptation and Self-Organizing Systems|CD Chaotic Dynamics|CG Cellular Automata and Lattice "
    "Gases|PS Pattern Formation and Solitons|SI Exactly Solvable and Integrable Systems",
    "q-bio": "BM Biomolecules|CB Cell Behavior|GN Genomics|MN Molecular Networks|NC Neurons and Cognition|PE "
    "Populations and Evolution|QM Quantitative Methods|SC Subcellular Processes|TO Tissues and Organs",
    "q-fin": "CP Computational Finance|GN General Finance|MF Mathematical Finance|PM Portfolio Management|PR "
    "Pricing of Securities|RM Risk Management|ST Statistical Finance|TR Trading and Market Microstructure",
    "stat": "AP Statistics Applications|CO Statistical Computation|ME Statistical Methodology|ML Machine "
    "Learning",
    "econ": "EM Econometrics|GN General Economics|TH Theoretical Economics",
    "eess": "AS Audio and Speech Processing|IV Image and Video Processing|SP Signal Processing",
    "": "gr-qc General Relativity and Quantum Cosmology|hep-ex High Energy Physics - Experiment|hep-lat High "
    "Energy Physics - Lattice|hep-ph High Energy Physics - Phenomenology|hep-th High Energy Physics - Theory|"
    "math-ph Mathematical Physics|nucl-ex Nuclear Experiment|nucl-th Nuclear Theory|quant-ph Quantum Physics",
}
ARXIV_FIELD = {
    "cs": {"cs"},
    "eess": {"cs"},
    "math": {"math"},
    "stat": {"stat"},
    "physics": {"physics"},
    "astro-ph": {"astro"},
    "cond-mat": {"physics"},
    "nlin": {"physics", "math"},
    "q-bio": {"bio"},
    "q-fin": {"finance"},
    "econ": {"finance"},
    "gr-qc": {"physics", "astro"},
    "hep-ex": {"physics", "astro"},
    "hep-lat": {"physics"},
    "hep-ph": {"physics", "astro"},
    "hep-th": {"physics", "astro", "math"},
    "math-ph": {"physics", "math"},
    "nucl-ex": {"physics"},
    "nucl-th": {"physics", "astro"},
    "quant-ph": {"physics", "cs"},
}
ARXIV_EXTRA_FIELD = {
    "stat.ML": {"cs"},
    "cs.IT": {"math"},
    "cs.NA": {"math"},
    "cs.DM": {"math"},
    "cs.LO": {"math"},
    "cs.CC": {"math"},
    "cs.SC": {"math"},
    "cs.FL": {"math"},
    "cs.CG": {"math"},
    "cs.DS": {"math"},
    "cs.CE": {"finance", "physics"},
    "cs.GT": {"finance", "math"},
    "cs.SY": {"math"},
    "cs.ET": {"physics"},
    "cs.SD": {"physics"},
    "math.OC": {"cs", "finance"},
    "math.PR": {"stat", "physics"},
    "math.ST": {"stat"},
    "math.DS": {"physics"},
    "math.AP": {"physics"},
    "math.NA": {"cs"},
    "math.CO": {"cs"},
    "math.LO": {"cs"},
    "physics.bio-ph": {"bio"},
    "physics.med-ph": {"bio"},
    "physics.chem-ph": {"bio"},
    "physics.soc-ph": {"cs", "finance"},
    "physics.data-an": {"stat", "cs"},
    "physics.comp-ph": {"cs"},
    "physics.ao-ph": {"astro"},
    "physics.geo-ph": {"astro"},
    "physics.space-ph": {"astro"},
    "physics.ed-ph": {"cs"},
    "physics.optics": {"cs"},
    "cond-mat.dis-nn": {"cs", "bio"},
    "cond-mat.soft": {"bio"},
    "cond-mat.stat-mech": {"math", "bio", "finance"},
    "astro-ph.EP": {"physics"},
    "astro-ph.IM": {"physics", "cs"},
    "q-bio.NC": {"cs", "physics"},
    "q-bio.QM": {"stat", "cs", "physics"},
    "q-bio.PE": {"math", "physics", "stat"},
    "q-bio.BM": {"physics"},
    "q-bio.MN": {"physics", "math"},
    "q-bio.GN": {"cs", "stat"},
    "q-bio.SC": {"physics"},
    "q-bio.CB": {"physics"},
    "q-bio.TO": {"physics"},
    "q-fin.ST": {"stat", "physics"},
    "q-fin.CP": {"cs", "math"},
    "q-fin.MF": {"math", "stat"},
    "q-fin.PR": {"math"},
    "q-fin.RM": {"math", "stat"},
    "q-fin.PM": {"math", "cs"},
    "q-fin.TR": {"physics", "cs"},
    "econ.EM": {"stat"},
    "econ.TH": {"math", "cs"},
    "stat.ME": {"bio", "math"},
    "stat.AP": {"bio", "physics", "finance"},
    "stat.CO": {"cs", "math"},
}
ARXIV_Q = [
    "Which arXiv subject categories is the paper above listed under?",
    "Under which of the following arXiv categories was the paper above filed?",
    "Which subject areas does the paper above belong to, according to its arXiv listing?",
]

DROP_STOP = set(
    "the a an in on at after before during however this that these those with from for by his her their its it "
    "he she they we as when while although despite following since until both each many most some other there "
    "then but and or of to also later first second third fourth final next last one two three four five six "
    "seven eight nine ten january february march april may june july august september october november december "
    "monday tuesday wednesday thursday friday saturday sunday quarter half".split()
)
CAP_PHRASE = re.compile(
    r"[A-Z][\w'\-]*(?:\s+(?:of\s+|de\s+|van\s+|von\s+|da\s+|al\s+)?[A-Z][\w'\-]*)*"
)
NUMBER = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])")


def retry(fn, *args, **kwargs):
    for attempt in range(6):
        try:
            return fn(*args, **kwargs)
        except Exception:
            if attempt == 5:
                raise
            time.sleep(20 * (attempt + 1))
    return None


def card_licence(readme: Path) -> str | None:
    if not readme.exists():
        return None
    text = readme.read_text(encoding="utf-8", errors="ignore")
    if not text.startswith("---"):
        return None
    head = text.split("---", 2)[1]
    match = re.search(
        r"^licen[cs]e:[ \t]*(.*?)[ \t]*$((?:\n[ \t]*-[ \t]*.*)*)", head, re.M
    )
    if not match:
        return None
    values = [match.group(1)] + re.findall(r"-[ \t]*(\S.*)", match.group(2))
    return ",".join(v.strip() for v in values if v.strip()) or None


def sata_source_check() -> dict[str, Any]:
    hits = [
        (repo, bad)
        for repo, _, _ in SOURCES.values()
        for bad in SATA_EXCLUDED
        if bad in repo.lower()
    ]
    if hits:
        raise SystemExit(f"SATA-Bench source used: {hits}")
    return {
        "excluded_patterns": SATA_EXCLUDED,
        "sata_bench_sources": {
            "d1": "MultiRC (cogcomp)",
            "d2": "allenai/real-toxicity-prompts",
            "d3": "Reuters-21578 (UCI 137)",
            "d4": "Kaggle owaiskhan9654/pubmed-multilabel-text-classification",
            "d5": "EURLEX57K",
            "d6": "knowledgator/events_classification_biotech",
        },
        "matches": [],
    }


# ------------------------------------------------------------------------------------------------ fetch


def fetch(raw: Path) -> dict[str, Any]:
    from huggingface_hub import HfApi, snapshot_download

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    report = {}
    for name, (repo, patterns, licence) in SOURCES.items():
        revision = api.dataset_info(repo).sha
        retry(
            snapshot_download,
            repo,
            repo_type="dataset",
            revision=revision,
            local_dir=str(raw / name),
            allow_patterns=patterns + ["README.md"],
            max_workers=8,
        )
        files = sorted(
            p
            for p in (raw / name).rglob("*")
            if p.is_file() and ".cache" not in p.parts
        )
        report[name] = {
            "repo": repo,
            "revision": revision,
            "licence": licence,
            "licence_card": card_licence(raw / name / "README.md"),
            "files": {str(p.relative_to(raw / name)): sha256_file(p) for p in files},
        }
        print(name, revision, report[name]["licence_card"], flush=True)
    write_json(raw / "fetch.json", report)
    return report


# ----------------------------------------------------------------------------------------------- sources


def parquet(raw: Path, pattern: str, columns: list[str] | None = None) -> list[dict]:
    import pyarrow.parquet as pq

    rows: list[dict] = []
    for path in sorted(raw.glob(pattern)):
        rows.extend(pq.read_table(path, columns=columns).to_pylist())
    return rows


def item(source, key, group, paragraph, question, options, gold, extra=None):
    return {
        "source": source,
        "key": str(key),
        "group": str(group),
        "paragraph": paragraph,
        "question": question,
        "options": options,
        "gold": gold,
        "extra": extra or {},
    }


def negatives_for(rng: random.Random, n_pos: int) -> int:
    return rng.choice({1: [2, 3], 2: [2, 3, 4], 3: [3, 4, 5]}.get(n_pos, [3, 4]))


def goemotions_examples(raw: Path) -> list[tuple[str, str, list[str], list[str]]]:
    names = list(EMOTIONS)
    train_ids = {r["id"] for r in parquet(raw, "simplified/train-*.parquet", ["id"])}
    agg: dict[str, dict] = {}
    for r in parquet(
        raw, "raw/train-*.parquet", ["id", "text", "example_very_unclear"] + names
    ):
        if r["id"] not in train_ids:
            continue
        a = agg.setdefault(
            r["id"], {"text": r["text"], "n": 0, "unclear": 0, "c": Counter()}
        )
        a["n"] += 1
        a["unclear"] += bool(r["example_very_unclear"])
        for e in names:
            if r[e]:
                a["c"][e] += 1
    out = []
    for i, a in agg.items():
        if a["n"] < 3 or 2 * a["unclear"] >= a["n"] or len(a["text"].strip()) < 12:
            continue
        pos = [e for e in names if a["c"][e] >= 2]
        neg = [e for e in names if a["c"][e] == 0]
        if not pos or ("neutral" in pos and len(pos) > 1):
            continue
        out.append((i, a["text"].strip(), pos, neg))
    return sorted(out, key=lambda x: rank(x[0], SEED))


def goemotions_items(examples, n: int, seed: int) -> list[dict]:
    multi = [x for x in examples if 2 <= len(x[2]) <= 4]
    single = [x for x in examples if len(x[2]) == 1]
    chosen = multi[: int(n * 0.7)]
    chosen += single[: n - len(chosen)]
    items = []
    for i, text, pos, neg in chosen:
        rng = rng_for(f"go:{i}", seed)
        k = min(negatives_for(rng, len(pos)), len(neg), 8 - len(pos))
        options = pos + rng.sample(neg, k)
        rng.shuffle(options)
        if rng.random() < 0.25:
            defs = "\n".join(f"{o}: {EMOTIONS[o]}" for o in options)
            paragraph = f"Emotion definitions:\n{defs}\n\nComment: {text}"
            question = rng.choice(GO_DEF_Q)
        else:
            paragraph, question = text, rng.choice(GO_Q)
        items.append(
            item(
                "goemotions",
                i,
                i,
                paragraph,
                question,
                options,
                [o in pos for o in options],
            )
        )
    return items


def goemotions_detect(examples, n: int, seed: int) -> list[dict]:
    out = []
    for i, text, pos, neg in sorted(examples, key=lambda x: rank(f"det:{x[0]}", seed))[
        :n
    ]:
        rng = rng_for(f"go-det:{i}", seed)
        yes = rng.random() < 0.08
        label = rng.choice(pos if yes else neg)
        out.append(
            {
                "source": "goemotions",
                "key": f"det:{i}",
                "group": i,
                "text": text,
                "label": label,
                "phrase": label,
                "yes": yes,
                "task": "emotion tagging",
            }
        )
    return out


def civil_rows(raw: Path) -> list[dict]:
    cols = ["text"] + list(CIVIL)
    rows = parquet(raw, "data/train-*.parquet", cols)
    for idx, r in enumerate(rows):
        r["_idx"] = idx
    return [r for r in rows if r["text"] and 20 <= len(r["text"]) <= 1500]


def civil_split(r: dict) -> tuple[list[str], list[str]]:
    pos = [a for a in CIVIL if (r[a] or 0.0) >= 0.5]
    neg = [a for a in CIVIL if (r[a] or 0.0) < 0.1]
    return pos, neg


def civil_items(rows: list[dict], n: int, seed: int) -> list[dict]:
    toxic = []
    for r in rows:
        pos, neg = civil_split(r)
        if pos and len(pos) + len(neg) >= 3:
            toxic.append((r, pos, neg))
    toxic.sort(key=lambda x: rank(f"cc:{x[0]['_idx']}", seed))
    multi = [x for x in toxic if len(x[1]) >= 2]
    single = [x for x in toxic if len(x[1]) == 1]
    chosen = multi[: int(n * 0.55)]
    chosen += single[: n - len(chosen)]
    items = []
    for r, pos, neg in chosen:
        rng = rng_for(f"cc:{r['_idx']}", seed)
        k = min(max(2, negatives_for(rng, len(pos))), len(neg), 7 - len(pos))
        options = pos + rng.sample(neg, k)
        rng.shuffle(options)
        names = [CIVIL[a][0] for a in options]
        text = r["text"].strip()
        if rng.random() < 0.6:
            defs = "\n".join(f"- {CIVIL[a][0]}: {CIVIL[a][1]}" for a in options)
            paragraph = f"Category definitions:\n{defs}\n\nComment: {text}"
            question = rng.choice(CIVIL_DEF_Q)
        else:
            paragraph, question = text, rng.choice(CIVIL_Q)
        items.append(
            item(
                "civil_comments",
                r["_idx"],
                r["_idx"],
                paragraph,
                question,
                names,
                [a in pos for a in options],
            )
        )
    return items


def civil_detect(rows: list[dict], n: int, seed: int) -> list[dict]:
    pool = sorted(rows, key=lambda r: rank(f"cc-det:{r['_idx']}", seed))
    toxic = [r for r in pool if civil_split(r)[0]]
    picked = toxic[: n // 4]
    seen = {r["_idx"] for r in picked}
    picked += [r for r in pool[: 2 * n] if r["_idx"] not in seen][: n - len(picked)]
    out = []
    for r in picked:
        rng = rng_for(f"cc-det:{r['_idx']}", seed)
        pos, neg = civil_split(r)
        if pos and rng.random() < 0.3:
            attr, yes = rng.choice(pos), True
        elif neg:
            attr, yes = rng.choice(neg), False
        else:
            continue
        out.append(
            {
                "source": "civil_comments",
                "key": f"det:{r['_idx']}",
                "group": r["_idx"],
                "text": r["text"].strip(),
                "label": CIVIL[attr][0],
                "phrase": CIVIL_PHRASE[attr],
                "yes": yes,
                "task": "content moderation",
            }
        )
    return out


def arxiv_tables() -> tuple[dict[str, str], dict[str, set[str]]]:
    names, fields = {}, {}
    for archive, spec in ARXIV_NAMES.items():
        for entry in spec.split("|"):
            code, name = entry.split(" ", 1)
            cat = code if not archive else f"{archive}.{code}"
            names[cat] = f"{name} ({cat})"
            base = ARXIV_FIELD[archive or code]
            fields[cat] = set(base) | ARXIV_EXTRA_FIELD.get(cat, set())
    return names, fields


def arxiv_items(raw: Path, n: int, seed: int) -> list[dict]:
    names, fields = arxiv_tables()
    cats = sorted(names)
    pool = []
    for r in read_jsonl(raw / "arxiv-abstracts.jsonl.gz"):
        pid = str(r.get("id") or "")
        if not pid or int(rank(f"ax:{pid}", seed)[:8], 16) >= int(0.02 * 2**32):
            continue
        raw_cats = r.get("categories") or []
        raw_cats = " ".join(raw_cats) if isinstance(raw_cats, list) else str(raw_cats)
        gold = list(dict.fromkeys(ARXIV_ALIAS.get(c, c) for c in raw_cats.split()))
        abstract = " ".join(str(r.get("abstract") or "").split())
        title = " ".join(str(r.get("title") or "").split())
        if (
            not 2 <= len(gold) <= 4
            or any(c not in names for c in gold)
            or not title
            or not 300 <= len(abstract) <= 1800
        ):
            continue
        pool.append((pid, title, abstract, gold))
    pool.sort(key=lambda x: rank(x[0], seed))
    items = []
    for pid, title, abstract, gold in pool[:n]:
        rng = rng_for(f"ax:{pid}", seed)
        touched = set().union(*(fields[c] for c in gold))
        far = [c for c in cats if not fields[c] & touched]
        k = min(negatives_for(rng, len(gold)), len(far), 8 - len(gold))
        if k < 2:
            continue
        options = gold + rng.sample(far, k)
        rng.shuffle(options)
        items.append(
            item(
                "arxiv",
                pid,
                pid,
                f"{title}\n\n{abstract}",
                rng.choice(ARXIV_Q),
                [names[c] for c in options],
                [c in gold for c in options],
            )
        )
    return items


def tokens_of(text: str) -> set[str]:
    return {
        t
        for t in re.findall(r"\w+", text.lower())
        if len(t) >= 3 and t not in DROP_STOP
    }


def drop_items(raw: Path, n: int, seed: int) -> tuple[list[dict], dict]:
    stats: Counter = Counter()
    pool = []
    for r in parquet(raw, "data/train-*.parquet"):
        answer = r.get("answers_spans") or {}
        spans = [s.strip() for s in answer.get("spans") or [] if s and s.strip()]
        gold = list(dict.fromkeys(spans))
        if (
            len(gold) < 2
            or len(gold) > 6
            or any(t != "span" for t in answer.get("types") or [])
        ):
            continue
        stats["multi_span"] += 1
        passage = r["passage"]
        if all(NUMBER.fullmatch(g) for g in gold):
            found = NUMBER.findall(passage)
            candidates = [c for c in found if c not in gold]
            mode = "number"
        elif all(g[:1].isupper() for g in gold):
            mid_sentence = set()
            for m in CAP_PHRASE.finditer(passage):
                words = m.group(0).strip(" .'-").split()
                initial = passage[: m.start()].rstrip()[-1:] in ("", ".", "!", "?", '"')
                while words and words[0].lower() in DROP_STOP:
                    words, initial = words[1:], False
                phrase = " ".join(words).strip(" .'-")
                if len(phrase) >= 3 and phrase.lower() not in DROP_STOP and not initial:
                    mid_sentence.add(phrase)
            candidates = sorted(mid_sentence)
            mode = "name"
        else:
            stats["other_type"] += 1
            continue
        gold_tokens = set().union(*(tokens_of(g) for g in gold))
        golds_low = [g.lower() for g in gold]
        clean = []
        for c in dict.fromkeys(candidates):
            low = c.lower()
            if any(low in g or g in low for g in golds_low):
                continue
            if mode == "name" and tokens_of(c) & gold_tokens:
                continue
            clean.append(c)
        if len(clean) < 2:
            stats["few_distractors"] += 1
            continue
        pool.append(
            (
                str(r["query_id"]),
                str(r["section_id"]),
                passage,
                r["question"],
                gold,
                clean,
            )
        )
    stats["pool"] = len(pool)
    pool.sort(key=lambda x: rank(x[0], seed))
    items = []
    for qid, section, passage, question, gold, clean in pool[:n]:
        rng = rng_for(f"dr:{qid}", seed)
        k = min(rng.randint(2, 4), len(clean), 8 - len(gold))
        options = gold + rng.sample(clean, k)
        rng.shuffle(options)
        items.append(
            item(
                "drop",
                qid,
                section,
                passage.strip(),
                question.strip(),
                options,
                [o in gold for o in options],
            )
        )
    return items, dict(stats)


# --------------------------------------------------------------------------------------------- rendering


def sata_rows(it: dict, seed: int) -> list[tuple[str, Any, dict, list[float], str]]:
    rng = rng_for(f"render:{it['source']}:{it['key']}", seed)
    p, q, options, gold = it["paragraph"], it["question"], it["options"], it["gold"]
    out = []
    if rng.random() < 0.7:
        state = {"paragraph": p, "question": q, "options": options}
        for j, (o, g) in enumerate(zip(options, gold)):
            question = choice_question(KIT_SATA.format(q=q, o=o), dict(YES_NO))
            out.append(
                (str(j), state, question, [0.0, 1.0] if g else [1.0, 0.0], "kit")
            )
        return out
    variant = rng.randrange(3)
    for j, (o, g) in enumerate(zip(options, gold)):
        if variant == 0:
            criteria = (
                {"yes": "Yes, it is correct", "no": "No, it is not correct"}
                if rng.random() < 0.5
                else {"no": "No, it is not correct", "yes": "Yes, it is correct"}
            )
            question = choice_question(
                f"{q}\nSeveral candidates may be correct. Candidate: {o}\nIs this candidate correct?",
                criteria,
            )
            target = [float(g == (key == "yes")) for key in criteria]
            state: Any = p
        elif variant == 1:
            state = {"text": p} if p else {}
            state.update(
                {
                    "question": q,
                    "candidates": options,
                    "instructions": "Select all that apply.",
                }
            )
            question = noul_question(f'Should "{o}" be selected?')
            target = [0.0, 1.0] if g else [1.0, 0.0]
        else:
            listing = "\n".join(f"- {x}" for x in options)
            state = (f"{p}\n\n" if p else "") + f"Question: {q}\nCandidates:\n{listing}"
            criteria = {"yes": None, "no": None} if j % 2 else dict(YES_NO)
            question = choice_question(
                f'Select all that apply. Is "{o}" one of the correct candidates?',
                criteria,
            )
            target = [float(g == (key == "yes")) for key in criteria]
        out.append((str(j), state, question, target, "natural"))
    return out


DETECT_KIT = [
    "Does this text exhibit {label}? Evaluate this category independently.",
    "Does this text express {label}? Evaluate this category independently.",
]


def detect_row(d: dict, seed: int) -> tuple[Any, dict, list[float], str]:
    rng = rng_for(f"render-det:{d['source']}:{d['key']}", seed)
    if rng.random() < 0.7:
        question = choice_question(
            rng.choice(DETECT_KIT).format(label=d["phrase"]), dict(YES_NO)
        )
        return d["text"], question, ([0.0, 1.0] if d["yes"] else [1.0, 0.0]), "kit"
    state = {"text": d["text"], "task": d["task"]}
    if rng.random() < 0.5:
        question = noul_question(f'Does the label "{d["label"]}" apply to this text?')
        return state, question, ([0.0, 1.0] if d["yes"] else [1.0, 0.0]), "natural"
    criteria = {"yes": "Applies", "no": "Does not apply"}
    question = choice_question(
        f'Judge only this label: is "{d["label"]}" present in the text?', criteria
    )
    return state, question, ([1.0, 0.0] if d["yes"] else [0.0, 1.0]), "natural"


def yes_of(row: dict) -> bool:
    question = row["question"]
    if question["type"] == "noul":
        return row["target"][1] >= 0.5
    return row["target"][list(question["criteria"]).index("yes")] >= 0.5


# ------------------------------------------------------------------------------------------------ build


def _tok_init(path: str) -> None:
    from d25.vega.data import tokens

    tokens._init(path)


def _tok_count(rows: list[dict]) -> list[dict]:
    from d25.vega.data import tokens

    return tokens.count(rows)


def describe_lengths(values: list[int]) -> dict[str, Any]:
    if not values:
        return {}
    ordered = sorted(values)
    pick = lambda q: ordered[min(len(ordered) - 1, int(q * len(ordered)))]  # noqa: E731
    return {
        "total": sum(ordered),
        "mean": round(statistics.mean(ordered), 1),
        "p50": pick(0.5),
        "p90": pick(0.9),
        "p99": pick(0.99),
        "max": ordered[-1],
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    from d25.vega.data.decontam import check_rows

    started = time.time()
    raw, out, seed = Path(args.raw), Path(args.out), args.seed
    if (out / "VERIFIED").exists():
        print(json.dumps({"out": str(out), "already": True}))
        return json.loads((out / "manifest.json").read_text())
    fetched = json.loads((raw / "fetch.json").read_text())
    exclusion = sata_source_check()
    holdouts = Holdouts(args.holdouts)

    parse_stats: dict[str, Any] = {}
    go = goemotions_examples(raw / "goemotions")
    civil = civil_rows(raw / "civil_comments")
    drop, parse_stats["drop"] = drop_items(raw / "drop", ITEMS["drop"], seed)
    items = (
        goemotions_items(go, ITEMS["goemotions"], seed)
        + civil_items(civil, ITEMS["civil_comments"], seed)
        + arxiv_items(raw / "arxiv", ITEMS["arxiv"], seed)
        + drop
    )
    detections = goemotions_detect(go, DETECT["goemotions"], seed) + civil_detect(
        civil, DETECT["civil_comments"], seed
    )
    parse_stats["goemotions_examples"] = len(go)
    parse_stats["civil_rows"] = len(civil)
    del go, civil

    def meta_for(source: str, group: str, key: str, part: str, render: str, extra):
        repo, _, _ = SOURCES[source]
        info = fetched[source]
        return {
            "dataset": f"{repo}@{info['revision'][:8]}",
            "licence": info["licence"],
            "licence_use": "commercial",
            "source_split": "train",
            "group": f"sk1:{source}:{group}",
            "item": f"sk1:{source}:{key}",
            "hf_ids": [repo],
            "soft": False,
            "part": part,
            "render": render,
            **extra,
        }

    rows: list[dict] = []
    for it in items:
        n_correct = sum(it["gold"])
        for j, state, question, target, render in sata_rows(it, seed):
            meta = meta_for(
                it["source"],
                it["group"],
                it["key"],
                "sk1",
                render,
                {
                    "orig_kind": question["type"],
                    "sata": {
                        "n_options": len(it["options"]),
                        "n_correct": n_correct,
                        "position": int(j),
                    },
                },
            )
            rows.append(
                make_row(
                    row_id=f"sk1:{it['source']}:{it['key']}:{j}",
                    source=f"sk1:{it['source']}",
                    family=f"sk1/{it['source']}",
                    state=state,
                    question=question,
                    target=target,
                    meta=meta,
                )
            )
    for d in detections:
        state, question, target, render = detect_row(d, seed)
        meta = meta_for(
            d["source"],
            d["group"],
            d["key"],
            "sk1-detect",
            render,
            {"orig_kind": question["type"], "label": d["label"]},
        )
        rows.append(
            make_row(
                row_id=f"sk1:{d['source']}:{d['key']}",
                source=f"sk1:{d['source']}",
                family=f"sk1/{d['source']}-detect",
                state=state,
                question=question,
                target=target,
                meta=meta,
            )
        )
    built = Counter(r["meta"]["dataset"].split("@")[0] for r in rows)

    held = Counter()
    for r in rows:
        reason = holdouts.check(r)
        if reason:
            held[reason] += 1
            r["_drop"] = f"holdout:{reason}"

    flags = list(check_rows(Path(args.index), rows, args.workers))
    by_reason: Counter = Counter()
    by_bench: Counter = Counter()
    bad_items = set()
    for row, flag in zip(rows, flags):
        if flag["drop"]:
            bad_items.add(row["meta"]["item"])
            by_reason.update(flag["reasons"])
            by_bench[flag["bench"] or "?"] += 1
    for row in rows:
        if row["meta"]["item"] in bad_items and "_drop" not in row:
            row["_drop"] = "decontam"
    decontam_rows = Counter(
        row["source"] for row in rows if row.get("_drop") == "decontam"
    )
    decontam_items = Counter(i.split(":")[1] for i in bad_items)

    seen, dup = set(), Counter()
    for row in rows:
        if "_drop" in row:
            continue
        key = sha(json.dumps([row["state"], row["question"]], sort_keys=True), 24)
        if key in seen:
            row["_drop"] = "duplicate"
            dup[row["source"]] += 1
        seen.add(key)

    kept = [r for r in rows if "_drop" not in r]
    with Pool(args.workers, initializer=_tok_init, initargs=(args.tokenizer,)) as pool:
        counts = [
            c
            for part in pool.map(
                _tok_count,
                [kept[i : i + 256] for i in range(0, len(kept), 256)],
                chunksize=1,
            )
            for c in part
        ]
    too_long = Counter()
    final = []
    for row, c in zip(kept, counts):
        if c["n"] < 0 or c["n"] > MAX_TOKENS:
            too_long[row["source"]] += 1
            continue
        row["meta"]["n_tokens"] = c["n"]
        final.append(row)

    train, dev = [], []
    for row in sorted(final, key=lambda r: rank(r["id"], seed)):
        bucket = int(rank(row["meta"]["group"], seed)[:8], 16) % 1000
        (dev if bucket < DEV_PER_MILLE else train).append(row)

    partial = out.with_name(out.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    train_name = "train-00000-of-00001.jsonl.gz"
    write_jsonl(partial / train_name, train)
    write_jsonl(partial / "dev.jsonl.gz", dev)

    def composition(rs: list[dict]) -> dict[str, Any]:
        return {
            "part": dict(Counter(r["meta"]["part"] for r in rs)),
            "source": dict(Counter(r["source"] for r in rs)),
            "render": dict(Counter(r["meta"]["render"] for r in rs)),
            "orig_kind": dict(Counter(r["meta"]["orig_kind"] for r in rs)),
        }

    def yes_rates(rs: list[dict]) -> dict[str, Any]:
        groups: dict[str, list[bool]] = defaultdict(list)
        for r in rs:
            y = yes_of(r)
            groups["all"].append(y)
            groups[f"part:{r['meta']['part']}"].append(y)
            groups[f"source:{r['source']}:{r['meta']['part']}"].append(y)
        return {k: round(sum(v) / len(v), 4) for k, v in sorted(groups.items())}

    sata = [r for r in train if r["meta"]["part"] == "sk1"]
    item_shapes = {
        r["meta"]["item"]: (
            r["meta"]["sata"]["n_options"],
            r["meta"]["sata"]["n_correct"],
        )
        for r in sata
    }
    samples = {}
    for r in train:
        key = f"{r['source']}|{r['meta']['part']}|{r['meta']['render']}"
        if key not in samples:
            samples[key] = r
    write_json(partial / "samples.json", samples)
    manifest = {
        "name": out.name,
        "version": "v1",
        "kind": "d25-vega same-skill multi-answer (select all that apply) set, ws-sk1",
        "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "code": {"module": "d25.vega.data.sk1", "tag": os.environ.get("D25_CODE_TAG")},
        "seed": seed,
        "rows": {"train": len(train), "dev": len(dev)},
        "composition": {"train": composition(train), "dev": composition(dev)},
        "yes_rate": {"train": yes_rates(train), "dev": yes_rates(dev)},
        "sata_items": {
            "train_items": len(item_shapes),
            "n_options": dict(
                sorted(Counter(s[0] for s in item_shapes.values()).items())
            ),
            "n_correct": dict(
                sorted(Counter(s[1] for s in item_shapes.values()).items())
            ),
        },
        "tokens": {"train": describe_lengths([r["meta"]["n_tokens"] for r in train])},
        "sources": {
            name: {
                **{k: v for k, v in fetched[name].items() if k != "files"},
                "files_sha256": fetched[name]["files"],
                "rows_built": built.get(SOURCES[name][0], 0),
                "rows_kept": sum(1 for r in final if r["source"] == f"sk1:{name}"),
            }
            for name in SOURCES
        },
        "parse_stats": parse_stats,
        "sata_bench_exclusion": exclusion,
        "holdouts": {**holdouts.report(), "dropped_rows": dict(held)},
        "decontamination": {
            "index": str(args.index),
            "index_meta": {
                k: v
                for k, v in json.loads(
                    (Path(args.index) / "meta.json").read_text()
                ).items()
                if k in ("suite_files", "stats", "benchmarks", "normalisation")
            },
            "rule": json.loads((Path(args.index) / "rule.json").read_text()),
            "drop_policy": "a flagged row drops every row of its item",
            "flagged_rows_by_reason": dict(by_reason),
            "flagged_rows_by_benchmark": dict(by_bench.most_common()),
            "dropped_items_by_source": dict(decontam_items),
            "dropped_rows_by_source": dict(decontam_rows),
        },
        "duplicates_dropped": dict(dup),
        "too_long_dropped": dict(too_long),
        "dev_rule": f"group-disjoint: rank(meta.group) % 1000 < {DEV_PER_MILLE}",
        "files": {},
        "seconds": None,
    }
    for name in (train_name, "dev.jsonl.gz", "samples.json"):
        manifest["files"][name] = {
            "sha256": sha256_file(partial / name),
            "bytes": (partial / name).stat().st_size,
        }
    manifest["seconds"] = round(time.time() - started, 1)
    write_json(partial / "manifest.json", manifest)
    (partial / "VERIFIED").write_text(
        f"sk1 {manifest['created']} train {len(train)} dev {len(dev)}\n"
    )
    if out.exists():
        shutil.rmtree(out)
    os.replace(partial, out)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--raw", type=Path, required=True)
    b = sub.add_parser("build")
    b.add_argument("--raw", type=Path, required=True)
    b.add_argument("--index", type=Path, required=True)
    b.add_argument("--holdouts", type=Path, required=True)
    b.add_argument("--tokenizer", required=True)
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--seed", type=int, default=SEED)
    b.add_argument("--workers", type=int, default=os.cpu_count() or 8)
    args = parser.parse_args()
    if args.cmd == "fetch":
        fetch(args.raw)
        return
    manifest = build(args)
    print(
        json.dumps(
            {
                k: manifest[k]
                for k in ("rows", "yes_rate", "sata_items", "tokens", "seconds")
                if k in manifest
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
