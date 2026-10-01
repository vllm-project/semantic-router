"""IB3 families: pinned publisher rows -> native candidate rows (prereg ``records/ib3-prereg-2026-10-01.md`` §2).

Every family is Noul and is balanced yes = no inside every declared cell (``audit_metadata["ib3"]["cell"]``), so no
declared surface property predicts the label. ``hash_key`` is the stable upstream key hash order uses.
"""

from __future__ import annotations

import ast
import collections
import html
import math
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

from v2.data.ib1.families import maud_question
from v2.data.sources.common import make_row, noul_options, sha
from v2.data.textnorm import normalize

ARM = "ib3"
MAX_STATE_CHARS = 12_000
Rows = list[dict[str, Any]]

WPD_INSTRUCTIONS = (
    "Is this link a phishing link, that is, a web address set up to impersonate a trusted site or to steal "
    "logins, payment details or personal data?"
)
PHIU_INSTRUCTIONS = (
    "Is this web page a phishing page, that is, one set up to impersonate a trusted site or to steal logins, "
    "payment details or personal data?"
)
FDIAL_INSTRUCTIONS = (
    "Is every statement in the response supported by the knowledge passage? Claims about the speaker's own "
    "experiences, feelings or opinions count as unsupported unless the passage supports them; questions and "
    "greetings need no support."
)
HALUQA_INSTRUCTIONS = "Is the answer to the question supported by the passages?"
ESCI_INSTRUCTIONS = (
    "Is this product an exact match for the shopper's search query, that is, the item the query asks for with "
    "every attribute the query specifies?"
)
MQA_INSTRUCTIONS = "Is the proposed answer the correct option for this math problem?"
MAUD_INSTRUCTIONS = "Does the merger-agreement excerpt support the proposed answer to the deal-point question?"
TEMPLATE_STRINGS = frozenset(
    [
        WPD_INSTRUCTIONS,
        PHIU_INSTRUCTIONS,
        FDIAL_INSTRUCTIONS,
        HALUQA_INSTRUCTIONS,
        ESCI_INSTRUCTIONS,
        MQA_INSTRUCTIONS,
        MAUD_INSTRUCTIONS,
        "No",
        "Yes",
    ]
)
# The state field each construction could leak the label through (G4 hypothesis view, prereg §3).
HYPOTHESIS_FIELDS = {
    "fdial": "response",
    "haluqa": "answer",
    "esci": "product",
    "mqa": "proposed_answer",
    "maud": "proposed_answer",
}
FDIAL_CAP = 6000
ESCI_QUERIES = 6000
MQA_PROBLEMS = 5000
MAUD_CAP = 4000
MAUD_MAX_TEXT = 6000
ESCI_ABOUT_CHARS = 800

# --------------------------------------------------------------------------- helpers


def order(salt: str, key: str) -> str:
    return sha(f"{salt}:{key}")


def ranked(rows: Iterable[dict[str, Any]], salt: str) -> Rows:
    return sorted(
        rows,
        key=lambda r: (order(salt, r["audit_metadata"]["ib3"]["hash_key"]), r["id"]),
    )


def collapse(text: Any) -> str:
    return " ".join(str(text or "").split())


def noul(
    *,
    yes: bool,
    source: str,
    family: str,
    group_key: str,
    key: str,
    state: dict[str, Any],
    instructions: str,
    template: str,
    cell: str,
    language: str = "en",
    **extra: Any,
) -> dict[str, Any]:
    return make_row(
        arm=ARM,
        source=source,
        family=family,
        task_type="noul",
        language=language,
        group_key=group_key,
        local_id=f"{family}:{key}",
        state=state,
        instructions=instructions,
        options=noul_options("en"),
        label=1 if yes else 0,
        render_template=template,
        audit={"ib3": {"hash_key": key, "cell": cell, **extra}},
    )


def resolve(rows: Rows, report: collections.Counter) -> Rows:
    """Drop over-long states; one row per id; ids whose copies disagree on input or label are dropped."""
    copies: dict[str, Rows] = collections.defaultdict(list)
    for item in rows:
        if sum(len(str(v)) for v in item["state"].values()) > MAX_STATE_CHARS:
            report["drop_length"] += 1
            continue
        copies[item["id"]].append(item)
    kept = []
    for members in copies.values():
        if len({(r["input_sha256"], r["label"]) for r in members}) > 1:
            report["drop_conflicting_duplicates"] += len(members)
            continue
        report["drop_exact_duplicates"] += len(members) - 1
        kept.append(members[0])
    return kept


def per_group(rows: Rows, cap: int, salt: str) -> Rows:
    kept, taken = [], collections.Counter()
    for item in ranked(rows, salt):
        if taken[item["group_id"]] < cap:
            kept.append(item)
            taken[item["group_id"]] += 1
    return kept


def cell_of(row: Mapping[str, Any]) -> str:
    return str(row["audit_metadata"]["ib3"]["cell"])


def cell_balance(rows: Rows, cap: int | None, salt: str) -> Rows:
    """yes = no inside every cell (hash order); a cap scales every cell down proportionally."""
    by: dict[str, dict[int, Rows]] = collections.defaultdict(lambda: {0: [], 1: []})
    for item in ranked(rows, salt):
        by[cell_of(item)][item["label"]].append(item)
    sizes = {c: min(len(v[0]), len(v[1])) for c, v in by.items()}
    total = 2 * sum(sizes.values())
    if cap is not None and total > cap:
        sizes = {c: math.floor(n * cap / total) for c, n in sizes.items()}
    out: Rows = []
    for c in sorted(by):
        out += by[c][0][: sizes[c]] + by[c][1][: sizes[c]]
    return out


def bucket(value: int, bounds: Sequence[int], labels: Sequence[str]) -> str:
    for bound, label in zip(bounds, labels, strict=False):
        if value <= bound:
            return label
    return labels[-1]


def finish(
    rows: Rows, report: collections.Counter, select: Callable[[Rows], Rows]
) -> Rows:
    rows = resolve(rows, report)
    report["eligible"] = len(rows)
    chosen = select(rows)
    report["selected"] = len(chosen)
    report["selected_yes"] = sum(r["label"] for r in chosen)
    return chosen


# --------------------------------------------------------------------------- phishing links and pages

URL_RE = re.compile(r"^([A-Za-z][A-Za-z0-9+.-]*)://([^/?#]*)(.*)$", re.S)
IPV4_RE = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")
SHORTENERS = frozenset(
    """
    bit.ly bitly.com bit.do goo.gl tinyurl.com ow.ly t.co is.gd buff.ly adf.ly cutt.ly rb.gy shorturl.at tiny.cc
    rebrand.ly s.id v.gd x.co lnkd.in qr.net 1url.com tr.im cli.gs short.to budurl.com post.ly snipr.com fic.kr
    migre.me ff.im tiny.pl url4.eu tweez.me lnk.co kl.am wp.me u.to j.mp db.tt qr.ae adcrun.ch ity.im q.gs po.st
    bc.vc u.bb yourls.org scrnch.me filoops.info vzturl.com link.zip.net shorte.st ouo.io clck.ru t.ly tiny.one
    rotf.lol shrtco.de 2no.co bl.ink soo.gd
    """.split()
)
WPD_LENGTHS = ((25, 35, 50, 80, 200), ("<=25", "26-35", "36-50", "51-80", "81-200"))
PHIU_LENGTHS = ((25, 35, 50, 80), ("<=25", "26-35", "36-50", "51-80"))
BAD_TITLE = re.compile(
    r"not found|\b404\b|\b403\b|suspended|phishing|deceptive|forbidden|access denied|\berror\b|for sale|parked|"
    r"coming soon|under construction|index of|just a moment|attention required|default page|welcome to nginx|"
    r"it works",
    re.I,
)


def url_parts(url: str) -> tuple[str, str, bool, bool] | None:
    """(scheme, host without a leading ``www.``, ``www.`` present, path or query present) or None."""
    match = URL_RE.match(url)
    if not match:
        return None
    scheme, netloc, rest = match.groups()
    scheme = scheme.lower()
    host = netloc.rsplit("@", 1)[-1].split(":", 1)[0].lower().rstrip(".")
    if scheme not in ("http", "https") or not host:
        return None
    www = host.startswith("www.")
    return scheme, host[4:] if www else host, www, rest not in ("", "/")


def wpd(records: Sequence[Mapping[str, str]], report: collections.Counter) -> Rows:
    rows: Rows = []
    for record in records:
        report["read"] += 1
        url = str(record.get("url") or "").strip()
        status = str(record.get("status") or "").strip()
        if not url or status not in ("phishing", "legitimate"):
            report["drop_shape"] += 1
            continue
        if len(url) > 200:
            report["drop_url_length"] += 1
            continue
        if IPV4_RE.search(url):
            report["drop_ipv4"] += 1
            continue
        parts = url_parts(url)
        if parts is None:
            report["drop_url_parse"] += 1
            continue
        scheme, host, www, path = parts
        if host in SHORTENERS:
            report["drop_shortener"] += 1
            continue
        cell = "|".join(
            (
                scheme,
                "www" if www else "nowww",
                "path" if path else "nopath",
                bucket(len(url), *WPD_LENGTHS),
            )
        )
        rows.append(
            noul(
                yes=status == "phishing",
                source="mendeley_web_page_phishing",
                family="wpd",
                group_key="wpd:" + host,
                key=sha(url)[:24],
                state={"url": url},
                instructions=WPD_INSTRUCTIONS,
                template="ib3_wpd_v1",
                cell=cell,
            )
        )
    return finish(
        rows,
        report,
        lambda rs: cell_balance(
            per_group(rs, 2, "ib3-wpd-group-v1"), None, "ib3-wpd-v1"
        ),
    )


PHIU_URL_RE = re.compile(r"https://www\.([^/?#:@\s]+?)/?")


def phiu(records: Sequence[Mapping[str, str]], report: collections.Counter) -> Rows:
    rows: Rows = []
    for record in records:
        report["read"] += 1
        url = str(record.get("URL") or "").strip()
        label = str(record.get("label") or "").strip()
        title = collapse(html.unescape(str(record.get("Title") or "")))
        match = PHIU_URL_RE.fullmatch(url)
        if label not in ("0", "1"):
            report["drop_shape"] += 1
            continue
        if not match or len(url) > 80:
            report["drop_shape_cell"] += 1
            continue
        if IPV4_RE.search(url):
            report["drop_ipv4"] += 1
            continue
        if not title or len(title) > 200 or BAD_TITLE.search(title):
            report["drop_title"] += 1
            continue
        host = match.group(1).lower().rstrip(".")
        rows.append(
            noul(
                yes=label == "0",
                source="uci_phiusiil",
                family="phiu",
                group_key="phiu:" + host,
                key=sha(url)[:24],
                state={"url": url, "page_title": title},
                instructions=PHIU_INSTRUCTIONS,
                template="ib3_phiu_v1",
                cell=bucket(len(url), *PHIU_LENGTHS),
            )
        )
    return finish(
        rows,
        report,
        lambda rs: cell_balance(
            per_group(rs, 1, "ib3-phiu-group-v1"), None, "ib3-phiu-v1"
        ),
    )


# --------------------------------------------------------------------------- grounding

FIRST_PERSON = re.compile(r"\b(?:i|i'm|i've|i'd|i'll|me|my|mine|myself)\b", re.I)
FDIAL_LENGTHS = ((10, 20, 30, 10**9), ("<=10", "11-20", "21-30", ">30"))
HALU_LENGTHS = ((1, 2, 3, 4, 6, 12), ("1", "2", "3", "4", "5-6", "7-12"))


def fdial(dialogues: Sequence[Mapping[str, Any]], report: collections.Counter) -> Rows:
    rows: Rows = []
    for dialogue in dialogues:
        for index, turn in enumerate(dialogue.get("utterances") or []):
            report["read"] += 1
            begin = tuple(sorted(turn.get("BEGIN") or []))
            if begin not in (("Entailment",), ("Hallucination",)):
                report["drop_begin_boundary"] += 1
                continue
            original = turn.get("original_response")
            text = collapse(original if original is not None else turn.get("response"))
            knowledge = collapse(turn.get("knowledge"))
            history = turn.get("history") or []
            previous = collapse(history[-1]) if history else ""
            if not text or not knowledge:
                report["drop_shape"] += 1
                continue
            state = {"knowledge": knowledge}
            if previous:
                state["previous_turn"] = previous
            state["response"] = text
            cell = "|".join(
                (
                    "fp" if FIRST_PERSON.search(text) else "nofp",
                    bucket(len(text.split()), *FDIAL_LENGTHS),
                )
            )
            rows.append(
                noul(
                    yes=begin == ("Entailment",),
                    source="faithdial_train",
                    family="fdial",
                    group_key="fdial:" + str(dialogue.get("dialog_idx")),
                    key=f"{dialogue.get('dialog_idx')}:{index}",
                    state=state,
                    instructions=FDIAL_INSTRUCTIONS,
                    template="ib3_fdial_v1",
                    cell=cell,
                )
            )
    return finish(
        rows,
        report,
        lambda rs: cell_balance(
            per_group(rs, 2, "ib3-fdial-group-v1"), FDIAL_CAP, "ib3-fdial-v1"
        ),
    )


def haluqa(records: Sequence[Mapping[str, Any]], report: collections.Counter) -> Rows:
    rows: Rows = []
    for record in records:
        report["read"] += 1
        question = collapse(record.get("question"))
        passages = collapse(record.get("knowledge"))
        right = collapse(record.get("right_answer"))
        wrong = collapse(record.get("hallucinated_answer"))
        if not question or not passages:
            report["drop_shape"] += 1
            continue
        if normalize(right) == normalize(wrong):
            report["drop_same_answers"] += 1
            continue
        for yes, answer in ((True, right), (False, wrong)):
            if not answer or answer.endswith(".") or len(answer.split()) > 12:
                report[f"drop_answer_form_{'right' if yes else 'hallucinated'}"] += 1
                continue
            rows.append(
                noul(
                    yes=yes,
                    source="halueval_qa",
                    family="haluqa",
                    group_key="haluqa:" + normalize(question),
                    key=sha(question + "\x1f" + answer + "\x1f" + str(yes))[:24],
                    state={
                        "passages": passages,
                        "question": question,
                        "answer": answer,
                    },
                    instructions=HALUQA_INSTRUCTIONS,
                    template="ib3_haluqa_v1",
                    cell=bucket(len(answer.split()), *HALU_LENGTHS),
                )
            )
    return finish(rows, report, lambda rs: cell_balance(rs, None, "ib3-haluqa-v1"))


# --------------------------------------------------------------------------- product search

ESCI_LANGUAGE = {"us": "en", "es": "es", "jp": "ja"}


def esci_product(product: Mapping[str, Any]) -> str | None:
    title = collapse(product.get("product_title"))
    if not title:
        return None
    lines = [f"Title: {title}"]
    for name, field in (("Brand", "product_brand"), ("Color", "product_color")):
        value = collapse(product.get(field))
        if value:
            lines.append(f"{name}: {value}")
    about = collapse(product.get("product_bullet_point"))[:ESCI_ABOUT_CHARS].strip()
    if about:
        lines.append(f"About: {about}")
    return "\n".join(lines)


def esci(
    examples: Sequence[Mapping[str, Any]],
    products: Mapping[tuple[str, str], Mapping[str, Any]],
    report: collections.Counter,
) -> Rows:
    by_query: dict[str, dict[str, list[Mapping[str, Any]]]] = collections.defaultdict(
        lambda: {"E": [], "I": []}
    )
    for example in examples:
        report["read"] += 1
        if example.get("split") != "train":
            report["drop_not_train"] += 1
            continue
        if example.get("esci_label") not in ("E", "I"):
            report["drop_boundary_class"] += 1
            continue
        by_query[str(example["query_id"])][example["esci_label"]].append(example)
    rows: Rows = []
    queries = sorted(
        (q for q, v in by_query.items() if v["E"] and v["I"]),
        key=lambda q: (order("ib3-esci-v1", q), q),
    )
    report["queries_with_both"] = len(queries)
    for query_id in queries:
        if len(rows) >= 2 * ESCI_QUERIES:
            break
        twins = []
        for label in ("E", "I"):
            pick = min(
                by_query[query_id][label],
                key=lambda e: (
                    order("ib3-esci-pick-v1", str(e["example_id"])),
                    str(e["example_id"]),
                ),
            )
            product = products.get(
                (str(pick["product_id"]), str(pick["product_locale"]))
            )
            text = esci_product(product) if product else None
            query = collapse(pick.get("query"))
            if not text or not query:
                break
            twins.append(
                noul(
                    yes=label == "E",
                    source="amazon_esci_train",
                    family="esci",
                    language=ESCI_LANGUAGE.get(str(pick["product_locale"]), "en"),
                    group_key="esci:" + query_id,
                    key=f"{query_id}:{pick['example_id']}",
                    state={"query": query, "product": text},
                    instructions=ESCI_INSTRUCTIONS,
                    template="ib3_esci_v1",
                    cell=query_id,
                    locale=str(pick["product_locale"]),
                )
            )
        if len(twins) == 2:
            rows += twins
        else:
            report["drop_query_missing_product"] += 1
    return finish(rows, report, lambda rs: cell_balance(rs, None, "ib3-esci-bal-v1"))


# --------------------------------------------------------------------------- maths MCQ (per-option check)

NUM_RE = re.compile(r"\d+(?:\.\d+)?")
OPTION_RE = re.compile(r"([a-e]) \) (.*?)(?= , [a-e] \) |\s*$)", re.S)
NON_NUMERIC = re.compile(
    r"none|inadequate|cannot|can not|not determined|insufficient|√|sqrt|π|pi\b", re.I
)
CONSTANTS = {"const_pi": math.pi, "const_deg_to_rad": math.pi / 180}


def plain_numbers(text: str) -> list[float]:
    return [float(n) for n in NUM_RE.findall(re.sub(r"(?<=\d),(?=\d)", "", text))]


def option_value(text: str) -> float | None:
    """The single number an option states (a / b as a fraction; ratios and other shapes give None)."""
    if NON_NUMERIC.search(text) or ":" in text:
        return None
    clean = re.sub(r"(?<=\d),(?=\d)", "", text)
    numbers = NUM_RE.findall(clean)
    negative = bool(re.match(r"^\s*-\s*\d", clean))
    if len(numbers) == 1:
        value = float(numbers[0])
    elif len(numbers) == 2 and re.fullmatch(
        r"[^\d]*\d+(?:\.\d+)?\s*/\s*\d+(?:\.\d+)?[^\d]*", clean
    ):
        if float(numbers[1]) == 0:
            return None
        value = float(numbers[0]) / float(numbers[1])
    else:
        return None
    return -value if negative else value


def parse_options(raw: str) -> list[tuple[str, str]]:
    text = raw.strip()
    if text.startswith("["):
        try:
            items = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return []
        text = " , ".join(str(item).strip() for item in items)
    return [
        (letter, " ".join(value.split())) for letter, value in OPTION_RE.findall(text)
    ]


def _int(value: float) -> int:
    if abs(value - round(value)) > 1e-9 or abs(value) > 10**6:
        raise ValueError("not a small integer")
    return int(round(value))


def _heron(a: float, b: float, c: float) -> float:
    s = (a + b + c) / 2
    return math.sqrt(s * (s - a) * (s - b) * (s - c))


OPS: dict[str, Callable[..., float]] = {
    "add": lambda a, b: a + b,
    "subtract": lambda a, b: a - b,
    "multiply": lambda a, b: a * b,
    "divide": lambda a, b: a / b,
    "power": lambda a, b: a**b,
    "sqrt": math.sqrt,
    "inverse": lambda a: 1 / a,
    "negate": lambda a: -a,
    "factorial": lambda a: float(math.factorial(_int(a))) if 0 <= a <= 50 else math.nan,
    "floor": lambda a: float(math.floor(a)),
    "reminder": lambda a, b: a % b,
    "choose": lambda a, b: float(math.comb(_int(a), _int(b))),
    "permutation": lambda a, b: float(math.perm(_int(a), _int(b))),
    "lcm": lambda a, b: float(math.lcm(_int(a), _int(b))),
    "gcd": lambda a, b: float(math.gcd(_int(a), _int(b))),
    "max": max,
    "min": min,
    "log": math.log,
    "rectangle_area": lambda a, b: a * b,
    "rectangle_perimeter": lambda a, b: 2 * (a + b),
    "square_area": lambda a: a * a,
    "square_perimeter": lambda a: 4 * a,
    "square_edge_by_perimeter": lambda a: a / 4,
    "square_edge_by_area": math.sqrt,
    "circle_area": lambda r: math.pi * r * r,
    "circumface": lambda r: 2 * math.pi * r,
    "volume_cube": lambda a: a**3,
    "surface_cube": lambda a: 6 * a * a,
    "cube_edge_by_volume": lambda v: v ** (1 / 3),
    "volume_cylinder": lambda r, h: math.pi * r * r * h,
    "surface_cylinder": lambda r, h: 2 * math.pi * r * (r + h),
    "volume_rectangular_prism": lambda a, b, c: a * b * c,
    "surface_rectangular_prism": lambda a, b, c: 2 * (a * b + b * c + a * c),
    "triangle_area": lambda b, h: b * h / 2,
    "triangle_perimeter": lambda a, b, c: a + b + c,
    "triangle_area_three_edges": _heron,
    "rhombus_area": lambda d1, d2: d1 * d2 / 2,
    "rhombus_perimeter": lambda a: 4 * a,
    "quadrilateral_area": lambda d, h1, h2: d * (h1 + h2) / 2,
    "diagonal": lambda a, b: math.sqrt(a * a + b * b),
    "speed": lambda d, t: d / t,
    "speed_in_still_water": lambda a, b: (a + b) / 2,
    "stream_speed": lambda a, b: (a - b) / 2,
    "negate_prob": lambda p: 1 - p,
    "volume_sphere": lambda r: 4 / 3 * math.pi * r**3,
    "surface_sphere": lambda r: 4 * math.pi * r * r,
    "volume_cone": lambda r, h: math.pi * r * r * h / 3,
    "p_after_gain": lambda g, p: p * (1 + g / 100),
    "original_price_before_gain": lambda g, p: p / (1 + g / 100),
    "original_price_before_loss": lambda g, p: p / (1 - g / 100),
}


def run_formula(formula: str, numbers: Sequence[float]) -> float | None:
    """Value of a MathQA ``linear_formula`` (steps ``op(arg,...)`` joined by ``|``; ``#k`` is step k's result)."""
    results: list[float] = []
    for step in [s for s in formula.strip().strip("|").split("|") if s.strip()]:
        match = re.fullmatch(r"\s*([a-z_]+)\((.*)\)\s*", step)
        if not match or match.group(1) not in OPS:
            return None
        args = []
        for token in [t.strip() for t in match.group(2).split(",") if t.strip()]:
            if re.fullmatch(r"n\d+", token):
                index = int(token[1:])
                if index >= len(numbers):
                    return None
                args.append(numbers[index])
            elif re.fullmatch(r"#\d+", token):
                index = int(token[1:])
                if index >= len(results):
                    return None
                args.append(results[index])
            elif token in CONSTANTS:
                args.append(CONSTANTS[token])
            elif re.fullmatch(r"const_\d+(?:_\d+)?", token):
                args.append(float(token[6:].replace("_", ".", 1)))
            else:
                return None
        try:
            value = float(OPS[match.group(1)](*args))
        except (ArithmeticError, ValueError, TypeError, OverflowError):
            return None
        if not math.isfinite(value):
            return None
        results.append(value)
    return results[-1] if results else None


def close(a: float, b: float, rel: float, absolute: float) -> bool:
    return abs(a - b) <= max(absolute, rel * abs(b))


def detok(text: str) -> str:
    return re.sub(r" ([.,?!%])", r"\1", " ".join(text.split()))


def mqa(records: Sequence[Mapping[str, Any]], report: collections.Counter) -> Rows:
    problems = []
    for record in records:
        report["read"] += 1
        problem = " ".join(str(record.get("Problem") or "").split())
        options = parse_options(str(record.get("options") or ""))
        key = str(record.get("correct") or "").strip()
        letters = [letter for letter, _ in options]
        if not problem or letters != ["a", "b", "c", "d", "e"] or key not in letters:
            report["drop_shape"] += 1
            continue
        values = [option_value(text) for _, text in options]
        if any(v is None for v in values):
            report["drop_non_numeric_option"] += 1
            continue
        result = run_formula(
            str(record.get("linear_formula") or ""), plain_numbers(problem)
        )
        if result is None:
            report["drop_formula_not_executable"] += 1
            continue
        matches = [i for i, v in enumerate(values) if close(v, result, 0.01, 0.01)]
        if matches != [letters.index(key)]:
            report["drop_key_not_verified"] += 1
            continue
        wrong = [
            i
            for i, v in enumerate(values)
            if i != matches[0] and not close(v, result, 0.05, 0.01)
        ]
        if not wrong:
            report["drop_no_distant_option"] += 1
            continue
        problems.append((sha(problem)[:24], problem, options, matches[0], wrong))
    report["verified"] = len(problems)
    problems.sort(key=lambda p: (order("ib3-mqa-v1", p[0]), p[0]))
    rows: Rows = []
    for pkey, problem, options, gold, wrong in problems[:MQA_PROBLEMS]:
        other = min(wrong, key=lambda i: order("ib3-mqa-other-v1", f"{pkey}:{i}"))
        shown = detok(problem)
        listing = "; ".join(f"{letter}) {text}" for letter, text in options)
        for yes, index in ((True, gold), (False, other)):
            rows.append(
                noul(
                    yes=yes,
                    source="mathqa_train",
                    family="mqa",
                    group_key="mqa:" + normalize(problem),
                    key=f"{pkey}:{'key' if yes else 'other'}",
                    state={
                        "problem": shown,
                        "options": listing,
                        "proposed_answer": options[index][1],
                    },
                    instructions=MQA_INSTRUCTIONS,
                    template="ib3_mqa_v1",
                    cell=pkey,
                )
            )
    return finish(rows, report, lambda rs: cell_balance(rs, None, "ib3-mqa-bal-v1"))


# --------------------------------------------------------------------------- contracts (per-option check)


def maud(records: Sequence[Mapping[str, str]], report: collections.Counter) -> Rows:
    answers: dict[tuple[str, str], set[str]] = collections.defaultdict(set)
    for record in records:
        if record.get("data_type") == "main" and record.get("answer", "").strip():
            answers[(record["question"], record["subquestion"])].add(
                record["answer"].strip()
            )
    rows: Rows = []
    for record in records:
        report["read"] += 1
        if record.get("data_type") != "main":
            report["drop_data_type"] += 1
            continue
        options = sorted(answers[(record["question"], record["subquestion"])])
        answer = str(record.get("answer") or "").strip()
        text = str(record.get("text") or "").strip()
        if not 2 <= len(options) <= 6 or answer not in options:
            report["drop_option_count"] += 1
            continue
        if not text or len(text) > MAUD_MAX_TEXT:
            report["drop_text_length"] += 1
            continue
        point = maud_question(record["question"], record["subquestion"])
        key = sha(record["contract_name"] + "\x1f" + point + "\x1f" + text)[:24]
        other = min(
            (o for o in options if o != answer),
            key=lambda o: order("ib3-maud-other-v1", f"{key}:{o}"),
        )
        for yes, proposed in ((True, answer), (False, other)):
            rows.append(
                noul(
                    yes=yes,
                    source="maud_train_main",
                    family="maud",
                    group_key="maud:" + record["contract_name"],
                    key=f"{key}:{'gold' if yes else 'other'}",
                    state={
                        "excerpt": text,
                        "deal_point": point,
                        "proposed_answer": proposed,
                    },
                    instructions=MAUD_INSTRUCTIONS,
                    template="ib3_maud_v1",
                    cell=point + "\x1f" + proposed,
                )
            )
    return finish(rows, report, lambda rs: cell_balance(rs, MAUD_CAP, "ib3-maud-v1"))
