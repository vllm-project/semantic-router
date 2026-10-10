"""KIE (CORD+FUNSD): 2,465 rows = 1,059 CORD receipt-field questions (4-way) + 1,406 FUNSD
form-field presence questions (yes / no).

The board's chance sum is 967.75, which with 2- and 4-option rows means exactly 1,406 binary and 1,059
four-way rows; the board text pairs "receipt fields" (multiple choice) with "form-field presence".
CORD test receipts: one question per annotated field (summary amounts, item prices, unit prices
and counts) with three distractors of the same kind from the same receipt, padded with same-format
values of the same field from other test receipts; 1,059 questions are taken round-robin over
receipts (summary fields first). FUNSD test forms: a present field is a QUESTION entity label; an
absent field is a label from another test form whose words never occur on this form; 703 present
labels are taken round-robin over forms, each form gets as many absent labels as present ones.
All candidates are kept in the ``kie-cord-all`` and ``kie-funsd-all`` variants.
"""

from __future__ import annotations

import glob
import io
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from d25.omni.suite import rows as R

BENCHMARK = "KIE (CORD+FUNSD)"
SOURCES = ("cord", "funsd")
CORD_ROWS, FUNSD_POSITIVES = 1059, 703

SUMMARY = (
    ("total", "total_price", "money", "What is the total price on this receipt?"),
    ("sub_total", "subtotal_price", "money", "What is the subtotal on this receipt?"),
    ("sub_total", "tax_price", "money", "What is the tax amount on this receipt?"),
    (
        "sub_total",
        "service_price",
        "money",
        "What is the service charge on this receipt?",
    ),
    (
        "sub_total",
        "discount_price",
        "money",
        "What is the discount amount on this receipt?",
    ),
    (
        "sub_total",
        "othersvc_price",
        "money",
        "What is the other service charge on this receipt?",
    ),
    ("total", "cashprice", "money", "How much cash was paid?"),
    ("total", "changeprice", "money", "How much change was given?"),
    ("total", "creditcardprice", "money", "How much was paid by card?"),
    ("total", "emoneyprice", "money", "How much was paid by e-money?"),
    ("total", "menuqty_cnt", "count", "How many items were bought in total?"),
    ("total", "menutype_cnt", "count", "How many different menu items were bought?"),
)
ITEM = (
    ("price", "money", "What is the price of “{nm}” on this receipt?"),
    ("unitprice", "money", "What is the unit price of “{nm}” on this receipt?"),
    ("cnt", "count", "How many “{nm}” were bought?"),
)
FUNSD_TAGS = [
    "O",
    "B-HEADER",
    "I-HEADER",
    "B-QUESTION",
    "I-QUESTION",
    "B-ANSWER",
    "I-ANSWER",
]
PRESENCE = "Does this form have a field labelled “{label}”?"
PRESENCE_CRITERIA = {
    "yes": "Yes, the form has this field.",
    "no": "No, the form does not have this field.",
}


def _first(value):
    if isinstance(value, list):
        value = value[0] if value else None
    return value.strip() if isinstance(value, str) and value.strip() else None


def digits(text: str) -> str:
    return re.sub(r"\D", "", text)


def shape(text: str) -> str:
    return re.sub(r"\d", "9", text)


def norm_words(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


# ---- CORD -------------------------------------------------------------------------------------


def cord_receipts(ctx):
    root = ctx.source("cord")
    for path in sorted(glob.glob(str(root / "data" / "test-*.parquet"))):
        name = str(Path(path).relative_to(root))
        for index, r in enumerate(pq.ParquetFile(path).read().to_pylist()):
            yield name, index, r


def cord_fields(gt: dict):
    """(field, kind, question, value) for every templated field of one receipt."""
    out = []
    for group, key, kind, question in SUMMARY:
        block = gt.get(group)
        value = _first(block.get(key)) if isinstance(block, dict) else None
        if value and digits(value):
            out.append((f"{group}.{key}", kind, question, value))
    menu = gt.get("menu") or []
    menu = menu if isinstance(menu, list) else [menu]
    names = Counter(_first(m.get("nm")) for m in menu if isinstance(m, dict))
    for m in menu:
        if not isinstance(m, dict):
            continue
        nm = _first(m.get("nm"))
        if not nm or names[nm] != 1:
            continue
        for key, kind, template in ITEM:
            value = _first(m.get(key))
            if value and digits(value):
                out.append((f"menu.{key}", kind, template.format(nm=nm), value))
    return out


def cord_build(ctx):
    receipts = []
    for name, index, r in cord_receipts(ctx):
        gt = json.loads(r["ground_truth"])["gt_parse"]
        receipts.append(
            {
                "file": name,
                "index": index,
                "image": r["image"]["bytes"],
                "fields": cord_fields(gt),
            }
        )
    by_field = defaultdict(list)
    for rec in receipts:
        for field, kind, _, value in rec["fields"]:
            by_field[field].append((rec["index"], value))
    candidates = []
    for rec in receipts:
        same = defaultdict(dict)
        for _, kind, _, value in rec["fields"]:
            same[kind].setdefault(digits(value), value)
        queue = []
        for priority, (field, kind, question, value) in enumerate(rec["fields"]):
            rng = R.stable_rng("cord", rec["index"], field, question)
            pool = [v for d, v in sorted(same[kind].items()) if d != digits(value)]
            rng.shuffle(pool)
            distractors = pool[:3]
            if len(distractors) < 3:
                used = {digits(value)} | {digits(v) for v in distractors}
                others = [
                    v
                    for i, v in by_field[field]
                    if i != rec["index"] and digits(v) not in used
                ]
                others = sorted(
                    set(others), key=lambda v: (shape(v) != shape(value), rng.random())
                )
                for v in others:
                    if digits(v) not in used:
                        distractors.append(v)
                        used.add(digits(v))
                    if len(distractors) == 3:
                        break
            if kind == "count" and len(distractors) < 3:
                n = int(digits(value))
                for d in (1, -1, 2, -2, 3):
                    if n + d >= 1 and str(n + d) not in {
                        digits(x) for x in distractors
                    } | {digits(value)}:
                        distractors.append(str(n + d))
                    if len(distractors) == 3:
                        break
            if len(distractors) < 3:
                continue
            options = [value] + distractors
            rng.shuffle(options)
            queue.append(
                (
                    priority,
                    field,
                    kind,
                    question,
                    options,
                    R.LETTERS[options.index(value)],
                    sum(1 for d in distractors if d in pool),
                )
            )
        candidates.append((rec, queue))
    rows_all = []
    for rec, queue in candidates:
        ref = None
        for priority, field, kind, question, options, gold, own in queue:
            ref = ref or ctx.store(rec["image"])
            source_id = f"{rec['index']}-{priority}"
            rows_all.append(
                R.make_row(
                    benchmark=BENCHMARK,
                    split="test",
                    subtask="CORD",
                    source_id=source_id,
                    images=[ref],
                    instructions=question,
                    criteria=R.letter_criteria(options),
                    gold=gold,
                    provenance=ctx.provenance(
                        "cord", rec["file"], source_id, receipt_index=rec["index"]
                    ),
                    tags=[f"field:{field}", f"kind:{kind}", f"own_distractors:{own}"],
                    extra={
                        "field": field,
                        "receipt_index": rec["index"],
                        "order": priority,
                    },
                )
            )
    by_receipt = defaultdict(list)
    for row in rows_all:
        by_receipt[row["metadata"]["receipt_index"]].append(row)
    order = sorted(by_receipt, key=lambda i: R.stable_rng("cord-receipt", i).random())
    chosen, depth = [], 0
    while len(chosen) < CORD_ROWS and any(depth < len(by_receipt[i]) for i in order):
        for i in order:
            if depth < len(by_receipt[i]) and len(chosen) < CORD_ROWS:
                chosen.append(by_receipt[i][depth])
        depth += 1
    keep = {r["id"] for r in chosen}
    return [r for r in rows_all if r["id"] in keep], rows_all


# ---- FUNSD ------------------------------------------------------------------------------------


def funsd_forms(ctx):
    root = ctx.source("funsd")
    for path in sorted(glob.glob(str(root / "data" / "test-*.parquet"))):
        name = str(Path(path).relative_to(root))
        for r in pq.ParquetFile(path).read().to_pylist():
            yield name, r


def form_labels(r) -> tuple[list[str], str]:
    """Distinct QUESTION labels (source text, first occurrence) and the form's normalized text."""
    labels, current = [], None
    for word, tag in zip(r["words"], r["ner_tags"]):
        name = FUNSD_TAGS[tag]
        if name == "B-QUESTION":
            current = [word]
            labels.append(current)
        elif name == "I-QUESTION" and current is not None:
            current.append(word)
        else:
            current = None
    seen, out = set(), []
    for words in labels:
        text = " ".join(words).strip()
        key = norm_words(text)
        if len(re.sub(r"[^a-z]", "", key)) >= 2 and key not in seen:
            seen.add(key)
            out.append(text)
    return out, " " + norm_words(" ".join(r["words"])) + " "


def funsd_build(ctx):
    forms = []
    for name, r in funsd_forms(ctx):
        labels, text = form_labels(r)
        forms.append(
            {
                "file": name,
                "id": str(r["id"]),
                "image": r["image"]["bytes"],
                "labels": labels,
                "text": text,
            }
        )
    pool = Counter()
    first_text = {}
    for f in forms:
        for label in f["labels"]:
            pool[norm_words(label)] += 1
            first_text.setdefault(norm_words(label), label)
    order = sorted(
        range(len(forms)),
        key=lambda i: R.stable_rng("funsd-form", forms[i]["id"]).random(),
    )
    shuffled = {}
    for i in order:
        labels = list(forms[i]["labels"])
        R.stable_rng("funsd-labels", forms[i]["id"]).shuffle(labels)
        shuffled[i] = labels
    positives = defaultdict(list)
    depth, taken = 0, 0
    while taken < FUNSD_POSITIVES and any(depth < len(shuffled[i]) for i in order):
        for i in order:
            if depth < len(shuffled[i]) and taken < FUNSD_POSITIVES:
                positives[i].append(shuffled[i][depth])
                taken += 1
        depth += 1
    board, everything = [], []
    for i, f in enumerate(forms):
        ref = ctx.store(f["image"])
        absent = sorted(
            k
            for k in pool
            if f" {k} " not in f["text"]
            and not any(k in norm_words(l) or norm_words(l) in k for l in f["labels"])
        )
        rng = R.stable_rng("funsd-absent", f["id"])
        weights = [pool[k] for k in absent]
        negatives_all = []
        remaining = list(zip(absent, weights))
        while remaining and len(negatives_all) < len(f["labels"]):
            total = sum(w for _, w in remaining)
            x, acc = rng.random() * total, 0.0
            for j, (k, w) in enumerate(remaining):
                acc += w
                if acc >= x:
                    negatives_all.append(first_text[k])
                    remaining.pop(j)
                    break
        for label in f["labels"]:
            row = _presence_row(ctx, f, ref, label, True)
            everything.append(row)
            if label in positives.get(i, []):
                board.append(row)
        for j, label in enumerate(negatives_all):
            row = _presence_row(ctx, f, ref, label, False)
            everything.append(row)
            if j < len(positives.get(i, [])):
                board.append(row)
    return board, everything


def _presence_row(ctx, form, ref, label, present):
    source_id = f"{form['id']}-{'yes' if present else 'no'}-{norm_words(label).replace(' ', '_')}"
    return R.make_row(
        benchmark=BENCHMARK,
        split="test",
        subtask="FUNSD",
        source_id=source_id,
        images=[ref],
        instructions=PRESENCE.format(label=label),
        criteria=PRESENCE_CRITERIA,
        gold="yes" if present else "no",
        provenance=ctx.provenance("funsd", form["file"], form["id"]),
        tags=[f"presence:{'yes' if present else 'no'}"],
        extra={"label": label, "form_id": form["id"]},
    )


def build(ctx) -> dict:
    cord_rows, cord_all = cord_build(ctx)
    funsd_rows, funsd_all = funsd_build(ctx)
    return {
        "rows": cord_rows + funsd_rows,
        "variants": {"kie-cord-all": cord_all, "kie-funsd-all": funsd_all},
        "notes": f"CORD test {len(cord_rows)} receipt-field MC of {len(cord_all)}; FUNSD test {len(funsd_rows)} presence rows of {len(funsd_all)}",
    }
