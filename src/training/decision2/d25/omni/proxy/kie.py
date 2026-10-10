"""KIE proxy (CORD + FUNSD style): synthetic receipts and business forms with known fields.

Receipts are rendered as HTML (thermal-paper layouts, several currencies and line formats), then
photographed onto a textured surface with perspective, light falloff and noise. Forms are typed
business forms (fax covers, memos, requests) with filled, blank and checkbox fields, degraded like a
photocopy scan. Questions ask for receipt fields (total, subtotal, tax, change, item price and
quantity, item counts) and for form-field presence, values and blanks, as 4-way multiple choice with
distractors taken from the same document.
"""

from __future__ import annotations

import datetime as dt
import html as htmlmod
import io
import random
from typing import Any

from PIL import Image

BENCHMARK = "KIE (CORD+FUNSD)"
NAME = "kie-proxy"
VERSION = "1"

STORE_WORDS = [
    "Kopi",
    "Warung",
    "Bakery",
    "Mart",
    "Bistro",
    "Cafe",
    "Noodle House",
    "Sushi Bar",
    "Pizzeria",
    "Grill",
    "Deli",
    "Tea House",
    "Dim Sum",
    "Burger Co.",
    "Ramen",
    "Juice Bar",
    "Kitchen",
    "Eatery",
]
STORE_NAMES = [
    "Sari",
    "Mawar",
    "Golden",
    "Lucky",
    "Blue Door",
    "Harbor",
    "Maple",
    "Sunrise",
    "Kenanga",
    "Melati",
    "Orchid",
    "Riverside",
    "Old Town",
    "Bamboo",
    "Lotus",
    "Corner",
    "Urban",
    "Hilltop",
    "Saffron",
    "Pandan",
    "Cempaka",
    "Tulip",
    "Garuda",
    "Mercury",
    "Nusa",
    "Bintang",
]
STREETS = [
    "Jl. Sudirman",
    "Jl. Thamrin",
    "Jl. Gatot Subroto",
    "Main St.",
    "Oak Ave.",
    "Market St.",
    "Jl. Merdeka",
    "High Street",
    "Elm Rd.",
    "Jl. Diponegoro",
    "Bay Blvd.",
    "Pine St.",
    "Jl. Ahmad Yani",
]
CITIES = [
    "Jakarta",
    "Bandung",
    "Surabaya",
    "Denpasar",
    "Springfield",
    "Riverton",
    "Kuala Lumpur",
    "Penang",
    "Yogyakarta",
    "Medan",
    "Portland",
    "Leeds",
    "Malang",
    "Semarang",
]
MENU = {
    "food": [
        "Nasi Goreng",
        "Mie Ayam",
        "Sate Ayam",
        "Gado Gado",
        "Ayam Bakar",
        "Soto Betawi",
        "Bakso",
        "Chicken Katsu",
        "Beef Burger",
        "Cheese Pizza",
        "Caesar Salad",
        "Fish & Chips",
        "Club Sandwich",
        "Spaghetti Bolognese",
        "Pad Thai",
        "Fried Rice",
        "Ramen Shoyu",
        "Gyoza (5)",
        "Spring Rolls",
        "French Fries",
        "Onion Rings",
        "Chicken Wings",
        "Tom Yum",
        "Beef Rendang",
        "Nasi Uduk",
        "Pisang Goreng",
        "Roti Bakar",
        "Croissant",
        "Pain au Choc",
        "Bagel Lox",
        "Waffle",
        "Pancakes",
    ],
    "drink": [
        "Es Teh Manis",
        "Kopi Susu",
        "Cappuccino",
        "Cafe Latte",
        "Americano",
        "Espresso",
        "Matcha Latte",
        "Lemon Tea",
        "Orange Juice",
        "Mineral Water",
        "Iced Chocolate",
        "Thai Tea",
        "Avocado Juice",
        "Soda Gembira",
        "Hot Tea",
        "Mango Smoothie",
        "Cola",
        "Ginger Ale",
    ],
    "grocery": [
        "Milk 1L",
        "Eggs (10)",
        "Bread Loaf",
        "Rice 5kg",
        "Sugar 1kg",
        "Cooking Oil 2L",
        "Instant Noodle",
        "Tissue Box",
        "Shampoo 340ml",
        "Toothpaste",
        "Detergent 1kg",
        "Coffee 200g",
        "Tea Bags (25)",
        "Butter 200g",
        "Cheese Slices",
        "Apples 1kg",
        "Bananas",
        "Yogurt 4x",
        "Chips 150g",
        "Soap Bar",
    ],
}
CURRENCIES = [
    {
        "symbol": "Rp",
        "decimals": 0,
        "thousands": ".",
        "point": ",",
        "step": 1000,
        "range": (5, 120),
        "cash": 10000,
    },
    {
        "symbol": "",
        "decimals": 0,
        "thousands": ",",
        "point": ".",
        "step": 1000,
        "range": (5, 120),
        "cash": 50000,
    },
    {
        "symbol": "$",
        "decimals": 2,
        "thousands": ",",
        "point": ".",
        "step": 0.25,
        "range": (8, 120),
        "cash": 10,
    },
    {
        "symbol": "RM",
        "decimals": 2,
        "thousands": ",",
        "point": ".",
        "step": 0.5,
        "range": (6, 120),
        "cash": 10,
    },
    {
        "symbol": "€",
        "decimals": 2,
        "thousands": ".",
        "point": ",",
        "step": 0.1,
        "range": (20, 300),
        "cash": 5,
    },
]
MONO = ["Liberation Mono", "DejaVu Sans Mono", "Noto Mono", "Courier New", "monospace"]
SANS = ["Liberation Sans", "DejaVu Sans", "Noto Sans", "Arial", "sans-serif"]
SERIF = ["Liberation Serif", "DejaVu Serif", "Noto Serif", "Times New Roman", "serif"]

FORM_TITLES = [
    "FAX TRANSMITTAL",
    "INTEROFFICE MEMORANDUM",
    "RESEARCH PROPOSAL",
    "PURCHASE REQUISITION",
    "SAMPLE REQUEST FORM",
    "EXPENSE REPORT",
    "PROJECT AUTHORIZATION",
    "BRAND REVIEW",
    "TEST MARKET SUMMARY",
    "VENDOR INFORMATION",
    "TRAVEL REQUEST",
    "CHANGE NOTICE",
    "ADVERTISING APPROVAL",
    "LABORATORY REQUEST",
    "SHIPPING ORDER",
]
FORM_FIELDS = {
    "TO": "person",
    "FROM": "person",
    "DATE": "date",
    "SUBJECT": "subject",
    "FAX NO.": "phone",
    "PHONE": "phone",
    "NO. OF PAGES": "small",
    "CC": "person",
    "COMPANY": "company",
    "ADDRESS": "address",
    "BRAND": "brand",
    "QUANTITY": "qty",
    "ACCOUNT NO.": "account",
    "APPROVED BY": "person",
    "DEPARTMENT": "dept",
    "PROJECT NO.": "account",
    "BUDGET": "money",
    "START DATE": "date",
    "COMPLETION DATE": "date",
    "REQUESTED BY": "person",
    "LOCATION": "city",
    "VENDOR": "company",
    "P.O. NUMBER": "account",
    "DELIVERY DATE": "date",
    "COST CENTER": "account",
    "CONTACT": "person",
    "TITLE": "jobtitle",
    "REFERENCE": "account",
    "PRIORITY": "priority",
    "FILE CODE": "account",
}
PEOPLE = [
    "J. R. Miller",
    "A. Thompson",
    "K. Nakamura",
    "L. Fischer",
    "M. Okafor",
    "R. Delgado",
    "S. Patel",
    "T. O'Brien",
    "D. Kowalski",
    "E. Laurent",
    "H. Becker",
    "P. Anderson",
    "C. Rossi",
    "B. Hughes",
    "G. Svensson",
    "N. Haddad",
    "F. Moreau",
    "W. Chen",
    "V. Ivanova",
    "Y. Tanaka",
]
COMPANIES = [
    "Atlas Research Corp.",
    "Benton Packaging Inc.",
    "Crescent Labs",
    "Delta Marketing Group",
    "Evergreen Supply Co.",
    "Fairview Printing",
    "Granite Industries",
    "Horizon Media",
    "Ironwood Analytics",
    "Juniper Foods",
    "Keystone Graphics",
    "Lakeside Distribution",
]
BRANDS = [
    "Winston Lights",
    "Ridgeway",
    "Northstar",
    "Silver Leaf",
    "Carlton Blue",
    "Harbor Mist",
    "Golden Field",
    "Summit 100s",
    "Meridian",
    "Copperline",
]
DEPTS = [
    "Marketing",
    "R&D",
    "Legal",
    "Finance",
    "Quality Assurance",
    "Sales",
    "Operations",
    "Purchasing",
]
SUBJECTS = [
    "Q3 sample shipment",
    "Revised test protocol",
    "Budget approval request",
    "Package redesign",
    "Consumer panel results",
    "Filter supplier change",
    "Promotion schedule",
    "Lab analysis report",
    "Trade show booth",
    "Pricing update",
    "Field audit findings",
    "Media plan review",
]
TITLES = [
    "Brand Manager",
    "Research Scientist",
    "Director",
    "Analyst",
    "Coordinator",
    "Supervisor",
]
PRIORITIES = ["High", "Normal", "Low", "Urgent"]


def money(value: float, cur: dict[str, Any]) -> str:
    text = f"{value:,.{cur['decimals']}f}"
    text = (
        text.replace(",", "\x00")
        .replace(".", cur["point"])
        .replace("\x00", cur["thousands"])
    )
    return f"{cur['symbol']} {text}".strip() if cur["symbol"] else text


def fake_value(kind: str, r: random.Random) -> str:
    if kind == "person":
        return r.choice(PEOPLE)
    if kind == "date":
        d = dt.date(r.randint(1985, 1999), r.randint(1, 12), r.randint(1, 28))
        return r.choice(
            [d.strftime("%m/%d/%y"), d.strftime("%B %d, %Y"), d.strftime("%d %b %Y")]
        )
    if kind == "subject":
        return r.choice(SUBJECTS)
    if kind == "phone":
        return f"({r.randint(200, 989)}) {r.randint(200, 989)}-{r.randint(1000, 9999)}"
    if kind == "small":
        return str(r.randint(1, 12))
    if kind == "company":
        return r.choice(COMPANIES)
    if kind == "address":
        return f"{r.randint(10, 9999)} {r.choice(STREETS)}, {r.choice(CITIES)}"
    if kind == "brand":
        return r.choice(BRANDS)
    if kind == "qty":
        return f"{r.choice([50, 100, 200, 250, 500, 1000, 1200, 2500])} {r.choice(['cartons', 'units', 'packs', 'cases'])}"
    if kind == "account":
        return f"{r.choice('ABCDEFGHJK')}{r.randint(100, 999)}-{r.randint(1000, 99999)}"
    if kind == "money":
        return f"${r.randint(5, 950) * 100:,}"
    if kind == "dept":
        return r.choice(DEPTS)
    if kind == "city":
        return r.choice(CITIES)
    if kind == "jobtitle":
        return r.choice(TITLES)
    if kind == "priority":
        return r.choice(PRIORITIES)
    raise ValueError(kind)


def receipt(r: random.Random) -> dict[str, Any]:
    cur = r.choice(CURRENCIES)
    kind = r.choice(["food", "food", "drink", "grocery"])
    pool = MENU[kind] + (MENU["drink"] if kind == "food" else [])
    names = r.sample(pool, r.randint(3, 9))
    items = []
    for name in names:
        unit = round(r.randint(*cur["range"]) * cur["step"], 2)
        qty = r.choices([1, 2, 3, 4, 5], weights=[6, 3, 1.5, 0.7, 0.4])[0]
        items.append(
            {"name": name, "qty": qty, "unit": unit, "total": round(unit * qty, 2)}
        )
    subtotal = round(sum(i["total"] for i in items), 2)
    rate = r.choice([0.1, 0.11, 0.08, 0.06, 0.05])
    service = round(subtotal * 0.05, cur["decimals"]) if r.random() < 0.35 else 0.0
    discount = (
        round(subtotal * r.choice([0.05, 0.1, 0.15]), cur["decimals"])
        if r.random() < 0.2
        else 0.0
    )
    tax = round((subtotal - discount + service) * rate, cur["decimals"])
    total = round(subtotal - discount + service + tax, cur["decimals"])
    cash = change = None
    if r.random() < 0.65:
        step = cur["cash"]
        cash = round((total // step + r.randint(1, 4)) * step, 2)
        change = round(cash - total, cur["decimals"])
    store = f"{r.choice(STORE_NAMES)} {r.choice(STORE_WORDS)}"
    when = dt.datetime(
        2026, r.randint(1, 9), r.randint(1, 28), r.randint(7, 22), r.randint(0, 59)
    )
    return {
        "cur": cur,
        "items": items,
        "subtotal": subtotal,
        "service": service,
        "discount": discount,
        "tax": tax,
        "rate": rate,
        "total": total,
        "cash": cash,
        "change": change,
        "store": store,
        "address": f"{r.choice(STREETS)} No. {r.randint(1, 250)}, {r.choice(CITIES)}",
        "phone": f"Tel. {r.randint(21, 89)}-{r.randint(1000000, 9999999)}",
        "when": when,
        "number": f"{r.choice(['INV', 'TRX', 'No.', 'Bill'])} {r.randint(10000, 999999)}",
        "cashier": r.choice(
            [
                "Ani",
                "Budi",
                "Citra",
                "Dewi",
                "Eko",
                "Fajar",
                "Maya",
                "Rina",
                "Tom",
                "Lisa",
            ]
        ),
        "card": r.random() < 0.3 and cash is None,
    }


def receipt_html(rc: dict[str, Any], r: random.Random) -> tuple[str, int]:
    cur = rc["cur"]
    family = r.choice(MONO + SANS[:2])
    size = r.randint(13, 16)
    width = r.randint(300, 380)
    two_line = r.random() < 0.45
    esc = htmlmod.escape
    rows = []
    for it in rc["items"]:
        if two_line:
            rows.append(
                f"<div class=l>{esc(it['name'])}</div>"
                f"<div class=r><span>{it['qty']} x {esc(money(it['unit'], cur))}</span><span>{esc(money(it['total'], cur))}</span></div>"
            )
        else:
            rows.append(
                f"<div class=r><span>{it['qty']} {esc(it['name'])}</span><span>{esc(money(it['total'], cur))}</span></div>"
            )
    totals = [("Subtotal", rc["subtotal"])]
    if rc["discount"]:
        totals.append(("Discount", -rc["discount"]))
    if rc["service"]:
        totals.append(("Service 5%", rc["service"]))
    totals.append((f"Tax {int(round(rc['rate'] * 100))}%", rc["tax"]))
    total_rows = "".join(
        f"<div class=r><span>{esc(k)}</span><span>{esc(money(v, cur))}</span></div>"
        for k, v in totals
    )
    pay = ""
    if rc["cash"] is not None:
        pay = (
            f"<div class=r><span>Cash</span><span>{esc(money(rc['cash'], cur))}</span></div>"
            f"<div class=r><span>Change</span><span>{esc(money(rc['change'], cur))}</span></div>"
        )
    elif rc["card"]:
        pay = f"<div class=r><span>VISA **** {r.randint(1000, 9999)}</span><span>{esc(money(rc['total'], cur))}</span></div>"
    when = rc["when"].strftime(
        r.choice(
            [
                "%d/%m/%Y %H:%M",
                "%Y-%m-%d %H:%M",
                "%d-%m-%y %H:%M:%S",
                "%b %d %Y %I:%M %p",
            ]
        )
    )
    sep = r.choice(["-", "=", "."]) * 60
    page = f"""<html><head><style>
    body {{ margin:0; background:#fff; }}
    .p {{ width:{width}px; padding:18px 16px 28px; font-family:'{family}'; font-size:{size}px; color:#1d1d1d;
          background:#fbfaf6; letter-spacing:{r.choice([0, 0, 0.3, 0.6])}px; line-height:1.35; }}
    .c {{ text-align:center; }} .b {{ font-weight:700; }} .big {{ font-size:{size + r.randint(2, 6)}px; }}
    .r {{ display:flex; justify-content:space-between; gap:8px; }} .l {{ text-align:left; }}
    .s {{ overflow:hidden; white-space:nowrap; color:#555; }}
    </style></head><body><div class=p>
    <div class="c b big">{esc(rc['store'].upper() if r.random() < 0.5 else rc['store'])}</div>
    <div class=c>{esc(rc['address'])}</div><div class=c>{esc(rc['phone'])}</div>
    <div class=s>{sep}</div>
    <div class=r><span>{esc(rc['number'])}</span><span>{esc(when)}</span></div>
    <div class=l>Cashier: {esc(rc['cashier'])}</div>
    <div class=s>{sep}</div>{''.join(rows)}<div class=s>{sep}</div>{total_rows}
    <div class="r b big"><span>TOTAL</span><span>{esc(money(rc['total'], cur))}</span></div>{pay}
    <div class=s>{sep}</div><div class=c>{esc(r.choice(['Thank you!', 'Terima kasih', 'Please come again', 'Thank you for your visit']))}</div>
    </div></body></html>"""
    return page, width + 32


def receipt_question(
    rc: dict[str, Any], r: random.Random
) -> tuple[str, str, str, list[str]]:
    cur = rc["cur"]
    amounts = {"subtotal": rc["subtotal"], "tax": rc["tax"], "total": rc["total"]}
    if rc["cash"] is not None:
        amounts.update({"cash": rc["cash"], "change": rc["change"]})
    if rc["service"]:
        amounts["service charge"] = rc["service"]
    for it in rc["items"]:
        amounts.setdefault(f"item:{it['name']}", it["total"])
    kinds = [
        "total",
        "total",
        "subtotal",
        "tax",
        "item_price",
        "item_price",
        "item_qty",
        "n_items",
        "top_item",
    ]
    if rc["cash"] is not None:
        kinds += ["change", "change"]
    kind = r.choice(kinds)

    def amount_options(key: str) -> tuple[str, list[str]]:
        correct = money(amounts[key], cur)
        pool = sorted(
            {money(v, cur) for k, v in amounts.items() if k != key} - {correct}
        )
        if len(pool) < 3:
            raise ValueError("not enough distinct amounts")
        return correct, r.sample(pool, 3)

    if kind in ("total", "subtotal", "tax", "change"):
        question = {
            "total": "What is the total amount to pay on this receipt?",
            "subtotal": "What is the subtotal on this receipt (before tax and other charges)?",
            "tax": "How much tax is charged on this receipt?",
            "change": "How much change was given to the customer?",
        }[kind]
        correct, distractors = amount_options(kind)
    elif kind == "item_price":
        it = r.choice(rc["items"])
        question = f"What is the line total for \"{it['name']}\" on this receipt?"
        correct, distractors = amount_options(f"item:{it['name']}")
    elif kind == "item_qty":
        it = r.choice(rc["items"])
        question = f"How many \"{it['name']}\" were bought according to this receipt?"
        correct = str(it["qty"])
        distractors = [
            str(q)
            for q in sorted(
                {1, 2, 3, 4, 5, 6} - {it["qty"]},
                key=lambda q: (abs(q - it["qty"]), r.random()),
            )[:3]
        ]
    elif kind == "n_items":
        n = len(rc["items"])
        question = "How many different products are listed on this receipt?"
        correct = str(n)
        distractors = [
            str(v)
            for v in sorted(
                {n - 2, n - 1, n + 1, n + 2} - {n},
                key=lambda v: (abs(v - n), r.random()),
            )
            if v > 0
        ][:3]
    else:
        top = max(rc["items"], key=lambda i: i["total"])
        others = [i for i in rc["items"] if i is not top]
        if (
            len(others) < 3
            or sorted(i["total"] for i in rc["items"])[-2] == top["total"]
        ):
            raise ValueError("ambiguous top item")
        question = "Which product has the highest line total on this receipt?"
        correct = top["name"]
        distractors = [i["name"] for i in r.sample(others, 3)]
    return f"receipt:{kind}", question, correct, distractors


def form(r: random.Random) -> dict[str, Any]:
    labels = r.sample(list(FORM_FIELDS), r.randint(7, 12))
    blanks = set(r.sample(labels, r.randint(1, max(1, len(labels) // 4))))
    values = {
        lab: ("" if lab in blanks else fake_value(FORM_FIELDS[lab], r))
        for lab in labels
    }
    absent = [lab for lab in FORM_FIELDS if lab not in labels]
    checks = r.sample(
        [
            "Urgent",
            "For review",
            "Please reply",
            "For your information",
            "Confidential",
            "Approved",
            "Rejected",
            "Pending",
        ],
        4,
    )
    return {
        "title": r.choice(FORM_TITLES),
        "labels": labels,
        "values": values,
        "blanks": blanks,
        "absent": absent,
        "company": r.choice(COMPANIES),
        "checks": checks,
        "checked": set(r.sample(checks, r.randint(1, 2))),
        "body": " ".join(
            r.sample(
                [
                    "Please find attached the requested materials.",
                    "Samples should be shipped no later than the date above.",
                    "Results will be summarized in the next quarterly report.",
                    "Contact the undersigned with any questions.",
                    "All figures are preliminary and subject to revision.",
                    "This authorization supersedes the previous request.",
                ],
                3,
            )
        ),
    }


def form_html(fm: dict[str, Any], r: random.Random) -> tuple[str, int]:
    family = r.choice(MONO[:3] + SERIF[:2] + SANS[:1])
    size = r.randint(14, 17)
    width = r.randint(760, 900)
    esc = htmlmod.escape
    two_col = r.random() < 0.5
    cells = []
    for lab in fm["labels"]:
        value = fm["values"][lab]
        shown = esc(value) if value else "&nbsp;"
        cells.append(
            f"<div class=f><span class=k>{esc(lab)}:</span><span class=v>{shown}</span></div>"
        )
    grid = (
        "grid-template-columns: 1fr 1fr;" if two_col else "grid-template-columns: 1fr;"
    )
    boxes = "".join(
        f"<span class=cb>[{'X' if c in fm['checked'] else '&nbsp;'}] {esc(c)}</span>"
        for c in fm["checks"]
    )
    page = f"""<html><head><style>
    body {{ margin:0; background:#fff; }}
    .p {{ width:{width}px; padding:48px 56px 64px; font-family:'{family}'; font-size:{size}px; color:#111; }}
    .h {{ text-align:center; font-weight:700; font-size:{size + 8}px; letter-spacing:2px; margin:6px 0 4px; }}
    .co {{ text-align:center; font-size:{size + 2}px; margin-bottom:22px; }}
    .g {{ display:grid; {grid} gap:12px 36px; margin-bottom:22px; }}
    .f {{ display:flex; gap:10px; align-items:flex-end; }} .k {{ font-weight:700; white-space:nowrap; }}
    .v {{ flex:1; border-bottom:1px solid #333; min-height:{size + 4}px; padding-left:4px; }}
    .cb {{ margin-right:24px; }} .body {{ margin-top:22px; line-height:1.6; }}
    .sig {{ margin-top:48px; width:260px; border-top:1px solid #333; padding-top:4px; }}
    </style></head><body><div class=p>
    <div class=co>{esc(fm['company'])}</div><div class=h>{esc(fm['title'])}</div>
    <div class=g>{''.join(cells)}</div><div>{boxes}</div><div class=body>{esc(fm['body'])}</div>
    <div class=sig>Signature</div></div></body></html>"""
    return page, width + 112


def form_question(
    fm: dict[str, Any], r: random.Random
) -> tuple[str, str, str, list[str]]:
    labels, values, blanks = fm["labels"], fm["values"], fm["blanks"]
    filled = [lab for lab in labels if values[lab]]
    kind = r.choice(["present", "present", "absent", "value", "value", "blank"])
    if kind == "present":
        correct = r.choice(labels)
        return (
            "form:present",
            "Which of the following fields appears on this form?",
            correct,
            r.sample(fm["absent"], 3),
        )
    if kind == "absent":
        correct = r.choice(fm["absent"])
        return (
            "form:absent",
            "Which of the following fields does NOT appear on this form?",
            correct,
            r.sample(labels, 3),
        )
    if kind == "value":
        lab = r.choice(filled)
        pool = sorted({values[x] for x in filled if x != lab} - {values[lab]})
        if len(pool) < 3:
            raise ValueError("not enough filled fields")
        return (
            "form:value",
            f'What is entered in the "{lab}" field of this form?',
            values[lab],
            r.sample(pool, 3),
        )
    if len(filled) < 3:
        raise ValueError("not enough filled fields")
    correct = r.choice(sorted(blanks))
    return (
        "form:blank",
        "Which of the following fields is left blank on this form?",
        correct,
        r.sample(filled, 3),
    )


def worker_init() -> None:
    from d25.omni.proxy import render

    render.start()


def worker_close() -> None:
    from d25.omni.proxy import render

    render.stop()


def build_item(index: int, seed: int, context: Any = None) -> dict[str, Any]:
    from d25.omni.proxy import augment, render
    from d25.omni.proxy.rows import lettered, rng

    for attempt in range(50):
        r = rng(NAME, seed, index, attempt)
        try:
            if r.random() < 0.6:
                doc = receipt(r)
                subtask, question, correct, distractors = receipt_question(doc, r)
                page, width = receipt_html(doc, r)
                png, _, _ = render.html(page, width, scale=r.choice([2.0, 2.5]))
                image = augment.photo(
                    Image.open(io.BytesIO(png)), r, target_long=r.randint(1200, 1600)
                )
            else:
                doc = form(r)
                subtask, question, correct, distractors = form_question(doc, r)
                page, width = form_html(doc, r)
                png, _, _ = render.html(page, width, scale=r.choice([1.25, 1.5]))
                image = augment.scan(
                    Image.open(io.BytesIO(png)), r, target_long=r.randint(1100, 1500)
                )
            criteria, answer = lettered(correct, distractors, r)
        except ValueError:
            continue
        payload = augment.jpeg(image, r.randint(85, 93))
        return {
            "item_id": f"{index:05d}",
            "subtask": subtask,
            "payloads": [(payload, "jpg")],
            "instructions": question,
            "criteria": criteria,
            "answer": answer,
            "provenance": [
                {
                    "source": "generated",
                    "generator": f"d25.omni.proxy.kie v{VERSION}",
                    "seed": seed,
                    "index": index,
                    "licence": "generated (no third-party content)",
                }
            ],
            "extra": {"attempt": attempt},
        }
    raise RuntimeError(f"no valid KIE item for {index}")
