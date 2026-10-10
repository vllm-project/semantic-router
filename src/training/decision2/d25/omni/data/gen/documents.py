"""Infographics, business documents, receipts and forms rendered with PIL, with known field values.

Every question is answerable from the page and its distractors come from the same page (other numbers,
dates, names, labels), except field-presence questions, whose absent labels come from a shared pool.
Receipts are photographed (perspective, lighting, blur) or scanned; forms get a scanner look.
"""

from __future__ import annotations

import random

from PIL import Image, ImageDraw

from d25.omni.data import render
from d25.omni.data.rows import Item, fmt_number, rng_for

GROUPS = (
    "adults",
    "employees",
    "students",
    "households",
    "small businesses",
    "teachers",
    "nurses",
    "commuters",
    "customers",
    "parents",
    "retirees",
    "farmers",
    "developers",
    "drivers",
    "young people",
    "patients",
)
ACTIONS = (
    "work from home at least once a week",
    "use a budgeting app",
    "recycle plastic regularly",
    "own an electric vehicle",
    "shop online every month",
    "read the news daily",
    "exercise three times a week",
    "use public transport",
    "have a pension plan",
    "volunteer locally",
    "speak a second language",
    "use cloud storage",
    "grow their own vegetables",
    "track their sleep",
    "cook at home most days",
    "plan to travel abroad this year",
    "pay mainly by phone",
    "feel optimistic about next year",
)
TOPICS = (
    "Digital Habits",
    "Health at Work",
    "The Green Transition",
    "Money Matters",
    "Urban Mobility",
    "Education Today",
    "Food and Farming",
    "Tech Adoption",
    "Travel Trends",
    "Community Life",
)
COUNTRIES = (
    "Canada",
    "Brazil",
    "Japan",
    "Kenya",
    "Germany",
    "India",
    "Mexico",
    "Norway",
    "Chile",
    "Vietnam",
    "Spain",
    "Egypt",
    "Poland",
    "Peru",
    "Ghana",
    "Sweden",
    "Turkey",
    "Thailand",
    "Italy",
    "Nigeria",
)
FIRST = (
    "James",
    "Maria",
    "Wei",
    "Aisha",
    "Carlos",
    "Elena",
    "Tom",
    "Priya",
    "Lukas",
    "Sofia",
    "Omar",
    "Grace",
    "Daniel",
    "Hana",
    "Pedro",
    "Fatima",
    "Ivan",
    "Chloe",
    "Kwame",
    "Mei",
    "Robert",
    "Laura",
    "Ahmed",
    "Nina",
)
LAST = (
    "Smith",
    "Garcia",
    "Chen",
    "Okafor",
    "Rossi",
    "Novak",
    "Kim",
    "Patel",
    "Muller",
    "Silva",
    "Haddad",
    "Johnson",
    "Tanaka",
    "Lopez",
    "Ivanova",
    "Brown",
    "Mensah",
    "Nguyen",
    "Wilson",
    "Moreau",
    "Kowalski",
)
ORGS = (
    "Northwind Logistics",
    "Bluefield Research",
    "Cedar Health Partners",
    "Orion Materials",
    "Harbor Foods Co.",
    "Summit Analytics",
    "Riverbend Energy",
    "Atlas Engineering",
    "Maple Street Bank",
    "Greenway Transit",
    "Silverline Pharma",
    "Pioneer Insurance",
    "Lakeside University",
    "Vertex Robotics",
    "Crescent Media",
)
DEPTS = (
    "Finance",
    "Human Resources",
    "Operations",
    "Research",
    "Legal",
    "Marketing",
    "Procurement",
    "IT Services",
    "Quality Assurance",
    "Sales",
)
MONTHS = (
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
STORES = (
    "Sunrise Mart",
    "Kopi Kenangan Corner",
    "Bella Pizza",
    "Green Leaf Cafe",
    "City Hardware",
    "Fresh Basket",
    "Noodle House 88",
    "Corner Bakery",
    "Blue Ocean Sushi",
    "Daily Grocer",
    "Taco Fiesta",
    "Book Nook",
    "Pharma Plus",
    "Burger Barn",
    "Tea Garden",
    "Metro Electronics",
)
FOODS = (
    "Iced Lemon Tea",
    "Fried Rice",
    "Chicken Wings",
    "Caesar Salad",
    "Cappuccino",
    "Mineral Water",
    "Beef Burger",
    "French Fries",
    "Green Tea Latte",
    "Chocolate Cake",
    "Spring Rolls",
    "Mushroom Soup",
    "Grilled Fish",
    "Orange Juice",
    "Pad Thai",
    "Club Sandwich",
    "Espresso",
    "Croissant",
    "Ramen Bowl",
    "Garlic Bread",
    "Milkshake",
    "Fruit Bowl",
    "Pancakes",
    "Hot Chocolate",
    "Dumplings",
    "Nachos",
)
GOODS = (
    "AA Batteries",
    "USB Cable",
    "Notebook A5",
    "Hand Soap",
    "Paper Towels",
    "Light Bulb",
    "Shampoo",
    "Toothpaste",
    "Bananas 1kg",
    "Whole Milk 1L",
    "Eggs x12",
    "Rice 5kg",
    "Olive Oil",
    "Coffee Beans",
    "Dish Sponge",
    "Trash Bags",
    "Phone Charger",
    "Sticky Notes",
    "Ballpoint Pens",
    "Yogurt",
)
FORM_TITLES = (
    "Request for Information",
    "Fax Cover Sheet",
    "Purchase Order",
    "Expense Report",
    "Contract Approval Form",
    "Laboratory Test Report",
    "Change Request",
    "Vendor Registration",
    "Travel Authorization",
    "Incident Report",
    "Product Evaluation",
    "Shipping Instructions",
)
FIELDS = (
    "Name",
    "Date",
    "Company",
    "Address",
    "Phone",
    "Fax",
    "Email",
    "Department",
    "Title",
    "Signature",
    "Project Number",
    "Account Code",
    "Purchase Order No.",
    "Invoice Number",
    "Amount",
    "Due Date",
    "Approved By",
    "Reviewed By",
    "Location",
    "Contact Person",
    "Reference No.",
    "Subject",
    "Total Pages",
    "Brand",
    "Sample Size",
    "Test Method",
    "Vendor ID",
    "Tax ID",
    "Ship To",
    "Bill To",
    "Cost Center",
    "Start Date",
    "End Date",
    "Destination",
    "Purpose",
    "Supervisor",
    "Employee ID",
    "Priority",
)


def _person(rng):
    return f"{rng.choice(FIRST)} {rng.choice(LAST)}"


def _date(rng, style=None):
    y, m, d = rng.randint(1995, 2024), rng.randint(1, 12), rng.randint(1, 28)
    style = style if style is not None else rng.randrange(4)
    return [
        f"{MONTHS[m - 1]} {d}, {y}",
        f"{d:02d}/{m:02d}/{y}",
        f"{y}-{m:02d}-{d:02d}",
        f"{d} {MONTHS[m - 1][:3]} {y}",
    ][style]


def _pick(gold: str, pool, rng, n=4):
    others = [p for p in dict.fromkeys(pool) if p != gold]
    if len(others) < n - 1:
        return None
    return [gold] + rng.sample(others, n - 1)


# ------------------------------------------------------------------ infographic


def infographic(index: int, ctx=None) -> Item | None:
    rng = rng_for("gen-document", "infographic", index)
    palette = rng.choice(render.PALETTES)
    width = rng.randint(760, 1100)
    bg = rng.choice(
        [
            (255, 255, 255),
            (247, 244, 236),
            (236, 242, 248),
            (24, 32, 48),
            (250, 240, 230),
        ]
    )
    dark = sum(bg) < 300
    ink = (240, 240, 240) if dark else (30, 30, 30)
    canvas = Image.new("RGB", (width, 3200), bg)
    draw = ImageDraw.Draw(canvas)
    title_font = render.font("display", rng.randint(46, 70), rng)
    head_font = render.font("sans_bold", rng.randint(24, 32), rng)
    body_font = render.font("sans", rng.randint(17, 22), rng)
    big_font = render.font("display", rng.randint(56, 84), rng)
    topic = rng.choice(TOPICS)
    region = rng.choice(COUNTRIES)
    year = rng.randint(2012, 2025)
    y = 40
    for line in render.wrap(
        draw, f"{topic.upper()} IN {region.upper()}", title_font, width - 80
    ):
        draw.text((40, y), line, font=title_font, fill=palette[0])
        y += render.line_height(title_font)
    draw.text(
        (40, y + 6),
        f"Survey of {rng.randint(5, 60) * 100:,} people, {year}",
        font=body_font,
        fill=ink,
    )
    y += 70
    facts: list[tuple[str, str]] = []
    numbers: list[str] = []
    lists: list[tuple[str, list[str]]] = []
    bars: list[tuple[str, str]] = []
    for s in range(rng.randint(3, 5)):
        kind = rng.choice(["stat", "stat", "bars", "rank", "ratio"])
        color = palette[(s + 1) % len(palette)]
        draw.rectangle((30, y, width - 30, y + 4), fill=color)
        y += 22
        if kind == "stat":
            for _ in range(rng.randint(1, 2)):
                value = f"{rng.randint(3, 97)}%"
                if value in numbers:
                    continue
                caption = f"of {rng.choice(GROUPS)} {rng.choice(ACTIONS)}"
                draw.text((50, y), value, font=big_font, fill=color)
                vx = 60 + render.text_width(draw, value, big_font)
                lines = render.wrap(draw, caption, body_font, width - vx - 60)
                render.draw_lines(draw, (vx + 10, y + 18), lines, body_font, fill=ink)
                facts.append((caption, value))
                numbers.append(value)
                y += max(render.line_height(big_font) + 10, 30 * len(lines) + 30)
        elif kind == "bars":
            label = f"{rng.choice(GROUPS).capitalize()} by region (%)"
            draw.text((50, y), label, font=head_font, fill=ink)
            y += render.line_height(head_font) + 10
            names = rng.sample(COUNTRIES, rng.randint(3, 5))
            vals = [rng.randint(5, 95) for _ in names]
            label_w = max(render.text_width(draw, n, body_font) for n in names) + 20
            maxw = width - 160 - label_w
            for n, v in zip(names, vals):
                draw.text((50, y), n, font=body_font, fill=ink)
                draw.rectangle(
                    (50 + label_w, y + 4, 50 + label_w + int(maxw * v / 100), y + 26),
                    fill=color,
                )
                draw.text(
                    (60 + label_w + int(maxw * v / 100), y + 2),
                    f"{v}%",
                    font=body_font,
                    fill=ink,
                )
                bars.append((f"{n} ({label.split(' by ')[0].lower()})", f"{v}%"))
                numbers.append(f"{v}%")
                y += 36
        elif kind == "rank":
            label = f"Top {rng.randint(4, 5)} {rng.choice(['destinations', 'priorities', 'concerns', 'sectors'])}"
            n = int(label.split()[1])
            draw.text((50, y), label, font=head_font, fill=ink)
            y += render.line_height(head_font) + 8
            items = rng.sample(
                [a.split(" ")[-1].capitalize() for a in ACTIONS] + list(COUNTRIES), n
            )
            for k, it in enumerate(items):
                draw.ellipse((50, y, 84, y + 34), fill=color)
                draw.text((60, y + 4), str(k + 1), font=body_font, fill=(255, 255, 255))
                draw.text((100, y + 5), it, font=body_font, fill=ink)
                y += 42
            lists.append((label, items))
        else:
            k = rng.randint(1, 9)
            caption = f"{k} in 10 {rng.choice(GROUPS)} {rng.choice(ACTIONS)}"
            for i in range(10):
                cx = 60 + i * 52
                fill = color if i < k else ((90, 90, 90) if dark else (200, 200, 200))
                draw.ellipse((cx, y, cx + 22, y + 22), fill=fill)
                draw.rectangle((cx + 3, y + 24, cx + 19, y + 60), fill=fill)
            y += 70
            draw.text((50, y), caption, font=body_font, fill=ink)
            facts.append((caption.split(" ", 3)[3], f"{k} in 10"))
            numbers.append(f"{k} in 10")
            y += 40
        y += 26
        if y > 2900:
            break
    draw.text(
        (40, y + 10), f"Source: {rng.choice(ORGS)} ({year})", font=body_font, fill=ink
    )
    image = canvas.crop((0, 0, width, min(3200, y + 60)))
    choices = []
    if facts:
        choices.append("fact")
    if bars:
        choices.append("bar")
    if lists:
        choices.append("rank")
    if not choices:
        return None
    qtype = rng.choice(choices)
    pool = numbers + [f"{rng.randint(3, 97)}%" for _ in range(4)]
    if qtype == "fact":
        caption, gold = rng.choice(facts)
        question = (
            f"According to the infographic, what share of respondents {caption.split(' ', 2)[-1]}?"
            if " in 10" not in gold
            else f"How many in 10 {caption}?"
        )
        if " in 10" in gold:
            pool = [f"{i} in 10" for i in range(1, 10)]
            question = (
                f"According to the infographic, how many in 10 {caption.split(' ', 1)[0]} "
                f"{caption.split(' ', 1)[1]}?"
            )
            options = _pick(gold, pool, rng)
        else:
            options = _pick(gold, pool, rng)
    elif qtype == "bar":
        label, gold = rng.choice(bars)
        question = f"What value is shown for {label}?"
        options = _pick(gold, pool, rng)
    else:
        label, items = rng.choice(lists)
        k = rng.randrange(len(items))
        gold = items[k]
        ordinal = ["first", "second", "third", "fourth", "fifth"][k]
        question = f"Which item is ranked {ordinal} in the list '{label}'?"
        options = _pick(gold, items, rng)
    if options is None:
        return None
    text = " ".join(
        [topic, region]
        + [c for c, _ in facts]
        + [b for b, _ in bars]
        + sum((i for _, i in lists), [])
    )
    return Item(
        source="gen-document",
        family="infographic",
        skill="infographic",
        images=[image],
        image_kinds=["png"],
        question=question,
        options=options,
        meta={"qtype": qtype, "benchmark_target": "InfographicVQA"},
        image_text=[text],
    )


# ------------------------------------------------------------------ business document


def document(index: int, ctx=None) -> Item | None:
    rng = rng_for("gen-document", "document", index)
    width, height = rng.choice([(850, 1100), (900, 1165), (1000, 1294)])
    page = render.paper((width, height), rng)
    draw = ImageDraw.Draw(page)
    serif = rng.random() < 0.5
    body = render.font("serif" if serif else "sans", rng.randint(17, 21), rng)
    bold = render.font("serif_bold" if serif else "sans_bold", rng.randint(20, 26), rng)
    org = rng.choice(ORGS)
    kind = rng.choice(["MEMORANDUM", "INTEROFFICE MEMO", "LETTER", "REPORT"])
    sender, recipient, cc = _person(rng), _person(rng), _person(rng)
    date = _date(rng)
    other_dates = [_date(rng) for _ in range(3)]
    dept = rng.choice(DEPTS)
    subject = rng.choice(
        [
            "Quarterly budget review",
            "Supplier contract renewal",
            "Safety inspection results",
            "Updated travel policy",
            "Project milestone report",
            "Equipment purchase request",
            "Annual audit schedule",
            "Customer survey findings",
        ]
    )
    x, y = 70, 60
    draw.text((x, y), org, font=bold, fill=(20, 20, 60))
    y += 34
    draw.text(
        (x, y),
        f"{rng.randint(10, 999)} {rng.choice(LAST)} Avenue, {rng.choice(COUNTRIES)}",
        font=body,
        fill=(60, 60, 60),
    )
    y += 50
    draw.text((x, y), kind, font=bold, fill=(0, 0, 0))
    y += 44
    fields = [
        ("DATE", date),
        ("TO", f"{recipient}, {rng.choice(DEPTS)}"),
        ("FROM", f"{sender}, {dept}"),
        ("CC", cc),
        ("SUBJECT", subject),
    ]
    for label, value in fields:
        draw.text((x, y), f"{label}:", font=bold, fill=(0, 0, 0))
        draw.text((x + 150, y + 2), value, font=body, fill=(0, 0, 0))
        y += 34
    draw.line((x, y + 6, width - 70, y + 6), fill=(0, 0, 0), width=2)
    y += 28
    rows = [
        (rng.choice(GOODS + FOODS), rng.randint(1, 40), round(rng.uniform(5, 900), 2))
        for _ in range(rng.randint(3, 6))
    ]
    names = list(dict.fromkeys(r[0] for r in rows))
    rows = [r for r in rows if r[0] in names][: len(names)]
    sentences = [
        f"As discussed during the meeting on {other_dates[0]}, {dept.lower()} will coordinate the next steps.",
        f"Please review the figures below before {other_dates[1]} and send comments to {sender.split()[0]}.",
        f"The attached schedule replaces the version circulated on {other_dates[2]}.",
        "All amounts are shown in US dollars and exclude applicable taxes.",
    ]
    rng.shuffle(sentences)
    y = (
        render.draw_lines(
            draw,
            (x, y),
            render.wrap(draw, " ".join(sentences), body, width - 140),
            body,
        )
        + 16
    )
    col = [x, x + 380, x + 520]
    for c, head in zip(col, ("Item", "Qty", "Amount")):
        draw.text((c, y), head, font=bold, fill=(0, 0, 0))
    y += 34
    for name, qty, amount in rows:
        draw.text((col[0], y), name, font=body, fill=(0, 0, 0))
        draw.text((col[1], y), str(qty), font=body, fill=(0, 0, 0))
        draw.text((col[2], y), f"${amount:,.2f}", font=body, fill=(0, 0, 0))
        y += 30
    total = round(sum(r[2] for r in rows), 2)
    draw.line((col[0], y + 4, col[2] + 160, y + 4), fill=(0, 0, 0), width=1)
    draw.text((col[0], y + 10), "Total", font=bold, fill=(0, 0, 0))
    draw.text((col[2], y + 10), f"${total:,.2f}", font=bold, fill=(0, 0, 0))
    y += 70
    draw.text((x, y), "Regards,", font=body, fill=(0, 0, 0))
    draw.text(
        (x, y + 40), sender, font=render.font("hand", 30, rng), fill=(20, 30, 120)
    )
    page = render.degrade_scan(page, rng) if rng.random() < 0.6 else page
    amounts = [f"${r[2]:,.2f}" for r in rows] + [f"${total:,.2f}"]
    qtype = rng.choice(
        ["date", "sender", "recipient", "amount", "total", "subject_dept"]
    )
    if qtype == "date":
        question, gold, pool = (
            "What is the date of this document?",
            date,
            other_dates + [_date(rng)],
        )
    elif qtype == "sender":
        question, gold, pool = (
            "Who sent this document?",
            sender,
            [recipient, cc, _person(rng)],
        )
    elif qtype == "recipient":
        question, gold, pool = (
            "To whom is this document addressed?",
            recipient,
            [sender, cc, _person(rng)],
        )
    elif qtype == "amount":
        name, qty, amount = rng.choice(rows)
        question, gold, pool = (
            f"What is the amount listed for {name}?",
            f"${amount:,.2f}",
            amounts,
        )
    elif qtype == "total":
        question, gold, pool = (
            "What is the total amount in the table?",
            f"${total:,.2f}",
            amounts,
        )
    else:
        question, gold, pool = "Which department is the sender from?", dept, list(DEPTS)
    options = _pick(gold, pool, rng)
    if options is None:
        return None
    text = " ".join(
        [org, kind, subject, sender, recipient, cc, date, *sentences, *names]
    )
    return Item(
        source="gen-document",
        family="document",
        skill="infographic",
        images=[page],
        image_kinds=["png"],
        question=question,
        options=options,
        meta={"qtype": qtype, "benchmark_target": "InfographicVQA"},
        image_text=[text],
    )


# ------------------------------------------------------------------ receipt


def _money(value: float, style: int) -> str:
    if style == 0:
        return f"{value:,.2f}"
    if style == 1:
        return f"{int(round(value)):,}"
    if style == 2:
        return f"{int(round(value)):,}".replace(",", ".")
    return f"{value:,.2f}".replace(",", " ").replace(".", ",")


def receipt(index: int, ctx=None) -> Item | None:
    rng = rng_for("gen-document", "receipt", index)
    style = rng.randrange(4)
    scale = {0: 1, 1: 1000, 2: 1000, 3: 1}[style]
    cur = {
        0: rng.choice(["$", "", "USD "]),
        1: rng.choice(["", "Rp ", "IDR "]),
        2: rng.choice(["", "Rp."]),
        3: rng.choice(["", "EUR "]),
    }[style]
    width = rng.randint(380, 520)
    face = render.font("receipt", rng.randint(19, 24), rng)
    big = render.font("receipt", rng.randint(26, 32), rng)
    lh = int(render.line_height(face) * 1.15)
    store = rng.choice(STORES)
    pool_items = FOODS if rng.random() < 0.6 else GOODS
    items = []
    for name in rng.sample(pool_items, rng.randint(2, 7)):
        qty = rng.choice([1, 1, 1, 2, 2, 3, 4])
        unit = (
            round(rng.uniform(1.0, 30.0), 2)
            if scale == 1
            else rng.randint(5, 95) * scale // 1
        )
        if scale > 1:
            unit = float(round(unit / 500) * 500 or 500)
        items.append((name, qty, unit, round(qty * unit, 2)))
    subtotal = round(sum(i[3] for i in items), 2)
    tax_rate = rng.choice([0, 5, 8, 10, 11])
    tax = round(subtotal * tax_rate / 100, 2 if scale == 1 else 0)
    service = round(subtotal * rng.choice([0, 0, 5]) / 100, 2 if scale == 1 else 0)
    total = round(subtotal + tax + service, 2)
    paid = (
        total
        if rng.random() < 0.5
        else (float(int(total / (10 * scale)) + 1) * 10 * scale)
    )
    change = round(paid - total, 2)
    lines: list[tuple[str, str, object]] = [
        ("c", store.upper(), big),
        ("c", f"{rng.randint(1, 300)} {rng.choice(LAST)} St.", face),
        ("c", f"Tel {rng.randint(100, 999)}-{rng.randint(1000, 9999)}", face),
        ("c", "-" * 40, face),
        (
            "l",
            f"{_date(rng, rng.choice([1, 2]))}  {rng.randint(7, 22):02d}:{rng.randint(0, 59):02d}",
            face,
        ),
        ("l", f"Cashier: {rng.choice(FIRST)}", face),
        ("c", "-" * 40, face),
    ]
    for name, qty, unit, line_total in items:
        lines.append(
            (
                "lr",
                (
                    f"{qty} x {name}" if qty > 1 or rng.random() < 0.5 else name,
                    f"{cur}{_money(line_total, style)}",
                ),
                face,
            )
        )
        if qty > 1 and rng.random() < 0.5:
            lines.append(("l", f"   @ {_money(unit, style)}", face))
    lines.append(("c", "-" * 40, face))
    lines.append(("lr", ("SUBTOTAL", f"{cur}{_money(subtotal, style)}"), face))
    if tax:
        lines.append(("lr", (f"TAX {tax_rate}%", f"{cur}{_money(tax, style)}"), face))
    if service:
        lines.append(("lr", ("SERVICE", f"{cur}{_money(service, style)}"), face))
    lines.append(("lr", ("TOTAL", f"{cur}{_money(total, style)}"), big))
    method = rng.choice(["CASH", "CARD", "DEBIT", "QRIS"])
    lines.append(("lr", (method, f"{cur}{_money(paid, style)}"), face))
    if change:
        lines.append(("lr", ("CHANGE", f"{cur}{_money(change, style)}"), face))
    lines.append(("c", "THANK YOU", face))
    height = sum(int(render.line_height(f) * 1.15) for _, _, f in lines) + 80
    page = render.paper(
        (width, height),
        rng,
        tint=rng.choice([(250, 250, 247), (245, 243, 235), (255, 255, 255)]),
    )
    draw = ImageDraw.Draw(page)
    y = 30
    for align, text, f in lines:
        step = int(render.line_height(f) * 1.15)
        if align == "c":
            draw.text(
                ((width - render.text_width(draw, text, f)) // 2, y),
                text,
                font=f,
                fill=(25, 25, 25),
            )
        elif align == "l":
            draw.text((24, y), text, font=f, fill=(25, 25, 25))
        else:
            left, right = text
            draw.text((24, y), left, font=f, fill=(25, 25, 25))
            draw.text(
                (width - 24 - render.text_width(draw, right, f), y),
                right,
                font=f,
                fill=(25, 25, 25),
            )
        y += step
    image = (
        render.photograph(page, rng)
        if rng.random() < 0.65
        else render.degrade_scan(page, rng)
    )
    money = [
        f"{cur}{_money(v, style)}"
        for v in [i[3] for i in items] + [subtotal, tax, service, total, paid, change]
        if v
    ]
    qtype = rng.choice(
        [
            "total",
            "total",
            "item_price",
            "subtotal",
            "count",
            "priciest",
            "store",
            "change",
        ]
    )
    if qtype == "total":
        question, gold, pool = (
            "What is the total amount on this receipt?",
            f"{cur}{_money(total, style)}",
            money,
        )
    elif qtype == "subtotal":
        question, gold, pool = (
            "What is the subtotal before tax and service?",
            f"{cur}{_money(subtotal, style)}",
            money,
        )
    elif qtype == "item_price":
        name, qty, unit, line_total = rng.choice(items)
        question, gold, pool = (
            f"What is the line amount for {name}?",
            f"{cur}{_money(line_total, style)}",
            money,
        )
    elif qtype == "count":
        n = sum(i[1] for i in items)
        question, gold, pool = (
            "How many units were purchased in total?",
            str(n),
            [str(n + d) for d in (-2, -1, 1, 2, 3) if n + d > 0],
        )
    elif qtype == "priciest":
        best = max(items, key=lambda i: i[3])
        if sum(1 for i in items if i[3] == best[3]) > 1 or len(items) < 4:
            return None
        question, gold, pool = (
            "Which item has the highest line amount?",
            best[0],
            [i[0] for i in items],
        )
    elif qtype == "store":
        question, gold, pool = "What is the name of the store?", store, list(STORES)
    else:
        if not change:
            return None
        question, gold, pool = (
            "How much change was given?",
            f"{cur}{_money(change, style)}",
            money,
        )
    options = _pick(gold, pool, rng)
    if options is None:
        return None
    text = " ".join(
        [store] + [i[0] for i in items] + ["SUBTOTAL TOTAL CHANGE THANK YOU"]
    )
    return Item(
        source="gen-document",
        family="receipt",
        skill="kie",
        images=[image],
        image_kinds=["jpeg"],
        question=question,
        options=options,
        meta={"qtype": qtype, "benchmark_target": "CORD"},
        image_text=[text],
    )


# ------------------------------------------------------------------ form


def form(index: int, ctx=None) -> Item | None:
    rng = rng_for("gen-document", "form", index)
    width, height = rng.choice([(850, 1100), (1000, 1294)])
    page = render.paper((width, height), rng)
    draw = ImageDraw.Draw(page)
    label_font = render.font(
        rng.choice(["sans", "serif", "mono"]), rng.randint(17, 21), rng
    )
    bold = render.font("sans_bold", rng.randint(26, 34), rng)
    hand = rng.random() < 0.6
    value_font = render.font(
        "hand" if hand else "mono",
        rng.randint(22, 28) if hand else rng.randint(17, 20),
        rng,
    )
    title = rng.choice(FORM_TITLES)
    org = rng.choice(ORGS)
    draw.text((60, 50), org, font=label_font, fill=(40, 40, 40))
    draw.text((60, 80), title.upper(), font=bold, fill=(0, 0, 0))
    labels = rng.sample(FIELDS, rng.randint(8, 13))
    values = {}
    for label in labels:
        if "Date" in label:
            values[label] = _date(rng, rng.choice([1, 2, 3]))
        elif label in (
            "Name",
            "Approved By",
            "Reviewed By",
            "Contact Person",
            "Supervisor",
            "Signature",
        ):
            values[label] = _person(rng)
        elif label in ("Phone", "Fax"):
            values[label] = (
                f"({rng.randint(200, 999)}) {rng.randint(200, 999)}-{rng.randint(1000, 9999)}"
            )
        elif label == "Amount":
            values[label] = f"${rng.uniform(50, 9000):,.2f}"
        elif label in ("Company", "Ship To", "Bill To"):
            values[label] = rng.choice(ORGS)
        elif label == "Department":
            values[label] = rng.choice(DEPTS)
        elif label in ("Location", "Destination", "Address"):
            values[label] = rng.choice(COUNTRIES)
        else:
            values[label] = f"{rng.choice('ABCDEFGHKMNPRST')}{rng.randint(1000, 99999)}"
    blank = set(rng.sample(labels, rng.randint(0, 2)))
    two_col = rng.random() < 0.5
    y = 150
    col_w = (width - 120) // (2 if two_col else 1)
    for k, label in enumerate(labels):
        cx = 60 + (k % 2) * col_w if two_col else 60
        if two_col and k % 2 == 0 and k:
            y += 62
        elif not two_col and k:
            y += 58
        draw.text((cx, y), f"{label}:", font=label_font, fill=(0, 0, 0))
        lx = cx + render.text_width(draw, f"{label}:", label_font) + 10
        if rng.random() < 0.5:
            draw.line((lx, y + 28, cx + col_w - 30, y + 28), fill=(0, 0, 0), width=1)
        else:
            draw.rectangle(
                (lx, y - 4, cx + col_w - 30, y + 30), outline=(0, 0, 0), width=1
            )
        if label not in blank:
            draw.text(
                (lx + 6, y - (6 if hand else 0)),
                values[label],
                font=value_font,
                fill=(20, 30, 130) if hand else (0, 0, 0),
            )
    y += 80
    check_q = rng.choice(
        ["Urgent", "Confidential", "Reply requested", "Sample attached", "Approved"]
    )
    answer = rng.choice(["Yes", "No", "N/A"])
    draw.text((60, y), f"{check_q}:", font=label_font, fill=(0, 0, 0))
    cx = 60 + render.text_width(draw, f"{check_q}:", label_font) + 20
    for opt in ("Yes", "No", "N/A"):
        draw.rectangle((cx, y + 2, cx + 20, y + 22), outline=(0, 0, 0), width=2)
        if opt == answer:
            draw.line((cx + 3, y + 12, cx + 9, y + 19), fill=(10, 10, 120), width=3)
            draw.line((cx + 9, y + 19, cx + 19, y + 3), fill=(10, 10, 120), width=3)
        draw.text((cx + 28, y), opt, font=label_font, fill=(0, 0, 0))
        cx += 110
    page = render.degrade_scan(page, rng)
    absent = [f for f in FIELDS if f not in labels]
    qtype = rng.choice(["presence", "presence_yes_no", "value", "value", "checkbox"])
    filled = [l for l in labels if l not in blank]
    text = " ".join([org, title, *labels, *[values[l] for l in filled], check_q])
    if qtype == "presence":
        gold = rng.choice(labels)
        options = [gold] + rng.sample(absent, 3)
        return Item(
            source="gen-document",
            family="form",
            skill="kie",
            images=[page],
            image_kinds=["png"],
            question="Which of these fields appears on the form?",
            options=options,
            meta={"qtype": qtype, "benchmark_target": "FUNSD"},
            image_text=[text],
        )
    if qtype == "presence_yes_no":
        present = rng.random() < 0.5
        label = rng.choice(labels if present else absent)
        return Item(
            source="gen-document",
            family="form",
            skill="kie",
            images=[page],
            image_kinds=["png"],
            question=f"Does the form contain a field labeled '{label}'?",
            noul=present,
            meta={"qtype": qtype, "benchmark_target": "FUNSD"},
            image_text=[text],
        )
    if qtype == "value" and filled:
        label = rng.choice(filled)
        pool = [values[l] for l in filled if l != label]
        same_kind = [v for v in pool if type(v) is type(values[label])]
        options = _pick(values[label], same_kind + pool, rng)
        if options is None:
            return None
        return Item(
            source="gen-document",
            family="form",
            skill="kie",
            images=[page],
            image_kinds=["png"],
            question=f"What is entered in the '{label}' field?",
            options=options,
            meta={"qtype": qtype, "benchmark_target": "FUNSD"},
            image_text=[text],
        )
    return Item(
        source="gen-document",
        family="form",
        skill="kie",
        images=[page],
        image_kinds=["png"],
        question=f"Which box is checked for '{check_q}'?",
        options=["Yes", "No", "N/A"],
        gold=["Yes", "No", "N/A"].index(answer),
        keys=["yes", "no", "n_a"],
        fixed_order=True,
        meta={"qtype": "checkbox", "benchmark_target": "FUNSD"},
        image_text=[text],
    )


__all__ = ["infographic", "document", "receipt", "form", "fmt_number"]
