"""Set-of-marks GUI decisions on fresh HTML pages rendered by headless Chromium (Mind2Web style).

A page is generated from one of several site templates (shop, travel, settings, news, jobs, restaurant)
with random content, theme and layout. A task and its previous actions define the gold next element; three
distractors are other visible interactive elements on the same screen, preferring the same element type.
The four elements are boxed and tagged on the screenshot, and the question asks which one to act on next.
"""

from __future__ import annotations

import html
import os
import random

from PIL import Image, ImageDraw

from d25.omni.data import render
from d25.omni.data.rows import Item, rng_for

FONTS = (
    "Lato",
    "Fira Sans",
    "Open Sans",
    "Roboto",
    "Noto Sans",
    "Inter",
    "Montserrat",
    "Poppins",
    "PT Sans",
    "Liberation Sans",
    "DejaVu Sans",
    "Nunito",
    "Raleway",
    "PT Serif",
    "Merriweather",
    "Source Serif 4",
)
CITIES = (
    "Mumbai",
    "New York",
    "London",
    "Tokyo",
    "Paris",
    "Sydney",
    "Toronto",
    "Berlin",
    "Madrid",
    "Dubai",
    "Singapore",
    "Chicago",
    "Seattle",
    "Rome",
    "Lisbon",
    "Seoul",
    "Cairo",
    "Lima",
    "Oslo",
    "Denver",
)
PRODUCTS = (
    "Wireless Earbuds",
    "Running Shoes",
    "Coffee Maker",
    "Desk Lamp",
    "Yoga Mat",
    "Backpack",
    "Smart Watch",
    "Bluetooth Speaker",
    "Water Bottle",
    "Office Chair",
    "Electric Kettle",
    "Phone Case",
    "Rain Jacket",
    "Air Fryer",
    "Gaming Mouse",
    "Mechanical Keyboard",
    "Sunglasses",
    "Tent",
    "Blender",
    "Monitor Stand",
    "Hiking Boots",
    "Travel Pillow",
    "Wall Clock",
    "Cutting Board",
    "Hair Dryer",
    "Notebook Set",
)
BRANDS = (
    "Acme",
    "Northpeak",
    "Lumina",
    "Vertex",
    "Orbit",
    "Kestrel",
    "Nimbus",
    "Solace",
    "Brava",
    "Tundra",
)
TOPICS = (
    "World",
    "Business",
    "Technology",
    "Science",
    "Sports",
    "Health",
    "Culture",
    "Travel",
    "Opinion",
    "Climate",
)
HEADLINES = (
    "Central bank holds rates steady",
    "New telescope captures distant galaxy",
    "City approves bike lane plan",
    "Local team wins championship",
    "Researchers unveil battery breakthrough",
    "Museum reopens after renovation",
    "Startup raises funding for clean water",
    "Heatwave expected this weekend",
    "Election debate draws record audience",
    "Study links sleep to memory",
    "Airline announces new routes",
    "Farmers adopt drought-resistant crops",
)
JOBS = (
    "Data Analyst",
    "Software Engineer",
    "Nurse",
    "Graphic Designer",
    "Project Manager",
    "Accountant",
    "Electrician",
    "Teacher",
    "Sales Associate",
    "UX Researcher",
    "Mechanical Engineer",
    "Chef",
)
RESTAURANTS = (
    "Olive & Thyme",
    "Sakura House",
    "The Copper Pot",
    "Casa Verde",
    "Blue Lagoon Grill",
    "Spice Route",
    "Little Paris Bistro",
    "Harbor Oyster Bar",
    "Golden Dragon",
    "Rustic Table",
)
SETTINGS = (
    "Email notifications",
    "Push notifications",
    "Promotional emails",
    "Two-factor authentication",
    "Dark mode",
    "Auto-play videos",
    "Location sharing",
    "Weekly summary",
    "Public profile",
    "Read receipts",
)


def _theme(rng: random.Random) -> str:
    palette = rng.choice(render.PALETTES)
    primary, accent = palette[0], palette[rng.randrange(1, len(palette))]
    bg = rng.choice(["#ffffff", "#f7f7f9", "#fbfaf7", "#f2f5f9"])
    radius = rng.choice([0, 3, 6, 10, 18])
    font = rng.choice(FONTS)
    size = rng.choice([14, 15, 16])
    return f"""
    * {{ box-sizing: border-box; }}
    body {{ margin:0; font-family:'{font}', sans-serif; font-size:{size}px; background:{bg}; color:#222; }}
    header {{ background:{primary}; color:#fff; padding:12px 24px; display:flex; align-items:center; gap:18px; }}
    header a {{ color:#fff; text-decoration:none; margin-right:14px; }}
    .logo {{ font-weight:700; font-size:22px; margin-right:20px; }}
    .wrap {{ display:flex; gap:20px; padding:18px 24px; }}
    aside {{ width:{rng.choice([200, 220, 240])}px; }}
    main {{ flex:1; }}
    button, .btn {{ background:{accent}; color:#fff; border:none; border-radius:{radius}px; padding:8px 14px;
                    font-size:{size}px; cursor:pointer; }}
    button.secondary {{ background:#e6e6e6; color:#222; }}
    input, select {{ border:1px solid #bbb; border-radius:{radius}px; padding:8px 10px; font-size:{size}px; }}
    .card {{ background:#fff; border:1px solid #ddd; border-radius:{radius}px; padding:12px; }}
    .grid {{ display:grid; grid-template-columns:repeat({rng.choice([2, 3, 4])}, 1fr); gap:14px; }}
    .thumb {{ height:{rng.choice([70, 90, 110])}px; border-radius:{radius}px; margin-bottom:8px; }}
    .muted {{ color:#666; font-size:{size - 2}px; }}
    .row {{ display:flex; gap:12px; align-items:center; margin:8px 0; flex-wrap:wrap; }}
    a {{ color:{primary}; }}
    """


def _el(tag: str, sid: str, content: str = "", **attrs) -> str:
    rendered = " ".join(
        f'{k.rstrip("_").replace("_", "-")}="{html.escape(str(v))}"'
        for k, v in attrs.items()
    )
    if tag == "input":
        return f'<input data-sid="{sid}" {rendered}>'
    return f'<{tag} data-sid="{sid}" {rendered}>{content}</{tag}>'


def _nav(rng, items, brand):
    links = "".join(
        _el("a", f"nav-{i}", html.escape(x), href="#") for i, x in enumerate(items)
    )
    return f'<header><span class="logo">{html.escape(brand)}</span>{links}</header>'


def shop(rng: random.Random):
    brand = rng.choice(BRANDS) + rng.choice([" Store", " Market", " Shop", ""])
    products = rng.sample(PRODUCTS, rng.choice([6, 8, 9]))
    brands = rng.sample(BRANDS, 5)
    cards = []
    for i, p in enumerate(products):
        colour = rng.choice(rng.choice(render.PALETTES))
        price = f"${rng.randint(9, 299)}.{rng.choice(['99', '49', '00'])}"
        cards.append(
            f'<div class="card"><div class="thumb" style="background:{colour}"></div>'
            f'{_el("a", f"title-{i}", html.escape(p), href="#")}<div class="muted">{price} · '
            f'{rng.uniform(3.2, 4.9):.1f}★</div><div class="row">{_el("button", f"cart-{i}", "Add to cart")}'
            f'{_el("button", f"wish-{i}", "♡", class_="secondary")}</div></div>'
        )
    filters = "".join(
        f'<div class="row">{_el("input", f"brand-{i}", type="checkbox")} {b}</div>'
        for i, b in enumerate(brands)
    )
    pages = "".join(_el("a", f"page-{n}", str(n), href="#") + " " for n in range(1, 6))
    body = (
        _nav(rng, ["Deals", "New", "Electronics", "Home", "Outdoor", "Sign in"], brand)
        + f'<div class="wrap"><aside><h3>Brand</h3>{filters}<h3>Price</h3>'
        + "".join(
            f'<div class="row">{_el("input", f"price-{k}", type="radio", name="p")} {r}</div>'
            for k, r in enumerate(["Under $25", "$25 to $100", "Over $100"])
        )
        + f'</aside><main><div class="row">{_el("input", "search", placeholder="Search products", style="width:360px")}'
        + f'{_el("button", "search-go", "Search")}</div><div class="grid">{"".join(cards)}</div>'
        + f'<div class="row">Page {pages}</div></main></div>'
    )
    kind = rng.choice(
        ["cart", "cart", "filter", "search_type", "search_click", "page", "open"]
    )
    if kind == "cart":
        i = rng.randrange(len(products))
        return (
            body,
            f"Add the {products[i]} to the shopping cart.",
            [],
            f"cart-{i}",
            "cart",
        )
    if kind == "open":
        i = rng.randrange(len(products))
        return (
            body,
            f"Open the product page for the {products[i]}.",
            [],
            f"title-{i}",
            "title",
        )
    if kind == "filter":
        i = rng.randrange(len(brands))
        return (
            body,
            f"Show only {brands[i]} products, then sort by price.",
            [],
            f"brand-{i}",
            "brand",
        )
    query = rng.choice(PRODUCTS).lower()
    if kind == "search_type":
        return body, f"Search for {query}.", [], "search", "search"
    if kind == "search_click":
        return (
            body,
            f"Search for {query}.",
            [f"[textbox] Search products -> TYPE: {query}"],
            "search-go",
            "search",
        )
    n = rng.randint(2, 5)
    return body, f"Go to page {n} of the results.", [], f"page-{n}", "page"


def travel(rng: random.Random):
    brand = rng.choice(["Skyway", "Jetset", "Horizon", "Wander", "Atlas"]) + rng.choice(
        [" Travel", " Air", " Trips"]
    )
    a, b = rng.sample(CITIES, 2)
    date = f"{rng.choice(['April', 'May', 'June', 'July', 'October'])} {rng.randint(1, 28)}"
    fields = [
        ("oneway", "radio", "One way"),
        ("from", "text", "From"),
        ("to", "text", "To"),
        ("depart", "text", "Depart date"),
        ("pax", "select", "Passengers"),
        ("go", "button", "Search flights"),
    ]
    form = (
        f'<div class="card" style="max-width:820px"><div class="row">{_el("input", "roundtrip", type="radio", name="t")} Round trip '
        f'{_el("input", "oneway", type="radio", name="t")} One way {_el("input", "multi", type="radio", name="t")} Multi-city</div>'
        f'<div class="row">{_el("input", "from", placeholder="From")}{_el("input", "to", placeholder="To")}'
        f'{_el("input", "depart", placeholder="Depart date")}{_el("input", "return", placeholder="Return date")}</div>'
        f'<div class="row">{_el("select", "pax", "".join(f"<option>{n} adult{"s" * (n > 1)}</option>" for n in range(1, 6)))}'
        f'{_el("select", "cabin", "<option>Economy</option><option>Business</option>")}'
        f'{_el("button", "go", "Search flights")}</div></div>'
    )
    deals = "".join(
        f'<div class="card">{_el("a", f"deal-{i}", f"{c} from ${rng.randint(99, 899)}", href="#")}</div>'
        for i, c in enumerate(rng.sample(CITIES, 4))
    )
    body = (
        _nav(rng, ["Flights", "Hotels", "Cars", "Deals", "Help", "Sign in"], brand)
        + f'<div class="wrap"><main><h2>Where to next?</h2>{form}<h3>Popular deals</h3><div class="grid">{deals}</div></main></div>'
    )
    task = f"Find a one-way flight from {a} to {b} on {date} for {rng.randint(1, 4)} adults."
    done = [
        f"[radio] One way -> CLICK",
        f"[textbox] From -> TYPE: {a}",
        f"[textbox] To -> TYPE: {b}",
        f"[textbox] Depart date -> TYPE: {date}",
        "[combobox] Passengers -> SELECT",
    ]
    k = rng.randrange(len(fields))
    return body, task, done[:k], fields[k][0], "form"


def settings(rng: random.Random):
    items = rng.sample(SETTINGS, rng.randint(5, 8))
    rows = "".join(
        f'<div class="row" style="justify-content:space-between;max-width:640px"><span>{s}</span>'
        f'{_el("input", f"toggle-{i}", type="checkbox")}</div>'
        for i, s in enumerate(items)
    )
    side = "".join(
        f'<div class="row">{_el("a", f"side-{i}", x, href="#")}</div>'
        for i, x in enumerate(
            ["Profile", "Security", "Notifications", "Billing", "Privacy"]
        )
    )
    body = (
        _nav(
            rng,
            ["Home", "Messages", "Settings", "Help"],
            rng.choice(BRANDS) + " Account",
        )
        + f'<div class="wrap"><aside>{side}</aside><main><h2>Settings</h2><div class="card">{rows}'
        + f'<div class="row">{_el("select", "lang", "<option>English</option><option>Español</option>")}</div>'
        + f'<div class="row">{_el("button", "save", "Save changes")}{_el("button", "cancel", "Cancel", class_="secondary")}</div>'
        + "</div></main></div>"
    )
    i = rng.randrange(len(items))
    kind = rng.choice(["toggle", "toggle", "save", "side"])
    if kind == "toggle":
        return (
            body,
            f"Turn {rng.choice(['on', 'off'])} {items[i].lower()}.",
            [],
            f"toggle-{i}",
            "toggle",
        )
    if kind == "save":
        return (
            body,
            f"Turn off {items[i].lower()} and keep the change.",
            [f"[checkbox] {items[i]} -> CLICK"],
            "save",
            "button",
        )
    j = rng.randrange(5)
    page = ["Profile", "Security", "Notifications", "Billing", "Privacy"][j]
    return body, f"Open the {page.lower()} settings page.", [], f"side-{j}", "side"


def news(rng: random.Random):
    heads = rng.sample(HEADLINES, rng.randint(5, 8))
    items = "".join(
        f'<div class="card">{_el("a", f"head-{i}", h, href="#")}<div class="muted">{rng.choice(TOPICS)} · '
        f"{rng.randint(2, 59)} min ago</div></div>"
        for i, h in enumerate(heads)
    )
    topics = rng.sample(TOPICS, 7)
    body = (
        _nav(
            rng,
            topics,
            rng.choice(["The Daily", "Metro News", "Globe Times", "Evening Post"]),
        )
        + f'<div class="wrap"><main><div class="row">{_el("input", "q", placeholder="Search news")}'
        + f'{_el("button", "q-go", "Search")}{_el("button", "sub", "Subscribe")}</div><div class="grid">{items}</div></main></div>'
    )
    kind = rng.choice(["head", "head", "topic", "subscribe"])
    if kind == "head":
        i = rng.randrange(len(heads))
        return body, f"Read the article titled '{heads[i]}'.", [], f"head-{i}", "head"
    if kind == "topic":
        i = rng.randrange(len(topics))
        return (
            body,
            f"Browse the latest {topics[i].lower()} stories.",
            [],
            f"nav-{i}",
            "nav",
        )
    return body, "Subscribe to the newsletter.", [], "sub", "button"


def jobs(rng: random.Random):
    roles = rng.sample(JOBS, rng.randint(4, 7))
    cards = "".join(
        f'<div class="card"><b>{r}</b><div class="muted">{rng.choice(CITIES)} · '
        f'{rng.choice(["Full-time", "Part-time", "Contract"])}</div><div class="row">'
        f'{_el("button", f"apply-{i}", "Apply")}{_el("button", f"save-{i}", "Save", class_="secondary")}</div></div>'
        for i, r in enumerate(roles)
    )
    city = rng.choice(CITIES)
    body = (
        _nav(
            rng,
            ["Find jobs", "Companies", "Salaries", "Post a job", "Sign in"],
            rng.choice(BRANDS) + " Careers",
        )
        + f'<div class="wrap"><aside><h3>Filters</h3><div class="row">{_el("input", "remote", type="checkbox")} Remote</div>'
        + f'<div class="row">{_el("input", "full", type="checkbox")} Full-time</div></aside><main><div class="row">'
        + f'{_el("input", "kw", placeholder="Job title or keyword")}{_el("input", "loc", placeholder="City")}'
        + f'{_el("button", "find", "Find jobs")}</div><div class="grid">{cards}</div></main></div>'
    )
    i = rng.randrange(len(roles))
    kind = rng.choice(["apply", "apply", "save", "type_loc", "remote"])
    if kind == "apply":
        return body, f"Apply for the {roles[i]} position.", [], f"apply-{i}", "apply"
    if kind == "save":
        return body, f"Save the {roles[i]} job for later.", [], f"save-{i}", "save"
    if kind == "type_loc":
        role = rng.choice(JOBS)
        return (
            body,
            f"Find {role.lower()} jobs in {city}.",
            [f"[textbox] Job title or keyword -> TYPE: {role}"],
            "loc",
            "search",
        )
    return body, "Show only remote jobs.", [], "remote", "filter"


def restaurant(rng: random.Random):
    names = rng.sample(RESTAURANTS, rng.randint(4, 6))
    cards = "".join(
        f'<div class="card"><b>{n}</b><div class="muted">{rng.choice(["Italian", "Japanese", "Mexican", "French", "Indian", "Seafood"])} · '
        f'{"$" * rng.randint(1, 4)}</div><div class="row">{_el("button", f"reserve-{i}", "Reserve")}'
        f'{_el("a", f"menu-{i}", "View menu", href="#")}</div></div>'
        for i, n in enumerate(names)
    )
    body = (
        _nav(
            rng,
            ["Discover", "Near me", "Offers", "Gift cards", "Sign in"],
            rng.choice(["TableNow", "DineOut", "Reservo"]),
        )
        + f'<div class="wrap"><main><div class="row">{_el("select", "date", "<option>Today</option><option>Tomorrow</option>")}'
        + f'{_el("select", "time", "".join(f"<option>{h}:00 PM</option>" for h in range(5, 10)))}'
        + f'{_el("select", "party", "".join(f"<option>{n} people</option>" for n in range(1, 9)))}'
        + f'{_el("button", "find", "Find a table")}</div><div class="grid">{cards}</div></main></div>'
    )
    i = rng.randrange(len(names))
    kind = rng.choice(["reserve", "reserve", "menu", "party"])
    if kind == "reserve":
        return body, f"Book a table at {names[i]}.", [], f"reserve-{i}", "reserve"
    if kind == "menu":
        return body, f"Look at the menu of {names[i]}.", [], f"menu-{i}", "menu"
    return (
        body,
        f"Reserve a table for {rng.randint(2, 8)} people tomorrow at 7 PM.",
        ["[combobox] Date -> SELECT: Tomorrow", "[combobox] Time -> SELECT: 7:00 PM"],
        "party",
        "select",
    )


SITES = (shop, shop, travel, settings, news, jobs, restaurant)
_BROWSER: dict = {}


def _page():
    if "page" not in _BROWSER:
        root = "/data/d25/omni/pylib"
        os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", f"{root}/ms-playwright")
        os.environ.setdefault("FONTCONFIG_FILE", f"{root}/fonts/fonts.conf")
        libs = f"{root}/syslibs/usr/lib/x86_64-linux-gnu:{root}/syslibs/lib/x86_64-linux-gnu"
        os.environ["LD_LIBRARY_PATH"] = (
            libs + ":" + os.environ.get("LD_LIBRARY_PATH", "")
        )
        from playwright.sync_api import sync_playwright

        _BROWSER["pw"] = sync_playwright().start()
        _BROWSER["browser"] = _BROWSER["pw"].chromium.launch(
            args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"]
        )
        _BROWSER["page"] = _BROWSER["browser"].new_page()
    return _BROWSER["page"]


COLLECT = """() => Array.from(document.querySelectorAll('[data-sid]')).map(e => {
  const r = e.getBoundingClientRect();
  return {sid: e.dataset.sid, tag: e.tagName.toLowerCase(), type: e.type || '', text: (e.innerText || e.placeholder || '').slice(0, 60),
          x: r.x, y: r.y, w: r.width, h: r.height};
}).filter(e => e.w > 4 && e.h > 4 && e.y >= 0 && e.y + e.h <= window.innerHeight && e.x >= 0 && e.x + e.w <= window.innerWidth)"""


def _overlaps(a, b, gap=2) -> bool:
    return not (
        a[2] + gap <= b[0]
        or b[2] + gap <= a[0]
        or a[3] + gap <= b[1]
        or b[3] + gap <= a[1]
    )


def draw_marks(
    image: Image.Image, boxes: list[tuple], labels: list[str], colours: list[str], face
) -> bool:
    """Boxes with tags placed outside every other box and tag; False when a tag cannot be placed."""
    draw = ImageDraw.Draw(image)
    w, h = image.size
    placed: list[tuple] = []
    tags = []
    for box, label in zip(boxes, labels):
        tw = render.text_width(draw, label, face) + 8
        th = render.line_height(face) + 4
        x0, y0, x1, y1 = box
        cy = (y0 + y1) // 2 - th // 2
        spots = [
            (x0 - tw - 3, cy),
            (x0, y0 - th - 2),
            (x1 + 3, cy),
            (x0, y1 + 2),
            (x1 - tw, y0 - th - 2),
        ]
        for sx, sy in spots:
            rect = (sx, sy, sx + tw, sy + th)
            if rect[0] < 0 or rect[1] < 0 or rect[2] > w or rect[3] > h:
                continue
            if any(_overlaps(rect, other) for other in boxes) or any(
                _overlaps(rect, t) for t in placed
            ):
                continue
            placed.append(rect)
            tags.append(rect)
            break
        else:
            return False
    for box, rect, label, colour in zip(boxes, tags, labels, colours):
        draw.rectangle(box, outline=colour, width=3)
        draw.rectangle(rect, fill=colour)
        draw.text((rect[0] + 4, rect[1] + 1), label, font=face, fill="white")
    return True


def describe(element: dict) -> str:
    role = {"a": "link", "button": "button", "select": "combobox"}.get(element["tag"])
    if element["tag"] == "input":
        role = {"checkbox": "checkbox", "radio": "radio"}.get(
            element["type"], "textbox"
        )
    return f"[{role}] {element['text'].strip()}".strip()


def generate(index: int, ctx=None) -> Item | None:
    rng = rng_for("gen-gui", index)
    site = rng.choice(SITES)
    body, task, history, gold_sid, group = site(rng)
    width, height = rng.choice(
        [(1280, 800), (1280, 900), (1366, 768), (1440, 900), (1200, 1000)]
    )
    page = _page()
    page.set_viewport_size({"width": width, "height": height})
    page.set_content(
        f"<html><head><style>{_theme(rng)}</style></head><body>{body}</body></html>"
    )
    elements = page.evaluate(COLLECT)
    by_sid = {e["sid"]: e for e in elements}
    if gold_sid not in by_sid:
        return None
    gold = by_sid[gold_sid]
    prefix = gold_sid.split("-")[0]
    same = [
        e for e in elements if e["sid"] != gold_sid and e["sid"].split("-")[0] == prefix
    ]
    other = [e for e in elements if e["sid"] != gold_sid and e not in same]
    rng.shuffle(same)
    other.sort(
        key=lambda e: abs(e["y"] - gold["y"])
        + abs(e["x"] - gold["x"])
        + rng.uniform(0, 300)
    )
    pad = 3
    rect = lambda e: (
        int(e["x"] - pad),
        int(e["y"] - pad),
        int(e["x"] + e["w"] + pad),
        int(e["y"] + e["h"] + pad),
    )
    chosen = [gold]
    for e in same[: rng.randint(1, 3)] + other:
        if len(chosen) == 4:
            break
        if all(not _overlaps(rect(e), rect(c), gap=6) for c in chosen):
            chosen.append(e)
    if len(chosen) < 4:
        return None
    shot = page.screenshot(type="png")
    from io import BytesIO

    image = Image.open(BytesIO(shot)).convert("RGB")
    order = list(range(4))
    rng.shuffle(order)
    labels = rng.choice([["A", "B", "C", "D"], ["1", "2", "3", "4"]])
    colours = rng.choice(
        [["#e6194b"] * 4, ["#e6194b", "#3cb44b", "#4363d8", "#f58231"], ["#ff00ff"] * 4]
    )
    face = render.font("sans_bold", rng.choice([13, 15, 17]), rng)
    if not draw_marks(image, [rect(chosen[k]) for k in order], labels, colours, face):
        return None
    actions = "\n".join(history) if history else "None"
    state = f"Task: {task}\nPrevious actions:\n{actions}"
    question = rng.choice(
        [
            "Which marked element should be acted on next to continue the task?",
            "Which of the boxed elements is the next one to interact with?",
            "To make progress on the task, which marked element should be used next?",
        ]
    )
    texts = [describe(e) for e in elements]
    return Item(
        source="gen-gui",
        family="gui-som",
        skill="gui",
        images=[image],
        image_kinds=["png"],
        question=question,
        options=[f"box {labels[i]}" for i in range(4)],
        gold=order.index(0),
        keys=labels,
        fixed_order=True,
        state=state,
        meta={
            "benchmark_target": "Mind2Web",
            "site": site.__name__,
            "group": group,
            "gold_element": describe(gold),
        },
        image_text=[" ".join(texts)],
    )
