"""Mind2Web proxy: rendered web pages, a task with its action history, and 4 marked candidate elements.

Each row is a freshly generated website page (flight, hotel and car-rental search, shopping lists with
filters, restaurant booking, job search) rendered by headless Chromium at 1280 px width after the
first k steps of a known plan were applied (filled fields, checked filters, updated results). The
next plan step's element is the answer; three real distractors are other interactive elements of the
same page, mostly from the same form or result list. Four lettered boxes are drawn on the screenshot.
"""

from __future__ import annotations

import datetime as dt
import html as htmlmod
import io
import random
from dataclasses import dataclass, field
from typing import Any

from PIL import Image

BENCHMARK = "Mind2Web"
NAME = "mind2web-proxy"
VERSION = "1"

CITIES = [
    "Boston",
    "Denver",
    "Seattle",
    "Chicago",
    "Austin",
    "Miami",
    "Atlanta",
    "Phoenix",
    "Portland",
    "Dallas",
    "New York",
    "San Diego",
    "Las Vegas",
    "Orlando",
    "Nashville",
    "Toronto",
    "London",
    "Paris",
    "Madrid",
    "Tokyo",
]
BRANDS = [
    "Acme",
    "Northwind",
    "Zenith",
    "Orbit",
    "Lumen",
    "Vertex",
    "Nimbus",
    "Polar",
    "Aurora",
    "Summit",
]
PRODUCTS = [
    "running shoes",
    "wireless headphones",
    "backpack",
    "coffee maker",
    "desk lamp",
    "yoga mat",
    "water bottle",
    "rain jacket",
    "office chair",
    "smart watch",
    "blender",
    "air fryer",
]
CUISINES = [
    "Italian",
    "Thai",
    "Mexican",
    "Japanese",
    "Indian",
    "French",
    "Greek",
    "Korean",
]
JOBS = [
    "data analyst",
    "nurse",
    "software engineer",
    "accountant",
    "graphic designer",
    "electrician",
    "project manager",
    "teacher",
    "pharmacist",
    "sales associate",
]
SITE_NAMES = [
    "Skyway",
    "TripNest",
    "GoFar",
    "ShopHub",
    "Cartly",
    "DineOut",
    "TableNow",
    "RoadRent",
    "WheelGo",
    "StayWell",
    "RoomFinder",
    "JobBoard",
    "HireLane",
    "Bazaar",
    "Voyago",
]
FONTS = ["Liberation Sans", "DejaVu Sans", "Noto Sans", "Arial", "sans-serif"]
THEMES = [
    "#0b5fff",
    "#d93025",
    "#0f9d58",
    "#6200ee",
    "#ff6d00",
    "#00796b",
    "#c2185b",
    "#37474f",
]
BOX_COLORS = [(230, 25, 75), (0, 130, 200), (60, 180, 75), (245, 130, 48)]


@dataclass
class Page:
    """HTML parts with interactive elements tagged ``data-el``; ``groups`` marks related elements."""

    title: str
    parts: list[str] = field(default_factory=list)
    labels: dict[str, str] = field(default_factory=dict)
    groups: dict[str, str] = field(default_factory=dict)
    n: int = 0

    def el(
        self,
        tag: str,
        label: str,
        group: str,
        inner: str = "",
        attrs: str = "",
        kind: str = "",
    ) -> tuple[str, str]:
        self.n += 1
        eid = f"e{self.n}"
        self.labels[eid] = f"[{kind or tag}] {label}"
        self.groups[eid] = group
        if tag == "input":
            return eid, f'<input data-el="{eid}" {attrs}>'
        return eid, f'<{tag} data-el="{eid}" {attrs}>{inner}</{tag}>'


def chrome(
    page: Page, r: random.Random, theme: str, site: str, nav: list[str]
) -> tuple[str, str]:
    links = "".join(
        page.el("a", n, "nav", htmlmod.escape(n), 'href="#"', "link")[1] for n in nav
    )
    signin = page.el("button", "Sign in", "nav", "Sign in", 'class="ghost"')[1]
    header = f'<header style="background:{theme}"><div class=logo>{htmlmod.escape(site)}</div><nav>{links}</nav>{signin}</header>'
    foot_links = r.sample(
        [
            "About us",
            "Careers",
            "Help center",
            "Privacy",
            "Terms",
            "Accessibility",
            "Gift cards",
            "Investors",
            "Contact",
            "Site map",
            "Press",
            "Affiliates",
        ],
        6,
    )
    footer = (
        "<footer>"
        + "".join(
            page.el("a", n, "footer", htmlmod.escape(n), 'href="#"', "link")[1]
            for n in foot_links
        )
        + "</footer>"
    )
    return header, footer


def filler(page: Page, r: random.Random) -> tuple[str, str]:
    """Promo banner and extra link/button strips, as real pages carry them."""
    promo = ""
    if r.random() < 0.6:
        text = r.choice(
            [
                "Members save up to 20% today",
                "Free shipping on orders over $35",
                "Download our app for exclusive offers",
                "New: earn points on every booking",
                "Holiday sale ends Sunday",
            ]
        )
        cta = r.choice(
            ["Join now", "Learn more", "Get the app", "Shop the sale", "See offers"]
        )
        promo = f'<div class=promo><span>{htmlmod.escape(text)}</span>{page.el("button", cta, "promo", cta, "class=sec")[1]}</div>'
    strips = ""
    for _ in range(r.randint(1, 2)):
        title = r.choice(
            [
                "Recently viewed",
                "Trending now",
                "Recommended for you",
                "Top picks",
                "Guides and tips",
            ]
        )
        tiles = "".join(
            f"<div class=tile><b>{htmlmod.escape(t)}</b><div class=muted>{r.randint(2, 98)} reviews</div>"
            f'{page.el("a", f"Read more: {t}", "filler", "Read more", "href=#", "link")[1]}</div>'
            for t in r.sample(
                [
                    "Packing list",
                    "Best time to visit",
                    "Budget tips",
                    "Gift ideas",
                    "How to choose",
                    "Top 10 lists",
                    "Seasonal picks",
                    "Staff favorites",
                    "Local guide",
                    "Buying guide",
                    "FAQ",
                    "Price alerts",
                ],
                r.randint(3, 5),
            )
        )
        strips += f"<div class=strip><h2>{title}</h2><div class=row>{tiles}</div></div>"
    return promo, strips


def css(theme: str, font: str) -> str:
    return f"""
    body {{ margin:0; font-family:'{font}'; color:#202124; background:#f5f6f8; }}
    header {{ display:flex; align-items:center; gap:28px; padding:14px 32px; color:#fff; }}
    .logo {{ font-size:26px; font-weight:800; }} nav {{ display:flex; gap:22px; flex:1; }} nav a {{ color:#fff; text-decoration:none; font-size:15px; }}
    button {{ font-family:inherit; font-size:15px; padding:9px 16px; border-radius:6px; border:1px solid {theme}; background:{theme}; color:#fff; cursor:pointer; }}
    button.ghost {{ background:transparent; border-color:#fff; color:#fff; }} button.sec {{ background:#fff; color:{theme}; }}
    main {{ display:flex; gap:24px; padding:24px 32px; }} .panel {{ background:#fff; border-radius:10px; padding:18px 22px; box-shadow:0 1px 3px #0002; }}
    .form {{ display:flex; flex-wrap:wrap; gap:14px; align-items:flex-end; }} .fld {{ display:flex; flex-direction:column; gap:4px; font-size:13px; color:#555; }}
    input, select {{ font-family:inherit; font-size:15px; padding:8px 10px; border:1px solid #bbb; border-radius:6px; min-width:150px; background:#fff; color:#202124; }}
    input[type=checkbox], input[type=radio] {{ min-width:0; width:auto; padding:0; margin:0 6px 0 0; }}
    .promo {{ display:flex; justify-content:space-between; align-items:center; padding:14px 32px; background:#fff8e1; font-size:15px; }}
    .strip {{ padding:0 32px; }} .strip h2 {{ font-size:19px; margin:18px 0 8px; }} .row {{ display:flex; gap:14px; flex-wrap:wrap; }}
    .tile {{ background:#fff; border-radius:8px; padding:12px; width:200px; box-shadow:0 1px 3px #0002; font-size:14px; }}
    .radio {{ display:flex; gap:16px; font-size:15px; margin-bottom:10px; }} .radio label {{ display:flex; gap:6px; align-items:center; }}
    .cards {{ display:grid; grid-template-columns:repeat(3, 1fr); gap:16px; margin-top:20px; }}
    .card {{ background:#fff; border-radius:10px; padding:14px; box-shadow:0 1px 3px #0002; font-size:14px; }}
    .card h4 {{ margin:4px 0 6px; font-size:16px; }} .price {{ font-size:18px; font-weight:700; color:{theme}; }}
    .side {{ width:230px; flex:none; }} .side h3 {{ font-size:15px; margin:14px 0 6px; }} .side label {{ display:block; font-size:14px; margin:5px 0; }}
    .res {{ display:flex; justify-content:space-between; align-items:center; background:#fff; padding:14px 18px; border-radius:10px; margin:10px 0; box-shadow:0 1px 3px #0002; }}
    .muted {{ color:#666; font-size:13px; }} h1 {{ font-size:26px; margin:0 0 14px; }}
    footer {{ display:flex; gap:26px; padding:24px 32px; background:#2b2f36; margin-top:24px; }} footer a {{ color:#cfd3da; font-size:13px; text-decoration:none; }}
    .slot {{ display:inline-block; margin:4px 6px 0 0; }}
    """


def date_text(r: random.Random) -> tuple[str, str]:
    d = dt.date(2026, r.randint(10, 12), r.randint(1, 28))
    return d.strftime("%B %d").replace(" 0", " "), d.strftime("%m/%d/%Y")


def flights(r: random.Random, theme: str) -> tuple[Page, list[tuple[str, str]], str]:
    page = Page("flights")
    a, b = r.sample(CITIES, 2)
    when, when_val = date_text(r)
    n = r.randint(1, 4)
    cabin = r.choice(["Economy", "Premium economy", "Business"])
    oneway = r.random() < 0.6
    task = f"Find a {'one-way' if oneway else 'round-trip'} flight from {a} to {b} on {when} for {n} adult{'s' * (n > 1)} in {cabin.lower()} class."
    steps_spec = []
    rt_id, rt = page.el(
        "input", "Round trip", "form", attrs='type="radio" name="t"', kind="radio"
    )
    ow_id, ow = page.el(
        "input", "One way", "form", attrs='type="radio" name="t"', kind="radio"
    )
    mc_id, mc = page.el(
        "input", "Multi-city", "form", attrs='type="radio" name="t"', kind="radio"
    )
    fr_id, _ = page.el("input", "From", "form", kind="textbox")
    to_id, _ = page.el("input", "To", "form", kind="textbox")
    dp_id, _ = page.el("input", "Depart", "form", kind="textbox")
    rd_id, _ = page.el("input", "Return", "form", kind="textbox")
    px_id, _ = page.el("select", "Travelers", "form", kind="combobox")
    cb_id, _ = page.el("select", "Cabin class", "form", kind="combobox")
    go_id, _ = page.el("button", "Search flights", "form")
    if oneway:
        steps_spec.append((ow_id, "CLICK"))
    steps_spec += [
        (fr_id, f"TYPE: {a}"),
        (to_id, f"TYPE: {b}"),
        (dp_id, f"TYPE: {when_val}"),
    ]
    if n > 1:
        steps_spec.append((px_id, f"SELECT: {n} adults"))
    if cabin != "Economy":
        steps_spec.append((cb_id, f"SELECT: {cabin}"))
    steps_spec.append((go_id, "CLICK"))
    return (
        page,
        steps_spec,
        task,
        {
            "ids": dict(
                rt=rt_id,
                ow=ow_id,
                mc=mc_id,
                fr=fr_id,
                to=to_id,
                dp=dp_id,
                rd=rd_id,
                px=px_id,
                cb=cb_id,
                go=go_id,
            ),
            "values": {
                fr_id: a,
                to_id: b,
                dp_id: when_val,
                px_id: f"{n} adults" if n > 1 else "1 adult",
                cb_id: cabin,
                ow_id: "on",
            },
        },
    )


def render_flights(
    page: Page, done: set[str], extra: dict, r: random.Random, theme: str
) -> str:
    ids, values = extra["ids"], extra["values"]
    v = lambda k: htmlmod.escape(values[ids[k]]) if ids[k] in done else ""  # noqa: E731
    checked_ow = "checked" if ids["ow"] in done else ""
    checked_rt = "" if checked_ow else "checked"
    pax = values[ids["px"]] if ids["px"] in done else "1 adult"
    cab = values[ids["cb"]] if ids["cb"] in done else "Economy"
    deals = "".join(
        f"<div class=card><h4>{htmlmod.escape(c)}</h4><div class=muted>Round trip from</div><div class=price>${r.randint(89, 899)}</div>"
        f'{page.el("button", f"View deals to {c}", "deals", "View deals", "class=sec")[1]}</div>'
        for c in r.sample(CITIES, 6)
    )
    return f"""<div class=panel style="width:100%"><h1>Where to next?</h1>
    <div class=radio><label><input data-el="{ids['rt']}" type=radio name=t {checked_rt}>Round trip</label>
    <label><input data-el="{ids['ow']}" type=radio name=t {checked_ow}>One way</label>
    <label><input data-el="{ids['mc']}" type=radio name=t>Multi-city</label></div>
    <div class=form>
    <div class=fld>From<input data-el="{ids['fr']}" placeholder="City or airport" value="{v('fr')}"></div>
    <div class=fld>To<input data-el="{ids['to']}" placeholder="City or airport" value="{v('to')}"></div>
    <div class=fld>Depart<input data-el="{ids['dp']}" placeholder="mm/dd/yyyy" value="{v('dp')}"></div>
    <div class=fld>Return<input data-el="{ids['rd']}" placeholder="mm/dd/yyyy" {'disabled' if checked_ow else ''}></div>
    <div class=fld>Travelers<select data-el="{ids['px']}"><option>{htmlmod.escape(pax)}</option></select></div>
    <div class=fld>Cabin<select data-el="{ids['cb']}"><option>{htmlmod.escape(cab)}</option></select></div>
    <button data-el="{ids['go']}">Search flights</button></div>
    <h2 style="margin-top:28px;font-size:20px">Popular destinations</h2><div class=cards>{deals}</div></div>"""


def shopping(r: random.Random, theme: str):
    page = Page("shopping")
    product = r.choice(PRODUCTS)
    brands = r.sample(BRANDS, 5)
    target_brand = brands[r.randrange(5)]
    items = []
    for i in range(r.randint(8, 12)):
        brand = target_brand if i < 4 else r.choice(brands)
        items.append(
            {
                "brand": brand,
                "name": f"{brand} {product.title()} {r.choice(['Pro', 'Lite', 'Max', 'Go', 'Air', 'Classic', 'X2', 'Plus'])}",
                "price": round(r.uniform(15, 300), 2),
                "stars": round(r.uniform(3.0, 5.0), 1),
            }
        )
    min_stars = r.choice([4.0, 4.5])
    eligible = [
        it for it in items if it["brand"] == target_brand and it["stars"] >= min_stars
    ]
    if len(eligible) < 2:
        return None
    best = min(eligible, key=lambda it: it["price"])
    task = f"Add the cheapest {target_brand} {product} rated {min_stars:g} stars or higher to the cart."
    search_id, _ = page.el("input", "Search products", "top", kind="searchbox")
    brand_ids = {
        b: page.el("input", b, "filters", attrs="type=checkbox", kind="checkbox")[0]
        for b in brands
    }
    star_ids = {
        s: page.el(
            "input", f"{s:g} stars & up", "filters", attrs="type=radio", kind="radio"
        )[0]
        for s in (3.0, 4.0, 4.5)
    }
    sort_id, _ = page.el("select", "Sort by", "results", kind="combobox")
    for it in items:
        it["add"] = page.el("button", f"Add to cart: {it['name']}", "results")[0]
        it["link"] = page.el("a", it["name"], "results", kind="link")[0]
    steps = [
        (brand_ids[target_brand], "CLICK"),
        (star_ids[min_stars], "CLICK"),
        (sort_id, "SELECT: Price: low to high"),
        (best["add"], "CLICK"),
    ]
    return (
        page,
        steps,
        task,
        {
            "items": items,
            "brands": brands,
            "brand_ids": brand_ids,
            "star_ids": star_ids,
            "sort_id": sort_id,
            "search_id": search_id,
            "product": product,
            "target": target_brand,
            "min_stars": min_stars,
        },
    )


def render_shopping(
    page: Page, done: set[str], extra: dict, r: random.Random, theme: str
) -> str:
    items = list(extra["items"])
    if extra["brand_ids"][extra["target"]] in done:
        items = [it for it in items if it["brand"] == extra["target"]]
    if extra["star_ids"][extra["min_stars"]] in done:
        items = [it for it in items if it["stars"] >= extra["min_stars"]]
    sorted_ = extra["sort_id"] in done
    if sorted_:
        items.sort(key=lambda it: it["price"])
    brands = "".join(
        f'<label><input data-el="{extra["brand_ids"][b]}" type=checkbox {"checked" if extra["brand_ids"][b] in done else ""}> {htmlmod.escape(b)}</label>'
        for b in extra["brands"]
    )
    stars = "".join(
        f'<label><input data-el="{extra["star_ids"][s]}" type=radio name=s {"checked" if extra["star_ids"][s] in done else ""}> {s:g} stars &amp; up</label>'
        for s in (3.0, 4.0, 4.5)
    )
    cards = "".join(
        f'<div class=card><a data-el="{it["link"]}" href="#"><h4>{htmlmod.escape(it["name"])}</h4></a><div class=muted>{"★" * int(it["stars"])} {it["stars"]:.1f}</div>'
        f'<div class=price>${it["price"]:.2f}</div><button data-el="{it["add"]}">Add to cart</button></div>'
        for it in items
    )
    return f"""<div class="panel side"><input data-el="{extra['search_id']}" placeholder="Search" value="{htmlmod.escape(extra['product'])}" style="min-width:0;width:180px">
    <h3>Brand</h3>{brands}<h3>Customer rating</h3>{stars}</div>
    <div style="flex:1"><div style="display:flex;justify-content:space-between;align-items:center"><h1>{htmlmod.escape(extra['product'].title())}</h1>
    <div class=fld>Sort by<select data-el="{extra['sort_id']}"><option>{'Price: low to high' if sorted_ else 'Featured'}</option></select></div></div>
    <div class=muted>{len(items)} results</div><div class=cards>{cards}</div></div>"""


def restaurant(r: random.Random, theme: str):
    page = Page("restaurant")
    cuisine = r.choice(CUISINES)
    city = r.choice(CITIES)
    when, when_val = date_text(r)
    party = r.randint(2, 8)
    time_ = r.choice(["6:00 PM", "6:30 PM", "7:00 PM", "7:30 PM", "8:00 PM"])
    task = (
        f"Book a table for {party} at an {cuisine} restaurant in {city} on {when} at {time_}."
        if cuisine[0] in "AEIOU"
        else f"Book a table for {party} at a {cuisine} restaurant in {city} on {when} at {time_}."
    )
    loc_id = page.el("input", "Location", "form", kind="textbox")[0]
    date_id = page.el("input", "Date", "form", kind="textbox")[0]
    time_id = page.el("select", "Time", "form", kind="combobox")[0]
    party_id = page.el("select", "Party size", "form", kind="combobox")[0]
    cuisine_id = page.el("select", "Cuisine", "form", kind="combobox")[0]
    find_id = page.el("button", "Find a table", "form")[0]
    restos = []
    names = r.sample(
        [
            "Bella",
            "Saffron",
            "Lotus",
            "Olive",
            "Maple",
            "Harbor",
            "Ember",
            "Juniper",
            "Basil",
            "Copper",
        ],
        4,
    )
    slots = ["6:00 PM", "6:30 PM", "7:00 PM", "7:30 PM", "8:00 PM"]
    for i, nm in enumerate(names):
        c = cuisine if i == 0 else r.choice(CUISINES)
        restos.append(
            {
                "name": f"{nm} {r.choice(['Kitchen', 'Bistro', 'House', 'Table'])}",
                "cuisine": c,
                "slots": {
                    s: page.el("button", f"{s} at {nm}", "results")[0] for s in slots
                },
            }
        )
    r.shuffle(restos)
    target = next(x for x in restos if x["cuisine"] == cuisine)
    steps = [
        (loc_id, f"TYPE: {city}"),
        (date_id, f"TYPE: {when_val}"),
        (time_id, f"SELECT: {time_}"),
        (party_id, f"SELECT: {party} people"),
        (cuisine_id, f"SELECT: {cuisine}"),
        (find_id, "CLICK"),
        (target["slots"][time_], "CLICK"),
    ]
    return (
        page,
        steps,
        task,
        {
            "ids": dict(
                loc=loc_id,
                date=date_id,
                time=time_id,
                party=party_id,
                cuisine=cuisine_id,
                find=find_id,
            ),
            "values": {
                loc_id: city,
                date_id: when_val,
                time_id: time_,
                party_id: f"{party} people",
                cuisine_id: cuisine,
            },
            "restos": restos,
            "find": find_id,
        },
    )


def render_restaurant(
    page: Page, done: set[str], extra: dict, r: random.Random, theme: str
) -> str:
    ids, values = extra["ids"], extra["values"]
    val = lambda k, default="": (
        htmlmod.escape(values[ids[k]]) if ids[k] in done else default
    )  # noqa: E731
    results = ""
    if extra["find"] in done:
        for x in extra["restos"]:
            slot_buttons = "".join(
                f'<button class="sec slot" data-el="{eid}">{s}</button>'
                for s, eid in x["slots"].items()
            )
            results += f'<div class=res><div><b>{htmlmod.escape(x["name"])}</b><div class=muted>{x["cuisine"]} · ${"$" * r.randint(1, 3)}</div></div><div>{slot_buttons}</div></div>'
    else:
        for x in extra["restos"]:
            for eid in x["slots"].values():
                page.labels.pop(eid, None)
    return f"""<div style="flex:1"><div class=panel><h1>Reserve a table</h1><div class=form>
    <div class=fld>Location<input data-el="{ids['loc']}" placeholder="City" value="{val('loc')}"></div>
    <div class=fld>Date<input data-el="{ids['date']}" placeholder="mm/dd/yyyy" value="{val('date')}"></div>
    <div class=fld>Time<select data-el="{ids['time']}"><option>{val('time', '7:00 PM')}</option></select></div>
    <div class=fld>Party size<select data-el="{ids['party']}"><option>{val('party', '2 people')}</option></select></div>
    <div class=fld>Cuisine<select data-el="{ids['cuisine']}"><option>{val('cuisine', 'Any cuisine')}</option></select></div>
    <button data-el="{ids['find']}">Find a table</button></div></div>{results}</div>"""


def jobs(r: random.Random, theme: str):
    page = Page("jobs")
    role = r.choice(JOBS)
    city = r.choice(CITIES)
    kind = r.choice(["Full-time", "Part-time", "Contract"])
    remote = r.random() < 0.4
    task = f"Search for {kind.lower()} {role} jobs in {city}{' that allow remote work' if remote else ''} and open the newest posting."
    kw_id = page.el("input", "Job title or keyword", "form", kind="textbox")[0]
    loc_id = page.el("input", "Location", "form", kind="textbox")[0]
    go_id = page.el("button", "Search jobs", "form")[0]
    type_ids = {
        t: page.el("input", t, "filters", attrs="type=checkbox", kind="checkbox")[0]
        for t in ["Full-time", "Part-time", "Contract", "Internship"]
    }
    remote_id = page.el(
        "input", "Remote", "filters", attrs="type=checkbox", kind="checkbox"
    )[0]
    posts = []
    for i in range(r.randint(5, 7)):
        days = r.randint(1, 30)
        posts.append(
            {
                "title": f"{role.title()}{r.choice(['', ' II', ' (Senior)', ' - Night shift', ' Lead'])}",
                "company": r.choice(
                    [
                        "Northwind",
                        "Contoso",
                        "Globex",
                        "Initech",
                        "Umbrella",
                        "Hooli",
                        "Vandelay",
                    ]
                ),
                "days": days,
                "id": page.el("a", f"job posting {i}", "results", kind="link")[0],
                "save": page.el("button", f"Save job {i}", "results")[0],
            }
        )
    newest = min(posts, key=lambda p: p["days"])
    if sorted(p["days"] for p in posts)[1] == newest["days"]:
        return None
    steps = [
        (kw_id, f"TYPE: {role}"),
        (loc_id, f"TYPE: {city}"),
        (go_id, "CLICK"),
        (type_ids[kind], "CLICK"),
    ]
    if remote:
        steps.append((remote_id, "CLICK"))
    steps.append((newest["id"], "CLICK"))
    return (
        page,
        steps,
        task,
        {
            "kw": kw_id,
            "loc": loc_id,
            "go": go_id,
            "types": type_ids,
            "remote": remote_id,
            "posts": posts,
            "role": role,
            "city": city,
        },
    )


def render_jobs(
    page: Page, done: set[str], extra: dict, r: random.Random, theme: str
) -> str:
    kw = htmlmod.escape(extra["role"]) if extra["kw"] in done else ""
    loc = htmlmod.escape(extra["city"]) if extra["loc"] in done else ""
    types = "".join(
        f'<label><input data-el="{eid}" type=checkbox {"checked" if eid in done else ""}> {t}</label>'
        for t, eid in extra["types"].items()
    )
    remote = f'<label><input data-el="{extra["remote"]}" type=checkbox {"checked" if extra["remote"] in done else ""}> Remote</label>'
    if extra["go"] in done:
        posts = "".join(
            f'<div class=res><div><a data-el="{p["id"]}" href="#"><b>{htmlmod.escape(p["title"])}</b></a><div class=muted>{p["company"]} · {htmlmod.escape(extra["city"])} · posted {p["days"]} day{"s" * (p["days"] > 1)} ago</div></div>'
            f'<button class=sec data-el="{p["save"]}">Save</button></div>'
            for p in extra["posts"]
        )
    else:
        posts = '<div class=muted style="margin-top:20px">Enter a job title and location to see results.</div>'
        for p in extra["posts"]:
            page.labels.pop(p["id"], None)
            page.labels.pop(p["save"], None)
    return f"""<div class="panel side"><h3>Job type</h3>{types}<h3>Workplace</h3>{remote}</div><div style="flex:1">
    <div class=panel><div class=form><div class=fld>What<input data-el="{extra['kw']}" placeholder="Job title or keyword" value="{kw}"></div>
    <div class=fld>Where<input data-el="{extra['loc']}" placeholder="City" value="{loc}"></div><button data-el="{extra['go']}">Search jobs</button></div></div>{posts}</div>"""


SITES = {
    "flights": (
        flights,
        render_flights,
        ["Flights", "Hotels", "Cars", "Deals", "Trips"],
    ),
    "shopping": (
        shopping,
        render_shopping,
        ["Today's deals", "Electronics", "Home", "Sports", "Orders"],
    ),
    "restaurant": (
        restaurant,
        render_restaurant,
        ["Restaurants", "Offers", "Gift cards", "Blog"],
    ),
    "jobs": (jobs, render_jobs, ["Find jobs", "Companies", "Salaries", "Post a job"]),
}


def worker_init() -> None:
    from d25.omni.proxy import render

    render.start()


def worker_close() -> None:
    from d25.omni.proxy import render

    render.stop()


def build_item(index: int, seed: int, context: Any = None) -> dict[str, Any]:
    from d25.omni.proxy import augment, render
    from d25.omni.proxy.rows import rng

    for attempt in range(80):
        r = rng(NAME, seed, index, attempt)
        site_key = r.choice(sorted(SITES))
        make, draw, nav = SITES[site_key]
        theme = r.choice(THEMES)
        spec = make(r, theme)
        if spec is None:
            continue
        page, steps, task, extra = spec
        k = r.randrange(len(steps))
        done = {eid for eid, _ in steps[:k]}
        gold = steps[k][0]
        header, footer = chrome(page, r, theme, r.choice(SITE_NAMES), nav)
        main = draw(page, done, extra, r, theme)
        promo, strips = filler(page, r)
        content = (
            f"<html><head><style>{css(theme, r.choice(FONTS))}</style></head><body>{header}{promo}<main>{main}</main>"
            f"{strips}{footer}</body></html>"
        )
        png, boxes, size = render.html(
            content, 1280, selectors={"els": "[data-el]"}, scale=1.0
        )
        el_boxes = _element_boxes(content, boxes["els"])
        visible = {
            eid: bx
            for eid, bx in el_boxes.items()
            if bx["visible"] and eid in page.labels
        }
        if gold not in visible:
            continue
        same = [
            e
            for e in visible
            if e != gold and page.groups.get(e) == page.groups.get(gold)
        ]
        other = [e for e in visible if e != gold and e not in same]
        pool = r.sample(same, min(len(same), r.choice([2, 3]))) + r.sample(
            other, min(len(other), 3)
        )
        distractors = []
        for e in pool:
            if len(distractors) == 3:
                break
            if not any(_overlap(visible[e], visible[x]) for x in [gold, *distractors]):
                distractors.append(e)
        if len(distractors) < 3:
            continue
        chosen = [gold, *distractors]
        r.shuffle(chosen)
        image = Image.open(io.BytesIO(png)).convert("RGB")
        letters = "ABCD"
        for i, eid in enumerate(chosen):
            bx = visible[eid]
            augment.box_marker(
                image,
                (
                    bx["x"] - 3,
                    bx["y"] - 3,
                    bx["x"] + bx["w"] + 3,
                    bx["y"] + bx["h"] + 3,
                ),
                BOX_COLORS[i],
                width=3,
                label=letters[i],
            )
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        history = [f"{page.labels[eid]} -> {op}" for eid, op in steps[:k]]
        state = f"Task: {task}\nPrevious actions:\n" + (
            "\n".join(history) if history else "(none)"
        )
        criteria = {letters[i]: f"the element in box {letters[i]}" for i in range(4)}
        answer = letters[chosen.index(gold)]
        return {
            "item_id": f"{index:05d}",
            "subtask": f"{site_key}:step{min(k, 5)}",
            "payloads": [(buffer.getvalue(), "png")],
            "instructions": "Four candidate elements are marked with lettered boxes on the screenshot. Which marked element should be acted on next to make progress on the task?",
            "criteria": criteria,
            "answer": answer,
            "state": state,
            "provenance": [
                {
                    "source": "generated",
                    "generator": f"d25.omni.proxy.web v{VERSION}",
                    "seed": seed,
                    "index": index,
                    "licence": "generated (no third-party content)",
                }
            ],
            "extra": {
                "attempt": attempt,
                "size": list(size),
                "gold_op": steps[k][1],
                "gold_label": page.labels[gold],
                "same_group_distractors": sum(
                    page.groups.get(e) == page.groups.get(gold) for e in distractors
                ),
            },
        }
    raise RuntimeError(f"no valid web item for {index}")


def _element_boxes(
    content: str, raw: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    import re

    order = re.findall(r'data-el="(e\d+)"', content)
    return {eid: bx for eid, bx in zip(order, raw)}


def _overlap(a: dict[str, Any], b: dict[str, Any], pad: float = 20.0) -> bool:
    return not (
        a["x"] + a["w"] + pad < b["x"]
        or b["x"] + b["w"] + pad < a["x"]
        or a["y"] + a["h"] + pad < b["y"]
        or b["y"] + b["h"] + pad < a["y"]
    )
