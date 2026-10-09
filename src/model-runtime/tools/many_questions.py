"""A many-question System One request for the latency records (public, deterministic; as the released cards use).

One customer-support ticket (about 300 tokens) and up to 128 typed questions about it: 16 question
templates (Choice, Yes / No and Score) about each of 8 aspects of the ticket, in that order, so the
first N questions mix every type.
"""

from __future__ import annotations

from typing import Any

TICKET = (
    "Order #48213 was delivered on Tuesday, two days later than promised. The outer box was crushed and "
    "the blender inside has a cracked jug, so it cannot be used. I called on Wednesday and was told a "
    "replacement would ship within 24 hours, but nothing has arrived and the tracking page still shows "
    "'label created'. I have the receipt and photos of the damage. The courier left the parcel by the back "
    "door instead of handing it to me, even though the order said a signature was required. I was also "
    "charged twice for express shipping on my card statement. I would like the replacement sent today with "
    "express delivery at no cost, the duplicate shipping charge refunded, and a call from someone who can "
    "confirm all of this. I am hosting a family event on Saturday and planned to use the blender then. If "
    "this cannot be sorted out by Friday I will cancel the order and ask my bank to reverse the payment. "
    "This is the second time this year that an order from your store has arrived damaged, and the last "
    "time it took three weeks to get my money back. - Dana Whitfield, customer since 2019"
)

ASPECTS = (
    ("delivery", "the delivery"),
    ("replacement", "the replacement"),
    ("refund", "the refund"),
    ("damage", "the damaged item"),
    ("charges", "the shipping charges"),
    ("courier", "the courier"),
    ("callback", "the requested call"),
    ("deadline", "the Saturday event"),
)

TEMPLATES = (
    (
        "team",
        "choice",
        "Which team should handle {a}?",
        {
            "returns": "Returns, replacements and damaged items",
            "billing": "Payments, charges and refunds",
            "logistics": "Couriers, tracking and delivery",
            "escalations": "Repeat problems and complaints",
        },
    ),
    ("mentioned", "noul", "Does the ticket say anything specific about {a}?", None),
    (
        "urgency",
        "score",
        "How urgent is {a} for the customer?",
        ["Not urgent", "Within a week", "Within a day", "Immediately"],
    ),
    (
        "concern",
        "choice",
        "What is the customer's main concern about {a}?",
        {
            "cost": "Money lost or charged",
            "timing": "Delays and deadlines",
            "quality": "Broken or unusable goods",
            "service": "How the store communicated",
        },
    ),
    ("satisfied", "noul", "Is the customer satisfied with {a} so far?", None),
    (
        "clarity",
        "score",
        "How clearly does the ticket describe {a}?",
        ["Not at all", "Vaguely", "Clearly", "In full detail"],
    ),
    (
        "next",
        "choice",
        "What should the next step be for {a}?",
        {
            "call": "Call the customer back",
            "ship": "Ship a replacement",
            "refund": "Issue a refund",
            "evidence": "Ask for photos or the receipt",
            "close": "Close the ticket",
        },
    ),
    ("approval", "noul", "Would resolving {a} need a manager's approval?", None),
    (
        "escalation",
        "score",
        "How likely is a formal complaint over {a}?",
        ["Unlikely", "Possible", "Likely", "Almost certain"],
    ),
    (
        "channel",
        "choice",
        "Which channel suits a reply about {a}?",
        {"email": "Email", "phone": "Phone call", "chat": "Live chat"},
    ),
    ("dated", "noul", "Does the customer give a deadline related to {a}?", None),
    (
        "effort",
        "score",
        "How much work will it take to resolve {a}?",
        ["Trivial", "Small", "Moderate", "Large", "Very large"],
    ),
    (
        "fault",
        "choice",
        "Who is responsible for the problems with {a}?",
        {
            "store": "The store",
            "courier": "The courier",
            "maker": "The manufacturer",
            "customer": "The customer",
            "unclear": "Unclear from the ticket",
        },
    ),
    (
        "more_info",
        "noul",
        "Is more information needed from the customer about {a}?",
        None,
    ),
    (
        "tone",
        "score",
        "How polite is the customer's tone about {a}?",
        ["Hostile", "Curt", "Neutral", "Polite", "Very polite"],
    ),
    (
        "priority",
        "choice",
        "Which priority label fits {a}?",
        {"p1": "P1: today", "p2": "P2: this week", "p3": "P3: when possible"},
    ),
)


def questions(count: int = 128) -> dict[str, Any]:
    """The first ``count`` (at most 128) questions: aspect by aspect, all 16 templates each."""
    out: dict[str, Any] = {}
    for key, text in ASPECTS:
        for name, kind, instructions, criteria in TEMPLATES:
            question: dict[str, Any] = {
                "type": kind,
                "instructions": instructions.format(a=text),
            }
            if criteria is not None:
                question["criteria"] = criteria
            out[f"{key}_{name}"] = question
    if not 1 <= count <= len(out):
        raise ValueError(f"count must be within 1..{len(out)}")
    return dict(list(out.items())[:count])


def request(count: int = 128) -> dict[str, Any]:
    return {"state": TICKET, "questions": questions(count)}
