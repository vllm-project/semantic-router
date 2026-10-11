"""The System One request shown in the card's Quickstart (the Decision 2.0 card example) and API probes."""

from __future__ import annotations

QUICKSTART = {
    "state": "The order arrived damaged yesterday. The customer has a receipt and asks for a replacement today.",
    "questions": {
        "route": {
            "type": "choice",
            "instructions": "Which team should handle this request?",
            "criteria": {
                "returns": "Refunds, replacements and damaged deliveries",
                "billing": "Payments, invoices and charges",
                "technical": "Product setup and faults",
            },
        },
        "receipt": {
            "type": "noul",
            "instructions": "Does the customer have a receipt?",
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgent is this request?",
            "criteria": ["Routine", "Soon", "Today"],
        },
    },
}

# The image example of an image-capable card: the repository's assets/example-receipt.png (an invented store and
# receipt, our own render) with the request the image smoke test runs (d25/omni/runtime/smoke.py).
EXAMPLE_IMAGE = "assets/example-receipt.png"
QUICKSTART_IMAGE = {
    "state": "The customer says the blender arrived cracked and attached the receipt.",
    "questions": {
        "route": QUICKSTART["questions"]["route"],
        "on_receipt": {
            "type": "noul",
            "instructions": "Does the receipt list the blender?",
        },
        "payment": {
            "type": "choice",
            "instructions": "How was the order paid?",
            "criteria": {"card": None, "cash": None, "gift card": None},
        },
    },
}

# The video example of a video-capable card: the repository's assets/example-video.mp4 (our own render: a blue
# square moves from left to right, stops and turns green; 4 s, 640 x 360, 8 frames per second, MPEG-4) with the
# request the video checks run (d25/vega/release/video_check.py).
EXAMPLE_VIDEO = "assets/example-video.mp4"
QUICKSTART_VIDEO = {
    "state": "A short clip from a test camera.",
    "questions": {
        "direction": {
            "type": "choice",
            "instructions": "Which way does the square move?",
            "criteria": {
                "right": "From left to right",
                "left": "From right to left",
                "still": "It does not move",
            },
        },
        "color_change": {
            "type": "noul",
            "instructions": "Does the square change color?",
        },
    },
}

# Edge cases of the product API: null descriptions, JSON state, noul criteria, a malformed question.
PROBES = {
    "state": {
        "ticket": 4182,
        "text": "Card payment failed twice at checkout.",
        "plan": "pro",
    },
    "questions": {
        "intent": {
            "type": "choice",
            "instructions": "What does the user want?",
            "criteria": {
                "fix_payment": None,
                "cancel": None,
                "upgrade": "Move to a higher plan",
            },
        },
        "repeat": {
            "type": "noul",
            "instructions": "Did the problem happen more than once?",
            "criteria": {
                "true": "It happened at least twice",
                "false": "It happened once",
            },
        },
        "severity": {
            "type": "score",
            "instructions": "How severe is it?",
            "criteria": [
                "Cosmetic",
                "Annoying",
                "Blocking",
                {"level": "Outage", "pages": True},
            ],
        },
        "broken": {
            "type": "score",
            "instructions": "Rate it.",
            "criteria": ["Only one level"],
        },
    },
}
