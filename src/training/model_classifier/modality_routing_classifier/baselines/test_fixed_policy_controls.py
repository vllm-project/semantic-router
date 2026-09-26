"""Regression tests for the modality routing fixed-policy controls.

These pin the lexical rule behaviour that the reported floor depends on. A
control used as a gate needs its own tests: if the rules drift, the number in
the decision record silently stops meaning what it said.
"""

import pytest

from fixed_policy_controls import LABELS, looks_like_prompt, predict_lexical, score


# --------------------------------------------------------------------------
# Rule ordering: a visual verb alone is DIFFUSION; with a text verb it is BOTH.
# Getting this backwards makes the control steal DIFFUSION examples.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "generate an image of a dragon, 8k, highly detailed",
        "cyberpunk corporate woman | | realistic shaded, poster by greg rutkowski",
        "a castle on a hill, oil painting, artstation, cinematic lighting",
        "forest at dusk, volumetric lighting, octane render, 4k",
    ],
)
def test_image_prompts_are_diffusion(text):
    assert predict_lexical(text) == "DIFFUSION"


@pytest.mark.parametrize(
    "text",
    [
        "Explain photosynthesis and show me a diagram",
        "What is the golden ratio? Draw a visual representation",
        "Describe a cross-section of the Earth and generate an image of it",
        "How does wound healing work? Include a diagram of each step",
        "Help me learn how transistors work - I need both text and pictures",
        "Give me a visual guide to French braid with explanations",
    ],
)
def test_text_plus_visual_is_both(text):
    assert predict_lexical(text) == "BOTH"


@pytest.mark.parametrize(
    "text",
    [
        "Generate a marketing material or advertisment that consists of both text "
        "and multiple images, where text and images can be interleaved in an "
        "arbitrary order.",
        "In this task, you are given a high-level goal 'How to Make Tea Eggs'",
    ],
)
def test_both_templates(text):
    assert predict_lexical(text) == "BOTH"


@pytest.mark.parametrize(
    "text",
    [
        "When was the 8088 processor released?",
        "Generate a list of methods to reduce food waste.",
        "Summarize the causes of World War I",
        "Explain how compilers work",
    ],
)
def test_plain_text_requests_are_ar(text):
    assert predict_lexical(text) == "AR"


def test_unsignalled_both_falls_through_to_ar():
    """Documents a known limit rather than asserting desired behaviour.

    These carry no request for images, so the lexical control cannot reach
    them. See README, finding 3. If a future rule change makes these BOTH,
    that rule is fitting the label rather than reading the request.
    """
    for text in [
        "How do I install a ceiling fan?",
        "Teach me julienne cutting for beginners",
        "How does a sewing machine thread path work?",
    ]:
        assert predict_lexical(text) == "AR"


# --------------------------------------------------------------------------
# Shape heuristic
# --------------------------------------------------------------------------


def test_prompt_shape_needs_comma_density():
    assert looks_like_prompt("knight, castle, sunset, oil painting, 8k")
    assert not looks_like_prompt("What are the causes of inflation?")
    assert not looks_like_prompt("Hello, world")


def test_question_mark_disqualifies_prompt_shape():
    assert not looks_like_prompt("knight, castle, sunset, oil painting, 8k?")


def test_pipe_is_prompt_shape():
    assert looks_like_prompt("knight | castle | sunset")


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def test_precision_is_none_when_class_never_predicted():
    rows = [
        {"text": "When was the 8088 released?", "label_name": "AR"},
        {"text": "What causes inflation?", "label_name": "DIFFUSION"},
    ]
    result = score(rows, lambda _t: "AR")
    assert result["per_class"]["DIFFUSION"]["predicted"] == 0
    assert result["per_class"]["DIFFUSION"]["precision"] is None
    assert result["per_class"]["DIFFUSION"]["recall"] == 0.0


def test_score_accounts_for_every_row():
    rows = [
        {"text": "When was the 8088 released?", "label_name": "AR"},
        {"text": "knight, castle, sunset, oil painting, 8k", "label_name": "DIFFUSION"},
        {"text": "Explain photosynthesis and show me a diagram", "label_name": "BOTH"},
    ]
    result = score(rows, predict_lexical)
    assert result["total"] == len(rows)
    assert sum(result["confusion"].values()) == len(rows)
    assert sum(result["per_class"][label]["support"] for label in LABELS) == len(rows)
