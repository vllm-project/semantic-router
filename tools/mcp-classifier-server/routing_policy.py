"""Shared example policy for MCP classifier routing recommendations."""

DEFAULT_MODEL = "openai/gpt-oss-20b"
COMPLEX_QUERY_MIN_WORDS = 20
SIMPLE_MATH_MAX_WORDS = 15
HIGH_CONFIDENCE = 0.9
LOW_CONFIDENCE = 0.6
COMPLEXITY_MARKERS = (
    "why",
    "how",
    "explain",
    "analyze",
    "compare",
    "evaluate",
    "describe",
)


def decide_routing(
    text: str, category_name: str, confidence: float
) -> tuple[str, bool]:
    """Choose the example backend and whether reasoning should be enabled."""
    word_count = len(text.split())
    has_complexity_marker = any(marker in text.lower() for marker in COMPLEXITY_MARKERS)

    if word_count > COMPLEX_QUERY_MIN_WORDS and has_complexity_marker:
        return DEFAULT_MODEL, True
    if category_name == "math" and word_count < SIMPLE_MATH_MAX_WORDS:
        return DEFAULT_MODEL, False
    if confidence > HIGH_CONFIDENCE:
        return DEFAULT_MODEL, False
    if confidence < LOW_CONFIDENCE:
        return DEFAULT_MODEL, True
    return DEFAULT_MODEL, True
