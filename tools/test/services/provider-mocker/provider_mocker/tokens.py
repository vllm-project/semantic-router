"""Deterministic token estimates for fixture accounting."""

import math


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))
