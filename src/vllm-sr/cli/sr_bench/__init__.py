"""sr-bench 1.0: shared, durable evaluation for single and routed models."""

VERSION = "sr-bench-1.0"
# nano and nano-holdout are the EXPERIMENTAL sr-bench-nano scope (see nano.py).
PROFILE_SPLITS = {
    "smoke": "dev",
    "quick": "dev",
    "standard": "holdout",
    "nano": "dev",
    "nano-holdout": "holdout",
}
