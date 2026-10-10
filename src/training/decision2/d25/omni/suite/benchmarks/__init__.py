"""One builder per board benchmark: ``build(ctx) -> {"rows", "variants", "notes"}``.

``rows`` are the board-matched rows (``suite/rows.py`` format); ``variants`` maps a name to alternate
rows kept so a construction can be switched later without a rebuild; ``notes`` records the
construction rule as built.
"""

from importlib import import_module

MODULES = {
    "CV-Bench": "cvbench",
    "BLINK": "blink",
    "RealWorldQA": "realworldqa",
    "CharXiv": "charxiv",
    "InfographicVQA": "infovqa",
    "Mind2Web": "mind2web",
    "Winoground": "winoground",
    "KIE (CORD+FUNSD)": "kie",
    "Moderation (Hateful Memes)": "hateful_memes",
    "R-Bench-M": "rbench",
    "MMMU-Pro vision": "mmmu_pro",
}


def builder(benchmark: str):
    return import_module(f"d25.omni.suite.benchmarks.{MODULES[benchmark]}")
