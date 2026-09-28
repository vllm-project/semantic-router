"""Family registry types and pinned source directories for the v2 arms.

A source module (``v2.data.m2.src_*``) exposes ``FAMILIES``: a tuple of
``FamilySpec``. ``build(dirs)`` receives every pinned source directory keyed by
the short names in ``SOURCE_DIRS`` and returns validated TRAIN rows plus a JSON
report; the orchestrator applies the whole-group cap and slices.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

ARMS = ("h1", "h3", "h5", "h6", "e11")

SOURCE_DIRS = {
    "hotpotqa": "hotpotqa@1908d6afbbead072334abe2965f91bd2709910ab",
    "twowiki": "twowiki@612bc5039a457880d9e7d84c3b0a4cf154b70e4f",
    "musique": "musique@22873a405dd809893b22ada0b499299fb612d2df",
    "tydiqa": "tydiqa@da78f23f9119363459acbaf46bf89426ff26c259",
    "miracl": "miracl@5be20db9509754dadad47689368639fcec739c00",
    "squad2": "squad2@3ffb306f725f7d2ce8394bc1873b24868140c412",
    "quac": "quac-v0.2",
    "jglue": "jglue@6f071c09316baae89c3d083a90985b4b1cb9968c",
    "klue": "klue@349481ec73fff722f88e0453ca05c77a447d967c",
    "cmrc2018": "cmrc2018@137f2c45a24275fb68f6961c4d357f46288886aa",
    "drcd": "drcd@b944790de5af02c5fbb7cd9cb1473d27d169eebf",
    "germanquad": "germanquad@a2f3a59f0be843fc305d0417d7292ef0b1a66884",
    "piaf": "piaf@bda8c063bc7297180796cd835d1974c0bc71c521",
    "sqac": "sqac@f9928e8819596a601b8887cc5f8598b15d589a82",
    "mtop": "mtop",
    "multiwoz": "multiwoz@fe0c8e65cfcd8462bd33c86e35f21addc84ca82b",
    "taskmaster": "taskmaster@d92cb6af3005f1dc09c39e75e7daf4a04905e00b",
    "dbpedia14": "dbpedia14@9abd46cf7fc8b4c64290f26993c540b92aa145ac",
    "winogrande": "winogrande@01e74176c63542e6b0bcb004dcdea22d94fb67b5",
    "csqa": "csqa@94630fe30dad47192a8546eb75f094926d47e155",
    "arc": "arc@210d026faf9955653af8916fad021475a3f00453",
    "obqa": "obqa@388097ea7776314e93a529163e0fea805b8a6454",
    "scitail": "scitail@0cc4353235b289165dfde1c7c5d1be983f99ce44",
    "aquarat": "aquarat@33301c6a050c96af81f63cad5562cb5363e88971",
    "gsm8k": "gsm8k@740312add88f781978c0658806c59bc2815b9866",
    "quartz": "quartz@28c1dbb56caf81799296cb17892fa73402e23464",
    "ropes": "ropes@d59f1e2ee2b423d7c6ba71edd47fceb4158b07dd",
    "mlqepe": "mlqepe@2a670a1140416cf80507b5a829659383c878feb8",
    "onestop": "onestop@37f8db3945cd2f3cc0caafe45674147b224349be",
    "argq": "argq30k@590726b3765b1b90c5e53a17e3b1f77d92d3aa8a",
}

Builder = Callable[[Mapping[str, Path]], tuple[list[dict[str, Any]], dict[str, Any]]]


@dataclasses.dataclass(frozen=True)
class FamilySpec:
    arm: str
    family: str
    source: str
    build: Builder
    cap_rows: int
    seed: str

    def __post_init__(self) -> None:
        if self.arm not in ARMS:
            raise ValueError(f"{self.family}: unknown arm {self.arm}")
        if self.cap_rows < 1:
            raise ValueError(f"{self.family}: cap_rows must be positive")


def resolve(root: Path) -> dict[str, Path]:
    return {name: root / sub for name, sub in SOURCE_DIRS.items()}
