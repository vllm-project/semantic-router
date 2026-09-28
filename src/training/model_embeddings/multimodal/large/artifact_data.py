"""Minimal record type packaged with the exported inference model."""

from dataclasses import dataclass
from typing import Any


@dataclass
class PairItem:
    modality: str
    value: Any
