"""The pico's one network: the UltraBERT backbone and its one marker head, word projections included.

``model.safetensors`` holds the encoder (``encoder.*``) and one readout: the
option projections and MLP that decisions, router questions and the one pass
read, and the word and label projections that span questions read. Every call
runs on it (``vela3_calls.MarkerNetworks``, network None).
"""

from __future__ import annotations

from ...plugins.base import EngineModel
from ..vela2.members import read_tensors
from ..vela3_calls import HEAD_PREFIXES, MarkerNetworks, marker_head
from .package import WEIGHTS_FILE, PicoPackage


class Network(MarkerNetworks):
    """The backbone's head, with spans, and the forward over a batch of one call."""

    def __init__(self, package: PicoPackage, engine_model: EngineModel):
        config = package.config
        tensors = read_tensors([package.root / WEIGHTS_FILE], HEAD_PREFIXES)
        head = marker_head(
            int(config["hidden_size"]),
            int(config["decision"]["proj_dim"]),
            tensors,
            spans=True,
        ).to(engine_model.device)
        super().__init__({None: head}, engine_model, int(config["pad_token_id"]))
