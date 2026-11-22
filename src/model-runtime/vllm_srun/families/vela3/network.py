"""The marker heads of vela3-fast's backbone and adapters (``vela3_calls`` runs them).

The backbone's head comes from ``model.safetensors``; an adapter's file holds
the readout tensors it trained apart from the backbone's (the others are the
backbone's own, as ``build_adapter`` shares them).
"""

from __future__ import annotations

from ...heads.marker import MarkerHead
from ...plugins.base import EngineModel
from ..vela2.members import read_tensors
from ..vela3_calls import HEAD_PREFIXES, MarkerNetworks, marker_head
from .package import BACKBONE_FILE, Vela3Package


class Networks(MarkerNetworks):
    """The marker heads of the backbone and of every adapter, and the forward over a batch."""

    def __init__(self, package: Vela3Package, engine_model: EngineModel):
        config = package.config
        hidden = int(config["hidden_size"])
        projection = int(config["decision"]["proj_dim"])
        device = engine_model.device
        base = read_tensors([package.root / BACKBONE_FILE], HEAD_PREFIXES)
        heads: dict[str | None, MarkerHead] = {
            None: marker_head(hidden, projection, base, spans=False).to(device)
        }
        for name, adapter in package.adapters.items():
            own = read_tensors([package.root / adapter.file], HEAD_PREFIXES)
            # An adapter's file holds the readout tensors it trained; the rest are the backbone's.
            heads[name] = marker_head(
                hidden, projection, {**base, **own}, spans="w_t.weight" in own
            ).to(device)
        super().__init__(heads, engine_model, int(config["pad_token_id"]))
