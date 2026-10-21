from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from gdn_prep_parity import conv_reference


def test_torch_only_mode_is_diagnostic_not_acceptance():
    conv_fn, source, validation = conv_reference(True)
    assert conv_fn is None
    assert source == "torch-reference"
    assert validation == "torch-diagnostic"
