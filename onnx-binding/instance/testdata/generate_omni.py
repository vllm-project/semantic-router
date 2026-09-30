"""Generate tiny constant graphs exercising the complete Omni deployment contract.

Run with numpy and onnx installed. These graphs test loading/ownership/contracts;
they do not represent Vela weights or establish model quality.
"""

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "tools/models/vela_omni"))
from contract import (  # noqa: E402 - Resolve the repository-local artifact contract first.
    artifact_manifest,
    inventory,
    write_json,
)

root = HERE / "omni"
(root / "onnx").mkdir(parents=True, exist_ok=True)
(root / "components/text").mkdir(parents=True, exist_ok=True)
(root / "processors").mkdir(parents=True, exist_ok=True)
shutil.copyfile(
    HERE / "embedding/tokenizer.json", root / "components/text/tokenizer.json"
)
manifest = artifact_manifest("nano", "right", 0)
manifest["source"] = {"repo_id": "test/constant-graphs", "revision": "0" * 40}
for name, graph in manifest["graphs"].items():
    dimension = graph["output"]["shape"][1]
    values = np.zeros((1, dimension), dtype=np.float32)
    values[0, 0] = 1
    inputs = [
        helper.make_tensor_value_info(
            p["name"],
            TensorProto.INT64 if p["dtype"] == "int64" else TensorProto.FLOAT,
            p["shape"],
        )
        for p in graph["inputs"]
    ]
    output = helper.make_tensor_value_info(
        "embedding", TensorProto.FLOAT, [1, dimension]
    )
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node(
                    "Constant", [], ["embedding"], value=numpy_helper.from_array(values)
                )
            ],
            name,
            inputs,
            [output],
        ),
        opset_imports=[helper.make_opsetid("", 17)],
        ir_version=10,
    )
    onnx.checker.check_model(model)
    onnx.save(model, root / graph["file"])
common = {
    "window": "periodic_hann",
    "center": True,
    "pad_mode": "reflect",
    "power": 2,
    "floor": 1e-10,
}


def filters(bins, mels):
    result = np.zeros((bins, mels))
    result[np.arange(mels), np.arange(mels)] = 1
    return result.tolist()


audio = {
    "format_version": 1,
    "windows": "endpoint_cover_v1",
    "whisper": {
        **common,
        "sampling_rate": 16000,
        "n_fft": 400,
        "hop_length": 160,
        "n_samples": 480000,
        "n_frames": 3000,
        "mel_filters": filters(201, 80),
        "log": "log10",
        "range": 8,
        "affine": [0.25, 1.0],
    },
    "clap": {
        **common,
        "sampling_rate": 48000,
        "n_fft": 1024,
        "hop_length": 480,
        "n_samples": 480000,
        "n_frames": 1001,
        "mel_filters": filters(513, 64),
        "log": "db",
        "reference": 1,
        "min_value": 1e-10,
        "db_range": None,
        "padding": "repeatpad",
    },
}
# Compact synthetic filter matrices keep this offline fixture small.
(root / "processors/audio.json").write_text(
    json.dumps(audio, separators=(",", ":")) + "\n"
)
write_json(
    root / "reference_parity.json",
    {
        "format_version": 1,
        "passed": True,
        "fixture_only": True,
        "source": manifest["source"],
        "variant": "nano",
        "tests": [{"name": "constant graph fixture", "passed": True}],
    },
)
manifest["reference_parity"]["passed"] = True
manifest["files"] = inventory(root)
write_json(root / "vela_omni_manifest.json", manifest)
