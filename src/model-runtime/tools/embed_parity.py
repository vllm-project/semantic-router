"""Parity records for the embedding and rerank models (embed workstream).

    python3 tools/embed_parity.py omni --bundle DIR --output OUT.json [--threads N]

``omni`` serves a prepared Vela Omni bundle that kept its goldens
(``VELA_OMNI_KEEP_GOLDEN=1``) through ``MultimodalEmbeddingFamily`` and the
``onnxruntime`` engine on the CPU, and compares every stage with the bundle's
official-reference goldens (``golden/index.json``): token IDs, image pixels,
16 kHz and 48 kHz audio, Whisper and CLAP features, CLAP window embeddings and
their aggregate, and every end-to-end embedding (text, padded text, Mini's
instruction roles, PNG and JPEG images, mono and stereo audio of one to three
CLAP windows). Media go through the request path as the router sends them:
images as their file bytes, audio as float32 WAV of the golden PCM. Passes when
every embedding has cosine >= 0.99999 and |delta| <= 1e-4 (design section 17).
"""

from __future__ import annotations

import argparse
import base64
import json
import struct
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vllm_sr_runtime.accel.cpu import CPUAccelerator  # noqa: E402
from vllm_sr_runtime.engines.onnxruntime.engine import OnnxRuntimeEngine  # noqa: E402
from vllm_sr_runtime.families.multimodal_embedding import audio  # noqa: E402
from vllm_sr_runtime.families.multimodal_embedding.family import (  # noqa: E402
    MultimodalEmbeddingFamily,
)
from vllm_sr_runtime.plugins.base import (  # noqa: E402
    DeviceInfo,
    EncoderBatch,
    EngineOptions,
    PackageRef,
    SurfaceRequest,
)

MIN_COSINE = 0.99999
MAX_ABS = 1e-4
CPU = DeviceInfo(accelerator="cpu", index=None, name="cpu")


def golden_array(root: Path, ref: dict[str, Any]) -> np.ndarray:
    return np.fromfile(root / "golden" / ref["file"], dtype="<f4").reshape(ref["shape"])


def compare(actual: Any, expected: np.ndarray) -> dict[str, float]:
    a = np.asarray(actual, dtype=np.float64).reshape(-1)
    b = np.asarray(expected, dtype=np.float64).reshape(-1)
    if a.shape != b.shape:
        return {"shape_mismatch": 1.0, "max_abs": float("inf"), "cosine": 0.0}
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    cosine = float(a @ b / denominator) if denominator else float(np.array_equal(a, b))
    return {
        "max_abs": float(np.abs(a - b).max()) if a.size else 0.0,
        "cosine": cosine,
        "equal": float(np.mean(a == b)),
    }


def float_wav(pcm: np.ndarray, rate: int) -> bytes:
    """Channels-first float32 PCM as an IEEE float WAV (what the router forwards)."""
    channels = pcm.shape[0]
    data = np.ascontiguousarray(pcm.T, dtype="<f4").tobytes()
    fmt = struct.pack(
        "<HHIIHH", 3, channels, rate, rate * 4 * channels, 4 * channels, 32
    )
    body = (
        b"WAVE"
        + b"fmt "
        + struct.pack("<I", len(fmt))
        + fmt
        + b"data"
        + struct.pack("<I", len(data))
        + data
    )
    return b"RIFF" + struct.pack("<I", len(body)) + body


class OmniParity:
    def __init__(self, bundle: Path, threads: int | None):
        family = MultimodalEmbeddingFamily()
        self.package = family.verify(PackageRef(bundle))
        spec = family.describe(self.package)
        engine_model = OnnxRuntimeEngine().load(
            spec, CPUAccelerator(), CPU, EngineOptions(threads=threads)
        )
        self.model = family.load(self.package, spec, engine_model)
        self.root = bundle
        self.index = json.loads(
            (bundle / "golden/index.json").read_text(encoding="utf-8")
        )
        self.cases: list[dict[str, Any]] = []

    def embed(
        self, part: Any, input_type: str | None = None
    ) -> tuple[list[float], float]:
        body: dict[str, Any] = {"input": [part]}
        if input_type:
            body["input_type"] = input_type
        request = SurfaceRequest(
            "embeddings", body, None, "exact", False, time.monotonic()
        )
        started = time.perf_counter()
        plan = self.model.plan_surface("embeddings", request)
        response = self.model.finish_surface(plan, self.model.run(plan.items))
        elapsed = (time.perf_counter() - started) * 1000
        entry = response["data"][0]
        if "embedding" not in entry:
            raise RuntimeError(f"{entry.get('error')} for a golden input")
        return entry["embedding"], elapsed

    def record(self, name: str, stages: dict[str, dict[str, float]], ms: float) -> None:
        final = stages["embedding"]
        passed = final["cosine"] >= MIN_COSINE and final["max_abs"] <= MAX_ABS
        self.cases.append(
            {"name": name, "passed": passed, "ms": round(ms, 3), **stages}
        )

    def texts(self) -> None:
        for index, case in enumerate(self.index["texts"]):
            role = case.get("role")
            ids_expected = [
                token
                for token, mask in zip(
                    case["input_ids"][0], case["attention_mask"][0], strict=True
                )
                if mask
            ]
            budget = self.model.info.limits["max_input_tokens"]
            ids, _ = self.model.text.encode(case["text"], budget, role)
            vector, ms = self.embed(case["text"], role)
            stages = {
                "token_ids": {"equal": float(ids == ids_expected), "tokens": len(ids)},
                "embedding": compare(
                    vector, golden_array(self.root, case["embedding"])
                ),
            }
            name = (
                f"text/{role}"
                if role
                else (
                    "text/padding"
                    if 0 in case["attention_mask"][0]
                    else f"text/{index}"
                )
            )
            self.record(name, stages, ms)

    def images(self) -> None:
        for index, case in enumerate(self.index["images"]):
            data = (self.root / "golden" / case["file"]).read_bytes()
            pixels = self.model.image.pixels(data)
            media = "image/jpeg" if case["file"].endswith(".jpg") else "image/png"
            url = f"data:{media};base64,{base64.b64encode(data).decode()}"
            vector, ms = self.embed({"type": "image_url", "image_url": {"url": url}})
            stages = {
                "pixels": compare(pixels, golden_array(self.root, case["pixels"])),
                "embedding": compare(
                    vector, golden_array(self.root, case["embedding"])
                ),
            }
            self.record(f"image/{index}", stages, ms)

    def clap_embeddings(self, features: np.ndarray) -> np.ndarray:
        import torch

        vectors = []
        for window in features:
            batch = EncoderBatch(
                input_ids=torch.zeros(1, 1, dtype=torch.long),
                attention_mask=torch.ones(1, 1, dtype=torch.long),
                graph="clap",
                graph_inputs={"input_features": torch.from_numpy(window)},
                outputs=("embedding",),
            )
            vectors.append(
                self.model.engine_model.encode(batch).outputs["embedding"].numpy()[0]
            )
        return np.stack(vectors)

    def audio(self) -> None:
        processor = self.model.audio
        for index, case in enumerate(self.index["audio"]):
            pcm = golden_array(self.root, case["pcm"])
            pcm = pcm[None] if pcm.ndim == 1 else pcm
            rate = case["sampling_rate"]
            wav = float_wav(pcm, rate)
            decoded = audio.decode_wav(wav)
            speech = audio.native_rate(decoded, audio.WHISPER_RATE)
            wide = audio.native_rate(decoded, audio.CLAP_RATE)
            features = processor.features(wav, "wav")
            clap = self.clap_embeddings(features.clap)
            vector, ms = self.embed(
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": base64.b64encode(wav).decode(),
                        "format": "wav",
                    },
                }
            )
            stages = {
                "decoded_pcm": compare(decoded.samples, pcm),
                "audio16": compare(speech, golden_array(self.root, case["audio16"])),
                "audio48": compare(wide, golden_array(self.root, case["audio48"])),
                "whisper_features": compare(
                    features.whisper, golden_array(self.root, case["whisper_features"])
                ),
                "clap_features": compare(
                    features.clap, golden_array(self.root, case["clap_features"])
                ),
                "clap_embeddings": compare(
                    clap, golden_array(self.root, case["clap_embeddings"])
                ),
                "embedding": compare(
                    vector, golden_array(self.root, case["embedding"])
                ),
            }
            self.record(
                f"audio/{index}@{rate}Hz/{pcm.shape[0]}ch/{case['seconds']}s",
                stages,
                ms,
            )

    def run(self) -> dict[str, Any]:
        self.texts()
        self.images()
        self.audio()
        return {
            "model": self.package.model_name,
            "source": self.package.details["bundle"].source,
            "bundle_sha256": self.package.model_sha256,
            "variant": self.index["variant"],
            "engine": "onnxruntime",
            "provider": "CPUExecutionProvider",
            "thresholds": {"min_cosine": MIN_COSINE, "max_abs": MAX_ABS},
            "passed": all(case["passed"] for case in self.cases),
            "cases": self.cases,
        }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    omni = commands.add_parser(
        "omni", help="Vela Omni bundle vs its official-reference goldens"
    )
    omni.add_argument("--bundle", type=Path, required=True)
    omni.add_argument("--output", type=Path, required=True)
    omni.add_argument("--threads", type=int, default=None)
    args = parser.parse_args()
    result = OmniParity(args.bundle.resolve(), args.threads).run()
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    failed = [case["name"] for case in result["cases"] if not case["passed"]]
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "cases": len(result["cases"]),
                "failed": failed,
            }
        )
    )
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
