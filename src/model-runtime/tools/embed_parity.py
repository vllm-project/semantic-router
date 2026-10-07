"""Parity records for the embedding and rerank models (embed workstream).

    python3 tools/embed_parity.py omni --bundle DIR [--snapshot DIR [--device rocm:N]] --output OUT.json [--threads N]
    python3 tools/embed_parity.py encoder --package DIR --output OUT.json [--threads N] [--device rocm:N]
    python3 tools/embed_parity.py reduced --package DIR --kind bfloat16 --output OUT.json [--device rocm:N]

``encoder`` serves an embedder or reranker package (Vela Embedding, Vela
Reranker, Qwen3-Embedding) through ``TaskHeadsFamily`` on the native engine
(every exit) and, when the package ships exit graphs, on the onnxruntime
engine (every graph), and compares both with a Transformers FP32 reference
fed the same token IDs (``tools/embed_corpus.py``): every exit's embedding
(raw or final-normed intermediate exits per the package contract, Matryoshka
views of the last one) or every pair scorer's logits and rerank order. With
``--device`` the native engine runs on that GPU against the CPU reference
under the GPU bar (cosine >= 0.9995); the onnxruntime side is a CPU record.

``omni`` serves Vela Omni through ``MultimodalEmbeddingFamily``: with
``--snapshot`` the published repository's files on the native engine (the
CPU, or ``--device``), else the prepared bundle on the ``onnxruntime``
engine on the CPU. It compares every stage with the official-reference
goldens of a bundle that kept them (``--bundle``, exported with
``VELA_OMNI_KEEP_GOLDEN=1``; ``golden/index.json``): token IDs, image pixels,
16 kHz and 48 kHz audio, Whisper and CLAP features, CLAP window embeddings and
their aggregate, and every end-to-end embedding (text, padded text, Mini's
instruction roles, PNG and JPEG images, mono and stereo audio of one to three
CLAP windows). Media go through the request path as the router sends them:
images as their file bytes, audio as float32 WAV of the golden PCM. Passes when
every embedding has cosine >= 0.99999 and |delta| <= 1e-4 on the CPU, and
cosine >= 0.9995 on a GPU (design section 17).
"""

from __future__ import annotations

import argparse
import base64
import json
import statistics
import struct
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from embed_corpus import RERANK, texts  # noqa: E402
from vllm_srun.accel.cpu import CPUAccelerator  # noqa: E402
from vllm_srun.engines.native.engine import NativeEngine  # noqa: E402
from vllm_srun.engines.onnxruntime.engine import OnnxRuntimeEngine  # noqa: E402
from vllm_srun.families.multimodal_embedding import audio  # noqa: E402
from vllm_srun.families.multimodal_embedding import bundle as bundles  # noqa: E402
from vllm_srun.families.multimodal_embedding.family import (  # noqa: E402
    MultimodalEmbeddingFamily,
)
from vllm_srun.families.multimodal_embedding.readout import unit  # noqa: E402
from vllm_srun.families.task_heads.family import TaskHeadsFamily  # noqa: E402
from vllm_srun.heads.embedding import matryoshka, pool  # noqa: E402
from vllm_srun.heads.relevance import RelevanceLayout  # noqa: E402
from vllm_srun.plugins.base import (  # noqa: E402
    DeviceInfo,
    EncoderBatch,
    EngineOptions,
    PackageRef,
    RegistryOptions,
    SurfaceRequest,
)

MIN_COSINE = 0.99999
MAX_ABS = 1e-4
# Design section 17: the GPU bar for embeddings; logits are reported, not gated, on GPUs.
GPU_MIN_COSINE = 0.9995
REDUCED_MIN_COSINE = 0.999
REDUCED_MIN_AGREEMENT = 0.99
TIE = 1e-3
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
    def __init__(
        self,
        bundle: Path,
        threads: int | None,
        snapshot: Path | None = None,
        device: str = "cpu",
    ):
        family = MultimodalEmbeddingFamily()
        self.root = bundle
        self.source = bundles.load(bundle).source
        self.device = device
        if snapshot is None:
            self.package = family.verify(PackageRef(bundle))
            spec = family.describe(self.package)
            engine_model = OnnxRuntimeEngine().load(
                spec, CPUAccelerator(), CPU, EngineOptions(threads=threads)
            )
            self.engine = "onnxruntime"
        else:
            self.package = family.verify(PackageRef(snapshot, *self.source))
            spec = family.describe(self.package)
            info = device_info(device)
            if device == "cpu":
                self.accelerator: Any = CPUAccelerator()
            else:
                from vllm_srun.accel.rocm import ROCmAccelerator

                self.accelerator = ROCmAccelerator()
            engine_model = self.accelerator.execute(
                info,
                lambda: NativeEngine().load(
                    spec, self.accelerator, info, EngineOptions(threads=threads)
                ),
            )
            self.engine = "native"
        self.model = family.load(self.package, spec, engine_model)
        self.index = json.loads(
            (bundle / "golden/index.json").read_text(encoding="utf-8")
        )
        self.cases: list[dict[str, Any]] = []
        self.min_cosine = MIN_COSINE if device == "cpu" else GPU_MIN_COSINE

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
        results = (
            self.model.run(plan.items)
            if self.engine == "onnxruntime"
            else self.accelerator.execute(
                device_info(self.device), lambda: self.model.run(plan.items)
            )
        )
        response = self.model.finish_surface(plan, results)
        elapsed = (time.perf_counter() - started) * 1000
        entry = response["data"][0]
        if "embedding" not in entry:
            raise RuntimeError(f"{entry.get('error')} for a golden input")
        return entry["embedding"], elapsed

    def record(self, name: str, stages: dict[str, dict[str, float]], ms: float) -> None:
        final = stages["embedding"]
        passed = final["cosine"] >= self.min_cosine and (
            self.device != "cpu" or final["max_abs"] <= MAX_ABS
        )
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
        """Each CLAP window's normalized embedding ``[windows, 512]`` (the goldens' stage)."""
        import torch

        if self.engine == "native":
            windows = features.reshape(-1, 1, *features.shape[-2:])
            pooled = self.model._tower("clap", input_features=windows)
            with torch.inference_mode():
                vectors = unit(self.model.readout.clap_projection(pooled).float())
            assert vectors is not None
            return vectors.cpu().numpy()
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
        import torch

        return {
            "model": self.package.model_name,
            "source": self.source,
            "model_sha256": self.package.model_sha256,
            "variant": self.index["variant"],
            "engine": self.engine,
            "device": self.device,
            "torch": torch.__version__,
            "provider": (
                "CPUExecutionProvider" if self.engine == "onnxruntime" else None
            ),
            "thresholds": {
                "min_cosine": self.min_cosine,
                "max_abs": MAX_ABS if self.device == "cpu" else None,
            },
            "passed": all(case["passed"] for case in self.cases),
            "cases": self.cases,
        }


def serve(
    model: Any, surface: str, body: dict[str, Any], reduced: bool = False
) -> tuple[Any, dict[str, Any]]:
    """One request through the exact path, or through the reduced copy (``max_speed``)."""
    request = SurfaceRequest(surface, body, None, "exact", False, time.monotonic())
    plan = model.plan_surface(surface, request)
    run = model.run_approximate if reduced else model.run
    return plan, model.finish_surface(plan, run(plan.items))


def device_info(name: str) -> DeviceInfo:
    """The device as placement reports it (architecture included, which selects fused kernels)."""
    if name == "cpu":
        return CPU
    from vllm_srun.accel.rocm import ROCmAccelerator

    return ROCmAccelerator().devices()[int(name.partition(":")[2] or 0)]


def load_task_model(
    package: Path,
    engine: Any,
    device: DeviceInfo,
    threads: int | None,
    options: dict[str, Any] | None = None,
    reduced: str | None = None,
) -> Any:
    """A ``task_heads`` model on one engine; ``reduced`` consents to that copy kind."""
    from vllm_srun.accel.rocm import ROCmAccelerator

    family = TaskHeadsFamily(RegistryOptions(model_options=options or {}))
    verified = family.verify(PackageRef(package))
    spec = family.describe(verified)
    if reduced is not None:
        field = "reduced_cpu" if device.accelerator == "cpu" else "reduced_gpu"
        spec = replace(spec, dtype=replace(spec.dtype, **{field: reduced}))
    accelerator = CPUAccelerator() if device.accelerator == "cpu" else ROCmAccelerator()
    engine_options = EngineOptions(
        threads=threads, reduced_precision=reduced is not None
    )
    return family.load(
        verified, spec, engine.load(spec, accelerator, device, engine_options)
    )


class EncoderParity:
    """One embedder or reranker on both engines against the Transformers reference."""

    def __init__(self, package: Path, threads: int | None, device: str = "cpu"):
        import torch

        torch.set_num_threads(threads or torch.get_num_threads())
        self.package = package
        self.threads = threads
        self.states: dict[tuple[int, ...], tuple[Any, ...]] = {}
        self.device = device_info(device)
        self.native = load_task_model(package, NativeEngine(), self.device, threads)
        layout = self.native.planners[next(iter(self.native.planners))].layout
        if isinstance(layout, RelevanceLayout):
            option = {
                "pair_scorers": [
                    {"layer": layer, "dimension": dim} for layer, dim in layout.graphs
                ]
            }
        else:
            option = {"layers": sorted(layout.graphs)}
        on_cpu = self.device.accelerator == "cpu"
        self.graph = (
            load_task_model(package, OnnxRuntimeEngine(), CPU, threads, option)
            if layout.graphs and on_cpu
            else None
        )
        self.layout = layout
        self.reference = self.reference_model()
        self.cases: list[dict[str, Any]] = []

    def reference_model(self) -> Any:
        import torch
        from transformers import AutoModel

        model = AutoModel.from_pretrained(
            str(self.package), dtype=torch.float32, attn_implementation="sdpa"
        )
        return model.eval()

    def row_states(self, row: tuple[int, ...]) -> tuple[Any, ...]:
        """One row's reference states by layer: raw exits, the last one final-normed."""
        import torch

        states = self.states.get(row)
        if states is None:
            with torch.inference_mode():
                out = self.reference(
                    input_ids=torch.tensor([row]), output_hidden_states=True
                )
            last = self.reference.config.num_hidden_layers
            states = (
                *(h[0] for h in out.hidden_states[:last]),
                out.last_hidden_state[0],
            )
            self.states[row] = states
        return states

    def hidden(
        self, rows: list[tuple[int, ...]], layer: int, normalize_exits: bool
    ) -> tuple[Any, Any]:
        """The reference's hidden states at ``layer`` as right-padded rows, and the mask.

        Each row runs alone once (no padding in the reference); every exit reads it.
        """
        import torch

        last = self.reference.config.num_hidden_layers
        width = max(len(row) for row in rows)
        mask = torch.zeros(len(rows), width, dtype=torch.long)
        hidden = None
        with torch.inference_mode():
            for index, row in enumerate(rows):
                value = self.row_states(row)[layer]
                if normalize_exits and layer != last:
                    value = self.reference.final_norm(value)
                if hidden is None:
                    hidden = value.new_zeros(len(rows), width, value.shape[-1])
                hidden[index, : len(row)] = value
                mask[index, : len(row)] = 1
        return hidden, mask

    def record(
        self,
        name: str,
        pairs: dict[str, dict[str, float]],
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Gate every comparison: CPU pairs on cosine and |delta|, GPU pairs on cosine."""
        gpu = self.device.accelerator != "cpu"
        passed = True
        for key, stats in pairs.items():
            on_gpu = gpu and key.startswith("native")
            if "min_cosine" in stats:
                passed &= stats["min_cosine"] >= (
                    GPU_MIN_COSINE if on_gpu else MIN_COSINE
                )
            if not on_gpu:
                passed &= stats.get("max_abs", 0.0) <= MAX_ABS
        self.cases.append(
            {"name": name, "passed": bool(passed), **pairs, **(extra or {})}
        )

    @staticmethod
    def rows_compare(a: Any, b: Any) -> dict[str, float]:
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        cosines = (a * b).sum(-1) / (
            np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
        )
        return {
            "min_cosine": float(cosines.min()),
            "max_abs": float(np.abs(a - b).max()),
        }

    def embeddings(self) -> None:
        corpus = texts()
        planner = self.native.planners["embeddings"]
        dims = sorted(
            {planner.info.dimensions[0], *planner.info.dimensions[-2:]}, reverse=True
        )
        for layer in planner.info.layers:
            for dimension in dims if layer == planner.info.layers[-1] else dims[:1]:
                body = {"input": corpus, "layer": layer, "dimensions": dimension}
                plan, native = serve(self.native, "embeddings", body)
                rows = [item.ids for item in plan.items]
                hidden, mask = self.hidden(rows, layer, self.layout.normalize_exits)
                pooled = pool(hidden, mask, self.layout.pooling)
                reference = matryoshka(pooled, dimension, self.layout.normalize)
                vectors = [entry["embedding"] for entry in native["data"]]
                pairs = {
                    "native_vs_reference": self.rows_compare(vectors, reference.numpy())
                }
                if (
                    self.graph is not None
                    and layer in self.graph.planners["embeddings"].info.layers
                ):
                    _, graph = serve(self.graph, "embeddings", body)
                    graph_vectors = [entry["embedding"] for entry in graph["data"]]
                    pairs["onnxruntime_vs_reference"] = self.rows_compare(
                        graph_vectors, reference.numpy()
                    )
                    pairs["native_vs_onnxruntime"] = self.rows_compare(
                        vectors, graph_vectors
                    )
                tokens = [len(row) for row in rows]
                self.record(
                    f"embeddings/layer{layer}/dim{dimension}",
                    pairs,
                    {"inputs": len(rows), "max_tokens": max(tokens)},
                )

    def rerank(self) -> None:
        import torch

        planner = self.native.planners["rerank"]
        exits = planner.info.exits
        scorers = self.layout.scorers(exits)
        graph_exits = self.graph.planners["rerank"].info.exits if self.graph else ()
        runs = {
            exit: {"native": [], "reference": [], "onnxruntime": [], "orders": []}
            for exit in exits
        }
        for query, documents in RERANK:
            for exit in exits:
                body = {
                    "query": query,
                    "documents": documents,
                    "layer": exit[0],
                    "dimensions": exit[1],
                }
                run = runs[exit]
                plan, native = serve(self.native, "rerank", body)
                by_index = {r["index"]: r["logit"] for r in native["results"]}
                run["native"] += [by_index[i] for i in range(len(documents))]
                rows = [item.ids for item in plan.items]
                hidden, _ = self.hidden(rows, exit[0], True)
                with torch.inference_mode():
                    logits = scorers[exit](hidden[:, 0, : exit[1]].float())[:, 0]
                run["reference"] += logits.tolist()
                reference_order = sorted(
                    range(len(documents)), key=lambda i: (-float(logits[i]), i)
                )
                served_order = [r["index"] for r in native["results"]]
                run["orders"].append(served_order == reference_order)
                if exit in graph_exits:
                    _, graph = serve(self.graph, "rerank", body)
                    graph_index = {r["index"]: r["logit"] for r in graph["results"]}
                    run["onnxruntime"] += [
                        graph_index[i] for i in range(len(documents))
                    ]
        for exit, run in runs.items():
            compared = [("native", "reference")]
            if run["onnxruntime"]:
                compared += [("onnxruntime", "reference"), ("native", "onnxruntime")]
            pairs = {
                f"{a}_vs_{b}": {
                    "max_abs": float(np.abs(np.subtract(run[a], run[b])).max())
                }
                for a, b in compared
            }
            self.record(
                f"rerank/layer{exit[0]}/dim{exit[1]}",
                pairs,
                {"identical_order": all(run["orders"]), "pairs": len(run["native"])},
            )

    def run(self) -> dict[str, Any]:
        if "embeddings" in self.native.planners:
            self.embeddings()
        else:
            self.rerank()
        return {
            "model": self.native.info.id,
            "model_sha256": self.native.info.model_sha256,
            "engines": ["native", *(["onnxruntime"] if self.graph is not None else [])],
            "reference": "transformers fp32 (sdpa) on the CPU, same token IDs",
            "device": self.device.label,
            "thresholds": {"min_cosine": MIN_COSINE, "max_abs": MAX_ABS},
            "passed": all(
                case["passed"] and case.get("identical_order", True)
                for case in self.cases
            ),
            "cases": self.cases,
        }


class ReducedParity:
    """A model's reduced copy (``max_speed``) against its exact path: agreement and latency.

    Embeddings need cosine >= 0.999 per vector, reranking 99 % pairwise order
    agreement on document pairs whose exact logits differ by more than ``TIE``
    (design section 5.4's floor), on the corpus of the encoder mode.
    """

    def __init__(self, package: Path, kind: str, device: str, threads: int | None):
        import torch

        torch.set_num_threads(threads or torch.get_num_threads())
        self.kind = kind
        self.device = device_info(device)
        self.model = load_task_model(
            package, NativeEngine(), self.device, threads, reduced=kind
        )
        self.cases: list[dict[str, Any]] = []

    def embeddings(self) -> list[dict[str, Any]]:
        planner = self.model.planners["embeddings"]
        dims = sorted(
            {planner.info.dimensions[0], *planner.info.dimensions[-2:]}, reverse=True
        )
        for layer in planner.info.layers:
            for dimension in dims if layer == planner.info.layers[-1] else dims[:1]:
                body = {"input": texts(), "layer": layer, "dimensions": dimension}
                vectors = [
                    [
                        entry["embedding"]
                        for entry in serve(self.model, "embeddings", body, r)[1]["data"]
                    ]
                    for r in (False, True)
                ]
                stats = EncoderParity.rows_compare(vectors[1], vectors[0])
                self.cases.append(
                    {
                        "name": f"embeddings/layer{layer}/dim{dimension}",
                        "passed": stats["min_cosine"] >= REDUCED_MIN_COSINE,
                        **stats,
                    }
                )
        return [{"input": text} for text in texts()]

    def rerank(self) -> list[dict[str, Any]]:
        planner = self.model.planners["rerank"]
        for exit in planner.info.exits:
            deltas, agree, pairs = [], 0, 0
            for query, documents in RERANK:
                body = {
                    "query": query,
                    "documents": documents,
                    "layer": exit[0],
                    "dimensions": exit[1],
                }
                logits = []
                for r in (False, True):
                    results = serve(self.model, "rerank", body, r)[1]["results"]
                    by_index = {result["index"]: result["logit"] for result in results}
                    logits.append([by_index[i] for i in range(len(documents))])
                exact, reduced = logits
                deltas.append(float(np.abs(np.subtract(exact, reduced)).max()))
                for i in range(len(documents)):
                    for j in range(i + 1, len(documents)):
                        if abs(exact[i] - exact[j]) > TIE:
                            pairs += 1
                            agree += (exact[i] - exact[j]) * (
                                reduced[i] - reduced[j]
                            ) > 0
            self.cases.append(
                {
                    "name": f"rerank/layer{exit[0]}/dim{exit[1]}",
                    "passed": agree >= REDUCED_MIN_AGREEMENT * pairs,
                    "max_logit_delta": max(deltas),
                    "pair_order_agreement": agree / max(pairs, 1),
                    "pairs_outside_ties": pairs,
                }
            )
        return [{"query": query, "documents": list(docs)} for query, docs in RERANK]

    def latency(self, surface: str, bodies: list[dict[str, Any]]) -> dict[str, float]:
        """p50 of one request at a time over the corpus, exact and reduced, three passes each."""
        medians = {}
        for name, reduced in (("exact", False), ("reduced", True)):
            samples = []
            for _ in range(3):
                for body in bodies:
                    started = time.perf_counter()
                    serve(self.model, surface, body, reduced)
                    samples.append((time.perf_counter() - started) * 1000)
            medians[f"{name}_p50_ms"] = round(statistics.median(samples), 3)
        return medians

    def run(self) -> dict[str, Any]:
        surface = "embeddings" if "embeddings" in self.model.planners else "rerank"
        bodies = self.embeddings() if surface == "embeddings" else self.rerank()
        return {
            "model": self.model.info.id,
            "model_sha256": self.model.info.model_sha256,
            "kind": self.kind,
            "device": self.device.label,
            "copy": self.model.engine_model.receipt().get("reduced"),
            "thresholds": {
                "min_cosine": REDUCED_MIN_COSINE,
                "min_pair_order_agreement": REDUCED_MIN_AGREEMENT,
            },
            "latency": self.latency(surface, bodies),
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
    omni.add_argument("--snapshot", type=Path, default=None)
    omni.add_argument("--device", default="cpu")
    omni.add_argument("--output", type=Path, required=True)
    omni.add_argument("--threads", type=int, default=None)
    encoder = commands.add_parser(
        "encoder", help="an embedder or reranker on both engines vs Transformers"
    )
    encoder.add_argument("--package", type=Path, required=True)
    encoder.add_argument("--output", type=Path, required=True)
    encoder.add_argument("--threads", type=int, default=None)
    encoder.add_argument("--device", default="cpu")
    reduced = commands.add_parser(
        "reduced", help="an embedder's or reranker's reduced copy vs its exact path"
    )
    reduced.add_argument("--package", type=Path, required=True)
    reduced.add_argument(
        "--kind", required=True, choices=("bfloat16", "int8", "float32-packed")
    )
    reduced.add_argument("--output", type=Path, required=True)
    reduced.add_argument("--threads", type=int, default=None)
    reduced.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.command == "omni":
        snapshot = args.snapshot.resolve() if args.snapshot else None
        result = OmniParity(
            args.bundle.resolve(), args.threads, snapshot, args.device
        ).run()
    elif args.command == "reduced":
        result = ReducedParity(
            args.package.resolve(), args.kind, args.device, args.threads
        ).run()
    else:
        result = EncoderParity(args.package.resolve(), args.threads, args.device).run()
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
