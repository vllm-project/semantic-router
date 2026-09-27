# Phase-local imports preserve memory limits and authenticated source loading.
# ruff: noqa: PLC0415
"""Compare full public native calls with the exported composition, including DSP.

Golden files are raw little-endian float32 tensors so Rust/Go acceptance tests
can independently verify preprocessing and end-to-end embeddings without Python.
"""

from __future__ import annotations

import gc
import importlib.metadata
import json
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from contract import (
    MANIFEST,
    PENDING_MANIFEST,
    digest,
    inventory,
    verify_inventory,
    write_json,
)
from PIL import Image
from processors import (
    aggregate_clap,
    image_processor,
    prepare_audio,
    store_array,
    waveform_fixture,
)
from torch.nn import functional

MINIMUM_COSINE = 0.99999
QUICK_TEXT_BUDGET = 1024


def comparison(name: str, actual, expected) -> dict:
    actual, expected = np.asarray(actual, dtype=np.float32), np.asarray(
        expected, dtype=np.float32
    )
    if (
        actual.shape != expected.shape
        or not np.isfinite(actual).all()
        or not np.isfinite(expected).all()
    ):
        raise ValueError(f"{name}: invalid output shape or nonfinite value")
    norms = np.linalg.norm(actual, axis=-1)
    reference_norms = np.linalg.norm(expected, axis=-1)
    cosine = float(
        np.min(
            np.sum(actual * expected, axis=-1)
            / np.maximum(norms * reference_norms, 1e-12)
        )
    )
    error = float(np.max(np.abs(actual - expected)))
    passed = bool(
        np.allclose(actual, expected, atol=1e-4, rtol=2e-4) and cosine >= MINIMUM_COSINE
    )
    result = {
        "name": name,
        "passed": passed,
        "shape": list(actual.shape),
        "max_absolute_error": error,
        "min_cosine": cosine,
    }
    if not passed:
        raise ValueError(f"reference parity failed: {result}")
    return result


def numpy_output(value):
    return value.detach().float().cpu().numpy()


def load_sessions(directory: Path, manifest: dict, provider: str, threads: int, names):
    import onnxruntime as ort

    if provider not in ort.get_available_providers():
        raise ValueError(f"requested ONNX provider is unavailable: {provider}")
    result = {}
    for name, graph in manifest["graphs"].items():
        if name not in names:
            continue
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        if provider != "CPUExecutionProvider":
            options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        session = ort.InferenceSession(
            str(directory / graph["file"]), sess_options=options, providers=[provider]
        )
        if [item.name for item in session.get_inputs()] != [
            item["name"] for item in graph["inputs"]
        ]:
            raise ValueError(f"{name}: exported graph inputs disagree with manifest")
        for actual, declared in zip(
            session.get_inputs() + session.get_outputs(),
            graph["inputs"] + [graph["output"]],
            strict=True,
        ):
            dtype = (
                "tensor(float)" if declared["dtype"] == "float32" else "tensor(int64)"
            )
            if actual.shape != declared["shape"] or actual.type != dtype:
                raise ValueError(
                    f"{name}: exported tensor shape/type disagrees with manifest"
                )
        result[name] = session
    return result


def run(session, **values):
    inputs = {
        key: (
            numpy_output(value)
            if isinstance(value, torch.Tensor) and value.dtype.is_floating_point
            else value.cpu().numpy() if isinstance(value, torch.Tensor) else value
        )
        for key, value in values.items()
    }
    return session.run(["embedding"], inputs)[0]


def text_fixture(tokenizer, budget: int) -> str:
    # Produce a public string at the exact supported budget, with no truncation.
    count = budget
    for _ in range(12):
        text = "x " * count
        length = len(tokenizer(text.strip(), truncation=False)["input_ids"])
        if length == budget:
            return text.strip()
        count += budget - length
    raise ValueError(f"could not construct exact {budget}-token boundary fixture")


def qualify_cpu_flash_reference(reference):
    from reference_backend import (
        cpu_flash_reference,
    )

    checks = []
    # Public batch encoding exercises actual tokenizer padding. Lengths on both
    # sides of the export block boundary also check full causal visibility.
    cases = [
        ["A short causal sentence."],
        [text_fixture(reference.tokenizer, 257)],
        ["短句。", text_fixture(reference.tokenizer, 513)],
    ]
    for index, texts in enumerate(cases):
        expected = numpy_output(reference.encode_text(texts))
        with cpu_flash_reference():
            actual = numpy_output(reference.encode_text(texts))
        checks.append(comparison(f"reference/cpu-flash-{index}", actual, expected))
    return checks


def verify_text(
    reference, sessions, root: Path, manifest: dict, full_context: bool, device: str
):
    texts = [
        "A cat is sitting beside a window.",
        "会议室白板旁边有一张桌子。",
        "  Python traceback:\n  ValueError: invalid input  ",
    ]
    probe_tokens = (
        manifest["max_text_length"]
        if full_context or reference.variant == "nano"
        else QUICK_TEXT_BUDGET
    )
    texts.append(text_fixture(reference.tokenizer, probe_tokens))
    records, checks = [], []
    bounded_reference = full_context and reference.variant == "mini" and device == "cpu"
    if bounded_reference:
        checks.extend(qualify_cpu_flash_reference(reference))
    for index, text in enumerate(texts):
        effective = text.strip() if reference.variant == "mini" else text
        tokens = reference.tokenizer(effective, return_tensors="pt", truncation=False)
        count = tokens["input_ids"].shape[1]
        actual = run(
            sessions["text"],
            input_ids=tokens["input_ids"],
            attention_mask=tokens["attention_mask"],
        )
        from reference_backend import (
            cpu_flash_reference,
        )

        context = (
            cpu_flash_reference()
            if bounded_reference and count > QUICK_TEXT_BUDGET
            else nullcontext()
        )
        with context:
            expected = numpy_output(reference.encode_text([text]))
        checks.append(comparison(f"text/{index}", actual, expected))
        records.append(
            {
                "text": text,
                "tokens": count,
                "input_ids": tokens["input_ids"].tolist(),
                "attention_mask": tokens["attention_mask"].tolist(),
                "embedding": store_array(root, f"text-{index}-embedding", expected),
            }
        )
    padded_text = "A short input padded for a fixed execution window."
    padded = reference.tokenizer(
        padded_text,
        padding="max_length",
        max_length=64,
        truncation=False,
        return_tensors="pt",
    )
    expected = numpy_output(reference.encode_text([padded_text]))
    actual = run(
        sessions["text"],
        input_ids=padded["input_ids"],
        attention_mask=padded["attention_mask"],
    )
    checks.append(comparison("text/padding", actual, expected))
    records.append(
        {
            "text": padded_text,
            "input_ids": padded["input_ids"].tolist(),
            "attention_mask": padded["attention_mask"].tolist(),
            "embedding": store_array(root, "text-padding-embedding", expected),
        }
    )
    if reference.variant == "mini":
        from omni_components.text_instructions import (
            format_texts,
        )

        for role in ("query", "document"):
            text = "  how to reset a router?"
            formatted = format_texts([text], task="retrieval", role=role)[0].strip()
            tokens = reference.tokenizer(
                formatted, return_tensors="pt", truncation=False
            )
            actual = run(
                sessions["text"],
                input_ids=tokens["input_ids"],
                attention_mask=tokens["attention_mask"],
            )
            expected = numpy_output(
                reference.encode_text([text], task="retrieval", role=role)
            )
            checks.append(comparison(f"text/instruction-{role}", actual, expected))
            records.append(
                {
                    "text": text,
                    "task": "retrieval",
                    "role": role,
                    "formatted_text": formatted,
                    "input_ids": tokens["input_ids"].tolist(),
                    "attention_mask": tokens["attention_mask"].tolist(),
                    "embedding": store_array(
                        root, f"text-instruction-{role}-embedding", expected
                    ),
                }
            )
    # The public limit is a reject contract, not a runtime truncation instruction.
    too_long = text_fixture(reference.tokenizer, manifest["max_text_length"] + 1)
    try:
        reference.encode_text([too_long])
    except ValueError:
        checks.append({"name": "text/overflow-rejected", "passed": True})
    else:
        raise ValueError(
            "official reference unexpectedly accepted an over-budget input"
        )
    return records, checks, probe_tokens


def verify_images(reference, sessions, root: Path):
    records, checks = [], []
    cases = ((137, 89, "png"), (31, 143, "png"), (151, 97, "jpg"))
    for index, (width, height, extension) in enumerate(cases):
        yy, xx = np.indices((height, width))
        rgb = np.stack(
            (
                (xx * 13 + yy * 7) % 256,
                ((xx // 3 + yy // 5) % 2) * 255,
                (xx * yy) % 256,
            ),
            axis=-1,
        ).astype(np.uint8)
        image = Image.fromarray(rgb)
        filename = f"image-{index}.{extension}"
        if extension == "jpg":
            image.save(root / filename, quality=85, subsampling=2)
            # Golden inputs are encoded bytes. Re-open JPEG through the public
            # Pillow decoder, so native tests also exercise chroma upsampling.
            image = Image.open(root / filename).convert("RGB")
        else:
            image.save(root / filename)
        pixels = image_processor(reference)(images=[image], return_tensors="pt")[
            "pixel_values"
        ]
        actual = run(sessions["image"], pixel_values=pixels)
        expected = numpy_output(reference.encode_image([image]))
        checks.append(comparison(f"image/{index}", actual, expected))
        records.append(
            {
                "file": filename,
                "pixels": store_array(root, f"image-{index}-pixels", pixels.numpy()),
                "embedding": store_array(root, f"image-{index}-embedding", expected),
            }
        )
    return records, checks


def verify_audio(reference, sessions, root: Path, device: str):
    records, checks = [], []
    # Exercises both resampling directions, original-rate stereo order, all
    # one/two/three endpoint-window cases, and a distinctive long-audio tail.
    cases = [
        (16000, 0.75, False),
        (44100, 1.25, True),
        (48000, 11.125, False),
        (48000, 21.0, False),
    ]
    if reference.variant == "mini":
        cases.append((22050, 0.8, False))
    for index, (rate, seconds, stereo) in enumerate(cases):
        print(
            f"Checking {reference.variant} audio parity {index}: {rate} Hz / {seconds} s",
            flush=True,
        )
        wave = waveform_fixture(rate, seconds, stereo)
        audio16, audio48, windows, whisper, clap_features = prepare_audio(
            reference, wave, rate
        )
        vectors = torch.from_numpy(
            np.concatenate(
                [
                    run(sessions["clap"], input_features=features.unsqueeze(0))
                    for features in clap_features
                ]
            )
        )
        aggregate = aggregate_clap(vectors)
        actual = run(
            sessions["audio"], input_features=whisper, clap_embedding=aggregate
        )
        expected = numpy_output(reference.encode_audio([wave], sampling_rate=rate))
        checks.append(comparison(f"audio/{index}/end-to-end", actual, expected))
        # Isolate CLAP export/rewrite parity so Whisper cannot hide a dropped or
        # substituted environmental-sound branch in the final vector.
        with torch.inference_mode():
            original_clap = functional.normalize(
                reference.model.audio_residual.clap(
                    input_features=clap_features.to(device)
                ).audio_embeds.float(),
                dim=-1,
            )
        checks.append(
            comparison(
                f"audio/{index}/clap", vectors.numpy(), numpy_output(original_clap)
            )
        )
        records.append(
            {
                "sampling_rate": rate,
                "seconds": seconds,
                "windows": windows,
                "pcm": store_array(root, f"audio-{index}-pcm", wave),
                "audio16": store_array(root, f"audio-{index}-16k", audio16),
                "audio48": store_array(root, f"audio-{index}-48k", audio48),
                "whisper_features": store_array(
                    root, f"audio-{index}-whisper", whisper.numpy()
                ),
                "clap_features": store_array(
                    root, f"audio-{index}-clap", clap_features.numpy()
                ),
                "clap_embeddings": store_array(
                    root, f"audio-{index}-clap-embeddings", numpy_output(original_clap)
                ),
                "clap_aggregate": store_array(
                    root, f"audio-{index}-clap-aggregate", aggregate.numpy()
                ),
                "embedding": store_array(root, f"audio-{index}-embedding", expected),
            }
        )
    return records, checks


def verify(
    reference,
    directory: Path,
    device: str = "cpu",
    provider: str = "CPUExecutionProvider",
    threads: int = 4,
    full_context: bool = False,
):
    pending = directory / PENDING_MANIFEST
    manifest = json.loads(
        (pending if pending.exists() else directory / MANIFEST).read_text()
    )
    if manifest["variant"] != reference.variant:
        raise ValueError("source variant differs from exported artifact")
    verify_inventory(directory, manifest)
    reference.to(device=device, dtype=torch.float32).eval()
    golden = directory / "golden"
    golden.mkdir(exist_ok=True)
    sessions = load_sessions(directory, manifest, provider, threads, ("text",))
    texts, text_checks, executed_budget = verify_text(
        reference, sessions, golden, manifest, full_context, device
    )
    del sessions
    gc.collect()
    sessions = load_sessions(directory, manifest, provider, threads, ("image",))
    images, image_checks = verify_images(reference, sessions, golden)
    del sessions
    gc.collect()
    sessions = load_sessions(directory, manifest, provider, threads, ("clap", "audio"))
    audio, audio_checks = verify_audio(reference, sessions, golden, device)
    del sessions
    gc.collect()
    write_json(
        golden / "index.json",
        {
            "format_version": 1,
            "variant": reference.variant,
            "texts": texts,
            "images": images,
            "audio": audio,
        },
    )
    report = {
        "format_version": 1,
        "passed": True,
        "source": manifest["source"],
        "variant": reference.variant,
        "provider": provider,
        "reference_device": device,
        "reference_attention_backend": (
            "torch_cpu_flash_sdpa_explicit_kv_repeat"
            if full_context and reference.variant == "mini" and device == "cpu"
            else "published_default"
        ),
        "precision": "float32",
        "exporter": {
            path.name: digest(path)
            for path in sorted(Path(__file__).parent.iterdir())
            if path.suffix in (".py", ".json") or path.name == "requirements.txt"
        },
        "versions": {
            name: importlib.metadata.version(name)
            for name in (
                "torch",
                "torchaudio",
                "transformers",
                "onnx",
                "onnxruntime",
                "numpy",
                "pillow",
            )
        },
        "tolerance": {
            "absolute": 1e-4,
            "relative": 2e-4,
            "minimum_cosine": MINIMUM_COSINE,
        },
        "full_context_executed": executed_budget == manifest["max_text_length"],
        "longest_text_tokens_executed": executed_budget,
        "tests": text_checks + image_checks + audio_checks,
    }
    write_json(directory / "reference_parity.json", report)
    manifest["reference_parity"]["passed"] = True
    manifest["files"] = inventory(directory)
    write_json(directory / MANIFEST, manifest)
    pending.unlink(missing_ok=True)
    return report
