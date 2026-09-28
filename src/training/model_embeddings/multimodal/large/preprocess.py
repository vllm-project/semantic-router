#!/usr/bin/env python3
import argparse
import json
import os
from glob import glob
from typing import Any

import librosa
import numpy as np
import torch
from PIL import Image
from torch.nn import functional
from torchaudio import transforms
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer

from .data import JsonlManifestDataset, PairItem, TrainRecord

IMAGE_SIZE_COMPONENTS = 2
WHISPER_MEL_FRAMES = 3000


def _resolve_image_processor(processor):
    return (
        processor.image_processor
        if hasattr(processor, "image_processor")
        else processor
    )


def _resolve_image_size(image_processor) -> tuple[int, int]:
    size = getattr(image_processor, "size", {}) or {}
    if hasattr(size, "height") or hasattr(size, "width"):
        height = int(
            getattr(size, "height", None) or getattr(size, "shortest_edge", None) or 384
        )
        width = int(
            getattr(size, "width", None)
            or getattr(size, "shortest_edge", None)
            or height
        )
        return height, width
    if hasattr(size, "to_dict"):
        size = size.to_dict()
    if hasattr(size, "items"):
        size = dict(size)
    if isinstance(size, dict):
        height = int(
            size.get("height") or size.get("shortest_edge") or size.get("size") or 384
        )
        width = int(
            size.get("width") or size.get("shortest_edge") or size.get("size") or height
        )
        return height, width
    if isinstance(size, (list, tuple)) and len(size) == IMAGE_SIZE_COMPONENTS:
        return int(size[0]), int(size[1])
    return int(size or 384), int(size or 384)


def _image_to_tensor(path: str) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def _preprocess_image_batch(
    paths: list[str], image_processor, device: torch.device
) -> list[torch.Tensor]:
    resolved = _resolve_image_processor(image_processor)
    image_mean = torch.tensor(
        getattr(resolved, "image_mean", [0.5, 0.5, 0.5]),
        dtype=torch.float32,
        device=device,
    ).view(1, 3, 1, 1)
    image_std = torch.tensor(
        getattr(resolved, "image_std", [0.5, 0.5, 0.5]),
        dtype=torch.float32,
        device=device,
    ).view(1, 3, 1, 1)
    target_h, target_w = _resolve_image_size(resolved)

    tensors = [_image_to_tensor(path) for path in paths]
    processed: list[torch.Tensor] = []
    for tensor in tensors:
        batch = tensor.unsqueeze(0).to(device)
        batch = functional.interpolate(
            batch, size=(target_h, target_w), mode="bilinear", align_corners=False
        )
        batch = (batch - image_mean) / image_std
        processed.append(batch.squeeze(0).cpu())
    return processed


def _load_audio_tensor(path: str, max_audio_len: int = 30 * 16000) -> torch.Tensor:
    wave, _sample_rate = librosa.load(path, sr=16000, mono=True)
    tensor = torch.tensor(wave, dtype=torch.float32)
    if tensor.numel() > max_audio_len:
        tensor = tensor[:max_audio_len]
    elif tensor.numel() < max_audio_len:
        tensor = functional.pad(tensor, (0, max_audio_len - tensor.numel()))
    return tensor


def _preprocess_audio_batch(
    paths: list[str], mel_transform, device: torch.device
) -> list[torch.Tensor]:
    waves = torch.stack([_load_audio_tensor(path) for path in paths], dim=0).to(device)
    with torch.no_grad():
        mel = mel_transform(waves)
        mel = torch.clamp(mel, min=1e-10).log10()
        mel = torch.maximum(mel, mel.amax(dim=(1, 2), keepdim=True) - 8.0)
        mel = (mel + 4.0) / 4.0
        if mel.size(-1) < WHISPER_MEL_FRAMES:
            mel = functional.pad(mel, (0, WHISPER_MEL_FRAMES - mel.size(-1)))
        else:
            mel = mel[..., :WHISPER_MEL_FRAMES]
    return [item.cpu() for item in mel]


def _serialize_text_item(
    item: PairItem, tokenizer, max_text_length: int
) -> dict[str, Any]:
    encoded = tokenizer(
        item.value,
        padding=False,
        truncation=True,
        max_length=max_text_length,
        return_tensors="pt",
    )
    return {
        "type": "text",
        "tokens": {
            key: value.squeeze(0).cpu()
            for key, value in encoded.items()
            if hasattr(value, "squeeze")
        },
    }


def serialize_records(
    records: list[TrainRecord],
    tokenizer,
    image_processor,
    mel_transform,
    device: torch.device,
    max_text_length: int,
) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = [
        {"query": None, "positive": None, "negative": None} for _ in records
    ]

    text_items: list[tuple[int, str, PairItem]] = []
    image_items: list[tuple[int, str, PairItem]] = []
    audio_items: list[tuple[int, str, PairItem]] = []

    for rec_idx, record in enumerate(records):
        for field_name in ("query", "positive", "negative"):
            item = getattr(record, field_name)
            if item is None:
                continue
            if item.modality == "text":
                text_items.append((rec_idx, field_name, item))
            elif item.modality == "image":
                image_items.append((rec_idx, field_name, item))
            elif item.modality == "audio":
                audio_items.append((rec_idx, field_name, item))

    for rec_idx, field_name, item in text_items:
        serialized[rec_idx][field_name] = _serialize_text_item(
            item, tokenizer, max_text_length
        )

    if image_items:
        processed = _preprocess_image_batch(
            [item.value for _, _, item in image_items], image_processor, device
        )
        for (rec_idx, field_name, _), tensor in zip(
            image_items, processed, strict=False
        ):
            serialized[rec_idx][field_name] = {"type": "image", "tensor": tensor}

    if audio_items:
        processed = _preprocess_audio_batch(
            [item.value for _, _, item in audio_items], mel_transform, device
        )
        for (rec_idx, field_name, _), tensor in zip(
            audio_items, processed, strict=False
        ):
            serialized[rec_idx][field_name] = {"type": "audio", "tensor": tensor}

    return serialized


def save_shard(path: str, records: list[dict[str, Any]]) -> None:
    torch.save({"records": records}, path)


def write_metadata(output_dir: str, metadata: dict[str, Any]) -> None:
    with open(
        os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(metadata, handle, indent=2)


def discover_existing_records(output_dir: str) -> tuple[int, int]:
    shard_paths = sorted(glob(os.path.join(output_dir, "shard_*.pt")))
    valid_shard_count = 0
    record_count = 0
    for shard_path in shard_paths:
        try:
            payload = torch.load(shard_path, map_location="cpu", weights_only=False)
        except Exception:
            continue
        record_count += len(payload.get("records", []))
        valid_shard_count += 1
    return valid_shard_count, record_count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cache manifest records into tensor shards"
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-root", default=None)
    parser.add_argument("--audio-root", default=None)
    parser.add_argument("--image-encoder-name", required=True)
    parser.add_argument("--audio-encoder-name", required=True)
    parser.add_argument(
        "--text-encoder-name",
        default="llm-semantic-router/mmbert-embed-32k-2d-matryoshka",
    )
    parser.add_argument("--max-text-length", type=int, default=32768)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--shard-size", type=int, default=512)
    parser.add_argument("--max-records", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    dataset = JsonlManifestDataset(
        manifest_path=args.manifest,
        image_root=args.image_root,
        audio_root=args.audio_root,
        allow_missing_negative=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.text_encoder_name, trust_remote_code=True
    )
    image_processor = AutoProcessor.from_pretrained(
        args.image_encoder_name, trust_remote_code=True
    )
    device = torch.device(args.device)
    mel_transform = transforms.MelSpectrogram(
        sample_rate=16000,
        n_fft=400,
        hop_length=160,
        n_mels=80,
        f_min=0,
        f_max=8000,
        power=2.0,
    ).to(device)

    shard_idx, processed_records = discover_existing_records(args.output_dir)
    shard_records: list[dict[str, Any]] = []
    total = (
        len(dataset) if args.max_records <= 0 else min(len(dataset), args.max_records)
    )

    pending_records: list[TrainRecord] = []
    for idx in tqdm(range(processed_records, total), desc="caching manifest"):
        pending_records.append(dataset[idx])
        if len(pending_records) >= args.shard_size:
            shard_records = serialize_records(
                pending_records,
                tokenizer=tokenizer,
                image_processor=image_processor,
                mel_transform=mel_transform,
                device=device,
                max_text_length=args.max_text_length,
            )
            save_shard(
                os.path.join(args.output_dir, f"shard_{shard_idx:05d}.pt"),
                shard_records,
            )
            shard_idx += 1
            pending_records = []

    if pending_records:
        shard_records = serialize_records(
            pending_records,
            tokenizer=tokenizer,
            image_processor=image_processor,
            mel_transform=mel_transform,
            device=device,
            max_text_length=args.max_text_length,
        )
        save_shard(
            os.path.join(args.output_dir, f"shard_{shard_idx:05d}.pt"), shard_records
        )
        shard_idx += 1

    write_metadata(
        args.output_dir,
        {
            "manifest": args.manifest,
            "num_records": total,
            "num_shards": shard_idx,
            "shard_size": args.shard_size,
            "text_encoder_name": args.text_encoder_name,
            "max_text_length": args.max_text_length,
            "image_encoder_name": args.image_encoder_name,
            "audio_encoder_name": args.audio_encoder_name,
            "device": args.device,
        },
    )


if __name__ == "__main__":
    main()
