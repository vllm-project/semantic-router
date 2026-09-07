"""Raw LLaVA-CC3M and COCO image-caption datasets."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from PIL import Image
from torch.utils.data import Dataset

LOGGER = logging.getLogger(__name__)
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


def _fallback_image(image_size: int) -> Image.Image:
    return Image.new("RGB", (image_size, image_size), color=(128, 128, 128))


def _load_image(path: Path, image_size: int) -> Image.Image:
    try:
        with Image.open(path) as image:
            return image.convert("RGB").resize(
                (image_size, image_size),
                Image.Resampling.LANCZOS,
            )
    except (OSError, ValueError):
        LOGGER.warning("Replacing unreadable image with a neutral fallback: %s", path)
        return _fallback_image(image_size)


class LLaVACC3MDataset(Dataset):
    """LLaVA-CC3M image-caption pairs from an explicitly supplied directory."""

    def __init__(
        self,
        data_dir: str,
        image_size: int = 256,
        max_samples: int | None = None,
    ) -> None:
        self.data_dir = Path(data_dir).expanduser()
        nested_images = self.data_dir / "images"
        self.images_dir = nested_images if nested_images.is_dir() else self.data_dir
        self.image_size = image_size
        self.samples = self._load_samples()
        if max_samples is not None:
            self.samples = self.samples[:max_samples]

    def _load_samples(self) -> list[dict[str, Any]]:
        chat_path = self.data_dir / "chat.json"
        if chat_path.is_file():
            return json.loads(chat_path.read_text(encoding="utf-8"))
        metadata_path = self.data_dir / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"Expected chat.json or metadata.json under {self.data_dir}"
            )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return [
            {
                "id": item["id"],
                "image": item["image"],
                "caption": item.get("caption", item.get("blip_caption", "")),
            }
            for item in metadata
        ]

    @staticmethod
    def _caption(sample: dict[str, Any]) -> str:
        for turn in sample.get("conversations", []):
            if turn.get("from") == "gpt":
                return str(turn.get("value", ""))
        return str(sample.get("caption", ""))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        image_name = str(sample.get("image", sample.get("id", f"{index}.jpg")))
        if not image_name.lower().endswith(IMAGE_SUFFIXES):
            image_name += ".jpg"
        return {
            "image": _load_image(self.images_dir / image_name, self.image_size),
            "caption": self._caption(sample),
        }


class COCOCaptionsDataset(Dataset):
    """One deterministic caption per image from a COCO captions manifest."""

    def __init__(
        self,
        annotations_file: str,
        images_dir: str,
        image_size: int = 256,
        max_samples: int | None = None,
    ) -> None:
        data = json.loads(Path(annotations_file).read_text(encoding="utf-8"))
        self.images_dir = Path(images_dir).expanduser()
        self.image_size = image_size
        id_to_file = {item["id"]: item["file_name"] for item in data["images"]}
        first_caption: dict[int, str] = {}
        for annotation in data["annotations"]:
            first_caption.setdefault(annotation["image_id"], annotation["caption"])
        self.samples = [
            {"image_file": id_to_file[image_id], "caption": caption}
            for image_id, caption in first_caption.items()
            if image_id in id_to_file
        ]
        if max_samples is not None:
            self.samples = self.samples[:max_samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        return {
            "image": _load_image(
                self.images_dir / sample["image_file"],
                self.image_size,
            ),
            "caption": sample["caption"],
        }


def raw_collate_fn(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "images": [item["image"] for item in batch],
        "captions": [item["caption"] for item in batch],
    }
