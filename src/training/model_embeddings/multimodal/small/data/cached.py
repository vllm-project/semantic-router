"""Random-access cached tensor datasets."""

from __future__ import annotations

import json
import random
import threading
import time
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from .shards import discover_complete_shards, load_shard

MAX_CACHED_SHARDS = 3


class CachedTensorDataset(Dataset):
    """Index cached tensor shards and discover newly completed shards."""

    def __init__(
        self,
        cache_dir: str,
        max_samples: int | None = None,
        shuffle_shards: bool = True,
        dynamic_discovery: bool = True,
        discovery_interval: int = 60,
        feature_key: str = "pixel_values",
        seed: int = 42,
    ) -> None:
        self.cache_dir = Path(cache_dir).expanduser()
        self.max_samples = max_samples
        self.shuffle_shards = shuffle_shards
        self.dynamic_discovery = dynamic_discovery
        self.discovery_interval = discovery_interval
        self.feature_key = feature_key
        self._rng = random.Random(seed)
        self._known_shards: set[Path] = set()
        self.shard_files: list[Path] = []
        self.shard_sizes: dict[Path, int] = {}
        self.index: list[tuple[int, int]] = []
        self._loaded: dict[int, dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._last_discovery = 0.0
        self.metadata = self._load_metadata()
        self._discover_shards()

    def _load_metadata(self) -> dict[str, Any]:
        path = self.cache_dir / "metadata.json"
        if not path.is_file():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def _discover_shards(self) -> int:
        if self.max_samples is not None and len(self.index) >= self.max_samples:
            self._last_discovery = time.time()
            return 0

        added = 0
        with self._lock:
            for path in discover_complete_shards(self.cache_dir):
                if path in self._known_shards:
                    continue
                try:
                    shard = load_shard(path)
                    size = len(shard["input_ids"])
                except (OSError, KeyError, RuntimeError, ValueError):
                    continue

                shard_index = len(self.shard_files)
                self.shard_files.append(path)
                self._known_shards.add(path)
                self.shard_sizes[path] = size
                for local_index in range(size):
                    if (
                        self.max_samples is not None
                        and len(self.index) >= self.max_samples
                    ):
                        break
                    self.index.append((shard_index, local_index))
                    added += 1
            if self.shuffle_shards and added:
                self._rng.shuffle(self.index)
        self._last_discovery = time.time()
        return added

    def check_for_new_shards(self) -> int:
        """Immediately scan the cache for newly completed shards."""
        return self._discover_shards()

    def _maybe_discover(self) -> None:
        if (
            self.dynamic_discovery
            and time.time() - self._last_discovery > self.discovery_interval
        ):
            self._discover_shards()

    def _load_shard(self, shard_index: int) -> dict[str, Any]:
        with self._lock:
            if shard_index not in self._loaded:
                self._loaded[shard_index] = load_shard(self.shard_files[shard_index])
                while len(self._loaded) > MAX_CACHED_SHARDS:
                    oldest = next(key for key in self._loaded if key != shard_index)
                    del self._loaded[oldest]
            return self._loaded[shard_index]

    def __len__(self) -> int:
        self._maybe_discover()
        return len(self.index)

    def __getitem__(self, index: int) -> dict[str, Any]:
        if not self.index:
            raise IndexError("No complete tensor shards are available")
        shard_index, local_index = self.index[index % len(self.index)]
        shard = self._load_shard(shard_index)
        return {
            self.feature_key: shard[self.feature_key][local_index],
            "input_ids": shard["input_ids"][local_index],
            "attention_mask": shard["attention_mask"][local_index],
        }

    def get_stats(self) -> dict[str, Any]:
        return {
            "total_samples": len(self.index),
            "num_shards": len(self.shard_files),
            "shard_files": [path.name for path in self.shard_files],
            "dynamic_discovery": self.dynamic_discovery,
        }


class CachedAudioDataset(CachedTensorDataset):
    """Random-access cached Whisper features paired with tokenized text."""

    def __init__(self, cache_dir: str, **kwargs: Any) -> None:
        super().__init__(cache_dir, feature_key="input_features", **kwargs)
