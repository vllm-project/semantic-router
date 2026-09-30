"""Bounded-memory sequential shard dataset with background prefetch."""

from __future__ import annotations

import random
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset

from .shards import discover_complete_shards, load_shard


class SequentialShardDataset(Dataset):
    """Expose one cached shard at a time and partition shard files across ranks."""

    def __init__(
        self,
        cache_dir: str,
        shuffle: bool = True,
        dynamic_discovery: bool = True,
        rank: int = 0,
        world_size: int = 1,
        prefetch: bool = True,
        prefetch_buffer_size: int = 3,
        feature_key: str = "pixel_values",
        min_shard_samples: int = 64,
        seed: int = 42,
    ) -> None:
        del prefetch_buffer_size  # A single future bounds memory to two shards.
        self.cache_dir = Path(cache_dir).expanduser()
        self.shuffle = shuffle
        self.dynamic_discovery = dynamic_discovery
        self.rank = rank
        self.world_size = world_size
        self.prefetch_enabled = prefetch
        self.feature_key = feature_key
        self.min_shard_samples = min_shard_samples
        self.seed = seed
        self.current_shard_idx = -1
        self.current_shard_data: dict[str, Any] | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._future: Future[dict[str, Any]] | None = None
        self._future_index: int | None = None
        self._prefetch_stats = {"hits": 0, "misses": 0}
        all_shards = self._ordered_shards()
        self._all_shard_count = len(all_shards)
        self.shard_files = self._partition_and_pad(all_shards)
        if self.prefetch_enabled:
            self._executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="shards"
            )
        if self.shard_files:
            self.next_shard()

    def _ordered_shards(self) -> list[Path]:
        shards = discover_complete_shards(self.cache_dir)
        if self.shuffle:
            random.Random(self.seed).shuffle(shards)
        return shards

    def _partition_and_pad(self, all_shards: list[Path]) -> list[Path]:
        if self.world_size <= 1:
            return all_shards
        assigned = all_shards[self.rank :: self.world_size]
        target = (len(all_shards) + self.world_size - 1) // self.world_size
        if not assigned and target:
            raise RuntimeError(
                f"Rank {self.rank} has no shard; cache needs at least {self.world_size} shards"
            )
        assigned.extend(
            assigned[index % len(assigned)] for index in range(target - len(assigned))
        )
        return assigned

    def _submit_prefetch(self, index: int) -> None:
        if self._executor is None or index >= len(self.shard_files):
            self._future = None
            self._future_index = None
            return
        self._future_index = index
        self._future = self._executor.submit(load_shard, self.shard_files[index])

    def discover_new_shards(self) -> int:
        """Append newly completed shards for this rank."""
        if not self.dynamic_discovery:
            return 0
        known = set(self.shard_files)
        ordered = self._ordered_shards()
        self._all_shard_count = len(ordered)
        candidates = ordered[self.rank :: self.world_size]
        additions = [path for path in candidates if path not in known]
        self.shard_files.extend(additions)
        return len(additions)

    def next_shard(self) -> bool:
        """Load the next assigned shard and start prefetching its successor."""
        next_index = self.current_shard_idx + 1
        if next_index >= len(self.shard_files):
            self.discover_new_shards()
        if next_index >= len(self.shard_files):
            return False

        if self._future is not None and self._future_index == next_index:
            data = self._future.result()
            self._prefetch_stats["hits"] += 1
        else:
            data = load_shard(self.shard_files[next_index])
            self._prefetch_stats["misses"] += 1

        self.current_shard_idx = next_index
        self.current_shard_data = data
        if len(data["input_ids"]) < self.min_shard_samples:
            raise RuntimeError(
                f"{self.shard_files[next_index].name} has fewer than "
                f"{self.min_shard_samples} samples"
            )
        self._submit_prefetch(next_index + 1)
        return True

    def reset(self) -> None:
        """Reset shard order and load the first shard for a new epoch."""
        self.close()
        if self.shuffle:
            random.Random(self.seed).shuffle(self.shard_files)
            self.seed += 1
        if self.prefetch_enabled:
            self._executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="shards"
            )
        self.current_shard_idx = -1
        self.current_shard_data = None
        if self.shard_files:
            self.next_shard()

    def close(self) -> None:
        if self._future is not None:
            self._future.cancel()
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
        self._future = None
        self._future_index = None
        self._executor = None

    def __len__(self) -> int:
        if self.current_shard_data is None:
            return 0
        return len(self.current_shard_data["input_ids"])

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self.current_shard_data is None:
            raise IndexError("No shard is loaded")
        return {
            self.feature_key: self.current_shard_data[self.feature_key][index],
            "input_ids": self.current_shard_data["input_ids"][index],
            "attention_mask": self.current_shard_data["attention_mask"][index],
        }

    @property
    def num_shards(self) -> int:
        return len(self.shard_files)

    @property
    def total_samples_estimate(self) -> int:
        return self._all_shard_count * 5000

    def get_prefetch_stats(self) -> dict[str, float]:
        total = self._prefetch_stats["hits"] + self._prefetch_stats["misses"]
        return {
            **self._prefetch_stats,
            "hit_rate": 100.0 * self._prefetch_stats["hits"] / max(1, total),
        }

    def __del__(self) -> None:
        self.close()
