"""Bounded-memory random-access and sequential cached-shard datasets."""

import json
import math
import os
import queue
import random
import threading
from bisect import bisect_right
from collections import OrderedDict
from typing import Any

import torch

from .records import PairItem, TrainRecord


class CachedShardDataset:
    def __init__(
        self, cache_dir: str, shard_cache_limit: int = 2, prefetch_shards: int = 0
    ) -> None:
        self.cache_dir = cache_dir
        self.shard_cache_limit = max(int(shard_cache_limit), 1)
        self.prefetch_shards = max(int(prefetch_shards), 0)
        self.metadata = self._load_metadata()
        self.shard_files = self._discover_shards()
        self.shard_sizes = self._resolve_shard_sizes()
        self.shard_offsets = self._build_offsets(self.shard_sizes)
        self.total_rows = sum(self.shard_sizes)
        self._shard_cache: OrderedDict[int, list[dict[str, Any]]] = OrderedDict()
        self._init_runtime_state()

    def _init_runtime_state(self) -> None:
        self._cache_lock = threading.Lock()
        self._prefetch_queue = None
        self._prefetch_thread = None
        self._prefetch_stop = threading.Event()
        self._prefetch_requested: set[int] = set()
        self._prefetch_hits = 0
        self._prefetch_misses = 0

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_shard_cache"] = OrderedDict(state.get("_shard_cache", OrderedDict()))
        state["_cache_lock"] = None
        state["_prefetch_queue"] = None
        state["_prefetch_thread"] = None
        state["_prefetch_stop"] = None
        state["_prefetch_requested"] = set()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._shard_cache = OrderedDict(self._shard_cache)
        self._init_runtime_state()

    def _load_metadata(self) -> dict[str, Any]:
        metadata_path = os.path.join(self.cache_dir, "metadata.json")
        if not os.path.exists(metadata_path):
            return {}
        with open(metadata_path, encoding="utf-8") as handle:
            return json.load(handle)

    def _discover_shards(self) -> list[str]:
        if not os.path.isdir(self.cache_dir):
            raise FileNotFoundError(f"Cache directory not found: {self.cache_dir}")
        shards: list[str] = []
        for name in sorted(os.listdir(self.cache_dir)):
            if not (name.startswith("shard_") and name.endswith(".pt")):
                continue
            shard_path = os.path.join(self.cache_dir, name)
            shards.append(shard_path)
        if not shards:
            raise ValueError(f"No cache shards found under {self.cache_dir}")
        return shards

    @staticmethod
    def _build_offsets(shard_sizes: list[int]) -> list[int]:
        offsets: list[int] = []
        running_total = 0
        for shard_size in shard_sizes:
            running_total += shard_size
            offsets.append(running_total)
        return offsets

    def _resolve_shard_sizes(self) -> list[int]:
        num_shards = len(self.shard_files)
        metadata_num_shards = self.metadata.get("num_shards")
        metadata_num_records = self.metadata.get("num_records")
        shard_size = self.metadata.get("shard_size")

        if (
            isinstance(metadata_num_shards, int)
            and isinstance(metadata_num_records, int)
            and isinstance(shard_size, int)
            and metadata_num_shards == num_shards
            and metadata_num_records > 0
            and shard_size > 0
        ):
            shard_sizes = [shard_size] * num_shards
            full_rows_before_last = shard_size * max(num_shards - 1, 0)
            shard_sizes[-1] = metadata_num_records - full_rows_before_last
            if shard_sizes[-1] <= 0:
                raise ValueError(
                    f"Invalid metadata in {self.cache_dir}: last shard size computed as {shard_sizes[-1]}"
                )
            return shard_sizes

        shard_sizes: list[int] = []
        for shard_path in self.shard_files:
            payload = torch.load(shard_path, map_location="cpu", weights_only=False)
            records = payload.get("records")
            if not isinstance(records, list):
                raise ValueError(f"Invalid shard format in {shard_path}")
            shard_sizes.append(len(records))
        return shard_sizes

    def _store_shard(self, shard_idx: int, records: list[dict[str, Any]]) -> None:
        with self._cache_lock:
            self._shard_cache[shard_idx] = records
            self._shard_cache.move_to_end(shard_idx)
            while len(self._shard_cache) > self.shard_cache_limit:
                self._shard_cache.popitem(last=False)

    def _ensure_prefetch_thread(self) -> None:
        if self.prefetch_shards <= 0:
            return
        if self._prefetch_thread is not None and self._prefetch_thread.is_alive():
            return

        self._prefetch_stop.clear()
        self._prefetch_queue = queue.Queue(maxsize=max(self.prefetch_shards * 2, 1))
        self._prefetch_thread = threading.Thread(
            target=self._prefetch_worker,
            daemon=True,
            name=f"cached-shard-prefetch-{os.getpid()}",
        )
        self._prefetch_thread.start()

    def _prefetch_worker(self) -> None:
        while not self._prefetch_stop.is_set():
            try:
                shard_idx = self._prefetch_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if shard_idx is None:
                continue

            try:
                with self._cache_lock:
                    if shard_idx in self._shard_cache:
                        self._prefetch_hits += 1
                        continue
                payload = torch.load(
                    self.shard_files[shard_idx], map_location="cpu", weights_only=False
                )
                records = payload["records"]
                self._store_shard(shard_idx, records)
                self._prefetch_hits += 1
            finally:
                with self._cache_lock:
                    self._prefetch_requested.discard(shard_idx)

    def _schedule_prefetch(self, shard_idx: int) -> None:
        if self.prefetch_shards <= 0:
            return

        self._ensure_prefetch_thread()
        if self._prefetch_queue is None:
            return

        for next_idx in range(
            shard_idx + 1,
            min(len(self.shard_files), shard_idx + 1 + self.prefetch_shards),
        ):
            with self._cache_lock:
                if (
                    next_idx in self._shard_cache
                    or next_idx in self._prefetch_requested
                ):
                    continue
                self._prefetch_requested.add(next_idx)
            try:
                self._prefetch_queue.put_nowait(next_idx)
            except queue.Full:
                with self._cache_lock:
                    self._prefetch_requested.discard(next_idx)
                break

    def _load_shard(self, shard_idx: int) -> list[dict[str, Any]]:
        cached = None
        with self._cache_lock:
            cached = self._shard_cache.get(shard_idx)
            if cached is not None:
                self._shard_cache.move_to_end(shard_idx)
        if cached is not None:
            self._schedule_prefetch(shard_idx)
            return cached

        self._prefetch_misses += 1
        payload = torch.load(
            self.shard_files[shard_idx], map_location="cpu", weights_only=False
        )
        records = payload["records"]
        self._store_shard(shard_idx, records)
        with self._cache_lock:
            self._prefetch_requested.discard(shard_idx)
        self._schedule_prefetch(shard_idx)
        return records

    @staticmethod
    def _deserialize_item(raw: dict[str, Any] | None) -> PairItem | None:
        if raw is None:
            return None
        modality = raw["type"]
        if modality == "text" and "tokens" in raw:
            value = raw["tokens"]
        elif modality == "text":
            value = raw["value"]
        elif "tensor" in raw:
            value = raw["tensor"]
        else:
            value = raw.get("value")
        return PairItem(modality=modality, value=value)

    def __len__(self) -> int:
        return self.total_rows

    def __getitem__(self, idx: int) -> TrainRecord:
        if idx < 0 or idx >= self.total_rows:
            raise IndexError(idx)
        shard_idx = bisect_right(self.shard_offsets, idx)
        shard_start = 0 if shard_idx == 0 else self.shard_offsets[shard_idx - 1]
        local_idx = idx - shard_start
        raw = self._load_shard(shard_idx)[local_idx]
        return TrainRecord(
            query=self._deserialize_item(raw["query"]),
            positive=self._deserialize_item(raw["positive"]),
            negative=self._deserialize_item(raw.get("negative")),
        )

    def get_prefetch_stats(self) -> dict[str, int]:
        with self._cache_lock:
            return {
                "cache_size": len(self._shard_cache),
                "cache_limit": self.shard_cache_limit,
                "prefetch_shards": self.prefetch_shards,
                "prefetch_hits": self._prefetch_hits,
                "prefetch_misses": self._prefetch_misses,
                "prefetch_pending": len(self._prefetch_requested),
            }

    def close(self) -> None:
        self._prefetch_stop.set()
        if self._prefetch_thread is not None and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=1.0)
        self._prefetch_thread = None
        self._prefetch_queue = None

    def __del__(self):
        self.close()


class SequentialShardDataset:
    def __init__(
        self,
        cache_dir: str,
        shuffle: bool = True,
        rank: int = 0,
        world_size: int = 1,
        prefetch_shards: int = 2,
        shard_cache_limit: int = 4,
    ) -> None:
        self.cache_dir = cache_dir
        self.shuffle = shuffle
        self.rank = rank
        self.world_size = max(world_size, 1)
        self.prefetch_shards = max(int(prefetch_shards), 0)
        self.shard_cache_limit = max(int(shard_cache_limit), 1)

        self.metadata = self._load_metadata()
        self.shard_files = self._discover_shards()
        self.shard_sizes = self._resolve_shard_sizes()
        self.total_rows = sum(self.shard_sizes)
        self.target_shard_size = int(
            self.metadata.get("shard_size") or max(self.shard_sizes)
        )

        self._shard_cache: OrderedDict[int, list[dict[str, Any]]] = OrderedDict()
        self._cache_lock = threading.Lock()
        self._prefetch_queue = None
        self._prefetch_thread = None
        self._prefetch_stop = threading.Event()
        self._prefetch_requested: set[int] = set()
        self._prefetch_hits = 0
        self._prefetch_misses = 0

        self._all_shard_indices = list(range(len(self.shard_files)))
        self._local_shard_indices: list[int] = []
        self.current_local_shard_pos = -1
        self.current_records: list[dict[str, Any]] | None = None

    def _load_metadata(self) -> dict[str, Any]:
        metadata_path = os.path.join(self.cache_dir, "metadata.json")
        if not os.path.exists(metadata_path):
            return {}
        with open(metadata_path, encoding="utf-8") as handle:
            return json.load(handle)

    def _discover_shards(self) -> list[str]:
        if not os.path.isdir(self.cache_dir):
            raise FileNotFoundError(f"Cache directory not found: {self.cache_dir}")
        shards: list[str] = []
        for name in sorted(os.listdir(self.cache_dir)):
            if not (name.startswith("shard_") and name.endswith(".pt")):
                continue
            shards.append(os.path.join(self.cache_dir, name))
        if not shards:
            raise ValueError(f"No cache shards found under {self.cache_dir}")
        return shards

    def _resolve_shard_sizes(self) -> list[int]:
        num_shards = len(self.shard_files)
        metadata_num_shards = self.metadata.get("num_shards")
        metadata_num_records = self.metadata.get("num_records")
        shard_size = self.metadata.get("shard_size")

        if (
            isinstance(metadata_num_shards, int)
            and isinstance(metadata_num_records, int)
            and isinstance(shard_size, int)
            and metadata_num_shards == num_shards
            and metadata_num_records > 0
            and shard_size > 0
        ):
            shard_sizes = [shard_size] * num_shards
            shard_sizes[-1] = metadata_num_records - shard_size * max(num_shards - 1, 0)
            return shard_sizes

        shard_sizes: list[int] = []
        for shard_path in self.shard_files:
            payload = torch.load(shard_path, map_location="cpu", weights_only=False)
            records = payload.get("records")
            if not isinstance(records, list):
                raise ValueError(f"Invalid shard format in {shard_path}")
            shard_sizes.append(len(records))
        return shard_sizes

    @staticmethod
    def _deserialize_item(raw: dict[str, Any] | None) -> PairItem | None:
        if raw is None:
            return None
        modality = raw["type"]
        if modality == "text" and "tokens" in raw:
            value = raw["tokens"]
        elif modality == "text":
            value = raw["value"]
        elif "tensor" in raw:
            value = raw["tensor"]
        else:
            value = raw.get("value")
        return PairItem(modality=modality, value=value)

    def _store_shard(self, shard_idx: int, records: list[dict[str, Any]]) -> None:
        with self._cache_lock:
            self._shard_cache[shard_idx] = records
            self._shard_cache.move_to_end(shard_idx)
            while len(self._shard_cache) > self.shard_cache_limit:
                self._shard_cache.popitem(last=False)

    def _ensure_prefetch_thread(self) -> None:
        if self.prefetch_shards <= 0:
            return
        if self._prefetch_thread is not None and self._prefetch_thread.is_alive():
            return
        self._prefetch_stop.clear()
        self._prefetch_queue = queue.Queue(maxsize=max(self.prefetch_shards * 2, 1))
        self._prefetch_thread = threading.Thread(
            target=self._prefetch_worker,
            daemon=True,
            name=f"sequential-shard-prefetch-{os.getpid()}",
        )
        self._prefetch_thread.start()

    def _prefetch_worker(self) -> None:
        while not self._prefetch_stop.is_set():
            try:
                shard_idx = self._prefetch_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if shard_idx is None:
                continue
            try:
                with self._cache_lock:
                    if shard_idx in self._shard_cache:
                        self._prefetch_hits += 1
                        continue
                payload = torch.load(
                    self.shard_files[shard_idx], map_location="cpu", weights_only=False
                )
                self._store_shard(shard_idx, payload["records"])
                self._prefetch_hits += 1
            finally:
                with self._cache_lock:
                    self._prefetch_requested.discard(shard_idx)

    def _stop_prefetch_thread(self) -> None:
        self._prefetch_stop.set()
        if self._prefetch_thread is not None and self._prefetch_thread.is_alive():
            self._prefetch_thread.join(timeout=1.0)
        self._prefetch_thread = None
        self._prefetch_queue = None

    def _schedule_prefetch_from_position(self, local_pos: int) -> None:
        if self.prefetch_shards <= 0:
            return
        self._ensure_prefetch_thread()
        if self._prefetch_queue is None:
            return
        for next_pos in range(
            local_pos + 1,
            min(len(self._local_shard_indices), local_pos + 1 + self.prefetch_shards),
        ):
            shard_idx = self._local_shard_indices[next_pos]
            with self._cache_lock:
                if (
                    shard_idx in self._shard_cache
                    or shard_idx in self._prefetch_requested
                ):
                    continue
                self._prefetch_requested.add(shard_idx)
            try:
                self._prefetch_queue.put_nowait(shard_idx)
            except queue.Full:
                with self._cache_lock:
                    self._prefetch_requested.discard(shard_idx)
                break

    def _build_local_shard_order(self, epoch: int) -> list[int]:
        shard_indices = list(self._all_shard_indices)
        if self.shuffle:
            random.Random(42 + epoch).shuffle(shard_indices)
        local_shards = shard_indices[self.rank :: self.world_size]
        max_shards = math.ceil(len(shard_indices) / self.world_size)
        if not local_shards:
            raise ValueError(
                f"Rank {self.rank} received no shards from {self.cache_dir}"
            )
        while len(local_shards) < max_shards:
            local_shards.append(local_shards[len(local_shards) % len(local_shards)])
        return local_shards

    def _load_records_for_shard(self, shard_idx: int) -> list[dict[str, Any]]:
        cached = None
        with self._cache_lock:
            cached = self._shard_cache.get(shard_idx)
            if cached is not None:
                self._shard_cache.move_to_end(shard_idx)
        if cached is not None:
            return cached

        self._prefetch_misses += 1
        payload = torch.load(
            self.shard_files[shard_idx], map_location="cpu", weights_only=False
        )
        records = payload["records"]
        self._store_shard(shard_idx, records)
        with self._cache_lock:
            self._prefetch_requested.discard(shard_idx)
        return records

    def _pad_records(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if len(records) >= self.target_shard_size:
            return records
        repeat = math.ceil(self.target_shard_size / len(records))
        return (records * repeat)[: self.target_shard_size]

    def reset(self, epoch: int) -> bool:
        self._stop_prefetch_thread()
        self._local_shard_indices = self._build_local_shard_order(epoch)
        self.current_local_shard_pos = -1
        self.current_records = None
        with self._cache_lock:
            self._prefetch_requested.clear()
        if self.prefetch_shards > 0:
            self._ensure_prefetch_thread()
        return self.next_shard()

    def next_shard(self) -> bool:
        self.current_local_shard_pos += 1
        if self.current_local_shard_pos >= len(self._local_shard_indices):
            self.current_records = None
            return False
        shard_idx = self._local_shard_indices[self.current_local_shard_pos]
        records = self._load_records_for_shard(shard_idx)
        self.current_records = self._pad_records(records)
        self._schedule_prefetch_from_position(self.current_local_shard_pos)
        return True

    def __len__(self) -> int:
        return len(self.current_records or [])

    def __getitem__(self, idx: int) -> TrainRecord:
        if self.current_records is None:
            raise IndexError(idx)
        raw = self.current_records[idx]
        return TrainRecord(
            query=self._deserialize_item(raw["query"]),
            positive=self._deserialize_item(raw["positive"]),
            negative=self._deserialize_item(raw.get("negative")),
        )

    def estimated_num_batches(self, batch_size: int, drop_last: bool) -> int:
        shard_batches = (
            self.target_shard_size // batch_size
            if drop_last
            else math.ceil(self.target_shard_size / batch_size)
        )
        return shard_batches * max(len(self._build_local_shard_order(0)), 1)

    def get_prefetch_stats(self) -> dict[str, int]:
        with self._cache_lock:
            return {
                "cache_size": len(self._shard_cache),
                "cache_limit": self.shard_cache_limit,
                "prefetch_shards": self.prefetch_shards,
                "prefetch_hits": self._prefetch_hits,
                "prefetch_misses": self._prefetch_misses,
                "prefetch_pending": len(self._prefetch_requested),
                "local_shards": len(self._local_shard_indices),
                "target_shard_size": self.target_shard_size,
            }

    def close(self) -> None:
        self._stop_prefetch_thread()

    def __del__(self):
        self.close()
