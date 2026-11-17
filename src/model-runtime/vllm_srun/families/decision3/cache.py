"""Content-addressed caches of a Decision 3.0 model: processed media on the host, vision features on its device.

Both hold values computed exactly as a miss computes them, so a hit changes no answer: a processed image or video
is a function of its encoded bytes and the package's processor settings, and the vision tower's features are a
function of the pixel rows it reads, including how many copies of a request's images one tower call holds.
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from collections.abc import Hashable
from typing import Any, Generic, TypeVar

V = TypeVar("V")

# Host bytes of processed images and videos (FP32 patch rows) one model keeps.
MEDIA_CACHE_BYTES = 4 << 30
# Device bytes of vision-tower features one model keeps.
FEATURE_CACHE_BYTES = 2 << 30


def url_digest(value: Any) -> str | None:
    """SHA-256 of a media data URL's text (None for anything else), the key of its processed form."""
    if not isinstance(value, str):
        return None
    return hashlib.sha256(value.encode("utf-8", "surrogatepass")).hexdigest()


class LRUCache(Generic[V]):
    """At most ``max_bytes`` of values, least recently used first out; safe to share between threads."""

    def __init__(self, max_bytes: int):
        self.max_bytes = max_bytes
        self._entries: OrderedDict[Hashable, tuple[V, int]] = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: Hashable) -> V | None:
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                self.misses += 1
                return None
            self._entries.move_to_end(key)
            self.hits += 1
            return entry[0]

    def put(self, key: Hashable, value: V, size: int) -> None:
        if size > self.max_bytes:
            return
        with self._lock:
            old = self._entries.pop(key, None)
            if old is not None:
                self._bytes -= old[1]
            self._entries[key] = (value, size)
            self._bytes += size
            while self._bytes > self.max_bytes:
                _, (_, evicted) = self._entries.popitem(last=False)
                self._bytes -= evicted

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "entries": len(self._entries),
                "bytes": self._bytes,
                "hits": self.hits,
                "misses": self.misses,
            }
