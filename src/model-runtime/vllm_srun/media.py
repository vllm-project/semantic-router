"""Worker processes that decode and resize request media, off the threads that plan requests.

A family hands decoding (a module-level function and picklable arguments) to
``submit``. With workers it runs in one of the process's media worker
processes, so decoding and resizing several images at once neither holds the
interpreter lock the request threads and the device worker share nor waits
for the CPU device thread; without workers it runs on the calling thread.
Either way the caller gets a future. Tensors come back through PyTorch's
shared-memory pickling, so a decoded image is not copied through a pipe.

``configure`` sets the count from ``ServeConfig.media_workers`` (None picks
``auto_workers``); the pool starts on first use or ``start`` (a model that
reads media starts it as it loads) and stops with the runtime (``stop``).
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import signal
import threading
from collections.abc import Callable
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from typing import Any

log = logging.getLogger("vllm_srun")

# Automatic count: one worker per this many CPUs the process may use, at most MAX_AUTO_WORKERS.
CPUS_PER_WORKER = 16
MAX_AUTO_WORKERS = 4
# Intra-op threads of each worker: decoding parallelizes across workers, not inside one.
WORKER_THREADS = 2


class _State:
    """The configured count and the running pool, guarded by ``lock``."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.workers: int | None = None
        self.pool: ProcessPoolExecutor | None = None


_state = _State()


def auto_workers() -> int:
    """Workers for the CPUs this process may use (its container's quota): one per ``CPUS_PER_WORKER``, at most ``MAX_AUTO_WORKERS``."""
    from .accel.cpu import container_cpus

    cpus = container_cpus() or os.cpu_count() or 1
    return min(MAX_AUTO_WORKERS, cpus // CPUS_PER_WORKER)


def configure(workers: int | None) -> None:
    """Set the worker count (None: ``auto_workers``); a running pool keeps its size until ``stop``."""
    with _state.lock:
        _state.workers = auto_workers() if workers is None else workers


def workers() -> int:
    """The configured worker count (0: decode on the calling thread)."""
    with _state.lock:
        return auto_workers() if _state.workers is None else _state.workers


def _initialize() -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    import torch

    torch.set_num_threads(WORKER_THREADS)


def _ready() -> int:
    return os.getpid()


def start() -> None:
    """Start the workers, if any are configured and they are not running, without waiting for them."""
    count = workers()
    with _state.lock:
        if _state.pool is not None or count < 1:
            return
        pool = _state.pool = ProcessPoolExecutor(
            max_workers=count,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize,
        )
        for _ in range(count):
            pool.submit(_ready)
    log.info("decoding request media in %d worker processes", count)


def submit(fn: Callable[..., Any], *args: Any) -> Future[Any]:
    """``fn(*args)`` in a media worker, or on this thread when there are none; its future either way."""
    if _state.pool is None and workers() > 0:
        start()
    pool = _state.pool
    if pool is not None:
        try:
            return pool.submit(fn, *args)
        except BrokenProcessPool:
            _discard(pool)
        except RuntimeError:
            pass  # shutting down: decode here
    future: Future[Any] = Future()
    try:
        future.set_result(fn(*args))
    except BaseException as exc:  # the caller reads it from the future
        future.set_exception(exc)
    return future


def outcome(future: Future[Any], fn: Callable[..., Any], *args: Any) -> Any:
    """A ``submit`` future's result; a job whose worker died runs again on this thread."""
    try:
        return future.result()
    except BrokenProcessPool:
        _discard(_state.pool)
        return fn(*args)


def _discard(pool: ProcessPoolExecutor | None) -> None:
    """Drop a broken pool; the next ``submit`` starts a new one."""
    with _state.lock:
        if pool is None or _state.pool is not pool:
            return
        _state.pool = None
    log.warning("a media worker process died; restarting the media workers")
    pool.shutdown(wait=False, cancel_futures=True)


def stop() -> None:
    """Stop the workers and wait for them to exit; later decoding starts them again if workers are configured.

    The server re-raises its stop signal right after the runtime stops, so a
    worker still exiting then outlives it as an orphan.
    """
    with _state.lock:
        pool, _state.pool = _state.pool, None
    if pool is not None:
        pool.shutdown(wait=True, cancel_futures=True)
