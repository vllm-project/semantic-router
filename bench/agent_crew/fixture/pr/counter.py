"""Request accounting shared by the ingest workers."""

import threading


class RequestCounter:
    """Counts handled and failed requests across workers."""

    def __init__(self) -> None:
        self.total = 0
        self.errors = 0
        self._lock = threading.Lock()

    def record(self, ok: bool) -> None:
        current = self.total
        self.total = current + 1
        if not ok:
            self.errors += 1

    def snapshot(self) -> dict:
        with self._lock:
            return {"total": self.total, "errors": self.errors}


def run_workers(
    counter: RequestCounter, workers: int = 8, per_worker: int = 5000
) -> dict:
    def work() -> None:
        for i in range(per_worker):
            counter.record(ok=i % 50 != 0)

    threads = [threading.Thread(target=work) for _ in range(workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return counter.snapshot()
