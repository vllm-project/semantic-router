"""Durable transport activity, distinct from final usage and billing evidence."""

import threading
import time
from datetime import datetime, timezone

CHECKPOINT_SECONDS = 1.0


def _timestamp(seconds):
    return datetime.fromtimestamp(seconds, timezone.utc).isoformat()


class CallActivity:
    def __init__(self, publish, *, clock=time.monotonic, wall_clock=time.time):
        self.publish = publish
        self.clock = clock
        self.origin = clock()
        self.wall_origin = wall_clock()
        self.lock = threading.Lock()
        self.last_checkpoint = self.origin
        self.published_bytes = 0
        self.state = {
            "phase": "preparing",
            "phase_started_at": _timestamp(self.wall_origin),
            "last_activity_at": None,
            "received_bytes": 0,
            "updated_at": _timestamp(self.wall_origin),
        }

    def snapshot(self):
        with self.lock:
            return dict(self.state)

    def _at(self, elapsed):
        return _timestamp(self.wall_origin + elapsed - self.origin)

    def waiting(self):
        with self.lock:
            when = self.clock()
            self.state.update(phase="waiting", phase_started_at=self._at(when))
            self._publish(when)

    def received(self, total, observed_at, *, force=False):
        with self.lock:
            if total < self.state["received_bytes"]:
                return
            if total == self.state["received_bytes"]:
                if force and total > self.published_bytes:
                    self._publish(self.clock())
                return
            first = self.state["phase"] != "streaming"
            if first:
                self.state.update(
                    phase="streaming", phase_started_at=self._at(observed_at)
                )
            self.state.update(
                received_bytes=total, last_activity_at=self._at(observed_at)
            )
            if (
                first
                or force
                or observed_at - self.last_checkpoint >= CHECKPOINT_SECONDS
            ):
                self._publish(self.clock())

    def _publish(self, when):
        self.state["updated_at"] = self._at(when)
        self.publish(dict(self.state))
        self.last_checkpoint = when
        self.published_bytes = self.state["received_bytes"]
