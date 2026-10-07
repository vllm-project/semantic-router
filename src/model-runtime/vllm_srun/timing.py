"""Where a request's server time goes: the ``Server-Timing`` header of its response.

The HTTP layer times reading the body (``parse``) and writing the answer
(``serialize``), the runtime times planning (``tokenize``: validation,
rendering and tokenization) and assembling answers (``post``), and the
scheduler times each job group's forwards (``forward``, readout included).
``queue`` is the rest of the time a group waited for its model: before its
first forward and between its forwards. ``total`` runs from the handler's
start until the response is ready to send, so ``total`` minus the phases is
the server's own overhead (event-loop and thread hand-offs, cache lookups).

A bundle whose tasks go to several models reports the queue, forward and
post of the group that was answered last, the one the response waited for.
"""

from __future__ import annotations

__all__ = ["RunTiming", "ServerTiming"]


class RunTiming:
    """A job group's forwards: their seconds and when the last one ended (``time.monotonic``).

    Only the runner that owns the group's model writes it, before it answers
    the group's jobs, so the submitter reads it once their futures are done.
    """

    __slots__ = ("batch", "done", "forward")

    def __init__(self) -> None:
        self.forward = 0.0
        self.done = 0.0
        self.batch = -1

    def ran(self, batch: int, seconds: float, ended: float) -> None:
        """Count one forward that ran items of the group, once however many of its jobs it held."""
        if batch != self.batch:
            self.batch = batch
            self.forward += seconds
            self.done = ended


class ServerTiming:
    """One request's phases in seconds."""

    __slots__ = ("forward", "parse", "post", "queue", "serialize", "tokenize")

    def __init__(self) -> None:
        self.parse = 0.0
        self.tokenize = 0.0
        self.queue = 0.0
        self.forward = 0.0
        self.post = 0.0
        self.serialize = 0.0

    def ran(
        self, run: RunTiming, submitted: float, answered: float, finished: float
    ) -> None:
        """Take a job group's queue, forward and post (``time.monotonic`` times).

        ``submitted`` is when the group reached its scheduler, ``answered``
        when its results were back and ``finished`` when its answers were
        assembled. Groups record in the order they finish, so the last one,
        which the response waited for, stays.
        """
        waited = (run.done or answered) - submitted
        self.forward = run.forward
        self.queue = max(0.0, waited - run.forward)
        self.post = finished - answered

    def header(self, total: float) -> str:
        """The ``Server-Timing`` value: every phase and ``total``, in milliseconds."""
        return (
            f"parse;dur={self.parse * 1e3:.3f}, "
            f"tokenize;dur={self.tokenize * 1e3:.3f}, "
            f"queue;dur={self.queue * 1e3:.3f}, "
            f"forward;dur={self.forward * 1e3:.3f}, "
            f"post;dur={self.post * 1e3:.3f}, "
            f"serialize;dur={self.serialize * 1e3:.3f}, "
            f"total;dur={total * 1e3:.3f}"
        )
