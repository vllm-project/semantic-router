"""Request dataclass and lifecycle states."""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


class RequestState(Enum):
    PENDING    = auto()   # arrived, waiting in global queue
    QUEUED     = auto()   # assigned to a pool instance queue
    PREFILLING = auto()   # actively being prefilled
    DECODING   = auto()   # prefill done, generating tokens
    DONE       = auto()   # all output tokens generated


@dataclass
class Request:
    """One inference request flowing through the fleet.

    Attributes
    ----------
    req_id       : globally unique request identifier
    arrival_time : simulation time when request arrived (s)
    l_in         : number of input (prompt) tokens
    l_out        : number of output tokens to generate
    category     : content category ("prose", "code", "rag", "mixed")
                   used by C&R router for compression safety decisions
    pool_id      : which pool this request was routed to (set by router)
    instance_id  : which instance within the pool (set by pool)
    compressed   : True if this request was compressed by C&R
    orig_l_in    : original l_in before compression (if compressed)

    Timestamps (set by simulation engine as request progresses):
    queue_time   : time entered instance queue
    start_time   : time prefill began
    first_token  : time first output token was generated (TTFT)
    end_time     : time last output token was generated
    """
    req_id: int
    arrival_time: float
    l_in: int
    l_out: int
    category: str = "prose"
    model_id: Optional[str] = None  # desired model/pool; used by ModelRouter

    # routing metadata (filled by router)
    pool_id: Optional[str] = None
    instance_id: Optional[int] = None
    compressed: bool = False
    orig_l_in: Optional[int] = None

    # timing (filled by simulation engine)
    queue_time: Optional[float] = None
    start_time: Optional[float] = None
    first_token_time: Optional[float] = None
    end_time: Optional[float] = None          # slot freed (= start + s_eff)
    physical_end_time: Optional[float] = None # last token generated (= start + s_raw)

    # reliability
    preempted: bool = False

    # state
    state: RequestState = field(default=RequestState.PENDING, init=False)

    @property
    def ttft(self) -> Optional[float]:
        """Time-to-first-token (s). None if not yet done."""
        if self.first_token_time is not None and self.arrival_time is not None:
            return self.first_token_time - self.arrival_time
        return None

    @property
    def e2e_latency(self) -> Optional[float]:
        """End-to-end latency (s). None if not yet done."""
        t = self.physical_end_time or self.end_time
        if t is not None:
            return t - self.arrival_time
        return None

    @property
    def tpot(self) -> Optional[float]:
        """Time-per-output-token (s/tok). Requires l_out > 1."""
        t = self.physical_end_time or self.end_time
        if t is not None and self.first_token_time is not None and self.l_out > 1:
            return (t - self.first_token_time) / (self.l_out - 1)
        return None

    @property
    def queue_wait(self) -> Optional[float]:
        """Time spent waiting in queue before service started (s)."""
        if self.start_time is not None and self.queue_time is not None:
            return self.start_time - self.queue_time
        return None

    def total_tokens(self) -> int:
        return self.l_in + self.l_out

    @property
    def l_total(self) -> int:
        """Alias for total_tokens(); convenience for routing logic."""
        return self.l_in + self.l_out
