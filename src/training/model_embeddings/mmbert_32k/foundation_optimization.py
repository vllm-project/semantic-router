"""Pure scheduling helpers for exact gradient-accumulation windows."""

from __future__ import annotations


def optimizer_steps_per_epoch(num_batches: int, accumulation_steps: int) -> int:
    """Return ceil(num_batches / accumulation_steps)."""
    if num_batches <= 0 or accumulation_steps <= 0:
        raise ValueError("num_batches and accumulation_steps must be positive")
    return (num_batches + accumulation_steps - 1) // accumulation_steps


def accumulation_window_size(
    batch_index: int, num_batches: int, accumulation_steps: int
) -> int:
    """Return the actual full-or-tail window divisor for one zero-based batch."""
    if batch_index < 0 or batch_index >= num_batches:
        raise ValueError("batch_index is outside the epoch")
    optimizer_steps_per_epoch(num_batches, accumulation_steps)
    window_start = (batch_index // accumulation_steps) * accumulation_steps
    return min(accumulation_steps, num_batches - window_start)


def should_optimizer_step(
    batch_index: int, num_batches: int, accumulation_steps: int
) -> bool:
    """Flush complete windows and the final partial window."""
    accumulation_window_size(batch_index, num_batches, accumulation_steps)
    return (batch_index + 1) % accumulation_steps == 0 or batch_index + 1 == num_batches
