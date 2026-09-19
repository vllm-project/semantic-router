"""Judging rows through the Claude API, with the client passed in."""

import sys
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from audit_lib.checkpoint import count_judged, read_checkpoint, select_ids
from audit_lib.constants import MAX_API_ATTEMPTS
from audit_lib.judgment import build_user_message, parse_lines, save_judgments

USAGE_KEYS = (
    "input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
)


@dataclass(frozen=True)
class ApiJudgeSettings:
    """How the API judging run behaves.

    Attributes:
        model: Model id.
        effort: Effort level, or None to omit it.
        batch_size: Rows per request.
        max_tokens: Output token limit per request.
        retries: How many times to re-ask for ids missing from a reply.
        limit: Stop after this many rows, or None for all.
        rejudge: Whether to run the self-consistency pass.
        seed: Seed for the re-judge sample.
        overwrite: Whether to re-judge rows already saved.
    """

    model: str = "claude-sonnet-5"
    effort: str | None = "medium"
    batch_size: int = 50
    max_tokens: int = 16000
    retries: int = 2
    limit: int | None = None
    rejudge: bool = False
    seed: int = 0
    overwrite: bool = False


def make_client():
    """Create the Anthropic client. The SDK is imported here so only `api` needs it."""
    import anthropic  # noqa: PLC0415  (lazy: only the api command needs the SDK)

    return anthropic.Anthropic()


def retryable_errors() -> tuple:
    """Return the SDK errors worth retrying, or an empty tuple if the SDK is missing."""
    try:
        import anthropic  # noqa: PLC0415  (lazy, as above)

        return anthropic.RateLimitError, anthropic.APIConnectionError
    except ImportError:
        return ()


def call_model(
    client,
    model: str,
    system_text: str,
    user_text: str,
    *,
    effort: str | None,
    max_tokens: int,
    retryable: tuple = (),
    sleep: Callable[[float], None] = time.sleep,
):
    """Send one batch to the model and return its reply.

    Retries the errors in `retryable` with a growing wait.

    Args:
        client: The Anthropic client.
        model: Model id.
        system_text: System prompt, the rubric.
        user_text: The batch, from build_user_message.
        effort: Effort level, or None to omit it.
        max_tokens: Output token limit.
        retryable: Exception types that are retried.
        sleep: Function that waits, replaceable in tests.

    Returns:
        (reply text, usage).

    Raises:
        RuntimeError: If the reply hit max_tokens.
    """
    kwargs = {
        "model": model,
        "max_tokens": max_tokens,
        "system": [
            {
                "type": "text",
                "text": system_text,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "messages": [{"role": "user", "content": user_text}],
    }
    if effort:
        kwargs["output_config"] = {"effort": effort}
    for attempt in range(MAX_API_ATTEMPTS):
        try:
            resp = client.messages.create(**kwargs)
            break
        except retryable as e:  # 429 / connection errors (SDK already retried twice)
            if attempt == MAX_API_ATTEMPTS - 1:
                raise
            wait = 20 * (attempt + 1)
            print(
                f"  retryable error ({type(e).__name__}); sleeping {wait}s",
                file=sys.stderr,
            )
            sleep(wait)
    if resp.stop_reason == "max_tokens":
        raise RuntimeError(
            "response hit max_tokens; lower --batch-size or raise --max-tokens"
        )
    text = "".join(b.text for b in resp.content if b.type == "text")
    return text, resp.usage


def judge_via_api(
    client,
    rows: list[dict],
    *,
    split: str,
    system_text: str,
    rubric: str,
    settings: ApiJudgeSettings,
    checkpoint: Path,
    rejudge_checkpoint: Path,
    target: Path,
    retryable: tuple = (),
    out: Callable[[str], None] = print,
    err: Callable[[str], None] = lambda message: print(message, file=sys.stderr),
) -> set[int]:
    """Judge rows through the API until the split is done or the limit is reached.

    Progress is written to the checkpoint after every reply, so a crash loses nothing
    and a rerun resumes where it stopped.

    Args:
        client: The Anthropic client.
        rows: The split's rows.
        split: Split name.
        system_text: System prompt, the rubric.
        rubric: Rubric hash, recorded on every record.
        settings: Model and batching settings.
        checkpoint: Primary checkpoint file.
        rejudge_checkpoint: Self-consistency checkpoint file.
        target: The checkpoint this run writes to.
        retryable: Exception types that are retried.
        out: Receives progress lines.
        err: Receives warnings.

    Returns:
        Ids that could not be judged even after retries.
    """
    skipped: set[int] = set()
    processed = 0
    totals: Counter = Counter()
    while settings.limit is None or processed < settings.limit:
        n = (
            settings.batch_size
            if settings.limit is None
            else min(settings.batch_size, settings.limit - processed)
        )
        ids = select_ids(
            split,
            rows,
            read_checkpoint(checkpoint),
            read_checkpoint(rejudge_checkpoint),
            n,
            rejudge=settings.rejudge,
            seed=settings.seed,
            exclude=skipped,
        )
        if not ids:
            break
        missing = list(ids)
        for _attempt in range(1 + settings.retries):
            text, usage = call_model(
                client,
                settings.model,
                system_text,
                build_user_message(rows, missing),
                effort=settings.effort,
                max_tokens=settings.max_tokens,
                retryable=retryable,
            )
            parsed, errors, _ = parse_lines(text, allowed=set(missing))
            records = save_judgments(
                parsed,
                rows,
                split=split,
                judge=settings.model,
                rubric=rubric,
                path=target,
                existing=read_checkpoint(target),
                overwrite=settings.overwrite,
            )
            for key in USAGE_KEYS:
                totals[key] += getattr(usage, key, 0) or 0
            got = {r["id"] for r in records}
            processed += len(got)
            missing = [i for i in missing if i not in got]
            if errors:
                err(f"  {len(errors)} unparseable/unexpected lines, e.g. {errors[0]}")
            if not missing:
                break
        if missing:
            err(f"  giving up on {len(missing)} ids after retries: {missing[:10]}")
            skipped.update(missing)
        done = count_judged(read_checkpoint(target), split)
        out(f"batch done: {done}/{len(rows)} judged | tokens {dict(totals)}")
    if skipped:
        out(f"unjudged (skipped) ids: {sorted(skipped)}")
    return skipped
