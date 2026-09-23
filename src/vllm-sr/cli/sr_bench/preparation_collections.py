"""Prepare a verified benchmark collection without planning or running models."""

from __future__ import annotations

from .datasets import DatasetReader
from .preparation_runtime import PreparationError

DEFAULT_SEED = 20260918
SELECTION_ERRORS = {
    "source_conflict": "Selected benchmarks have conflicting frozen sources. Resolve the sources before preparing a collection.",
    "seed_conflict": "Selected datasets and the requested collection must use one common seed.",
    "source_size_limit": "A selected source exceeds the supported verification limit.",
    "scan_budget_exhausted": "Dataset selection could not verify the selected sources within its scan budget.",
}


class CollectionError(PreparationError):
    def __init__(self, code, message=None):
        self.code = code
        super().__init__(
            message
            or SELECTION_ERRORS.get(code, "Dataset collection could not be verified.")
        )


def pending_items(request):
    return [
        {
            "benchmark": benchmark,
            "status": "queued",
            "phase": "queued",
            "reused": False,
            "source_ids": [],
        }
        for benchmark in request["benchmarks"]
    ]


def _selection(reader, request):
    try:
        selection = reader.selection(request["profile"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise CollectionError("selection_unverified") from exc
    entries = {item["id"]: item for item in selection["benchmarks"]}
    return selection, [entries[benchmark] for benchmark in request["benchmarks"]]


def _require_running(stopping):
    if stopping.is_set():
        raise CollectionError(
            "service_stopped",
            "Service stopped during collection preparation. Retry explicitly.",
        )


def execute_collection(request, store, executor, update, stopping):
    """Pin reusable sources, prepare missing members serially, then verify composition.

    Only `not_prepared` permits acquisition. Ambiguous or unverified inventory
    never becomes an implicit download or an arbitrary choice of source revision.
    """
    reader = DatasetReader(store)
    items = pending_items(request)
    update(phase="resolving_datasets", items=items)
    _require_running(stopping)
    selection, entries = _selection(reader, request)
    blocked = None
    for item, entry in zip(items, entries, strict=True):
        if entry["eligible"]:
            item.update(
                status="completed",
                phase="reused",
                reused=True,
                source_ids=entry["source_ids"],
            )
        elif entry["reason_code"] != "not_prepared":
            code = entry["reason_code"] or "selection_unverified"
            item.update(
                status="failed",
                phase="failed",
                error_code=code,
                error=SELECTION_ERRORS.get(
                    code, "Dataset selection could not be verified."
                ),
            )
            blocked = blocked or code
    seed = request.get(
        "seed", selection["seed"] if selection["seed"] is not None else DEFAULT_SEED
    )
    # Selection's seed is collection-wide. When existing benchmarks disagree,
    # even an entirely missing requested subset cannot create a valid collection
    # by inventing the default seed and adding more incompatible sources.
    seed_conflict = any(
        entry.get("reason_code") == "seed_conflict" for entry in selection["benchmarks"]
    )
    if seed_conflict or (selection["seed"] is not None and seed != selection["seed"]):
        blocked = "seed_conflict"
        for item in items:
            item.update(
                status="failed",
                phase="failed",
                error_code=blocked,
                error=SELECTION_ERRORS[blocked],
            )
    update(seed=seed, items=items)
    if blocked:
        raise CollectionError(blocked)

    for item in items:
        _require_running(stopping)
        if item["reused"]:
            continue
        item.update(status="running", phase="checking_dependencies")
        update(phase=item["phase"], items=items)

        def progress(phase, member=item):
            member["phase"] = phase
            update(phase=phase, items=items)

        try:
            dataset = executor(
                {
                    "benchmark": item["benchmark"],
                    "profile": request["profile"],
                    "seed": seed,
                },
                store,
                progress,
                stopping,
            )
            _require_running(stopping)
            # Do not trust a downloader receipt alone. Verify the registered
            # bytes and requested profile before admitting this member.
            verified = reader.compose([dataset["id"]], [item["benchmark"]])
            if (
                verified["profile"] != request["profile"]
                or verified["seed"] != seed
                or verified.get("custom_subset")
            ):
                raise CollectionError("prepared_dataset_mismatch")
            item.update(
                status="completed",
                phase="completed",
                source_ids=[verified["id"]],
                dataset_id=verified["id"],
            )
            update(items=items)
        except Exception as exc:
            code = (
                exc.code if isinstance(exc, CollectionError) else "preparation_failed"
            )
            message = (
                str(exc)
                if isinstance(exc, PreparationError)
                else "Dataset preparation failed. Check the worker and retry explicitly."
            )
            item.update(status="failed", phase="failed", error_code=code, error=message)
            update(items=items)
            raise CollectionError(code, message) from exc

    _require_running(stopping)
    update(phase="composing")
    # Re-check selection after acquisition, including concurrent local imports.
    # Composition below uses the original pinned IDs, never a new revision pick.
    current, entries = _selection(reader, request)
    for entry in entries:
        if not entry["eligible"]:
            raise CollectionError(entry["reason_code"] or "selection_unverified")
    if current["seed"] != seed:
        raise CollectionError("seed_conflict")
    identifiers = list(
        dict.fromkeys(source for item in items for source in item["source_ids"])
    )
    try:
        dataset = reader.compose(identifiers, request["benchmarks"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise CollectionError("composition_unverified") from exc
    _require_running(stopping)
    if (
        dataset["profile"] != request["profile"]
        or dataset["seed"] != seed
        or dataset.get("custom_subset")
    ):
        raise CollectionError("prepared_dataset_mismatch")
    return dataset
