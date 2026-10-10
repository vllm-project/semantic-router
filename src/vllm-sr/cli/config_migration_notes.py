"""Notes that `vllm-sr config migrate` reports for changes an operator should review."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MigrationNote:
    """One rewritten, removed or unresolved field.

    ``action_required`` marks a value the tool could not rewrite safely; the
    migrated file keeps it, and the router rejects it until the operator acts.
    """

    path: str
    message: str
    action_required: bool = False


class MigrationNotes:
    """Collects notes in the order the migration produced them."""

    def __init__(self) -> None:
        self._notes: list[MigrationNote] = []

    def changed(self, path: str, message: str) -> None:
        self._notes.append(MigrationNote(path, message))

    def action(self, path: str, message: str) -> None:
        self._notes.append(MigrationNote(path, message, action_required=True))

    def __iter__(self):
        return iter(self._notes)

    def __len__(self) -> int:
        return len(self._notes)

    def as_list(self) -> list[MigrationNote]:
        return list(self._notes)
