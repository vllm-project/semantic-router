"""Public durable-filesystem boundary for evaluation artifact stores."""

from __future__ import annotations

from cli.evaluation.private_filesystem_publication import (
    PrivateFilesystemPublicationPrimitives,
)


class DurablePrivateFilesystem(PrivateFilesystemPublicationPrimitives):
    """Anchor all artifact operations beneath one immutable root descriptor."""
