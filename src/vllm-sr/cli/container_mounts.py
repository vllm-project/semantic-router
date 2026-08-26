"""One way to ask the container runtime what an existing container mounts.

Two callers need the same ``inspect --format '{{json .Mounts}}'`` round trip and
want opposite things from a failure, so the transport lives here while the
judgement stays with them. :func:`cli.storage_secrets.adopted_volume_name` fails
closed, because reading an unreachable daemon as "there is no container" would
mint an empty volume beside data that is still on disk;
:func:`cli.container_services.container_mount_destinations` answers "unknown"
and lets its caller keep the container. Sharing the round trip is what keeps the
two from drifting apart on the format string, the timeout, or what a reply that
is not a JSON array means.
"""

from __future__ import annotations

import json
import subprocess

from cli.container_runtime import get_container_runtime
from cli.utils import get_logger

log = get_logger(__name__)

# A local inspect answers in milliseconds. The bound is here so an unresponsive
# runtime socket cannot hang a `serve` or a `stop` indefinitely.
RUNTIME_INSPECT_TIMEOUT_SECONDS = 10

# Docker and Podman both word a missing container as "no such object".
_NO_SUCH_OBJECT = "no such object"


class ContainerMountsUnavailableError(RuntimeError):
    """The runtime did not report what a container mounts.

    ``container_absent`` separates the one failure that is really an answer --
    the container is gone -- from every failure that is not. Only a caller that
    can act on that difference should look at it.
    """

    def __init__(self, message: str, *, container_absent: bool = False) -> None:
        super().__init__(message)
        self.container_absent = container_absent


def inspect_container_mounts(container_name: str) -> list[dict]:
    """Return the ``.Mounts`` entries the runtime reports for *container_name*.

    An empty list is a normal answer, not a failure: an image that declares no
    ``VOLUME`` and takes no bind mount genuinely mounts nothing, and so does a
    runtime that renders the field as ``null``. Everything the runtime could not
    answer raises instead, so no caller can mistake a failure for an empty mount
    table -- that difference decides whether data is safe to remove.
    """

    runtime = get_container_runtime()
    try:
        result = subprocess.run(
            [runtime, "inspect", "--format", "{{json .Mounts}}", container_name],
            capture_output=True,
            text=True,
            check=True,
            timeout=RUNTIME_INSPECT_TIMEOUT_SECONDS,
        )
        mounts = json.loads(result.stdout or "null")
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr or ""
        raise ContainerMountsUnavailableError(
            stderr.strip() or str(exc),
            container_absent=_NO_SUCH_OBJECT in stderr.lower(),
        ) from exc
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError) as exc:
        raise ContainerMountsUnavailableError(str(exc)) from exc

    if not isinstance(mounts, list):
        return []
    return [mount for mount in mounts if isinstance(mount, dict)]
