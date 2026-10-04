"""HTTP probes the CLI runs inside the router container.

The router image ships Python for its model runtime but no curl, so the probes
use the standard library. A bearer token is read from the container's own
environment by name and never appears in the probe's arguments.
"""

from __future__ import annotations

PROBE = """
import os, sys, urllib.request
url, timeout, token_env = sys.argv[1], float(sys.argv[2]), sys.argv[3:]
headers = {}
if token_env:
    token = os.environ.get(token_env[0], "")
    if not token:
        sys.exit(1)
    headers["Authorization"] = "Bearer " + token
try:
    urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=timeout)
except Exception:
    sys.exit(1)
"""


def router_probe_command(
    url: str, timeout: float = 5.0, token_env: str | None = None
) -> list[str]:
    """A command that exits 0 only when `url` answers with a 2xx status."""
    command = ["python3", "-c", PROBE, url, f"{max(0.001, timeout):.3f}"]
    return [*command, token_env] if token_env else command
