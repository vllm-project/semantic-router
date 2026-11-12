"""Serve the ASGI app over TCP or a Unix domain socket."""

from __future__ import annotations

import logging
import os
import socket
import stat
import time
from pathlib import Path
from typing import Any

from ..config import ServeConfig
from ..runtime import Runtime
from .app import create_app

log = logging.getLogger("vllm_srun")
# The exit status of a server that stopped with a forward still running (a device error exits with 3).
ABANDONED_EXIT = 4


def prepare_socket(path: str) -> None:
    """Create the socket's directory (0700 when new) and remove a stale socket file."""
    socket_path = Path(path)
    parent = socket_path.parent
    if not parent.exists():
        parent.mkdir(parents=True, mode=0o700)
    if socket_path.exists() or socket_path.is_symlink():
        if not stat.S_ISSOCK(socket_path.lstat().st_mode):
            raise FileExistsError(f"{path} exists and is not a socket")
        socket_path.unlink()


def stop(runtime: Runtime, timeout: float) -> None:
    """Stop the runtime; exit the process at once if a forward still runs after ``timeout`` seconds.

    Python's own exit would wait for that forward: it joins the device's
    threads, and a GPU runtime's teardown waits for the device's work.
    """
    abandoned = runtime.stop(timeout)
    if abandoned:
        log.error(
            "exiting with a forward still running after the shutdown grace "
            "(--shutdown-grace-seconds): %s",
            "; ".join(abandoned),
        )
        os._exit(ABANDONED_EXIT)


def serve(config: ServeConfig) -> None:
    import uvicorn

    runtime = Runtime(config)
    runtime.start(background=True)
    app = create_app(runtime)
    grace = config.shutdown_grace_seconds

    def cleanup(timeout: float) -> None:
        if config.uds and os.path.exists(config.uds):
            os.unlink(config.uds)
        stop(runtime, timeout)

    class Server(uvicorn.Server):
        # uvicorn re-raises the stop signal after serve() returns, so clean up inside shutdown.
        async def shutdown(self, sockets: list[socket.socket] | None = None) -> None:
            deadline = time.monotonic() + grace
            await super().shutdown(sockets=sockets)
            cleanup(max(0.0, deadline - time.monotonic()))

    options: dict[str, Any] = {
        "log_level": config.log_level,
        "access_log": False,
        "timeout_keep_alive": 30,
        "timeout_graceful_shutdown": grace,
        "lifespan": "off",
    }
    if config.uds:
        prepare_socket(config.uds)
        server_config = uvicorn.Config(app, uds=config.uds, **options)
        log.info("listening on unix:%s", config.uds)
    else:
        server_config = uvicorn.Config(
            app, host=config.host, port=config.port, **options
        )
        log.info("listening on http://%s:%d", config.host, config.port)
    server = Server(server_config)
    try:
        server.run()
    finally:
        cleanup(grace)
