"""Dual-home the Router container across the application and data networks.

Router is the only container on both of a stack's networks: inbound traffic
(Envoy ext_proc, the Dashboard management API, the Prometheus scrape) arrives
over the application network, and every storage connection leaves over the data
network. A container runtime accepts one ``--network`` at creation, and passing
several is only supported on Podman and Docker 25 or newer, so the second one
has to be attached with ``network connect``.
"""

from __future__ import annotations


def router_data_network_commands(
    runtime: str, container_name: str, data_network_name: str, *, start_now: bool
) -> tuple[list[str], ...]:
    """Return the commands that follow Router's ``create``, in order.

    The sequence is create, connect, start, and the order is not cosmetic.
    Router opens its Postgres pool with an immediate ``PingContext``, so it
    dials the store the moment the process comes up; attaching the data network
    after the start would be a race the router loses on a fast host. The
    application network stays on the creation command instead, because that
    direction is inbound -- Envoy's ext_proc calls, the Dashboard management
    API, the Prometheus scrape -- and nothing there is dialled during startup.

    *start_now* is false in setup mode, where the container is deliberately
    left created for the activation reconciler to start later. The connect
    still happens now: a created container accepts ``network connect``, so
    whoever starts it afterwards finds both networks already in place.

    *runtime* is the resolved runtime the container was created with, not a
    freshly detected one, so a stack cannot end up half built by two runtimes.
    """

    commands = [[runtime, "network", "connect", data_network_name, container_name]]
    if start_now:
        commands.append([runtime, "start", container_name])
    return tuple(commands)
