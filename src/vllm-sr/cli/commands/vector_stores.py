"""Router-managed vector-store inspection."""

from __future__ import annotations

from cli.router_management_client import RouterManagementClient
from cli.terminal import echo, fields, heading

VECTOR_STORES_PATH = "/api/v1/storage/vector-stores"


def list_vector_stores(
    endpoint: str | None = None,
    timeout: int = 15,
    token_env: str = "VSR_MGMT_TOKEN",
) -> None:
    """List vector stores through the canonical Router management API."""

    client = RouterManagementClient(endpoint, timeout=timeout, token_env=token_env)
    data = client.request("GET", VECTOR_STORES_PATH).payload
    stores = data.get("data") if isinstance(data, dict) else None
    if not isinstance(stores, list):
        stores = []

    heading(f"Vector stores ({len(stores)})")
    fields((("Endpoint", client.base_url + VECTOR_STORES_PATH),))
    echo()
    _print_vector_stores(stores)


def _print_vector_stores(stores: list) -> None:
    if not stores:
        echo("No vector stores have been created.")
        return

    for index, store in enumerate(stores):
        if not isinstance(store, dict):
            continue
        if index:
            echo()
        name = str(store.get("name") or "(unnamed)")
        store_id = str(store.get("id") or "-")
        heading(name)
        details: list[tuple[str, object]] = [("ID", store_id)]
        status = store.get("status")
        if status:
            details.append(("Status", status))

        backend = store.get("backend_type")
        if backend:
            details.append(("Backend", backend))

        counts = store.get("file_counts")
        if isinstance(counts, dict):
            total = counts.get("total", 0)
            completed = counts.get("completed", 0)
            in_progress = counts.get("in_progress", 0)
            failed = counts.get("failed", 0)
            details.append(
                (
                    "Files",
                    f"{total} total ({completed} completed, "
                    f"{in_progress} in progress, {failed} failed)",
                )
            )
        fields(details)
