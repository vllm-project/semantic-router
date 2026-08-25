"""Store-specific local runtime default tests."""

from cli.runtime_stack import resolve_runtime_stack
from cli.service_defaults import inject_local_store_runtime_defaults
from cli.storage_secrets import POSTGRES_PASSWORD_PLACEHOLDER


def test_populates_milvus_connection():
    config = {
        "version": "v0.3",
        "listeners": [{"name": "http-8899", "address": "0.0.0.0", "port": 8899}],
        "global": {
            "stores": {"response_cache": {"enabled": True, "backend_type": "milvus"}}
        },
    }
    changed = inject_local_store_runtime_defaults(config, resolve_runtime_stack())
    cache = config["global"]["stores"]["response_cache"]
    milvus = cache["milvus"]
    connection = milvus["connection"]
    assert changed is True
    assert cache["backend_type"] == "milvus"
    assert connection["host"] == "vllm-sr-milvus"
    assert connection["port"] == 19530
    assert connection["database"] == "default"
    assert connection["timeout"] == 30
    assert milvus["collection"]["name"] == "semantic_cache"
    assert milvus["collection"]["vector_field"]["dimension"] == 768
    assert milvus["search"]["params"]["ef"] == 64
    assert milvus["search"]["topk"] == 10
    assert milvus["development"]["auto_create_collection"] is True


def test_populates_vector_store_metadata_postgres():
    config = _vector_store_config()
    changed = inject_local_store_runtime_defaults(config, resolve_runtime_stack())
    metadata = config["global"]["stores"]["vector_store"]["metadata_postgres"]
    assert changed is True
    assert metadata["host"] == "vllm-sr-postgres"
    assert metadata["port"] == 5432
    assert metadata["database"] == "vsr"
    assert metadata["user"] == "router"
    assert metadata["password"] == POSTGRES_PASSWORD_PLACEHOLDER
    assert metadata["ssl_mode"] == "disable"


def test_backfills_vector_store_metadata_postgres():
    config = _vector_store_config(metadata_postgres={"host": "", "database": "custom"})
    changed = inject_local_store_runtime_defaults(config, resolve_runtime_stack())
    metadata = config["global"]["stores"]["vector_store"]["metadata_postgres"]
    assert changed is True
    assert metadata["host"] == "vllm-sr-postgres"
    assert metadata["database"] == "custom"
    assert metadata["user"] == "router"


def test_preserves_user_milvus_config():
    config = {
        "version": "v0.3",
        "global": {
            "stores": {
                "response_cache": {
                    "enabled": True,
                    "backend_type": "milvus",
                    "milvus": {
                        "connection": {
                            "host": "custom-milvus-host",
                            "port": 19531,
                        },
                        "collection": {"name": "my_custom_collection"},
                    },
                }
            }
        },
    }
    changed = inject_local_store_runtime_defaults(config, resolve_runtime_stack())
    milvus = config["global"]["stores"]["response_cache"]["milvus"]
    connection = milvus["connection"]
    assert changed is True
    assert connection["host"] == "custom-milvus-host"
    assert connection["port"] == 19531
    assert connection["database"] == "default"
    assert connection["timeout"] == 30
    assert milvus["collection"]["name"] == "my_custom_collection"


def _vector_store_config(metadata_postgres=None):
    vector_store = {"enabled": True, "metadata_store": "postgres"}
    if metadata_postgres is not None:
        vector_store["metadata_postgres"] = metadata_postgres
    return {
        "version": "v0.3",
        "global": {
            "stores": {
                "response_cache": {"enabled": False},
                "vector_store": vector_store,
            }
        },
    }
