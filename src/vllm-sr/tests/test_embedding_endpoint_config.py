import pytest
from cli.models import EmbeddingModelsConfig


def test_embedding_endpoint_accepts_limits() -> None:
    config = EmbeddingModelsConfig(
        endpoint={
            "base_url": "https://embedding.example/v1",
            "model": "embedding-model",
            "api_key_env": "VLLM_SR_EMBEDDING_API_KEY",
            "timeout_seconds": 3600,
            "max_retries": 10,
        }
    )

    assert config.endpoint is not None
    assert config.endpoint.api_key_env == "VLLM_SR_EMBEDDING_API_KEY"
    assert config.endpoint.timeout_seconds == 3600
    assert config.endpoint.max_retries == 10


@pytest.mark.parametrize(
    "value",
    ["OPENAI_API_KEY", "VLLM_SR_LOOPER_SHARED_SECRET", "INVALID-NAME", " KEY"],
)
def test_embedding_endpoint_rejects_invalid_api_key_env(value: str) -> None:
    with pytest.raises(ValueError, match="VLLM_SR_EMBEDDING_API_KEY"):
        EmbeddingModelsConfig(endpoint={"api_key_env": value})


def test_embedding_endpoint_requires_https_for_api_key() -> None:
    with pytest.raises(ValueError, match="must use https"):
        EmbeddingModelsConfig(
            endpoint={
                "base_url": "http://embedding.example/v1",
                "api_key_env": "VLLM_SR_EMBEDDING_API_KEY",
            }
        )


@pytest.mark.parametrize(
    "base_url",
    ["ftp://embedding.example/v1", "https:///missing-host", "embedding.example/v1"],
)
def test_embedding_endpoint_rejects_invalid_scheme_or_host(base_url: str) -> None:
    with pytest.raises(
        ValueError, match="must use http or https and include a valid host"
    ) as exc_info:
        EmbeddingModelsConfig(endpoint={"base_url": base_url})

    assert base_url not in str(exc_info.value)


def test_embedding_endpoint_rejects_userinfo_without_exposing_url() -> None:
    marker = "DO-NOT-EXPOSE-URL-CONTENT"
    with pytest.raises(ValueError, match="must not include userinfo") as exc_info:
        EmbeddingModelsConfig(
            endpoint={
                "base_url": f"https://user:{marker}@embedding.example/v1?key={marker}",
            }
        )

    assert marker not in str(exc_info.value)


@pytest.mark.parametrize("separator", ["?token=", "#"])
def test_embedding_endpoint_rejects_url_components_without_exposure(
    separator: str,
) -> None:
    marker = "DO-NOT-EXPOSE-URL-CONTENT"
    with pytest.raises(
        ValueError, match="must not include query or fragment"
    ) as exc_info:
        EmbeddingModelsConfig(
            endpoint={"base_url": f"https://embedding.example/v1{separator}{marker}"}
        )

    assert marker not in str(exc_info.value)


def test_embedding_endpoint_allows_empty_api_key_env_without_https() -> None:
    config = EmbeddingModelsConfig(
        endpoint={"base_url": "http://embedding.example/v1", "api_key_env": ""}
    )

    assert config.endpoint is not None
    assert config.endpoint.api_key_env == ""


@pytest.mark.parametrize(
    ("field", "value"),
    [("timeout_seconds", 3601), ("max_retries", 11)],
)
def test_embedding_endpoint_rejects_values_above_limits(field: str, value: int) -> None:
    with pytest.raises(ValueError, match="less than or equal to"):
        EmbeddingModelsConfig(
            endpoint={
                "base_url": "https://embedding.example/v1",
                "model": "embedding-model",
                field: value,
            }
        )
