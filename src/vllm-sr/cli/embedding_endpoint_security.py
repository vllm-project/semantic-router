"""Security policy for OpenAI-compatible embedding endpoint URLs."""

from urllib.parse import urlsplit


def validate_embedding_endpoint_security(
    base_url: str | None, api_key_env: str | None
) -> None:
    """Reject unsafe or ambiguous remote embedding endpoint URLs."""
    if not base_url:
        return
    try:
        parsed = urlsplit(base_url.strip())
    except ValueError as exc:
        raise ValueError("embedding endpoint base_url must be a valid URL") from exc
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        raise ValueError(
            "embedding endpoint base_url must use http or https and include a valid host"
        )
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("embedding endpoint base_url must not include userinfo")
    if "?" in base_url or "#" in base_url:
        raise ValueError(
            "embedding endpoint base_url must not include query or fragment components"
        )
    if api_key_env and parsed.scheme.lower() != "https":
        raise ValueError(
            "embedding endpoint base_url must use https when api_key_env is set"
        )
