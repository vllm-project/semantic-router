"""Public error boundary for durable evaluation artifact storage."""


class StoreError(ValueError):
    """Evaluation artifact storage rejected unsafe or corrupt data."""
