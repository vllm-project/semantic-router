"""Evaluation storage trust-boundary errors."""


class StoreError(ValueError):
    """The run artifact store rejected unsafe or corrupt data."""


class SuiteStoreError(ValueError):
    """The benchmark suite store rejected unsafe or inconsistent data."""
