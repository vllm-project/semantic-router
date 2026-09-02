"""Errors raised at the normalized-suite storage trust boundary."""


class SuiteStoreError(ValueError):
    """A normalized suite was unsafe, corrupt, or inconsistent."""
