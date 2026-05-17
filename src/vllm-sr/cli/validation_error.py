"""Shared validation error type for CLI configuration checks."""


class ValidationError:
    """Validation error."""

    def __init__(self, message: str, field: str | None = None):
        self.message = message
        self.field = field

    def __str__(self):
        if self.field:
            return f"[{self.field}] {self.message}"
        return self.message
