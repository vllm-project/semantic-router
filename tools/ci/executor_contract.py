"""Shared executor contract definitions."""

# Executors whose ci.yml job iterates with a matrix (matrix.label).
# All other executors support exactly one verification.
MATRIX_EXECUTORS = frozenset({"local", "tools", "platform", "e2e"})
