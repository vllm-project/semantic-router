"""Thin entrypoint for onboarding evaluation (imports modular implementation)."""

from .onboard import ModelConfig, OnboardEvaluate, TestResult

__all__ = ["ModelConfig", "OnboardEvaluate", "TestResult"]


def _run_cli() -> int:
    from .onboard.cli import main

    return main()


if __name__ == "__main__":
    raise SystemExit(_run_cli())
