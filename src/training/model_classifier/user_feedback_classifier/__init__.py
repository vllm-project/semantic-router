"""Four-class user feedback training and inference.

Inference dependencies are imported lazily so dataset contracts can be checked
without PyTorch. The historical binary dissatisfaction module is not included.
"""

__all__ = ["FeedbackDetector", "FeedbackResult"]


def __getattr__(name):
    if name in __all__:
        from . import inference_feedback  # noqa: PLC0415

        return getattr(inference_feedback, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
