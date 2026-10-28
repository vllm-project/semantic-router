"""Keep training schedules compatible with Transformers 4 and 5."""

import inspect
import os


def create_training_arguments(arguments_type, **kwargs):
    """Construct arguments without dropping the requested warmup or log path.

    Transformers 5.15 removed ``warmup_ratio`` and ``logging_dir``. Its
    ``warmup_steps`` accepts a fraction, and TensorBoard now reads its directory
    from ``TENSORBOARD_LOGGING_DIR``. Older versions retain the original fields.
    These command-line training entrypoints run one Trainer per process.
    """
    parameters = inspect.signature(arguments_type).parameters
    if "warmup_ratio" in kwargs and "warmup_ratio" not in parameters:
        ratio = kwargs.pop("warmup_ratio")
        if ratio is not None:
            if not 0 <= ratio <= 1:
                raise ValueError("warmup_ratio must be between 0 and 1")
            if not kwargs.get("warmup_steps", 0):
                # In the new API 1 means one step, not 100% of the run.
                if ratio == 1:
                    raise ValueError(
                        "For full-run warmup, set warmup_steps to the total "
                        "training steps with this Transformers version"
                    )
                kwargs["warmup_steps"] = ratio
    logging_dir = kwargs.get("logging_dir")
    if "logging_dir" in kwargs and "logging_dir" not in parameters:
        del kwargs["logging_dir"]
    arguments = arguments_type(**kwargs)
    if logging_dir is not None and "logging_dir" not in parameters:
        os.environ["TENSORBOARD_LOGGING_DIR"] = str(logging_dir)
    return arguments
