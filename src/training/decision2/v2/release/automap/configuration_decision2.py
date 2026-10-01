"""Decision 2.0 configuration for 🤗 Transformers (``trust_remote_code=True``).

The repository's root ``config.json`` is the package pointer the native runtime reads first (model name,
model files by role, input limit) plus ``model_type``, ``auto_map`` and ``custom_pipelines``. This class
only exposes those fields: the model is described by ``decision_config.json`` and every file is checked
against ``MODEL_MANIFEST.json`` when ``Decision2Model`` loads.
"""

try:
    from transformers import PreTrainedConfig
except ImportError:  # Transformers 4
    from transformers import PretrainedConfig as PreTrainedConfig


class Decision2Config(PreTrainedConfig):
    model_type = "decision2"
