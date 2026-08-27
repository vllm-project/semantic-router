import importlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

UserConfig = importlib.import_module("cli.models").UserConfig


def _decision(**overrides):
    decision = {
        "name": "vision_request",
        "priority": 1000,
        "rules": {
            "operator": "AND",
            "conditions": [{"type": "conversation", "name": "image_input"}],
        },
        "modelRefs": [{"model": "vision-model"}],
    }
    decision.update(overrides)
    return decision


def test_decision_without_description_parses():
    config = UserConfig(version="0.3", routing={"decisions": [_decision()]})

    assert config.decisions[0].description is None


def test_decision_with_description_still_parses():
    config = UserConfig(
        version="0.3",
        routing={"decisions": [_decision(description="Routes vision requests.")]},
    )

    assert config.decisions[0].description == "Routes vision requests."
