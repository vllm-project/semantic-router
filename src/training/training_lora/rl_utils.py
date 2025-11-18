"""
Simple RL configuration helper for training scripts.
Reads `config/config.yaml` and returns a normalized RL config dictionary.
"""
import yaml
from typing import Dict

DEFAULTS = {
    "enabled": False,
    "algorithm": "ppo",
    "learning_rate": 1e-5,
    "gamma": 0.99,
    "batch_size": 16,
    "update_epochs": 4,
    "reward_metric": "accuracy",
}


def load_rl_config(config_path: str = "config/config.yaml") -> Dict:
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except FileNotFoundError:
        return DEFAULTS.copy()

    classifier = cfg.get("classifier", {})
    rl = classifier.get("rl_training", {}) if isinstance(classifier, dict) else {}

    out = DEFAULTS.copy()
    out.update({k: rl.get(k, v) for k, v in DEFAULTS.items()})

    # Normalize types
    out["enabled"] = bool(out["enabled"])
    out["algorithm"] = str(out["algorithm"])
    try:
        out["learning_rate"] = float(out["learning_rate"])
    except Exception:
        out["learning_rate"] = DEFAULTS["learning_rate"]
    try:
        out["gamma"] = float(out["gamma"])
    except Exception:
        out["gamma"] = DEFAULTS["gamma"]
    try:
        out["batch_size"] = int(out["batch_size"])
    except Exception:
        out["batch_size"] = DEFAULTS["batch_size"]
    try:
        out["update_epochs"] = int(out["update_epochs"])
    except Exception:
        out["update_epochs"] = DEFAULTS["update_epochs"]
    out["reward_metric"] = str(out["reward_metric"])

    return out


def is_rl_enabled(config_path: str = "config/config.yaml") -> bool:
    return load_rl_config(config_path).get("enabled", False)
