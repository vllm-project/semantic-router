from cli.models import UserConfig
from cli.validator import validate_user_config


def build_meta_routing_config(meta: dict) -> UserConfig:
    return UserConfig.model_validate(
        {
            "version": "v0.3",
            "listeners": [],
            "providers": {
                "defaults": {"default_model": "qwen3-8b"},
                "models": [
                    {
                        "name": "qwen3-8b",
                        "backend_refs": [{"endpoint": "127.0.0.1:8000"}],
                    }
                ],
            },
            "routing": {
                "modelCards": [{"name": "qwen3-8b"}],
                "signals": {
                    "domains": [{"name": "general", "description": "General requests"}]
                },
                "meta": meta,
                "decisions": [
                    {
                        "name": "general_route",
                        "description": "General route",
                        "priority": 1,
                        "rules": {
                            "operator": "AND",
                            "conditions": [{"type": "domain", "name": "general"}],
                        },
                        "modelRefs": [{"model": "qwen3-8b"}],
                    }
                ],
            },
        }
    )


def test_validate_user_config_accepts_meta_routing_contract():
    config = build_meta_routing_config(
        {
            "mode": "shadow",
            "max_passes": 2,
            "trigger_policy": {
                "decision_margin_below": 0.18,
                "required_families": [{"type": "preference", "min_confidence": 0.6}],
                "family_disagreements": [
                    {"cheap": "keyword", "expensive": "embedding"}
                ],
            },
            "allowed_actions": [
                {"type": "disable_compression"},
                {
                    "type": "rerun_signal_families",
                    "signal_families": ["preference", "jailbreak"],
                },
            ],
        }
    )

    assert validate_user_config(config) == []


def test_validate_user_config_rejects_unknown_meta_signal_family():
    config = build_meta_routing_config(
        {
            "mode": "observe",
            "trigger_policy": {
                "required_families": [{"type": "unknown_family"}],
            },
            "allowed_actions": [],
        }
    )

    errors = validate_user_config(config)

    assert errors
    assert "unknown_family" in errors[0].message
