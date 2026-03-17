from types import SimpleNamespace

from bench.reasoning.canonical_patch import generate_vsr_canonical_patch


def make_comparison() -> SimpleNamespace:
    return SimpleNamespace(
        dataset_name="MMLU-Pro",
        model_name="qwen3-14b",
        timestamp="2026-03-16T00:00:00Z",
        standard_mode=SimpleNamespace(
            mode_name="standard",
            accuracy=0.71,
            token_usage_ratio=1.2,
            time_per_output_token=16.0,
        ),
        reasoning_mode=SimpleNamespace(
            mode_name="reasoning",
            accuracy=0.79,
            token_usage_ratio=1.45,
            time_per_output_token=18.5,
        ),
    )


def test_generate_vsr_canonical_patch_emits_canonical_patch() -> None:
    recommendation = generate_vsr_canonical_patch(
        [make_comparison()],
        model_name="qwen3-14b",
        reasoning_family="qwen3",
    )

    assert recommendation["reasoning_family"] == "qwen3"
    assert "manual_follow_up" not in recommendation
    assert recommendation["suggested_vsr_patch"] == {
        "providers": {
            "defaults": {
                "reasoning_families": {
                    "qwen3": {
                        "type": "chat_template_kwargs",
                        "parameter": "enable_thinking",
                    }
                }
            },
            "models": [{"name": "qwen3-14b", "reasoning_family": "qwen3"}],
        },
        "routing": {"modelCards": [{"name": "qwen3-14b"}]},
    }


def test_generate_vsr_canonical_patch_requires_follow_up_for_unknown_family() -> None:
    recommendation = generate_vsr_canonical_patch(
        [make_comparison()],
        model_name="custom-model",
        reasoning_family=None,
    )

    assert recommendation["reasoning_family"] == "auto"
    assert recommendation["suggested_vsr_patch"] == {
        "providers": {"models": [{"name": "custom-model"}]},
        "routing": {"modelCards": [{"name": "custom-model"}]},
    }
    assert "manual_follow_up" in recommendation
