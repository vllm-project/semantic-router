from cli.evaluation.target_arm_resolution import (
    resolve_target_arm,
    resolve_target_arm_id,
)
from cli.evaluation.target_contracts import EvaluationTargetArm


def _arm(arm_id: str, model: str) -> EvaluationTargetArm:
    return EvaluationTargetArm(
        id=arm_id,
        model=model,
        provider_model_id_digest="sha256:" + "a" * 64,
        input_cost_per_million_tokens_usd=0.0,
        output_cost_per_million_tokens_usd=0.0,
    )


def test_resolve_target_arm_accepts_unique_id_or_model() -> None:
    arms = (_arm("fast", "public-fast"), _arm("strong", "public-strong"))

    assert resolve_target_arm("fast", arms) is arms[0]
    assert resolve_target_arm("public-strong", arms) is arms[1]
    assert resolve_target_arm_id("fast", arms) == "fast"
    assert resolve_target_arm_id("public-strong", arms) == "strong"
    assert resolve_target_arm_id(None, arms) is None
    assert resolve_target_arm_id("missing", arms) is None


def test_resolve_target_arm_rejects_ambiguous_public_selector() -> None:
    arms = (_arm("fast", "shared-model"), _arm("strong", "shared-model"))

    assert resolve_target_arm("shared-model", arms) is None
    assert resolve_target_arm_id("shared-model", arms) is None

    cross_kind_collision = (_arm("fast", "public-fast"), _arm("strong", "fast"))
    assert resolve_target_arm("fast", cross_kind_collision) is None
    assert resolve_target_arm_id("fast", cross_kind_collision) is None
