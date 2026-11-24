"""Regression checks for the no-dataset, two-backend Speed draft."""

import copy
import importlib.util
import sys
from pathlib import Path

import pytest
import yaml

LAB = Path(__file__).resolve().parents[1]
# Test these checkout bytes even when a historical recipe_lab is installed.
spec = importlib.util.spec_from_file_location(
    "speed_draft", LAB / "recipe_lab/speed_draft.py"
)
assert spec is not None and spec.loader is not None
speed_draft = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = speed_draft
spec.loader.exec_module(speed_draft)


STRONG, FAST, PUBLIC = "actual/confident", "actual/quick", "router/ours"


def providers():
    def card(name, port, tools):
        return {
            "name": name,
            "provider_model_id": name,
            "api_format": "openai",
            "capabilities": ["chat", "text", "streaming"]
            + (["tools"] if tools else []),
            "modalities": {"input": ["text"], "output": ["text"]},
            "context_window_size": 65536,
            "max_output_tokens": 8192,
            "backend_refs": [
                {
                    "name": f"local-{port}",
                    "provider": "vllm",
                    "endpoint": f"localhost:{port}",
                    "protocol": "http",
                    "weight": 1,
                }
            ],
        }

    return {
        "version": "v0.5",
        "listeners": [{"name": "http", "address": "127.0.0.1", "port": 8899}],
        "providers": {
            "defaults": {"model": FAST},
            "models": [card(STRONG, 8001, True), card(FAST, 8002, False)],
        },
    }


def compiled():
    # Preserve the installed policy structure: produce the same modelRef shape as
    # native init and put an unrelated, colliding recipe first in the document.
    root = LAB.parents[1]
    mom = yaml.safe_load(
        (root / "config/recipes/built-in/latest/mom-v1/config.yaml").read_text("utf-8")
    )
    speed = next(r for r in mom["recipes"] if r["name"] == "speed")
    for decision in speed["decisions"]:
        decision["modelRefs"] = [
            {"model": FAST if decision["name"] == "fast" else STRONG}
        ]
    neighbor = copy.deepcopy(speed)
    neighbor["name"] = "unrelated"
    doc = providers()
    doc["recipes"] = [neighbor, speed]
    doc["entrypoints"] = [
        {"model_names": ["router/other"], "recipe": "unrelated"},
        {"model_names": [PUBLIC], "recipe": "speed"},
    ]
    return doc


def test_scoped_specialization_preserves_other_entrypoint_and_default():
    source = compiled()
    original = copy.deepcopy(source)
    result = speed_draft.specialize(source, PUBLIC, STRONG, FAST, STRONG)
    assert source == original, "work on a private copy; deployment inputs may be reused"
    assert result["recipes"][0] == original["recipes"][0]
    assert result["providers"] == original["providers"]
    recipe = result["recipes"][1]
    assert recipe["candidate_requirements"] == {
        "capabilities": "declared",
        "context": "known_limits",
    }
    by_name = {d["name"]: d for d in recipe["decisions"]}
    assert {n: [r["model"] for r in d["modelRefs"]] for n, d in by_name.items()} == {
        "reasoning": [STRONG],
        "tools": [STRONG],
        "fast": [FAST, STRONG],
    }
    assert all(d["algorithm"]["type"] == "static" for d in by_name.values())
    for old in original["recipes"][1]["decisions"]:
        changed = by_name[old["name"]]
        assert (changed.get("rules"), changed["priority"]) == (
            old.get("rules"),
            old["priority"],
        )


@pytest.mark.parametrize("both_tool_capable", [False, True])
def test_fast_tool_backend_can_serve_hard_requests_without_changing_rules(
    both_tool_capable,
):
    doc = compiled()
    strong, fast = doc["providers"]["models"]
    fast["capabilities"].append("tool_use")
    if not both_tool_capable:
        strong["capabilities"].remove("tools")
    tools = next(d for d in doc["recipes"][1]["decisions"] if d["name"] == "tools")
    tools["modelRefs"] = [{"model": FAST}]
    result = speed_draft.specialize(doc, PUBLIC, STRONG, FAST, FAST)
    for old, lane in zip(
        doc["recipes"][1]["decisions"],
        result["recipes"][1]["decisions"],
        strict=True,
    ):
        assert (old.get("rules"), old["priority"]) == (
            lane.get("rules"),
            lane["priority"],
        )
        expected = [r["model"] for r in old["modelRefs"]]
        if not both_tool_capable and lane["name"] == "reasoning":
            expected.append(FAST)
        assert [r["model"] for r in lane["modelRefs"]] == expected


def test_omitted_output_default_fits_the_tool_candidate_in_the_fast_lane():
    doc = compiled()
    doc["providers"]["models"][0]["max_output_tokens"] = 512
    result = speed_draft.specialize(doc, PUBLIC, STRONG, FAST, STRONG)
    fast = next(d for d in result["recipes"][1]["decisions"] if d["name"] == "fast")
    assert fast["plugins"] == [
        {"type": "request_params", "configuration": {"default_max_tokens": 512}}
    ]


def test_agent_streaming_removes_only_the_selected_decisions_total_timeout():
    doc = compiled()
    doc["listeners"][0]["timeout"] = "300s"
    for recipe in doc["recipes"]:
        recipe["decisions"][0]["reliability"] = {
            "total_timeout": "180s",
            "first_byte_timeout": "10s",
        }
    original = copy.deepcopy(doc)
    standard = speed_draft.specialize(doc, PUBLIC, STRONG, FAST, STRONG)
    streaming = speed_draft.specialize(
        doc, PUBLIC, STRONG, FAST, STRONG, agent_streaming=True
    )
    assert doc == original
    for before, after in zip(
        original["recipes"][1]["decisions"],
        standard["recipes"][1]["decisions"],
        strict=True,
    ):
        assert after.get("reliability") == before.get("reliability")
    expected = copy.deepcopy(standard)
    for decision in expected["recipes"][1]["decisions"]:
        decision.setdefault("reliability", {})["total_timeout"] = "0s"
    assert streaming == expected


def test_scoped_scores_cannot_silently_override_ordered_fallback():
    doc = compiled()
    scores = [{"name": "fast", "model_scores": [{"model": STRONG, "score": 1}]}]
    doc["recipes"][0]["signals"]["domains"] = scores
    speed_draft.specialize(doc, PUBLIC, STRONG, FAST, STRONG)
    doc["recipes"][1]["signals"]["domains"] = scores
    with pytest.raises(ValueError, match="model_scores"):
        speed_draft.specialize(doc, PUBLIC, STRONG, FAST, STRONG)


@pytest.mark.parametrize(
    "field", ["context_window_size", "max_output_tokens", "capabilities"]
)
def test_legacy_shadow_card_cannot_supply_missing_provider_facts(field):
    doc = providers()
    model = doc["providers"]["models"][0]
    # A legacy consumer might wrongly recover the missing field from here.
    doc["routing"] = {"modelCards": [copy.deepcopy(model)]}
    del model[field]
    with pytest.raises(ValueError):
        speed_draft.validate_cards(doc, STRONG, FAST, STRONG)


def test_tools_must_be_assigned_to_an_actually_tool_capable_model():
    doc = providers()
    speed_draft.validate_cards(doc, STRONG, FAST, STRONG)
    with pytest.raises(ValueError):
        speed_draft.validate_cards(doc, STRONG, FAST, FAST)


def test_chat_capability_alone_declares_text_generation():
    doc = providers()
    for card in doc["providers"]["models"]:
        card["capabilities"].remove("text")
    assert speed_draft.validate_cards(doc, STRONG, FAST, STRONG) == {STRONG}


def test_speed_yaml_anchor_cannot_modify_a_neighbor():
    source = compiled()
    source["recipes"][0]["decisions"] = source["recipes"][1]["decisions"]
    neighbor = copy.deepcopy(source["recipes"][0])
    result = speed_draft.specialize(source, PUBLIC, STRONG, FAST, STRONG)
    assert result["recipes"][0] == neighbor
    assert all(
        d["algorithm"]["type"] == "static" for d in result["recipes"][1]["decisions"]
    )


def test_duplicate_served_identity_is_not_two_backends():
    doc = providers()
    first, second = doc["providers"]["models"]
    second["provider_model_id"] = first["provider_model_id"]
    second["backend_refs"][0]["endpoint"] = first["backend_refs"][0]["endpoint"]
    with pytest.raises(ValueError, match="same declared model and backend"):
        speed_draft.validate_cards(doc, STRONG, FAST, STRONG)
    # A shared vendor gateway can serve independent models.
    second["provider_model_id"] = FAST
    speed_draft.validate_cards(doc, STRONG, FAST, STRONG)


def test_other_recipe_cannot_supply_missing_eligibility_policy():
    doc = compiled()
    del doc["recipes"][1]["candidate_requirements"]
    with pytest.raises(ValueError):
        speed_draft.specialize(doc, PUBLIC, STRONG, FAST, STRONG)


@pytest.mark.parametrize("separate_entry", [True, False])
def test_shared_speed_policy_requires_isolated_public_name(separate_entry):
    doc = compiled()
    if separate_entry:
        doc["entrypoints"].append(
            {"recipe": "speed", "model_names": ["router/sibling"]}
        )
    else:
        doc["entrypoints"][1]["model_names"].append("router/sibling")
    with pytest.raises(ValueError):
        speed_draft.specialize(doc, PUBLIC, STRONG, FAST, STRONG)


def test_legacy_recipe_and_unmapped_entrypoint_cannot_masquerade_as_speed():
    source = compiled()
    with pytest.raises(ValueError):
        speed_draft.specialize(source, "router/missing", STRONG, FAST, STRONG)
    source["version"] = "v0.3"
    with pytest.raises(ValueError):
        speed_draft.specialize(source, PUBLIC, STRONG, FAST, STRONG)
