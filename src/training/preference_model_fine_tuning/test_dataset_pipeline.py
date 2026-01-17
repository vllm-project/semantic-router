import json
import random
from pathlib import Path

import pytest

from dataset_pipeline import (
    AugmentationEngine,
    ConversationSynthesizer,
    PreferenceModelDataPipeline,
    RoutePolicyGenerator,
    TopicPool,
)


class DummyLLMClient:
    """Deterministic stand-in for LLMClient used in tests."""

    def chat(
        self, messages, temperature: float = 0.7, max_tokens: int = 512
    ) -> str:  # noqa: ARG002
        prompt = "\n".join(msg["content"] for msg in messages)

        if "Generate concise routing policies" in prompt:
            seeds_text = prompt.split("Topics:")[-1]
            seeds = [seed.strip() for seed in seeds_text.split(",") if seed.strip()]
            payload = []
            for seed in seeds:
                for idx in range(3):
                    payload.append(
                        {
                            "label": f"{seed}_policy_{idx}",
                            "description": f"Policy for {seed} #{idx}",
                            "seed": f"seed:{seed}",
                        }
                    )
            return json.dumps(payload)

        if "gatekeeper ensuring routing policies" in prompt:
            json_start = prompt.find("Policies:")
            raw_json = prompt[json_start + len("Policies:") :].strip()
            try:
                policies = json.loads(raw_json)
            except json.JSONDecodeError:
                policies = []
            refined = [
                {
                    "label": p.get("label", ""),
                    "description": p.get("description", ""),
                    "seed": p.get("seed", ""),
                }
                for p in policies
                if p.get("label")
            ]
            return json.dumps(refined)

        if "crisp user intents" in prompt:
            label = "intent"
            if "Policy label:" in prompt:
                label = (
                    prompt.split("Policy label:", maxsplit=1)[1].split(".")[0].strip()
                )
            return f"Need help with {label}"

        if "Return JSON list of turns" in prompt:
            intent_text = prompt.split("Intent:")[-1].strip()
            return json.dumps(
                [
                    {"role": "user", "content": intent_text or "Need help"},
                    {
                        "role": "assistant",
                        "content": f"Here is guidance for {intent_text or 'the request'}",
                    },
                ]
            )

        if "score and reason" in prompt:
            return json.dumps({"score": 0.9, "reason": "mock alignment"})

        return "mock-response"


def test_pipeline_runs_with_mock_llm(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Run the full pipeline with a dummy LLM client to ensure it dry-runs."""

    monkeypatch.setattr(TopicPool, "all", classmethod(lambda cls: ["alpha", "beta"]))
    random.seed(1234)

    dummy_llm = DummyLLMClient()

    policy_generator = RoutePolicyGenerator(proposer=dummy_llm, validator=dummy_llm)
    synthesizer = ConversationSynthesizer(
        intent_model=dummy_llm, dialogue_model=dummy_llm, verifier=dummy_llm
    )
    augmentation_engine = AugmentationEngine(available_policies=[])

    pipeline = PreferenceModelDataPipeline(
        policy_generator=policy_generator,
        conversation_synthesizer=synthesizer,
        augmentation_engine=augmentation_engine,
        output_path=tmp_path / "dataset.jsonl",
        clean_samples=2,
        augmented_samples=1,
    )

    pipeline.run()

    assert pipeline.output_path.exists()

    with pipeline.output_path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]

    assert len(rows) == 3
    assert {row["phase"] for row in rows} == {"clean", "augmented"}

    for row in rows:
        assert row["conversation"]
        assert all(
            turn["role"] in {"user", "assistant"} for turn in row["conversation"]
        )
        assert row["ground_truth_label"]
        labels = [policy["label"] for policy in row["all_policies"]]
        assert row["ground_truth_label"] in labels

    augmented_rows = [row for row in rows if row["phase"] == "augmented"]
    assert augmented_rows
    assert augmented_rows[0]["augmentations"]
    assert augmented_rows[0]["ground_truth_policy"]
