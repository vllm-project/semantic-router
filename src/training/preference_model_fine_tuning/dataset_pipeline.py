"""Synthetic data pipeline for routing preference SFT.

Reference: Arch-Router: Aligning LLM Routing with Human Preferences

This module builds a ~50k training set that maps multi-turn conversations
to the correct routing policy. The pipeline has two phases:

Phase 1 (clean synthesis):
1. Build a diverse route policy pool from seed topic sources (industry
   classifications, MMLU-like academic topics, and common API surfaces).
2. Ask an LLM to expand these seeds into concrete policies, then refine
   them with a validator LLM to enforce clarity and granularity.
3. For each curated policy, synthesize a conversational intent, then a full
   dialogue, and finally verify the alignment between the dialogue and the
   intended policy. Failed samples are regenerated.

Phase 2 (augmentation):
1. Irrelevance injection: insert off-topic user turns or remove the
   ground-truth policy from the candidate set to create ambiguity.
2. Policy modification: inject irrelevant/misleading policies into the
   candidate set to sharpen decision boundaries.
3. Scenario mixing: stitch segments from multiple conversations to create
   longer, noisier dialogues with topic shifts and abandoned intents.

The pipeline emits JSONL with fields:
        {
                "conversation": [{"role": "user|assistant", "content": "..."}, ...],
                "route_policies": [{"label": "...", "description": "..."}, ...],
                "ground_truth_label": "...",
                "phase": "clean" | "augmented",
                "augmentations": ["irrelevance_injection", ...],
                "meta": {"policy_seed": "mmlu:physics", "verification": {"score": 0.92}}
        }
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from openai import OpenAI


###############################################################################
# Data structures
###############################################################################


@dataclass
class RoutePolicy:
    """A candidate routing policy option.

    Args:
            label: Short, unique policy name (e.g., "code_generation").
            description: Optional concise description to disambiguate the policy.
            seed: Source tag describing how this policy was derived (for auditing).
    """

    label: str
    description: Optional[str] = None
    seed: Optional[str] = None


@dataclass
class ConversationMessage:
    """Single dialogue turn."""

    role: str  # "user" or "assistant"
    content: str


@dataclass
class ConversationSample:
    """Training sample pairing a conversation with a routed policy."""

    conversation: List[ConversationMessage]
    ground_truth_policy: RoutePolicy
    # ground truth policy for this conversation
    all_policies: List[RoutePolicy]
    # All candidate policies for this conversation, possibly not include ground truth policy
    ground_truth_label: str
    phase: str  # "clean" or "augmented"
    augmentations: List[str] = field(default_factory=list)
    meta: Dict[str, object] = field(default_factory=dict)


###############################################################################
# LLM client and helpers
###############################################################################


class LLMClient:
    """Thin wrapper over the OpenAI-compatible chat API.

    The client is intentionally minimal; swap it out with another provider by
    re-implementing ``chat``.
    """

    def __init__(
        self, model: str, base_url: Optional[str] = None, api_key: Optional[str] = None
    ):
        self.model = model
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""


def batched(iterable: Sequence, batch_size: int) -> Iterable[Sequence]:
    """Yield fixed-size slices from a sequence."""

    for start in range(0, len(iterable), batch_size):
        yield iterable[start : start + batch_size]


def safe_json_loads(payload: str) -> object:
    """Parse the first JSON payload found in text; fallback to empty structures."""

    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        # Try to recover by trimming to first/last brace.
        first = payload.find("[")
        last = payload.rfind("]")
        if first != -1 and last != -1:
            snippet = payload[first : last + 1]
            try:
                return json.loads(snippet)
            except json.JSONDecodeError:
                return []
        first = payload.find("{")
        last = payload.rfind("}")
        if first != -1 and last != -1:
            snippet = payload[first : last + 1]
            try:
                return json.loads(snippet)
            except json.JSONDecodeError:
                return {}
    return []


###############################################################################
# Phase 1: Policy synthesis and conversation generation
###############################################################################


class TopicPool:
    """Builds seed topics from industry, academic, and tool domains."""

    INDUSTRY = [
        "finance_risk",
        "healthcare_diagnostics",
        "retail_personalization",
        "supply_chain",
        "customer_support",
        "legal_contracts",
        "cyber_security",
        "gaming_design",
        "media_entertainment",
    ]
    ACADEMIC = [
        # mmlu-like topics
        "business",
        "law",
        "psychology",
        "biology",
        "chemistry",
        "history",
        "health",
        "economics",
        "math",
        "physics",
        "computer_science",
        "philosophy",
        "engineering",
        "other",
    ]
    TOOL = [
        "code_generation",
        "code_debugging",
        "data_analysis",
        "image_generation",
        "text_summarization",
        "translation",
        "email_composition",
        "web_search",
    ]

    @classmethod
    def all(cls) -> List[str]:
        return list(dict.fromkeys(cls.INDUSTRY + cls.ACADEMIC + cls.TOOL))


class RoutePolicyGenerator:
    """Expands seed topics into refined routing policies using two LLM passes."""

    def __init__(self, proposer: LLMClient, validator: LLMClient):
        self.proposer = proposer
        self.validator = validator

    def generate(
        self, seeds: List[str], policies_per_seed: int = 4
    ) -> List[RoutePolicy]:
        candidates = []
        for seed_batch in batched(seeds, batch_size=6):
            prompt = (
                "Generate concise routing policies for the following topics. "
                "Each policy must have a unique `label` (snake_case) and a 1-2 sentence `description`. "
                "Return JSON list with objects {label, description}. Topics: "
                + ", ".join(seed_batch)
            )
            raw = self.proposer.chat(
                [
                    {"role": "system", "content": "You write clean JSON only."},
                    {"role": "user", "content": prompt},
                ]
            )
            parsed = safe_json_loads(raw)
            for item in parsed or []:
                label = (item.get("label") or "").strip()
                desc = (item.get("description") or "").strip() or None
                if label:
                    candidates.append(
                        RoutePolicy(
                            label=label,
                            description=desc,
                            seed="seed:" + ",".join(seed_batch),
                        )
                    )
        for candidate in candidates:
            logging.debug(
                f"[RoutePolicyGenerator] Proposed policy: {candidate.label} - {candidate.description} - {candidate.seed}"
            )
        return self._refine(candidates, policies_per_seed=policies_per_seed)

    def _refine(
        self, candidates: List[RoutePolicy], policies_per_seed: int
    ) -> List[RoutePolicy]:
        refined: List[RoutePolicy] = []
        prompt_base = (
            "You are a gatekeeper ensuring routing policies are atomic, clear, and non-overlapping. "
            "Given policies, keep only those with distinct intent boundaries, rewrite labels to snake_case, "
            "and trim descriptions to <=35 words. Return JSON list {label, description}."
        )
        for batch in batched(candidates, batch_size=10):
            policies_json = json.dumps(
                [
                    {
                        "label": p.label,
                        "description": p.description or "",
                        "seed": p.seed or "",
                    }
                    for p in batch
                ]
            )
            raw = self.validator.chat(
                [
                    {"role": "system", "content": "Return only JSON"},
                    {
                        "role": "user",
                        "content": f"{prompt_base}\nPolicies: {policies_json}",
                    },
                ],
                temperature=0.3,
            )
            parsed = safe_json_loads(raw)
            for item in parsed or []:
                label = (item.get("label") or "").strip()
                desc = (item.get("description") or "").strip() or None
                if label:
                    refined.append(
                        RoutePolicy(
                            label=label, description=desc, seed=item.get("seed")
                        )
                    )

        # Deduplicate and cap per seed
        seen = set()
        per_seed: Dict[str, int] = {}
        output: List[RoutePolicy] = []
        for policy in refined:
            if policy.label in seen:
                continue
            seed_key = policy.seed or "generic"
            if per_seed.get(seed_key, 0) >= policies_per_seed:
                continue
            seen.add(policy.label)
            per_seed[seed_key] = per_seed.get(seed_key, 0) + 1
            output.append(policy)
            logging.debug(
                f"[RoutePolicyRefiner] Refined to {policy.label} - {policy.description} - {policy.seed}"
            )
        return output


class ConversationSynthesizer:
    """Creates dialogues aligned to curated route policies."""

    def __init__(
        self, intent_model: LLMClient, dialogue_model: LLMClient, verifier: LLMClient
    ):
        self.intent_model = intent_model
        self.dialogue_model = dialogue_model
        self.verifier = verifier

    def synthesize(
        self,
        policies: List[RoutePolicy],
        target_samples: int,  # total samples to generate
        turns: Tuple[int, int] = (4, 8),
    ) -> List[ConversationSample]:
        samples: List[ConversationSample] = []
        policies_cycle = policies * ((target_samples // len(policies)) + 1)
        # number of samples for this policy
        policies_cycle = policies_cycle[:target_samples]

        for policy in policies_cycle:
            intent = self._draft_intent(policy)
            logging.debug(
                f"[ConversationSynthesizer] Drafted intent for policy {policy.label}: {intent}"
            )
            conversation = self._draft_dialogue(intent, policy, turns=turns)
            logging.debug(
                f"[ConversationSynthesizer] Drafted conversation for policy {policy.label}"
            )
            verified, score = self._verify(conversation, policy, policies=policies)
            logging.debug(
                f"[ConversationSynthesizer] Verified conversation for policy {policy.label}: {verified} (score={score})"
            )
            if not verified:
                logging.warning(
                    f"[ConversationSynthesizer] Verification failed for policy {policy.label}, regenerating conversation."
                )
                # Regenerate once; if it still fails, skip to keep dataset clean.
                conversation = self._draft_dialogue(intent, policy, turns=turns)
                verified, score = self._verify(conversation, policy)
                if not verified:
                    logging.warning(
                        f"[ConversationSynthesizer] Skipping policy {policy.label} after failed re-verification."
                    )
                    continue

            # pick random candidate policies including ground truth
            # this is in theory part of data augmentation, but we do it here for simplicity
            all_policies = [policy] + random.sample(
                [p for p in policies if p.label != policy.label],
                k=random.randint(5, min(20, len(policies) - 1)),
            )
            sample = ConversationSample(
                conversation=conversation,
                ground_truth_policy=policy,
                all_policies=all_policies,
                ground_truth_label=policy.label,
                phase="clean",
                meta={"policy_seed": policy.seed, "verification": {"score": score}},
            )
            samples.append(sample)
        return samples

    def _draft_intent(self, policy: RoutePolicy) -> str:
        prompt = (
            "Generate a specific user intent that clearly matches the policy. The user intent will be used to start a conversation."
            "Keep it one sentence. Policy label: {label}. Description: {desc}."
        ).format(label=policy.label, desc=policy.description or "")
        return self.intent_model.chat(
            [
                {"role": "system", "content": "You create crisp user intents."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.6,
            max_tokens=64,
        ).strip()

    def _draft_dialogue(
        self, intent: str, policy: RoutePolicy, turns: Tuple[int, int]
    ) -> List[ConversationMessage]:
        min_turns, max_turns = turns
        prompt = (
            "Write a natural multi-turn dialogue that starts from the user intent and naturally leads to the routing policy. "
            f"Use between {min_turns} and {max_turns} turns total. Keep the assistant concise and actionable. "
            f"Policy: {policy.label} - {policy.description or 'no description provided'}. Intent: {intent}"
        )
        text = self.dialogue_model.chat(
            [
                {
                    "role": "system",
                    "content": "Return JSON list of turns {role, content}. Roles are user or assistant.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.8,
            max_tokens=512,
        )
        parsed = safe_json_loads(text)
        conversation: List[ConversationMessage] = []
        for turn in parsed or []:
            role = (turn.get("role") or "").strip().lower()
            content = (turn.get("content") or "").strip()
            if role in {"user", "assistant"} and content:
                conversation.append(ConversationMessage(role=role, content=content))
        return conversation

    def _verify(
        self,
        conversation: List[ConversationMessage],
        policy: RoutePolicy,
        policies: List[RoutePolicy],
    ) -> Tuple[bool, float]:
        prompt = (
            "Given a conversation and a candidate policy for intent routing, rate alignment 0-1 where 1 means the selected policy is the "
            'only reasonable routing choice. Respond with JSON {"score": float, "reason": str}.'
        )
        convo_text = "\n".join(f"{m.role}: {m.content}" for m in conversation)
        raw = self.verifier.chat(
            [
                {
                    "role": "system",
                    "content": "Return only JSON with score and reason.",
                },
                {
                    "role": "user",
                    "content": f"{prompt}\nConversation:\n{convo_text}\nSelected Policy: {policy.label} - {policy.description}\nAll Policies: {', '.join(p.label for p in policies)}",
                },
            ],
            temperature=0.1,
            max_tokens=200,
        )
        parsed = safe_json_loads(raw) or {}
        score = float(parsed.get("score", 0.0)) if isinstance(parsed, dict) else 0.0
        return score >= 0.7, score


###############################################################################
# Phase 2: Augmentations
###############################################################################


class AugmentationEngine:
    """Applies augmentation strategies to clean samples."""

    def __init__(self, available_policies: List[RoutePolicy]):
        self.available_policies = available_policies

    def augment(
        self,
        samples: List[ConversationSample],
        target_size: int,  # total samples to generate
    ) -> List[ConversationSample]:
        augmented: List[ConversationSample] = []
        rng = random.Random(1337)

        strategies = [
            self._irrelevance_injection,
            self._policy_modification,
            self._scenario_mixing,
        ]

        while len(augmented) < target_size:
            base = rng.choice(samples)
            strategy = rng.choice(strategies)
            maybe_sample = strategy(base, rng)
            if maybe_sample:
                augmented.append(maybe_sample)
        return augmented

    def _irrelevance_injection(
        self, sample: ConversationSample, rng: random.Random
    ) -> Optional[ConversationSample]:
        convo = list(sample.conversation)
        if not convo:
            return None
        # TODO: agument with LLM-generated irrelevant content
        injection = ConversationMessage(
            role="user",
            content=rng.choice(
                [
                    "By the way, did you see the sports game last night?",
                    "Ignore that, I actually just wanted to check my calendar.",
                    "Unrelated question: what's the weather in Paris?",
                ]
            ),
        )
        insertion_idx = rng.randrange(0, len(convo) + 1)
        convo.insert(insertion_idx, injection)

        policies = [p for p in sample.all_policies]
        # Optionally drop the ground truth to simulate missing label
        if rng.random() < 0.3:
            policies = [p for p in policies if p.label != sample.ground_truth_label]

        return ConversationSample(
            conversation=convo,
            all_policies=policies,
            ground_truth_policy=sample.ground_truth_policy,
            ground_truth_label=sample.ground_truth_label,
            phase="augmented",
            augmentations=sample.augmentations + ["irrelevance_injection"],
            meta=sample.meta,
        )

    def _policy_modification(
        self, sample: ConversationSample, rng: random.Random
    ) -> Optional[ConversationSample]:
        policies = list(sample.all_policies)
        # TODO: augment with LLM-generated misleading policies
        distractors = rng.sample(
            self.available_policies, k=min(3, len(self.available_policies))
        )
        for d in distractors:
            if d.label not in [p.label for p in sample.all_policies]:
                policies.append(d)

        return ConversationSample(
            conversation=sample.conversation,
            all_policies=policies,
            ground_truth_policy=sample.ground_truth_policy,
            ground_truth_label=sample.ground_truth_label,
            phase="augmented",
            augmentations=sample.augmentations + ["policy_modification"],
            meta=sample.meta,
        )

    def _scenario_mixing(
        self, sample: ConversationSample, rng: random.Random
    ) -> Optional[ConversationSample]:
        if len(sample.conversation) < 2:
            return None
        # TODO: augment with LLM-generated scenario mixing content
        mix_with = rng.choice(self.available_policies)
        mixed_turn = ConversationMessage(
            role="user",
            content=f"Switching topics: also need help with {mix_with.label.replace('_', ' ')}",
        )
        convo = list(sample.conversation)
        convo.insert(rng.randrange(1, len(convo)), mixed_turn)
        return ConversationSample(
            conversation=convo,
            all_policies=sample.all_policies + [mix_with],
            ground_truth_policy=sample.ground_truth_policy,
            ground_truth_label=sample.ground_truth_label,
            phase="augmented",
            augmentations=sample.augmentations + ["scenario_mixing"],
            meta=sample.meta,
        )


###############################################################################
# Orchestration
###############################################################################


class PreferenceModelDataPipeline:
    """End-to-end driver that emits JSONL to disk."""

    def __init__(
        self,
        policy_generator: RoutePolicyGenerator,
        conversation_synthesizer: ConversationSynthesizer,
        augmentation_engine: AugmentationEngine,
        output_path: Path,
        clean_samples: int = 30000,
        augmented_samples: int = 20000,
        policy_checkpoint: Optional[Path] = None,
        phase1_checkpoint: Optional[Path] = None,
        phase2_checkpoint: Optional[Path] = None,
    ):
        """
        If checkpoints are provided:
        If they exists on disk, load from them instead of regenerating.
        If they do not exists on disk, write them after generation.
        """
        self.policy_generator = policy_generator
        self.conversation_synthesizer = conversation_synthesizer
        self.augmentation_engine = augmentation_engine
        self.output_path = output_path
        self.clean_samples = clean_samples
        self.augmented_samples = augmented_samples
        self.policy_checkpoint = policy_checkpoint
        self.phase1_checkpoint = phase1_checkpoint
        self.phase2_checkpoint = phase2_checkpoint

    def run(self) -> None:
        policies = self._maybe_load_policies()
        if policies is None:
            logging.info("Starting policy synthesis")
            seeds = TopicPool.all()
            policies = self.policy_generator.generate(seeds, policies_per_seed=4)
            logging.info("Curated %d policies", len(policies))
            self._maybe_write_policies(policies)

        logging.info("Generating clean conversations")
        clean = self._maybe_load_samples(self.phase1_checkpoint)
        if clean is None:
            clean = self.conversation_synthesizer.synthesize(
                policies, target_samples=self.clean_samples
            )
            logging.info("Generated %d clean samples", len(clean))
            if self.phase1_checkpoint:
                logging.info(
                    f"Writing phase-1 checkpoint with {len(clean)} clean samples to {self.phase1_checkpoint}"
                )
                self._write_jsonl(clean, path=self.phase1_checkpoint)

        logging.info("Applying augmentations")
        self.augmentation_engine.available_policies = policies
        augmented = self._maybe_load_samples(self.phase2_checkpoint)
        if augmented is None:
            augmented = self.augmentation_engine.augment(
                clean, target_size=self.augmented_samples
            )
            logging.info(f"Generated {len(augmented)} augmented samples")
            if self.phase2_checkpoint:
                logging.info(
                    f"Writing phase-2 checkpoint with {len(augmented)} augmented samples to {self.phase2_checkpoint}"
                )
                self._write_jsonl(augmented, path=self.phase2_checkpoint)

        all_samples = clean + augmented
        logging.info(f"Writing {len(all_samples)} samples to {self.output_path}")
        self._write_jsonl(all_samples, path=self.output_path)

    def _write_jsonl(self, samples: List[ConversationSample], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for sample in samples:
                f.write(json.dumps(sample_to_dict(sample), ensure_ascii=False) + "\n")

    def _maybe_write_policies(self, policies: List[RoutePolicy]) -> None:
        if not self.policy_checkpoint:
            return
        self.policy_checkpoint.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            {"label": p.label, "description": p.description, "seed": p.seed}
            for p in policies
        ]
        with self.policy_checkpoint.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    def _maybe_load_policies(self) -> Optional[List[RoutePolicy]]:
        if not self.policy_checkpoint or not self.policy_checkpoint.exists():
            return None
        with self.policy_checkpoint.open("r", encoding="utf-8") as f:
            data = json.load(f)
        policies: List[RoutePolicy] = []
        for item in data or []:
            label = item.get("label")
            if not label:
                continue
            policies.append(
                RoutePolicy(
                    label=label,
                    description=item.get("description"),
                    seed=item.get("seed"),
                )
            )
        logging.info(
            f"Loaded {len(policies)} policies from checkpoint {self.policy_checkpoint}"
        )
        return policies

    def _maybe_load_samples(
        self, checkpoint: Optional[Path]
    ) -> Optional[List[ConversationSample]]:
        if not checkpoint or not checkpoint.exists():
            return None
        samples: List[ConversationSample] = []
        with checkpoint.open("r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                conversation = [
                    ConversationMessage(role=t["role"], content=t["content"])
                    for t in obj.get("conversation", [])
                ]
                policies = [
                    RoutePolicy(
                        label=p.get("label", ""),
                        description=p.get("description"),
                        seed=p.get("seed"),
                    )
                    for p in obj.get("route_policies", [])
                    if p.get("label")
                ]
                samples.append(
                    ConversationSample(
                        conversation=conversation,
                        route_policies=policies,
                        ground_truth_label=obj.get("ground_truth_label", ""),
                        phase=obj.get("phase", "clean"),
                        augmentations=obj.get("augmentations", []),
                        meta=obj.get("meta", {}),
                    )
                )
        logging.info(f"Loaded {len(samples)} samples from {checkpoint}")
        return samples


###############################################################################
# Serialization helpers
###############################################################################


def sample_to_dict(sample: ConversationSample) -> Dict[str, object]:
    return {
        "conversation": [
            {"role": t.role, "content": t.content} for t in sample.conversation
        ],
        "route_policies": [
            {"label": p.label, "description": p.description, "seed": p.seed}
            for p in sample.route_policies
        ],
        "ground_truth_label": sample.ground_truth_label,
        "phase": sample.phase,
        "augmentations": sample.augmentations,
        "meta": sample.meta,
    }


###############################################################################
# Entry point example
###############################################################################


def build_default_pipeline(
    output_path: str,
    policy_checkpoint: Optional[str] = None,
    phase1_checkpoint: Optional[str] = None,
    phase2_checkpoint: Optional[str] = None,
) -> PreferenceModelDataPipeline:
    """Factory with sensible defaults for quick runs."""

    proposer = LLMClient(model="gpt-4o-mini")
    validator = LLMClient(model="gpt-4o-mini")
    intent_model = LLMClient(model="gpt-4o-mini")
    dialogue_model = LLMClient(model="gpt-4o-mini")
    verifier = LLMClient(model="gpt-4o-mini")

    policy_generator = RoutePolicyGenerator(proposer=proposer, validator=validator)
    synthesizer = ConversationSynthesizer(
        intent_model=intent_model, dialogue_model=dialogue_model, verifier=verifier
    )

    # Policies are unknown until generated; instantiate AugmentationEngine after policies exist
    dummy_policies = [RoutePolicy(label="placeholder")]
    augmentation_engine = AugmentationEngine(available_policies=dummy_policies)

    pipeline = PreferenceModelDataPipeline(
        policy_generator=policy_generator,
        conversation_synthesizer=synthesizer,
        augmentation_engine=augmentation_engine,
        output_path=Path(output_path),
        policy_checkpoint=Path(policy_checkpoint) if policy_checkpoint else None,
        phase1_checkpoint=Path(phase1_checkpoint) if phase1_checkpoint else None,
        phase2_checkpoint=Path(phase2_checkpoint) if phase2_checkpoint else None,
    )
    return pipeline


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    pipeline = build_default_pipeline(output_path="./preference_router_dataset.jsonl")
    # Run in steps so we can attach the real policy list to augmentation
    seeds = TopicPool.all()
    policies = pipeline.policy_generator.generate(seeds, policies_per_seed=4)
    pipeline.augmentation_engine.available_policies = policies
    clean = pipeline.conversation_synthesizer.synthesize(
        policies, target_samples=pipeline.clean_samples
    )
    augmented = pipeline.augmentation_engine.augment(
        clean, target_size=pipeline.augmented_samples
    )
    pipeline._write_jsonl(clean + augmented)
