"""ShareGPT preference label synthesis pipeline.

This module implements a two-pass pipeline that converts ShareGPT JSON
conversations into (conversation, preference_label) pairs suitable for
fine-tuning a routing preference model. The process mirrors the methodology
outlined in the module docstring:

1. Group ShareGPT conversations into configurable, fixed-size batches suitable
    for prompt construction.
2. Topic discovery pass: prompt an LLM to summarize each conversation with a
   concise topic that reflects the preferred routing target.
3. Cluster/deduplicate all discovered topics into a canonical label set.
4. Label attribution pass: prompt the LLM again to map each conversation to one
   of the canonical preference labels.
5. Persist the results as JSONL rows containing the original conversation and
   the assigned preference label plus useful metadata for auditability.

The pipeline is intentionally modular: swap in different LLM providers by
passing alternative ``LLMClient`` implementations, adjust batching heuristics,
or reuse components for other datasets.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple
from dataset_pipeline import LLMClient, safe_json_loads


@dataclass
class ShareGPTConversation:
    """Normalized ShareGPT sample."""

    sample_id: str
    messages: List[Turn]

    def to_dict(self) -> Dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "messages": [message.to_dict() for message in self.messages],
        }


@dataclass
class RoutePolicySample:
    """Sample labeled with a route policy."""

    sample_id: str
    label: str
    description: str


@dataclass
class RoutePolicy:
    label: str
    description: str


@dataclass
class Turn:
    role: str
    content: str

    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


def get_sample_id_hash(sample_id: str) -> str:
    """Get a hash for a sample ID."""
    # example: "QWJhYvA_0" -> "QWJhYvA"
    return sample_id.split("_")[0]


class ShareGPTPreferencePipeline:
    """Two-pass ShareGPT preference labeling pipeline."""

    def __init__(
        self,
        policy_label_model: LLMClient,
        policy_refine_model: LLMClient,
        batch_size: int,
        max_sample_counts: int,
        start_sample_index: int,
    ):
        self.policy_label_model = policy_label_model
        self.policy_refine_model = policy_refine_model
        self.batch_size = batch_size
        self.max_sample_counts = max_sample_counts
        self.start_sample_index = start_sample_index

    def _load_existing_policies(self, path: Path) -> List[RoutePolicy]:
        """Load existing route policies from a JSONL file."""
        policies: List[RoutePolicy] = []
        if not path.exists():
            logging.info(f"No existing policy file found at {path}, starting fresh.")
            return policies
        with open(path, "r") as f:
            for line in f:
                item = json.loads(line)
                label = item.get("label", "").strip()
                description = item.get("description", "").strip()
                if label:
                    policies.append(RoutePolicy(label=label, description=description))
        logging.info(f"Loaded {len(policies)} existing policies from {path}")
        return policies

    def _load_policy_label_candidates(self, path: Path) -> List[RoutePolicySample]:
        """Load existing policy label candidates from a JSONL file."""
        candidates: List[RoutePolicySample] = []
        if not path.exists():
            logging.info(
                f"No existing candidates file found at {path}, starting fresh."
            )
            return candidates
        with open(path, "r") as f:
            for line in f:
                item = json.loads(line)
                sample_id = item.get("sample_id", "").strip()
                label = item.get("label", "").strip()
                description = item.get("description", "").strip()
                if sample_id and label:
                    candidates.append(
                        RoutePolicySample(
                            sample_id=sample_id, label=label, description=description
                        )
                    )
        logging.info(f"Loaded {len(candidates)} existing candidates from {path}")
        return candidates

    async def run(
        self,
        existing_policy_label_path: Path,
        raw_dataset_path: Path,
        output_path: Path,
    ) -> Path:
        """Execute the full pipeline and return the output path."""
        all_policy_candidates: List[RoutePolicy] = self._load_existing_policies(
            existing_policy_label_path
        )
        batch_count = 0
        start_index = self.start_sample_index
        for batch in self._get_one_batch(raw_dataset_path, start_index):
            batch_count += 1
            logging.info(f"Processing batch {batch_count} of size {len(batch)}")
            (
                added_policies,
                policy_label_candidates,
            ) = await self.generate_policy_label_for_batch(batch, all_policy_candidates)
            all_policy_candidates.extend(added_policies)
            # store the policy_label_candidate to local
            with open(output_path, "a") as f:
                for candidate in policy_label_candidates:
                    f.write(json.dumps(candidate) + "\n")
            # write all policies to to existing_policy_label_path
            with open(existing_policy_label_path, "w") as f:
                for policy in all_policy_candidates:
                    f.write(
                        json.dumps(
                            {"label": policy.label, "description": policy.description}
                        )
                        + "\n"
                    )
        return output_path

    # --------------------------- dataset utilities -------------------------

    def _get_one_batch(
        self, raw_dataset_path: Path, start_index: int
    ) -> Generator[ShareGPTConversation]:
        raw = json.loads(raw_dataset_path.read_text())
        current_batch: List[ShareGPTConversation] = []
        seen_sample_id_hashes = set()
        for idx, item in enumerate(raw):
            if idx < start_index:
                continue
            convo = item.get("conversations") or []
            normalized_messages: List[Turn] = []
            for turn in convo:
                speaker_raw = turn.get("from")
                speaker = str(speaker_raw).lower().strip()
                content_raw = turn.get("value")
                content = str(content_raw).strip()
                if not content:
                    logging.info(f"Skipping empty content in sample {item.get('id')}")
                    continue
                role = "assistant" if speaker == "gpt" else "user"
                normalized_messages.append(Turn(role=role, content=content))
            if not normalized_messages:
                logging.info(f"Skipping sample {item.get('id')} with no valid messages")
                continue
            sample_id = str(item.get("id"))

            # note: here we assume that all sample_ids with the same hash have the same preference label to boost performance
            sample_id_hash = get_sample_id_hash(sample_id)
            if sample_id_hash in seen_sample_id_hashes:
                logging.info(f"Skipping duplicate sample_id {sample_id}")
                continue
            seen_sample_id_hashes.add(sample_id_hash)
            current_batch.append(
                ShareGPTConversation(
                    sample_id=sample_id,
                    messages=normalized_messages,
                )
            )
            if idx + 1 >= self.max_sample_counts:
                break
            if self.batch_size and len(current_batch) >= self.batch_size:
                yield current_batch
                current_batch = []
        if current_batch:
            yield current_batch

    async def generate_policy_label_for_batch(
        self,
        batch: List[ShareGPTConversation],
        existing_policies: List[RoutePolicy],
    ) -> Tuple[List[RoutePolicy], List[Dict[str, str]]]:
        """Generate policy labels by querying each conversation concurrently."""

        tasks = [
            asyncio.to_thread(self._request_policy_label, conversation)
            for conversation in batch
        ]
        raw_candidates = await asyncio.gather(*tasks, return_exceptions=True)

        policy_label_candidates: List[Dict[str, str]] = []
        for conversation, candidate in zip(batch, raw_candidates):
            if isinstance(candidate, Exception):
                logging.error(
                    "Policy label generation failed for sample %s: %s",
                    conversation.sample_id,
                    candidate,
                )
                continue
            if candidate is None:
                logging.warning(
                    "Policy label model returned no candidate for sample %s",
                    conversation.sample_id,
                )
                continue
            policy_label_candidates.append(candidate)

        logging.info(
            "Received policy label candidates: %s",
            json.dumps(policy_label_candidates),
        )

        all_valid_sample_ids = {convo.sample_id for convo in batch}
        valid_policy_label_candidates: List[Dict[str, str]] = []
        new_policies: List[RoutePolicy] = []
        for candidate in policy_label_candidates:
            sample_id = candidate.get("sample_id", "").strip()
            label = candidate.get("label", "").strip()
            description = candidate.get("description", "").strip()
            if sample_id not in all_valid_sample_ids:
                logging.warning(
                    "Skipping unknown sample_id in candidate: %s",
                    json.dumps(candidate),
                )
                continue
            if not label:
                logging.warning(
                    "Skipping empty label in candidate: %s",
                    json.dumps(candidate),
                )
                continue
            valid_policy_label_candidates.append(candidate)
            if any(p.label == label for p in existing_policies + new_policies):
                logging.info("Skipping duplicate label: %s", label)
                continue
            new_policies.append(RoutePolicy(label=label, description=description))
        return new_policies, valid_policy_label_candidates

    def _request_policy_label(
        self, conversation: ShareGPTConversation
    ) -> Optional[Dict[str, str]]:
        """Issue a single policy label request for one conversation."""

        prompt = self._build_policy_label_prompt(conversation)
        response_payload = self.policy_label_model.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        parsed_payload = safe_json_loads(response_payload)
        candidate: Optional[Dict[str, str]] = None
        if isinstance(parsed_payload, dict):
            candidate = parsed_payload
        elif isinstance(parsed_payload, list):
            candidate = next(
                (item for item in parsed_payload if isinstance(item, dict)), None
            )
        if candidate is None:
            logging.warning(
                "Policy label model returned unexpected payload for sample %s: %s",
                conversation.sample_id,
                response_payload,
            )
            return None
        if candidate.get("sample_id", "").strip() != conversation.sample_id:
            logging.warning(
                "Mismatched sample_id in candidate for sample %s: %s",
                conversation.sample_id,
                json.dumps(candidate),
            )
            return None
        return candidate

    def _build_policy_label_prompt(self, conversation: ShareGPTConversation) -> str:
        conversation_json = json.dumps(conversation.to_dict(), indent=2)
        return f"""You are an expert at summarizing user intents into concise routing policy labels.
Given the following single conversation, generate a short generalized label that best describes the user's intent for routing purposes.

### Instructions
Return exactly one JSON object in the format:
{{"sample_id": <sample_id>, "label": <label>, "description": <brief description on the generalized label, not on the specific conversation>}}
Copy the sample_id faithfully from the input conversation.
The label should consist of a "domain" (e.g. legal, finance) plus an "action" (e.g. summarization, inquiry, code_generation) so it can generalize to similar conversations.

### Conversation
{conversation_json}
"""

    def refine(self, refined_dataset_path: Path) -> None:
        """Refine policy labels for the entire dataset."""
        refined_sample_labels = self._refine_sample_labels()
        # save refined labels to file
        with open(refined_dataset_path, "w") as f:
            for sample_id, policy in refined_sample_labels.items():
                f.write(
                    json.dumps(
                        {
                            "sample_id": sample_id,
                            "label": policy.label,
                            "description": policy.description,
                        }
                    )
                    + "\n"
                )

    def _refine_sample_labels(
        self,
    ) -> Dict[str, RoutePolicy]:
        """Refine sample labels for a batch of conversations."""
        raw_policies = self._load_existing_policies()
        policy_label_candidates = self._load_policy_label_candidates()
        refined_policies = self._refine_policies(raw_policies)

        # remmap sample labels to refined labels
        sample_id_to_refined_policy: Dict[str, RoutePolicy] = {}
        for candidate in policy_label_candidates:
            sample_id = candidate.sample_id
            original_label = candidate.label
            if original_label not in refined_policies:
                logging.warning(
                    f"Original label {original_label} not found in refined policies"
                )
                continue
            refined_policy = refined_policies[original_label]
            sample_id_to_refined_policy[sample_id] = refined_policy
        return sample_id_to_refined_policy

    def _refine_policies(
        self,
        raw_policies: List[RoutePolicy],
    ) -> List[RoutePolicy]:
        """Refine and deduplicate policies using the refine LLM."""
        refine_prompt = f"""You are an expert at refining and deduplicating routing policies.
### Instructions
Given the following list of routing policies, identify and merge duplicates or highly similar policies into a single canonical policy.
For each policy, map it to a refined label and description. The refined label should be more general and concise, suitable for routing purposes. Represent the refined label in snake case.
Return a JSON array for each policy in the format:
{{"original_label": <original_label>, "refined_label": <refined_label>, "description": <refined_description>}}
### Policies
{json.dumps([{"label": p.label, "description": p.description} for p in raw_policies], indent=2)}
"""
        refine_response = self.policy_refine_model.chat(
            messages=[{"role": "user", "content": refine_prompt}],
            temperature=0.3,
        )
        refined_policy_mappings: List[Dict[str, str]] = safe_json_loads(refine_response)
        assert len(refined_policy_mappings) == len(
            raw_policies
        ), "Refined policies count must match raw policies count"
        raw_policies_to_refined: Dict[str, RoutePolicy] = {}
        for mapping in refined_policy_mappings:
            original_label = mapping.get("original_label", "").strip()
            refined_label = mapping.get("refined_label", "").strip()
            description = mapping.get("description", "").strip()
            if not original_label or not refined_label:
                logging.warning(f"Skipping invalid mapping: {json.dumps(mapping)}")
                continue
            raw_policies_to_refined[original_label] = RoutePolicy(
                label=refined_label, description=description
            )
        return raw_policies_to_refined


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    policy_label_model = LLMClient(
        model="Qwen/Qwen3-8B",
        api_key="token-abc123",
        base_url="http://localhost:8000/v1",
    )
    policy_refine_model = LLMClient(
        model="Qwen/Qwen3-8B",
        api_key="token-abc123",
        base_url="http://localhost:8000/v1",
    )
    pipeline = ShareGPTPreferencePipeline(
        policy_label_model=policy_label_model,
        policy_refine_model=policy_refine_model,
        batch_size=10,
        max_sample_counts=50000,
        start_sample_index=0,
    )

    existing_policy_label_path = Path("existing_sharegpt_policies.jsonl")
    dataset_path = Path("ShareGPT_V3_unfiltered_cleaned_split.json")
    output_path = Path("sharegpt_preference_labeled.jsonl")
    refined_dataset_path = Path("refined_sharegpt_policy_labels.jsonl")
    asyncio.run(
        pipeline.run(
            existing_policy_label_path=existing_policy_label_path,
            raw_dataset_path=dataset_path,
            output_path=output_path,
        )
    )
    pipeline.refine(refined_dataset_path=refined_dataset_path)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
