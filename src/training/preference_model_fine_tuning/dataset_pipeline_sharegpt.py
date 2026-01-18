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

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Generator, List
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


class ShareGPTPreferencePipeline:
    """Two-pass ShareGPT preference labeling pipeline."""

    def __init__(
        self,
        policy_label_model: LLMClient,
        policy_refine_model: LLMClient,
        batch_size: int,
        max_sample_counts: int,
    ):
        self.policy_label_model = policy_label_model
        self.policy_refine_model = policy_refine_model
        self.batch_size = batch_size
        self.max_sample_counts = max_sample_counts

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

    def run(
        self,
        existing_policy_label_path: Path,
        dataset_path: Path,
        output_path: Path,
    ) -> Path:
        """Execute the full pipeline and return the output path."""
        all_policy_candidates: List[RoutePolicy] = self._load_existing_policies(
            existing_policy_label_path
        )
        for batch in self._get_one_batch(dataset_path):
            logging.info(f"Processing batch of size {len(batch)}")
            added_policies, policy_label_candidates = (
                self.generate_policy_label_for_batch(batch, all_policy_candidates)
            )
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

    def _get_one_batch(self, dataset_path: Path) -> Generator[ShareGPTConversation]:
        raw = json.loads(dataset_path.read_text())
        current_batch: List[ShareGPTConversation] = []
        for idx, item in enumerate(raw):
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
            sample_id = str(item.get("id") or f"sample_{idx}")
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

    def generate_policy_label_for_batch(
        self,
        batch: List[ShareGPTConversation],
        existing_policies: List[RoutePolicy],
    ) -> List[RoutePolicy]:
        """Generate policy labels for a batch of conversations."""
        policy_label_prompt = f"""You are an expert at summarizing user intents into concise routing policy labels.
Given the following conversations(samples), generate a short generalized label for each conversation(sample) that best describes the user's intent of that conversation for routing purposes.
Example labels include "code_generation", "writing_emails", "text_summarization", "academic_qa", "translation", etc.

Here's a list of existing policy labels. If a new label is similar to an existing one, reuse the existing label.
### Existing Policy Labels
{json.dumps([{"label": p.label, "description": p.description} for p in existing_policies], indent=2)}

### Conversations
{json.dumps([convo.to_dict() for convo in batch], indent=2)}

Return a JSON array where each element has the following format:
{{"sample_id": <sample_id>, "label": <label>, "description": <brief description on the generalized label, not on the specific conversation>}}
The label should be concise (ideally 3 words or fewer) and accurately reflect the user's intent.
"""

        policy_label_response = self.policy_label_model.chat(
            messages=[{"role": "user", "content": policy_label_prompt}],
            temperature=0.3,
        )
        logging.info(f"Policy label response: {policy_label_response}")
        policy_label_candidates: List[RoutePolicySample] = safe_json_loads(
            policy_label_response
        )
        new_policies: List[RoutePolicy] = []
        logging.info(
            f"Received policy label candidates: {json.dumps(policy_label_candidates)}"
        )
        for candidate in policy_label_candidates:
            logging.info(f"Processing candidate: {json.dumps(candidate)}")
            label = candidate.get("label", "").strip()
            description = candidate.get("description", "").strip()
            if not label:
                logging.warning(
                    f"Skipping empty label in candidate: {json.dumps(candidate)}"
                )
                continue
            if any(p.label == label for p in existing_policies + new_policies):
                logging.info(f"Skipping duplicate label: {label}")
                continue
            new_policies.append(RoutePolicy(label=label, description=description))
        return new_policies, policy_label_candidates


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    policy_label_model = LLMClient(model="gpt-4o-mini")
    policy_refine_model = LLMClient(model="gpt-4o-mini")
    pipeline = ShareGPTPreferencePipeline(
        policy_label_model=policy_label_model,
        policy_refine_model=policy_refine_model,
        batch_size=5,
        max_sample_counts=11,
    )

    existing_policy_label_path = Path("existing_sharegpt_policies.jsonl")
    dataset_path = Path("ShareGPT_V3_unfiltered_cleaned_split.json")
    output_path = Path("sharegpt_preference_labeled.jsonl")
    pipeline.run(
        existing_policy_label_path=existing_policy_label_path,
        dataset_path=dataset_path,
        output_path=output_path,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
