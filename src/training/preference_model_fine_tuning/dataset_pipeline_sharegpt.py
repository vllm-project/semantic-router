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

            """
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
                "other"
            """
            return [
                RoutePolicy(label="code_generation", description="Code generation"),
                RoutePolicy(label="creative_writing", description="Creative writing"),
                RoutePolicy(
                    label="text_summarization", description="Text summarization"
                ),
                RoutePolicy(
                    label="math_problem_solving", description="Math problem solving"
                ),
                RoutePolicy(label="legal_inquiry", description="Legal inquiry"),
                RoutePolicy(label="medical_inquiry", description="Medical inquiry"),
                RoutePolicy(label="financial_inquiry", description="Financial inquiry"),
                RoutePolicy(label="business_inquiry", description="Business inquiry"),
                RoutePolicy(label="history_inquiry", description="History inquiry"),
                RoutePolicy(
                    label="psychology_inquiry", description="Psychology inquiry"
                ),
                RoutePolicy(label="biology_inquiry", description="Biology inquiry"),
                RoutePolicy(label="chemistry_inquiry", description="Chemistry inquiry"),
                RoutePolicy(label="health_inquiry", description="Health inquiry"),
                RoutePolicy(label="economics_inquiry", description="Economics inquiry"),
                RoutePolicy(label="physics_inquiry", description="Physics inquiry"),
                RoutePolicy(
                    label="computer_science_inquiry",
                    description="Computer science inquiry",
                ),
                RoutePolicy(
                    label="philosophy_inquiry", description="Philosophy inquiry"
                ),
                RoutePolicy(
                    label="engineering_inquiry", description="Engineering inquiry"
                ),
            ]
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
        input_dataset_path: Path,
        output_path: Path,
    ) -> Path:
        """Execute the full pipeline and return the output path."""
        all_policy_candidates: List[RoutePolicy] = self._load_existing_policies(
            existing_policy_label_path
        )
        batch_count = 0
        start_index = self.start_sample_index
        for batch in self._get_one_batch(input_dataset_path, start_index):
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
        self, input_dataset_path: Path, start_index: int
    ) -> Generator[list[ShareGPTConversation]]:
        raw = json.loads(input_dataset_path.read_text())
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
            asyncio.to_thread(
                self._request_policy_label, conversation, existing_policies
            )
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
        self,
        conversation: ShareGPTConversation,
        existing_policies: List[RoutePolicy],
    ) -> Optional[Dict[str, str]]:
        """Issue a single policy label request for one conversation."""

        prompt = self._build_policy_label_prompt(
            conversation, existing_policies=existing_policies
        )
        response_payload = self.policy_label_model.chat(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "RoutePolicySample",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "sample_id": {"type": "string"},
                            "label": {"type": "string"},
                            "description": {"type": "string"},
                        },
                        "required": ["sample_id", "label", "description"],
                        "additionalProperties": False,
                    },
                },
            },
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
            )
            return None
        return candidate

    def _build_policy_label_prompt(
        self, conversation: ShareGPTConversation, existing_policies: list[RoutePolicy]
    ) -> str:
        conversation_json = json.dumps(conversation.to_dict(), indent=2)
        return f"""You are an expert at summarizing user intents into concise routing policy labels.
Given the following single conversation, generate a short generalized label that best describes the user's intent for routing purposes.

### Instructions
Return exactly one JSON object in the format:
{{"sample_id": <sample_id>, "label": <label>, "description": <brief description on the generalized label, not on the specific conversation>}}
Copy the sample_id faithfully from the input conversation.
The label should consist of a "domain" (e.g. legal, finance) plus an "action" (e.g. summarization, inquiry, code_generation) so it can generalize to similar conversations.
Try to make the label (both domain and action) general enough. If there's no specific domain, use "general" as the domain.
### Current Policy Labels
Try to reuse one of the existing policy labels if it fits well.
{json.dumps([policy.label for policy in existing_policies])}

### Conversation
{conversation_json}
"""

    def refine(
        self,
        refined_dataset_path: Path,
        existing_policy_label_path: Path,
        input_dataset_path: Path,
        batch_size: int,
    ) -> None:
        """
        Refine policy labels for the entire dataset.
        This function produces a mapping from original policy labels to refined labels.
        """
        refined_policy_mappings = self._refine_policies_with_preprocessing(
            existing_policy_label_path, input_dataset_path, batch_size
        )
        with open(refined_dataset_path, "w") as f:
            json.dump(refined_policy_mappings, f)

    def _refine_policies_with_preprocessing(
        self,
        existing_policy_label_path: Path,
        input_dataset_path: Path,
        batch_size: int,
    ) -> dict[str, str]:
        """Refine sample labels for a batch of conversations."""
        raw_policies = self._load_existing_policies(existing_policy_label_path)

        # TODO: preprocessing to filter out long tail labels
        # input_dataset_path is used to count the occurrence of each label
        all_refined_policies: dict[str, str] = {}
        for batch_start in range(0, len(raw_policies), batch_size):
            batched_policies = raw_policies[batch_start : batch_start + batch_size]
            logging.info(
                f"Refining policies for batch {batch_start // batch_size + 1} of size {len(batched_policies)}"
            )
            raw_policies_to_refined = self._refine_policies(batched_policies)
            all_refined_policies.update(raw_policies_to_refined)

        return all_refined_policies

    def _refine_policies(
        self,
        raw_policies: List[RoutePolicy],
    ) -> dict[str, RoutePolicy]:
        """Refine and deduplicate policies using the refine LLM."""
        refine_prompt = f"""You are an expert at refining and deduplicating routing policies.
### Instructions
Given the following list of routing policies, identify and merge duplicates or highly similar policies into a single canonical policy.
For each policy, map it to a refined label and description. The refined label should be more general and concise, suitable for routing purposes. Represent the refined label in snake case.
For example, "art_history_inquiry", "history_inquiry", "history_questions", "history_information", "history_summarization" should be all grouped to "history_inquiry".
"cpp_code_generation", "engineering_software_development", "python_coding", "programming_string_replacement" should be all grouped into "code_generation".
Return a JSON dict that represents the mapping in the format:
{{<original_label>: <refined_label>}}
### Policies
{json.dumps([p.label for p in raw_policies])}
"""
        refine_response = self.policy_refine_model.chat(
            messages=[{"role": "user", "content": refine_prompt}],
            temperature=0.3,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "StringToStringMap",
                    "schema": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                    },
                },
            },
        )
        logging.info(f"Refine response: {refine_response}")
        refined_policy_mappings = safe_json_loads(refine_response)
        return refined_policy_mappings

    async def verify_policies(
        self,
        original_dataset_path: Path,
        input_dataset_path: Path,
        refined_dataset_path: Path,  # this should actually be renamed to refined_policy_mapping_path
        output_path: Path,
        batch_size: int,
        start_index: int,
    ) -> None:
        """
        original_dataset_path: shareGPT dataset path
        input_dataset_path: dataset with original labels (produced by first pass)

        Verify the integrity of refined policies.
        After refinement, we merge the semantically similar policies, so the canonical labels should be a valid label set.
        We need to ensure that the mapping still works, which means among all labels in the canonical label set
        The label for each sample should be the most suitable one.
        We need this because when generating the labels, some early samples may use labels that are too general, and a more specific label may appear later.

        This function produces the final sample_id to canonical label mapping.
        """
        # generate the mapping from sample_id to canonical label
        # input_dataset_path is a jsonl file with each line being a json object
        with open(input_dataset_path, "r") as f:
            route_policy_samples = [json.loads(line) for line in f]

        refined_policy_label_mappings = json.loads(refined_dataset_path.read_text())
        # build sample_id_to_label_mapping
        sample_id_to_canonical_label: dict[str, str] = {}
        canonical_labels_set = set(refined_policy_label_mappings.values())
        for item in route_policy_samples:
            sample_id = item.get("sample_id")
            label = item.get("label")
            if label in refined_policy_label_mappings:
                refined_label = refined_policy_label_mappings[label]
                sample_id_to_canonical_label[sample_id] = refined_label
            else:
                # if it's not in the mapping, it means it's long tail label, so we filter them out
                continue
        # call LLM to verify the refined label for each sample concurrently
        batch_count = 0
        for batch in self._get_one_batch(original_dataset_path, start_index):
            batch_count += 1
            logging.info(f"Verifying batch {batch_count} of size {len(batch)}")
            # if the sample is in long tail label set, we skip it
            batch_filtered = [
                sample
                for sample in batch
                if sample.sample_id in sample_id_to_canonical_label
            ]
            tasks = [
                asyncio.to_thread(
                    self._verify_policies_for_sample,
                    sample,
                    sample_id_to_canonical_label[sample.sample_id],
                    canonical_labels_set,
                )
                for sample in batch_filtered
            ]
            verified_route_policy_samples = await asyncio.gather(
                *tasks, return_exceptions=True
            )
            valid_verified_route_policy_samples: List[RoutePolicySample] = []
            for conversation, verified_route_policy_sample in zip(
                batch, verified_route_policy_samples
            ):
                if isinstance(verified_route_policy_sample, Exception):
                    logging.error(
                        "Policy label generation failed for sample %s: %s",
                        conversation.sample_id,
                        verified_route_policy_sample,
                    )
                    continue
                if verified_route_policy_sample is None:
                    logging.warning(
                        "Policy label model returned no candidate for sample %s",
                        conversation.sample_id,
                    )
                    continue
                valid_verified_route_policy_samples.append(verified_route_policy_sample)

            # write verified_route_policy_samples to file
            with open(output_path, "a") as f:
                for verified_route_policy_sample in valid_verified_route_policy_samples:
                    f.write(
                        json.dumps(
                            {
                                "sample_id": verified_route_policy_sample.sample_id,
                                "label": verified_route_policy_sample.label,
                            }
                        )
                        + "\n"
                    )

    def _verify_policies_for_sample(
        self,
        sample: ShareGPTConversation,
        canonical_label: str,
        canonical_labels_set: set[str],
    ) -> RoutePolicySample:
        """Verify the policy label for one conversation."""
        verify_policy_prompt = f"""You are an expert at grouping conversations with routing policy labels.
### Instructions
Given the following conversation, pick the most suitable label from the canonical label set that best describes the user's intent for routing purposes. If none of the labels apply, respond with 'none_of_the_above'. Do not create new labels that is not in the canonical label set.
{json.dumps(list(canonical_labels_set))}

### Conversation
{json.dumps(sample.to_dict(), indent=2)}
"""
        verify_response = self.policy_refine_model.chat(
            messages=[{"role": "user", "content": verify_policy_prompt}],
            temperature=0.3,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "RoutePolicySample",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "sample_id": {"type": "string"},
                            "label": {"type": "string"},
                        },
                        "required": ["sample_id", "label"],
                        "additionalProperties": False,
                    },
                },
            },
        )

        logging.info(f"Verify response: {verify_response}")
        verified_route_policy_sample = safe_json_loads(verify_response)
        if verified_route_policy_sample.get("label") != canonical_label:
            logging.warning(
                "Sample %s: verified label %s does not match canonical label %s",
                sample.sample_id,
                verified_route_policy_sample.get("label"),
                canonical_label,
            )
        if (
            verified_route_policy_sample.get("label") not in canonical_labels_set
            and verified_route_policy_sample.get("label") != "none_of_the_above"
        ):
            logging.warning(
                "Sample %s: verified label %s is not in canonical label set",
                sample.sample_id,
                verified_route_policy_sample.get("label"),
            )
            verified_route_policy_sample["label"] = "none_of_the_above"
        return RoutePolicySample(
            sample_id=verified_route_policy_sample["sample_id"],
            label=verified_route_policy_sample["label"],
            description="",
        )


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
        batch_size=20,
        max_sample_counts=50000,
        start_sample_index=0,
    )

    existing_policy_label_path = Path("existing_sharegpt_policies_2.jsonl")
    dataset_path = Path("ShareGPT_V3_unfiltered_cleaned_split.json")
    output_path = Path("sharegpt_preference_labeled_2.jsonl")
    refined_dataset_path = Path("label_canonical_map.json")
    # asyncio.run(
    #     pipeline.run(
    #         existing_policy_label_path=existing_policy_label_path,
    #         input_dataset_path=dataset_path,
    #         output_path=output_path,
    #     )
    # )
    # pipeline.refine(
    #     refined_dataset_path=refined_dataset_path,
    #     existing_policy_label_path=existing_policy_label_path,
    #     input_dataset_path=output_path,
    #     batch_size=1000,
    # )
    asyncio.run(
        pipeline.verify_policies(
            original_dataset_path=dataset_path,
            input_dataset_path=output_path,
            refined_dataset_path=refined_dataset_path,
            output_path=Path("verified_sharegpt_policy_labels_2.jsonl"),
            batch_size=20,
            start_index=34066,
        )
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
