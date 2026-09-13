"""Contract tests tying the evaluation registry to the checkpoints the router serves.

The repo writes the classifier model names down in three places and, until
issue #3644, nothing tied them together:

  * `tools/make/models.mk` decides what `make download-models` pulls,
  * `config/config.yaml` decides what the router loads at runtime,
  * `src/training/model_eval/constants.py` decides what the evaluation scores.

The third drifted. For roughly seven months the registry named the older 8K
`mmbert-*` repos while the router served the `mmbert32k-*` ones, so every
accuracy, F1 and confusion matrix `mom_collection_eval.py` produced was a
correct measurement of a checkpoint nobody runs. Nothing failed along the way,
because the 8K repos still exist on the Hub and still load.

These tests fail if the three lists come apart again.

See https://github.com/vllm-project/semantic-router/issues/3644.

`make test-training-contracts` runs on a stdlib-only system python3, so the
makefile and the router config are read with `re` rather than with a make
parser or PyYAML.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

from src.training.model_eval.constants import MODEL_REGISTRY

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
MODELS_MK = REPOSITORY_ROOT / "tools/make/models.mk"
ROUTER_CONFIG = REPOSITORY_ROOT / "config/config.yaml"

# Registry role -> the `model_catalog.system` key in config.yaml that names the
# checkpoint the router loads for that role. The two vocabularies grew up
# separately, so the mapping has to be written out rather than derived.
ROLE_TO_CONFIG_KEY = {
    "feedback": "feedback_detector",
    "jailbreak": "prompt_guard",
    "fact-check": "fact_check_classifier",
    "intent": "domain_classifier",
    "pii": "pii_classifier",
}

# What the registry used to say. These repos still resolve, which is exactly
# why the drift stayed invisible.
LEGACY_PREFIX = "llm-semantic-router/mmbert-"

MERGED_SUFFIX = "-merged"
LORA_SUFFIX = "-lora"


def read_make_variable(name: str) -> list[str]:
    """Return the words assigned to a `:=` variable in models.mk.

    Understands the backslash-continued list style the file uses for the model
    name lists.
    """
    collecting = False
    words: list[str] = []
    for line in MODELS_MK.read_text(encoding="utf-8").splitlines():
        if not collecting:
            assignment = re.match(rf"{re.escape(name)}\s*:=(.*)$", line)
            if assignment is None:
                continue
            collecting = True
            rest = assignment.group(1)
        else:
            rest = line
        stripped = rest.rstrip()
        continued = stripped.endswith("\\")
        words.extend(stripped.rstrip("\\").split())
        if not continued:
            return words
    raise AssertionError(f"{name} is not assigned in {MODELS_MK}")


def read_served_model(config_key: str) -> str:
    """Return the checkpoint basename config.yaml gives one system classifier."""
    config = ROUTER_CONFIG.read_text(encoding="utf-8")
    matches = re.findall(
        rf"^\s*{re.escape(config_key)}:\s*models/(\S+)\s*$", config, re.MULTILINE
    )
    if len(matches) != 1:
        raise AssertionError(
            f"expected exactly one '{config_key}: models/...' line in "
            f"{ROUTER_CONFIG}, found {len(matches)}"
        )
    return matches[0]


HF_ORG = read_make_variable("HF_ORG")[0]


class EvaluationRegistryMatchesServedModels(unittest.TestCase):
    def test_registry_covers_every_role_that_has_a_served_checkpoint(self):
        self.assertEqual(
            set(MODEL_REGISTRY),
            set(ROLE_TO_CONFIG_KEY),
            "a role gained or lost a registry entry; update ROLE_TO_CONFIG_KEY "
            "so the rest of this file keeps checking it",
        )

    def test_merged_ids_match_the_makefile_download_list(self):
        expected = {
            f"{HF_ORG}/{name}"
            for name in read_make_variable("MMBERT_32K_MERGED_MODELS")
        }
        actual = {config["id"] for config in MODEL_REGISTRY.values()}
        self.assertEqual(
            actual,
            expected,
            "the evaluation registry and MMBERT_32K_MERGED_MODELS name "
            "different checkpoints",
        )

    def test_lora_ids_match_the_makefile_adapter_list(self):
        expected = {
            f"{HF_ORG}/{name}"
            for name in read_make_variable("MMBERT_32K_LORA_ADAPTERS")
        }
        actual = {config["lora_id"] for config in MODEL_REGISTRY.values()}
        self.assertEqual(
            actual,
            expected,
            "the evaluation registry and MMBERT_32K_LORA_ADAPTERS name "
            "different adapters",
        )

    def test_each_role_scores_the_checkpoint_the_router_loads(self):
        for role, config_key in ROLE_TO_CONFIG_KEY.items():
            with self.subTest(role=role):
                self.assertEqual(
                    MODEL_REGISTRY[role]["id"],
                    f"{HF_ORG}/{read_served_model(config_key)}",
                    f"the evaluation scores a different checkpoint than the "
                    f"router serves as {config_key}",
                )

    def test_merged_and_lora_ids_name_the_same_artifact(self):
        # Catches a half-finished rename: fact-check is the one role whose repo
        # is not its 8K name with a prefix bolted on, so it is the one most
        # likely to be updated on only one of its two lines.
        for role, config in MODEL_REGISTRY.items():
            with self.subTest(role=role):
                self.assertTrue(config["id"].endswith(MERGED_SUFFIX))
                self.assertTrue(config["lora_id"].endswith(LORA_SUFFIX))
                self.assertEqual(
                    config["id"].removesuffix(MERGED_SUFFIX),
                    config["lora_id"].removesuffix(LORA_SUFFIX),
                )

    def test_no_role_still_points_at_a_legacy_8k_repo(self):
        for role, config in MODEL_REGISTRY.items():
            with self.subTest(role=role):
                self.assertFalse(config["id"].startswith(LEGACY_PREFIX))
                self.assertFalse(config["lora_id"].startswith(LEGACY_PREFIX))
