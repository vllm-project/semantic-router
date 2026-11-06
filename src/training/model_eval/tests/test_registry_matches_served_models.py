"""Tie default evaluation IDs and immutable revisions to the Router's registry.

The Router no longer serves a Vela 1.0 specialist for any of these roles (its
decision model answers them), so the checkpoints are pinned to the revisions
its model registry records rather than to a served role. Legacy MoM download
targets remain separately checked, not silently renamed to mean Vela. The
dependency-light checks need no Hub access or model libraries.
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

from src.training.model_eval.constants import (
    LEGACY_MODEL_REGISTRY,
    MODEL_REGISTRY,
    VELA_RELEASE_REVISIONS,
    model_registry,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
MODELS_MK = REPOSITORY_ROOT / "tools/make/models.mk"

# What the registry used to say. These repos still resolve, which is exactly
# why the drift stayed invisible.
LEGACY_PREFIX = "vllm-sr/mmbert-"

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


HF_ORG = read_make_variable("HF_ORG")[0]


class EvaluationRegistryMatchesServedModels(unittest.TestCase):
    def test_lora_ids_match_the_makefile_adapter_list(self):
        expected = {
            f"{HF_ORG}/{name}"
            for name in read_make_variable("MMBERT_32K_LORA_ADAPTERS")
        }
        actual = {config["lora_id"] for config in LEGACY_MODEL_REGISTRY.values()}
        self.assertEqual(
            actual,
            expected,
            "the evaluation registry and MMBERT_32K_LORA_ADAPTERS name "
            "different adapters",
        )

    def test_merged_and_lora_ids_name_the_same_artifact(self):
        # Catches a half-finished rename: fact-check is the one role whose repo
        # is not its 8K name with a prefix bolted on, so it is the one most
        # likely to be updated on only one of its two lines.
        for role, config in LEGACY_MODEL_REGISTRY.items():
            with self.subTest(role=role):
                self.assertTrue(config["id"].endswith(MERGED_SUFFIX))
                self.assertTrue(config["lora_id"].endswith(LORA_SUFFIX))
                self.assertEqual(
                    config["id"].removesuffix(MERGED_SUFFIX),
                    config["lora_id"].removesuffix(LORA_SUFFIX),
                )

    def test_no_role_still_points_at_a_legacy_8k_repo(self):
        for role, config in LEGACY_MODEL_REGISTRY.items():
            with self.subTest(role=role):
                self.assertFalse(config["id"].startswith(LEGACY_PREFIX))
                self.assertFalse(config["lora_id"].startswith(LEGACY_PREFIX))

    def test_served_is_default_and_legacy_requires_an_explicit_collection(self):
        self.assertIs(model_registry(), MODEL_REGISTRY)
        self.assertIs(model_registry("legacy-mom"), LEGACY_MODEL_REGISTRY)
        self.assertEqual(len(MODEL_REGISTRY["feedback"]["labels"]), 5)
        self.assertEqual(MODEL_REGISTRY["feedback"]["labels"][4], "NO_FEEDBACK")
        self.assertEqual(len(LEGACY_MODEL_REGISTRY["feedback"]["labels"]), 4)
        self.assertEqual(MODEL_REGISTRY["jailbreak"]["labels"], ["benign", "jailbreak"])
        self.assertNotIn("hf_dataset", MODEL_REGISTRY["jailbreak"])
        self.assertNotIn("dataset_label_aliases", MODEL_REGISTRY["jailbreak"])
        self.assertEqual(
            LEGACY_MODEL_REGISTRY["jailbreak"]["dataset_label_aliases"],
            {"safe": "benign", "unsafe": "jailbreak"},
        )
        for role in MODEL_REGISTRY:
            self.assertNotIn("lora_id", MODEL_REGISTRY[role])
            self.assertRegex(MODEL_REGISTRY[role]["revision"], r"^[0-9a-f]{40}$")
        with self.assertRaises(ValueError):
            model_registry("misspelled")

    def test_all_vela_pins_match_the_native_registry(self):
        source = (
            REPOSITORY_ROOT / "src/semantic-router/pkg/config/registry.go"
        ).read_text()
        pins = dict(
            re.findall(
                r'RepoID:\s*"(vllm-sr/Vela-[^"]+)"[,]\s*Revision:\s*"([0-9a-f]{40})"',
                source,
            )
        )
        self.assertEqual(VELA_RELEASE_REVISIONS, pins)
        for entry in MODEL_REGISTRY.values():
            if entry["id"] in pins:
                self.assertEqual(entry["revision"], pins[entry["id"]])

    def test_eval_download_uses_the_registry(self):
        makefile = MODELS_MK.read_text()
        self.assertIn("python3 -m src.training.model_eval.download_models", makefile)
