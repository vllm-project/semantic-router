"""Fail-closed checks for the GB10 qualification profile."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import qualification

REPO = Path(__file__).resolve().parents[3]
DIGEST = (
    "ghcr.io/vllm-project/semantic-router/vllm-sr-cuda"
    "@sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
)


def gpu_row(**overrides: object) -> dict:
    row = {
        "index": 0,
        "name": "NVIDIA GB10",
        "compute_capability": "sm_121",
        "driver_version": "580.65.06",
        "memory_used_mib": 1024,
    }
    row.update(overrides)
    return row


def passing_receipt(**overrides: object) -> dict:
    cases = [
        {
            "domain": "code",
            "reasoning_noul": 0.8,
            "difficulty_score": 1.4,
            "decision": "external_code",
            "alias": qualification.EXTERNAL_ALIAS,
            "upstream_received": True,
            "prompt": "prove the merge",
        },
        {
            "domain": "math",
            "reasoning_noul": 0.1,
            "difficulty_score": 0.2,
            "decision": "local_generation",
            "alias": qualification.LOCAL_ALIAS,
            "upstream_received": True,
            "prompt": "two plus two",
        },
    ]
    receipt = qualification.build_receipt(
        image=DIGEST,
        gpu=gpu_row(),
        reference_key="cuda:sm_121",
        source_revision="abc123",
        cli_version="vllm-sr version: 0.0.0",
        card={
            "engine": "native",
            "dtype": "auto",
            "device": "cuda:0",
            "accelerator_validated": False,
            "revision": qualification.MODEL_REVISION,
        },
        startup_seconds=12.5,
        memory_before_mib=1000,
        memory_after_mib=4200,
        cases=cases,
        latency=qualification.latency_distribution([10.0] * 20, "warm prompt"),
        upstreams={
            qualification.LOCAL_ALIAS: "host.docker.internal:1",
            qualification.EXTERNAL_ALIAS: "host.docker.internal:2",
        },
    )
    receipt.update(overrides)
    return receipt


class HostAndImageTests(unittest.TestCase):
    def test_linux_arm64_gb10_is_the_recorded_host(self) -> None:
        self.assertEqual(
            qualification.observed_architecture("Linux", "aarch64"), "linux/arm64"
        )
        rows = qualification.parse_nvidia_smi("NVIDIA GB10, 12.1, 580.65.06, 2048\n")
        gpu = qualification.require_gb10_gpu(rows)
        self.assertEqual(gpu["compute_capability"], "sm_121")
        self.assertEqual(gpu["index"], 0)

    def test_other_hosts_and_gpus_fail(self) -> None:
        with self.assertRaises(qualification.QualificationError):
            qualification.observed_architecture("Darwin", "arm64")
        with self.assertRaises(qualification.QualificationError):
            qualification.observed_architecture("Linux", "x86_64")
        with self.assertRaises(qualification.QualificationError):
            qualification.require_gb10_gpu([gpu_row(name="NVIDIA GeForce RTX 4090")])
        with self.assertRaises(qualification.QualificationError):
            qualification.require_gb10_gpu([gpu_row(index=1)])

    def test_digest_pin_rejects_latest(self) -> None:
        self.assertEqual(qualification.require_digest_image(DIGEST), DIGEST)
        with self.assertRaises(qualification.QualificationError):
            qualification.require_digest_image(
                "ghcr.io/vllm-project/semantic-router/vllm-sr-cuda:latest"
            )
        with self.assertRaises(qualification.QualificationError):
            qualification.require_digest_image(
                DIGEST.replace("@sha256:", ":latest@sha256:")
            )

    def test_image_platform_must_be_linux_arm64(self) -> None:
        qualification.require_arm64_image("arm64", "linux")
        with self.assertRaises(qualification.QualificationError):
            qualification.require_arm64_image("amd64", "linux")


class RoutingTests(unittest.TestCase):
    def test_thresholds_partition_kai_answers(self) -> None:
        aliases = {
            qualification.select_alias(domain, reasoning, difficulty)
            for domain in ("code", "math", "other")
            for reasoning in (0.0, 0.5, 0.9)
            for difficulty in (0.0, 1.0, 2.0)
        }
        self.assertEqual(
            aliases, {qualification.LOCAL_ALIAS, qualification.EXTERNAL_ALIAS}
        )
        self.assertEqual(
            qualification.select_alias("code", 0.5, 1.0),
            qualification.EXTERNAL_ALIAS,
        )
        self.assertEqual(
            qualification.select_alias("code", 0.4999572038651558, 0.9390606098685782),
            qualification.LOCAL_ALIAS,
        )
        self.assertEqual(
            qualification.select_alias("math", 0.9, 2.0), qualification.LOCAL_ALIAS
        )
        with self.assertRaises(qualification.QualificationError):
            qualification.select_alias(None, 0.9, 2.0)

    def test_routed_request_must_match_the_answer_and_both_aliases(self) -> None:
        external = qualification.prove_route(
            {
                "domain": "code",
                "reasoning": 0.8,
                "difficulty": 1.4,
                "selected_model": qualification.EXTERNAL_ALIAS,
                "decision_name": "external_code",
            },
            qualification.EXTERNAL_ALIAS,
        )
        local = qualification.prove_route(
            {
                "domain": "other",
                "reasoning": 0.1,
                "difficulty": 0.2,
                "selected_model": qualification.LOCAL_ALIAS,
                "decision_name": "local_generation",
            },
            qualification.LOCAL_ALIAS,
        )
        qualification.require_both_cases([external, local])
        with self.assertRaises(qualification.QualificationError):
            qualification.prove_route(
                {
                    "domain": "code",
                    "reasoning": 0.8,
                    "difficulty": 1.4,
                    "selected_model": qualification.LOCAL_ALIAS,
                    "decision_name": "local_generation",
                },
                qualification.LOCAL_ALIAS,
            )
        with self.assertRaises(qualification.QualificationError):
            qualification.require_both_cases([local])

    def test_preview_reads_choice_noul_and_score(self) -> None:
        answers = qualification.answers_from_preview(
            {
                "selected_model": qualification.EXTERNAL_ALIAS,
                "decision_result": {
                    "decision_name": "external_code",
                    "matched_signals": {
                        "decision": [
                            "gb10_domain:code",
                            "gb10_reasoning",
                            "gb10_difficulty",
                        ]
                    },
                },
                "signal_values": {
                    "decision:gb10_reasoning": 0.8,
                    "decision:gb10_difficulty": 1.4,
                },
            }
        )
        self.assertEqual(answers["domain"], "code")
        self.assertEqual(answers["reasoning"], 0.8)
        self.assertEqual(answers["difficulty"], 1.4)


class ReceiptTests(unittest.TestCase):
    def test_passing_receipt_keeps_measurements_apart_from_the_catalog_floor(
        self,
    ) -> None:
        receipt = passing_receipt()
        self.assertEqual(receipt["id"], qualification.RECEIPT_ID)
        self.assertEqual(receipt["architecture"], "linux/arm64")
        self.assertEqual(receipt["device"]["placement"], "cuda:0")
        self.assertNotIn("min_device_memory_gib", receipt["resources"])
        self.assertEqual(receipt["modeled_capacity"]["min_device_memory_gib"], 4)
        self.assertEqual(receipt["resources"]["gpu_memory_footprint_mib"], 3200)
        self.assertEqual(receipt["resources"]["latency"]["samples"], 20)
        self.assertFalse(receipt["qualifies_all_nvidia_devices"])
        self.assertNotIn(qualification.FORBIDDEN_EVIDENCE, json.dumps(receipt))

    def test_cpu_labels_and_a_single_route_fail(self) -> None:
        receipt = passing_receipt()
        receipt["architecture"] = "linux/amd64"
        with self.assertRaises(qualification.QualificationError):
            qualification.validate_passing_receipt(receipt)
        receipt = passing_receipt()
        receipt["cli_version"] = qualification.FORBIDDEN_EVIDENCE
        with self.assertRaises(qualification.QualificationError):
            qualification.validate_passing_receipt(receipt)
        receipt = passing_receipt()
        receipt["resources"]["min_device_memory_gib"] = 4
        with self.assertRaises(qualification.QualificationError):
            qualification.validate_passing_receipt(receipt)
        with self.assertRaises(qualification.QualificationError):
            qualification.latency_distribution([1.0, 2.0], "prompt")

    def test_missing_cuda_golden_is_a_failure(self) -> None:
        table = json.loads(
            (
                REPO / "src/model-runtime/vllm_srun/registry/golden_answers.json"
            ).read_text(encoding="utf-8")
        )
        entry = table[qualification.MODEL_REPO]
        self.assertEqual(entry["revision"], qualification.MODEL_REVISION)
        self.assertIn("cpu", entry["answers"])
        self.assertIn("rocm", entry["answers"])
        self.assertFalse(qualification.golden_reference_present(table, "cuda:sm_121"))
        message = qualification.missing_reference_message("cuda:sm_121")
        self.assertIn(qualification.RECORDER_COMMAND, message)
        self.assertIn("src/model-runtime/docs/records/", message)
        with self.assertRaises(qualification.QualificationError) as caught:
            qualification.require_kai_card(
                [
                    {
                        "repo": qualification.MODEL_REPO,
                        "id": qualification.KAI_DEPLOYMENT,
                        "revision": qualification.MODEL_REVISION,
                        "device": "cpu",
                        "golden": {"status": "unverified", "reference": None},
                    }
                ],
                "cuda:sm_121",
            )
        self.assertIn("cuda", str(caught.exception))

    def test_matched_cuda_card_passes(self) -> None:
        card = qualification.require_kai_card(
            [
                {
                    "repo": qualification.MODEL_REPO,
                    "revision": qualification.MODEL_REVISION,
                    "device": "cuda:0",
                    "golden": {"status": "matched", "reference": "cuda:sm_121"},
                }
            ],
            "cuda:sm_121",
        )
        self.assertEqual(card["device"], "cuda:0")


class RecipeAndRegistryTests(unittest.TestCase):
    def test_recipe_pins_cuda_without_fallback(self) -> None:
        config = (REPO / "config/recipes/gb10/config.yaml").read_text(encoding="utf-8")
        self.assertIn("device: cuda:0", config)
        self.assertIn(qualification.MODEL_REVISION, config)
        self.assertNotIn("device: auto", config)
        self.assertNotIn("fallback", config)
        self.assertIn("endpoint: local-generation:8000", config)
        self.assertIn("endpoint: external-generation:8000", config)
        rendered = qualification.rewrite_endpoints(
            config, "host.docker.internal:41001", "host.docker.internal:41002"
        )
        self.assertNotIn("local-generation:8000", rendered)
        self.assertIn("host.docker.internal:41001", rendered)
        self.assertIn("host.docker.internal:41002", rendered)

    def test_receipt_id_stays_out_of_the_cpu_inventory(self) -> None:
        catalog = (REPO / "tools/ci/verification_catalog.yaml").read_text(
            encoding="utf-8"
        )
        targets = (REPO / "tools/ci/execution_targets.yaml").read_text(encoding="utf-8")
        cuda = (REPO / "src/model-runtime/vllm_srun/accel/cuda.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn(qualification.RECEIPT_ID, catalog)
        self.assertIn("linux-amd64-cpu:", targets)
        self.assertIn("devices: [none, cpu]", targets)
        self.assertIn("validated = False", cuda)

    def test_teardown_lists_only_processes_this_run_started(self) -> None:
        stack = qualification.OwnedStack()
        stack.start("gb10-stub")
        stack.start("gb10-serve")
        self.assertEqual(stack.stop_names(), ["gb10-stub", "gb10-serve"])
        self.assertEqual(stack.stop_names(), [])

    def test_stub_records_one_chat(self) -> None:
        with qualification.RecordingStub(qualification.LOCAL_ALIAS) as stub:
            status, _body, _headers = qualification._http_json(
                "POST",
                f"http://127.0.0.1:{stub.port}/v1/chat/completions",
                {"model": qualification.LOCAL_ALIAS, "messages": []},
            )
        self.assertEqual(status, 200)
        self.assertEqual(stub.requests, ["/v1/chat/completions"])


if __name__ == "__main__":
    unittest.main()
