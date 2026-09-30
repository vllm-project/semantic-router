#!/usr/bin/env python3
"""Execute the published-model contract and reject silent skips or empty selection."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FAMILIES = (
    "Domain",
    "Guard",
    "PII",
    "FactCheck",
    "Feedback",
    "Modality",
    "Safety",
    "Hazard",
    "Embedding",
    "Reranker",
)
CLASSIFIER_TESTS = (
    "TestJailbreakDistributionRealModel",
    "TestJailbreakRiskRealModelContract",
    "TestClassifyPIIWithDetails_RealModelFindsPIIPastTheWindow",
    "TestFeedbackPredictionAndAbstentionRealModel",
    "TestFeedbackForcedAbstentionPreservesPredictionRealModel",
    "TestFactCheckPolicyRealModel",
    "TestFactCheckEmptyInputRealModel",
    "TestLocalClassifierMaintainedCPU",
    "TestUnifiedClassifierPublishedModels",
)
CANDLE_CACHE_TESTS = ("TestNegationFalseHitRegressionInMemory",)
MULTIMODAL_BINDING_TESTS = (
    "TestMultiModalEmbeddingInit",
    "TestMultiModalEncodeText",
    "TestMultiModalInputValidation",
)

# Supported CPU additions have distinct task contracts and artifact formats.
# They deliberately do not imply Halu/ORT or Omni/Candle qualification.
RUNTIME_ADDITIONS = {"candle": ("Halu",), "ort": ("OmniNano", "OmniMini")}
OWNED_OMNI_TESTS = {
    "classification": (
        "TestEmbeddingClassifier_IntegrationImageQueryEndToEnd",
        "TestEmbeddingClassifier_IntegrationTextRulesIgnoredOnImagePath",
    ),
    "cache": ("TestOmniStorageIntegrationUsesArtifactDimensionAndIdentity",),
    "modeldownload": ("TestPublishedOmniPreparedInventory",),
}
HALU_CASES = tuple(
    "TestPublishedVelaHalu/" + name
    for name in ("supported", "unsupported", "unicode", "input-budget")
)
OMNI_CASES = tuple(
    "TestPublishedOmniModels/" + name for name in ("nano", "mini", "nano-revisit")
)


def runtime_families(provider: str) -> tuple[str, ...]:
    return FAMILIES + RUNTIME_ADDITIONS[provider]


def required_inventory(provider: str, suite: str = "runtime") -> set[tuple[str, str]]:
    """The receipt cannot shorten the checked-in mandatory case inventory."""
    if suite == "multimodal" and provider == "candle":
        return {(".", name) for name in MULTIMODAL_BINDING_TESTS}
    if suite != "runtime" or provider not in RUNTIME_ADDITIONS:
        raise ValueError(f"unsupported model suite {suite}/{provider}")
    cases = {
        ("./pkg/modelruntime/native", "TestPublishedVelaModels/" + name)
        for name in FAMILIES
    }
    cases.update(("./pkg/classification", name) for name in CLASSIFIER_TESTS)
    if provider == "candle":
        cases.update(("./pkg/cache", name) for name in CANDLE_CACHE_TESTS)
        cases.update(("./pkg/modelruntime/native", name) for name in HALU_CASES)
    else:
        cases.add(
            (
                "./pkg/modelruntime",
                "TestOwnedImplicitORTEmbeddingAndExplicitCandleOverride",
            )
        )
        cases.update(("./pkg/modelruntime/native", name) for name in OMNI_CASES)
        cases.update(
            ("./pkg/" + package, name)
            for package, names in OWNED_OMNI_TESTS.items()
            for name in names
        )
    return cases


def validate_results(events: list[dict], expected: set[str]) -> dict:
    passed = {e["Test"] for e in events if e.get("Action") == "pass" and "Test" in e}
    skipped = sorted(
        {e["Test"] for e in events if e.get("Action") == "skip" and "Test" in e}
    )
    failed = sorted(
        {
            e.get("Test", e.get("Package", "unknown"))
            for e in events
            if e.get("Action") == "fail"
        }
    )
    missing = sorted(expected - passed)
    return {
        "passed": sorted(passed),
        "skipped": skipped,
        "failed": failed,
        "missing": missing,
        "success": bool(expected) and not (skipped or failed or missing),
    }


def run_suite(
    package: str,
    names: tuple[str, ...],
    expected: set[str],
    env: dict,
    output: Path,
    *,
    cwd: Path,
) -> dict:
    args = [
        "go",
        "test",
        "-json",
        "-count=1",
        "-timeout=45m",
        "-p=1",
        "-ldflags=-X github.com/vllm-project/semantic-router/src/semantic-router/pkg/config.defaultModelProvider="
        + env["VLLM_SR_MODEL_TEST_PROVIDER"],
        "-run",
        "^(" + "|".join(names) + ")$",
        package,
    ]
    events = []
    with output.open("w") as log:
        process = subprocess.Popen(
            args,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                print(line, end="", flush=True)
                continue
            events.append(event)
            if event.get("Output"):
                print(event["Output"], end="", flush=True)
        code = process.wait()
    result = validate_results(events, expected)
    result.update(package=package, exit_code=code, expected=sorted(expected))
    result["success"] = result["success"] and code == 0
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", choices=("runtime", "multimodal"), default="runtime")
    # Adding a matrix entry cannot qualify a GPU through the CPU-only contract.
    parser.add_argument("--device", choices=("cpu",), default="cpu")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    provider = manifest["provider"]
    providers = {"candle", "ort"} if args.suite == "runtime" else {"candle"}
    if provider not in providers:
        raise ValueError(f"unsupported runtime {provider!r} for {args.suite} suite")
    families = (
        runtime_families(provider) if args.suite == "runtime" else ("Multimodal",)
    )
    models = manifest["models"]
    if len(models) != len(families) or {m["name"] for m in models} != set(families):
        raise ValueError(f"{args.suite} manifest must cover exactly {families}")
    if args.suite == "multimodal" and (
        models[0].get("env") != "MULTIMODAL_MODEL_PATH" or not models[0].get("path")
    ):
        raise ValueError(
            "multimodal manifest requires an explicit MULTIMODAL_MODEL_PATH"
        )
    env = {
        **os.environ,
        "VLLM_SR_REQUIRE_MODEL_TESTS": "1",
        "VLLM_SR_MODEL_TEST_PROVIDER": provider,
        "VLLM_SR_MODEL_TEST_DEVICE": args.device,
        "CGO_ENABLED": "1",
    }
    for model in manifest["models"]:
        env[model["env"]] = model["path"]
        if model["name"] == "Domain":
            env["CANDLE_GENERIC_CLASSIFIER_MODEL"] = model["path"]
        if model["name"] == "Embedding" and provider == "candle":
            env["VLLM_SR_MMBERT_TEST_MODEL"] = model["path"]
    args.output.mkdir(parents=True, exist_ok=True)
    suites = []
    selections = (
        (
            ROOT / "src/semantic-router",
            "./pkg/modelruntime/native",
            ("TestPublishedVelaModels",),
            {"TestPublishedVelaModels/" + name for name in FAMILIES},
            "native.jsonl",
        ),
        (
            ROOT / "src/semantic-router",
            "./pkg/classification",
            CLASSIFIER_TESTS,
            set(CLASSIFIER_TESTS),
            "classification.jsonl",
        ),
    )
    if args.suite == "multimodal":
        selections = (
            (
                ROOT / "candle-binding",
                ".",
                MULTIMODAL_BINDING_TESTS,
                set(MULTIMODAL_BINDING_TESTS),
                "binding.jsonl",
            ),
        )
    elif provider == "candle":
        selections += (
            (
                ROOT / "src/semantic-router",
                "./pkg/cache",
                CANDLE_CACHE_TESTS,
                set(CANDLE_CACHE_TESTS),
                "cache.jsonl",
            ),
        )
    elif provider == "ort":
        selections += (
            (
                ROOT / "src/semantic-router",
                "./pkg/modelruntime",
                ("TestOwnedImplicitORTEmbeddingAndExplicitCandleOverride",),
                {"TestOwnedImplicitORTEmbeddingAndExplicitCandleOverride"},
                "default-execution.jsonl",
            ),
        )
    if args.suite == "runtime" and provider == "candle":
        env.update(VLLM_SR_REQUIRE_HALU_TESTS="1", VLLM_SR_HALU_REFERENCE="")
        selections += (
            (
                ROOT / "src/semantic-router",
                "./pkg/modelruntime/native",
                ("TestPublishedVelaHalu",),
                set(HALU_CASES),
                "grounding.jsonl",
            ),
        )
    elif args.suite == "runtime" and provider == "ort":
        env.update(
            REQUIRE_OMNI_TESTS="1",
            VELA_OMNI_ARTIFACT=next(
                model["path"] for model in models if model["name"] == "OmniNano"
            ),
        )
        selections += (
            (
                ROOT / "src/semantic-router",
                "./pkg/modelruntime/native",
                ("TestPublishedOmniModels",),
                set(OMNI_CASES),
                "omni.jsonl",
            ),
        )
        selections += tuple(
            (
                ROOT / "src/semantic-router",
                "./pkg/" + package,
                names,
                set(names),
                "omni-" + package + ".jsonl",
            )
            for package, names in OWNED_OMNI_TESTS.items()
        )
    for cwd, package, names, expected, filename in selections:
        suites.append(
            run_suite(package, names, expected, env, args.output / filename, cwd=cwd)
        )
    report = {
        "source_sha": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "environment": "native",
        "suite": args.suite,
        "provider": provider,
        "device": args.device,
        "models": manifest["models"],
        "suites": suites,
        "success": all(s["success"] for s in suites),
    }
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0 if report["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
