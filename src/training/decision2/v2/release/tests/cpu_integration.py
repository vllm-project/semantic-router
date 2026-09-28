"""CPU integration: tiny real qwen-full and qwen-adapter packages, built, loaded and queried.

Run in the pinned image (torch, Transformers 5.x, PEFT) with PYTHONPATH at the
mirror's src/training/decision2 and a real Qwen3 tokenizer directory:

    python3 -m v2.release.tests.cpu_integration --tokenizer QWEN3_DIR --work NEW_DIR

A random two-layer Qwen3 backbone is saved through the shared training code,
packaged exactly as a release would be (the size-name rule is relaxed for the
toy model only), then each package is loaded in fresh isolated interpreters
(``python -I -B``) for the native examples, a cross-process comparison and the
card's own Python block. Writes RESULT.json; exits non-zero on any failure.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EXAMPLES = ROOT / "v2/release/examples.py"
REPORTS = ROOT / "v2/eval/records/m1-reports"


def spec_for(
    profile: str,
    checkpoint: Path,
    identity: str,
    cal: Path,
    licence: Path,
    base: dict | None,
) -> dict:
    from v2.release import build, layout

    spec = {
        "schema": build.SPEC_SCHEMA,
        "kind": "staging",
        "repo_id": "llm-semantic-router/dev2-release-staging",
        "model_name": "DEV2.0-0.6B",
        "profile": profile,
        "checkpoint": str(checkpoint),
        "expected_identity": {"model_sha256": identity},
        "calibration": {"path": str(cal), "sha256": layout.sha_file(cal)},
        "max_input_tokens": 2048,
        "origin": {
            "repo_id": "llm-semantic-router/dev2-release-staging",
            "revision": "0" * 40,
            "relation": "finetune",
            "summary": "Toy random model for pipeline tests.",
        },
        "licence": {
            "components": [{"component": "toy weights", "licence": "apache-2.0"}],
            "files": [
                {
                    "source": str(licence),
                    "path": "LICENSE",
                    "sha256": layout.sha_file(licence),
                }
            ],
            "attributions": ["Toy test model."],
        },
        "card": {
            "reports": [
                {
                    "key": "cand",
                    "role": "candidate",
                    "report": str(REPORTS / "lex.json"),
                    "label": "DEV2.0-0.6B (test)",
                },
                {
                    "key": "kai1",
                    "role": "own-1.0",
                    "report": str(REPORTS / "kai1.json"),
                    "repo_id": "llm-semantic-router/Decision-1.0-Kai-0.6B",
                },
            ],
            "roster": "v2/eval/records/decision-index-peer-roster-2026-09-28.json",
            "text": {"tagline": "Toy.", "staging_notice": "Integration test package."},
            "requirements_text": "Tested with Transformers 5.",
        },
    }
    if base:
        spec["base"] = base
    return spec


def isolated(args: list[str]) -> dict:
    completed = subprocess.run(
        [sys.executable, "-I", "-B", str(EXAMPLES), *args],
        capture_output=True,
        text=True,
    )
    return {
        "exit": completed.returncode,
        "stdout": completed.stdout[-500:],
        "stderr": completed.stderr[-3000:],
    }


def check_package(pkg: Path, out: Path, extra: list[str], card: bool) -> dict:
    result = {
        "run_a": isolated(
            [
                "run",
                "--package",
                str(pkg),
                "--output",
                str(out / "a.json"),
                "--device",
                "cpu",
                *extra,
            ]
        ),
        "run_b": isolated(
            [
                "run",
                "--package",
                str(pkg),
                "--output",
                str(out / "b.json"),
                "--device",
                "cpu",
                *extra,
            ]
        ),
    }
    result["compare"] = isolated(
        [
            "compare",
            str(out / "a.json"),
            str(out / "b.json"),
            "--output",
            str(out / "cmp.json"),
        ]
    )
    if card:
        result["card"] = isolated(
            [
                "card",
                "--package",
                str(pkg),
                "--reference",
                str(out / "a.json"),
                "--output",
                str(out / "card.json"),
            ]
        )
    result["passed"] = all(
        step["exit"] == 0 for step in result.values() if isinstance(step, dict)
    )
    if (out / "a.json").is_file():
        run = json.loads((out / "a.json").read_text())
        result["loaded_parameters"] = run["loaded_parameters"]
        result["checks"] = run["checks"]
        result["sample"] = run["outputs"][0]["response"]["answers"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    import torch
    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

    from training.model.decision_model import DecisionModel
    from training.model.infer import checkpoint_fingerprint
    from training.model.lora import attach_lora
    from training.model.source import source_fingerprint
    from v2.release import build, layout

    work = args.work
    work.mkdir(parents=True)
    base = work / "base"
    base.mkdir()
    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
        if (args.tokenizer / name).is_file():
            shutil.copyfile(args.tokenizer / name, base / name)
    tokenizer = AutoTokenizer.from_pretrained(base)
    torch.manual_seed(20260928)
    config = Qwen3Config(
        vocab_size=151936,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=4096,
        tie_word_embeddings=True,
    )
    Qwen3ForCausalLM(config).save_pretrained(base, safe_serialization=True)
    config_path = base / "config.json"
    model, tokenizer = DecisionModel.from_base(base, "0" * 40, head_dim=16)
    model.save(work / "full", tokenizer)
    licence = work / "LICENSE"
    licence.write_text("Apache License 2.0 (test)\n")

    def calibration(path: Path, model_sha: str) -> Path:
        path.write_text(
            json.dumps(
                {
                    "calibration_version": "decision2-per-type-temperature/1",
                    "model_sha256": model_sha,
                    **{
                        k: "0" * 64
                        for k in (
                            "checkpoint_sha256",
                            "cal_sha256",
                            "best_sha256",
                            "complete_sha256",
                            "provenance_sha256",
                        )
                    },
                    "fit_split": "cal",
                    "selection_policy": "completed_run_best_only",
                    "temperature_by_type": {"choice": 1.2, "noul": 0.8, "score": 1.5},
                    "inference": {"max_length": 2048},
                }
            )
        )
        return path

    layout.name_for = lambda parameters: "DEV2.0-0.6B"
    results = {"config_sha256": layout.sha_file(config_path)}

    full_id = checkpoint_fingerprint(work / "full")["model_sha256"]
    spec = spec_for(
        "qwen-full",
        work / "full",
        full_id,
        calibration(work / "cal-full.json", full_id),
        licence,
        None,
    )
    (work / "spec-full.json").write_text(json.dumps(spec))
    build.build(work / "spec-full.json", work / "pkg-full" / "dev2-release-staging")
    (work / "out-full").mkdir()
    results["qwen-full"] = check_package(
        work / "pkg-full" / "dev2-release-staging", work / "out-full", [], card=True
    )

    model, tokenizer = DecisionModel.from_checkpoint(work / "full")
    attach_lora(
        model,
        rank=4,
        alpha=8,
        dropout=0.0,
        source_kind="decision2",
        source_fingerprint=source_fingerprint(work / "full"),
    )
    with torch.no_grad():
        for name, parameter in model.backbone.named_parameters():
            if "lora_B" in name:
                parameter.normal_(0, 0.05)
    model.save(work / "lora", tokenizer)
    lora_id = checkpoint_fingerprint(work / "lora", work / "full")["model_sha256"]
    base_block = {
        "repo_id": "llm-semantic-router/dev2-release-staging-base",
        "revision": "0" * 40,
        "path": str(work / "full"),
        "licence": "apache-2.0",
        "redistribution": "own model",
    }
    spec = spec_for(
        "qwen-adapter",
        work / "lora",
        lora_id,
        calibration(work / "cal-lora.json", lora_id),
        licence,
        base_block,
    )
    (work / "spec-lora.json").write_text(json.dumps(spec))
    build.build(work / "spec-lora.json", work / "pkg-lora" / "dev2-release-staging")
    (work / "out-lora").mkdir()
    results["qwen-adapter"] = check_package(
        work / "pkg-lora" / "dev2-release-staging",
        work / "out-lora",
        ["--base-path", str(work / "full")],
        card=False,
    )
    full_sample = results["qwen-full"].get("sample", {})
    lora_sample = results["qwen-adapter"].get("sample", {})
    results["adapter_changes_outputs"] = full_sample != lora_sample
    results["passed"] = (
        results["qwen-full"]["passed"]
        and results["qwen-adapter"]["passed"]
        and results["adapter_changes_outputs"]
    )
    (work / "RESULT.json").write_text(json.dumps(results, indent=2, sort_keys=True))
    print(
        json.dumps(
            {k: (v["passed"] if isinstance(v, dict) else v) for k, v in results.items()}
        )
    )
    sys.exit(0 if results["passed"] else 1)


if __name__ == "__main__":
    main()
