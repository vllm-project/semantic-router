"""GPU integration: BF16-resident Linear weights give byte-identical answers.

Run in the pinned image on one leased GPU, with PYTHONPATH at the mirror's
src/training/decision2 and a real Qwen3 tokenizer directory:

    python3 -m v2.release.tests.gpu_bf16_resident --tokenizer QWEN3_DIR --work NEW_DIR

Builds tiny real packages exactly as a release would: a two-layer Qwen3
backbone and a two-layer Qwen3.5 hybrid text backbone (gated-delta + full
attention) as ``qwen-full``, and a LoRA continuation of the first as
``qwen-adapter``, with every backbone Linear weight stored as BF16 rounds it,
like the released BF16 packages, and FP32 LoRA factors. Each package answers
the release examples in fresh isolated interpreters (``examples.py run``) on
cuda:0 with the default runtime (BF16-resident) and with ``--fp32-master``
(the previous runtime), and the Qwen3 packages once on CPU (the image's
causal-conv1d kernel is GPU-only). Passes if both GPU runs give byte-identical
answers, the BF16-resident run holds every backbone Linear weight in BF16
except the LoRA factors, and the FP32-master and CPU runs hold every
parameter in FP32. Writes RESULT.json; exits non-zero on any failure.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from v2.release.tests.cpu_integration import isolated, spec_for


def round_linear_weights(module, torch) -> None:
    with torch.no_grad():
        for layer in module.modules():
            if isinstance(layer, torch.nn.Linear):
                layer.weight.copy_(layer.weight.to(torch.bfloat16).float())


def qwen3_5_checkpoint(path: Path, tokenizer) -> None:
    """A Qwen3.5 hybrid text backbone with a shared head, saved as a full checkpoint."""
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextModel

    import torch
    from training.model.data import MAX_OPTIONS
    from training.model.decision_model import (
        ARCHITECTURE,
        PROMPT_VERSION,
        CandidateHead,
        DecisionModel,
    )

    config = Qwen3_5TextConfig(
        vocab_size=151936,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        layer_types=["linear_attention", "full_attention"],
        max_position_embeddings=4096,
    )
    backbone = Qwen3_5TextModel(config)
    backbone.config.use_cache = False
    metadata = {
        "architecture": ARCHITECTURE,
        "head_variant": "shared",
        "backbone_model_type": "qwen3_5",
        "prompt_version": PROMPT_VERSION,
        "base_revision": "0" * 40,
        "source_stage": "base",
        "initialization": "base-random-head",
        "head_dim": 16,
        "max_options": MAX_OPTIONS,
        "text_parameter_count": sum(p.numel() for p in backbone.parameters()),
        "parameter_dtype": "float32",
        "autocast_dtype": "bfloat16",
        "head_compute_dtype": "float32",
        "attention": "sdpa",
    }
    model = DecisionModel(backbone, CandidateHead(64, 16), metadata)
    round_linear_weights(model.backbone, torch)
    model.save(path, tokenizer)


def check(
    pkg: Path, out: Path, extra: list[str], lora_factors: int, cpu_leg: bool = True
) -> dict:
    def receipt(name: str) -> dict:
        path = out / f"{name}.json"
        return json.loads(path.read_text()) if path.is_file() else {}

    steps = {
        "gpu_resident": isolated(
            [
                "run",
                "--package",
                str(pkg),
                "--output",
                str(out / "gpu-resident.json"),
                "--device",
                "cuda:0",
                *extra,
            ]
        ),
        "gpu_fp32_master": isolated(
            [
                "run",
                "--package",
                str(pkg),
                "--output",
                str(out / "gpu-fp32.json"),
                "--device",
                "cuda:0",
                "--fp32-master",
                *extra,
            ]
        ),
    }
    if cpu_leg:
        steps["cpu"] = isolated(
            [
                "run",
                "--package",
                str(pkg),
                "--output",
                str(out / "cpu.json"),
                "--device",
                "cpu",
                *extra,
            ]
        )
    steps["compare"] = isolated(
        [
            "compare",
            str(out / "gpu-resident.json"),
            str(out / "gpu-fp32.json"),
            "--output",
            str(out / "compare.json"),
            "--tolerance",
            "0",
        ]
    )
    resident, master, cpu = (receipt(n) for n in ("gpu-resident", "gpu-fp32", "cpu"))
    compared = receipt("compare")
    weights = {
        name: (r.get("runtime") or {}).get("residency")
        for name, r in (
            ("gpu_resident", resident),
            ("gpu_fp32_master", master),
            ("cpu", cpu),
        )
    }
    dtypes = {
        name: (r.get("runtime") or {}).get("parameter_dtypes")
        for name, r in (
            ("gpu_resident", resident),
            ("gpu_fp32_master", master),
            ("cpu", cpu),
        )
    }
    counts = weights["gpu_resident"] or {}
    checks = {
        "steps_passed": all(step["exit"] == 0 for step in steps.values()),
        "bit_identical_answers": compared.get("bit_identical_answers") is True,
        "resident_backbone_linear_bf16": counts.get("linear_bf16", 0) > 0
        and counts.get("linear_fp32") == lora_factors,
        "resident_holds_bf16": (dtypes["gpu_resident"] or {}).get("bfloat16", 0) > 0,
        "fp32_master_all_fp32": weights["gpu_fp32_master"] is None
        and set(dtypes["gpu_fp32_master"] or {"?": 0}) == {"float32"},
    }
    if cpu_leg:
        checks["cpu_all_fp32"] = weights["cpu"] is None and set(
            dtypes["cpu"] or {"?": 0}
        ) == {"float32"}
    return {
        "steps": steps,
        "residency": weights,
        "parameter_dtypes": dtypes,
        "answers_sha256": {
            "gpu_resident": resident.get("answers_sha256"),
            "gpu_fp32_master": master.get("answers_sha256"),
            "cpu": cpu.get("answers_sha256"),
        },
        "checks": checks,
        "passed": all(checks.values()),
    }


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

    if not torch.cuda.is_available():
        raise SystemExit("needs one GPU")
    work = args.work
    work.mkdir(parents=True)
    base = work / "base"
    base.mkdir()
    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
        if (args.tokenizer / name).is_file():
            shutil.copyfile(args.tokenizer / name, base / name)
    tokenizer = AutoTokenizer.from_pretrained(base)
    torch.manual_seed(20261001)
    Qwen3ForCausalLM(
        Qwen3Config(
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
    ).save_pretrained(base, safe_serialization=True)
    model, tokenizer = DecisionModel.from_base(base, "0" * 40, head_dim=16)
    round_linear_weights(model.backbone, torch)
    model.save(work / "full", tokenizer)
    qwen3_5_checkpoint(work / "full35", tokenizer)
    licence = work / "LICENSE"
    licence.write_text("Apache License 2.0 (test)\n")
    layout.name_for = lambda parameters: "DEV2.0-0.6B"

    def package(name: str, profile: str, checkpoint: Path, identity: str, base_block):
        cal = work / f"cal-{name}.json"
        cal.write_text(
            json.dumps(
                {
                    "calibration_version": "decision2-per-type-temperature/1",
                    "model_sha256": identity,
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
        spec = spec_for(profile, checkpoint, identity, cal, licence, base_block)
        (work / f"spec-{name}.json").write_text(json.dumps(spec))
        target = work / f"pkg-{name}" / "dev2-release-staging"
        build.build(work / f"spec-{name}.json", target)
        (work / f"out-{name}").mkdir()
        return target

    results = {}
    for name, checkpoint in (
        ("qwen3-full", work / "full"),
        ("qwen3_5-full", work / "full35"),
    ):
        identity = checkpoint_fingerprint(checkpoint)["model_sha256"]
        pkg = package(name, "qwen-full", checkpoint, identity, None)
        # The image's causal-conv1d kernel is GPU-only (its Qwen3.5 cards say CPU is not verified).
        results[name] = check(pkg, work / f"out-{name}", [], 0, name == "qwen3-full")

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
    factors = 2 * len(model.metadata["lora"]["target_modules"])
    identity = checkpoint_fingerprint(work / "lora", work / "full")["model_sha256"]
    base_block = {
        "repo_id": "llm-semantic-router/dev2-release-staging-base",
        "revision": "0" * 40,
        "path": str(work / "full"),
        "licence": "apache-2.0",
        "redistribution": "own model",
    }
    pkg = package("qwen3-adapter", "qwen-adapter", work / "lora", identity, base_block)
    results["qwen3-adapter"] = check(
        pkg, work / "out-qwen3-adapter", ["--base-path", str(work / "full")], factors
    )
    results["passed"] = all(r["passed"] for r in results.values())
    (work / "RESULT.json").write_text(json.dumps(results, indent=2, sort_keys=True))
    print(
        json.dumps(
            {k: (v["passed"] if isinstance(v, dict) else v) for k, v in results.items()}
        )
    )
    sys.exit(0 if results["passed"] else 1)


if __name__ == "__main__":
    main()
