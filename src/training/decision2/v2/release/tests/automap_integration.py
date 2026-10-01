"""Integration: tiny real packages through stock Transformers with ``trust_remote_code=True``.

Run in the pinned image with PYTHONPATH at the mirror's src/training/decision2 and a real Qwen3 tokenizer
directory (``--device cuda:0`` with ``--site`` kernel directories for the GPU leg):

    python3 -m v2.release.tests.automap_integration --tokenizer QWEN3_DIR --work NEW_DIR [--device cuda:0]

Builds a Qwen3 and a Qwen3.5 hybrid ``qwen-full`` package and a Qwen3 LoRA ``qwen-adapter`` package with
the release builder. For each: the native examples (``examples.py run``), ``examples.py automap`` against
them (AutoConfig / AutoTokenizer / AutoModel / pipeline, bit-identical answers) and the card's
Transformers block (``automap-card``). The native CPU reference of the Qwen3.5 package runs with the
image's GPU-only causal-conv1d kernel hidden (as on a machine without it), while ``automap`` keeps it
installed, so its per-layer reference forwards must reproduce that run exactly. Then, in this process:
loading by repository ID from an offline Hugging Face cache layout of the package (files linked to
blobs, so through the hard-link view); the same with an ``adapter_config.json`` at the adapter
repository's root, where Transformers' PEFT detection loads the named base instead (the pitfall the
``adapter/`` layout avoids); dtype / cast / save refusals and a device move. Writes RESULT.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

from v2.release.tests.cpu_integration import isolated, spec_for
from v2.release.tests.gpu_bf16_resident import qwen3_5_checkpoint, round_linear_weights

STAGING = "llm-semantic-router/dev2-release-staging"


def hub_cache(package: Path, cache: Path, repo_id: str, commit: str) -> Path:
    """An offline Hugging Face cache entry for ``repo_id``: snapshot files linked to their blobs."""
    folder = cache / ("models--" + repo_id.replace("/", "--"))
    snapshot = folder / "snapshots" / commit
    (folder / "refs").mkdir(parents=True)
    (folder / "refs" / "main").write_text(commit)
    for path in sorted(package.rglob("*")):
        if path.is_dir() or ".cache" in path.parts:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        blob = folder / "blobs" / digest
        blob.parent.mkdir(parents=True, exist_ok=True)
        if not blob.exists():
            shutil.copyfile(path, blob)
        link = snapshot / path.relative_to(package)
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(os.path.relpath(blob, link.parent))
    return snapshot


def cpu_lapack(torch) -> bool:
    try:
        torch.linalg.solve_triangular(torch.eye(2), torch.ones(2, 1), upper=False)
        return True
    except RuntimeError:
        return False


def step(name: str, args: list[str], results: dict) -> dict:
    results[name] = isolated(args)
    return results[name]


def receipt(path: Path) -> dict:
    return json.loads(path.read_text()) if path.is_file() else {}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--site", action="append", default=[])
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
    licence = work / "LICENSE"
    licence.write_text("Apache License 2.0 (test)\n")
    layout.name_for = lambda parameters: "Decision-2.0-Kai-0.6B"
    hidden = work / "no-causal-conv1d"
    (hidden / "causal_conv1d").mkdir(parents=True)
    (hidden / "causal_conv1d" / "__init__.py").write_text(
        'raise ImportError("hidden: a machine without the GPU-only causal-conv1d kernel")\n'
    )
    base_block = {
        "repo_id": "llm-semantic-router/dev2-release-staging-base",
        "revision": "0" * 40,
        "path": str(work / "full"),
        "licence": "apache-2.0",
        "redistribution": "own model",
    }

    def package(name: str, profile: str, checkpoint: Path, base: dict | None) -> Path:
        identity = checkpoint_fingerprint(
            checkpoint, Path(base["path"]) if base else None
        )["model_sha256"]
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
        spec = spec_for(profile, checkpoint, identity, cal, licence, base)
        spec["remote_code"] = {"tested": ["(integration test)"]}
        (work / f"spec-{name}.json").write_text(json.dumps(spec))
        target = work / f"pkg-{name}" / "dev2-release-staging"
        build.build(work / f"spec-{name}.json", target)
        (work / f"out-{name}").mkdir()
        return target

    device = ["--device", args.device, *sum((["--site", s] for s in args.site), [])]
    results: dict = {}
    packages = {
        "qwen3-full": package("qwen3-full", "qwen-full", work / "full", None),
        "qwen3_5-full": package("qwen3_5-full", "qwen-full", work / "full35", None),
        "qwen3-adapter": package(
            "qwen3-adapter", "qwen-adapter", work / "lora", base_block
        ),
    }
    for name, pkg in packages.items():
        out = work / f"out-{name}"
        if name == "qwen3_5-full" and args.device == "cpu" and not cpu_lapack(torch):
            results[name] = {
                "skipped": "this PyTorch has no CPU LAPACK (solve_triangular), which the Qwen3.5 "
                "reference gated-delta path needs on CPU",
                "passed": True,
            }
            continue
        extra = ["--base-path", str(work / "full")] if "adapter" in name else []
        native_sites = (
            ["--site", str(hidden)]
            if name == "qwen3_5-full" and args.device == "cpu"
            else []
        )
        legs = {}
        step(
            "native",
            ["run", "--package", str(pkg), "--output", str(out / "native.json")]
            + device
            + native_sites
            + extra,
            legs,
        )
        step(
            "automap",
            [
                "automap",
                "--package",
                str(pkg),
                "--reference",
                str(out / "native.json"),
                "--output",
                str(out / "automap.json"),
            ]
            + device
            + extra,
            legs,
        )
        if not extra:
            step(
                "automap_card",
                [
                    "automap-card",
                    "--package",
                    str(pkg),
                    "--reference",
                    str(out / "native.json"),
                    "--output",
                    str(out / "automap-card.json"),
                ]
                + sum((["--site", s] for s in args.site), []),
                legs,
            )
        automap = receipt(out / "automap.json")
        results[name] = {
            "steps": {k: v["exit"] for k, v in legs.items()},
            "stderr": {k: v["stderr"] for k, v in legs.items() if v["exit"]},
            "checks": automap.get("checks"),
            "cpu_reference_layers": (automap.get("checks") or {})
            .get("model", {})
            .get("cpu_reference_layers"),
            "passed": all(v["exit"] == 0 for v in legs.values()),
        }
    if args.device == "cpu" and "skipped" not in results["qwen3_5-full"]:
        results["qwen3_5-full"]["passed"] &= (
            results["qwen3_5-full"]["cpu_reference_layers"] or 0
        ) > 0
    cache = work / "hf-cache"
    commit = "1" * 40
    hub_cache(packages["qwen3-full"], cache, STAGING, commit)
    hub_cache(packages["qwen3-adapter"], cache, f"{STAGING}-adapter", commit)
    hub_cache(work / "base", cache, f"{STAGING}-base", commit)
    rooted = work / "rooted"
    shutil.copytree(packages["qwen3-adapter"], rooted)
    config = json.loads((rooted / "adapter" / "adapter_config.json").read_text())
    (rooted / "adapter_config.json").write_text(
        json.dumps({**config, "base_model_name_or_path": f"{STAGING}-base"})
    )
    shutil.copyfile(
        rooted / "adapter" / "adapter_model.safetensors",
        rooted / "adapter_model.safetensors",
    )
    hub_cache(rooted, cache, f"{STAGING}-rooted", commit)
    env = {
        **os.environ,
        "HF_HUB_CACHE": str(cache),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    import subprocess

    completed = subprocess.run(
        [sys.executable, "-m", "v2.release.tests.automap_integration", "--hub-checks"]
        + [
            "--tokenizer",
            str(args.tokenizer),
            "--work",
            str(work),
            "--device",
            args.device,
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    results["hub_layout"] = {
        **receipt(work / "hub-checks.json"),
        "exit": completed.returncode,
        "stderr": completed.stderr[-3000:] if completed.returncode else "",
    }
    results["hub_layout"]["passed"] = (
        completed.returncode == 0 and results["hub_layout"].get("passed") is True
    )
    results["passed"] = all(r["passed"] for r in results.values())
    (work / "RESULT.json").write_text(json.dumps(results, indent=2, sort_keys=True))
    print(
        json.dumps(
            {k: (v["passed"] if isinstance(v, dict) else v) for k, v in results.items()}
        )
    )
    sys.exit(0 if results["passed"] else 1)


def hub_checks(work: Path, device: str) -> None:
    """By repository ID from the offline cache (HF_HUB_CACHE); the PEFT root-adapter pitfall; refusals."""
    import torch
    from transformers import AutoModel

    from v2.release.examples import EXAMPLES, compare_answers

    cache = Path(os.environ["HF_HUB_CACHE"])
    first = EXAMPLES[0]
    expected = receipt(work / "out-qwen3-full" / "native.json")["outputs"][0][
        "response"
    ]
    checks: dict = {}
    model = AutoModel.from_pretrained(STAGING, trust_remote_code=True, device=device)
    response = model.system_one(state=first["state"], questions=first["questions"])
    checks["by_repo_id"] = response == expected
    checks["view_removed"] = not any(
        p.name.startswith(".decision2-view-") for p in cache.rglob("*")
    )
    refused = []
    for call in (
        model.half,
        model.float,
        model.bfloat16,
        lambda: model.to(torch.float16),
    ):
        try:
            call()
            refused.append(False)
        except TypeError:
            refused.append(True)
    checks["casts_refused"] = all(refused)
    try:
        model.save_pretrained(work / "saved")
        checks["save_refused"] = False
    except NotImplementedError:
        checks["save_refused"] = True
    if device != "cpu":
        model.to("cpu")
        checks["moved_to_cpu"] = str(model.device) == "cpu"
        moved = model.system_one(state=first["state"], questions=first["questions"])
        checks["moved_same_decisions"] = (
            compare_answers(moved["answers"], expected["answers"])["category_changes"]
            == 0
        )
    del model
    try:
        AutoModel.from_pretrained(STAGING, trust_remote_code=True, dtype=torch.bfloat16)
        checks["dtype_refused"] = False
    except ValueError:
        checks["dtype_refused"] = True
    loaded = AutoModel.from_pretrained(
        f"{STAGING}-adapter",
        trust_remote_code=True,
        base_path=str(work / "full"),
        device=device,
    )
    checks["adapter_layout_loads_decision_model"] = (
        type(loaded).__name__ == "Decision2Model"
    )
    del loaded
    # The same adapter repository with its adapter (config and weights) also at the root: Transformers'
    # PEFT detection loads the base the adapter names, with the adapter, instead of the Decision model.
    try:
        redirected = type(
            AutoModel.from_pretrained(f"{STAGING}-rooted", trust_remote_code=True)
        ).__name__
    except Exception as exc:
        redirected = f"error: {type(exc).__name__}"
    checks["root_adapter_config_redirects"] = redirected == "Qwen3Model"
    (work / "hub-checks.json").write_text(
        json.dumps(
            {
                "checks": checks,
                "root_adapter_config_loaded": redirected,
                "passed": all(checks.values()),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    if "--hub-checks" in sys.argv:
        parser = argparse.ArgumentParser()
        parser.add_argument("--hub-checks", action="store_true")
        parser.add_argument("--tokenizer", type=Path)
        parser.add_argument("--work", type=Path, required=True)
        parser.add_argument("--device", default="cpu")
        options = parser.parse_args()
        hub_checks(options.work, options.device)
    else:
        main()
