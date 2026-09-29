"""Derive the DEV2.0-9B and DEV2.0-27B specs from the released DEV2.0-8B / DEV2.0-26B specs.

Only the name (repository, model name, name_basis base, gate receipt, own licence component,
banner, candidate label, naming sentence) changes, plus the three 27B peers' mlx-diag score
files and the 27B multilingual paragraph. Checkpoint, identity, scored run, runtime and every
other card input are copied unchanged. Run from src/training/decision2:

  python3 v2/release/records/dev2-rename-9b-27b-2026-09-29/ops/make_specs.py [--check]
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

SPECS = Path("v2/release/specs")
DECISIONS = "/data/dev2/runs/release/decisions"
PEERS_MLX = "/data/dev2/runs/release/inputs/dev2-27b/peers-mlx"
RENAME = (
    "user directive 2026-09-29 16:05 UTC+8: names follow the base model's size. "
    "llm-semantic-router/{old} was moved to llm-semantic-router/{new} with move_repo on "
    "2026-09-29 (history, revisions and privacy unchanged; the old ID redirects). This spec "
    "builds the card-only revision: name, banner, labels and the naming sentence{extra}; the "
    "weights are byte-identical to the released revision {main}."
)
NINE_B_SIZE = (
    "**Size:** the model is the Qwen3.5 text backbone with a decision head; the Qwen3.5 "
    "vision tower is not part of it."
)
MULTILINGUAL_27B = (
    "**Multilingual.** On the mlx-diag diagnostic (public test splits in seven languages, "
    "English instructions over target-language states), non-English Choice is level with "
    "English (79.1% vs 79.4%) and non-English Noul (paraphrase pairs) is 1.5 points lower "
    "(85.5% vs 87.0%). The weakest languages are Korean Noul (79%, 100 items) and Spanish and "
    "Arabic Choice (76.2% and 77.8%, 126 items each). The three 27B peers ran the same "
    "diagnostic (Choice and Noul parts; no paired test): non-English Choice is 80.6% for "
    "AutoJev-27B, 78.4% for Eikos-27B and 77.6% for Jebadiah-27B (79.1% here), and non-English "
    "Noul is 88.7%, 85.0% and 85.3% (85.5% here). Korean Noul is 85%, 84% and 80% for the "
    "peers (79% here) and Arabic Choice 78.6%, 75.4% and 75.4% (77.8% here)."
)
OLD_MULTILINGUAL_END = "No other 27B model shown has an mlx-diag run yet."


def insert_after(d: dict, anchor: str, key: str, value) -> dict:
    out = {}
    for k, v in d.items():
        if k != key:
            out[k] = v
        if k == anchor:
            out[key] = value
    return out


def rename(spec: dict, old: str, new: str, base_model: str) -> dict:
    spec = copy.deepcopy(spec)
    spec["repo_id"] = f"llm-semantic-router/{new}"
    spec["model_name"] = new
    spec["name_basis"] = "base"
    spec = insert_after(spec, "name_basis", "name_base_model", base_model)
    spec["gate_receipt"] = f"{DECISIONS}/{new}.decision.json"
    own = spec["licence"]["components"][0]
    assert own["source"] == f"llm-semantic-router/{old}", own
    own["component"] = own["component"].replace(old, new)
    own["source"] = f"llm-semantic-router/{new}"
    spec["card"]["banner"] = f"{new}-owl-banner.png"
    candidate = spec["card"]["reports"][0]
    assert candidate["role"] == "candidate" and candidate["label"] == old, candidate
    candidate["label"] = new
    return spec


def nine_b() -> dict:
    spec = rename(
        json.loads((SPECS / "dev2-8b-release.json").read_text(encoding="utf-8")),
        "DEV2.0-8B",
        "DEV2.0-9B",
        "Qwen/Qwen3.5-9B",
    )
    details = spec["card"]["text"]["details"]
    assert details[0].startswith(
        "**Name:** DEV2.0 names follow the loaded parameter count"
    ), details[0]
    details[0] = NINE_B_SIZE
    note = RENAME.format(
        old="DEV2.0-8B",
        new="DEV2.0-9B",
        extra="",
        main="53bac735be58def53673d0d290b9baa3f2af1cf9",
    )
    return insert_after(spec, "schema", "_release", {"rename": note})


def twenty_seven_b() -> dict:
    spec = rename(
        json.loads((SPECS / "dev2-26b-release.json").read_text(encoding="utf-8")),
        "DEV2.0-26B",
        "DEV2.0-27B",
        "Qwen/Qwen3.8-27B",
    )
    assert spec["base"]["repo_id"] == spec["name_base_model"]
    spec["_release"]["name"] = (
        "DEV2.0-27B: named after its base model Qwen/Qwen3.8-27B (name_basis base); it loads "
        "25,746,591,744 parameters (text backbone 25,624,600,064 + rank-16 LoRA 116,727,808 + "
        "head 5,263,872; no vision tower, MTP or LM head). Released first as DEV2.0-26B "
        "(name_basis loaded-parameters)."
    )
    spec["_release"]["rename"] = RENAME.format(
        old="DEV2.0-26B",
        new="DEV2.0-27B",
        extra=(
            ", plus the three 27B peers' mlx-diag Choice / Noul results (eval record "
            "v2/eval/records/m5-mlx-diag-27b-peers-2026-09-29.md; score files copied to node B "
            "and re-hashed)"
        ),
        main="6931828d7e41a5d31cc8e5acdc36f5eebae70fad",
    )
    files = {
        "autojev27": "autojev27.mlx-diag.score.json",
        "eikos27": "eikos27b.mlx-diag.score.json",
        "jebadiah27": "jebadiah27b.mlx-diag.score.json",
    }
    for entry in spec["card"]["reports"][1:]:
        assert "mlx" not in entry, entry
        entry["mlx"] = f"{PEERS_MLX}/{files[entry['key']]}"
    limitations = spec["card"]["text"]["limitations"]
    (index,) = [
        i for i, item in enumerate(limitations) if item.startswith("**Multilingual.**")
    ]
    assert limitations[index].endswith(OLD_MULTILINGUAL_END), limitations[index]
    limitations[index] = MULTILINGUAL_27B
    return spec


def leftovers(spec: dict, old: str) -> list[str]:
    body = {k: v for k, v in spec.items() if k != "_release"}
    text = json.dumps(body, ensure_ascii=False)
    return [old] if old in text else []


def main() -> int:
    check = "--check" in sys.argv[1:]
    problems = []
    for name, spec, old in (
        ("dev2-9b-release.json", nine_b(), "DEV2.0-8B"),
        ("dev2-27b-release.json", twenty_seven_b(), "DEV2.0-26B"),
    ):
        problems += [f"{name}: still names {o}" for o in leftovers(spec, old)]
        rendered = json.dumps(spec, ensure_ascii=False, indent=2) + "\n"
        target = SPECS / name
        if check:
            if not target.is_file() or target.read_text(encoding="utf-8") != rendered:
                problems.append(f"{name}: differs from the derivation")
        else:
            target.write_text(rendered, encoding="utf-8")
            print(f"wrote {target}")
    for p in problems:
        print(p, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
