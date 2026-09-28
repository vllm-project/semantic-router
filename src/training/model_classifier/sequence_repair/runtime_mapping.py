"""Serialize classifier label sidecars consumed by the router's Go loaders."""

import json
from pathlib import Path

RUNTIME_TASKS = (
    "domain",
    "fact-check",
    "modality",
    "feedback",
    "prompt-guard",
    "safety",
    "hazard",
)


def write_runtime_mappings(destination, labels, task):
    """Write both HF and router layouts from one contiguous label bijection."""
    if task not in RUNTIME_TASKS:
        raise ValueError("Unknown runtime classifier task")
    if not labels or set(labels.values()) != set(range(len(labels))):
        raise ValueError("Runtime label IDs must be a contiguous bijection")
    if task == "modality" and labels != {"AR": 0, "DIFFUSION": 1, "BOTH": 2}:
        raise ValueError("Modality must preserve the native AR/DIFFUSION/BOTH IDs")
    if task == "fact-check" and set(labels) != {
        "NO_FACT_CHECK_NEEDED",
        "FACT_CHECK_NEEDED",
    }:
        raise ValueError("FactCheck requires the router's two semantic labels")
    reverse = {str(index): label for label, index in labels.items()}
    generic = {
        "label2id": labels,
        "id2label": reverse,
        "label_to_idx": labels,
        "idx_to_label": reverse,
    }
    files = {"label_mapping.json": generic}
    if task == "domain":
        files["category_mapping.json"] = {
            "category_to_idx": labels,
            "idx_to_category": reverse,
        }
    sidecars = {
        "fact-check": "fact_check_mapping.json",
        "modality": "modality_mapping.json",
        "prompt-guard": "jailbreak_type_mapping.json",
    }
    if task in sidecars:
        files[sidecars[task]] = {
            "label_to_idx": labels,
            "idx_to_label": reverse,
        }
    for name, data in files.items():
        path = Path(destination) / name
        if path.exists():
            raise ValueError("Refusing to overwrite an existing runtime mapping")
        path.write_text(json.dumps(data, indent=2) + "\n")
    return sorted(files)
