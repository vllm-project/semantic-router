"""Build reviewed text-only control and ordinary-request contrast families."""

import argparse
import hashlib
import json
import unicodedata
from collections import Counter
from pathlib import Path

DEFAULT_CONFIG = Path(__file__).parent / "configs" / "vela-source-boundary-v1.json"


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def build_rows(config):
    """Keep translations, ordinary requests, attacks and quotations in one group."""
    rows = []
    family_ids = set()
    seen_texts = set()
    for family in config["families"]:
        identifier = family["id"]
        split = family["split"]
        if not identifier or identifier in family_ids:
            raise ValueError("Family IDs must be nonempty and unique")
        family_ids.add(identifier)
        if split not in {"train", "development"}:
            raise ValueError("Unknown family split")
        if set(family["languages"]) != {"en", "zh"}:
            raise ValueError("Each family must contain both translations")
        for language, texts in family["languages"].items():
            required = {
                "ordinary_safe",
                "ordinary_harmful",
                "control",
                "quoted_control",
            }
            if set(texts) != required or any(
                not isinstance(value, str) or not value.strip()
                for value in texts.values()
            ):
                raise ValueError("A family has an incomplete text contract")
            if texts["control"] not in texts["quoted_control"]:
                raise ValueError("Quotation must preserve the complete control text")
            contrasts = [
                ("ordinary_safe", texts["ordinary_safe"], "benign", "benign"),
                ("ordinary_harmful", texts["ordinary_harmful"], "benign", "harmful"),
                (
                    "attack_safe",
                    texts["control"] + "\n" + texts["ordinary_safe"],
                    "jailbreak",
                    "benign",
                ),
                (
                    "attack_harmful",
                    texts["control"] + "\n" + texts["ordinary_harmful"],
                    "jailbreak",
                    "harmful",
                ),
                ("quoted_control", texts["quoted_control"], "benign", "benign"),
            ]
            for condition, text, label, risk in contrasts:
                key = normalized(text)
                if key in seen_texts:
                    raise ValueError("Duplicate normalized text across contrast rows")
                seen_texts.add(key)
                digest = hashlib.sha256(text.encode()).hexdigest()
                rows.append(
                    {
                        "id": "vela-pg-boundary:" + digest,
                        "group_id": "vela-pg-boundary:" + identifier,
                        "family": identifier,
                        "split": split,
                        "text": text,
                        "text_sha256": digest,
                        "label": label,
                        "source": "authored_promptguard_source_boundary_v1",
                        "language": language,
                        "length_bucket": "authored_short",
                        "condition": condition,
                        "content_risk": risk,
                        "supervision": "reviewed_authored_current_visible_text",
                    }
                )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Refusing to overwrite a frozen corpus")
    config = json.loads(args.config.read_text())
    rows = build_rows(config)
    args.output.mkdir(parents=True)
    manifest = {
        "version": config["version"],
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "implementation_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "split_unit": config["split_unit"],
        "limitations": config["development_limit"],
        "model_predictions_used_for_authoring": False,
        "files": {},
    }
    for split in ["train", "development"]:
        selected = [row for row in rows if row["split"] == split]
        path = args.output / (split + ".jsonl")
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=True) + "\n" for row in selected)
        )
        manifest["files"][path.name] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "rows": len(selected),
            "groups": len({row["group_id"] for row in selected}),
            "labels": dict(Counter(row["label"] for row in selected)),
            "conditions": dict(Counter(row["condition"] for row in selected)),
        }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
