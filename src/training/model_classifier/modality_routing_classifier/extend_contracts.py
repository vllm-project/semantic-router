"""Add explicitly supplied output-contract variants to training captions."""

import argparse
import hashlib
import json
from pathlib import Path
from string import Formatter

from ..sequence_repair.corpus import write_corpus
from ..sequence_repair.data import normalized_text

LABELS = {"AR", "DIFFUSION", "BOTH"}


def validate_templates(templates):
    if not isinstance(templates, list) or not templates:
        raise ValueError("Supply a nonempty list of authored templates")
    for template in templates:
        if not isinstance(template, str) or not template.strip():
            raise ValueError("Templates must be nonempty strings")
        fields = [
            (field, spec, conversion)
            for _, field, spec, conversion in Formatter().parse(template)
            if field is not None
        ]
        if fields != [("", "", None)]:
            raise ValueError("Each template must contain exactly one {} caption field")


def validate_registry(registry):
    if not isinstance(registry, dict) or registry.get("version") != 1:
        raise ValueError("Expected output-contract registry version 1")
    source = registry.get("source_templates")
    contracts = registry.get("contracts")
    if not isinstance(source, dict) or not source or not isinstance(contracts, dict):
        raise ValueError("Supply source_templates and contracts by language")
    if set(source) != set(contracts):
        raise ValueError("Source templates and contracts must cover the same languages")
    for language, templates in source.items():
        validate_templates(templates)
        labels = contracts[language]
        if not isinstance(labels, dict) or set(labels) != LABELS:
            raise ValueError("Each language must declare AR, DIFFUSION, and BOTH")
        for values in labels.values():
            validate_templates(values)


def recover_caption(row, source_templates):
    family = int(row["template_family"].rsplit("-", 1)[1])
    templates = source_templates.get(row["language"], [])
    if not 0 <= family < len(templates):
        raise ValueError("Source template family is absent from the supplied registry")
    template = templates[family]
    validate_templates([template])
    parts, side = [[], []], 0
    for literal, field, _, _ in Formatter().parse(template):
        parts[side].append(literal)
        if field is not None:
            side = 1
    prefix, suffix = ("".join(part) for part in parts)
    text = row["text"]
    if not text.startswith(prefix) or not text.endswith(suffix):
        raise ValueError("Source caption wrapper differs from the supplied registry")
    caption = text[len(prefix) : len(text) - len(suffix)]
    if (
        hashlib.sha256(normalized_text(caption).encode()).hexdigest()
        != row["caption_sha256"]
    ):
        raise ValueError("Recovered caption differs from its frozen source hash")
    return caption


def build_additions(rows, registry):
    validate_registry(registry)
    additions = []
    for row in rows:
        if (
            row["source"] != "authored-output-contract-with-gallery-caption"
            or row["label"] != "AR"
        ):
            continue
        caption = recover_caption(row, registry["source_templates"])
        for label, templates in registry["contracts"][row["language"]].items():
            for index, template in enumerate(templates):
                additions.append(
                    {
                        **row,
                        "id": f'contract:{row["id"]}:{label}:{index}',
                        "text": template.format(caption),
                        "label": label,
                        "source": "authored-train-contract-diversity",
                        "template_family": f"modality-contract-{index}",
                    }
                )
    return additions


def extend_corpus(corpus, output, registry_path):
    registry_bytes = registry_path.read_bytes()
    registry = json.loads(registry_bytes)
    splits = {}
    for split in ("train", "dev", "test"):
        with (corpus / f"{split}.jsonl").open(encoding="utf-8") as stream:
            splits[split] = [json.loads(line) for line in stream if line.strip()]
    additions = build_additions(splits["train"], registry)
    splits["train"].extend(additions)
    contract = json.loads((corpus / "contract.json").read_text())
    provenance = {
        "parent_manifest_sha256": hashlib.sha256(
            (corpus / "manifest.json").read_bytes()
        ).hexdigest(),
        "registry_sha256": hashlib.sha256(registry_bytes).hexdigest(),
        "training_additions": len(additions),
        "scope": "Supplied authored templates on existing training caption groups; development and test requests retain their content and labels",
        "annotation_scope": "Registry labels are caller-supplied judgments, not evidence of independent review or natural user requests",
        "test_used": False,
    }
    return write_corpus(output, splits, contract, provenance)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(extend_corpus(args.corpus, args.output, args.registry), indent=2))


if __name__ == "__main__":
    main()
