"""Export a verified private v3 package to a small public Hugging Face tree.

The full package and its review receipts remain in private storage. This
export copies every native model byte unchanged and keeps only model-card,
license, chart and concise evaluation material at the public repository root.
Run the private package verifier before calling this command; this exporter
also rehashes the complete input inventory and its own output inventory.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import shutil
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from . import bundle_arena as common
from .generate_arena_v3 import FIGURES

VERSION = "decision2-hf-slim-export/1"
MODEL_ID = "llm-semantic-router/DEV2.0-4B"
OWL_SHA256 = "58dbd7cf5ff49b760a162c32366bd5312a2acfded1273b79b937afcd31ca6ee4"
OWL_SOURCE_REVISION = "cde2a68dbaa557ea65dc458104d410a0802ee259"
BANNER = "decision-2-4b-banner.svg"
CARD_FOX = (
    "![Decision 2.0 crossroads fox mosaic sticker]"
    "(decision-2-sticker-crossroads-fox-v5.png)"
)
CHARTS = tuple(FIGURES)
ROOT_SOURCES = {
    "LICENSE": "LICENSE",
    "NOTICE": "NOTICE",
    "LICENSE-Eikos": "native/LICENSE",
    "LICENSE-Qwen": "native/LICENSE-Qwen",
}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path.name}")
    return value


def _inventory(root: Path) -> dict[str, str]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Package must be a regular directory")
    result = {}
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("Linked package entries are forbidden")
        if path.is_file():
            result[path.relative_to(root).as_posix()] = _sha(path)
    return dict(sorted(result.items()))


def _checked_source(source: Path) -> tuple[dict[str, Any], dict[str, str]]:
    manifest_path = source / "PACKAGE_MANIFEST.json"
    manifest = _json(manifest_path)
    expected = manifest.get("files_sha256")
    if not isinstance(expected, dict) or not expected:
        raise ValueError("Private package lacks a complete byte inventory")
    actual = _inventory(source)
    actual.pop("PACKAGE_MANIFEST.json", None)
    if actual != expected:
        raise ValueError("Private package bytes differ from PACKAGE_MANIFEST.json")
    if manifest.get("model_id") != MODEL_ID:
        raise ValueError("Private package is not the 4B release model")
    record = _json(source / "release-record.json")
    if record.get("model_id") != MODEL_ID or record.get("license_id") != "apache-2.0":
        raise ValueError("Reviewed package does not declare Apache-2.0 for 4B")
    gate = _json(source / "release-gate.json")
    if gate.get("status") not in ("passed", "passed_postkey_user_directed"):
        raise ValueError("Private package release gate is not passed")
    if not (source / "native" / "SHA256SUMS").is_file():
        raise ValueError("Native model lacks its exact SHA256SUMS")
    for name, relative in ROOT_SOURCES.items():
        if relative not in actual:
            raise ValueError(f"Private package lacks inherited license/notice: {name}")
    return manifest, actual


def _banner(owl_png: bytes) -> str:
    """Compose new SVG typography around the unmodified Decision 1.0 owl."""
    encoded = base64.b64encode(owl_png).decode("ascii")
    # The old 2172 x 724 banner contains the owl entirely left of x=640.
    # Cover its former wordmark, leaving the original owl pixels untouched.
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="2172" height="724" viewBox="0 0 2172 724" role="img" aria-labelledby="title desc">
<title id="title">DEV2.0-4B Decision 2.0 mosaic owl</title>
<desc id="desc">The Decision 1.0 mosaic owl returns beside the DEV2.0-4B model name.</desc>
<rect width="2172" height="724" fill="#070a14"/>
<image width="2172" height="724" href="data:image/png;base64,{encoded}"/>
<rect x="640" y="0" width="1532" height="724" fill="#070a14"/>
<rect x="1465" y="61" width="622" height="105" rx="50" fill="#fffaf0" stroke="#0c2054" stroke-width="12"/>
<text x="1776" y="133" text-anchor="middle" fill="#0c2054" font-family="DejaVu Sans, Arial, sans-serif" font-size="57" font-weight="900" letter-spacing="2">DECISION 2.0</text>
<text x="704" y="447" fill="#0b1b47" stroke="#0b1b47" stroke-width="19" paint-order="stroke fill" font-family="DejaVu Sans, Arial, sans-serif" font-size="312" font-weight="900" letter-spacing="-12">DEV2.0</text>
<text x="704" y="425" fill="#fffaf0" font-family="DejaVu Sans, Arial, sans-serif" font-size="312" font-weight="900" letter-spacing="-12">DEV2.0</text>
<text x="1684" y="622" fill="#0b1b47" stroke="#0b1b47" stroke-width="23" paint-order="stroke fill" font-family="DejaVu Sans, Arial, sans-serif" font-size="210" font-weight="900">4B</text>
<text x="1684" y="602" fill="#baa5f4" font-family="DejaVu Sans, Arial, sans-serif" font-size="210" font-weight="900">4B</text>
</svg>\n"""
    ET.fromstring(svg)
    return svg


def _card(source_card: str) -> str:
    if source_card.count(CARD_FOX) != 1:
        raise ValueError("Source card does not have the expected old sticker")
    if not re.search(r"(?m)^license: apache-2\.0$", source_card):
        raise ValueError("Source card license metadata is not Apache-2.0")
    card = source_card.replace(
        CARD_FOX,
        f"![DEV2.0-4B mosaic owl](assets/{BANNER})",
        1,
    )
    for name in CHARTS:
        old = f"]({name})"
        if card.count(old) != 1:
            raise ValueError(f"Source card has no unique chart reference: {name}")
        card = card.replace(old, f"](assets/{name})", 1)
    old_manifest_sentence = "bound in `card-artifacts/manifest.json`."
    if card.count(old_manifest_sentence) != 1:
        raise ValueError("Source card lacks its exact generated-figure provenance")
    card = card.replace(
        old_manifest_sentence,
        "bound by the [public evaluation manifest](evaluation/manifest.json).",
        1,
    )
    old_package_sentence = (
        "`PACKAGE_MANIFEST.json` binds this card, native model, provenance, package\n"
        "parity and all three scored native runs. The packager checks reviewed release\n"
        "receipts and exact bytes; it does not rerun GPU inference or independently\n"
        "authenticate the declared reviewer and timestamp log."
    )
    if card.count(old_package_sentence) != 1:
        raise ValueError("Source card lacks its exact private verification disclosure")
    card = card.replace(
        old_package_sentence,
        "The [evaluation summary](evaluation/EVALUATION.md) and "
        "[public manifest](evaluation/manifest.json) identify the frozen panels, "
        "model bytes and generated charts. Full release evidence remains in the "
        "private verification package; the public repository contains no raw "
        "evaluation predictions or temporary review receipts.",
        1,
    )
    source_table = re.compile(
        r"\| Source \| Terms \| Attribution \| Use \| Redistribution \|\n"
        r"\| --- \| --- \| --- \| --- \| --- \|\n"
        r"(?:\|[^\n]*\|\n)+",
    )
    card, replaced = source_table.subn(
        "Training-source provenance and redistribution records are retained "
        "in the private research dataset and release evidence. No raw source "
        "records are included here.\n",
        card,
        count=1,
    )
    if replaced > 1:
        raise ValueError("Source card has ambiguous training-provenance tables")
    card = card.replace(
        "## Training, rights and limitations", "## Training and limitations"
    )
    if "CC BY-SA" in card:
        raise ValueError("Public card still contains private training-source detail")
    card += (
        "\n## Download and native inference\n\n"
        f"`hf download {MODEL_ID} --local-dir DEV2.0-4B` downloads the complete "
        "model. The byte-identical native model files are in `model/`; point the "
        "Decision 2.0 Eikos-compatible native loader at that directory. The "
        "model uses a Decision-specific readout, so a generic text-generation "
        "answer is not its evaluated decision output.\n\n"
        "[Apache-2.0 license](LICENSE) · [Upstream notices](ATTRIBUTIONS.md).\n"
    )
    common._public_text(card, "README.md")
    return card


def _attributions(record: dict[str, Any]) -> str:
    base = record.get("base_model", {})
    if (
        base.get("id") != "caiovicentino1/Eikos-4B"
        or base.get("revision") != "582ffb13f19a4da3f455e3db198584190bd7755b"
    ):
        raise ValueError("Unrecognized inherited model attribution")
    lines = [
        "# Model and artwork attributions",
        "",
        "This model adapts [Eikos-4B](https://huggingface.co/caiovicentino1/Eikos-4B)",
        "at revision `582ffb13f19a4da3f455e3db198584190bd7755b`.",
        "Its underlying Qwen model and tokenizer retain their original",
        "Apache-2.0 terms. The Eikos MIT license and native NOTICE are preserved",
        "byte-for-byte in `model/`; these inherited terms remain alongside our Apache-2.0",
        "license. No raw training records are included in this model repository.",
        "",
        "Inherited license texts:",
        "[Eikos MIT](LICENSE-Eikos), [Qwen Apache-2.0](LICENSE-Qwen),",
        "and [Eikos native notice](model/NOTICE). Our modification notice is",
        "in the root [NOTICE](NOTICE).",
        "",
        "The owl artwork is reused from the Decision 1.0 Nox-4B banner",
        f"at revision `{OWL_SOURCE_REVISION}` (source PNG SHA-256",
        f"`{OWL_SHA256}`); the model-name layout is new.",
        "",
    ]
    value = "\n".join(lines)
    common._public_text(value, "ATTRIBUTIONS.md")
    return value


def _evaluation(source_manifest: dict[str, Any]) -> str:
    panel = source_manifest.get("panel_sha256", {})
    if not isinstance(panel, dict):
        raise ValueError("Package has no panel digests")
    lines = [
        "# Evaluation protocol",
        "",
        "JevArena v3 first release uses 1,600 typed FINAL items and 6,547",
        "human-labeled transfer items across 15 tasks. Its score is",
        "`100 × sqrt(T × H)`, where T is the four-family macro accuracy of",
        "typed FINAL and H is the median of 15 task macro-F1 scores.",
        "JevBench's 231 public questions are scored separately. The displayed",
        "rank and Pareto frontier are relative to the same evaluated roster;",
        "they are not an upstream closed-set ranking.",
        "",
        "Source panel digests and all public repository file hashes are in",
        "[manifest.json](manifest.json). The full predictions, independent",
        "review receipts and original scorer outputs remain in the private",
        "verification bundle; the reproducible scorer and protocol live in",
        "[vLLM Semantic Router](https://github.com/vllm-project/semantic-router)",
        "and the [research log](https://gist.github.com/Xunzhuo/cd90fce0fa548616d8a4f1b2d2398dea).",
        "",
    ]
    value = "\n".join(lines)
    common._public_text(value, "EVALUATION.md")
    return value


def export(source: Path, owl: Path, output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(output)
    manifest, source_files = _checked_source(source)
    owl_bytes = owl.read_bytes()
    if owl.is_symlink() or hashlib.sha256(owl_bytes).hexdigest() != OWL_SHA256:
        raise ValueError("1.0 owl banner differs from the pinned owned asset")
    record = _json(source / "release-record.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.export-", dir=output.parent))
    try:
        for relative in source_files:
            if not relative.startswith("native/"):
                continue
            destination = stage / "model" / relative.removeprefix("native/")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / relative, destination)
            if _sha(destination) != source_files[relative]:
                raise ValueError(f"Native model changed during export: {relative}")
        for name, relative in ROOT_SOURCES.items():
            shutil.copyfile(source / relative, stage / name)
            if _sha(stage / name) != source_files[relative]:
                raise ValueError(f"License or notice changed during export: {name}")
        (stage / "assets").mkdir()
        (stage / "assets" / BANNER).write_text(_banner(owl_bytes), encoding="utf-8")
        for name in CHARTS:
            relative = f"card-artifacts/{name}"
            if relative not in source_files:
                raise ValueError(f"Missing frozen chart: {name}")
            shutil.copyfile(source / relative, stage / "assets" / name)
            if _sha(stage / "assets" / name) != source_files[relative]:
                raise ValueError(f"Chart changed during export: {name}")
        (stage / "README.md").write_text(
            _card((source / "README.md").read_text(encoding="utf-8")),
            encoding="utf-8",
        )
        (stage / "ATTRIBUTIONS.md").write_text(_attributions(record), encoding="utf-8")
        (stage / "evaluation").mkdir()
        (stage / "evaluation" / "EVALUATION.md").write_text(
            _evaluation(manifest), encoding="utf-8"
        )
        files = _inventory(stage)
        public = {
            "schema_version": VERSION,
            "model_id": MODEL_ID,
            "model_revision": manifest["model_revision"],
            "parameter_count": manifest["parameter_count"],
            "native_model_sha256": manifest["native_model_sha256"],
            "source_private_package_manifest_sha256": _sha(
                source / "PACKAGE_MANIFEST.json"
            ),
            "source_owl_banner_sha256": OWL_SHA256,
            "source_owl_model_revision": OWL_SOURCE_REVISION,
            "panel_sha256": manifest["panel_sha256"],
            "files_sha256": files,
        }
        (stage / "evaluation" / "manifest.json").write_text(
            json.dumps(public, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        verify(stage)
        stage.rename(output)
        return public
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def verify(root: Path) -> dict[str, Any]:
    manifest = _json(root / "evaluation" / "manifest.json")
    if (
        manifest.get("schema_version") != VERSION
        or manifest.get("model_id") != MODEL_ID
    ):
        raise ValueError("Unknown HF export identity")
    observed = _inventory(root)
    observed.pop("evaluation/manifest.json", None)
    if observed != manifest.get("files_sha256"):
        raise ValueError("Public HF export inventory changed")
    card = (root / "README.md").read_text(encoding="utf-8")
    if (
        CARD_FOX in card
        or f"assets/{BANNER}" not in card
        or not re.search(r"(?m)^license: apache-2\.0$", card)
        or any(f"](assets/{name})" not in card for name in CHARTS)
        or "PACKAGE_MANIFEST.json" in card
    ):
        raise ValueError("Public card differs from the slim release layout")
    if "Apache License" not in (root / "LICENSE").read_text(encoding="utf-8"):
        raise ValueError("Public root license is not an Apache license text")
    if set(root.iterdir()) != {
        root / "README.md",
        root / "ATTRIBUTIONS.md",
        *(root / name for name in ROOT_SOURCES),
        root / "assets",
        root / "model",
        root / "evaluation",
    }:
        raise ValueError("Public root contains an unexpected temporary file")
    for relative in ("README.md", "ATTRIBUTIONS.md", "evaluation/EVALUATION.md"):
        document = root / relative
        for target in re.findall(
            r"\]\(([^)]+)\)", document.read_text(encoding="utf-8")
        ):
            if target.startswith(("https://", "http://", "#", "mailto:")):
                continue
            target_path = (document.parent / target.split("#", 1)[0]).resolve()
            if (
                not target_path.is_relative_to(root.resolve())
                or not target_path.is_file()
            ):
                raise ValueError(f"Broken local model-card reference: {relative}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-package", type=Path)
    parser.add_argument("--owl-banner", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify is not None:
        result = verify(args.verify)
    else:
        if any(
            value is None
            for value in (args.private_package, args.owl_banner, args.output)
        ):
            parser.error("Export requires --private-package, --owl-banner and --output")
        result = export(args.private_package, args.owl_banner, args.output)
    print(json.dumps({"model_id": result["model_id"], "version": VERSION}))


if __name__ == "__main__":
    main()
