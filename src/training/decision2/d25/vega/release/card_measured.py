"""Model card of a Decision 3.0 release (d3 family) in the Decision 2.0 card design, measured numbers only.

    python -m d25.vega.release.card_measured --measured measured.json --board index.json --vision-board vision.json \
        --manifest MODEL_MANIFEST.json --model-name d3 --repo-id vllm-sr/d3 \
        --banner decision-3.0.png --example example-receipt.png --logo vllm-sr-logo.dark.png --fonts <Inter TTFs> \
        --out <dir>

writes ``<dir>/README.md``, ``<dir>/assets/{banner,index-pareto,index-areas,example-receipt}.png``, the assembled
input (``card-input.json``) and ``card-assets.json`` (input and output digests). ``measured.json`` (schema
``d25-measured/1``) holds the public text index and its areas from a complete official-kit run, per-benchmark vision
scores on our reconstruction of the public vision suite, latency on one RTX PRO 6000, and the internal evaluation
shown in d3's board rows under one label (the Highlights say the official Full scores are pending); peers come
from the live boards' data files (Space ``data/index.json`` and ``data/vision.json``).

README, in order: YAML metadata; the banner; the title; one product paragraph; the at-a-glance table; Highlights;
a code-only Quickstart (text, then text and an image) that the verifier executes (``smoke.py --card``);
Evaluation (text board, the two charts, vision board, per-benchmark vision table, footnote); License (the base
model line); Citation. ``card.lint_readme`` and the package trace lint (``build.PUBLIC_TRACES``) apply.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import shutil
import sys
from pathlib import Path
from typing import Any

from d25.vega.release.card import PROJECT_URL, AUTHOR, CITATION_YEAR, lint_readme, one
from d25.vega.release.card_input import (
    AREAS,
    PREVIOUS_ENGINE,
    board_point,
    decision_name,
)
from d25.vega.release.examples import EXAMPLE_IMAGE, QUICKSTART, QUICKSTART_IMAGE

SCHEMA = "d25-measured/1"
ASSETS_SCHEMA = "d25-card-assets/1"
BANNER = "assets/banner.png"
PARETO = "assets/index-pareto.png"
AREAS_CHART = "assets/index-areas.png"
GENERATION = "3.0"
FAMILY = {
    "d3": "27B",
    "d3-flash": "9B",
    "d3-mini": "4B",
    "d3-nano": "2B",
    "d3-lite": "0.8B",
    "d3-edge": "0.6B",
}
PENDING = "pending official evaluation"
TAGS = (
    "zero-shot-classification",
    "decision-model",
    "classification",
    "system-one",
    "multimodal",
    "vision",
    "safetensors",
)
PIP = (
    'pip install "transformers==5.17.0" torch torchvision pillow safetensors accelerate',
    "pip install flash-linear-attention  # optional: fast GPU kernels for the linear-attention layers",
)
REQUIRED_SECTIONS = (
    "## Highlights",
    "## Quickstart",
    "## Evaluation",
    "## License",
    "## Citation",
)
TEXT_PEERS = (
    "pplx-decider-v1.1-27b",
    "decision-27b-ckpt-nothink",
    "jev",
    "torchcast-decision-27b",
)
VISION_PEERS = 3
PPLX_VISION = "pplx-decider-v1.1-27b"
PREVIOUS_NAME = "Decision 2.0 (27B)"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def py_literal(value: Any, indent: int = 0) -> str:
    """A Python literal laid out like ``json.dumps(..., indent=4)`` (``None`` instead of ``null``)."""
    pad, inner = " " * indent, " " * (indent + 4)
    if isinstance(value, dict):
        if not value:
            return "{}"
        items = [
            f"{inner}{json.dumps(k)}: {py_literal(v, indent + 4)}"
            for k, v in value.items()
        ]
        return "{\n" + ",\n".join(items) + f"\n{pad}}}"
    if isinstance(value, list):
        if not value:
            return "[]"
        items = [f"{inner}{py_literal(v, indent + 4)}" for v in value]
        return "[\n" + ",\n".join(items) + f"\n{pad}]"
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "True" if value else "False"
    return json.dumps(value, ensure_ascii=False)


def assemble(
    measured: dict,
    board: dict,
    vision_board: dict,
    manifest: dict,
    model_name: str,
    repo_id: str,
    snapshot: str,
    vision_snapshot: str,
) -> dict[str, Any]:
    if measured.get("schema") != SCHEMA:
        raise ValueError(f"not a {SCHEMA} file")
    if model_name not in FAMILY:
        raise ValueError(f"not a Decision {GENERATION} model name: {model_name}")
    text = measured["text"]
    if (
        not text.get("complete")
        or text.get("unsupported") != 0
        or text["ok"] != text["requests"]
    ):
        raise ValueError(
            "the public text run must be complete, with every request answered"
        )
    models = [
        m
        for m in [*board["models"], *([board["jev"]] if board.get("jev") else [])]
        if m.get("v03") and m["v03"].get("v") is not None
    ]
    points = {
        m["engine"]: {**board_point(m), "name": m["v03"].get("name") or m["name"]}
        for m in models
    }
    previous = {**points[PREVIOUS_ENGINE], "name": PREVIOUS_NAME}
    peers = [points[e] for e in TEXT_PEERS]
    others = [p for e, p in points.items() if not e.startswith("decision-2.0-")]
    family = [
        {**points[e], "name": decision_name(e)}
        for e in points
        if e.startswith("decision-2.0-") and points[e]["parameters"]
    ]
    ranked_vision = sorted(vision_board["entrants"], key=lambda e: -e["full"])
    pplx = next(e for e in vision_board["entrants"] if e["engine"] == PPLX_VISION)
    benchmarks = []
    for row in measured["vision"]["benchmarks"]:
        board_row = pplx["bench"][row["name"]]
        benchmarks.append({**row, "pplx_board": board_row["pub"]})
    return {
        "schema": "d25-card-input-measured/1",
        "model_name": model_name,
        "repo_id": repo_id,
        "model_sha256": manifest["identity"]["model_sha256"],
        "base_model": manifest["base_model"]["repo_id"],
        "parameters": manifest["parameters"],
        "size": FAMILY[model_name],
        "text": {
            "edition": text["edition"],
            "board_edition": text["board_edition"],
            "snapshot": snapshot,
            "board_generated_utc": board.get("generated_utc"),
            "public": text["public"],
            "requests": text["requests"],
            "areas": text["areas"],
            "previous": previous,
            "peers": peers,
            "family": family,
            "entrants": [
                {"parameters": p["parameters"], "full": p["full"]}
                for p in others
                if p["parameters"]
            ],
        },
        "vision": {
            "edition": vision_board["edition"],
            "snapshot": vision_snapshot,
            "board_generated_utc": vision_board.get("generated_utc"),
            "peers": [
                {
                    "name": e["name"],
                    "full": e["full"],
                    "public": e["pub"],
                    "private": e["priv"],
                }
                for e in ranked_vision[:VISION_PEERS]
            ],
            "benchmarks": benchmarks,
            "index": {**measured["vision"]["index"], "pplx_board": pplx["pub"]},
            "rows": measured["vision"]["rows"],
            "approximate": [b["name"] for b in benchmarks if b.get("approximate")],
        },
        "internal": measured["internal"],
        "latency": measured["latency"],
        "context": measured["context"],
        "images": measured["images"],
    }


def front_matter(card: dict) -> list[str]:
    return [
        "---",
        "pipeline_tag: zero-shot-classification",
        "license: apache-2.0",
        f"base_model: {card['base_model']}",
        "library_name: transformers",
        "tags:",
        *[f"- {t}" for t in TAGS],
        "---",
    ]


def pitch(card: dict) -> str:
    return (
        f"**{card['model_name']}** is the {card['size']} multimodal foundation decision model of Decision "
        f"{GENERATION}, the decision models of "
        f"[vLLM Semantic Router]({PROJECT_URL}). Give it an input (text or JSON, optionally with images) and "
        "the questions you need answered: pick one of several options, "
        "say yes or no, or rate on a scale. It answers them all in one call and returns a probability for every "
        "answer, without generating text."
    )


def at_a_glance(card: dict) -> list[str]:
    p = card["parameters"]
    return [
        "| | |",
        "| --- | --- |",
        f"| **Parameters** | {p['loaded'] / 1e9:.2f}B, including the {p['vision'] / 1e9:.2f}B vision encoder |",
        "| **Inputs** | Text or JSON, plus images (several per request) |",
        "| **Decision types** | Choice · Yes / No · Score |",
        "| **License** | Apache-2.0 |",
    ]


def ms(value: float) -> str:
    return f"{value:,.1f} ms"


def highlights(card: dict) -> list[str]:
    text, vision, latency = card["text"], card["vision"], card["latency"]
    previous = text["previous"]
    ahead = sum(1 for a in AREAS if text["areas"][a] > previous["areas"][a])
    areas_ahead = (
        "all five areas"
        if ahead == len(AREAS)
        else f"{ahead} of the {len(AREAS)} areas"
    )
    t, one_image = latency["text"], latency["one_image"]
    return [
        f"**Jev Decision Index {text['edition']}, public suite: {text['public']:.2f}**, measured with the official "
        f"{text['edition']} kit on the released weights: all {text['requests']:,} public requests answered, none "
        f"unsupported. The official Full scores on the text and vision boards are {PENDING}.",
        f"**+{one(text['public'] - previous['public'])} on the public suite over Decision 2.0** "
        f"(its 27B model: {previous['public']:.2f} on the board), ahead in {areas_ahead}.",
        "**Reads images:** multiple images per request (PNG, JPEG or WebP), given as paths, URLs, PIL images or "
        "base64 data URLs; every question of the request sees all of them.",
        f"**Speed:** text requests take a median of {ms(t['median_ms'])} (mean {ms(t['mean_ms'])}, 80th percentile "
        f"{ms(t['p80_ms'])}); requests with an image a median of {ms(one_image['median_ms'])} (mean "
        f"{ms(one_image['mean_ms'])}, 80th percentile {ms(one_image['p80_ms'])}). One {latency['gpu']}, one "
        "request at a time.",
        "**Many questions, one call:** Choice, Yes / No and Score questions about the same input are answered "
        "together, each from its own forward pass over the input, with a probability for every option.",
    ]


def quickstart(repo: str) -> str:
    def call(example: dict, images: str | None) -> str:
        lines = [
            "result = model.system_one(",
            f"    state={json.dumps(example['state'], ensure_ascii=False)},",
        ]
        if images:
            lines.append(
                f"    images={images},  # local paths, http(s) URLs, PIL images or base64 data URLs"
            )
        lines += [
            f"    questions={py_literal(example['questions'], 4)},",
            ")",
            'print(json.dumps(result["answers"], indent=2))',
        ]
        return "\n".join(lines)

    return "\n".join(
        [
            "import json",
            "",
            "from huggingface_hub import hf_hub_download",
            "from transformers import AutoModel",
            "",
            f'model = AutoModel.from_pretrained("{repo}", trust_remote_code=True)',
            "",
            "# Text",
            call(QUICKSTART, None),
            "",
            "# Text and an image (or several)",
            f'receipt = hf_hub_download("{repo}", "{EXAMPLE_IMAGE}")',
            call(QUICKSTART_IMAGE, "[receipt]"),
            "",
            "# Or as a pipeline:",
            f'# transformers.pipeline("decision", model="{repo}", trust_remote_code=True)(state=..., questions=..., '
            "images=...)",
        ]
    )


def image_note(card: dict) -> str:
    images = card["images"]
    return (
        f"Images go before the text of the request, each read at up to {images['max_pixels'] / 1e6:.1f} "
        "megapixels; every question of the request sees all of them."
    )


def text_table(card: dict) -> list[str]:
    text, internal = card["text"], card["internal"]["text"]
    rows = [
        "| Model | Jev Decision Index ↑ | Public ↑ | Same-skill tests ↑ | New-domain tasks ↑ |",
        "| --- | ---: | ---: | ---: | ---: |",
        "| "
        + " | ".join(
            f"**{c}**"
            for c in (
                card["model_name"],
                one(internal["full"]),
                one(text["public"]),
                one(internal["same_skill"]),
                one(internal["new_domain"]),
            )
        )
        + " |",
    ]
    for peer in sorted([*text["peers"], text["previous"]], key=lambda p: -p["full"]):
        cells = [
            peer["name"],
            *(one(peer[k]) for k in ("full", "public", "same_skill", "new_domain")),
        ]
        rows.append("| " + " | ".join(cells) + " |")
    return rows


def vision_tables(card: dict) -> list[str]:
    vision, internal = card["vision"], card["internal"]["vision"]
    rows = [
        "| Model | Vision Index ↑ | Public ↑ | Private ↑ |",
        "| --- | ---: | ---: | ---: |",
        "| "
        + " | ".join(
            f"**{c}**"
            for c in (
                card["model_name"],
                one(internal["full"]),
                one(internal["public"]),
                one(internal["private"]),
            )
        )
        + " |",
    ]
    for peer in vision["peers"]:
        rows.append(
            "| "
            + " | ".join(
                [
                    peer["name"],
                    one(peer["full"]),
                    one(peer["public"]),
                    one(peer["private"]),
                ]
            )
            + " |"
        )
    rows += [
        "",
        f"{card['model_name']} on the public vision benchmarks († approximate rebuild):",
        "",
        f"| Benchmark | {card['model_name']} |",
        "| --- | ---: |",
    ]
    for b in vision["benchmarks"]:
        name = b["name"] + (" †" if b.get("approximate") else "")
        rows.append(f"| {name} | {one(b['ours'])} |")
    return rows


def footnote(card: dict) -> str:
    text, vision = card["text"], card["vision"]
    return f"{card['internal']['label']}. Others: live board data, text {text['snapshot']}, vision {vision['snapshot']}."


def chart_note(card: dict, measured: bool) -> str:
    """One short line per sentence (the 2.0 chart footnote layout); the first stays clear of the x-axis label."""
    own = (
        f"{card['model_name']}: public suite measured with the official {card['text']['edition']} kit."
        if measured
        else f"{card['internal']['label']}."
    )
    return f"{own} Others: public board snapshot, {card['text']['snapshot']}."


def citation(card: dict) -> list[str]:
    name, repo = card["model_name"], card["repo_id"]
    key = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return [
        "```bibtex",
        f"@misc{{{key}_{CITATION_YEAR},",
        f"  title        = {{{{{name}}}: A Multimodal Foundation Decision Model}},",
        f"  author       = {{{{{AUTHOR}}}}},",
        f"  year         = {{{CITATION_YEAR}}},",
        f"  howpublished = {{\\url{{https://huggingface.co/{repo}}}}}",
        "}",
        "```",
    ]


def render_readme(card: dict) -> str:
    name, repo = card["model_name"], card["repo_id"]
    previous = card["text"]["previous"]["name"]
    base = card["base_model"]
    return "\n".join(
        [
            *front_matter(card),
            "",
            f"![{name}: Decision {GENERATION}]({BANNER})",
            "",
            f"# {name}",
            "",
            pitch(card),
            "",
            *at_a_glance(card),
            "",
            "## Highlights",
            "",
            *[f"- {item}" for item in highlights(card)],
            "",
            "## Quickstart",
            "",
            "```bash",
            *PIP,
            "```",
            "",
            "```python",
            quickstart(repo),
            "```",
            "",
            image_note(card),
            "",
            "## Evaluation",
            "",
            f"### Text: Jev Decision Index {card['text']['board_edition']}",
            "",
            *text_table(card),
            "",
            f"![Jev Decision Index against model size]({PARETO})",
            "",
            f"![Jev Decision Index by area: {name} and {previous}]({AREAS_CHART})",
            "",
            f"### Images: Jev Decision Index vision board {card['vision']['edition']}",
            "",
            *vision_tables(card),
            "",
            f"<sub>{footnote(card)}</sub>",
            "",
            "## License",
            "",
            f"Apache-2.0 ([LICENSE](LICENSE)). Built on [{base}](https://huggingface.co/{base}) (Apache-2.0).",
            "",
            "## Citation",
            "",
            *citation(card),
            "",
        ]
    )


def check_rendered(readme: str, files: set[str]) -> list[str]:
    from d25.vega.release.build import PUBLIC_TRACES

    problems = lint_readme(readme)
    problems += [
        f"{PUBLIC_TRACES[1]}: {line.strip()[:80]}"
        for line in readme.splitlines()
        if PUBLIC_TRACES[0].search(line)
    ]
    problems += [
        f"image count cap: {line.strip()[:80]}"
        for line in readme.splitlines()
        if re.search(
            r"up to \d+ (images|per request)|at most \d+ images|max_images", line
        )
    ]
    if not readme.startswith("---\n") or "\n---\n" not in readme[4:]:
        problems.append("missing YAML front matter")
    body = readme.split("\n---\n", 1)[-1].lstrip("\n")
    if not body.startswith("![") or f"]({BANNER})" not in body.split("\n", 1)[0]:
        problems.append("the banner must open the card")
    for link in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", readme):
        if not link.startswith("http") and link not in files:
            problems.append(f"image does not resolve: {link}")
    for link in re.findall(r"(?<!!)\[[^\]]*\]\(([^)#]+)\)", readme):
        if not link.startswith("http") and link not in files:
            problems.append(f"link does not resolve: {link}")
    headings = re.findall(r"^#{1,3} .+$", readme, flags=re.M)
    if [h for h in headings if h in REQUIRED_SECTIONS] != list(REQUIRED_SECTIONS):
        problems.append("sections missing or out of order")
    if readme.count("```python") != 1:
        problems.append("the quickstart needs exactly one Python block")
    for chart in (PARETO, AREAS_CHART):
        if f"]({chart})" not in readme:
            problems.append(f"missing chart: {chart}")
    if f'hf_hub_download("' not in readme or EXAMPLE_IMAGE not in files:
        problems.append("the image example must resolve to a repository file")
    return problems


def chart_view(card: dict) -> dict[str, Any]:
    text = card["text"]
    return {
        "own": {
            "name": card["model_name"],
            "parameters": text["previous"]["parameters"],
            "full": card["internal"]["text"]["full"],
        },
        "previous": text["previous"],
        "family": text["family"],
        "entrants": text["entrants"],
        "generation": f"Decision {GENERATION}",
        "previous_generation": "Decision 2.0",
        "own_mark": "",
        "footnote": chart_note(card, measured=False),
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    from d25.vega.release.card_assets import Renderer

    read = lambda p: json.loads(Path(p).read_text(encoding="utf-8"))  # noqa: E731
    card = assemble(
        read(args.measured),
        read(args.board),
        read(args.vision_board),
        read(args.manifest),
        args.model_name,
        args.repo_id,
        args.snapshot,
        args.vision_snapshot,
    )
    out = args.out
    (out / "assets").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.banner, out / BANNER)
    shutil.copyfile(args.example, out / EXAMPLE_IMAGE)
    renderer = Renderer(args.logo, args.fonts)
    view = chart_view(card)
    renderer.pareto25(view, out / PARETO)
    renderer.areas25(
        {
            **view,
            "footnote": chart_note(card, measured=True),
            "own": {
                "name": card["model_name"],
                "public": card["text"]["public"],
                "areas": card["text"]["areas"],
            },
        },
        out / AREAS_CHART,
    )
    readme = render_readme(card)
    problems = check_rendered(
        readme, {BANNER, PARETO, AREAS_CHART, EXAMPLE_IMAGE, "LICENSE"}
    )
    if problems:
        raise ValueError(f"README failed the card checks: {problems}")
    (out / "README.md").write_text(readme, encoding="utf-8")
    (out / "card-input.json").write_text(json.dumps(card, indent=1) + "\n")
    import matplotlib
    import PIL

    receipt = {
        "schema": ASSETS_SCHEMA,
        "model_name": card["model_name"],
        "model_sha256": card["model_sha256"],
        "inputs": {
            name: sha256_file(getattr(args, name))
            for name in (
                "measured",
                "board",
                "vision_board",
                "manifest",
                "banner",
                "example",
                "logo",
            )
        },
        "software": {
            "python": platform.python_version(),
            "matplotlib": matplotlib.__version__,
            "pillow": PIL.__version__,
        },
        "files": {
            name: sha256_file(out / name)
            for name in (
                "README.md",
                BANNER,
                PARETO,
                AREAS_CHART,
                EXAMPLE_IMAGE,
                "card-input.json",
            )
        },
        "lint": [],
    }
    (out / "card-assets.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    return receipt


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    for name in (
        "measured",
        "board",
        "vision-board",
        "manifest",
        "banner",
        "example",
        "logo",
        "fonts",
        "out",
    ):
        ap.add_argument(f"--{name}", required=True, type=Path)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--repo-id", required=True)
    ap.add_argument(
        "--snapshot", required=True, help="date of the text board data, e.g. 2026-10-10"
    )
    ap.add_argument(
        "--vision-snapshot", required=True, help="date of the vision board data"
    )
    args = ap.parse_args(argv)
    receipt = build(args)
    print(json.dumps({"files": receipt["files"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
