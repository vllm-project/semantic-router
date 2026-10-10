"""Product model card of a Decision 2.5 release, in the Decision 2.0 card design.

    python -m d25.vega.release.card --input card-input.json --out <dir> --logo <vllm-sr-logo.dark.png> \
        --fonts <dir with the Inter TTFs> [--notice "SAMPLE: placeholder numbers"]

writes ``<dir>/README.md``, ``<dir>/assets/{banner,index-pareto,index-areas}.png`` and ``<dir>/card-assets.json``
(input and output digests only). ``card-input.json`` (schema ``d25-card-input/1``, built by ``card_input.py``)
holds the Index values; they appear only on the published card, never in git.

README, in order: YAML metadata; the banner (eyebrow "DECISION 2.5"); the title; one product paragraph; the
at-a-glance table; four Highlights (Index standing, gain over the previous generation, latency on one RTX PRO
6000, many questions in one call); a code-only Quickstart that the verifier executes (``smoke.py --card``);
Evaluation (Full, public, same-skill and new-domain scores of this model and the board's licence-eligible
leaders and the previous generation; the Index against model size and by area; the footnote, which marks a
local estimate as such until the board scores the model); License with the one-line teacher disclosure;
Citation. ``lint_readme`` refuses internal vocabulary, machine details, revision hashes and precision or
training-recipe details (the 2.0 rules plus the 2.5 campaign's terms).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

from v2.release.card import FORBIDDEN, README_FORBIDDEN

from d25.vega.release.examples import QUICKSTART

SCHEMA = "d25-card-input/1"
ASSETS_SCHEMA = "d25-card-assets/1"
BANNER = "assets/banner.png"
PARETO = "assets/index-pareto.png"
AREAS_CHART = "assets/index-areas.png"
PROJECT_URL = "https://github.com/vllm-project/semantic-router"
AUTHOR = "vLLM Semantic Router Team"
CITATION_YEAR = 2026
TAGS = (
    "zero-shot-classification",
    "decision-model",
    "classification",
    "system-one",
    "safetensors",
)
MODEL_NAME = re.compile(
    r"Decision-(?P<generation>\d\.\d)-(?P<codename>[A-Z][a-z]+)-(?P<size>[0-9.]+B)\Z"
)
REQUIRED_SECTIONS = (
    "## Highlights",
    "## Quickstart",
    "## Evaluation",
    "## License",
    "## Citation",
)
PIP = (
    'pip install "transformers==5.17.0" torch safetensors accelerate',
    "pip install flash-linear-attention  # optional: fast GPU kernels for the linear-attention layers",
)
FOOTNOTE_BOARD = (
    "Jev Decision Index {edition}: the maintainers' runs, public board snapshot {snapshot}. Training data "
    "audited at row level against all Index test items."
)
CHART_NOTE_ESTIMATE = (
    "{name}: local estimate, not a board score: public part measured with the official {edition} kit, Index "
    "estimated against Perplexity Decider v1.1 on the same measurements (paired SE {uncertainty:.1f}). Others: "
    "public board snapshot, {snapshot}. Training data audited at row level against all Index test items."
)
FOOTNOTE_ESTIMATE = (
    "{name} (*): local estimate, not a board score. Public part measured with the official {edition} kit on the "
    "released weights; same-skill and new-domain parts estimated from held-out proxy tests calibrated on open "
    "board models; the Index is Perplexity Decider v1.1's board score plus the paired difference between the two "
    "models on the same measurements (paired standard error {uncertainty:.1f} Full points). Others: public board "
    "snapshot, {snapshot}. Training data audited at row level against all Index test items."
)
README_FORBIDDEN_25 = (
    (
        re.compile(r"\bprox(?:y|ies)\b|\banchors?\b|\bstand-?in\b", re.I),
        "internal evaluation vocabulary",
    ),
    (
        re.compile(r"\bws-[a-z]+|\bwave-?\d|\bcampaign\b|\bgate\b", re.I),
        "internal process vocabulary",
    ),
    (
        re.compile(r"\bM[1-9]T?(?:-[a-z0-9.]+)?\b|\bSYN\d\b|\bA[1-9]\b"),
        "internal data or arm names",
    ),
    (
        re.compile(r"MI3\d\dX|\bROCm\b|S_hat|O_hat|\beq[SO]\b|Full_hat"),
        "internal hardware or estimator names",
    ),
)
CARD_LICENCES = {
    "apache-2.0",
    "mit",
    "bsd-2-clause",
    "bsd-3-clause",
    "cc-by-4.0",
    "cc-by-sa-4.0",
    "openrail",
    "llama3",
    "gemma",
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def lint_readme(text: str) -> list[str]:
    return sorted(
        {
            reason
            for pattern, reason in (*FORBIDDEN, *README_FORBIDDEN, *README_FORBIDDEN_25)
            if pattern.search(text)
        }
    )


def load_input(path: Path) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema") != SCHEMA:
        raise ValueError(f"{path}: not a {SCHEMA} file")
    if not MODEL_NAME.match(data["model_name"]):
        raise ValueError(f"not a Decision model name: {data['model_name']}")
    index = data["index"]
    if index["status"] not in ("estimate", "board", "pending"):
        raise ValueError("index.status must be estimate, board or pending")
    for key in ("full", "public", "same_skill", "new_domain"):
        value = index["own"].get(key)
        if index["status"] == "pending" and value is not None:
            raise ValueError("a pending card carries no values for this model")
        optional = index["status"] == "estimate" and key in ("same_skill", "new_domain")
        if (
            index["status"] != "pending"
            and not optional
            and not isinstance(value, (int, float))
        ):
            raise ValueError(f"index.own.{key} must be a number")
    for peer in index["peers"]:
        if (
            peer.get("selected") != "lead"
            and (peer.get("licence") or "").lower() not in CARD_LICENCES
        ):
            raise ValueError(
                f"peer {peer['name']}: licence {peer.get('licence')!r} is not card-eligible"
            )
    if not data.get("teachers"):
        raise ValueError(
            "teachers must list every teacher model (the card discloses them)"
        )
    return data


def one(value: float) -> str:
    """One decimal, rounding halves up as the board's two-decimal values are read (62.25 -> 62.3)."""
    return str(Decimal(str(value)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def parts(name: str) -> dict[str, str]:
    return MODEL_NAME.match(name).groupdict()


def front_matter(card: dict[str, Any]) -> list[str]:
    return [
        "---",
        "pipeline_tag: zero-shot-classification",
        "license: apache-2.0",
        f"base_model: {card['base_model']}",
        "base_model_relation: finetune",
        "library_name: transformers",
        "tags:",
        *[f"- {t}" for t in TAGS],
        "---",
    ]


def pitch(card: dict[str, Any]) -> str:
    p = parts(card["model_name"])
    family = f"Decision {p['generation']}"
    if card.get("collection_url"):
        family = f"[{family}]({card['collection_url']})"
    return (
        f"**{card['model_name']}** is the {p['size']} model of {family}, the decision models of "
        f"[vLLM Semantic Router]({PROJECT_URL}). Give it an input (text or JSON) and the questions you need "
        "answered: pick one of several options, say yes or no, or rate on a scale. It answers them all in one "
        "call and returns a probability for every answer, without generating text."
    )


def at_a_glance(card: dict[str, Any]) -> list[str]:
    return [
        "| | |",
        "| --- | --- |",
        f"| **Parameters** | {card['parameters_loaded'] / 1e9:.2f}B |",
        f"| **Context length** | {card['max_input_tokens']:,} tokens per question |",
        "| **Decision types** | Choice · Yes / No · Score |",
        "| **License** | Apache-2.0 |",
    ]


def highlights(card: dict[str, Any]) -> list[str]:
    index = card["index"]
    if index["status"] == "pending":
        return [
            f"**Open weights:** fine-tuned from {card['base_model']}, Apache-2.0.",
            "**Many questions, one call:** Choice, Yes / No and Score questions about the same input are "
            "answered together, each from its own forward pass over the input, with a probability for every "
            "option.",
        ]
    own, previous, peers = index["own"], index["previous"], index["peers"]
    board_top = index["board_top"]
    lines = []
    if index["status"] == "board":
        rank = index["own"]["rank"]
        if rank == 1:
            runner = index["runner_up"]
            lines.append(
                f"**#1 on the Jev Decision Index:** {one(own['full'])} Full score on the {index['edition']} "
                f"board, ahead of {runner['name']} ({one(runner['full'])})."
            )
        else:
            lines.append(
                f"**#{rank} on the Jev Decision Index:** {one(own['full'])} Full score on the "
                f"{index['edition']} board."
            )
    elif own.get("paired_with"):
        paired = own["paired_with"]
        lines.append(
            f"**Jev Decision Index {index['edition']}, local estimate:** {one(own['full'])} Full score, paired "
            f"with {paired['name']} (board {paired['full']:.2f}); not an official score. Public part measured "
            f"with the official kit: {one(own['public'])}."
        )
    else:
        lines.append(
            f"**Jev Decision Index, local estimate:** {one(own['full'])} Full score (board snapshot "
            f"{index['snapshot']}: #1 at {one(board_top['full'])})."
        )
    gain = own["full"] - previous["full"]
    if gain > 0.05:
        domain = (
            None
            if own["new_domain"] is None
            else own["new_domain"] - previous["new_domain"]
        )
        extra = (
            f", +{one(domain)} on new-domain tasks"
            if domain is not None and domain > 0.05
            else ""
        )
        if index["status"] == "estimate":
            extra += "; local estimate"
        lines.append(
            f"**The strongest Decision model:** +{one(gain)} Full score over {previous['name']} "
            f"({one(previous['full'])}){extra}."
        )
    speed = card.get("speed")
    if speed:
        lines.append(
            f"**Speed:** a median of {speed['median_ms']:.1f} ms per request (mean {speed['mean_ms']:.1f} ms, "
            f"80th percentile {speed['p80_ms']:.1f} ms) on one {speed['gpu']}, one request at a time."
        )
    lines.append(
        "**Many questions, one call:** Choice, Yes / No and Score questions about the same input are "
        "answered together, each from its own forward pass over the input, with a probability for every "
        "option."
    )
    return lines


def quickstart(repo: str) -> str:
    state = json.dumps(QUICKSTART["state"], ensure_ascii=False)
    questions = json.dumps(
        QUICKSTART["questions"], ensure_ascii=False, indent=4
    ).replace("\n", "\n    ")
    return (
        "import json\n\nfrom transformers import AutoModel\n\n"
        f'model = AutoModel.from_pretrained("{repo}", trust_remote_code=True)\n'
        "result = model.system_one(\n"
        f"    state={state},\n"
        f"    questions={questions},\n"
        ")\n"
        'print(json.dumps(result["answers"], indent=2))\n\n'
        "# Or as a pipeline:\n"
        f'# transformers.pipeline("decision", model="{repo}", trust_remote_code=True)(state=..., questions=...)'
    )


def results_table(card: dict[str, Any]) -> list[str]:
    index = card["index"]
    star = "*" if index["status"] == "estimate" else ""
    rows = [
        "| Model | Jev Decision Index ↑ | Public ↑ | Same-skill tests ↑ | New-domain tasks ↑ |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    own = index["own"]
    cells = (
        [card["model_name"]] + ["pending"] * 4
        if index["status"] == "pending"
        else [
            card["model_name"],
            one(own["full"]) + star,
            one(own["public"]),
            "—" if own["same_skill"] is None else one(own["same_skill"]) + star,
            "—" if own["new_domain"] is None else one(own["new_domain"]) + star,
        ]
    )
    rows.append("| " + " | ".join(f"**{c}**" for c in cells) + " |")
    shown = sorted([*index["peers"], index["previous"]], key=lambda p: -p["full"])
    for peer in shown:
        rows.append(
            "| "
            + " | ".join(
                [
                    peer["name"],
                    one(peer["full"]),
                    one(peer["public"]),
                    one(peer["same_skill"]),
                    one(peer["new_domain"]),
                ]
            )
            + " |"
        )
    return rows


FOOTNOTE_PENDING = (
    "{name}: evaluation of this checkpoint is pending; this card shows no values for it. Others: public board "
    "snapshot, {snapshot}. Training data audited at row level against all Index test items."
)


FOOTNOTE_PAIRED = (
    "{name} (*): local estimate, not an official score. Public part measured with the official {edition} kit on "
    "the released weights. Full estimated against {pplx} (board {pplx_full:.2f}) from the paired difference "
    "between the two models on the same measurements (public suite and held-out tests calibrated on open board "
    "models); {full:.1f} is the "
    "conservative value: the public gain on benchmarks whose training splits were used in training is not carried "
    "over to the private same-skill part ({unadjusted:.1f} without this adjustment). Same-skill and new-domain parts "
    "are not estimated separately. Others: public board snapshot, {snapshot}. Training data audited at row level "
    "against all Index test items."
)
CHART_NOTE_PAIRED = (
    "{name}: local estimate, not an official score (paired with {pplx}, conservative). Others: public board "
    "snapshot, {snapshot}. Training data audited at row level against all Index test items."
)


def paired_note(card: dict[str, Any], template: str) -> str:
    index = card["index"]
    own = index["own"]
    return template.format(
        name=card["model_name"],
        edition=index["edition"],
        pplx=own["paired_with"]["name"],
        pplx_full=own["paired_with"]["full"],
        full=own["full"],
        unadjusted=own["full_unadjusted"],
        snapshot=index["snapshot"],
    )


def footnote(card: dict[str, Any]) -> str:
    index = card["index"]
    if index["status"] == "estimate" and index["own"].get("paired_with"):
        return paired_note(card, FOOTNOTE_PAIRED)
    if index["status"] == "pending":
        return FOOTNOTE_PENDING.format(
            name=card["model_name"], snapshot=index["snapshot"]
        )
    if index["status"] == "board":
        return FOOTNOTE_BOARD.format(
            edition=index["edition"], snapshot=index["snapshot"]
        )
    return FOOTNOTE_ESTIMATE.format(
        name=card["model_name"],
        edition=index["edition"],
        snapshot=index["snapshot"],
        uncertainty=index["own"]["uncertainty"],
    )


def chart_note(card: dict[str, Any]) -> str:
    """The footnote on the charts: one short line per sentence (the 2.0 chart footnote layout)."""
    index = card["index"]
    if index["status"] == "estimate" and index["own"].get("paired_with"):
        return paired_note(card, CHART_NOTE_PAIRED)
    if index["status"] == "pending":
        return FOOTNOTE_PENDING.format(
            name=card["model_name"], snapshot=index["snapshot"]
        )
    if index["status"] == "board":
        return FOOTNOTE_BOARD.format(
            edition=index["edition"], snapshot=index["snapshot"]
        )
    return CHART_NOTE_ESTIMATE.format(
        name=card["model_name"],
        edition=index["edition"],
        snapshot=index["snapshot"],
        uncertainty=index["own"]["uncertainty"],
    )


def teacher_line(card: dict[str, Any]) -> str:
    items = [
        f"[{t['model']}](https://huggingface.co/{t['model']}) ({t['licence']}; {t['use']})"
        for t in card["teachers"]
    ]
    return "Teacher models: " + "; ".join(items) + "."


def citation(card: dict[str, Any]) -> list[str]:
    name, repo, p = card["model_name"], card["repo_id"], parts(card["model_name"])
    key = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return [
        "```bibtex",
        f"@misc{{{key}_{CITATION_YEAR},",
        f"  title        = {{{{{name}}}: A Decision {p['generation']} Model for Structured Decisions}},",
        f"  author       = {{{{{AUTHOR}}}}},",
        f"  year         = {{{CITATION_YEAR}}},",
        f"  howpublished = {{\\url{{https://huggingface.co/{repo}}}}}",
        "}",
        "```",
    ]


def render_readme(card: dict[str, Any], notice: str | None = None) -> str:
    name, repo = card["model_name"], card["repo_id"]
    previous = card["index"]["previous"]["name"]
    lines = [*front_matter(card), "", f"![{name}]({BANNER})", "", f"# {name}", ""]
    if notice:
        lines += [f"> **{notice}**", ""]
    lines += [
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
        "## Evaluation",
        "",
        *results_table(card),
        "",
        "### Jev Decision Index",
        "",
        f"![Jev Decision Index against model size]({PARETO})",
        "",
        *(
            []
            if card["index"]["status"] == "pending"
            else [
                f"![Jev Decision Index by area: {name} and {previous}]({AREAS_CHART})",
                "",
            ]
        ),
        f"<sub>{footnote(card)}</sub>",
        "",
        "## License",
        "",
        f"Apache-2.0 ([LICENSE](LICENSE)). Fine-tuned from [{card['base_model']}]"
        f"(https://huggingface.co/{card['base_model']}) (Apache-2.0). {teacher_line(card)}",
        "",
        "## Citation",
        "",
        *citation(card),
        "",
    ]
    return "\n".join(lines)


def check_rendered(readme: str, files: set[str]) -> list[str]:
    """Structural card checks against the repository file list (the 2.0 checks, with the 2.5 charts)."""
    problems = lint_readme(readme)
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
    order = [h for h in headings if h in REQUIRED_SECTIONS]
    if order != list(REQUIRED_SECTIONS):
        problems.append(f"sections missing or out of order: {order}")
    if readme.count("```python") != 1:
        problems.append("the quickstart needs exactly one Python block")
    pending = "evaluation of this checkpoint is pending" in readme
    for chart in (PARETO,) if pending else (PARETO, AREAS_CHART):
        if f"]({chart})" not in readme:
            problems.append(f"missing chart: {chart}")
    return problems


def chart_view(card: dict[str, Any]) -> dict[str, Any]:
    index = card["index"]
    p = parts(card["model_name"])
    previous = index["previous"]
    return {
        "own": (
            None
            if index["status"] == "pending"
            else {
                "name": card["model_name"],
                "parameters": card["parameters_served"],
                **index["own"],
            }
        ),
        "previous": previous,
        "family": index["family"],
        "entrants": index["entrants"],
        "generation": f"Decision {p['generation']}",
        "previous_generation": f"Decision {parts(previous['name'])['generation']}",
        "own_mark": " (local estimate)" if index["status"] == "estimate" else "",
        "footnote": chart_note(card),
    }


def build(
    input_path: Path, out: Path, logo: Path, fonts: Path, notice: str | None = None
) -> dict[str, Any]:
    from d25.vega.release.card_assets import Renderer

    card = load_input(input_path)
    p = parts(card["model_name"])
    out.mkdir(parents=True, exist_ok=True)
    renderer = Renderer(logo, fonts)
    renderer.banner(
        card["model_name"],
        out / BANNER,
        codename=p["codename"],
        size=p["size"],
        eyebrow=f"DECISION {p['generation']}",
    )
    view = chart_view(card)
    renderer.pareto25(view, out / PARETO)
    if view["own"] is not None:
        renderer.areas25(view, out / AREAS_CHART)
    readme = render_readme(card, notice)
    problems = check_rendered(readme, {BANNER, PARETO, AREAS_CHART, "LICENSE"})
    if problems:
        raise ValueError(f"README failed the card checks: {problems}")
    (out / "README.md").write_text(readme, encoding="utf-8")
    import matplotlib
    import PIL

    from v2.release.card_assets import FONTS

    receipt = {
        "schema": ASSETS_SCHEMA,
        "model_name": card["model_name"],
        "model_sha256": card.get("model_sha256"),
        "inputs": {
            "card_input_sha256": sha256_file(input_path),
            "logo_sha256": sha256_file(logo),
            "fonts_sha256": {n: sha256_file(fonts / f"{n}.ttf") for n in FONTS},
        },
        "software": {
            "python": platform.python_version(),
            "matplotlib": matplotlib.__version__,
            "pillow": PIL.__version__,
        },
        "files": {
            name: sha256_file(out / name)
            for name in ("README.md", BANNER, PARETO, AREAS_CHART)
            if (out / name).exists()
        },
    }
    (out / "card-assets.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    return receipt


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--logo", required=True, type=Path)
    ap.add_argument("--fonts", required=True, type=Path)
    ap.add_argument(
        "--notice", help="a line shown under the title (staging or sample cards)"
    )
    args = ap.parse_args(argv)
    receipt = build(args.input, args.out, args.logo, args.fonts, args.notice)
    print(json.dumps({"files": receipt["files"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
