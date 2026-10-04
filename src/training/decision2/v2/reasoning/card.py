"""Product card of a Decision 2.0 Reasoning release: banner, four charts and README.md.

Same composition, palette, fonts and logo as the Decision 2.0 cards (``v2.release.card_assets``): white background,
the vLLM-SR logo bottom-right of every chart. The values come from one private input file (they appear only on the
card, never in a commit); a receipt records input and output digests.

    python -m v2.reasoning.card --inputs CARD.json --logo LOGO --fonts DIR --output DIR

Input file (``rsn-card/1``): model_name, repo_id, base_name, base_repo, collection_url, parameters_label,
context_tokens; ``index`` {own, base: {balanced_skill, areas}, delta_ci: [lo, hi], area_ci: {area: [lo, hi]},
rank_text}; ``pareto`` {family, decision1, entrants, own_parameters, footnote}; ``jevarena`` rows {role, label,
score, choice, noul, score_accuracy, transfer}; ``latency_ms`` {own, base}; ``index_footnote``; ``example``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

from v2.release import card_assets
from v2.release.card_assets import (
    BANNER_INCHES,
    BANNER_UNITS,
    DPI,
    GRADIENT,
    INK,
    MUTED,
    YELLOW,
)

SCHEMA = "rsn-card/1"
NAME = re.compile(
    r"Decision-2\.0-(?P<codename>[A-Z][a-z]+)-(?P<size>[0-9.]+B)-(?P<variant>[A-Za-z]+)\Z"
)
TAGLINE = "Multi-step reasoning in one forward pass"
FIGURES = (
    "banner.png",
    "jevarena.png",
    "jevarena-types.png",
    "index-pareto.png",
    "index-areas.png",
)


class Renderer(card_assets.Renderer):
    def save(self, fig, path: Path) -> None:
        """Category labels longer than the base layout allows move the axes right, level with the title."""
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        lefts = [
            label.get_window_extent(renderer).x0 / fig.bbox.width
            for ax in fig.axes
            if ax.axison
            for label in ax.get_yticklabels()
            if label.get_visible() and label.get_text()
        ]
        if lefts and min(lefts) < 0:
            title = min(text.get_position()[0] for text in fig.texts)
            fig.subplots_adjust(left=fig.subplotpars.left + title - min(lefts))
        super().save(fig, path)

    def banner_variant(self, name: str, path: Path) -> None:
        """The Decision 2.0 banner with the variant in the eyebrow and the Reasoning tagline."""
        import numpy as np
        from matplotlib.font_manager import FontProperties
        from matplotlib.patches import Circle, PathPatch
        from matplotlib.textpath import TextPath
        from matplotlib.transforms import Affine2D

        match = NAME.fullmatch(name)
        if not match:
            raise ValueError(f"not a Decision 2.0 variant name: {name}")
        plt = self.plt
        fig = plt.figure(figsize=BANNER_INCHES, dpi=DPI)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, BANNER_UNITS[0])
        ax.set_ylim(0, BANNER_UNITS[1])
        ax.axis("off")
        vmark = self.vmark()
        vh, vw = vmark.shape[:2]
        aspect = BANNER_INCHES[0] / BANNER_INCHES[1]
        height = 1.25
        width = height * (vw / vh) / aspect
        vax = fig.add_axes([0.985 - 0.92 * width, -0.14, width, height])
        vax.imshow(vmark, interpolation="lanczos")
        vax.axis("off")
        w, h = self.logo.size
        lax = fig.add_axes([0.052, 0.82, 0.12, 0.12 * (h / w) * aspect])
        lax.imshow(self.logo)
        lax.axis("off")
        x0, baseline = 5.6, 10.6
        ax.text(
            x0 + 0.5,
            27.6,
            f"DECISION 2.0  ·  {match['variant'].upper()}",
            fontsize=12.5,
            fontweight="bold",
            color=GRADIENT[0],
        )
        display = FontProperties(family="Inter Display", weight="bold")
        code = TextPath((0, 0), match["codename"], size=15.5, prop=display)
        cb = code.get_extents()
        patch = PathPatch(
            code,
            transform=Affine2D().translate(x0 - cb.x0, baseline) + ax.transData,
            facecolor="none",
            edgecolor="none",
        )
        ax.add_patch(patch)
        t = np.linspace(0, 1, 512)[None, :, None]
        ramp = (
            card_assets._rgb(GRADIENT[0]) * (1 - t) + card_assets._rgb(GRADIENT[1]) * t
        )
        image = ax.imshow(
            np.repeat(ramp, 8, axis=0),
            extent=(x0, x0 + cb.width, baseline + cb.y0, baseline + cb.y1),
            aspect="auto",
            interpolation="bicubic",
            zorder=3,
        )
        image.set_clip_path(patch)
        tier = TextPath((0, 0), match["size"], size=8.4, prop=display)
        sb = tier.get_extents()
        size_left = x0 + cb.width + 3.2
        ax.add_patch(
            PathPatch(
                Affine2D().translate(size_left - sb.x0, baseline).transform_path(tier),
                facecolor=INK,
                edgecolor="none",
                zorder=3,
            )
        )
        tagline = ax.text(x0 + 0.5, 3.4, TAGLINE, fontsize=13.5, color=MUTED)
        fig.canvas.draw()
        box = tagline.get_window_extent(fig.canvas.get_renderer())
        right = ax.transData.inverted().transform((box.x1, box.y0))[0]
        ax.add_patch(
            Circle(
                (right + 0.75, 3.85), 0.45, facecolor=YELLOW, edgecolor="none", zorder=3
            )
        )
        self.save(fig, path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def render_assets(
    inputs: dict[str, Any], logo: Path, fonts: Path, output: Path
) -> dict[str, str]:
    renderer = Renderer(logo, fonts)
    output.mkdir(parents=True, exist_ok=True)
    renderer.banner_variant(inputs["model_name"], output / "banner.png")
    renderer.jevarena(inputs["jevarena"], output / "jevarena.png")
    renderer.types(inputs["jevarena"], output / "jevarena-types.png")
    index = inputs["index"]
    own = {
        "name": inputs["model_name"],
        "balanced_skill": index["own"]["balanced_skill"],
        "areas": index["own"]["areas"],
        "parameters": inputs["pareto"]["own_parameters"],
    }
    base = {
        "name": inputs["base_name"],
        "balanced_skill": index["base"]["balanced_skill"],
        "areas": index["base"]["areas"],
        "parameters": inputs["pareto"]["own_parameters"],
    }
    renderer.areas(
        {
            "own": own,
            "compare": base,
            "delta": own["balanced_skill"] - base["balanced_skill"],
            "footnote": inputs["index_footnote"],
        },
        output / "index-areas.png",
    )
    pareto = inputs["pareto"]
    renderer.pareto(
        {
            "own": own,
            "compare": base,
            "compare_kind": "base",
            "family": pareto["family"],
            "decision1": pareto["decision1"],
            "entrants": pareto["entrants"],
            "footnote": pareto["footnote"],
        },
        output / "index-pareto.png",
    )
    return {name: sha256(output / name) for name in FIGURES}


def _signed(value: float) -> str:
    return f"{value:+.1f}".replace("-", "−")


def readme(inputs: dict[str, Any]) -> str:
    name, repo, base_name, base_repo = (
        inputs["model_name"],
        inputs["repo_id"],
        inputs["base_name"],
        inputs["base_repo"],
    )
    index = inputs["index"]
    own_i, base_i = index["own"]["balanced_skill"], index["base"]["balanced_skill"]
    lo, hi = index["delta_ci"]
    know_own, know_base = (
        index["own"]["areas"]["knowledge"],
        index["base"]["areas"]["knowledge"],
    )
    arena = {row["role"]: row for row in inputs["jevarena"]}
    own_a, base_a = arena["candidate"], arena["own-1.0"]
    latency = inputs["latency_ms"]
    example = json.dumps(inputs["example"]["questions"], indent=4)
    example = "\n".join(
        "    " + line if i else line for i, line in enumerate(example.splitlines())
    )
    lines = [
        "---",
        "license: apache-2.0",
        f"base_model: {base_repo}",
        "base_model_relation: finetune",
        "library_name: transformers",
        "tags:",
        "- decision-model",
        "- classification",
        "- system-one",
        "- reasoning",
        "- safetensors",
        "---",
        "",
        f"![{name}](assets/banner.png)",
        "",
        f"# {name}",
        "",
        f"**{name}** is the reasoning model of [{base_name}](https://huggingface.co/{base_repo}) in "
        f"[Decision 2.0]({inputs['collection_url']}), the decision models of "
        "[vLLM Semantic Router](https://github.com/vllm-project/semantic-router). It works through multi-step "
        "problems (arithmetic, code execution, causal questions, logical deduction) and still answers in a single "
        "forward pass: give it an input (text or JSON) and the questions you need answered, pick one of several "
        "options, say yes or no, or rate on a scale, and it returns a probability for every answer without "
        f"generating text. It is a drop-in replacement for {base_name}: same inputs, same API, same speed.",
        "",
        "| | |",
        "| --- | --- |",
        f"| **Parameters** | {inputs['parameters_label']} |",
        f"| **Context length** | {inputs['context_tokens']:,} tokens |",
        "| **Decision types** | Choice · Yes / No · Score |",
        "| **License** | Apache-2.0 |",
        "",
        "## Highlights",
        "",
        f"- **Better decisions than {base_name}:** {_signed(own_i - base_i)} on the Jev Decision Index "
        f"({own_i:.1f} vs. {base_i:.1f}; paired 95% interval {_signed(lo)} to {_signed(hi)}), "
        f"{index['rank_text']}.",
        f"- **Stronger multi-step reasoning:** Knowledge & Reasoning {know_own:.1f} vs. {know_base:.1f} "
        f"({_signed(know_own - know_base)}).",
        f"- **No extra cost:** the same architecture, size and single forward pass as {base_name}; a median of "
        f"{latency['own']:.1f} ms per request on one GPU ({base_name}: {latency['base']:.1f} ms on the same GPU "
        "and runtime).",
        "- **Many questions, one pass:** Choice, Yes / No and Score questions about the same input are answered "
        "together, with a probability for every option.",
        "",
        "## Quickstart",
        "",
        "```bash",
        'pip install "transformers>=5.17" torch safetensors',
        "```",
        "",
        "```python",
        "import json",
        "",
        "from transformers import AutoModel",
        "",
        f'model = AutoModel.from_pretrained("{repo}", trust_remote_code=True)',
        "result = model.system_one(",
        f"    state={json.dumps(inputs['example']['state'])},",
        f"    questions={example},",
        ")",
        'print(json.dumps(result["answers"], indent=2))',
        "",
        "# Or as a pipeline:",
        f'# transformers.pipeline("decision", model="{repo}", trust_remote_code=True)(state=..., questions=...)',
        "```",
        "",
        "## Evaluation",
        "",
        "| Model | Jev Decision Index ↑ | Knowledge & Reasoning ↑ | JevArena ↑ | Human-labelled transfer ↑ |",
        "| --- | ---: | ---: | ---: | ---: |",
        f"| **{name}** | **{own_i:.1f}** | **{know_own:.1f}** | **{own_a['score']:.1f}** | "
        f"**{own_a['transfer']:.1f}** |",
        f"| {base_name} | {base_i:.1f} | {know_base:.1f} | {base_a['score']:.1f} | {base_a['transfer']:.1f} |",
        "",
        "### Jev Decision Index",
        "",
        f"![Jev Decision Index by area: {name} and {base_name}](assets/index-areas.png)",
        "",
        "![Jev Decision Index against model size](assets/index-pareto.png)",
        "",
        f"<sub>{inputs['index_footnote']}</sub>",
        "",
        "### JevArena",
        "",
        f"![JevArena: {name} and same-size models](assets/jevarena.png)",
        "",
        f"![JevArena by decision type: {name} and same-size models](assets/jevarena-types.png)",
        "",
        "<sub>Every model answers the same frozen prompts, scored the same way; missing or invalid answers count as "
        "errors. JevArena answers were known to the project before this model (post-key comparison). "
        "Human-labelled transfer is the median macro-F1 over 15 human-labelled tasks (×100).</sub>",
        "",
        "## License",
        "",
        "Apache-2.0 ([LICENSE](LICENSE)).",
        "",
        "## Citation",
        "",
        "```bibtex",
        f"@misc{{{name.lower().replace('-', '_').replace('.', '_')}_2026,",
        f"  title        = {{{{{name}}}: A Decision 2.0 Reasoning Model for Structured Decisions}},",
        "  author       = {{vLLM Semantic Router Team}},",
        "  year         = {2026},",
        f"  howpublished = {{\\url{{https://huggingface.co/{repo}}}}}",
        "}",
        "```",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--logo", type=Path, required=True)
    parser.add_argument("--fonts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = json.loads(args.inputs.read_text(encoding="utf-8"))
    if inputs.get("schema") != SCHEMA:
        raise ValueError("unknown card input schema")
    figures = render_assets(inputs, args.logo, args.fonts, args.output / "assets")
    (args.output / "README.md").write_text(readme(inputs), encoding="utf-8")
    receipt = {
        "schema": "rsn-card-receipt/1",
        "model_name": inputs["model_name"],
        "inputs_sha256": sha256(args.inputs),
        "figures_sha256": figures,
        "readme_sha256": sha256(args.output / "README.md"),
    }
    (args.output / "card-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=1))


if __name__ == "__main__":
    main()
