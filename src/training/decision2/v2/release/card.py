"""Product model card of a Decision 2.0 release.

README layout: YAML metadata (licence, ``base_model``); the banner; the title; one
paragraph; an at-a-glance table; highlights; a code-only quickstart; evaluation
(JevArena overall and by decision type, the Jev Decision Index against model
size and by area, one compact table and the Index footnote); licence; citation.
Training details, methods, intervals and comparator notes stay in the release
records.

The banner and charts are PNGs rendered by ``card_assets`` outside the build; the
build copies them only after checking their receipt against the same reports,
Index input and weights. Index values come from a private input file
(``card_index``) and appear only on the published card.

Every comparator passes the licence filter. At a size without a Decision 1.0
model (``facts["comparison"] == "no-1.0"``) the JevArena counterpart slot is the
release gate's reference peer (role ``reference``) and the Index compares with
the family's next size down.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

from v2.release import card_index
from v2.release import licence as licence_policy
from v2.release.examples import EXAMPLES
from v2.release.layout import (
    CHART_FILES,
    CODENAMES,
    FORMER_ORG,
    MODEL_NAME,
    ORG,
    current_repo,
    sha_file,
)

COLLECTION_URL = (
    f"https://huggingface.co/collections/{ORG}/decision-20-6ab7cf7bdfb506bf8269cb00"
)
PROJECT_URL = "https://github.com/vllm-project/semantic-router"
FORBIDDEN = (
    (
        re.compile(r"gate ledger|release gate|promotion gate", re.I),
        "internal gate ledger",
    ),
    (re.compile(r"GPU-hours?", re.I), "internal compute accounting"),
    (re.compile(r"\bnode [A-F]\b|\b\d{1,3}(?:\.\d{1,3}){3}\b"), "machine identity"),
    (re.compile(r"/data/|/root/|/home/"), "private path"),
    (
        re.compile(r"hf_[A-Za-z0-9]{20,}|jv_live_|gho_[A-Za-z0-9]|apikey_"),
        "credential-like text",
    ),
    (
        re.compile(r"\bSOTA\b|state[- ]of[- ]the[- ]art", re.I),
        "unsupported leaderboard claim",
    ),
    (re.compile(r"JevArena[ -]v3|\bv3\b(?!\.)", re.I), "internal panel version"),
)
# The product README carries none of the internal evaluation or training vocabulary.
README_FORBIDDEN = (
    (re.compile(r"post-key", re.I), "post-key wording"),
    (re.compile(r"\bBrier\b|\bECE\b"), "calibration metrics"),
    (re.compile(r"mlx-diag", re.I), "development diagnostic"),
    (re.compile(r"Download and decide|from decision2 import"), "native runtime usage"),
    (re.compile(r"\bowl\b", re.I), "banner artwork"),
    (re.compile(r"\b[0-9a-f]{40}\b"), "revision hash"),
    (re.compile(r"JevBench", re.I), "JevBench results"),
    (
        re.compile(r"\bLoRA\b|\bBF16\b|\bFP32\b|\bbf16\b|fp32", re.I),
        "training or precision details",
    ),
    (re.compile(r"^## (?:Limitations|Training data)", re.M), "removed card section"),
    (re.compile(r"ATTRIBUTIONS\.md|\]\(NOTICE\)|evaluation/"), "removed card file"),
    (
        re.compile(r"stock 🤗 Transformers|stock Transformers", re.I),
        "plumbing highlight",
    ),
    (re.compile(re.escape(FORMER_ORG)), "former Hugging Face organization"),
)
TEXT_KEYS = {"description", "staging_notice"}
CITATION_YEAR = 2026
AUTHOR = "vLLM Semantic Router Team"
NO_OWN_1_0 = "no-1.0"
MLX_SCHEMA = "dev2-mlx-diag-score/1"
MLX_CARD_TYPES = ("choice", "noul")
TRANSFORMERS_HEADING = "## Quickstart"
BANNER = "assets/banner.png"
ASSETS_SCHEMA = "dev2-card-assets/1"
ASSETS_RECEIPT = "card-assets.json"
CHART_JEVARENA, CHART_TYPES, CHART_PARETO, CHART_AREAS = CHART_FILES
# Index gains below this do not round to a positive one-decimal value and are not stated.
INDEX_GAIN_MIN = 0.05
JEVARENA_NOTE = (
    "Every model answers the same frozen prompts, scored the same way; missing or invalid answers count as "
    "errors. Human-labelled transfer is the median macro-F1 over 15 human-labelled tasks (×100)."
)


def lint(text: str) -> list[str]:
    return sorted({reason for pattern, reason in FORBIDDEN if pattern.search(text)})


def lint_readme(text: str) -> list[str]:
    return sorted(
        set(lint(text))
        | {reason for pattern, reason in README_FORBIDDEN if pattern.search(text)}
    )


def comparison_of(spec: dict[str, Any]) -> str:
    """``no-1.0`` for a gated tier without a Decision 1.0 model, else ``own-1.0``."""
    tier = (spec.get("gate_profile") or {}).get("tier") or {}
    return (
        NO_OWN_1_0
        if tier.get("no_own_1_0") is True or tier.get("no_1_0") is True
        else "own-1.0"
    )


def _report(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _mlx(entries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Optional mlx-diag scores per entry key; all must score the same gold panel."""
    scores = {}
    for entry in entries:
        if not entry.get("mlx"):
            continue
        value = _report(Path(entry["mlx"]))
        if value.get("schema") != MLX_SCHEMA or value.get("invalid_or_missing") is None:
            raise ValueError(f"{entry['key']}: not an mlx-diag score file")
        scores[entry["key"]] = {"data": value, "sha256": sha_file(Path(entry["mlx"]))}
    if (
        len({(s["data"]["gold_sha256"], s["data"]["items"]) for s in scores.values()})
        > 1
    ):
        raise ValueError("mlx-diag scores come from different panels")
    return scores


def select_reports(
    entries: list[dict[str, Any]], roster_path: Path, comparison: str = "own-1.0"
) -> dict[str, Any]:
    """Validate one same panel, apply the licence filter, keep display order."""
    from v2.eval.charts import load_reports

    load_reports([Path(e["report"]) for e in entries])
    mlx = _mlx(entries)
    roster = licence_policy.load_roster(roster_path)
    shown, excluded = [], []
    roles = [e.get("role") for e in entries]
    if comparison == NO_OWN_1_0:
        if (
            roles.count("candidate") != 1
            or roles.count("reference") != 1
            or "own-1.0" in roles
        ):
            raise ValueError(
                "A card without a Decision 1.0 model needs exactly one candidate, "
                "one reference peer and no own Decision 1.0 comparator"
            )
    elif roles.count("candidate") != 1 or roles.count("own-1.0") != 1:
        raise ValueError(
            "A card needs exactly one candidate and one own Decision 1.0 comparator"
        )
    for entry in entries:
        report = _report(Path(entry["report"]))
        declared = report["model"].get("model_id")
        repo = entry.get("repo_id")
        if (
            entry["role"] != "candidate"
            and declared
            and repo
            and current_repo(declared) != current_repo(repo)
        ):
            raise ValueError(
                f"{entry['key']}: report model {declared} differs from {repo}"
            )
        decision = licence_policy.card_eligibility(
            {
                "repo_id": repo,
                "family": report["model"].get("family"),
                "label": report["model"].get("label"),
                "candidate": entry["role"] == "candidate",
                "board_entry": entry.get("board_entry"),
            },
            roster,
        )
        item = {
            **entry,
            "data": report,
            "sha256": sha_file(Path(entry["report"])),
            "licence": decision["licence"],
            "reason": decision["reason"],
            "mlx": (mlx.get(entry["key"]) or {}).get("data"),
            "mlx_sha256": (mlx.get(entry["key"]) or {}).get("sha256"),
        }
        (shown if decision["eligible"] else excluded).append(item)
    if comparison != NO_OWN_1_0 and not any(e["role"] == "own-1.0" for e in shown):
        raise ValueError(
            "The own Decision 1.0 comparator failed the card licence filter"
        )
    return {"shown": shown, "excluded": excluded}


def _label(entry: dict[str, Any]) -> str:
    return entry.get("label") or entry["data"]["model"]["label"]


def _score(entry: dict[str, Any]) -> float:
    return entry["data"]["v3"]["score"]


def _transfer(entry: dict[str, Any]) -> float:
    return entry["data"]["v3"]["H"]


def _pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def _typed(report: dict[str, Any], kind: str) -> tuple[int, int]:
    item = report["panels"]["typed-final"]["by_type"][kind]
    return item["correct"], item["n"]


def loaded_parameters(entry: dict[str, Any], candidate_loaded: int | None) -> int:
    """The packaged candidate shows the count its runtime asserts; others their reports'."""
    if entry["role"] == "candidate" and candidate_loaded:
        return candidate_loaded
    return entry["data"]["parameters"]["loaded"]


TASK_NAMES = {
    "conv_go_awry": "Conversations Gone Awry",
    "emotion": "Emotion",
    "flute": "FLUTE figurative language",
    "ibc": "Ideological Books Corpus",
    "indian_english_dialect": "Indian English dialect",
    "media_ideology": "Media ideology",
    "mrf": "Misinfo Reaction Frames",
    "persuasion": "Persuasion",
    "raop": "Random Acts of Pizza",
    "reddit_humor": "Reddit humour",
    "talklife": "TalkLife empathy",
    "tempowic": "TempoWiC",
    "tropes": "Character tropes",
    "wiki_corpus": "Wikipedia power",
    "wiki_politeness": "Wikipedia politeness",
}


def _task(name: str) -> str:
    return TASK_NAMES.get(name, name.replace("_", " "))


def tradeoff_rows(
    candidate: dict[str, Any],
    own: dict[str, Any],
    candidate_mlx: dict[str, Any] | None = None,
    own_mlx: dict[str, Any] | None = None,
) -> list[tuple[str, str, str]]:
    """Every per-type, transfer, per-task, public and mlx-diag result below the counterpart."""
    rows = []
    for kind in ("choice", "noul", "score"):
        (c, n), (o, _) = _typed(candidate, kind), _typed(own, kind)
        if c < o:
            rows.append((f"Typed {kind.title()} (correct)", f"{c}/{n}", f"{o}/{n}"))
    if candidate["v3"]["H"] < own["v3"]["H"]:
        rows.append(
            (
                "Human-labelled transfer (median task macro-F1)",
                f"{candidate['v3']['H']:.3f}",
                f"{own['v3']['H']:.3f}",
            )
        )
    if candidate_mlx and own_mlx:
        for kind in MLX_CARD_TYPES:
            mine = candidate_mlx["by_type"][kind]["non_english_mean_accuracy"]
            theirs = own_mlx["by_type"][kind]["non_english_mean_accuracy"]
            if mine < theirs:
                rows.append(
                    (
                        f"mlx-diag non-English {kind.title()} (accuracy)",
                        _pct(mine),
                        _pct(theirs),
                    )
                )
    tasks_c = candidate["panels"]["css15"]["tasks"]
    tasks_o = own["panels"]["css15"]["tasks"]
    for task in sorted(
        tasks_c, key=lambda t: tasks_c[t]["macro_f1"] - tasks_o[t]["macro_f1"]
    ):
        if tasks_c[task]["macro_f1"] < tasks_o[task]["macro_f1"]:
            rows.append(
                (
                    f"Transfer: {_task(task)} (macro-F1)",
                    _pct(tasks_c[task]["macro_f1"]),
                    _pct(tasks_o[task]["macro_f1"]),
                )
            )
    public_c, public_o = candidate["panels"]["public231"], own["panels"]["public231"]
    if public_c["correct"] < public_o["correct"]:
        rows.append(
            (
                "JevBench public 231 (correct)",
                f"{public_c['correct']}/{public_c['items']}",
                f"{public_o['correct']}/{public_o['items']}",
            )
        )
    tiers_c, tiers_o = public_c["tiers"], public_o["tiers"]
    for tier in ("easy", "standard", "hard"):
        if tier in tiers_c and tiers_c[tier]["correct"] < tiers_o[tier]["correct"]:
            rows.append(
                (
                    f"JevBench public {tier} (correct)",
                    f"{tiers_c[tier]['correct']}/{tiers_c[tier]['items']}",
                    f"{tiers_o[tier]['correct']}/{tiers_o[tier]['items']}",
                )
            )
    return rows


def tradeoffs(
    candidate: dict[str, Any],
    own: dict[str, Any],
    candidate_mlx: dict[str, Any] | None = None,
    own_mlx: dict[str, Any] | None = None,
) -> list[str]:
    return [
        f"{name}: {mine} versus {theirs}"
        for name, mine, theirs in tradeoff_rows(candidate, own, candidate_mlx, own_mlx)
    ]


def _paired(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    value = _report(path)
    delta, ci95 = value["point"]["delta"], value["ci95"]
    return {
        "delta": delta["score"] if isinstance(delta, dict) else delta,
        "ci95": [ci95["low"], ci95["high"]] if isinstance(ci95, dict) else list(ci95),
        "replicates": value.get("replicates"),
        "sha256": sha_file(path),
    }


def _paired_peers(
    shown: list[dict[str, Any]], paths: dict[str, Path]
) -> dict[str, dict[str, Any]]:
    """Candidate-minus-peer intervals for shown peers; each file must pair exactly those two reports."""
    candidate = _score(next(e for e in shown if e["role"] == "candidate"))
    result = {}
    for entry in shown:
        path = paths.get(entry["key"])
        if path is None or entry["role"] == "candidate":
            continue
        point = _report(Path(path))["point"]
        if (
            abs(point["left"]["score"] - candidate) > 1e-9
            or abs(point["right"]["score"] - _score(entry)) > 1e-9
        ):
            raise ValueError(
                f"{entry['key']}: paired file is not candidate minus this peer"
            )
        result[entry["key"]] = _paired(Path(path))
    return result


def _example_arguments() -> tuple[str, str]:
    example = EXAMPLES[0]
    state = json.dumps(example["state"], ensure_ascii=False)
    questions = json.dumps(example["questions"], ensure_ascii=False, indent=4).replace(
        "\n", "\n    "
    )
    return state, questions


def pip_line(facts: dict[str, Any]) -> str:
    requirements = facts.get("runtime_requirements") or {}
    packages = ['"transformers>=5.17"', "torch", "safetensors"]
    if "peft" in requirements:
        packages.append("peft")
    return "pip install " + " ".join(packages)


def _front_matter(facts: dict[str, Any]) -> list[str]:
    lic, repo = facts["licence"], facts["repo_id"]
    front = ["---", f"license: {lic['spdx']}"]
    if lic["spdx"] == "other":
        # The Hub's metadata validator accepts only an https URI here.
        front += [
            f"license_name: {lic['license_name']}",
            f"license_link: https://huggingface.co/{repo}/blob/main/LICENSING.md",
        ]
    origin = facts["origin"]
    front.append(f"base_model: {origin['repo_id']}")
    if origin.get("relation") in ("finetune", "adapter", "merge"):
        front.append(f"base_model_relation: {origin['relation']}")
    front += [
        "library_name: transformers",
        "tags:",
        "- decision-model",
        "- classification",
        "- system-one",
        "- safetensors",
        "---",
    ]
    return front


def citation(facts: dict[str, Any]) -> list[str]:
    name, repo = facts["model_name"], facts["repo_id"]
    key = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return [
        "```bibtex",
        f"@misc{{{key}_{CITATION_YEAR},",
        f"  title        = {{{{{name}}}: A Decision 2.0 Model for Structured Decisions}},",
        f"  author       = {{{{{AUTHOR}}}}},",
        f"  year         = {{{CITATION_YEAR}}},",
        f"  howpublished = {{\\url{{https://huggingface.co/{repo}}}}}",
        "}",
        "```",
    ]


def transformers_example(repo: str, shared: bool = False) -> str:
    state, questions = _example_arguments()
    many = (
        "\n# Many questions about one long input: share_context=True computes the input they share once\n"
        "# (several times faster at dozens of questions; near-tie answers can differ slightly):\n"
        "# model.system_one(state=..., questions=..., share_context=True)\n"
        if shared
        else ""
    )
    return (
        "import json\n\n"
        "from transformers import AutoModel\n\n"
        f'model = AutoModel.from_pretrained("{repo}", trust_remote_code=True)\n'
        "result = model.system_one(\n"
        f"    state={state},\n"
        f"    questions={questions},\n"
        ")\n"
        'print(json.dumps(result["answers"], indent=2))\n\n'
        "# Or as a pipeline:\n"
        f'# transformers.pipeline("decision", model="{repo}", trust_remote_code=True)(state=..., questions=...)\n'
        f"{many}"
    )


def _significance(paired: dict[str, Any] | None) -> str | None:
    if not paired:
        return None
    low, high = paired["ci95"]
    return "above" if low > 0 else "below" if high < 0 else "level"


def _size(name: str) -> str:
    return f"{MODEL_NAME.match(name)['size']}B"


def _one(value: float) -> str:
    return f"{value:.1f}"


def _ms(value: float) -> str:
    """Milliseconds at a precision a reader can compare: one decimal below 100, whole above."""
    return f"{value:.1f}" if value < 100 else f"{value:,.0f}"


def highlights(ctx: dict[str, Any]) -> list[str]:
    """Product claims, each true for this model: rank, generation gain, speed, one pass."""
    facts, shown = ctx["facts"], ctx["shown"]
    candidate, own, index = ctx["candidate"], ctx["own"], ctx["index"]
    score = _score(candidate)
    lines = []
    ordered = sorted(shown, key=lambda e: -_score(e))
    if ordered[0] is candidate and len(ordered) > 1:
        runner = ordered[1]
        paired = (
            ctx["paired"] if runner is own else ctx["paired_peers"].get(runner["key"])
        )
        if _significance(paired) == "level" or (
            paired is None and score - _score(runner) < 1
        ):
            line = (
                f"**Top JevArena score of its size:** {_one(score)} among the {len(ordered)} same-size "
                f"models compared, statistically level with {_label(runner)} ({_one(_score(runner))})."
            )
        else:
            others = (
                "the other same-size model compared"
                if len(ordered) == 2
                else f"the {len(ordered) - 1} other same-size models compared"
            )
            line = (
                f"**Top JevArena score of its size:** {_one(score)}, ahead of {others}."
            )
        lines.append(line)
    if ctx["comparison"] != NO_OWN_1_0 and own is not None:
        gains = []
        delta = score - _score(own)
        state = _significance(ctx["paired"])
        if state == "above" or (state is None and delta > 0):
            gains.append(f"+{_one(delta)} on JevArena")
        if index["delta"] >= INDEX_GAIN_MIN:
            gains.append(f"+{_one(index['delta'])} on the Jev Decision Index")
        if gains:
            lines.append(f"**Ahead of {_label(own)}:** " + " and ".join(gains) + ".")
    elif (
        index["compare_kind"] == "family"
        and index["delta"] >= INDEX_GAIN_MIN
        and index["own"]["balanced_skill"]
        == max(p["balanced_skill"] for p in index["family"])
    ):
        lines.append(
            f"**The strongest Decision 2.0 model:** +{_one(index['delta'])} on the Jev Decision Index "
            f"over {index['compare_name']}."
        )
    speed = facts.get("speed")
    if speed:
        line = f"**Speed:** a median of {_one(speed['median_ms'])} ms per single-question request on a single GPU"
        shared = facts.get("speed_shared")
        if shared:
            line += (
                f"; {shared['questions']} questions about one input take {_ms(shared['on_ms'])} ms with "
                f"`share_context=True` instead of {_ms(shared['off_ms'])} ms"
            )
        lines.append(line + ".")
    lines.append(
        "**Many questions, one pass:** Choice, Yes / No and Score questions about the same input are "
        "answered together in one forward pass, with a probability for every option."
    )
    return lines


def pitch(facts: dict[str, Any], text: dict[str, Any], staging: bool) -> str:
    if text.get("description"):
        return text["description"]
    name = facts["model_name"]
    family = "Decision 2.0" if staging else f"[Decision 2.0]({COLLECTION_URL})"
    return (
        f"**{name}** is the {_size(name)} model of {family}, the decision models of "
        f"[vLLM Semantic Router]({PROJECT_URL}). Give it an input (text or JSON) and the questions you "
        "need answered: pick one of several options, say yes or no, or rate on a scale. It answers them "
        "all at once and returns a probability for every answer, without generating text."
    )


def at_a_glance(facts: dict[str, Any]) -> list[str]:
    lic = facts["licence"]
    licence_cell = (
        "Apache-2.0"
        if lic["spdx"] == "apache-2.0"
        else f"{lic['spdx']} ([components](LICENSING.md))"
    )
    return [
        "| | |",
        "| --- | --- |",
        f"| **Parameters** | {facts['parameters']['loaded'] / 1e9:.2f}B |",
        f"| **Context length** | {facts['max_input_tokens']:,} tokens |",
        "| **Decision types** | Choice · Yes / No · Score |",
        f"| **License** | {licence_cell} |",
    ]


def results_table(ctx: dict[str, Any]) -> list[str]:
    index = ctx["index"]
    decision1 = {p["name"]: p["balanced_skill"] for p in index["decision1"]}
    rows = [
        "| Model | JevArena ↑ | Human-labelled transfer ↑ | Jev Decision Index ↑ |",
        "| --- | ---: | ---: | ---: |",
    ]
    for entry in sorted(ctx["shown"], key=lambda e: -_score(e)):
        value = "—"
        if entry["role"] == "candidate":
            value = _one(index["own"]["balanced_skill"])
        elif _label(entry) in decision1:
            value = _one(decision1[_label(entry)])
        cells = [
            _label(entry),
            _one(_score(entry)),
            _one(100 * _transfer(entry)),
            value,
        ]
        if entry["role"] == "candidate":
            cells = [f"**{c}**" for c in cells]
        rows.append("| " + " | ".join(cells) + " |")
    return rows


def render_readme(ctx: dict[str, Any]) -> str:
    facts, text = ctx["facts"], ctx["text"]
    name, repo = facts["model_name"], facts["repo_id"]
    staging = text.get("staging_notice")
    lic = facts["licence"]
    lines = [*_front_matter(facts), "", f"![{name}]({BANNER})", "", f"# {name}", ""]
    if staging:
        lines += [f"> **{staging}**", ""]
    lines += [pitch(facts, text, bool(staging)), "", *at_a_glance(facts), ""]
    lines += ["## Highlights", "", *[f"- {item}" for item in highlights(ctx)], ""]
    lines += [
        TRANSFORMERS_HEADING,
        "",
        "```bash",
        pip_line(facts),
        "```",
        "",
        "```python",
        transformers_example(repo, bool(facts.get("speed_shared"))).rstrip("\n"),
        "```",
        "",
        "## Evaluation",
        "",
        *results_table(ctx),
        "",
        "### JevArena",
        "",
        f"![JevArena: {name} and same-size models]({CHART_JEVARENA})",
        "",
        f"![JevArena by decision type: {name} and same-size models]({CHART_TYPES})",
        "",
        f"<sub>{JEVARENA_NOTE}</sub>",
        "",
        "### Jev Decision Index",
        "",
        f"![Jev Decision Index against model size]({CHART_PARETO})",
        "",
        f"![Jev Decision Index by area: {name} and {ctx['index']['compare_name']}]({CHART_AREAS})",
        "",
        f"<sub>{ctx['index']['footnote']}</sub>",
        "",
        "## License",
        "",
        (
            "Apache-2.0 ([LICENSE](LICENSE))."
            if lic["spdx"] == "apache-2.0"
            else f"`{lic['spdx']}` ([LICENSE](LICENSE); per-component terms in [LICENSING.md](LICENSING.md))."
        ),
        "",
        "## Citation",
        "",
        *citation(facts),
        "",
    ]
    return "\n".join(lines)


def check_assets(
    receipt_path: Path,
    shown: list[dict[str, Any]],
    index_sha256: str,
    facts: dict[str, Any],
) -> dict[str, Any]:
    """The rendered banner and charts must come from exactly this card's inputs."""
    receipt = _report(receipt_path)
    inputs = receipt.get("inputs") or {}
    problems = []
    if receipt.get("schema") != ASSETS_SCHEMA:
        problems.append("not a card-assets receipt")
    if receipt.get("model_name") != facts["model_name"]:
        problems.append("rendered for another model name")
    if receipt.get("model_sha256") != facts["model_sha256"]:
        problems.append("rendered for other weights")
    if inputs.get("reports") != {e["key"]: e["sha256"] for e in shown}:
        problems.append("rendered from other reports")
    if inputs.get("index_sha256") != index_sha256:
        problems.append("rendered from another Index input")
    if set(receipt.get("files") or {}) != {BANNER, *CHART_FILES}:
        problems.append("receipt does not list the banner and the four charts")
    for name, digest in (receipt.get("files") or {}).items():
        if sha_file(receipt_path.parent / name) != digest:
            problems.append(f"{name} differs from its receipt")
    if problems:
        raise ValueError(f"Card assets: {'; '.join(problems)}")
    return receipt


def build_card(
    *,
    entries: list[dict[str, Any]],
    roster: Path,
    paired: Path | None,
    facts: dict[str, Any],
    text: dict[str, Any],
    work: Path,
    output: Path,
    index: Path,
    assets: Path,
    paired_peers: dict[str, Path] | None = None,
    files: set[str] | None = None,
) -> dict[str, Any]:
    """Write README.md and assets/ into ``output``; return digests (no Index values).

    ``index`` is the private Index input, ``assets`` the directory ``card_assets``
    rendered with its receipt. ``paired_peers`` (report key -> candidate-minus-peer
    paired file) tells whether the runner-up is level. ``files`` lists the package
    files the README may link to (default: those under ``output``).
    """
    unknown = set(text) - TEXT_KEYS
    if unknown:
        raise ValueError(f"Card text keys not used by this card: {sorted(unknown)}")
    if facts.get("remote_code") is None:
        raise ValueError("The card's quickstart needs the Transformers remote code")
    if not MODEL_NAME.match(facts["model_name"]):
        raise ValueError(f"Not a Decision 2.0 model name: {facts['model_name']}")
    comparison = facts.get("comparison", "own-1.0")
    selection = select_reports(entries, roster, comparison)
    shown = selection["shown"]
    if len(shown) < 2:
        raise ValueError(
            "A card needs the candidate and at least one eligible comparator"
        )
    candidate = next(e for e in shown if e["role"] == "candidate")
    if _label(candidate) != facts["model_name"]:
        raise ValueError("The candidate's card label must be the model name")
    tier = next(
        t
        for t, c in CODENAMES.items()
        if c == MODEL_NAME.match(facts["model_name"])["codename"]
    )
    view = card_index.view(card_index.load(index), tier, facts["model_sha256"])
    slot = "reference" if comparison == NO_OWN_1_0 else "own-1.0"
    ctx = {
        "facts": facts,
        "text": text,
        "shown": shown,
        "excluded": selection["excluded"],
        "comparison": comparison,
        "candidate": candidate,
        "own": next((e for e in shown if e["role"] == slot), None),
        "paired": _paired(paired),
        "paired_peers": _paired_peers(shown, paired_peers or {}),
        "index": view,
    }
    receipt = check_assets(Path(assets) / ASSETS_RECEIPT, shown, view["sha256"], facts)
    for name in receipt["files"]:
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(assets) / name, target)
    work.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(Path(assets) / ASSETS_RECEIPT, work / ASSETS_RECEIPT)
    present = (
        files
        if files is not None
        else {
            p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()
        }
    )
    readme = render_readme(ctx)
    problems = check_rendered(readme, present | {"LICENSE"})
    if problems:
        raise ValueError(f"README failed the card checks: {problems}")
    (output / "README.md").write_text(readme, encoding="utf-8")
    return {
        "readme_sha256": sha_file(output / "README.md"),
        "figures_sha256": dict(receipt["files"]),
        "assets_receipt_sha256": sha_file(Path(assets) / ASSETS_RECEIPT),
        "index_sha256": view["sha256"],
        "shown": [
            {"key": e["key"], "licence": e["licence"], "report_sha256": e["sha256"]}
            for e in shown
        ],
        "excluded": [
            {"key": e["key"], "reason": e["reason"]} for e in selection["excluded"]
        ],
        "tradeoffs": (
            tradeoffs(
                ctx["candidate"]["data"],
                ctx["own"]["data"],
                ctx["candidate"].get("mlx"),
                ctx["own"].get("mlx"),
            )
            if ctx["own"]
            else []
        ),
        "paired": ctx["paired"],
    }


REQUIRED_SECTIONS = (
    "## Highlights",
    "## Quickstart",
    "## Evaluation",
    "## License",
    "## Citation",
)


def check_rendered(readme: str, files: set[str]) -> list[str]:
    """Structural card checks on a rendered README against the repository file list."""
    problems = lint_readme(readme)
    if not readme.startswith("---\n") or "\n---\n" not in readme[4:]:
        problems.append("missing YAML front matter")
    body = readme.split("\n---\n", 1)[-1].lstrip("\n")
    if not body.startswith(f"![") or f"]({BANNER})" not in body.split("\n", 1)[0]:
        problems.append("the banner must open the card")
    for link in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", readme):
        if not link.startswith("http") and link not in files:
            problems.append(f"image does not resolve: {link}")
    for link in re.findall(r"(?<!!)\[[^\]]*\]\(([^)#]+)\)", readme):
        if not link.startswith("http") and link not in files:
            problems.append(f"link does not resolve: {link}")
    headings = re.findall(r"^#{1,3} .+$", readme, flags=re.M)
    for required in REQUIRED_SECTIONS:
        if required not in headings:
            problems.append(f"missing section: {required}")
    order = [h for h in headings if h in REQUIRED_SECTIONS]
    if order != [h for h in REQUIRED_SECTIONS if h in order]:
        problems.append("sections out of order")
    if readme.count("```python") != 1:
        problems.append("the quickstart needs exactly one Python block")
    for chart in CHART_FILES:
        if f"]({chart})" not in readme:
            problems.append(f"missing chart: {chart}")
    return problems
