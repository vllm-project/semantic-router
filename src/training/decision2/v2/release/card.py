"""Product model card in the Decision 1.0 design, from same-panel reports only.

Layout: owl banner, what the model is for, the three decision types, measured
same-panel results (score table, rank chart, model x task chart, public-231
rank chart), tradeoffs versus the tier's own Decision 1.0 model, a runnable
local System One example, then short model details and limits. No Pareto chart,
no internal gate ledger. Every comparator passes the licence filter; charts come
from the eval track's generator (``v2.eval.charts``) on relabeled copies of the
exact reports (display names only).
"""

from __future__ import annotations

import json
import math
import re
import shutil
from pathlib import Path
from typing import Any

from v2.release import licence as licence_policy
from v2.release.examples import EXAMPLES
from v2.release.layout import CHART_FILES, sha_file

COLLECTION_URL = "https://huggingface.co/collections/llm-semantic-router/decision-20-6ab7cf7bdfb506bf8269cb00"
FORBIDDEN = (
    (re.compile(r"pareto", re.I), "Pareto content"),
    (
        re.compile(r"gate ledger|release gate|promotion gate", re.I),
        "internal gate ledger",
    ),
    (re.compile(r"GPU-hours?", re.I), "internal compute accounting"),
    (re.compile(r"\bnode [AB]\b|\b\d{1,3}(?:\.\d{1,3}){3}\b"), "machine identity"),
    (re.compile(r"/data/|/root/|/home/"), "private path"),
    (
        re.compile(r"hf_[A-Za-z0-9]{20,}|jv_live_|gho_[A-Za-z0-9]|apikey_"),
        "credential-like text",
    ),
    (
        re.compile(r"\bSOTA\b|state[- ]of[- ]the[- ]art", re.I),
        "unsupported leaderboard claim",
    ),
)
TYPED_N = {"choice": 800, "noul": 800, "score": 400}
ARCHITECTURE = {
    "kai-native": (
        "Three 22-layer bidirectional encoder paths share multilingual embeddings. "
        "Each decision type has its own interaction layers and candidate readout; "
        "the candidates of a question are scored together in one pass."
    ),
    "qwen-full": (
        "A causal text backbone reads the state, the question and every supplied "
        "candidate once. A shared candidate head scores the candidates against a "
        "global query and returns probabilities without generating text."
    ),
    "qwen-adapter": (
        "A LoRA adapter over the pinned base text backbone reads the state, the "
        "question and every supplied candidate once. A shared candidate head scores "
        "the candidates against a global query and returns probabilities without "
        "generating text."
    ),
}


def lint(text: str) -> list[str]:
    return sorted({reason for pattern, reason in FORBIDDEN if pattern.search(text)})


def _report(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def select_reports(entries: list[dict[str, Any]], roster_path: Path) -> dict[str, Any]:
    """Validate one same panel, apply the licence filter, keep display order."""
    from v2.eval.charts import load_reports

    load_reports([Path(e["report"]) for e in entries])
    roster = licence_policy.load_roster(roster_path)
    shown, excluded = [], []
    roles = [e.get("role") for e in entries]
    if roles.count("candidate") != 1 or roles.count("own-1.0") != 1:
        raise ValueError(
            "A card needs exactly one candidate and one own Decision 1.0 comparator"
        )
    for entry in entries:
        report = _report(Path(entry["report"]))
        declared = report["model"].get("model_id")
        repo = entry.get("repo_id")
        if entry["role"] != "candidate" and declared and repo and declared != repo:
            raise ValueError(
                f"{entry['key']}: report model {declared} differs from {repo}"
            )
        decision = licence_policy.card_eligibility(
            {
                "repo_id": repo,
                "family": report["model"].get("family"),
                "label": report["model"].get("label"),
                "candidate": entry["role"] == "candidate",
            },
            roster,
        )
        item = {
            **entry,
            "data": report,
            "sha256": sha_file(Path(entry["report"])),
            "licence": decision["licence"],
            "reason": decision["reason"],
        }
        (shown if decision["eligible"] else excluded).append(item)
    if not any(e["role"] == "own-1.0" for e in shown):
        raise ValueError(
            "The own Decision 1.0 comparator failed the card licence filter"
        )
    return {"shown": shown, "excluded": excluded}


def render_charts(
    shown: list[dict[str, Any]], work: Path, assets: Path
) -> dict[str, str]:
    """Eval-track chart generator on relabeled copies; its receipt stays private."""
    from v2.eval.charts import render

    relabeled = work / "relabeled-reports"
    relabeled.mkdir(parents=True, exist_ok=False)
    paths = []
    for entry in shown:
        data = json.loads(json.dumps(entry["data"]))
        data["model"]["label"] = entry.get("label") or data["model"]["label"]
        path = relabeled / f"{entry['key']}.json"
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        paths.append(path)
    output = work / "charts"
    receipt = render(paths, output)
    shutil.move(str(output / "charts.json"), str(work / "charts-receipt.json"))
    assets.mkdir(parents=True, exist_ok=True)
    figures = {}
    for relative in CHART_FILES:
        name = Path(relative).name
        if not (output / name).is_file():
            raise ValueError(f"Chart generator did not produce {name}")
        shutil.copyfile(output / name, assets / name)
        figures[relative] = sha_file(assets / name)
        if lint((assets / name).read_text(encoding="utf-8")):
            raise ValueError(f"{name} failed the card content lint")
    if set(receipt["figures"]) != {Path(r).name for r in CHART_FILES}:
        raise ValueError(
            "Unexpected chart set; cards carry exactly rank, model x task and public-231 rank"
        )
    return figures


def _pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def _typed(report: dict[str, Any], kind: str) -> tuple[int, int]:
    item = report["panels"]["typed-final"]["by_type"][kind]
    return item["correct"], item["n"]


def score_table(shown: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    ordered = sorted(shown, key=lambda e: -e["data"]["v3"]["score"])
    lines = [
        "| Rank | Model | Parameters | JevArena v3 ↑ | Choice | Noul | Score | Human transfer | JevBench public 231 ↑ |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, entry in enumerate(ordered, 1):
        report = entry["data"]
        label = entry.get("label") or report["model"]["label"]
        bold = entry["role"] == "candidate"
        name = f"**{label}**" if bold else label
        loaded = report["parameters"]["loaded"]
        typed = " | ".join(
            f"{c}/{n}"
            for c, n in (_typed(report, k) for k in ("choice", "noul", "score"))
        )
        public = report["panels"]["public231"]
        v3 = f"{report['v3']['score']:.2f}"
        lines.append(
            f"| {rank} | {name} | {loaded / 1e9:.2f}B | {'**' + v3 + '**' if bold else v3} | {typed} | "
            f"{_pct(report['v3']['H'])} | {public['correct']}/{public['items']} |"
        )
    return "\n".join(lines), ordered


def tradeoff_rows(
    candidate: dict[str, Any], own: dict[str, Any]
) -> list[tuple[str, str, str]]:
    """Every per-type, per-task and public-tier regression versus the tier's own 1.0 model."""
    rows = []
    for kind in ("choice", "noul", "score"):
        (c, n), (o, _) = _typed(candidate, kind), _typed(own, kind)
        if c < o:
            rows.append((f"Typed {kind.title()} (correct)", f"{c}/{n}", f"{o}/{n}"))
    tasks_c = candidate["panels"]["css15"]["tasks"]
    tasks_o = own["panels"]["css15"]["tasks"]
    for task in sorted(
        tasks_c, key=lambda t: tasks_c[t]["macro_f1"] - tasks_o[t]["macro_f1"]
    ):
        if tasks_c[task]["macro_f1"] < tasks_o[task]["macro_f1"]:
            rows.append(
                (
                    f"Transfer: {task.replace('_', ' ')} (macro-F1)",
                    _pct(tasks_c[task]["macro_f1"]),
                    _pct(tasks_o[task]["macro_f1"]),
                )
            )
    tiers_c = candidate["panels"]["public231"]["tiers"]
    tiers_o = own["panels"]["public231"]["tiers"]
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


def tradeoffs(candidate: dict[str, Any], own: dict[str, Any]) -> list[str]:
    return [
        f"{name}: {mine} versus {theirs}"
        for name, mine, theirs in tradeoff_rows(candidate, own)
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


def code_example(local: str) -> str:
    example = EXAMPLES[0]
    state = json.dumps(example["state"], ensure_ascii=False)
    questions = json.dumps(example["questions"], ensure_ascii=False, indent=4).replace(
        "\n", "\n    "
    )
    return (
        "import json\n"
        "import os\n"
        "import sys\n\n"
        f'sys.path.insert(0, os.path.abspath("{local}"))\n'
        "from decision2 import Decision2\n\n"
        f'model = Decision2.from_pretrained("{local}")  # CPU, or one CUDA/ROCm GPU if present\n'
        "result = model.system_one(\n"
        f"    state={state},\n"
        f"    questions={questions},\n"
        ")\n"
        'print(json.dumps(result["answers"], indent=2))\n'
    )


def auto_limits(candidate: dict[str, Any], cap: int) -> list[str]:
    limits = [
        "JevArena v3 answers were available during development, so these are post-key "
        "same-panel comparisons, not an untouched blind test.",
    ]
    screen = candidate["slices"].get("language_screen") or {}
    total = sum(sum(counts.values()) for counts in screen.values())
    english = sum(counts.get("en", 0) for counts in screen.values())
    if not total or english / total >= 0.95:
        limits.append(
            "The v3 and public panels are almost entirely English; they do not measure multilingual ability."
        )
    over = 0
    for panel in ("typed-final", "css15", "public231"):
        reasons = (candidate.get("invalid", {}).get(panel) or {}).get("reasons") or {}
        over += sum(
            v for k, v in reasons.items() if "overflow" in k or "max_length" in k
        )
    invalid = sum(
        (candidate.get("invalid", {}).get(p) or {}).get("invalid_or_missing", 0)
        for p in ("typed-final", "css15", "public231")
    )
    limits.append(
        f"Complete inputs above {cap:,} tokens are rejected, never truncated"
        + (
            f"; {invalid:,} evaluation answers were invalid (including over-budget inputs) and counted as failures."
            if invalid
            else "."
        )
    )
    limits.append(
        "It decides from the supplied state only and does not retrieve missing facts. "
        "Probabilities are estimates; review consequential decisions against the evidence."
    )
    return limits


def render_readme(ctx: dict[str, Any]) -> str:
    facts, text = ctx["facts"], ctx["text"]
    name, repo = facts["model_name"], facts["repo_id"]
    local = repo.rsplit("/", 1)[1]
    candidate, own = ctx["candidate"]["data"], ctx["own"]["data"]
    own_label = ctx["own"].get("label") or own["model"]["label"]
    lic = facts["licence"]
    front = ["---", f"license: {lic['spdx']}"]
    if lic["spdx"] == "other":
        front += [f"license_name: {lic['license_name']}", "license_link: LICENSING.md"]
    origin = facts["origin"]
    front.append(f"base_model: {origin['repo_id']}")
    if origin.get("relation") in ("finetune", "adapter", "merge"):
        front.append(f"base_model_relation: {origin['relation']}")
    front += [
        "tags:",
        "- decision-model",
        "- classification",
        "- system-one",
        "- safetensors",
        "---",
    ]
    paired = ctx["paired"]
    v3, own_v3 = candidate["v3"]["score"], own["v3"]["score"]
    public, own_public = candidate["panels"]["public231"], own["panels"]["public231"]
    summary = (
        f"On the same 8,147-item JevArena v3 panel, {name} scores **{v3:.2f}** versus "
        f"**{own_v3:.2f}** for {own_label}"
    )
    if paired:
        lo, hi = paired["ci95"]
        summary += (
            f" ({paired['delta']:+.2f}; paired 95% interval [{lo:+.2f}, {hi:+.2f}])"
        )
    summary += (
        f". On the separate 231 public JevBench questions it answers **{public['correct']}/231** "
        f"correctly versus **{own_public['correct']}/231**."
    )
    table, ordered = score_table(ctx["shown"])
    rank = next(i for i, e in enumerate(ordered, 1) if e["role"] == "candidate")
    summary += f" It ranks {rank} of {len(ordered)} models shown."
    regressions = tradeoff_rows(candidate, own)
    tradeoff_text = (
        "\n".join(
            [
                f"Results below {own_label} on this panel:",
                "",
                f"| Result | {name} | {own_label} |",
                "| --- | ---: | ---: |",
                *(
                    f"| {row} | {mine} | {theirs} |"
                    for row, mine, theirs in regressions
                ),
            ]
        )
        if regressions
        else f"No Choice, Noul, Score, transfer-task or public-tier result is below {own_label}."
    )
    limits = [
        *text.get("limitations", []),
        *auto_limits(candidate, facts["max_input_tokens"]),
    ]
    staging = text.get("staging_notice")
    links = [] if staging else [f"[Decision 2.0 collection]({COLLECTION_URL})"]
    links.append("[Download](#download-and-decide)")
    components = facts["parameters"]["components_text"]
    requirements = facts["requirements_text"]
    lines = [
        *front,
        "",
        f"![{name} owl banner](assets/{facts['banner']})",
        "",
        f"# {name}",
        "",
    ]
    if staging:
        lines += [f"> **{staging}**", ""]
    lines += [
        text["tagline"],
        "",
        " · ".join(links),
        "",
        "| Type | Use it for | Output |",
        "| --- | --- | --- |",
        "| **Choice** | Route a request or choose among 2–255 supplied options. | Selected ID + distribution |",
        "| **Noul** | Check a condition against the supplied evidence. | P(yes) |",
        "| **Score** | Apply 2–10 ordered rubric levels. | Expected level + distribution |",
        "",
        "Ask one or many named questions about one state; options and rubrics are supplied at request "
        "time. Answers are typed decisions with probabilities, not generated text.",
        "",
        "## Measured decisions",
        "",
        summary,
        "",
        table,
        "",
        "![JevArena v3 same-panel ranking](assets/jevarena-v3-rank.svg)",
        "",
        "![JevArena v3 model by task](assets/jevarena-v3-model-task.svg)",
        "",
        "![JevBench public 231 ranking](assets/jevbench-public231-rank.svg)",
        "",
        "Every model ran natively on the same frozen prompts with the same scorers; missing, invalid and "
        "over-budget answers count as failures. JevBench covers its 231 public questions only and is not "
        "the official closed-set rank. Ranks include only the models shown. "
        "[Methods and per-model results](evaluation/EVALUATION.md)",
        "",
        f"### Tradeoffs versus {own_label}",
        "",
        tradeoff_text,
        "",
        "## Download and decide",
        "",
        "```bash",
        f"hf download {repo} --local-dir {local}",
        "```",
        "",
        "```python",
        code_example(local).rstrip("\n"),
        "```",
        "",
        "The download includes a small local runtime (`decision2/`); it runs on CPU or one CUDA/ROCm GPU "
        f"and does not start a hosted endpoint. {requirements}",
        "",
        "## Model details",
        "",
        f"- **Architecture:** {text.get('architecture') or ARCHITECTURE[facts['profile']]}",
        f"- **Parameters:** {facts['parameters']['loaded']:,} loaded ({components}).",
        f"- **Direct weight origin:** [{origin['repo_id']}](https://huggingface.co/{origin['repo_id']}) at "
        f"`{origin['revision']}`. {origin['summary']}",
        f"- **Input limit:** {facts['max_input_tokens']:,} tokens for the complete state, question and candidates.",
        f"- **Calibration:** {facts['calibration_text']}",
        "- **Files:** the root `config.json` maps the model files; `MODEL_MANIFEST.json` lists the SHA-256 "
        "of every file and the runtime verifies them before loading.",
        "",
        "### Limits",
        "",
        *[f"- {item}" for item in limits],
        "",
        "[License](LICENSE)"
        + (" · [Component licences](LICENSING.md)" if lic["spdx"] == "other" else "")
        + " · [Notice](NOTICE) · [Attributions](ATTRIBUTIONS.md) · [Evaluation](evaluation/EVALUATION.md)",
        "",
    ]
    return "\n".join(lines)


def render_evaluation(ctx: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    shown, candidate = ctx["shown"], ctx["candidate"]["data"]
    rows = [
        "| Model | Loaded parameters | v3 | T | H | Choice / Noul / Score | Public 231 (easy/standard/hard) | Invalid typed / transfer / public |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    models = []
    for entry in sorted(shown, key=lambda e: -e["data"]["v3"]["score"]):
        report = entry["data"]
        label = entry.get("label") or report["model"]["label"]
        tiers = report["panels"]["public231"]["tiers"]
        invalid = "/".join(
            str((report.get("invalid", {}).get(p) or {}).get("invalid_or_missing", "—"))
            for p in ("typed-final", "css15", "public231")
        )
        rows.append(
            f"| {label} | {report['parameters']['loaded']:,} | {report['v3']['score']:.3f} | "
            f"{report['v3']['T']:.4f} | {report['v3']['H']:.4f} | "
            + " / ".join(
                f"{c}/{n}"
                for c, n in (_typed(report, k) for k in ("choice", "noul", "score"))
            )
            + f" | {report['panels']['public231']['correct']} ("
            + "/".join(
                str(tiers[t]["correct"])
                for t in ("easy", "standard", "hard")
                if t in tiers
            )
            + f") | {invalid} |"
        )
        models.append(
            {
                "label": label,
                "role": entry["role"],
                "repo_id": entry.get("repo_id"),
                "revision": report["model"].get("revision"),
                "family": report["model"].get("family"),
                "licence": entry["licence"],
                "report_sha256": entry["sha256"],
                "seal_sha256": report.get("seal_sha256"),
                "v3": report["v3"]["score"],
                "T": report["v3"]["T"],
                "H": report["v3"]["H"],
                "public231_correct": report["panels"]["public231"]["correct"],
                "loaded_parameters": report["parameters"]["loaded"],
            }
        )
    paired = ctx["paired"]
    paired_text = (
        f"The candidate-minus-{ctx['own'].get('label') or ctx['own']['data']['model']['label']} v3 "
        f"difference is {paired['delta']:+.3f} with paired 95% interval "
        f"[{paired['ci95'][0]:+.3f}, {paired['ci95'][1]:+.3f}]."
        if paired
        else "No paired interval was supplied."
    )
    text = "\n".join(
        [
            "# Evaluation",
            "",
            "**Scope: post-key same-panel.** JevArena v3 has 1,600 typed original items (2,000 answer "
            "slots) and 15 human-labeled transfer tasks (6,547 items). Its scalar is `100 × sqrt(T × H)`, "
            "where T is the four-family macro accuracy of typed decisions and H is the median task "
            "macro-F1 of human transfer. Missing, invalid and over-budget answers are failures in every "
            "denominator. The answers of this panel were available during development, so results are "
            "same-panel comparisons rather than an untouched blind test.",
            "",
            "JevBench public 231 is reported separately as raw accuracy by easy / standard / hard tier. "
            "It is an independent rerun of the public questions, not the upstream four-axis score or the "
            "official closed-set rank.",
            "",
            "Every model ran through its own native inference path on the same frozen prompts and was "
            "scored by the same scorers; each row reports the parameters its native loader instantiates. "
            "The paired interval is a joint bootstrap over typed groups (within family) and transfer "
            "tasks then items, 5,000 replicates. " + paired_text,
            "",
            "Comparators under non-commercial or research-only licences are not shown on this card.",
            "",
            *rows,
            "",
            "Report, panel and figure digests: [manifest.json](manifest.json).",
            "",
        ]
    )
    manifest = {
        "schema": "dev2-card-evaluation/1",
        "scope": "post-key same-panel",
        "panel_sha256": candidate["panel_sha256"],
        "scorer_sha256": {
            key: candidate["sources"].get(key)
            for key in (
                "benchmark/score.py",
                "transfer/score.py",
                "jev_arena/jevbench_public.py",
                "jev_arena/compare_v3.py",
            )
        },
        "models": models,
        "paired_vs_own_1_0": paired,
        "excluded_comparators": len(ctx["excluded"]),
    }
    return text, manifest


def build_card(
    *,
    entries: list[dict[str, Any]],
    roster: Path,
    paired: Path | None,
    facts: dict[str, Any],
    text: dict[str, Any],
    banner: Path,
    work: Path,
    output: Path,
) -> dict[str, Any]:
    """Write README.md, assets/ and evaluation/ into ``output``; return digests."""
    selection = select_reports(entries, roster)
    shown = selection["shown"]
    ctx = {
        "facts": facts,
        "text": text,
        "shown": shown,
        "excluded": selection["excluded"],
        "candidate": next(e for e in shown if e["role"] == "candidate"),
        "own": next(e for e in shown if e["role"] == "own-1.0"),
        "paired": _paired(paired),
    }
    if len(shown) < 2:
        raise ValueError(
            "A card needs the candidate and at least one eligible comparator"
        )
    figures = render_charts(shown, work, output / "assets")
    banner_target = output / "assets" / facts["banner"]
    shutil.copyfile(banner, banner_target)
    readme = render_readme(ctx)
    evaluation, manifest = render_evaluation(ctx)
    manifest["figures_sha256"] = figures
    manifest["banner_sha256"] = sha_file(banner_target)
    for content, label in ((readme, "README"), (evaluation, "EVALUATION")):
        problems = lint(content)
        if problems:
            raise ValueError(f"{label} failed the card content lint: {problems}")
    (output / "README.md").write_text(readme, encoding="utf-8")
    (output / "evaluation").mkdir(parents=True, exist_ok=True)
    (output / "evaluation/EVALUATION.md").write_text(evaluation, encoding="utf-8")
    (output / "evaluation/manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "readme_sha256": sha_file(output / "README.md"),
        "figures_sha256": figures,
        "shown": [
            {"key": e["key"], "licence": e["licence"], "report_sha256": e["sha256"]}
            for e in shown
        ],
        "excluded": [
            {"key": e["key"], "reason": e["reason"]} for e in selection["excluded"]
        ],
        "tradeoffs": tradeoffs(ctx["candidate"]["data"], ctx["own"]["data"]),
        "paired": ctx["paired"],
    }


def check_rendered(readme: str, files: set[str]) -> list[str]:
    """Structural card checks on a rendered README against the repository file list."""
    problems = lint(readme)
    if not readme.startswith("---\n") or "\n---\n" not in readme[4:]:
        problems.append("missing YAML front matter")
    for link in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", readme):
        if not link.startswith("http") and link not in files:
            problems.append(f"image does not resolve: {link}")
    for link in re.findall(r"(?<!!)\[[^\]]*\]\(([^)#]+)\)", readme):
        if not link.startswith("http") and link not in files:
            problems.append(f"link does not resolve: {link}")
    for required in (
        "## Measured decisions",
        "## Download and decide",
        "## Model details",
        "```python",
    ):
        if required not in readme:
            problems.append(f"missing section: {required}")
    for chart in CHART_FILES:
        if f"]({chart})" not in readme:
            problems.append(f"missing chart: {chart}")
    if not math.isfinite(len(readme)):
        problems.append("invalid README")
    return problems
