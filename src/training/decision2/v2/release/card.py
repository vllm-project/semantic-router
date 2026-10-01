"""Model card of a Decision 2.0 release, from same-panel reports only.

README layout (a standard model-release card): YAML metadata; title, one
paragraph and a link row; highlights; a model overview table; evaluation (a
JevArena bar chart, a small JevBench public-231 chart and one compact table of
this model, its Decision 1.0 counterpart and the same-size peers, with one
footnote line); a Transformers quickstart; short limitations; training data with
the licence attributions; licence; citation. Methods, per-task results, paired
intervals, calibration, comparator notes and every result below the counterpart
go to ``evaluation/EVALUATION.md``. Internal release facts (gate items,
decision and weights hashes, runtime notes) stay in the release records.

Every comparator passes the licence filter. At a size without a Decision 1.0
model (``facts["comparison"] == "no-1.0"``) the counterpart slot is the release
gate's reference peer (role ``reference``).
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

from v2.release import card_charts
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
# The README carries none of the internal evaluation vocabulary (EVALUATION.md may).
README_FORBIDDEN = (
    (re.compile(r"post-key", re.I), "post-key wording"),
    (re.compile(r"\bBrier\b|\bECE\b"), "calibration metrics"),
    (re.compile(r"mlx-diag", re.I), "development diagnostic"),
    (re.compile(r"Download and decide|from decision2 import"), "native runtime usage"),
    (re.compile(r"\bowl\b", re.I), "banner artwork"),
    (re.compile(r"\b[0-9a-f]{40}\b"), "revision hash"),
)
TEXT_KEYS = {
    "description",
    "model_type",
    "base_model",
    "precision",
    "limitations",
    "training_summary",
    "comparator_note",
    "c1_result",
    "transformers_note",
    "staging_notice",
}
CITATION_YEAR = 2026
AUTHOR = "vLLM Semantic Router Team"
NO_OWN_1_0 = "no-1.0"
NO_OWN_1_0_TEXT = "There is no Decision 1.0 model at this size."
PUBLIC231_NOTE = (
    "JevBench public 231 is an independent rerun of the 231 public questions (about a third of the "
    "official Intelligence inputs), not the official JevBench score; its easy tier is at ceiling, and "
    "totals within about 10 items are not distinguishable."
)
# Public-231 totals closer than this are not distinguishable (PUBLIC231_NOTE).
PUBLIC_MARGIN = 10
# A same-size peer's human-labelled transfer lead below this (median macro-F1) is not listed.
TRANSFER_MARGIN = 0.02
MAX_LIMITATIONS = 5
MLX_SCHEMA = "dev2-mlx-diag-score/1"
# mlx-diag Score is built from XNLI (CC BY-NC 4.0, internal use only); EVALUATION.md shows Choice and Noul.
MLX_CARD_TYPES = ("choice", "noul")
MLX_NAMES = {"choice": "Choice", "noul": "Noul"}
DEFAULT_PRECISION = "BF16 backbone compute, FP32 decision head"
TRANSFORMERS_HEADING = "### Use with 🤗 Transformers"
CHART_RANK, CHART_PUBLIC = CHART_FILES


def lint(text: str) -> list[str]:
    return sorted({reason for pattern, reason in FORBIDDEN if pattern.search(text)})


def lint_readme(text: str) -> list[str]:
    return sorted(
        set(lint(text))
        | {reason for pattern, reason in README_FORBIDDEN if pattern.search(text)}
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


def _public(entry: dict[str, Any]) -> int:
    return entry["data"]["panels"]["public231"]["correct"]


def render_charts(
    shown: list[dict[str, Any]], work: Path, assets: Path
) -> dict[str, str]:
    """The JevArena and JevBench public-231 bar charts; their receipt stays private."""
    from v2.eval.charts import DEFAULT_LICENCES, licence_class

    model_ids = {}
    for entry in shown:
        model = dict(entry["data"]["model"])
        if entry["role"] == "candidate":
            continue
        if entry.get("repo_id") and not model.get("model_id"):
            # Some adopted reports name no model id; the licence lookup needs one.
            model_ids[model["label"]] = entry["repo_id"]
            model["model_id"] = entry["repo_id"]
        if licence_class({"model": model}) not in DEFAULT_LICENCES:
            raise ValueError(f"{_label(entry)}: licence class not shown on cards")
    rows = [{"label": _label(e), "highlight": e["role"] == "candidate"} for e in shown]
    scores = [_score(e) for e in shown]
    figures_svg = {
        CHART_RANK: card_charts.bar_chart(
            title="JevArena",
            subtitle="Typed decisions and 15 human-labelled transfer tasks, same frozen panel for every model (higher is better)",
            rows=[{**r, "value": v} for r, v in zip(rows, scores)],
            maximum=card_charts.axis_maximum(scores, 10, 50),
            step=10,
            digits=2,
        ),
        CHART_PUBLIC: card_charts.bar_chart(
            title="JevBench (public 231)",
            subtitle="Correct answers out of 231 public questions (higher is better)",
            rows=[{**r, "value": float(_public(e))} for r, e in zip(rows, shown)],
            maximum=231,
            step=50,
            digits=0,
        ),
    }
    assets.mkdir(parents=True, exist_ok=True)
    figures = {}
    for relative, svg in figures_svg.items():
        if lint(svg):
            raise ValueError(f"{relative} failed the card content lint")
        target = assets.parent / relative
        target.write_text(svg, encoding="utf-8")
        figures[relative] = sha_file(target)
    work.mkdir(parents=True, exist_ok=True)
    (work / "charts-receipt.json").write_text(
        json.dumps(
            {
                "schema": "dev2-card-charts/2",
                "reports": {e["key"]: e["sha256"] for e in shown},
                "model_id_overrides": model_ids,
                "values": {
                    _label(e): {"jevarena": _score(e), "public231": _public(e)}
                    for e in shown
                },
                "figures": figures,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return figures


def _pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def _f1(value: float) -> str:
    return f"{100 * value:.1f}"


def _typed(report: dict[str, Any], kind: str) -> tuple[int, int]:
    item = report["panels"]["typed-final"]["by_type"][kind]
    return item["correct"], item["n"]


def loaded_parameters(entry: dict[str, Any], candidate_loaded: int | None) -> int:
    """The packaged candidate shows the count its runtime asserts; others their reports'."""
    if entry["role"] == "candidate" and candidate_loaded:
        return candidate_loaded
    return entry["data"]["parameters"]["loaded"]


def _tiers(public: dict[str, Any]) -> str:
    tiers = public["tiers"]
    return " / ".join(
        str(tiers[t]["correct"]) for t in ("easy", "standard", "hard") if t in tiers
    )


def _mlx_cell(entry: dict[str, Any]) -> str:
    mlx = entry.get("mlx")
    if not mlx:
        return "—"
    return " / ".join(
        f"{100 * mlx['by_type'][kind]['non_english_mean_accuracy']:.1f}"
        for kind in MLX_CARD_TYPES
    )


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


def transformers_example(repo: str) -> str:
    state, questions = _example_arguments()
    return (
        "import json\n\n"
        "from transformers import AutoModel\n\n"
        f'model = AutoModel.from_pretrained("{repo}", trust_remote_code=True)\n'
        "result = model.system_one(\n"
        f"    state={state},\n"
        f"    questions={questions},\n"
        ")\n"
        'print(json.dumps(result["answers"], indent=2))\n'
    )


def pip_line(facts: dict[str, Any]) -> str:
    requirements = facts.get("runtime_requirements") or {}
    packages = ['"transformers>=5.17"', "torch", "safetensors"]
    if "peft" in requirements:
        packages.append("peft")
    return "pip install " + " ".join(packages)


def transformers_note(facts: dict[str, Any], text: dict[str, Any]) -> str:
    repo = facts["repo_id"]
    remote = facts.get("remote_code") or {}
    requirements = facts.get("runtime_requirements") or {}
    note = (
        f'`pipeline("decision", model="{repo}", trust_remote_code=True)` accepts '
        '`{"state": ..., "questions": {...}}` and returns the same answers.'
    )
    if "flash-linear-attention" in requirements:
        note += (
            ' The model loads on `cuda:0`; pass `device_map="cuda:1"` to choose another GPU. Install '
            "`flash-linear-attention` and `causal-conv1d` for the kernels the evaluation used; without them "
            "Transformers runs slower reference code whose probabilities differ in the last digits. CPU "
            "inference was not verified."
        )
    else:
        note += (
            " The model loads on `cuda:0` when a GPU is visible and on the CPU otherwise; pass "
            '`device_map="cpu"` or `device_map="cuda:1"` to choose.'
        )
    base = remote.get("base")
    if base:
        note += (
            f" On first load it downloads the pinned base [{base['repo_id']}]"
            f"(https://huggingface.co/{base['repo_id']}) and checks every file's SHA-256."
        )
    tested = remote.get("tested")
    if tested:
        note += f" Tested with Transformers {' and '.join(tested)}."
    if text.get("transformers_note"):
        note += f" {text['transformers_note']}"
    return note


def _significance(paired: dict[str, Any] | None) -> str | None:
    if not paired:
        return None
    low, high = paired["ci95"]
    return "above" if low > 0 else "below" if high < 0 else "level"


def highlights(ctx: dict[str, Any]) -> list[str]:
    shown = ctx["shown"]
    candidate, own = ctx["candidate"], ctx["own"]
    score = _score(candidate)
    no_own = ctx.get("comparison") == NO_OWN_1_0
    lines = []
    if own is None:
        lines.append(f"**JevArena {score:.2f}.** {NO_OWN_1_0_TEXT}")
    else:
        other, delta = _label(own), score - _score(own)
        strongest = _score(own) >= max(_score(e) for e in shown if e is not candidate)
        role = (
            "its Decision 1.0 counterpart"
            if not no_own
            else (
                "the strongest other same-size model"
                if strongest
                else "the reference same-size model"
            )
        )
        paired, state = ctx["paired"], _significance(ctx["paired"])
        interval = (
            f"paired 95% CI {paired['ci95'][0]:+.2f} to {paired['ci95'][1]:+.2f}"
            if paired
            else None
        )
        if state == "above":
            text = f"{delta:+.2f} over {role}, {other} ({interval})"
        elif state == "below":
            text = f"{delta:+.2f} below {role}, {other} ({interval})"
        elif state == "level":
            text = f"level with {role}, {other} ({_score(own):.2f}; difference {delta:+.2f}, {interval})"
        else:
            text = f"versus {_score(own):.2f} for {role}, {other} ({delta:+.2f})"
        lines.append(
            f"**JevArena {score:.2f}**, {text}."
            + (f" {NO_OWN_1_0_TEXT}" if no_own else "")
        )
    ordered = sorted(shown, key=lambda e: -_score(e))
    rank = ordered.index(candidate) + 1
    above = [
        _label(e)
        for e in ordered
        if e is not candidate
        and e is not own
        and _significance(ctx["paired_peers"].get(e["key"])) == "above"
    ]
    standing = (
        f"Highest JevArena score of the {len(ordered)} same-size models compared"
        if rank == 1
        else f"Ranks {rank} of {len(ordered)} on JevArena among the same-size models compared"
    )
    if above:
        standing += f"; significantly above {_join(above)} (paired 95% CIs)"
    lines.append(standing + ".")
    lines.append(
        'Runs with stock 🤗 Transformers through `AutoModel` or `pipeline("decision")` with '
        "`trust_remote_code=True`."
    )
    return lines


def _join(items: list[str]) -> str:
    if len(items) <= 2:
        return " and ".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def weaker_than_counterpart(ctx: dict[str, Any]) -> str | None:
    own = ctx["own"]
    if own is None:
        return None
    candidate = ctx["candidate"]
    c, o = candidate["data"], own["data"]
    parts = []
    typed = []
    for kind in ("choice", "noul", "score"):
        (mine, n), (theirs, _) = _typed(c, kind), _typed(o, kind)
        if mine < theirs:
            typed.append(f"{kind.title()} {mine} vs {theirs} of {n}")
    if typed:
        parts.append(f"typed decisions ({'; '.join(typed)})")
    if c["v3"]["H"] < o["v3"]["H"]:
        parts.append(
            f"human-labelled transfer overall ({_f1(c['v3']['H'])} vs {_f1(o['v3']['H'])})"
        )
    tasks_c, tasks_o = c["panels"]["css15"]["tasks"], o["panels"]["css15"]["tasks"]
    lower = sorted(
        (t for t in tasks_c if tasks_c[t]["macro_f1"] < tasks_o[t]["macro_f1"]),
        key=lambda t: tasks_c[t]["macro_f1"] - tasks_o[t]["macro_f1"],
    )
    if lower:
        largest = ", ".join(
            f"{_task(t)} {_f1(tasks_c[t]['macro_f1'])} vs {_f1(tasks_o[t]['macro_f1'])}"
            for t in lower[:2]
        )
        parts.append(
            f"{len(lower)} of {len(tasks_c)} human-labelled transfer tasks (largest gaps, macro-F1: {largest})"
        )
    if _public(candidate) < _public(own):
        parts.append(f"JevBench public 231 ({_public(candidate)} vs {_public(own)})")
    cm, om = candidate.get("mlx"), own.get("mlx")
    if cm and om:
        below = [
            (MLX_NAMES[kind], mine, theirs)
            for kind in MLX_CARD_TYPES
            for mine, theirs in [
                (
                    cm["by_type"][kind]["non_english_mean_accuracy"],
                    om["by_type"][kind]["non_english_mean_accuracy"],
                )
            ]
            if mine < theirs
        ]
        if below:
            parts.append(
                f"non-English {_join([k for k, _, _ in below])} on a multilingual diagnostic ("
                + "; ".join(f"{_pct(m)} vs {_pct(t)}" for _, m, t in below)
                + ")"
            )
    if not parts:
        return None
    return (
        f"**Below {_label(own)} in places:** {_join(parts)}. "
        "Every such result is listed in [EVALUATION.md](evaluation/EVALUATION.md)."
    )


def weaker_than_peers(ctx: dict[str, Any]) -> str | None:
    candidate, own = ctx["candidate"], ctx["own"]
    parts = []
    for entry in sorted(ctx["shown"], key=lambda e: -_score(e)):
        if entry is candidate or entry is own:
            continue
        gaps = []
        if _score(entry) > _score(candidate):
            gaps.append(f"JevArena ({_score(candidate):.2f} vs {_score(entry):.2f})")
        if _transfer(entry) - _transfer(candidate) >= TRANSFER_MARGIN:
            gaps.append(
                f"human-labelled transfer ({_f1(_transfer(candidate))} vs {_f1(_transfer(entry))})"
            )
        if _public(entry) - _public(candidate) > PUBLIC_MARGIN:
            gaps.append(
                f"JevBench public 231 ({_public(candidate)} vs {_public(entry)})"
            )
        if gaps:
            parts.append(f"{_label(entry)} on {_join(gaps)}")
    if not parts:
        return None
    return f"**Against other same-size models:** trails {'; '.join(parts)}."


def limitations(ctx: dict[str, Any]) -> list[str]:
    facts, text = ctx["facts"], ctx["text"]
    candidate = ctx["candidate"]["data"]
    lines = [
        line for line in (weaker_than_counterpart(ctx), weaker_than_peers(ctx)) if line
    ]
    lines += text.get("limitations", [])
    screen = candidate["slices"].get("language_screen") or {}
    total = sum(sum(counts.values()) for counts in screen.values())
    english = sum(counts.get("en", 0) for counts in screen.values())
    scope = f"Complete inputs above {facts['max_input_tokens']:,} tokens are rejected, never truncated."
    if not total or english / total >= 0.95:
        scope += " The evaluation panels are almost entirely English, so other languages are less well measured."
    trust = (
        "It decides only from the input it is given and does not retrieve missing facts; "
        "probabilities are estimates, so review consequential decisions against the evidence."
    )
    if len(lines) + 2 <= MAX_LIMITATIONS:
        lines += [scope, trust]
    else:
        lines.append(f"{scope} {trust}")
    if len(lines) > MAX_LIMITATIONS:
        raise ValueError(
            f"The card allows at most {MAX_LIMITATIONS} limitations; shorten the spec's own lines"
        )
    return lines


CC = re.compile(r"CC BY(?:-SA)?(?: [0-9.]+)?(?![-\w])")


def _source_name(segment: str) -> str:
    """The dataset name before a citation parenthesis, e.g. ', and the labels of KLUE STS, MRC and YNAT '."""
    name = re.sub(r"^[\s,;.]*(?:(?:and|plus)\s+)?", "", segment).strip()
    for separator in (": ", ". "):
        name = name.rsplit(separator, 1)[-1]
    name = name.rsplit(" of ", 1)[-1]
    return re.sub(r"^(?:the|and|plus)\s+", "", name).strip(" ,.")


def _expand(name: str) -> list[str]:
    """'KLUE YNAT, MRC and STS' -> KLUE YNAT, KLUE MRC, KLUE STS (one shared prefix word)."""
    parts = [p.strip() for p in re.split(r",\s*|\s+and\s+", name) if p.strip()]
    words = parts[0].split()
    if len(parts) == 1 or len(words) < 2:
        return parts
    return [parts[0]] + [p if " " in p else f"{words[0]} {p}" for p in parts[1:]]


def cc_sources(attributions: list[str]) -> dict[str, list[str]]:
    """CC BY / CC BY-SA training sources named in the spec's attributions, grouped by licence."""
    groups: dict[str, list[str]] = {}
    for body in attributions:
        if not body.startswith("Training data"):
            continue
        start = 0
        for match in re.finditer(r"\(([^()]*)\)", body):
            name = _source_name(body[start : match.start()])
            start = match.end()
            terms = match.group(1).split(";")[-1]
            for fragment in terms.split(","):
                fragment = fragment.strip()
                found = CC.search(fragment)
                if not found or not name or not name[0].isupper():
                    continue
                key = found.group(0)
                if fragment.lower().startswith("wikipedia"):
                    key = f"Wikipedia text under {key}"
                for item in _expand(name):
                    if item not in groups.setdefault(key, []):
                        groups[key].append(item)
    return {k: sorted(v, key=str.lower) for k, v in sorted(groups.items())}


def base_notice(attributions: list[str], files: set[str] | None = None) -> list[str]:
    """Qwen (or other upstream) base-model licence lines from the spec's attributions."""
    lines = []
    for entry in attributions:
        found = re.match(r"\[([^\]]+)\]\((https://huggingface\.co/[^)]+)\)", entry)
        if not found or "/Qwen/" not in found.group(2):
            continue
        licence_file = re.search(r"`(LICENSES/[^`]+)`", entry)
        terms = re.search(r"\((Apache-2\.0|[^,;()]*Licen[cs]e[^,;()]*)", entry)
        line = f"[{found.group(1)}]({found.group(2)})"
        if terms:
            line += f" ({terms.group(1)}"
            if licence_file and (files is None or licence_file.group(1) in files):
                line += f"; [licence text]({licence_file.group(1)})"
            line += ")"
        lines.append(line)
    return lines


def training_section(
    facts: dict[str, Any], text: dict[str, Any], files: set[str]
) -> list[str]:
    attributions = facts.get("attributions") or []
    lines = [text["training_summary"], ""]
    base = base_notice(attributions, files)
    if base:
        lines += [
            f"**Base model licence.** Built on {_join(base)}; the upstream licence terms and "
            "copyright notices are kept with this repository.",
            "",
        ]
    groups = cc_sources(attributions)
    if groups:
        lines += [
            "**Attribution.** The training data include material under Creative Commons licences: "
            + "; ".join(f"{k}: {', '.join(v)}" for k, v in groups.items())
            + ". These datasets are not redistributed here; full credits, citations and licences are in "
            "[ATTRIBUTIONS.md](ATTRIBUTIONS.md).",
            "",
        ]
    else:
        lines += [
            "Full data credits and licences: [ATTRIBUTIONS.md](ATTRIBUTIONS.md).",
            "",
        ]
    return lines


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


def overview(facts: dict[str, Any], text: dict[str, Any]) -> list[str]:
    origin = facts["origin"]
    base = text.get("base_model") or (
        f"[{origin['repo_id'].rsplit('/', 1)[-1]}](https://huggingface.co/{origin['repo_id']})"
    )
    loaded = facts["parameters"]["loaded"]
    parameters = f"{loaded / 1e9:.2f}B ({loaded:,})"
    if facts.get("name_basis") == "base":
        named = facts["name_base_model"]
        parameters += f"; named after its base model, [{named.rsplit('/', 1)[-1]}](https://huggingface.co/{named})"
    lic = facts["licence"]
    licence_cell = (
        "[Apache-2.0](LICENSE)"
        if lic["spdx"] == "apache-2.0"
        else f"[{lic['spdx']}](LICENSE)"
        + (" ([components](LICENSING.md))" if lic["spdx"] == "other" else "")
    )
    return [
        "| | |",
        "| --- | --- |",
        f"| **Model type** | {text['model_type']} |",
        f"| **Base model** | {base} |",
        f"| **Parameters** | {parameters} |",
        f"| **Context length** | {facts['max_input_tokens']:,} tokens (state, questions and options together) |",
        "| **Decision types** | Choice (2–255 options), Noul (yes/no), Score (2–10 ordered levels) |",
        f"| **Precision** | {text.get('precision') or DEFAULT_PRECISION} |",
        f"| **License** | {licence_cell} |",
    ]


def results_table(ctx: dict[str, Any]) -> list[str]:
    candidate_loaded = ctx["facts"]["parameters"]["loaded"]
    rows = [
        "| Model | Parameters | JevArena ↑ | Human-labelled transfer ↑ | JevBench (public 231) ↑ |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for entry in sorted(ctx["shown"], key=lambda e: -_score(e)):
        cells = [
            _label(entry),
            f"{loaded_parameters(entry, candidate_loaded) / 1e9:.2f}B",
            f"{_score(entry):.2f}",
            _f1(_transfer(entry)),
            f"{_public(entry)}",
        ]
        if entry["role"] == "candidate":
            cells = [f"**{c}**" for c in cells]
        rows.append("| " + " | ".join(cells) + " |")
    return rows


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


def render_readme(ctx: dict[str, Any], files: set[str]) -> str:
    facts, text = ctx["facts"], ctx["text"]
    name, repo = facts["model_name"], facts["repo_id"]
    staging = text.get("staging_notice")
    lic = facts["licence"]
    links = [] if staging else [f"[Decision 2.0 collection]({COLLECTION_URL})"]
    links.append("[Evaluation details](evaluation/EVALUATION.md)")
    description = text.get("description") or (
        f"{name} is a Decision 2.0 model. It answers structured decision questions about one input "
        "(**Choice** among named options, **Noul** yes/no checks and **Score** ratings on ordered levels) and "
        "returns a probability for every answer, all in one forward pass and without generating text. Options "
        "and rubrics are supplied with each request."
    )
    lines = [*_front_matter(facts), "", f"# {name}", ""]
    if staging:
        lines += [f"> **{staging}**", ""]
    lines += [description, "", " · ".join(links), "", "## Highlights", ""]
    lines += [f"- {item}" for item in highlights(ctx)]
    lines += ["", "## Model overview", "", *overview(facts, text), ""]
    lines += [
        "## Evaluation",
        "",
        f"![JevArena scores of {name} and same-size models]({CHART_RANK})",
        "",
        f"![JevBench public 231 scores of {name} and same-size models]({CHART_PUBLIC})",
        "",
        *results_table(ctx),
        "",
        "<sub>Every model ran on the same frozen prompts with the same scorers; missing or invalid answers "
        "count as errors. JevArena combines typed-decision accuracy with the median macro-F1 of 15 "
        "human-labelled transfer tasks (shown ×100). Methods, per-task results and comparator notes: "
        "[EVALUATION.md](evaluation/EVALUATION.md).</sub>",
        "",
        "## Quickstart",
        "",
    ]
    lines += [
        TRANSFORMERS_HEADING,
        "",
        "```bash",
        pip_line(facts),
        "```",
        "",
        "```python",
        transformers_example(repo).rstrip("\n"),
        "```",
        "",
        transformers_note(facts, text),
        "",
        "## Limitations",
        "",
    ]
    lines += [f"- {item}" for item in limitations(ctx)]
    lines += ["", "## Training data", "", *training_section(facts, text, files)]
    lines += [
        "## License",
        "",
        (
            "Apache-2.0 ([LICENSE](LICENSE))."
            if lic["spdx"] == "apache-2.0"
            else f"`{lic['spdx']}` ([LICENSE](LICENSE); per-component terms in [LICENSING.md](LICENSING.md))."
        )
        + " Third-party notices: [NOTICE](NOTICE) and [ATTRIBUTIONS.md](ATTRIBUTIONS.md).",
        "",
        "## Citation",
        "",
        *citation(facts),
        "",
    ]
    return "\n".join(lines)


def render_evaluation(ctx: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    shown, candidate = ctx["shown"], ctx["candidate"]["data"]
    name = ctx["facts"]["model_name"]
    candidate_loaded = ctx["facts"]["parameters"]["loaded"]
    mlx = any(e.get("mlx") for e in shown)
    ordered = sorted(shown, key=lambda e: -_score(e))
    rows = [
        "| Model | Loaded parameters | JevArena | Typed decisions | Human-labelled transfer | Choice / Noul / Score | "
        "JevBench public 231 (easy / standard / hard) | Typed Brier / ECE | "
        + ("mlx-diag non-English Choice / Noul | " if mlx else "")
        + "Invalid typed / transfer / public |",
        "| --- | ---: | ---: | ---: | ---: | --- | --- | --- | "
        + ("--- | " if mlx else "")
        + "--- |",
    ]
    models = []
    for entry in ordered:
        report = entry["data"]
        label = _label(entry)
        final = report["panels"]["typed-final"]
        loaded = loaded_parameters(entry, candidate_loaded)
        invalid = " / ".join(
            str((report.get("invalid", {}).get(p) or {}).get("invalid_or_missing", "—"))
            for p in ("typed-final", "css15", "public231")
        )
        rows.append(
            f"| {label} | {loaded:,} | {_score(entry):.2f} | "
            f"{_f1(report['v3']['T'])} | {_f1(report['v3']['H'])} | "
            + " / ".join(
                f"{c}/{n}"
                for c, n in (_typed(report, k) for k in ("choice", "noul", "score"))
            )
            + f" | {_public(entry)} ({_tiers(report['panels']['public231'])}) | "
            f"{final['brier']:.3f} / {final['ece_10']:.3f} | "
            + (f"{_mlx_cell(entry)} | " if mlx else "")
            + f"{invalid} |"
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
                "jevarena": _score(entry),
                "typed": report["v3"]["T"],
                "transfer": report["v3"]["H"],
                "public231_correct": _public(entry),
                "typed_brier": final["brier"],
                "typed_ece_10": final["ece_10"],
                "loaded_parameters": loaded,
                "report_loaded_parameters": report["parameters"]["loaded"],
                "mlx_diag_sha256": entry.get("mlx_sha256"),
                "mlx_diag_non_english": (
                    {
                        kind: entry["mlx"]["by_type"][kind]["non_english_mean_accuracy"]
                        for kind in MLX_CARD_TYPES
                    }
                    if entry.get("mlx")
                    else None
                ),
            }
        )
    tasks = list(candidate["panels"]["css15"]["tasks"])
    per_task = [
        "| Task | " + " | ".join(_label(e) for e in ordered) + " |",
        "| --- | " + " | ".join("---:" for _ in ordered) + " |",
    ]
    for kind in ("choice", "noul", "score"):
        per_task.append(
            f"| Typed {kind.title()} (accuracy) | "
            + " | ".join(
                _f1(e["data"]["panels"]["typed-final"]["by_type"][kind]["accuracy"])
                for e in ordered
            )
            + " |"
        )
    for task in tasks:
        per_task.append(
            f"| {_task(task)} (macro-F1) | "
            + " | ".join(
                _f1(e["data"]["panels"]["css15"]["tasks"][task]["macro_f1"])
                for e in ordered
            )
            + " |"
        )
    reported = candidate["parameters"]["loaded"]
    parameter_note = (
        f"The {name} row shows the {candidate_loaded:,} parameters its packaged runtime loads and "
        f"asserts; its same-panel report counted {reported:,} from the backbone safetensors only."
        if reported != candidate_loaded
        else ""
    )
    mlx_text = (
        "**mlx-diag** (a development diagnostic, not a release score: 2,275 prompts in seven languages, "
        "English instructions over target-language states): Choice comes from the MASSIVE 1.1 test split "
        "(CC BY 4.0) and Noul from the PAWS-X test split; the column shows mean non-English accuracy. Its "
        "Score part is built from XNLI (CC BY-NC 4.0) and is not shown. Public test splits may appear in "
        "backbone pretraining data, so this is not a sealed test."
        if mlx
        else ""
    )
    paired = ctx["paired"]
    no_own = ctx.get("comparison") == NO_OWN_1_0
    own = ctx["own"]
    paired_text = (
        NO_OWN_1_0_TEXT
        if own is None
        else (f"{NO_OWN_1_0_TEXT} " if no_own else "")
        + (
            f"The {name} minus {_label(own)} JevArena difference is {paired['delta']:+.2f} "
            f"(paired 95% interval [{paired['ci95'][0]:+.2f}, {paired['ci95'][1]:+.2f}])."
            if paired
            else "No paired interval was supplied."
        )
    )
    peer_pairs = [
        (_label(e), ctx["paired_peers"][e["key"]])
        for e in shown
        if e["key"] in ctx["paired_peers"] and e is not own
    ]
    if peer_pairs:
        paired_text += (
            " Against the other models shown: "
            + "; ".join(
                f"{label} {value['delta']:+.2f} [{value['ci95'][0]:+.2f}, {value['ci95'][1]:+.2f}]"
                for label, value in peer_pairs
            )
            + "."
        )
    below = (
        tradeoff_rows(
            candidate, own["data"], ctx["candidate"].get("mlx"), own.get("mlx")
        )
        if own
        else []
    )
    below_lines = (
        [
            f"## Results below {_label(own)}",
            "",
            *(
                [
                    f"| Result | {name} | {_label(own)} |",
                    "| --- | ---: | ---: |",
                    *(f"| {row} | {mine} | {theirs} |" for row, mine, theirs in below),
                ]
                if below
                else [
                    f"No typed, transfer, public or diagnostic result is below {_label(own)}."
                ]
            ),
            "",
        ]
        if own
        else []
    )
    text = ctx["text"]
    lines = [
        f"# {name}: evaluation",
        "",
        "## Method",
        "",
        "**JevArena** scores 1,600 typed decision items (Choice, Noul and Score; 2,000 answer slots) and 15 "
        "human-labelled transfer tasks (6,547 items) as `100 × sqrt(typed × transfer)`: *typed* is the "
        "macro accuracy over the four typed families and *transfer* the median macro-F1 over the 15 tasks. "
        "Missing, invalid and over-budget answers count as errors in every denominator. The panel's answers "
        "were available during development (post-key), so the results are same-panel comparisons, not an "
        "untouched blind test.",
        "",
        PUBLIC231_NOTE,
        "",
        "Every model ran through its own native inference path on the same frozen prompts and was scored "
        "by the same scorers; each row reports the parameters its native loader instantiates. "
        + (parameter_note + " " if parameter_note else "")
        + "Typed Brier and ECE (10 bins) measure calibration on the typed decisions. Paired intervals "
        "come from a joint bootstrap over typed groups (within family) and transfer tasks then items, "
        "5,000 replicates. " + paired_text,
        "",
        *([text["c1_result"], ""] if text.get("c1_result") else []),
        *([mlx_text, ""] if mlx_text else []),
        "Comparators under non-commercial, research-only or unknown licences are not shown."
        + (f" {text['comparator_note']}" if text.get("comparator_note") else ""),
        "",
        "## Results",
        "",
        *rows,
        "",
        "Typed decisions and human-labelled transfer are shown ×100.",
        "",
        "## Per-task results",
        "",
        *per_task,
        "",
        *below_lines,
        "Report, panel and figure digests: [manifest.json](manifest.json).",
        "",
    ]
    manifest = {
        "schema": "dev2-card-evaluation/2",
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
        **(
            {
                "decision_1_0": None,
                "paired_vs_reference": paired if own else None,
            }
            if no_own
            else {"paired_vs_own_1_0": paired}
        ),
        **(
            {"paired_vs_peers": {label: value for label, value in peer_pairs}}
            if peer_pairs
            else {}
        ),
        "excluded_comparators": len(ctx["excluded"]),
    }
    return "\n".join(lines), manifest


def build_card(
    *,
    entries: list[dict[str, Any]],
    roster: Path,
    paired: Path | None,
    facts: dict[str, Any],
    text: dict[str, Any],
    work: Path,
    output: Path,
    paired_peers: dict[str, Path] | None = None,
    files: set[str] | None = None,
) -> dict[str, Any]:
    """Write README.md, assets/ and evaluation/ into ``output``; return digests.

    ``paired_peers`` (report key -> candidate-minus-peer paired file) adds those
    intervals to the evaluation page for peers the licence filter shows. ``files``
    lists the package files the README may link to (default: those under ``output``).
    """
    unknown = set(text) - TEXT_KEYS
    if unknown:
        raise ValueError(f"Card text keys not used by this card: {sorted(unknown)}")
    for key in ("model_type", "training_summary"):
        if not text.get(key):
            raise ValueError(f"The card needs text.{key}")
    if facts.get("remote_code") is None:
        raise ValueError("The card's quickstart needs the Transformers remote code")
    comparison = facts.get("comparison", "own-1.0")
    selection = select_reports(entries, roster, comparison)
    shown = selection["shown"]
    slot = "reference" if comparison == NO_OWN_1_0 else "own-1.0"
    ctx = {
        "facts": facts,
        "text": text,
        "shown": shown,
        "excluded": selection["excluded"],
        "comparison": comparison,
        "candidate": next(e for e in shown if e["role"] == "candidate"),
        "own": next((e for e in shown if e["role"] == slot), None),
        "paired": _paired(paired),
        "paired_peers": _paired_peers(shown, paired_peers or {}),
    }
    if len(shown) < 2:
        raise ValueError(
            "A card needs the candidate and at least one eligible comparator"
        )
    figures = render_charts(shown, work, output / "assets")
    present = (
        files
        if files is not None
        else {
            p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()
        }
    )
    readme = render_readme(ctx, present)
    evaluation, manifest = render_evaluation(ctx)
    manifest["figures_sha256"] = figures
    for content, label, check in (
        (readme, "README", lint_readme),
        (evaluation, "EVALUATION", lint),
    ):
        problems = check(content)
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
    "## Model overview",
    "## Evaluation",
    "## Quickstart",
    "## Limitations",
    "## Training data",
    "## License",
    "## Citation",
)


def check_rendered(readme: str, files: set[str]) -> list[str]:
    """Structural card checks on a rendered README against the repository file list."""
    problems = lint_readme(readme)
    if not readme.startswith("---\n") or "\n---\n" not in readme[4:]:
        problems.append("missing YAML front matter")
    for link in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", readme):
        if not link.startswith("http") and link not in files:
            problems.append(f"image does not resolve: {link}")
    for link in re.findall(r"(?<!!)\[[^\]]*\]\(([^)#]+)\)", readme):
        if not link.startswith("http") and link not in files:
            problems.append(f"link does not resolve: {link}")
    headings = re.findall(r"^#{1,3} .+$", readme, flags=re.M)
    for required in (*REQUIRED_SECTIONS, TRANSFORMERS_HEADING):
        if required not in headings:
            problems.append(f"missing section: {required}")
    order = [h for h in headings if h in REQUIRED_SECTIONS]
    if order != [h for h in REQUIRED_SECTIONS if h in order]:
        problems.append("sections out of order")
    if "```python" not in readme:
        problems.append("missing Python example")
    for chart in CHART_FILES:
        if f"]({chart})" not in readme:
            problems.append(f"missing chart: {chart}")
    return problems
