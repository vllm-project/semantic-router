"""Render the public 0.6B card and figures from three pinned same-panel reports.

Only compact scored aggregates and hashes enter the product package. Raw
predictions, gold, private paths, and training records stay outside it.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from statistics import median
from typing import Any

from jev_arena.render import ranking_svg as public_rank_svg

from .bundle import sha_file
from .full_bundle import CONTENT_FILES, QWEN_LICENSE_SHA256
from .product_card import ProductCard, render
from .render_arena_v3 import ranking_svg, task_matrix_svg

REPORT_HASHES = {
    "new": (
        "1cdb39ee11ee027e49233093b08c2b09b5e09684a421f3ed94c228f802c2229b",
        "80e7e66fe529b610defdc5c50f28965f112a21ec7ee431f1b43a360c066d17f4",
        "17075505c67717fa3efe78760995ce53d994fa52c2cd183d0621b7025196289f",
    ),
    "kai": (
        "5c2e4db8f935e4bc284d98cca8dba4eeea2ee44234cbaf533b54538f700d3d77",
        "88b40e6561a13a51c8639c21ff80539dd12d8d4d8120f86c1f33302fe465f64a",
        "05378fed31c6ef390740831f29c8e41271e5789af48f2d431057c0075da7e441",
    ),
    "bosun": (
        "67c53af9e75bc63a15b1589261d3185c2f8017cf299123255404af8403e401d8",
        "bee85bd314ab9ed742e518a45342f0d9c7f56f17a883550d7938ea84f7bf792b",
        "5a6e2f770c4a16d934b544a04428e8d683372cdba06b902ba95c4035e71ce87f",
    ),
}
ROSTER = {
    "new": ("DEV2.0-0.6B", "decision2", 597_103_104),
    "kai": ("Decision 1.0 Kai-0.6B", "decision1", 571_909_635),
    "bosun": ("Bosun-v3.1-0.6B", "open", 606_131_200),
}
SOURCE_REVISION = "da87bfb608c14b7cf20ba1ce41287e8de496c0cd"
BANNER_SHA256 = "c8a91da11d38499a6e9162f67114d3da328be54c5c608dcae899dfaf810e6ad7"


def _read(path: Path, digest: str) -> dict[str, Any]:
    if sha_file(path) != digest:
        raise ValueError(f"Scored report hash mismatch: {path.name}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Scored report must be an object")
    return value


def _row(
    key: str, typed: dict[str, Any], css: dict[str, Any], public: dict[str, Any]
) -> dict[str, Any]:
    label, group, parameters = ROSTER[key]
    if (
        typed.get("schema_version") != "typed-decision-report/2"
        or typed.get("items") != 1600
        or css.get("score_schema_version") != "css-transfer-score/2"
        or len(css.get("tasks", {})) != 15
        or public.get("score_version") != "jevarena-jevbench-public-score/1"
        or public.get("items") != 231
    ):
        raise ValueError(f"{key}: reports differ from the frozen panel")
    if key in {"kai", "bosun"}:
        expected_id = (
            "llm-semantic-router/Decision-1.0-Kai-0.6B"
            if key == "kai"
            else "Hanno-Labs/bosun-v3.1-0.6b"
        )
        if (
            typed.get("model", {}).get("id") != expected_id
            or public.get("model_id") != expected_id
        ):
            raise ValueError(f"{key}: peer identity differs")
    elif typed.get("model", {}).get("id") != "research/qwen3-06b-official-full466":
        raise ValueError("Official 0.6B scored candidate differs")
    task_scores = {task: item["macro_f1_all"] for task, item in css["tasks"].items()}
    typed_scores = {
        kind: item["accuracy_all"] for kind, item in typed["by_type"].items()
    }
    if set(typed_scores) != {"choice", "noul", "score"} or any(
        not isinstance(value, (float, int))
        or not math.isfinite(value)
        or not 0 <= value <= 1
        for value in (*typed_scores.values(), *task_scores.values())
    ):
        raise ValueError(f"{key}: invalid task score")
    t, h = typed["macro_family_accuracy"], median(task_scores.values())
    score = 100 * math.sqrt(t * h)
    if public["correct"] < 0 or public["correct"] > 231:
        raise ValueError(f"{key}: invalid public correct count")
    return {
        "key": key,
        "label": label,
        "group": group,
        "parameters": parameters,
        "score": score,
        "axes": {"typed": t, "transfer": h},
        "task_scores": {"typed": typed_scores, "transfer": task_scores},
        "public_correct": public["correct"],
        "public_tiers": public["tiers"],
        "typed_valid": typed["overall"]["valid_n"],
        "css_invalid": sum(
            item["invalid_or_missing_n"] for item in css["tasks"].values()
        ),
        "public_valid": public["valid"],
    }


def build(
    *,
    report_paths: dict[str, tuple[Path, Path, Path]],
    banner: Path,
    qwen_license: Path,
    apache_license: Path,
    output: Path,
) -> dict[str, Any]:
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    if (
        sha_file(banner) != BANNER_SHA256
        or sha_file(qwen_license) != QWEN_LICENSE_SHA256
    ):
        raise ValueError(
            "Banner or official Qwen license differs from the pinned source"
        )
    reports = {
        key: tuple(
            _read(path, digest)
            for path, digest in zip(report_paths[key], REPORT_HASHES[key], strict=True)
        )
        for key in ROSTER
    }
    for panel, field in (
        (0, "gold_sha256"),
        (1, "gold_sha256"),
        (2, "panel_manifest_sha256"),
    ):
        identities = {reports[key][panel].get(field) for key in ROSTER}
        if len(identities) != 1 or not isinstance(next(iter(identities)), str):
            raise ValueError("Models were not scored on the same panel")
    if (
        reports["new"][0].get("predictions_sha256")
        != "e7380958e5273ad89eaf2c2b80cf8812111d88ac351de6bf1bfa06cfaf718552"
        or reports["new"][1].get("predictions_sha256")
        != "ef521866fc50d8fb1e2f21705b44141a0fac04fe0e11595b68d872f3e3e512ef"
        or reports["new"][2].get("predictions_sha256")
        != "2cf2ab0a2ba097ecb71ea02124d0674fc7d11e64fc29b89d41a09a40390b0244"
    ):
        raise ValueError("Candidate reports are not bound to the sealed predictions")
    rows = [_row(key, *reports[key]) for key in ROSTER]
    ordered = sorted(rows, key=lambda row: (-row["score"], row["key"]))
    for rank, row in enumerate(ordered, 1):
        row["rank"] = rank
    arena = {
        "schema_version": "jevarena-ranking/3",
        "scope": "post-key same-panel",
        "models": ordered,
    }
    public_rows = sorted(rows, key=lambda row: (-row["public_correct"], row["key"]))
    public = {
        "schema_version": "jevarena-jevbench-public-rank/1",
        "edition": "v1.2",
        "models": [],
    }
    for rank, row in enumerate(public_rows, 1):
        public["models"].append(
            {
                "key": row["key"],
                "label": row["label"],
                "group": row["group"],
                "rank": rank,
                "score": 100 * row["public_correct"] / 231,
            }
        )

    output.mkdir(parents=True)
    (output / "assets").mkdir()
    (output / "evaluation").mkdir()
    shutil.copy2(banner, output / "assets/DEV2.0-0.6B-owl-banner.png")
    shutil.copy2(qwen_license, output / "LICENSE-Qwen")
    shutil.copy2(apache_license, output / "LICENSE")
    (output / "assets/jevarena-rank.svg").write_text(
        ranking_svg(arena), encoding="utf-8"
    )
    (output / "assets/jevarena-task-matrix.svg").write_text(
        task_matrix_svg(arena), encoding="utf-8"
    )
    (output / "assets/jevbench-public-rank.svg").write_text(
        public_rank_svg(public), encoding="utf-8"
    )

    def pct(value: float) -> str:
        return f"{value * 100:.2f}%"

    table = [
        "| Rank | Model | Parameters | JevArena v3 ↑ | Human transfer ↑ | JevBench public ↑ |",
        "| ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in ordered:
        table.append(
            f"| {row['rank']} | {row['label']} | {row['parameters']:,} | "
            f"{row['score']:.3f} | {pct(row['axes']['transfer'])} | "
            f"{row['public_correct']}/231 |"
        )
    table += [
        "",
        "| Typed decision | DEV2.0-0.6B | Decision 1.0 Kai | Bosun |",
        "| --- | ---: | ---: | ---: |",
    ]
    for kind in ("choice", "noul", "score"):
        denominator = 400 if kind == "score" else 800
        table.append(
            "| "
            + kind.title()
            + " | "
            + " | ".join(
                f"{round(next(row for row in rows if row['key'] == key)['task_scores']['typed'][kind] * denominator)}/{denominator}"
                for key in ("new", "kai", "bosun")
            )
            + " |"
        )
    card = ProductCard(
        model_id="llm-semantic-router/DEV2.0-0.6B",
        banner="assets/DEV2.0-0.6B-owl-banner.png",
        tagline="A compact model for choices, yes/no checks and ordered scores.",
        measured_summary=(
            "**38.52** on JevArena v3 versus **35.94** for Kai 1.0. "
            "Human-task transfer improves; Choice and Score decline. The "
            "paired interval includes zero, so the overall difference "
            "remains uncertain."
        ),
        use_cases=(
            "Choose among supplied routes or options with explicit criteria.",
            "Judge a yes/no proposition and return a probability estimate.",
            "Score an ordered set of levels from the evidence in context.",
        ),
        direct_weight_source="Qwen/Qwen3-0.6B-Base",
        source_revision=SOURCE_REVISION,
        loaded_parameters=597_103_104,
        method=(
            "A causal text backbone reads the state and each question. A native "
            "candidate head scores supplied options or ordered levels and "
            "returns probabilities without generating chat text."
        ),
        limitations=(
            "Choice and Score are weaker than Kai 1.0 on this panel; Noul and human-task transfer are stronger.",
            "The +2.582-point JevArena difference has a paired 95% interval of [-2.032, +7.846].",
            "Fifteen transfer inputs exceeded the 8,192-token limit and counted as failures. Review important decisions against source evidence.",
        ),
        evidence_table="\n".join(table),
        evaluation_scope=(
            "JevBench covers its public 231 questions, not the full closed "
            "benchmark. JevArena answers were available during development, "
            "so this comparison still needs independent confirmation."
        ),
    )
    (output / "README.md").write_text(render(card), encoding="utf-8")
    (output / "NOTICE").write_text(
        "Decision 2.0 0.6B: original package code, card and visual are released under Apache-2.0.\n"
        "The full checkpoint starts from the separately licensed official Qwen3-0.6B-Base weights; see LICENSE-Qwen.\n"
        "Decision 2.0 owl motif derives from our Decision 1.0 Nox-4B artwork. See ATTRIBUTIONS.md.\n",
        encoding="utf-8",
    )
    (output / "ATTRIBUTIONS.md").write_text(
        "# Source and artwork credits\n\n"
        "- [Qwen3-0.6B-Base](https://huggingface.co/Qwen/Qwen3-0.6B-Base) "
        f"at `{SOURCE_REVISION}`: original starting weights; see [LICENSE-Qwen](LICENSE-Qwen).\n"
        "- [Decision 1.0 Nox-4B](https://huggingface.co/llm-semantic-router/Decision-1.0-Nox-4B) "
        "at `cde2a68dbaa557ea65dc458104d410a0802ee259`: our mosaic owl motif, "
        "adapted into this model-name banner.\n"
        "- Supervised training uses internally generated rule data, plus "
        "labeled records from [GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions) "
        "(CC BY 4.0), [CLINC150](https://github.com/clinc/oos-eval) (CC BY 3.0), "
        "[BANKING77](https://huggingface.co/datasets/PolyAI/banking77) (CC BY 4.0), "
        "[CosmosQA](https://huggingface.co/datasets/allenai/cosmos_qa) (CC BY 4.0), "
        "[SNLI](https://nlp.stanford.edu/projects/snli/) (CC BY-SA 4.0), "
        "[SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/) (CC BY-SA 4.0), "
        "and [FLUTE](https://huggingface.co/datasets/ColumbiaNLP/FLUTE) (AFL 3.0). "
        "Their source terms remain applicable; our Apache-2.0 declaration does not relicense upstream material.\n",
        encoding="utf-8",
    )
    (output / "evaluation/EVALUATION.md").write_text(
        "# Evaluation\n\n"
        "JevArena v3 uses 1,600 typed original items (2,000 answer slots) and 15 "
        "human-labeled transfer tasks (6,547 items). Its scalar is "
        "`100 × sqrt(T × H)`, where T is four-family macro accuracy and H is "
        "the median task macro-F1. Missing, invalid and over-budget outputs "
        "are failures. The paired interval resamples independent typed groups "
        "and transfer tasks/items (5,000 replicates). The candidate-minus-Kai "
        "point estimate is +2.582 with 95% interval [-2.032, +7.846].\n\n"
        "JevBench v1.2 public is a separate, independently rerun 231-question "
        "subset (easy 48, standard 72, hard 111). The displayed raw accuracy "
        "is not the upstream four-axis official score or closed-set ranking. "
        "DEV2.0-0.6B scores 48/48, 51/72 and 44/111 respectively. "
        "Native-valid answers were 231/231 for DEV2.0-0.6B and Bosun, and "
        "187/231 for Kai; invalid answers remain failures in the 231 denominator.\n\n"
        "This is a prospective post-key same-panel rerun. Project-level formal "
        "answer-key access predates this candidate; its prediction files were "
        "sealed before same-panel scoring. It is not an untouched blind test. "
        "The model's direct source and source-data credits appear in the card "
        "and ATTRIBUTIONS.md. Its TRAIN data include FLUTE training examples, "
        "while the human-task panel contains disjoint FLUTE evaluation items; "
        "the 15 tasks therefore are not uniformly unseen-source zero-shot transfer. "
        "Exact public panel and report hashes are in "
        "[manifest.json](manifest.json).\n",
        encoding="utf-8",
    )
    public_manifest = {
        "schema_version": "decision2-06b-product-evaluation/1",
        "scope": "post-key same-panel",
        "typed_items": 1600,
        "typed_answer_slots": 2000,
        "transfer_items": 6547,
        "transfer_tasks": 15,
        "jevbench_public_items": 231,
        "jevbench_scope": "v1.2 independently rerun public 231; raw accuracy, not official closed rank",
        "candidate_checkpoint_sha256": "5380e01e3fbb5f541d6548144dfb9e0776292d28a52d490f90c8929f764f37ab",
        "candidate_prediction_seal_sha256": "3bd2e8e2ee992ef170bfccbf801d142cdecddad317e7186bd6bc0c9cb46ec33c",
        "calibration_sha256": "dc29fc12c65f2cf0a676f547360703fde5e846b1df0cd4fb557f830fb3d13da5",
        "report_sha256": {
            key: dict(
                zip(("typed", "transfer", "jevbench_public"), values, strict=True)
            )
            for key, values in REPORT_HASHES.items()
        },
        "jevbench_public_valid_answers": {
            row["key"]: row["public_valid"] for row in rows
        },
        "figure_sha256": {
            path.name: sha_file(path)
            for path in sorted((output / "assets").glob("*.svg"))
        },
        "banner_sha256": BANNER_SHA256,
        "source_revision": SOURCE_REVISION,
        "loaded_parameters": {key: value[2] for key, value in ROSTER.items()},
    }
    (output / "evaluation/manifest.json").write_text(
        json.dumps(public_manifest, indent=2, sort_keys=True, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )
    if {
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file()
    } != CONTENT_FILES:
        raise ValueError("Rendered product files differ from the publication whitelist")
    return {
        "report_sha256": public_manifest["report_sha256"],
        "figure_sha256": public_manifest["figure_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ROSTER:
        for panel in ("typed", "css", "public"):
            parser.add_argument(f"--{key}-{panel}", type=Path, required=True)
    for name in ("banner", "qwen-license", "apache-license", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    paths = {
        key: tuple(
            getattr(args, f"{key}_{panel}") for panel in ("typed", "css", "public")
        )
        for key in ROSTER
    }
    print(
        json.dumps(
            build(
                report_paths=paths,
                banner=args.banner,
                qwen_license=args.qwen_license,
                apache_license=args.apache_license,
                output=args.output,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
