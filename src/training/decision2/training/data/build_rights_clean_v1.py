"""Build a rights-audited Decision 2.0 TRAIN/SELECT/CAL from frozen inputs.

TRAIN removes all MultiNLI and TweetEval-origin groups. SELECT and CAL are
fresh, independently seeded oracle problems; CSS and pressure references are
opened only as gold-free overlap exclusions. No source rows are redistributed.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
from pathlib import Path
from typing import Any

from training.data import build_nox4b_no_mnli as no_mnli
from training.data import build_pilot as pilot
from training.data import build_targeted_candidate as targeted
from training.model.data import check_partition_isolation, load_partition

PARENT_SHA256 = "d8d670f965a69aa08ed1f9704696b02179999fc1cc2ba2aad3bf1c0b479ca12c"
PARENT_MANIFEST_SHA256 = (
    "a702223b3b6712779b5fbcd57f85661f5d1d004820371e8202cb7a5396f1bcfa"
)
REFERENCE_HASHES = {
    "synthetic_dev": "a17ec4b675bbc3da96dba8f31af8f25c9b02cc96ff048fb7de899bdd8b6cf79a",
    "css_pilot": "598319a429de16c659b59ede0eac3c269939356b0599f4e08d1983e44def3dda",
    "css_evaluation": "7a527357e8ac3ca8da8f8663da66684d04c568a8c728261125c194294dd34af6",
}
SEED = "decision2-rights-clean-v1"
TRAIN_NAME = "rights_clean.train.jsonl"


def _counts(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(collections.Counter(row[field] for row in rows).items()))


def _is_excluded(row: dict[str, Any]) -> bool:
    return no_mnli.is_mnli_origin(row) or row["source"].startswith("tweeteval_train:")


def _set_role(row: dict[str, Any], role: str) -> dict[str, Any]:
    row = dict(row)
    row["split"] = role
    row["evaluation_role"] = "calibrate" if role == "cal" else "select"
    row["source"] = "decision2_rights_clean_oracle_holdout_v1"
    row["audit_metadata"] = {**row["audit_metadata"], "holdout_only": True}
    return row


def _candidate_groups(seed: str) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    generators = (
        ("pilot_string_composition", pilot.make_composition),
        ("pilot_narrative_reading", pilot.make_reading),
        ("pilot_open_world_abstention", pilot.make_abstention),
    )
    for family, generator in generators:
        for index in range(2400):
            row = generator(seed, index)
            groups[row["group_id"]] = [row]
    for index in range(2400):
        rows = targeted.quantized_median(seed, index)
        if len(rows) != 2 or len({row["group_id"] for row in rows}) != 1:
            raise AssertionError("Score oracle pair changed group shape")
        groups[rows[0]["group_id"]] = rows
    return groups


def _exclude_context_groups(
    groups: dict[str, list[dict[str, Any]]],
    protected: list[dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    protected_ids = {row["id"] for row in protected}
    protected_groups = {
        row.get("group_id") for row in protected if isinstance(row.get("group_id"), str)
    }
    protected_inputs = {
        row.get("input_sha256")
        for row in protected
        if isinstance(row.get("input_sha256"), str)
    }
    hashes = {targeted.text_hashes(row["state"]) for row in protected}
    raw, norm = ({pair[index] for pair in hashes} for index in (0, 1))
    eligible = {}
    rejected = collections.Counter()
    for group_id, rows in groups.items():
        if any(
            row["id"] in protected_ids
            or row["group_id"] in protected_groups
            or row["input_sha256"] in protected_inputs
            for row in rows
        ):
            rejected["id_group_input"] += len(rows)
        elif any(
            (pair := targeted.text_hashes(row["state"]))[0] in raw or pair[1] in norm
            for row in rows
        ):
            rejected["exact_context"] += len(rows)
        else:
            eligible[group_id] = rows
    near = pilot.near_duplicates(
        targeted.context_rows([row for rows in eligible.values() for row in rows]),
        targeted.context_rows(protected),
        collect_left_ids=True,
    )
    near_ids = set(near.pop("left_ids"))
    for group_id in list(eligible):
        if any(row["id"] in near_ids for row in eligible[group_id]):
            rejected["near_context"] += len(eligible.pop(group_id))
    return eligible, {
        "candidate_groups": len(groups),
        "eligible_groups": len(eligible),
        "eligible_rows": sum(map(len, eligible.values())),
        "rejected_rows": dict(sorted(rejected.items())),
        "near_before_quarantine": near,
    }


def _select_holdouts(
    groups: dict[str, list[dict[str, Any]]],
    seed: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    by_family: dict[str, list[str]] = collections.defaultdict(list)
    for group_id, rows in groups.items():
        by_family[(rows[0]["family"], rows[0]["task_type"])].append(group_id)
    for family, group_ids in by_family.items():
        group_ids.sort(
            key=lambda group_id: pilot.sha_bytes(
                f"{seed}\0{family}\0{group_id}".encode()
            )
        )
    quotas = {
        ("pilot_string_composition", "choice"): 40,
        ("pilot_narrative_reading", "choice"): 40,
        ("pilot_open_world_abstention", "choice"): 40,
        ("pilot_narrative_reading", "noul"): 90,
        ("targeted_quantized_median", "score"): 45,
    }
    selected_groups = []
    for family, group_count in quotas.items():
        if len(by_family[family]) < 2 * group_count:
            raise ValueError(f"Insufficient clean oracle groups for {family}")
        selected_groups.extend(by_family[family][:group_count])
    select = [
        _set_role(row, "select")
        for group_id in selected_groups
        for row in groups[group_id]
    ]
    available = {
        group_id: rows
        for group_id, rows in groups.items()
        if group_id not in set(selected_groups)
    }
    available, cal_filter = _exclude_context_groups(available, select)
    by_family = collections.defaultdict(list)
    for group_id, rows in available.items():
        by_family[(rows[0]["family"], rows[0]["task_type"])].append(group_id)
    cal_groups = []
    for family, group_count in quotas.items():
        ordered = sorted(
            by_family[family],
            key=lambda group_id: pilot.sha_bytes(
                f"{seed}\0cal\0{family}\0{group_id}".encode()
            ),
        )
        if len(ordered) < group_count:
            raise ValueError(f"Insufficient CAL groups for {family}: {len(ordered)}")
        cal_groups.extend(ordered[:group_count])
    cal = [
        _set_role(row, "cal") for group_id in cal_groups for row in available[group_id]
    ]
    if (len(select), len(cal)) != (300, 300):
        raise AssertionError("Unexpected clean SELECT/CAL size")
    return (
        select,
        cal,
        {
            "groups_per_partition": {
                f"{family}/{task_type}": count
                for (family, task_type), count in quotas.items()
            },
            "selected_group_count": len(selected_groups),
            "cal_group_count": len(cal_groups),
            "cal_candidate_filter": cal_filter,
        },
    )


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    frozen = {
        "parent_train": (args.parent_train, PARENT_SHA256),
        "parent_manifest": (args.parent_manifest, PARENT_MANIFEST_SHA256),
        "old_select": (args.old_select, no_mnli.structured.SELECT_SHA256),
        "old_cal": (args.old_cal, no_mnli.structured.CAL_SHA256),
        "synthetic_dev": (args.dev_prompts, REFERENCE_HASHES["synthetic_dev"]),
        "css_pilot": (args.css_pilot_prompts, REFERENCE_HASHES["css_pilot"]),
        "css_evaluation": (
            args.css_evaluation_prompts,
            REFERENCE_HASHES["css_evaluation"],
        ),
    }
    for role, (path, digest) in frozen.items():
        if pilot.sha_file(path) != digest:
            raise ValueError(f"Frozen {role} SHA-256 differs")
    parent_manifest = json.loads(args.parent_manifest.read_text())
    if parent_manifest["outputs"][no_mnli.OUTPUT_NAME]["sha256"] != PARENT_SHA256:
        raise ValueError("Parent manifest TRAIN bytes differ")
    parent = load_partition(args.parent_train, "train")
    if len(parent) != 8255:
        raise ValueError("Parent TRAIN row count differs")
    by_group: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in parent:
        by_group[row["group_id"]].append(row)
    train, removed = [], []
    for rows in by_group.values():
        flags = {_is_excluded(row) for row in rows}
        if len(flags) != 1:
            raise ValueError("Restricted source shares a group with eligible source")
        (removed if True in flags else train).extend(rows)
    if (
        len(train) != 4655
        or len(removed) != 3600
        or any(_is_excluded(row) for row in train)
    ):
        raise ValueError("Expected exact TweetEval removal")
    expected_tweet = {
        "tweeteval_train:hate": 500,
        "tweeteval_train:sentiment": 400,
        "tweeteval_train:emotion": 400,
        "tweeteval_train:irony": 400,
        "tweeteval_train:offensive": 400,
        **{
            f"tweeteval_train:stance/{topic}": 300
            for topic in ("climate", "hillary", "abortion", "atheism", "feminist")
        },
    }
    if _counts(removed, "source") != dict(sorted(expected_tweet.items())):
        raise ValueError("TweetEval task inventory changed")
    references = {}
    for role, path in (
        ("old_select", args.old_select),
        ("old_cal", args.old_cal),
        ("synthetic_dev", args.dev_prompts),
        ("css_pilot", args.css_pilot_prompts),
        ("css_evaluation", args.css_evaluation_prompts),
        ("rq1", args.rq1_prompts),
        ("rq2", args.rq2_prompts),
        ("rq3", args.rq3_prompts),
    ):
        references[role], _ = targeted.load_context_reference(path)
    protected = [*train, *(row for rows in references.values() for row in rows)]
    candidates, candidate_filter = _exclude_context_groups(
        _candidate_groups(args.seed), protected
    )
    select, cal, selection = _select_holdouts(candidates, args.seed)
    check_partition_isolation({"train": train, "select": select, "cal": cal})
    if pilot.train_consistency_audit(train)["conflicting_gold_groups"]:
        raise ValueError("TRAIN gold conflicts")
    audits = {}
    for name, rows in {"select": select, "cal": cal, **references}.items():
        audits[f"train_vs_{name}"] = targeted.context_overlap(
            train, rows, approximate=True
        )
    for name, rows in references.items():
        audits[f"select_vs_{name}"] = targeted.context_overlap(
            select, rows, approximate=True
        )
        audits[f"cal_vs_{name}"] = targeted.context_overlap(cal, rows, approximate=True)
    audits["select_vs_cal"] = targeted.context_overlap(select, cal, approximate=True)
    payloads = {
        TRAIN_NAME: pilot.jsonl_bytes(train),
        "select.jsonl": pilot.jsonl_bytes(select),
        "cal.jsonl": pilot.jsonl_bytes(cal),
    }
    report = {
        "schema_version": "decision2-rights-clean-splits/1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "generator_code_sha256": pilot.sha_file(Path(__file__)),
        "seed": args.seed,
        "inputs": {
            role: {"file": path.name, "sha256": digest}
            for role, (path, digest) in frozen.items()
        },
        "additional_gold_free_references": {
            name: {"file": path.name, "sha256": pilot.sha_file(path)}
            for name, path in (
                ("rq1", args.rq1_prompts),
                ("rq2", args.rq2_prompts),
                ("rq3", args.rq3_prompts),
            )
        },
        "derivation": "remove all TweetEval and MultiNLI origins; retain every other frozen TRAIN row unchanged; generate independent oracle SELECT/CAL",
        "removed_from_no_mnli_parent": _counts(removed, "source"),
        "counts": {
            role: {
                field: _counts(rows, field)
                for field in ("source", "family", "task_type", "language")
            }
            for role, rows in (("train", train), ("select", select), ("cal", cal))
        },
        "source_rights": [
            {
                "source": "original_programmatic",
                "rows": sum(
                    row["source"]
                    in (
                        "legacy:stage4-general-composition-v2",
                        "decision2_targeted_programmatic_v1",
                        "decision2_programmatic_original_v1",
                    )
                    for row in train
                ),
                "license": "internally generated",
                "evidence": "pinned generators and objective oracle audits",
            },
            {
                "source": "Stage3 replay",
                "rows": 644,
                "license": "435 internal; CLINC150 CC BY 3.0; BANKING77 CC BY 4.0",
                "evidence": "https://github.com/clinc/oos-eval/blob/master/LICENSE ; https://huggingface.co/datasets/PolyAI/banking77",
            },
            {
                "source": "CosmosQA",
                "rows": 448,
                "license": "CC BY 4.0",
                "evidence": "https://huggingface.co/datasets/allenai/cosmos_qa",
            },
            {
                "source": "SNLI",
                "rows": 272,
                "license": "CC BY-SA 4.0",
                "evidence": "https://nlp.stanford.edu/projects/snli/",
            },
            {
                "source": "SQuAD 2.0",
                "rows": 334,
                "license": "CC BY-SA 4.0",
                "evidence": "https://rajpurkar.github.io/SQuAD-explorer/",
            },
            {
                "source": "FLUTE",
                "rows": 120,
                "license": "AFL 3.0",
                "evidence": "https://huggingface.co/datasets/ColumbiaNLP/FLUTE",
            },
            {
                "source": "oracle SELECT/CAL",
                "rows": 1200,
                "license": "internally generated",
                "evidence": "generator_code_sha256",
            },
        ],
        "excluded_rights": {
            "MultiNLI": "267 excluded in parent for unresolved annotation/derivative terms",
            "TweetEval": "3600 excluded; original irony and HatEval have NC terms, NRC emotion/stance research-only, other task/platform terms not affirmatively cleared",
        },
        "publication_eligible": True,
        "publication_scope": "trained weights/modelcard only; no raw source rows, SELECT/CAL rows, or individual text predictions",
        "publication_conditions": [
            "attribute all named datasets",
            "retain original source licenses and notices",
            "document CC BY-SA sources and model-weight derivation uncertainty",
            "audit separately the chosen base model's training lineage and license",
        ],
        "candidate_filter": candidate_filter,
        "selection": selection,
        "overlap_audits": audits,
        "outputs": {
            name: {
                "sha256": pilot.sha_bytes(payload),
                "rows": len(payload.splitlines()),
                "bytes": len(payload),
            }
            for name, payload in payloads.items()
        },
        "limitations": [
            "The new SELECT/CAL are narrow synthetic oracles, not natural-distribution calibration.",
            "Approximate near-context screening cannot prove semantic independence.",
            "The 120 FLUTE TRAIN rows make CSS FLUTE same-task supervised.",
            "The base model's pretraining and earlier fine-tuning rights require separate audit.",
        ],
    }
    args.output_dir.mkdir(parents=True, mode=0o700)
    for name, payload in payloads.items():
        pilot._atomic_write(args.output_dir / name, payload)
    pilot._atomic_write(
        args.output_dir / "rights_clean.manifest.json",
        (json.dumps(report, ensure_ascii=False, indent=2) + "\n").encode(),
    )
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "parent-train",
        "parent-manifest",
        "old-select",
        "old-cal",
        "dev-prompts",
        "css-pilot-prompts",
        "css-evaluation-prompts",
        "rq1-prompts",
        "rq2-prompts",
        "rq3-prompts",
        "output-dir",
    ):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--seed", default=SEED)
    args = parser.parse_args(argv)
    report = build(args)
    print(
        json.dumps(
            {
                "rows": {k: v["rows"] for k, v in report["outputs"].items()},
                "hashes": {k: v["sha256"] for k, v in report["outputs"].items()},
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
