"""Build a separate harder CAL while retaining the 300 CSS pilot Choice rows.

The new Noul and Score rows are generated from predeclared oracle mechanisms.
No model prediction or score is used to select them. Synthetic final and CSS
fifteen-task evaluation files are neither accepted nor opened.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
from pathlib import Path
from typing import Any

from training.data import build_pilot as pilot
from training.data import build_targeted_candidate as targeted
from training.model.data import check_partition_isolation, load_partition, validate_row

SOURCE = "decision2_cal_hard_original_v2"
FROZEN_OLD_CAL_SHA256 = targeted.FROZEN_CAL_SHA256
FROZEN_SELECT_SHA256 = targeted.FROZEN_SELECT_SHA256
FROZEN_TRAIN_SHA256 = "18714248d5afdc803a91e7c0ed4d5613d19d113e6b42deb684c48524dad665bc"
FROZEN_COMBINED_SHA256 = targeted.FROZEN_COMBINED_SHA256
FROZEN_PURE_LEGACY_SHA256 = (
    "437055a3d8d222e7e181ff7484f11988d830596e4301708feb5fccf84edfb5f6"
)
FROZEN_TARGETED_SHA256 = (
    "59d40112ab4d9d0688b3b121212ad5cc36329fe18170839485c9906544f0dee0"
)
FROZEN_DEV_PROMPTS_SHA256 = (
    "a17ec4b675bbc3da96dba8f31af8f25c9b02cc96ff048fb7de899bdd8b6cf79a"
)
FROZEN_CSS_PILOT_PROMPTS_SHA256 = (
    "598319a429de16c659b59ede0eac3c269939356b0599f4e08d1983e44def3dda"
)
NAMES = (
    "Ada",
    "Basil",
    "Cora",
    "Davin",
    "Elin",
    "Farah",
    "Galen",
    "Hana",
    "Ivo",
    "Jorin",
)
DOCUMENTS = (
    "map sheet",
    "field journal",
    "catalog slip",
    "survey packet",
    "site diagram",
)
TIERS = ("direct", "boundary", "conditional")


def identifiers(seed: str, family: str, index: int, variant: int) -> tuple[str, str]:
    digest = pilot.sha_bytes(f"{seed}\0{family}\0{index}".encode())[:20]
    return f"d2calh_{digest}_{variant}", f"d2calhg_{digest}"


def cal_row(
    seed: str,
    family: str,
    index: int,
    variant: int,
    *,
    state: str,
    instructions: str,
    options: list[dict[str, str]],
    answer: str,
    task_type: str,
    language: str,
    tier: str,
    oracle: dict[str, Any],
) -> dict[str, Any]:
    identifier, group = identifiers(seed, family, index, variant)
    row = {
        "id": identifier,
        "state": state,
        "instructions": instructions,
        "options": options,
        "label": next(i for i, option in enumerate(options) if option["key"] == answer),
        "task_type": task_type,
        "family": f"cal_hard_{family}",
        "group_id": group,
        "language": language,
        "split": "cal",
        "source": SOURCE,
        "evaluation_role": "calibrate",
        "render_template": f"cal_hard_{family}_{tier}_v2",
        "audit_metadata": {
            "holdout_only": True,
            "generator": "cal-hard-oracle-v2",
            "case_index": index,
            "paired_variant": variant,
            "difficulty_tier": tier,
            "oracle": oracle,
        },
    }
    row["input_sha256"] = pilot.input_sha256(row)
    validate_row(row, "cal")
    return row


def noul_case(seed: str, index: int) -> list[dict[str, Any]]:
    """Approval requires signature, threshold, and an embargo waiver when needed."""
    rng = pilot.rng_for(seed, "cal-hard-noul", index)
    tier = TIERS[index % 3]
    language = "zh" if index % 5 == 0 else "en"
    name = rng.choice(NAMES)
    document = rng.choice(DOCUMENTS)
    case_code = f"AR-{1000 + rng.randint(0, 8999)}"
    minimum = rng.randint(12, 34)
    if tier == "direct":
        pages = (minimum + rng.randint(1, 4),) * 2
        signed = (True, False)
        embargo = waiver = (False, False)
    elif tier == "boundary":
        pages = (minimum - 1, minimum)
        signed = (True, True)
        embargo = waiver = (False, False)
    else:
        pages = (minimum, minimum)
        signed = embargo = (True, True)
        waiver = (False, True)
    if index % 2:
        pages, signed, embargo, waiver = (
            values[::-1] for values in (pages, signed, embargo, waiver)
        )
    result = []
    for variant in range(2):
        page_count, is_signed = pages[variant], signed[variant]
        is_embargoed, has_waiver = embargo[variant], waiver[variant]
        approved = (
            is_signed and page_count >= minimum and (not is_embargoed or has_waiver)
        )
        if language == "zh":
            state = (
                f"档案室审核 {name} 的 {document}（编号 {case_code}）。规则："
                f"提交页数至少 {minimum} 页，而且必须有正式签名；"
                "若有封存标记，还必须附已核实的豁免函。"
                f"这份材料有 {page_count} 页；正式签名{'已核实' if is_signed else '缺失'}；"
                f"封存标记{'存在' if is_embargoed else '不存在'}；"
                f"豁免函{'已核实' if has_waiver else '未核实'}。"
                "旁边另有一份未签署的草稿，不计入本次审核。"
            )
            instructions = "根据上述全部审核规则，这份正式材料可以通过吗？"
            options = [
                {"key": "true", "description": "可以通过"},
                {"key": "false", "description": "不能通过"},
            ]
        else:
            state = (
                f"The archive reviews {name}'s {document}, case {case_code}. The official submission "
                f"needs at least {minimum} pages and a verified signature. An embargoed submission "
                "also needs a verified waiver; a waiver cannot replace the page or signature rule. "
                f"This submission has {page_count} pages, a signature that is "
                f"{'verified' if is_signed else 'missing'}, an embargo flag that is "
                f"{'set' if is_embargoed else 'clear'}, and a waiver that is "
                f"{'verified' if has_waiver else 'unverified'}. An unsigned draft nearby is excluded."
            )
            instructions = (
                "Does the official submission pass all applicable approval rules?"
            )
            options = [
                {"key": "true", "description": "Approved"},
                {"key": "false", "description": "Not approved"},
            ]
        if variant:
            options.reverse()
        result.append(
            cal_row(
                seed,
                "archive_approval",
                index,
                variant,
                state=state,
                instructions=instructions,
                options=options,
                answer="true" if approved else "false",
                task_type="noul",
                language=language,
                tier=tier,
                oracle={
                    "minimum_pages": minimum,
                    "submitted_pages": page_count,
                    "signature_verified": is_signed,
                    "embargo": is_embargoed,
                    "waiver_verified": has_waiver,
                    "approved": approved,
                    "page_margin": page_count - minimum,
                },
            )
        )
    if {row["options"][row["label"]]["key"] for row in result} != {"true", "false"}:
        raise AssertionError("Noul pair fails to cross its decision boundary")
    return result


def score_case(seed: str, index: int) -> list[dict[str, Any]]:
    """One-unit net-credit changes cross an explicit ordinal cutoff."""
    rng = pilot.rng_for(seed, "cal-hard-score", index)
    tier = TIERS[index % 3]
    language = "zh" if index % 5 == 0 else "en"
    name = rng.choice(NAMES)
    first = rng.randint(14, 26)
    step = rng.randint(7, 12)
    cutoffs = [first + step * k for k in range(4)]
    boundary = index % 4
    nets = (cutoffs[boundary] - 1, cutoffs[boundary])
    if index % 2:
        nets = nets[::-1]
    fee = rng.randint(3, 9)
    bonus = rng.randint(1, 6)
    hold = rng.randint(4, 8)
    hold_applies = index % 2 == 0
    result = []
    for variant, net in enumerate(nets):
        grade = sum(net >= cutoff for cutoff in cutoffs)
        opening = (
            net + fee + (hold if tier == "conditional" and hold_applies else 0) - bonus
        )
        if tier == "direct":
            if language == "zh":
                state = (
                    f"{name} 的最终净积分为 {net}。四个升级阈值依次为 "
                    + "、".join(map(str, cutoffs))
                    + "。"
                )
            else:
                state = (
                    f"{name}'s final net credit is {net}. The four grade cutoffs are "
                    + ", ".join(map(str, cutoffs))
                    + "."
                )
        elif tier == "boundary":
            if language == "zh":
                state = (
                    f"{name} 的账户初始积分 {opening}，本次增加 {bonus}，扣除手续费 {fee}。"
                    "按加分后减手续费计算净积分。四个升级阈值依次为 "
                    + "、".join(map(str, cutoffs))
                    + "。"
                )
            else:
                state = (
                    f"{name}'s account starts at {opening} credits, receives {bonus} bonus credits, "
                    f"and pays a fee of {fee} credits. Compute opening plus bonus minus fee. "
                    "The four grade cutoffs are " + ", ".join(map(str, cutoffs)) + "."
                )
        else:
            tag = "MATCH" if hold_applies else "OTHER"
            if language == "zh":
                state = (
                    f"{name} 的账户初始积分 {opening}，增加 {bonus}，扣除手续费 {fee}。"
                    f"另有 {hold} 分的保留款，仅当标记为 MATCH 时才扣除；本次标记为 {tag}。"
                    "草稿写有另一金额，但不是正式记录。四个升级阈值依次为 "
                    + "、".join(map(str, cutoffs))
                    + "。"
                )
            else:
                state = (
                    f"{name}'s official account starts at {opening} credits, gains {bonus}, "
                    f"and pays a fee of {fee}. A separate hold of {hold} is deducted only "
                    f"when the tag reads MATCH; the tag here reads {tag}. A draft lists a "
                    "different amount but is not official. The four grade cutoffs are "
                    + ", ".join(map(str, cutoffs))
                    + "."
                )
        if language == "zh":
            state += "净积分每达到一个阈值升一级，低于全部阈值为 0 级。"
            instructions = "按正式记录计算最终净积分，这个账户属于 0 到 4 中的哪一级？"
        else:
            state += " The grade rises by one for each reached cutoff; below every cutoff is grade 0."
            instructions = (
                "What grade from 0 to 4 does the official net credit receive?"
            )
        options = [
            {
                "key": str(level),
                "description": f"{'等级' if language == 'zh' else 'Grade'} {level}",
            }
            for level in range(5)
        ]
        result.append(
            cal_row(
                seed,
                "net_credit_grade",
                index,
                variant,
                state=state,
                instructions=instructions,
                options=options,
                answer=str(grade),
                task_type="score",
                language=language,
                tier=tier,
                oracle={
                    "opening": opening,
                    "bonus": bonus,
                    "fee": fee,
                    "hold": hold if tier == "conditional" else 0,
                    "hold_applies": hold_applies if tier == "conditional" else False,
                    "net": net,
                    "cutoffs": cutoffs,
                    "grade": grade,
                    "distance_to_crossed_cutoff": abs(net - cutoffs[boundary]),
                },
            )
        )
    grades = [row["audit_metadata"]["oracle"]["grade"] for row in result]
    if abs(grades[0] - grades[1]) != 1:
        raise AssertionError("Score pair must cross one ordinal cutoff")
    return result


def generate(seed: str) -> list[dict[str, Any]]:
    rows = [row for index in range(150) for row in noul_case(seed, index)]
    rows.extend(row for index in range(150) for row in score_case(seed, index))
    if len(rows) != 600 or len({row["id"] for row in rows}) != 600:
        raise AssertionError("Fresh CAL needs 600 unique oracle rows")
    group_sizes = collections.Counter(row["group_id"] for row in rows)
    if len(group_sizes) != 300 or set(group_sizes.values()) != {2}:
        raise AssertionError("Fresh CAL must preserve 300 complete paired groups")
    return rows


def retained_choice(old_cal: list[dict[str, Any]]) -> list[dict[str, Any]]:
    choice = [row for row in old_cal if row["task_type"] == "choice"]
    if (
        len(choice) != 300
        or collections.Counter(row["source"] for row in choice)
        != {
            "css_pilot:semeval_stance": 100,
            "css_pilot:implicit_hate": 100,
            "css_pilot:discourse": 100,
        }
        or len({row["group_id"] for row in choice}) != 300
    ):
        raise ValueError(
            "Old CAL CSS Choice lineage is not the frozen 300-row partition"
        )
    return choice


def zero_audit(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    description: str,
    *,
    context_near: bool = False,
) -> dict[str, Any]:
    audit = pilot.overlap_audit(left, right)
    if context_near:
        audit["near_duplicate"] = pilot.near_duplicates(
            targeted.context_rows(left), targeted.context_rows(right)
        )
    if pilot.audit_has_exact_overlap(audit) or audit["near_duplicate"]["count"]:
        raise ValueError(f"{description} overlap: {audit}")
    return audit


def train_inventory_context(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read flattened TRAIN or Kev's nested private TRAIN for exclusion only."""
    rows = []
    shapes = collections.Counter()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            source = json.loads(line)
            if not isinstance(source, dict) or "state" not in source:
                raise ValueError(f"{path.name}:{line_number}: TRAIN row has no state")
            if isinstance(source.get("id"), str):
                identity = source
                if source.get("split") != "train":
                    raise ValueError(
                        f"{path.name}:{line_number}: non-TRAIN flattened row"
                    )
                shapes["flattened"] += 1
            else:
                identity = source.get("_meta")
                if not isinstance(identity, dict) or identity.get("split") != "train":
                    raise ValueError(
                        f"{path.name}:{line_number}: unrecognized nested TRAIN"
                    )
                shapes["kev_nested"] += 1
            if not isinstance(identity.get("id"), str) or not identity["id"]:
                raise ValueError(
                    f"{path.name}:{line_number}: TRAIN row has no source ID"
                )
            rows.append(
                {
                    "id": identity["id"],
                    "group_id": identity.get("group_id"),
                    "input_sha256": identity.get("input_sha256"),
                    "state": source["state"],
                    "instructions": "",
                    "options": [],
                    "task_type": "context",
                }
            )
    if not rows:
        raise ValueError(f"{path.name}: empty TRAIN inventory source")
    return rows, {
        "file": path.name,
        "sha256": pilot.sha_file(path),
        "rows": len(rows),
        "shapes": dict(sorted(shapes.items())),
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    frozen = (
        (args.old_cal, FROZEN_OLD_CAL_SHA256),
        (args.select_file, FROZEN_SELECT_SHA256),
        (args.targeted_anchor_train, FROZEN_TRAIN_SHA256),
        (args.combined_6k, FROZEN_COMBINED_SHA256),
        (args.pure_legacy_1024, FROZEN_PURE_LEGACY_SHA256),
        (args.targeted_2k, FROZEN_TARGETED_SHA256),
        (args.dev_prompts, FROZEN_DEV_PROMPTS_SHA256),
        (args.css_pilot_prompts, FROZEN_CSS_PILOT_PROMPTS_SHA256),
    )
    for path, expected in frozen:
        if pilot.sha_file(path) != expected:
            raise ValueError(f"Frozen reference differs: {path.name}")
    if args.legacy_v1_train.name != "train.jsonl":
        raise ValueError("Only the original 1.0 source train.jsonl is allowed")
    old_cal = load_partition(args.old_cal, "cal")
    select = load_partition(args.select_file, "select")
    choice = retained_choice(old_cal)
    fresh = generate(args.seed)
    zero_audit(fresh, old_cal, "Fresh CAL versus old CAL", context_near=True)
    cal = [*choice, *fresh]
    cal.sort(
        key=lambda row: (
            pilot.sha_bytes(f"{args.seed}\0output\0{row['id']}".encode()),
            row["id"],
        )
    )
    if len(cal) != 900 or collections.Counter(row["task_type"] for row in cal) != {
        "choice": 300,
        "noul": 300,
        "score": 300,
    }:
        raise AssertionError("New CAL must balance the three native types at 300 each")
    retained_ids = {row["id"] for row in choice}
    if {row["id"] for row in cal if row["task_type"] == "choice"} != retained_ids:
        raise AssertionError("Original CSS Choice CAL was altered")
    selection_audit = zero_audit(
        cal, select, "New CAL versus frozen SELECT", context_near=True
    )
    train_audits = {}
    for name, path in (
        ("targeted_anchor_3024", args.targeted_anchor_train),
        ("combined_6k", args.combined_6k),
        ("pure_legacy_1024", args.pure_legacy_1024),
        ("targeted_2k", args.targeted_2k),
    ):
        train_rows = load_partition(path, "train")
        train_audits[name] = zero_audit(cal, train_rows, f"New CAL versus {name}")
        if name == "targeted_anchor_3024":
            check_partition_isolation(
                {"train": train_rows, "select": select, "cal": cal}
            )
    # Audit every materialized TRAIN JSONL in the designated data directory,
    # plus caller-named runs outside it. Duplicated arms are recorded by path
    # and SHA rather than silently skipped.
    inventory_paths = sorted(
        set(args.train_root.resolve().rglob("*.train.jsonl"))
        | {path.resolve() for path in args.extra_train}
    )
    if (
        not inventory_paths
        or args.targeted_anchor_train.resolve() not in inventory_paths
    ):
        raise ValueError("TRAIN inventory must include the frozen targeted anchor")
    train_inventory = []
    for path in inventory_paths:
        if (
            not path.name.endswith(".train.jsonl")
            or "final" in path.name
            or "evaluation" in path.name
        ):
            raise ValueError(
                f"TRAIN inventory contains an unapproved filename: {path.name}"
            )
        reference, receipt = train_inventory_context(path)
        train_inventory.append(
            {
                "path_from_train_root": (
                    str(path.relative_to(args.train_root.resolve()))
                    if path.is_relative_to(args.train_root.resolve())
                    else path.name
                ),
                "source": receipt,
                "overlap": targeted.context_overlap(cal, reference, approximate=False),
            }
        )
    legacy, legacy_receipt = targeted.load_context_reference(
        args.legacy_v1_train, expected_name="train.jsonl"
    )
    legacy_audit = targeted.context_overlap(cal, legacy, approximate=False)
    open_refs = {}
    for name, path, expected in (
        ("synthetic_dev", args.dev_prompts, "dev.prompts.jsonl"),
        ("css_3task_pilot", args.css_pilot_prompts, "css-pilot.prompts.jsonl"),
    ):
        reference, receipt = targeted.load_context_reference(
            path, expected_name=expected
        )
        new_audit = targeted.context_overlap(fresh, reference, approximate=True)
        open_refs[name] = {"source": receipt, "fresh_noul_score_overlap": new_audit}
        if name == "css_3task_pilot":
            by_id = {row["id"]: row for row in reference}
            if any(
                row["id"] not in by_id
                or targeted.text_hashes(row["state"])
                != targeted.text_hashes(by_id[row["id"]]["state"])
                for row in choice
            ):
                raise ValueError(
                    "Retained Choice CAL differs from the CSS pilot source"
                )
            open_refs[name]["retained_css_choice_rows"] = len(choice)
        else:
            open_refs[name]["retained_choice_overlap"] = targeted.context_overlap(
                choice, reference, approximate=True
            )
    if pilot.train_consistency_audit(cal)["conflicting_gold_groups"]:
        raise ValueError("CAL has conflicting gold for identical inputs")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.tokenizer.resolve()), local_files_only=True, trust_remote_code=False
    )
    lengths = {row["id"]: pilot.count_tokens(row, tokenizer) for row in cal}
    if max(lengths.values()) > args.max_row_tokens:
        raise ValueError("New CAL row exceeds exact token cap")
    token_by_type = collections.Counter()
    difficulty = collections.Counter()
    labels = collections.defaultdict(collections.Counter)
    for row in cal:
        token_by_type[row["task_type"]] += lengths[row["id"]]
        if row["source"] == SOURCE:
            difficulty[
                (row["task_type"], row["audit_metadata"]["difficulty_tier"])
            ] += 1
            labels[row["task_type"]][row["options"][row["label"]]["key"]] += 1
    payloads = {
        "cal.jsonl": pilot.jsonl_bytes(cal),
        "select.jsonl": args.select_file.read_bytes(),
    }
    output_dir.mkdir(parents=True, mode=0o700)
    for name, payload in payloads.items():
        pilot._atomic_write(output_dir / name, payload)
    loaded_cal = load_partition(output_dir / "cal.jsonl", "cal")
    loaded_select = load_partition(output_dir / "select.jsonl", "select")
    check_partition_isolation(
        {
            "train": load_partition(args.targeted_anchor_train, "train"),
            "select": loaded_select,
            "cal": loaded_cal,
        }
    )
    manifest = {
        "schema_version": "decision2-cal-hard-v2/1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "generator_code_sha256": pilot.sha_file(Path(__file__)),
        "seed_sha256": pilot.sha_bytes(args.seed.encode()),
        "inputs": {
            path.name + "::" + label: pilot.sha_file(path)
            for label, path in (
                ("old_cal", args.old_cal),
                ("select", args.select_file),
                ("targeted_anchor", args.targeted_anchor_train),
                ("combined_6k", args.combined_6k),
                ("pure_legacy_1024", args.pure_legacy_1024),
                ("targeted_2k", args.targeted_2k),
                ("legacy_v1_train", args.legacy_v1_train),
                ("dev_prompts", args.dev_prompts),
                ("css_pilot_prompts", args.css_pilot_prompts),
            )
        },
        "source_attribution": {
            "retained_choice": "CSS three-task pilot CAL, unchanged from frozen old CAL",
            "fresh_types": "original programmatic oracle; private internal research, no public data license assigned",
        },
        "retained_choice": {
            "rows": 300,
            "id_sha256": pilot.sha_bytes(
                pilot.canonical(sorted(retained_ids)).encode()
            ),
            "row_payload_sha256": pilot.sha_bytes(pilot.jsonl_bytes(choice)),
            "source_counts": dict(
                sorted(collections.Counter(row["source"] for row in choice).items())
            ),
        },
        "new_oracle": {
            "rows": 600,
            "complete_pair_groups": 300,
            "family_counts": dict(
                sorted(collections.Counter(row["family"] for row in fresh).items())
            ),
            "difficulty_counts": {
                f"{kind}/{tier}": count
                for (kind, tier), count in sorted(difficulty.items())
            },
            "label_counts": {
                kind: dict(sorted(count.items()))
                for kind, count in sorted(labels.items())
            },
            "noul_page_boundary_rows": sum(
                row["audit_metadata"]["difficulty_tier"] == "boundary"
                for row in fresh
                if row["task_type"] == "noul"
            ),
            "score_one_unit_cutoff_rows": 300,
        },
        "counts": {
            field: dict(sorted(collections.Counter(row[field] for row in cal).items()))
            for field in ("family", "task_type", "language", "source")
        },
        "token_audit": {
            "method": "decoder-v2 segmented exact Qwen tokenizer",
            "tokenizer_revision": args.tokenizer_revision,
            "total": sum(lengths.values()),
            "minimum": min(lengths.values()),
            "maximum": max(lengths.values()),
            "max_row_tokens": args.max_row_tokens,
            "by_task_type": dict(sorted(token_by_type.items())),
        },
        "old_cal_fresh_overlap": zero_audit(
            fresh, old_cal, "Fresh CAL versus old CAL", context_near=True
        ),
        "select_overlap": selection_audit,
        "train_audits": train_audits,
        "train_inventory_exact_audit": train_inventory,
        "legacy_v1_audit": {"source": legacy_receipt, "overlap": legacy_audit},
        "open_reference_audits": open_refs,
        "sealed_holdouts": {
            "synthetic_final": "not read",
            "css_15task_evaluation": "not read",
        },
        "outputs": {
            name: {
                "sha256": pilot.sha_bytes(payload),
                "bytes": len(payload),
                "rows": len(payload.splitlines()),
            }
            for name, payload in payloads.items()
        },
        "interpretation": "A candidate CAL for a future checkpoint; it has not been fit or selected by model scores.",
        "limitations": [
            "Rule and arithmetic examples are generated, shorter and cleaner than deployment requests.",
            "The old CSS Choice CAL remains pilot-derived; a fit is not an external release evaluation.",
            "Hardness is structural, not measured against a target model; temperature can still hit its optimization bound.",
            "Only the named TRAIN sources and original 1.0 train source were audited; approximate near-duplicate search is not complete.",
        ],
    }
    pilot._atomic_write(
        output_dir / "cal_hard_v2.manifest.json",
        (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode(),
    )
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-cal", type=Path, required=True)
    parser.add_argument("--select-file", type=Path, required=True)
    parser.add_argument("--targeted-anchor-train", type=Path, required=True)
    parser.add_argument("--combined-6k", type=Path, required=True)
    parser.add_argument("--pure-legacy-1024", type=Path, required=True)
    parser.add_argument("--targeted-2k", type=Path, required=True)
    parser.add_argument(
        "--train-root",
        type=Path,
        required=True,
        help="Audit every *.train.jsonl beneath this private data root",
    )
    parser.add_argument(
        "--extra-train",
        type=Path,
        action="append",
        default=[],
        help="Additional private TRAIN JSONL outside train-root; repeat as needed",
    )
    parser.add_argument("--legacy-v1-train", type=Path, required=True)
    parser.add_argument("--dev-prompts", type=Path, required=True)
    parser.add_argument("--css-pilot-prompts", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--max-row-tokens", type=int, default=8192)
    parser.add_argument("--seed", default="decision2-cal-hard-v2")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = build(args)
    print(
        pilot.canonical(
            {
                "cal_sha256": manifest["outputs"]["cal.jsonl"]["sha256"],
                "rows": manifest["outputs"]["cal.jsonl"]["rows"],
                "tokens": manifest["token_audit"]["total"],
            }
        )
    )


if __name__ == "__main__":
    main()
