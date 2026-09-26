"""Build a fresh, oracle-checked Decision 2.0 targeted TRAIN candidate.

The only external text read is used for exclusion audits. Generation never
conditions on DEV or CSS pilot examples, predictions, or labels. This builder
has no argument or code path for benchmark final or CSS evaluation files.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import statistics
from pathlib import Path
from typing import Any

from transfer.build import normalized_context_sha256

from training.data import build_pilot as pilot
from training.model.data import check_partition_isolation, load_partition

SOURCE = "decision2_targeted_programmatic_v1"
FAMILIES = {
    "interval_conjunction": 300,
    "quantized_median": 200,
    "attributed_stance": 250,
    "dialogue_function": 250,
}
FROZEN_COMBINED_SHA256 = (
    "4e82651181fd4f9b11e82718275a7370b82cbe810f20bbe87db9786cfdf888ad"
)
FROZEN_SELECT_SHA256 = (
    "d8b1197830fe96a6554b49ee72c12f4755da00d0a819fb514725dc957b687e38"
)
FROZEN_CAL_SHA256 = "b5235c72269d8f13a28fa18019e29172f7c75ccc7204a63fa6fba9f20f096ebb"
DISCOURSE_LABELS = (
    "question",
    "answer",
    "agreement",
    "disagreement",
    "appreciation",
    "elaboration",
    "humor",
)
TOPICS = (
    "adding refill taps to the library",
    "opening the reading room on Sundays",
    "planting shade trees near the station",
    "moving the market to the riverside",
    "providing free repair workshops",
    "painting the cycle lanes blue",
    "extending the evening bus route",
    "reusing empty shops as classrooms",
    "installing benches at the plaza",
    "publishing the city budget monthly",
)
NAMES = (
    "Mira",
    "Tomas",
    "Leena",
    "Omar",
    "Ravi",
    "Sana",
    "Yara",
    "Nico",
    "Iris",
    "Pavel",
)
OBJECTS = (
    "blue folder",
    "red map",
    "green notebook",
    "silver key",
    "orange badge",
    "violet card",
)


def row_id(seed: str, family: str, index: int, variant: int) -> tuple[str, str]:
    stem = pilot.sha_bytes(f"{seed}\0{family}\0{index}".encode())[:20]
    return f"d2t_{stem}_{variant}", f"d2tg_{stem}"


def make_row(
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
    template: str,
    oracle: dict[str, Any],
) -> dict[str, Any]:
    identifier, group = row_id(seed, family, index, variant)
    row = {
        "id": identifier,
        "state": state,
        "instructions": instructions,
        "options": options,
        "label": next(i for i, option in enumerate(options) if option["key"] == answer),
        "task_type": task_type,
        "family": f"targeted_{family}",
        "group_id": group,
        "language": language,
        "split": "train",
        "source": SOURCE,
        "evaluation_role": "train",
        "render_template": f"targeted_{template}_v1",
        "audit_metadata": {
            "oracle": oracle,
            "generator": "targeted-oracle-v1",
            "case_index": index,
            "paired_variant": variant,
        },
    }
    row["input_sha256"] = pilot.input_sha256(row)
    pilot.validate_train_row(row)
    return row


def shuffled_options(
    descriptions: list[str], answer_description: str, rotation: int
) -> tuple[list[dict[str, str]], str]:
    if (
        len(set(descriptions)) != len(descriptions)
        or answer_description not in descriptions
    ):
        raise ValueError("Choice descriptions must be distinct with one gold")
    ordered = list(descriptions)
    ordered = ordered[rotation:] + ordered[:rotation]
    options = [
        {"key": f"K{i + 1}", "description": description}
        for i, description in enumerate(ordered)
    ]
    return options, next(
        option["key"]
        for option in options
        if option["description"] == answer_description
    )


def interval_conjunction(seed: str, index: int) -> list[dict[str, Any]]:
    """Inclusive/exclusive time window AND independent certification flag."""
    rng = pilot.rng_for(seed, "targeted-interval-conjunction", index)
    language = "zh" if index % 5 == 0 else "en"
    start = rng.randint(8, 21)
    end = start + rng.randint(5, 10)
    permit = "P" + str(20000 + rng.randint(0, 69999))
    name = rng.choice(NAMES)
    boundary = index % 2 == 0
    if boundary:
        dates = (end - 1, end) if index % 4 == 0 else (start, start - 1)
        certified = (True, True)
    else:
        day = rng.randint(start, end - 1)
        dates = (day, day)
        certified = (True, False)
    if index % 2:
        dates = dates[::-1]
        certified = certified[::-1]
    result = []
    for variant in range(2):
        date, stamp = dates[variant], certified[variant]
        valid = start <= date < end and stamp
        if language == "zh":
            state = (
                f"{name} 的许可编号为 {permit}。许可从第 {start} 天起生效，"
                f"到第 {end} 天开始时失效。第 {date} 天的检查记录："
                f"独立的认证印章{'已核实' if stamp else '未核实'}。"
                "两项条件必须同时成立，才可当日使用该许可。"
            )
            instructions = "按记录，这张许可在检查当天可以使用吗？"
            options = [
                {"key": "true", "description": "可以"},
                {"key": "false", "description": "不可以"},
            ]
        else:
            state = (
                f"{name} holds permit {permit}. It is valid starting on day {start} "
                f"and expires at the start of day {end}. The inspection is on day {date}. "
                f"The independent certification stamp is {'verified' if stamp else 'unverified'}. "
                "Both the time window and the verified stamp are required for use that day."
            )
            instructions = "Can this permit be used on the inspection day?"
            options = [
                {"key": "true", "description": "Yes"},
                {"key": "false", "description": "No"},
            ]
        if variant:
            options.reverse()
        result.append(
            make_row(
                seed,
                "interval_conjunction",
                index,
                variant,
                state=state,
                instructions=instructions,
                options=options,
                answer="true" if valid else "false",
                task_type="noul",
                language=language,
                template="permit_window_check",
                oracle={
                    "start_inclusive": start,
                    "end_exclusive": end,
                    "inspection_day": date,
                    "stamp_verified": stamp,
                    "valid": valid,
                    "contrast_dimension": "boundary" if boundary else "stamp",
                },
            )
        )
    if {row["options"][row["label"]]["key"] for row in result} != {"true", "false"}:
        raise AssertionError("Noul contrast does not change truth value")
    return result


def quantized_median(seed: str, index: int) -> list[dict[str, Any]]:
    """Median of five measurements mapped through explicit ordinal cutoffs."""
    rng = pilot.rng_for(seed, "targeted-quantized-median", index)
    language = "zh" if index % 5 == 0 else "en"
    first = rng.randint(18, 28)
    step = rng.randint(6, 11)
    cutoffs = [first + step * i for i in range(4)]
    levels = (index % 5, (index + 1) % 5)
    result = []
    for variant, level in enumerate(levels):
        median = (
            first - rng.randint(1, step)
            if level == 0
            else first + (level - 1) * step + rng.randrange(step)
        )
        readings = [
            rng.randint(0, 3),
            rng.randint(4, 7),
            median,
            first + 5 * step + rng.randint(0, 3),
            first + 6 * step + rng.randint(0, 3),
        ]
        rng.shuffle(readings)
        if (
            sorted(readings)[2] != median
            or sum(median >= cutoff for cutoff in cutoffs) != level
        ):
            raise AssertionError("Score oracle construction failed")
        if language == "zh":
            state = (
                "一台仪器有五次读数：" + "、".join(map(str, readings)) + "。"
                "只取五个数的中位数。分档规则：中位数每达到一个阈值就升一级，"
                "四个阈值依次是 " + "、".join(map(str, cutoffs)) + "；"
                "低于全部阈值为 0 级，达到全部阈值为 4 级。"
            )
            instructions = "这次仪器记录属于 0 到 4 中的哪一级？"
        else:
            state = (
                "An instrument records five readings: "
                + ", ".join(map(str, readings))
                + ". "
                "Use only the median of the five values. Raise the grade by one for each "
                "cutoff the median reaches; the four cutoffs are "
                + ", ".join(map(str, cutoffs))
                + ". Below all cutoffs is grade 0; reaching all four is grade 4."
            )
            instructions = "Which grade from 0 to 4 does this record receive?"
        options = [
            {
                "key": str(k),
                "description": f"{'等级' if language == 'zh' else 'Grade'} {k}",
            }
            for k in range(5)
        ]
        result.append(
            make_row(
                seed,
                "quantized_median",
                index,
                variant,
                state=state,
                instructions=instructions,
                options=options,
                answer=str(level),
                task_type="score",
                language=language,
                template="median_threshold_grade",
                oracle={
                    "readings": readings,
                    "median": median,
                    "cutoffs": cutoffs,
                    "grade": level,
                },
            )
        )
    return result


def attributed_stance(seed: str, index: int) -> list[dict[str, Any]]:
    """Author stance despite quoted or attributed opposing language."""
    rng = pilot.rng_for(seed, "targeted-attributed-stance", index)
    topic = rng.choice(TOPICS)
    author, other = rng.sample(NAMES, 2)
    target = ("Favor", "Against", "None")[index % 3]
    quote = ("unnecessary", "helpful", "divisive", "practical")[index % 4]
    if target == "Favor":
        state = (
            f"At a neighborhood meeting about {topic}, {other} called the plan '{quote}'. "
            f"{author} wrote: 'That is someone else's view. I support this plan because it "
            "would help residents.'"
        )
    elif target == "Against":
        state = (
            f"At a neighborhood meeting about {topic}, {other} called the plan '{quote}'. "
            f"{author} wrote: 'That is someone else's view. I oppose this plan because it "
            "would divert funds from urgent repairs.'"
        )
    else:
        state = (
            f"At a neighborhood meeting about {topic}, {other} called the plan '{quote}'. "
            f"{author} recorded both supporters' and critics' comments, then asked for the "
            "budget and timeline. The record gives no preference from the author."
        )
    instructions = f"Classify {author}'s own stated stance toward {topic}, ignoring views attributed to others."
    descriptions = ["Favor", "Against", "None"]
    rng.shuffle(descriptions)
    result = []
    for variant in range(2):
        options, answer = shuffled_options(descriptions, target, variant)
        result.append(
            make_row(
                seed,
                "attributed_stance",
                index,
                variant,
                state=state,
                instructions=instructions,
                options=options,
                answer=answer,
                task_type="choice",
                language="en",
                template="speaker_stance_attribution",
                oracle={
                    "author": author,
                    "semantic_stance": target,
                    "quoted_speaker": other,
                    "option_permutation": variant,
                },
            )
        )
    if (
        result[0]["options"][result[0]["label"]]["key"]
        == result[1]["options"][result[1]["label"]]["key"]
    ):
        raise AssertionError("Stance option reversal did not change answer key")
    return result


def dialogue_function(seed: str, index: int) -> list[dict[str, Any]]:
    """Controlled reply acts with stable semantics under option-key reversal."""
    rng = pilot.rng_for(seed, "targeted-dialogue-function", index)
    object_name = rng.choice(OBJECTS)
    speaker, responder = rng.sample(NAMES, 2)
    shelf = rng.choice(("upper shelf", "supply desk", "side cabinet", "archive room"))
    target = DISCOURSE_LABELS[index % len(DISCOURSE_LABELS)]
    turns = {
        "question": (
            f"I placed the {object_name} on the {shelf}.",
            f"When did you put the {object_name} there?",
        ),
        "answer": (
            f"Where did you put the {object_name}?",
            f"I put the {object_name} on the {shelf}.",
        ),
        "agreement": (
            f"We should label the {object_name} before storing it.",
            f"I agree; labeling the {object_name} first is sensible.",
        ),
        "disagreement": (
            f"We should label the {object_name} before storing it.",
            f"I disagree; leave the {object_name} unlabelled for now.",
        ),
        "appreciation": (
            f"I finished organizing the {object_name} on the {shelf}.",
            "Thank you for taking care of that work.",
        ),
        "elaboration": (
            f"The {object_name} is ready for visitors.",
            f"It is on the {shelf}, next to the new catalog so visitors can find it.",
        ),
        "humor": (
            f"The {object_name} wandered around the office again.",
            "Perhaps it is applying for a job as our travel guide.",
        ),
    }
    first, second = turns[target]
    state = f"{speaker}: {first}\n{responder}: {second}"
    instructions = f"What conversational role does {responder}'s reply mainly play?"
    descriptions = list(DISCOURSE_LABELS)
    rng.shuffle(descriptions)
    result = []
    for variant in range(2):
        options, answer = shuffled_options(descriptions, target, variant)
        result.append(
            make_row(
                seed,
                "dialogue_function",
                index,
                variant,
                state=state,
                instructions=instructions,
                options=options,
                answer=answer,
                task_type="choice",
                language="en",
                template="two_turn_reply_function",
                oracle={"reply_function": target, "option_permutation": variant},
            )
        )
    if (
        result[0]["options"][result[0]["label"]]["key"]
        == result[1]["options"][result[1]["label"]]["key"]
    ):
        raise AssertionError("Dialogue option reversal did not change answer key")
    return result


GENERATORS = {
    "interval_conjunction": interval_conjunction,
    "quantized_median": quantized_median,
    "attributed_stance": attributed_stance,
    "dialogue_function": dialogue_function,
}


def generate(seed: str) -> list[dict[str, Any]]:
    rows = [
        row
        for family, count in FAMILIES.items()
        for index in range(count)
        for row in GENERATORS[family](seed, index)
    ]
    if len(rows) != 2000 or len({row["id"] for row in rows}) != 2000:
        raise AssertionError("Targeted candidate needs 2,000 unique rows")
    if set(collections.Counter(row["group_id"] for row in rows).values()) != {2}:
        raise AssertionError("Every targeted group must contain two paired rows")
    rows.sort(
        key=lambda row: (
            pilot.sha_bytes(f"{seed}\0output\0{row['id']}".encode()),
            row["id"],
        )
    )
    return rows


def context_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "id": row["id"],
            "state": row["state"],
            "instructions": "",
            "options": [],
            "task_type": "context",
        }
        for row in rows
    ]


def text_hashes(value: Any) -> tuple[str, str]:
    text = value if isinstance(value, str) else pilot.canonical(value)
    return pilot.sha_bytes(text.encode()), normalized_context_sha256(text)


def load_context_reference(
    path: Path, *, expected_name: str | None = None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if expected_name is not None and path.name != expected_name:
        raise ValueError(f"Expected only {expected_name} for this reference role")
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            source = json.loads(line)
            if not isinstance(source.get("id"), str) or "state" not in source:
                raise ValueError(f"Malformed audit-only context in {path.name}")
            rows.append(
                {
                    "id": source["id"],
                    "state": source["state"],
                    "group_id": source.get("group_id"),
                    "input_sha256": source.get("input_sha256"),
                    "instructions": "",
                    "options": [],
                    "task_type": "context",
                }
            )
    return rows, {"file": path.name, "sha256": pilot.sha_file(path), "rows": len(rows)}


def context_overlap(
    train: list[dict[str, Any]], references: list[dict[str, Any]], *, approximate: bool
) -> dict[str, Any]:
    references_hashes = {text_hashes(row["state"]) for row in references}
    raw = {pair[0] for pair in references_hashes}
    normalized = {pair[1] for pair in references_hashes}
    train_hashes = [text_hashes(row["state"]) for row in train]
    ids = {row["id"] for row in references}
    groups = {
        row["group_id"] for row in references if isinstance(row.get("group_id"), str)
    }
    inputs = {
        row["input_sha256"]
        for row in references
        if isinstance(row.get("input_sha256"), str)
    }
    receipt = {
        "id_rows": sum(row["id"] in ids for row in train),
        "group_id_rows": sum(row["group_id"] in groups for row in train),
        "input_sha256_rows": sum(row["input_sha256"] in inputs for row in train),
        "raw_context_rows": sum(pair[0] in raw for pair in train_hashes),
        "normalized_context_rows": sum(pair[1] in normalized for pair in train_hashes),
    }
    if approximate:
        receipt["near_context"] = pilot.near_duplicates(context_rows(train), references)
    if any(
        receipt[key]
        for key in (
            "id_rows",
            "group_id_rows",
            "input_sha256_rows",
            "raw_context_rows",
            "normalized_context_rows",
        )
    ) or receipt.get("near_context", {}).get("count"):
        raise ValueError(f"Targeted TRAIN overlaps audit-only reference: {receipt}")
    return receipt


def summarize_counts(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    return {
        field: dict(sorted(collections.Counter(row[field] for row in rows).items()))
        for field in ("family", "task_type", "language", "source")
    }


def build(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    for path, expected in (
        (args.combined_train, FROZEN_COMBINED_SHA256),
        (args.select_file, FROZEN_SELECT_SHA256),
        (args.cal_file, FROZEN_CAL_SHA256),
    ):
        if pilot.sha_file(path) != expected:
            raise ValueError(f"Frozen reference differs: {path.name}")
    if args.legacy_train.name != "train.jsonl":
        raise ValueError("Legacy source must be its original train.jsonl")
    train = generate(args.seed)
    combined = load_partition(args.combined_train, "train")
    select = load_partition(args.select_file, "select")
    cal = load_partition(args.cal_file, "cal")
    check_partition_isolation(
        {"train": [*combined, *train], "select": select, "cal": cal}
    )
    cross = {}
    for name, reference in (
        ("combined_train", combined),
        ("select", select),
        ("cal", cal),
    ):
        audit = pilot.overlap_audit(train, reference)
        if pilot.audit_has_exact_overlap(audit) or audit["near_duplicate"]["count"]:
            raise ValueError(f"Targeted TRAIN overlaps {name}: {audit}")
        cross[name] = audit
    if pilot.train_consistency_audit(train)["conflicting_gold_groups"]:
        raise ValueError("Targeted TRAIN has conflicting labels for identical inputs")
    contexts = {}
    for name, path, expected in (
        ("legacy_v1_train_source", args.legacy_train, "train.jsonl"),
        ("synthetic_dev", args.dev_prompts, "dev.prompts.jsonl"),
        ("css_3task_pilot", args.css_pilot_prompts, "css-pilot.prompts.jsonl"),
    ):
        reference, source = load_context_reference(path, expected_name=expected)
        contexts[name] = {
            "source": source,
            "overlap": context_overlap(
                train, reference, approximate=name != "legacy_v1_train_source"
            ),
        }
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(args.tokenizer.resolve()), local_files_only=True, trust_remote_code=False
    )
    lengths = [pilot.count_tokens(row, tokenizer) for row in train]
    if min(lengths) <= 0 or max(lengths) > args.max_row_tokens:
        raise ValueError("Exact tokenizer row length outside allowed range")
    by_family = collections.defaultdict(list)
    for row, length in zip(train, lengths):
        by_family[row["family"]].append(length)
    payload = pilot.jsonl_bytes(train)
    output_dir.mkdir(parents=True, mode=0o700)
    pilot._atomic_write(output_dir / "targeted_2k.train.jsonl", payload)
    manifest = {
        "schema_version": "decision2-targeted-programmatic-candidate/1",
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "generator_code_sha256": pilot.sha_file(Path(__file__)),
        "seed_sha256": pilot.sha_bytes(args.seed.encode()),
        "source_attribution": {
            "license": "private internal research; no public data license assigned",
            "attribution": "Decision 2.0 research team",
            "evidence": "Only synthetic oracle generation; all external text used for exclusion checks",
        },
        "reference_sha256": {
            "combined_train": FROZEN_COMBINED_SHA256,
            "select": FROZEN_SELECT_SHA256,
            "cal": FROZEN_CAL_SHA256,
        },
        "reference_context_audits": contexts,
        "sealed_holdouts": {
            "benchmark_final": "not read",
            "css_15task_evaluation": "not read",
        },
        "counts": summarize_counts(train),
        "groups": 1000,
        "pair_properties": {
            "noul_contrasts": 300,
            "score_contrasts": 200,
            "choice_option_reversals": 500,
        },
        "token_audit": {
            "method": "decoder-v2 segmented exact Qwen tokenizer",
            "tokenizer_revision": args.tokenizer_revision,
            "total": sum(lengths),
            "minimum": min(lengths),
            "median": statistics.median(lengths),
            "p95": sorted(lengths)[int(0.95 * (len(lengths) - 1))],
            "maximum": max(lengths),
            "max_row_tokens": args.max_row_tokens,
            "family_tokens": {
                family: sum(values) for family, values in sorted(by_family.items())
            },
        },
        "cross_partition_audit": cross,
        "output": {
            "file": "targeted_2k.train.jsonl",
            "sha256": pilot.sha_bytes(payload),
            "bytes": len(payload),
            "rows": len(train),
        },
        "interpretation": "Training candidate only. It is not evidence of model improvement or a causal ablation.",
        "limitations": [
            "Programmatic stance and dialogue examples are simpler than human CSS labels.",
            "No reliable implicit-hate oracle is generated; that pilot weakness remains unaddressed.",
            "Approximate near-duplicate detection cannot prove semantic independence.",
            "Benchmark final and CSS 15-task evaluation data are not read by this builder.",
        ],
    }
    pilot._atomic_write(
        output_dir / "targeted_2k.manifest.json",
        (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode(),
    )
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--combined-train", type=Path, required=True)
    parser.add_argument("--legacy-train", type=Path, required=True)
    parser.add_argument("--select-file", type=Path, required=True)
    parser.add_argument("--cal-file", type=Path, required=True)
    parser.add_argument("--dev-prompts", type=Path, required=True)
    parser.add_argument("--css-pilot-prompts", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--max-row-tokens", type=int, default=8192)
    parser.add_argument("--seed", default="decision2-targeted-2k-v1")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = build(args)
    print(
        pilot.canonical(
            {
                "rows": manifest["output"]["rows"],
                "train_sha256": manifest["output"]["sha256"],
                "tokens": manifest["token_audit"]["total"],
            }
        )
    )


if __name__ == "__main__":
    main()
