from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .adapters import AdapterContext, BenchmarkAdapter
from .canonical import digest
from .dataset_io import MAX_ROW_BYTES, read_small
from .draco_rubric import (
    DEFAULT_BATCH_SIZE,
    MAX_ANSWER_CHARS,
    CriterionVerdict,
    DracoCriterion,
    DracoRubric,
    _parse_rubric,
    aggregate,
    decode_verdicts,
    judge_messages,
    parse_samples,
    select_samples,
)
from .grading import WEIGHTED_RUBRIC_RESULT
from .sources import _write_dataset
from .target_contracts import resolve_auxiliary_target
from .task_identity import native_task_identity, source_identity
from .transport import CallFailure

BENCHMARK_ID: str = "draco-routed-answer-v1"
GRADER_VERSION: str = "draco-legacy-rubric-strict-v1"
SELECTION_VERSION: str = "draco-legacy-shuffle-v1"
SOURCE_URL: str = "https://huggingface.co/datasets/perplexity-ai/draco"
SENTINEL: str = "deliberate-eval"


def validated_rubric(raw: object, identity: str) -> DracoRubric:
    if not isinstance(raw, dict) or not isinstance(raw.get("sections"), list):
        raise ValueError("DRACO requires a structured weighted rubric")
    ids: set[str] = set()
    for section in raw["sections"]:
        if not isinstance(section, dict) or not isinstance(
            section.get("criteria"), list
        ):
            raise ValueError("DRACO rubric sections require criteria")
        for criterion in section["criteria"]:
            if (
                not isinstance(criterion, dict)
                or not isinstance(criterion.get("id"), str)
                or not criterion["id"]
                or criterion["id"] in ids
                or not isinstance(criterion.get("requirement"), str)
                or not criterion["requirement"]
            ):
                raise ValueError("DRACO criteria require unique IDs and requirements")
            weight = criterion.get("weight")
            if (
                isinstance(weight, bool)
                or not isinstance(weight, (int, float))
                or not math.isfinite(weight)
            ):
                raise ValueError("DRACO criterion weights must be finite numbers")
            ids.add(criterion["id"])
    rubric = _parse_rubric(rubric_id=identity, raw=raw)
    if (
        not ids
        or not math.isfinite(rubric.max_positive_score)
        or rubric.max_positive_score <= 0
        or not math.isfinite(rubric.min_negative_score)
    ):
        raise ValueError("DRACO requires a finite positive rubric denominator")
    return rubric


def preflight(
    case: dict[str, Any], manifest: dict[str, Any], cache: dict[str, Any]
) -> None:
    if manifest["profile"] == "standard":
        raise ValueError("DRACO diagnostics do not establish a Standard holdout")
    metadata = case.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("DRACO requires original problem metadata")
    problem = metadata.get("draco_problem")
    if not isinstance(problem, str) or not problem:
        raise ValueError("DRACO requires its original problem text")
    if case["messages"] != [{"role": "user", "content": f"{SENTINEL}\n\n{problem}"}]:
        raise ValueError("DRACO subject messages differ from the legacy prompt")
    rubric = validated_rubric(raw=case.get("answer"), identity=case["id"])
    options = manifest.setdefault("benchmark_options", {})
    if not isinstance(options, dict):
        raise ValueError("benchmark_options must be an object")
    config = options.setdefault(BENCHMARK_ID, {})
    if not isinstance(config, dict) or set(config) - {
        "judge",
        "grader_version",
        "batch_size",
    }:
        raise ValueError("Invalid DRACO benchmark options")
    if config.get("grader_version") != GRADER_VERSION:
        raise ValueError(f"DRACO requires grader_version={GRADER_VERSION}")
    resolve_auxiliary_target(config=config, role="judge", manifest=manifest)
    batch_size = config.setdefault("batch_size", DEFAULT_BATCH_SIZE)
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("DRACO batch_size must be a positive integer")
    required = 1 + math.ceil(len(rubric.criteria) / batch_size)
    if required > manifest["limits"]["max_calls_per_case"]:
        raise ValueError(
            "DRACO subject and rubric batches exceed the frozen call budget"
        )


def parse_batch(content: str, batch: list[DracoCriterion]) -> list[CriterionVerdict]:
    try:
        parsed = decode_verdicts(content)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            "DRACO judge returned malformed verdicts; response retained"
        ) from exc
    expected = {criterion.id for criterion in batch}
    found: dict[str, dict[str, Any]] = {}
    if not isinstance(parsed, list):
        raise ValueError("DRACO judge must return a verdict array")
    for item in parsed:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("id"), str)
            or item["id"] not in expected
            or item["id"] in found
            or not isinstance(item.get("matched"), bool)
            or not isinstance(item.get("evidence", ""), str)
        ):
            raise ValueError(
                "DRACO judge returned invalid or duplicate criterion verdicts"
            )
        found[item["id"]] = item
    if set(found) != expected:
        raise ValueError(
            "DRACO judge omitted criterion verdicts; grading is incomplete"
        )
    return [
        CriterionVerdict(
            id=c.id,
            matched=found[c.id]["matched"],
            weight=c.weight,
            section_title=c.section_title,
            is_negative=c.is_negative,
            evidence=found[c.id].get("evidence", "")[:120],
        )
        for c in batch
    ]


def execute(case: dict[str, Any], context: AdapterContext) -> dict[str, Any]:
    rubric = validated_rubric(raw=case["answer"], identity=case["id"])
    generated = context.call(messages=case["messages"])
    if not generated.get("output_complete", True):
        raise CallFailure("Subject output limit reached; DRACO grading is incomplete")
    judge = resolve_auxiliary_target(
        config=context.config, role="judge", manifest=context.manifest
    )
    verdicts: list[CriterionVerdict] = []
    criteria = rubric.criteria
    batch_size = context.config["batch_size"]
    for index in range(0, len(criteria), batch_size):
        batch = criteria[index : index + batch_size]
        response = context.call(
            messages=judge_messages(
                problem=case["metadata"]["draco_problem"],
                answer=generated["final"][:MAX_ANSWER_CHARS],
                batch=batch,
            ),
            role="judge",
            target=judge,
        )
        verdicts.extend(parse_batch(content=response["final"], batch=batch))
    score = aggregate(rubric=rubric, verdicts=verdicts)
    return {
        "answer": generated["final"],
        "correct": None,
        "score": score.normalized,
        "details": {
            "grader_version": GRADER_VERSION,
            "judge": judge["id"],
            "answer_clip_chars": MAX_ANSWER_CHARS,
            "rubric": {
                **asdict(score),
                "n_negative_triggered": score.n_negative_triggered,
            },
        },
    }


def adapter() -> BenchmarkAdapter:
    return BenchmarkAdapter(
        id=BENCHMARK_ID,
        title="DRACO routed-answer diagnostic",
        kind="weighted-rubric",
        source_url=SOURCE_URL,
        version=GRADER_VERSION,
        execute=execute,
        preflight=preflight,
        requires_answer=True,
        result_kind=WEIGHTED_RUBRIC_RESULT,
    )


def prepare(
    *,
    source_path: Path,
    revision: str,
    source_partition: str,
    store: Path,
    profile: str = "quick",
    domains: list[str] | None = None,
    seed: int = 42,
    max_samples: int | None = None,
) -> dict[str, Any]:
    if profile not in {"smoke", "quick"}:
        raise ValueError("DRACO preparation supports smoke/quick diagnostics only")
    if type(seed) is not int or (
        max_samples is not None and (type(max_samples) is not int or max_samples <= 0)
    ):
        raise ValueError("DRACO requires an integer seed and a positive sample cap")
    source_path = source_path.expanduser().absolute()
    content = read_small(path=source_path, maximum=MAX_ROW_BYTES)
    samples = parse_samples(json.loads(content))
    identities = [sample.id for sample in samples]
    if len(identities) != len(set(identities)):
        raise ValueError("Duplicate DRACO source task IDs")
    selected = select_samples(
        samples=samples, domains=domains, seed=seed, max_samples=max_samples
    )
    if not selected:
        raise ValueError("DRACO selection contains no tasks")
    source = {
        "url": SOURCE_URL,
        "revision": revision,
        "partition": source_partition,
        "files": [
            {"name": source_path.name, "sha256": hashlib.sha256(content).hexdigest()}
        ],
        "normalizer": "draco-legacy-json-v1",
        "revision_verification": "declared-revision-with-content-digest",
        "access_note": "Operator-supplied DRACO export; retain its upstream access and license records.",
        "selection": {
            "version": SELECTION_VERSION,
            "domains": domains,
            "seed": seed,
            "max_samples": max_samples,
            "ordered_source_ids_sha256": digest([sample.id for sample in selected]),
        },
    }
    task_source = source_identity(source=source, partition=source_partition)
    cases: list[dict[str, Any]] = []
    for sample in selected:
        rubric = asdict(sample.rubric)
        validated_rubric(raw=rubric, identity=sample.id)
        cases.append(
            {
                "id": f"{BENCHMARK_ID}/{sample.id}",
                "benchmark": BENCHMARK_ID,
                "messages": [
                    {"role": "user", "content": f"{SENTINEL}\n\n{sample.problem}"}
                ],
                "answer": rubric,
                "metadata": {
                    "stratum": sample.domain,
                    "split": "dev",
                    "source_revision": revision,
                    "draco_problem": sample.problem,
                    "task_identity": native_task_identity(
                        benchmark=BENCHMARK_ID,
                        row={"id": sample.id},
                        source=task_source,
                    ),
                },
            }
        )
    return _write_dataset(
        root=store,
        cases=cases,
        profile=profile,
        seed=seed,
        sources={BENCHMARK_ID: source},
        custom=True,
        selection=SELECTION_VERSION,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a local DRACO sr-bench diagnostic"
    )
    parser.add_argument("--source-path", required=True, type=Path)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--source-partition", required=True)
    parser.add_argument("--store", required=True, type=Path)
    parser.add_argument("--profile", choices=("smoke", "quick"), default="quick")
    parser.add_argument("--domains")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()
    try:
        result = prepare(
            source_path=args.source_path,
            revision=args.revision,
            source_partition=args.source_partition,
            store=args.store,
            profile=args.profile,
            domains=args.domains.split(",") if args.domains else None,
            seed=args.seed,
            max_samples=args.max_samples,
        )
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
